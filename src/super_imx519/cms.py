"""85store-cms の撮影の元データ（RAW）を取りに行き、補正した商品画像を返す。

CMS の「写真から下書きを作る」で「下書きを作る」を押すと、商品ごとに保留の下書きができ、RAW の写真は
「処理待ち」になる（作り方 wall = 正面・背面、develop = アップ）。これを 1 分ごとに取りに行く。

  .venv/bin/python -m super_imx519.cms            # 処理待ちを処理して終わる（無ければすぐ終わる）

- 商品ごとに、正面・背面（wall）を先に処理する: カラーカードで色、4 隅のマーカーで形を直し、構図を
  整えて仕上げる（pipeline/phone.py）。着丈・身幅（おおよそ）も測って report に入れる
- アップ（develop）は、同じ商品の正面（無ければ背面）のカードの補正を、露出をそろえてから当てる
  （pipeline/closeup.py）。正面がまだ処理されていなければ、その回は待つ（ほかが処理中のとき）
- 仕上がりは POST /api/photoSources/:id/result（multipart の file と _payload { report }）。
  CMS が商品写真にして、商品の「写真」に区分の順に並べる。失敗は /:id/error
- CMS の API は 85store-cms の src/photo-intake/endpoints.ts。PC は tailnet の whois で通る
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import dataset
from .pipeline import closeup, exposure, phone

CMS_URL = os.getenv("CMS_URL", "https://cms.85-store.com").rstrip("/")
TIMEOUT_S = 300
# 正面・背面を先に（アップの色の補正に、そのカードの値を使う）
ORDER = {("wall", "front"): 0, ("wall", "back"): 1}


class Cms:
    def __init__(self, url: str = CMS_URL, opener=None):
        self.url = url
        self.open = opener or urllib.request.urlopen

    def _json(self, path: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self.open(req, timeout=TIMEOUT_S) as res:
                return res.status, json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def queue(self, limit: int = 100) -> list[dict]:
        status, body = self._json(f"/api/photoSources/queue?limit={limit}")
        if status != 200:
            raise RuntimeError(f"処理待ちを読めません: {status} {body}")
        return body["docs"]

    def product_sources(self, product: int) -> list[dict]:
        q = urllib.parse.urlencode(
            {"where[product][equals]": product, "depth": 0, "limit": 500, "pagination": "false"}
        )
        status, body = self._json(f"/api/photoSources?{q}")
        if status != 200:
            raise RuntimeError(f"商品 {product} の元データを読めません: {status}")
        return body["docs"]

    def claim(self, source_id: int) -> bool:
        status, _ = self._json(f"/api/photoSources/{source_id}/claim", "POST")
        return status == 200

    def download(self, url: str, dest: Path) -> None:
        # CMS のファイルの URL は、非公開のバケットの署名 URL に転送される
        full = url if url.startswith("http") else f"{self.url}{url}"
        with (
            self.open(urllib.request.Request(full), timeout=TIMEOUT_S) as res,
            open(dest, "wb") as f,
        ):
            while chunk := res.read(1 << 20):
                f.write(chunk)

    def result(self, source_id: int, jpeg: Path, report: dict) -> dict:
        boundary = uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="_payload"\r\n\r\n'.encode()
            + json.dumps({"report": report}, ensure_ascii=False).encode()
            + b"\r\n",
            f"--{boundary}\r\n".encode()
            + f'Content-Disposition: form-data; name="file"; filename="{jpeg.name}"\r\n'.encode()
            + b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.read_bytes()
            + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(
            f"{self.url}/api/photoSources/{source_id}/result", data=b"".join(parts), method="POST"
        )
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with self.open(req, timeout=TIMEOUT_S) as res:
                return json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"仕上がりを返せません: {e.code} {e.read()[:300]!r}") from e

    def error(self, source_id: int, message: str) -> None:
        self._json(f"/api/photoSources/{source_id}/error", "POST", {"message": message[:2000]})


def reference_of(sources: list[dict]) -> dict | None:
    """済んだ正面（無ければ背面）の report から、アップに当てる補正の値。"""
    done = [
        s
        for s in sources
        if s.get("recipe") == "wall"
        and s.get("status") == "done"
        and isinstance(s.get("report"), dict)
    ]
    done.sort(key=lambda s: 0 if s.get("role") == "front" else 1)
    for s in done:
        ref = closeup.reference_from(s["report"], s["report"].get("exposure_scale"))
        if ref:
            return ref
    return None


def waiting_for_reference(sources: list[dict], mine: set[int]) -> bool:
    """正面・背面がまだ処理されていない（ほかが処理中か、この回で処理しない）。"""
    return any(
        s.get("recipe") == "wall"
        and s.get("status") in ("pending", "processing")
        and s["id"] not in mine
        for s in sources
    )


def process_source(cms: Cms, doc: dict, reference: dict | None, work: Path) -> dict:
    src = work / (doc.get("filename") or f"photo-{doc['id']}.dng")
    cms.download(doc["url"], src)
    out = work / f"out-{doc['id']}"
    if doc["recipe"] == "wall":
        try:
            # 採寸のモデル用に、壁の正面図・服の形も残す（dataset.py）
            report = phone.process_upload(
                src,
                out,
                dataset_dir=dataset.scene_dir(
                    doc["product"], doc["id"], doc.get("role") or "front"
                ),
                dataset_extra=dataset.extra_of(doc["product"], doc),
            )
        except Exception as e:  # noqa: BLE001 マーカーが見つからないなど。形は直さず、色だけ合わせる
            report = closeup.process_closeup(src, out, reference)
            report["warnings"].insert(
                0, f"壁のマーカーで整えられなかったので、色だけ合わせました: {e}"
            )
        if src.suffix.lower() == ".dng":
            report["exposure_scale"] = exposure.exposure_scale(src)
    else:
        report = closeup.process_closeup(src, out, reference)
    cms.result(doc["id"], out / "final.jpg", report)
    src.unlink(missing_ok=True)
    return report


def run_once(cms: Cms | None = None, log=print) -> int:
    """処理待ちを商品ごとに処理する。処理した枚数を返す。"""
    cms = cms or Cms()
    docs = cms.queue()
    if not docs:
        return 0
    by_product: dict[int, list[dict]] = {}
    for d in docs:
        by_product.setdefault(d["product"], []).append(d)
    done = 0
    with tempfile.TemporaryDirectory(prefix="cms-intake-") as tmp:
        work = Path(tmp)
        for product, todo in by_product.items():
            todo.sort(key=lambda d: ORDER.get((d["recipe"], d.get("role")), 2))
            sources = cms.product_sources(product)
            reference = reference_of(sources)
            mine = {d["id"] for d in todo}
            for doc in todo:
                if (
                    doc["recipe"] == "develop"
                    and reference is None
                    and waiting_for_reference(sources, mine)
                ):
                    log(f"商品 {product}: 正面の処理を待ちます（{doc.get('filename')}）")
                    continue
                if not cms.claim(doc["id"]):
                    continue
                try:
                    report = process_source(cms, doc, reference, work)
                except Exception as e:  # noqa: BLE001 CMS に知らせて、次へ
                    cms.error(doc["id"], f"補正できませんでした: {e}")
                    log(f"商品 {product}: {doc.get('filename')} 失敗: {e}")
                    continue
                done += 1
                if doc["recipe"] == "wall" and reference is None:
                    reference = closeup.reference_from(report, report.get("exposure_scale"))
                m = (report.get("measurements") or {}).get("values")
                log(
                    f"商品 {product}: {doc.get('filename')}（{doc['recipe']}）"
                    + (f" 採寸 {m}" if m else "")
                    + (f" 注意: {'／'.join(report['warnings'])}" if report.get("warnings") else "")
                )
    return done


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    n = run_once(log=lambda s: print(s, flush=True))
    if n:
        print(f"{n} 枚を補正して CMS に返しました", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
