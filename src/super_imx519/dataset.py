"""採寸のモデルを作るためのデータセット（壁の正面図・服の形・手で測った採寸の組）。

正面・背面の写真を補正するとき（pipeline/phone.py の process_upload に dataset_dir を渡す）に、
壁の正面図（4 隅のマーカーで決めた壁面の座標。1 px = mm_per_px mm）のうち服のまわりと、
服の形（マスク）、座標の情報を残す。手で測った採寸は 85store-cms の商品の「採寸」の欄が正で、
export でつなぐ。

  .venv/bin/python -m super_imx519.dataset rebuild 497 498 ...  # 正面・背面を RAW から作り直す
  .venv/bin/python -m super_imx519.dataset export               # 採寸がある商品を書き出す

置き場所は DATASET_ROOT（既定 ~/data/super_imx519/dataset）の cms/<商品 ID>/<元データ ID>-<区分>/:
- wall.jpg: 壁の正面図の、服のまわり（服の範囲 + MARGIN_MM）。色はカードで補正した sRGB。
  マーカーの板は消さない
- mask.png: 服の形（255 が服。ハンガーを含むことがある）
- meta.json: mm_per_px・切り出しの位置（正面図の画素）・ハンガーの位置・服の範囲・
  マーカーの中心（壁面の mm）・元の写真 → 切り出しの変換（3 × 3）・壁面の mm → 切り出しの変換・
  マーカーの誤差・自動の採寸・CMS の ID など。画素の座標は切り出し（wall.jpg）の左上が原点
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "~/data/super_imx519/dataset")).expanduser()
MARGIN_MM = 100
SIZES = ("着丈", "身幅", "肩幅", "袖丈", "裄丈")


def scene_dir(product: int, source_id: int, role: str) -> Path:
    return DATASET_ROOT / "cms" / str(product) / f"{source_id}-{role}"


def save_scene(
    out: Path,
    wall: np.ndarray,
    garment: np.ndarray,
    view,
    hook: tuple[float, float],
    bbox: tuple[int, int, int, int],
    extra: dict | None = None,
) -> None:
    """wall は正面図（BGR、0〜255 の float）、garment は服の形（bool）、view は WallView。"""
    mm = view.mm_per_px
    h, w = garment.shape
    pad = round(MARGIN_MM / mm)
    x0, y0 = max(bbox[0] - pad, 0), max(min(bbox[1], int(hook[1])) - pad, 0)
    x1, y1 = min(bbox[2] + pad, w - 1), min(bbox[3] + pad, h - 1)
    out.mkdir(parents=True, exist_ok=True)
    crop = np.clip(wall[y0 : y1 + 1, x0 : x1 + 1], 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "wall.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / "mask.png"), garment[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    meta = {
        "mm_per_px": mm,
        "crop_px": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
        "hook_px": [round(hook[0] - x0, 1), round(hook[1] - y0, 1)],
        "garment_bbox_px": [
            int(bbox[0] - x0),
            int(bbox[1] - y0),
            int(bbox[2] - x0),
            int(bbox[3] - y0),
        ],
        "markers_mm": {
            str(i): np.round(view.to_mm(q).mean(axis=0), 1).tolist()
            for i, q in view.markers.items()
        },
        "marker_residual_mm": [round(float(v), 2) for v in view.residual_mm],
        # 元の写真の画素 → 切り出しの画素 / 壁面の mm → 切り出しの画素
        "source_to_crop": np.round(shift @ view.matrix, 8).tolist(),
        "mm_to_crop": np.round(shift @ view.mm_to_px, 8).tolist(),
        **(extra or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def measured_values(product: dict) -> dict | None:
    """CMS の商品の採寸（1 点もの = サイズが空の 1 行）→ {"着丈": 72.0, ...}。無ければ None。"""
    table = product.get("measurementTable") or {}
    points, rows = table.get("points") or [], table.get("rows") or []
    if len(rows) != 1:
        return None
    values = {
        name.strip(): float(v)
        for name, v in zip(points, rows[0].get("values") or [], strict=False)
        if name and isinstance(v, (int, float))
    }
    return values or None


def export(cms=None, root: Path = DATASET_ROOT) -> Path:
    """採寸が入った商品の正面図を、1 行 1 枚で index.jsonl に書き出す。"""
    from .cms import Cms

    cms = cms or Cms()
    rows = []
    for product_dir in sorted((root / "cms").glob("*")):
        status, product = cms._json(f"/api/products/{product_dir.name}?depth=0")
        if status != 200:
            continue
        measured = measured_values(product)
        for scene in sorted(product_dir.glob("*")):
            meta_path = scene / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            rows.append(
                {
                    "product": int(product_dir.name),
                    "title": product.get("title"),
                    "product_type": product.get("productType"),
                    "scene": str(scene.relative_to(root)),
                    "role": meta.get("role"),
                    "measured": {k: v for k, v in (measured or {}).items() if k in SIZES} or None,
                    "auto": (meta.get("measurements") or {}).get("values"),
                }
            )
    path = root / "index.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def rebuild(products: list[int], cms=None) -> int:
    """補正済みの商品の正面・背面を RAW から処理し直し、データセットだけ作る（CMS は変えない）。"""
    from .cms import Cms
    from .pipeline import phone

    cms = cms or Cms()
    done = 0
    with tempfile.TemporaryDirectory(prefix="dataset-") as tmp:
        work = Path(tmp)
        for product in products:
            for s in cms.product_sources(product):
                if s.get("recipe") != "wall" or s.get("status") != "done":
                    continue
                src = work / (s.get("filename") or f"{s['id']}.dng")
                cms.download(s["url"], src)
                out = scene_dir(product, s["id"], s.get("role") or "front")
                report = phone.process_upload(
                    src,
                    work / f"out-{s['id']}",
                    dataset_dir=out,
                    dataset_extra=extra_of(product, s),
                )
                src.unlink(missing_ok=True)
                done += 1
                print(
                    f"商品 {product}: {s.get('filename')}（{s.get('role')}）→ {out}"
                    f"  採寸 {(report.get('measurements') or {}).get('values')}",
                    flush=True,
                )
    return done


def extra_of(product: int, source: dict) -> dict:
    return {
        "product": product,
        "source": source["id"],
        "role": source.get("role"),
        "filename": source.get("filename"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rb = sub.add_parser("rebuild", help="補正済みの商品の正面・背面から、データセットを作り直す")
    rb.add_argument("products", type=int, nargs="+")
    sub.add_parser("export", help="採寸が入った商品を index.jsonl に書き出す")
    args = parser.parse_args()
    if args.command == "rebuild":
        print(f"{rebuild(args.products)} 枚を作り直しました")
    else:
        path = export()
        lines = path.read_text().splitlines()
        with_measure = sum(1 for line in lines if json.loads(line)["measured"])
        print(f"{path}: {len(lines)} 枚（手で測った採寸あり {with_measure} 枚）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
