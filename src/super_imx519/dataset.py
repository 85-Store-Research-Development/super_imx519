"""採寸のモデル用のデータセット（壁の正面図・服の形・手で測った採寸）を CMS から組み立てる。

データの正は CMS にある（ここにはファイルを持たない）:
- 撮影の元データ（photoSources）: RAW（非公開の R2 `85store-dataset`）と、補正の記録（report）。
  記録には、マーカーの位置と誤差・mm/px・元の写真 → 正面図の変換（wall.matrix・wall.size）・
  ハンガーの位置・服の範囲（crop）・自動の採寸（measurements）が入る（cms.py が返す）
- 商品（products）の「採寸」: 手で測った値（1 点もの = サイズが空の 1 行）。これが正解
- 着画（モデルが写る写真）は使わない（正面・背面の壁の写真 = 作り方 wall だけ）

  .venv/bin/python -m super_imx519.dataset export 497 498 ...  # 一覧（index.jsonl）を書き出す
  .venv/bin/python -m super_imx519.dataset scene 10 OUT_DIR    # 元データ 10 の正面図・服の形を作る

scene は RAW から正面図と服の形をその場で作り直す（色の補正 → マーカー → 正面図 → 服の形）。
記録の変換と作り直した変換がずれていれば知らせる（マーカーの見つけ方が変わったとき）。
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIZES = ("着丈", "身幅", "肩幅", "袖丈", "裄丈")
MATRIX_TOLERANCE_PX = 2.0  # 記録の変換と作り直した変換の、元の写真の上でのずれの許容


def measured_values(product: dict) -> dict | None:
    """CMS の商品の採寸（1 点もの = サイズが空の 1 行）→ {"着丈": 72.0, ...}。無ければ None。"""
    table = product.get("measurementTable") or {}
    points, rows = table.get("points") or [], table.get("rows") or []
    if len(rows) != 1:
        return None
    values = {
        name.strip(): float(v)
        for name, v in zip(points, rows[0].get("values") or [], strict=False)
        if name and name.strip() in SIZES and isinstance(v, (int, float))
    }
    return values or None


def rows_for(product: dict, sources: list[dict]) -> list[dict]:
    """1 つの商品の、正面・背面の壁の写真（補正済み）ごとの行。"""
    measured = measured_values(product)
    rows = []
    for s in sources:
        report = s.get("report") if isinstance(s.get("report"), dict) else None
        if s.get("recipe") != "wall" or s.get("status") != "done" or not report:
            continue
        wall = report.get("wall") or {}
        rows.append(
            {
                "product": product["id"],
                "title": product.get("title"),
                "product_type": product.get("productType"),
                "source": s["id"],
                "role": s.get("role"),
                "filename": s.get("filename"),
                "measured": measured,
                "auto": (report.get("measurements") or {}).get("values"),
                "mm_per_px": wall.get("mm_per_px"),
                "marker_residual_mm": wall.get("residual_mm"),
                "has_matrix": "matrix" in wall,
                "crop": report.get("crop"),
            }
        )
    return rows


def export(products: list[int], out: Path, cms=None) -> list[dict]:
    from .cms import Cms

    cms = cms or Cms()
    rows = []
    for product_id in products:
        status, product = cms._json(f"/api/products/{product_id}?depth=0")
        if status != 200:
            print(f"商品 {product_id}: 読めません（{status}）", file=sys.stderr)
            continue
        rows += rows_for(product, cms.product_sources(product_id))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def matrix_shift_px(a, b: np.ndarray, size: tuple[int, int]) -> float:
    """2 つの変換（元の写真 → 正面図）で、正面図の 4 隅が元の写真の上でどれだけずれるか（画素）。"""
    w, h = size
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], float)
    pa = cv2.perspectiveTransform(corners, np.linalg.inv(np.array(a, float)))
    pb = cv2.perspectiveTransform(corners, np.linalg.inv(b))
    return float(np.abs(pa - pb).max())


def scene(source: dict, cms=None) -> dict:
    """元データ（補正済みの壁の写真）の RAW から、正面図（BGR、0〜255）と服の形を作り直す。"""
    from .cms import Cms
    from .pipeline import phone, rectify

    cms = cms or Cms()
    report = source.get("report") or {}
    with tempfile.TemporaryDirectory(prefix="scene-") as tmp:
        src = Path(tmp) / (source.get("filename") or f"{source['id']}.dng")
        cms.download(source["url"], src)
        rgb, kind = phone.load(src)
    layout = rectify.WallLayout.load(phone.DEFAULT_WALL)
    detect, white = phone.detection_image(rgb, kind)
    bgr, quads = phone.correct_color(rgb, kind, white, detect, {"warnings": []})
    view, _ = phone.make_view(detect, layout)
    notes = []
    recorded = (report.get("wall") or {}).get("matrix")
    if recorded is not None:
        shift = matrix_shift_px(recorded, view.matrix, view.size)
        if shift > MATRIX_TOLERANCE_PX:
            notes.append(
                f"記録の変換と {shift:.1f} px ずれています（マーカーの見つけ方が変わった）"
            )
    wall = np.clip(view.warp(bgr), 0, 255)
    found = phone.analyze(wall, view.valid(bgr.shape), view, quads, layout)
    hook = phone.hook_position(found, view, layout, {"warnings": []})
    return {
        "wall": wall,
        "garment": found.garment,
        "hook": hook,
        "bbox": found.bbox,
        "mm_per_px": view.mm_per_px,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="商品ごとの正面・背面と採寸の一覧を書き出す")
    ex.add_argument("products", type=int, nargs="+")
    ex.add_argument("--out", type=Path, default=Path("index.jsonl"))
    sc = sub.add_parser("scene", help="元データの正面図・服の形を作り直して保存する")
    sc.add_argument("source", type=int)
    sc.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        rows = export(args.products, args.out)
        measured = sum(1 for r in rows if r["measured"])
        print(f"{args.out}: {len(rows)} 枚（手で測った採寸あり {measured} 枚）")
        return 0
    from .cms import Cms

    cms = Cms()
    status, source = cms._json(f"/api/photoSources/{args.source}?depth=0")
    if status != 200:
        print(f"元データ {args.source} を読めません（{status}）", file=sys.stderr)
        return 1
    result = scene(source, cms)
    args.out.mkdir(parents=True, exist_ok=True)
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, 92]
    cv2.imwrite(str(args.out / "wall.jpg"), result["wall"].astype(np.uint8), jpeg)
    cv2.imwrite(str(args.out / "mask.png"), result["garment"].astype(np.uint8) * 255)
    print(f"{args.out}: mm/px={result['mm_per_px']} {' '.join(result['notes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
