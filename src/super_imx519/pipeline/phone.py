"""スマホ（iPhone）で撮った 1 枚から、商品画像を作る（スタジオがアップロードごとに呼ぶ）。

1. 読み込み: DNG（ProRAW）はリニアに現像し、JPEG はそのまま（表示用）として扱う
2. 色: カラーカードを見つけて補正する（color.py）。無ければ補正せず、report に書く
3. 形: 4 隅のマーカーで、壁の正面図（0.5 mm/px）にする（rectify.py）
4. 構図: 服とフックを見つけ、フックが上の端に掛かる縦 2:3 を実寸で決める（framing.py）
5. 掃除: マーカーの板とカードを、影ごと壁で埋める。切り出しが写っていない所に掛かれば、そこも埋める
6. 長辺 2048px 以内の final.jpg（品質 95）、画面用の preview.jpg、report.json を書く

iPhone の ProRAW は、保存前に明暗の差が圧縮されている（推定）ので、色の正しさは近似にとどまる。
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..markers import BOARD_MM, MARKER_MM
from . import cleanup, color, framing, rectify, stacking
from .process import FINAL_SIDE, PREVIEW_SIDE, fit, write_jpeg

SUFFIXES = {".jpg", ".jpeg", ".png", ".dng"}
CARD_BORDER_MM = 12.0  # パッチの外側の、カードの黒い台紙の幅
MAX_VIEW_SIDE = 8000  # 正面図がこれより大きくなるなら、縮尺を粗くする

Progress = Callable[[float, float, str], None]


def load(path: Path) -> tuple[np.ndarray, str]:
    """RGB の float32（0〜1）と、"linear"（DNG）か "display"（JPEG など）。"""
    if path.suffix.lower() == ".dng":
        return np.ascontiguousarray(stacking.read_dng(path)[..., ::-1]), "linear"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # EXIF の向きは OpenCV が直す
    if bgr is None:
        raise ValueError("画像を読めません（JPEG か DNG で送ってください。HEIC は読めません）")
    return np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32) / 255, "display"


def detection_image(rgb: np.ndarray, source: str) -> tuple[np.ndarray, float]:
    """マーカーとカードを探すための 8 bit の BGR。リニアなら明るさを合わせてガンマをかける。"""
    white = float(np.percentile(rgb, 99.5)) if source == "linear" else 1.0
    srgb = color.to_srgb(rgb / white) if source == "linear" else rgb
    return (np.clip(srgb[..., ::-1], 0, 1) * 255 + 0.5).astype(np.uint8), white


def board_poly(quad: np.ndarray, mm_per_px: float) -> np.ndarray:
    """マーカーの 4 角から、板（一辺 BOARD_MM）と縁の分の外形。"""
    center = quad.mean(axis=0)
    return center + (quad - center) * (BOARD_MM / 2 + cleanup.EDGE_MM) / (MARKER_MM / 2)


def card_poly(points: np.ndarray, mm_per_px: float) -> np.ndarray:
    """パッチの角（正面図）から、カードの台紙と縁の分の外形。"""
    (cx, cy), (w, h), angle = cv2.minAreaRect(points.astype(np.float32))
    grow = 2 * (CARD_BORDER_MM + cleanup.EDGE_MM) / mm_per_px
    return cv2.boxPoints(((cx, cy), (w + grow, h + grow), angle)).astype(np.float64)


def process_upload(
    src: Path,
    out_dir: Path,
    progress: Progress | None = None,
    layout: framing.Layout | None = None,
) -> dict:
    layout = layout or framing.Layout()
    started = time.time()
    steps = ["読み込み", "色の補正", "傾きの補正", "服とフックを探す", "マーカーを消す", "書き出し"]
    report: dict = {"source_file": src.name, "warnings": []}
    warnings = report["warnings"]

    def step(i: int) -> None:
        if progress:
            progress(i, len(steps), steps[i] if i < len(steps) else "完成")

    step(0)
    rgb, source = load(src)
    report["source"] = source
    report["source_size"] = [rgb.shape[1], rgb.shape[0]]
    detect, white = detection_image(rgb, source)

    step(1)
    quads = color.find_card(detect)
    if quads is not None:
        correction = color.fit_card(color.sample_patches(rgb, quads), source)
        linear = correction.apply(rgb.reshape(-1, 3)).reshape(rgb.shape)
        report["color"] = correction.report()
    else:
        warnings.append("カラーカードが見つからないため、色は補正していません")
        linear = rgb / white if source == "linear" else color.to_linear(rgb)
        report["color"] = None
    del rgb
    bgr = (color.to_srgb(linear)[..., ::-1] * 255).astype(np.float32)
    del linear

    step(2)
    gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
    mm = rectify.MM_PER_PX
    view = rectify.wall_view(gray, mm)
    while max(view.size) > MAX_VIEW_SIDE:
        mm *= 1.5
        view = rectify.wall_view(gray, mm)
    if len(view.markers) < 4:
        warnings.append(
            f"マーカーが {len(view.markers)} 枚しか見つかりません（傾きの補正の精度が落ちます）"
        )
    report["wall"] = view.report()
    wall = np.clip(view.warp(bgr), 0, 255)
    valid = view.valid(bgr.shape)
    del bgr
    polys = [board_poly(q, mm) for q in view.markers.values()]
    if quads is not None:
        card_points = cv2.perspectiveTransform(quads.reshape(1, -1, 2), view.matrix)[0]
        polys.append(card_poly(card_points, mm))

    step(3)
    objects = np.zeros(wall.shape[:2], np.uint8)
    for p in polys:
        cv2.fillConvexPoly(objects, np.round(p).astype(np.int32), 1)
    foreground = framing.foreground_mask(wall, valid, exclude=objects > 0)
    garment = framing.garment_component(foreground)
    edges = framing.sharp_edges(wall, mm) & valid & (objects == 0)
    hook = framing.find_hook(foreground, garment, mm, edges)
    gy, gx = np.nonzero(garment)
    bbox = (int(gx.min()), int(gy.min()), int(gx.max()), int(gy.max()))
    (x0, y0, cw, ch), scale = framing.crop_box(hook, mm, layout, bbox)
    if scale > 1:
        warnings.append(
            f"服が決まった大きさ（幅 {layout.width_mm:.0f} mm）に収まらないので、"
            f"{scale:.2f} 倍に引いて切り出しました（ハンガーの位置は同じ）"
        )
    if bbox[3] >= np.nonzero(valid.any(axis=1))[0].max() - 2:
        warnings.append("服の裾が写真の下の端で切れています（下まで入るように撮ってください）")
    # 切り出しが正面図の外に出るなら、その分だけ広げておく（あとで壁で埋める）
    h, w = wall.shape[:2]
    pad = [max(-y0, 0), max(y0 + ch - h, 0), max(-x0, 0), max(x0 + cw - w, 0)]
    if any(pad):
        top, bottom, left, right = pad
        wall = cv2.copyMakeBorder(wall, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        valid, foreground = (np.pad(m, ((top, bottom), (left, right))) for m in (valid, foreground))
        polys = [p + [left, top] for p in polys]
        x0, y0 = x0 + left, y0 + top
    report["crop"] = {
        "hook_px": [round(hook[0], 1), round(hook[1], 1)],
        "garment_px": list(bbox),
        "box_px": [x0, y0, cw, ch],
        "scale": round(scale, 3),
        "width_mm": round(cw * mm, 1),
        "height_mm": round(ch * mm, 1),
    }

    step(4)
    filler, report["removed"] = cleanup.remove_objects(wall, polys, valid, foreground, mm)
    missing = np.zeros(valid.shape, bool)
    missing[y0 : y0 + ch, x0 : x0 + cw] = ~valid[y0 : y0 + ch, x0 : x0 + cw]
    share = missing.sum() / (cw * ch)
    if share > 0:
        filler.fill(missing, keep_protected=False, degree=1)
    if share > 0.005:
        warnings.append(f"切り出しの {share:.1%} が写っていなかったので、壁で埋めました")

    step(5)
    final = fit(wall[y0 : y0 + ch, x0 : x0 + cw] / 255, min(layout.side, FINAL_SIDE))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jpeg(out_dir / "preview.jpg", fit(final, PREVIEW_SIDE), 85)
    report["final_size"] = [final.shape[1], final.shape[0]]
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_jpeg(out_dir / "final.jpg", final, 95)  # 最後に書く（final.jpg があれば完成）
    step(len(steps))
    return report
