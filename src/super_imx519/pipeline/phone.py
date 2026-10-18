"""スマホ（iPhone）で撮った 1 枚から、商品画像を作る（スタジオがアップロードごとに呼ぶ）。

1. 読み込み: DNG（ProRAW）はリニアに現像し、JPEG はそのまま（表示用）として扱う
2. 色: カラーカードを見つけて補正する（color.py）。無ければ補正せず、report に書く
3. 形: 4 隅のマーカーで、壁の正面図（0.5 mm/px）にする（rectify.py）。壁の配置の較正値
   （DEFAULT_WALL。calibrate.py）があれば使う
4. 構図: フックが上の端に掛かる縦 2:3 の枠を、壁面座標（実寸）で決める（framing.py）。服の大きさ
   では変えない。較正値にフックの位置があれば、その位置を基準にする（どの写真でも枠が同じ位置になる）
5. 掃除: マーカーの板とカードを、影ごと壁で埋める。切り出しが写っていない所に掛かれば、そこも埋める
6. 長辺 2048px 以内の final.jpg（品質 95）、画面用の preview.jpg、report.json を書く

iPhone の ProRAW は、保存前に明暗の差が圧縮されている（推定）ので、色の正しさは近似にとどまる。
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..markers import BOARD_MM, MARKER_MM
from . import cleanup, color, framing, rectify, stacking
from .process import FINAL_SIDE, PREVIEW_SIDE, fit, write_jpeg

SUFFIXES = {".jpg", ".jpeg", ".png", ".dng"}
DEFAULT_WALL = Path("~/data/super_imx519/wall.json").expanduser()  # 壁の配置の較正値
HOOK_MOVED_MM = 30.0  # 見つけたフックが較正値からこれ以上離れていたら知らせる
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


@dataclass
class Scene:
    """正面図で見つけたもの。座標は正面図の画素。"""

    polys: list[np.ndarray]  # 消す物（マーカーの板とカード）の外形
    objects: np.ndarray  # polys を塗ったもの
    foreground: np.ndarray
    garment: np.ndarray
    hook: tuple[float, float]  # 見つけたフックの上端
    bbox: tuple[int, int, int, int]  # 服の範囲 (x0, y0, x1, y1)


def make_view(
    detect: np.ndarray, wall: rectify.WallLayout | None
) -> tuple[rectify.WallView, float]:
    gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
    mm = rectify.MM_PER_PX
    view = rectify.wall_view(gray, mm, wall)
    while max(view.size) > MAX_VIEW_SIDE:
        mm *= 1.5
        view = rectify.wall_view(gray, mm, wall)
    return view, mm


def analyze(
    image: np.ndarray, valid: np.ndarray, view: rectify.WallView, quads: np.ndarray | None
) -> Scene:
    """正面図（BGR、0〜255）から、消す物・服・フックを見つける。"""
    mm = view.mm_per_px
    polys = [board_poly(q, mm) for q in view.markers.values()]
    if quads is not None:
        card_points = cv2.perspectiveTransform(quads.reshape(1, -1, 2), view.matrix)[0]
        polys.append(card_poly(card_points, mm))
    objects = np.zeros(image.shape[:2], np.uint8)
    for p in polys:
        cv2.fillConvexPoly(objects, np.round(p).astype(np.int32), 1)
    foreground = framing.foreground_mask(image, valid, exclude=objects > 0)
    garment = framing.garment_component(foreground)
    edges = framing.sharp_edges(image, mm) & valid & (objects == 0)
    hook = framing.find_hook(foreground, garment, mm, edges)
    gy, gx = np.nonzero(garment)
    bbox = (int(gx.min()), int(gy.min()), int(gx.max()), int(gy.max()))
    return Scene(polys, objects > 0, foreground, garment, hook, bbox)


def process_upload(
    src: Path,
    out_dir: Path,
    progress: Progress | None = None,
    composition: framing.Layout | None = None,
    wall_path: Path | None = DEFAULT_WALL,
) -> dict:
    composition = composition or framing.Layout()
    wall_layout = rectify.WallLayout.load(wall_path)
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
    view, mm = make_view(detect, wall_layout)
    expected = len(wall_layout.markers) if wall_layout else 4
    found = len(view.markers)
    if found < expected:
        if found >= 2 and wall_layout:
            note = "較正値があるので補正はできます"
        elif found == 1:
            note = "離れた所ほど傾きの補正がずれます。2 枚以上写してください"
        else:
            note = "傾きの補正の精度が落ちます"
        warnings.append(f"マーカーが {found} 枚しか見つかりません（{note}）")
    report["wall"] = view.report()
    wall = np.clip(view.warp(bgr), 0, 255)
    valid = view.valid(bgr.shape)
    del bgr

    step(3)
    scene = analyze(wall, valid, view, quads)
    polys, foreground, bbox = scene.polys, scene.foreground, scene.bbox
    hook = scene.hook
    if wall_layout and wall_layout.hook_mm:
        hook = tuple(view.to_view([wall_layout.hook_mm])[0])
        moved = float(np.hypot(*(np.subtract(scene.hook, hook)))) * mm
        report["hook_detected_offset_mm"] = round(moved, 1)
        if moved > HOOK_MOVED_MM:
            warnings.append(
                f"見つけたフックが較正の位置から {moved:.0f} mm 離れています"
                "（フックやマーカーを動かしたなら、較正し直してください）"
            )
    x0, y0, cw, ch = framing.crop_box(hook, mm, composition)
    sides = framing.overflow((x0, y0, cw, ch), bbox)
    if sides:
        warnings.append(
            f"服が枠（幅 {composition.width_mm:.0f} mm）の{'・'.join(sides)}からはみ出しています"
            "（縮尺はどの商品も同じにしているので、引いては切り出しません）"
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
        "hook_from": "較正値" if wall_layout and wall_layout.hook_mm else "検出",
        "garment_px": list(bbox),
        "box_px": [x0, y0, cw, ch],
        "width_mm": round(cw * mm, 1),
        "height_mm": round(ch * mm, 1),
    }

    step(4)
    filler, report["removed"] = cleanup.remove_objects(wall, polys, valid, foreground, mm)
    missing = np.zeros(valid.shape, bool)
    missing[y0 : y0 + ch, x0 : x0 + cw] = ~valid[y0 : y0 + ch, x0 : x0 + cw]
    share = missing.sum() / (cw * ch)
    if share > 0:
        filler.fill(missing, keep_protected=False)
    if share > 0.005:
        warnings.append(f"切り出しの {share:.1%} が写っていなかったので、壁で埋めました")

    step(5)
    final = fit(wall[y0 : y0 + ch, x0 : x0 + cw] / 255, min(composition.side, FINAL_SIDE))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jpeg(out_dir / "preview.jpg", fit(final, PREVIEW_SIDE), 85)
    report["final_size"] = [final.shape[1], final.shape[0]]
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_jpeg(out_dir / "final.jpg", final, 95)  # 最後に書く（final.jpg があれば完成）
    step(len(steps))
    return report
