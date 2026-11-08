"""スマホ（iPhone）で撮った 1 枚から、商品画像を作る（スタジオがアップロードごとに呼ぶ）。

1. 読み込み: DNG（ProRAW）はリニアに現像し、JPEG はそのまま（表示用）として扱う
2. 色: カラーカードを見つけて補正する（color.py）。無ければ補正せず、report に書く
3. 形: 4 隅のマーカーで、壁の正面図（0.5 mm/px）にする（rectify.py）。壁の配置の較正値
   （DEFAULT_WALL。calibrate.py）があれば使う
4. 構図: フックが上の端に掛かる縦 2:3 の枠を、壁面座標（実寸）で決める（framing.py）。服が
   はみ出すときだけ引く。較正値にフックの位置があれば、その位置を基準にする
5. 掃除: マーカーの板とカードを、影ごと壁で埋める。切り出しが写っていない所に掛かれば、そこも埋める
6. 仕上げ: 背景（壁）の明るさと色味を既存の商品画像に揃え、軽くシャープにする（finish.py）
7. 長辺 2048px 以内の final.jpg（品質 95）、画面用の preview.jpg、report.json を書く

iPhone の ProRAW は、保存前に明暗の差が圧縮されている（推定）ので、色の正しさは近似にとどまる。
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..markers import BOARD_MM, MARKER_MM, MODULES, marker_modules
from . import cleanup, color, framing, rectify, stacking
from .finish import Finish, finish
from .process import FINAL_SIDE, PREVIEW_SIDE, fit, write_jpeg

SUFFIXES = {".jpg", ".jpeg", ".png", ".dng"}
DEFAULT_WALL = Path("~/data/super_imx519/wall.json").expanduser()  # 壁の配置の較正値
HOOK_MOVED_MM = 30.0  # 見つけたフックが較正値からこれ以上離れていたら知らせる
CARD_BORDER_MM = 12.0  # パッチの外側の、カードの黒い台紙の幅
DEFAULT_LOOK = Finish()
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


def board_poly(quad: np.ndarray) -> np.ndarray:
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

    erase: list[np.ndarray]  # 消す物（マーカーの板とカード）の画素。服に隠れた所は除く
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


WALL, PAPER, INK = 0, 1, 2  # 板の周りの見えるはずのもの


def board_pattern(
    view: rectify.WallView, marker: tuple[float, float, float], marker_id: int, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """region の画素に見えるはずのもの。(ys, xs, 種類（WALL・PAPER・INK）, 確かか)。

    境目の近く（位置のずれやぼけで入れ替わりうる所。2 mm 以内）は「確かでない」とする。
    """
    ys, xs = np.nonzero(region)
    mm = view.to_mm(np.stack([xs, ys], axis=1).astype(np.float64))
    x, y, angle = marker
    local = (mm - [x, y]) @ rectify.rotation(angle)  # マーカーの向きにそろえた座標（mm）
    module = MARKER_MM / MODULES
    u = (local + MARKER_MM / 2) / module  # マス単位（0〜MODULES がマーカー）
    inside = ((u >= 0) & (u < MODULES)).all(axis=1)
    reach = np.abs(local).max(axis=1)
    kind = np.where(reach < BOARD_MM / 2, PAPER, WALL)
    cells = np.clip(np.floor(u[inside]).astype(int), 0, MODULES - 1)
    kind[inside] = np.where(marker_modules(marker_id)[cells[:, 1], cells[:, 0]], INK, PAPER)
    edge = np.minimum(np.abs(reach - MARKER_MM / 2), np.abs(reach - BOARD_MM / 2))
    grid = np.abs(u - np.round(u)).min(axis=1) * module
    edge = np.where(inside, np.minimum(edge, grid), edge)
    return ys, xs, kind, edge > 2.0


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image / 255, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)


def ink_color(image: np.ndarray, view: rectify.WallView, layout: rectify.WallLayout) -> np.ndarray:
    """見えているマーカーで測った、印刷の黒の色（L* は板の白に対する比、a*、b*）。"""
    lab = _lab(image)
    samples = []
    for i, quad in view.markers.items():
        if i not in layout.markers:
            continue
        region = cleanup.poly_mask(image.shape, board_poly(quad))
        ys, xs, kind, sure = board_pattern(view, layout.markers[i], i, region)
        values = lab[ys, xs]
        paper, ink = values[sure & (kind == PAPER)], values[sure & (kind == INK)]
        if len(paper) > 50 and len(ink) > 50:
            ink = np.median(ink, axis=0)
            samples.append([ink[0] / max(np.median(paper[:, 0]), 1.0), ink[1], ink[2]])
    return np.median(samples, axis=0) if samples else np.array([0.35, 0.0, 0.0])


def covered_part(
    image: np.ndarray,
    region: np.ndarray,
    view: rectify.WallView,
    marker: tuple[float, float, float],
    marker_id: int,
    garment: np.ndarray,
    ink: np.ndarray,
) -> np.ndarray:
    """板（region）のうち、服に隠れている所。

    板の柄は分かっているので、画素ごとに「見えるはずの色」（板の白・印刷の黒・壁）と「服の色」の
    どちらに近いか（Lab の距離）で決める。黒は、見えているマーカーで測った色（ink_color）から
    求める。両者が近くて見分けられない画素（黒い服の上の黒いマスなど）と、柄の境目は、周りの決まった
    画素から埋めて決める。板の外側は、服なら隠れている側、壁なら見えている側の手がかりにする。
    """
    ring = cleanup.dilate(region, 15 / view.mm_per_px) & ~region
    near_garment = ring & garment
    if near_garment.sum() < 20:
        return np.zeros(region.shape, bool)
    ys, xs, kind, sure = board_pattern(view, marker, marker_id, region)
    lab = _lab(image)
    values = lab[ys, xs]
    paper = values[sure & (kind == PAPER)]
    if len(paper) < 50:
        return np.zeros(region.shape, bool)
    white = paper[paper[:, 0] >= np.percentile(paper[:, 0], 75)]  # 隠れていない白
    levels = np.zeros((3, 3), np.float32)
    levels[WALL] = np.median(lab[ring & ~garment], axis=0)
    levels[PAPER] = np.median(white, axis=0)
    levels[INK] = [levels[PAPER][0] * ink[0], ink[1], ink[2]]
    cloth = np.median(lab[near_garment], axis=0)
    expected = levels[kind]
    to_board = np.linalg.norm(values - expected, axis=1)
    to_cloth = np.linalg.norm(values - cloth, axis=1)
    decisive = sure & (np.linalg.norm(expected - cloth, axis=1) > 12)
    vote = np.where(to_cloth < to_board, -1.0, 1.0)
    evidence = np.zeros(region.shape, np.float32)  # +1 見えている、-1 隠れている、0 分からない
    evidence[ys[decisive], xs[decisive]] = vote[decisive]
    kernel = np.ones((5, 5), np.uint8)
    for sign in (-1, 1):  # 細かい誤り（ノイズ・映り込み）を除く
        keep = cv2.morphologyEx((evidence == sign).astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        evidence[(evidence == sign) & ~keep] = 0
    evidence[near_garment] = -1
    evidence[ring & ~garment] = 1
    y0, y1 = max(ys.min() - 40, 0), ys.max() + 41
    x0, x1 = max(xs.min() - 40, 0), xs.max() + 41
    e = evidence[y0:y1, x0:x1]
    stack = np.repeat(e[..., None], 3, axis=2)  # pull_push は 3 チャネルで扱う
    field = cleanup.pull_push(stack, (e != 0).astype(np.float32))[..., 0]
    covered = np.zeros(region.shape, bool)
    covered[y0:y1, x0:x1] = (field < 0) & region[y0:y1, x0:x1]
    # 服とつながった所だけを残す
    n, labels = cv2.connectedComponents((covered | near_garment).astype(np.uint8))
    touching = np.unique(labels[near_garment])
    return covered & np.isin(labels, touching[touching > 0])


def analyze(
    image: np.ndarray,
    valid: np.ndarray,
    view: rectify.WallView,
    quads: np.ndarray | None,
    layout: rectify.WallLayout | None = None,
) -> Scene:
    """正面図（BGR、0〜255）から、消す物・服・フックを見つける。

    較正値（layout）があれば、見つからなかったマーカー（服の裾などに隠れたもの）の板も、較正の位置
    で消す。服に隠れている所は消さない。
    """
    mm = view.mm_per_px
    erase = [cleanup.poly_mask(image.shape, board_poly(q)) for q in view.markers.values()]
    if quads is not None:
        card_points = cv2.perspectiveTransform(quads.reshape(1, -1, 2), view.matrix)[0]
        erase.append(cleanup.poly_mask(image.shape, card_poly(card_points, mm)))
    hidden = []  # (ID, 板の範囲)
    for i in sorted(set(layout.markers) - set(view.markers) if layout else []):
        region = cleanup.poly_mask(image.shape, board_poly(view.to_view(layout.corners(i))))
        if region.sum() and (region & valid).sum() > 0.8 * region.sum():
            hidden.append((i, region & valid))
    empty = np.zeros(image.shape[:2], bool)
    objects = np.logical_or.reduce([empty, *erase, *(r for _, r in hidden)])
    foreground = framing.foreground_mask(image, valid, exclude=objects)
    garment = framing.garment_component(foreground)
    ink = ink_color(image, view, layout) if hidden else None
    for i, region in hidden:
        covered = covered_part(image, region, view, layout.markers[i], i, garment, ink)
        foreground |= covered
        erase.append(region & ~covered)
    if hidden:
        garment = framing.garment_component(foreground)
    edges = framing.sharp_edges(image, mm) & valid & ~objects
    hook = framing.find_hook(foreground, garment, mm, edges)
    gy, gx = np.nonzero(garment)
    bbox = (int(gx.min()), int(gy.min()), int(gx.max()), int(gy.max()))
    return Scene(erase, foreground, garment, hook, bbox)


def correct_color(
    rgb: np.ndarray, source: str, white: float, detect: np.ndarray, report: dict
) -> tuple[np.ndarray, np.ndarray | None]:
    """カラーカードで色を合わせた BGR（0〜255、sRGB）と、カードのパッチの角（無ければ None）。"""
    quads = color.find_card(detect)
    if quads is not None:
        correction = color.fit_card(color.sample_patches(rgb, quads), source)
        linear = correction.apply(rgb.reshape(-1, 3)).reshape(rgb.shape)
        report["color"] = correction.report()
    else:
        report["warnings"].append("カラーカードが見つからないため、色は補正していません")
        linear = rgb / white if source == "linear" else color.to_linear(rgb)
        report["color"] = None
    return (color.to_srgb(linear)[..., ::-1] * 255).astype(np.float32), quads


def marker_warning(view: rectify.WallView, wall_layout: rectify.WallLayout | None) -> str | None:
    expected = len(wall_layout.markers) if wall_layout else 4
    found = len(view.markers)
    usable = found + len(view.located)
    if found >= expected:
        return None
    if usable >= expected:
        return None  # 隠れたマーカーも、板の柄で位置が分かった
    if usable >= 2 and wall_layout:
        note = "較正値があるので補正はできますが、下の板が隠れていると縦の縮尺が数 % ずれます"
    elif usable == 1:
        note = "離れた所ほど傾きの補正がずれます。2 枚以上写してください"
    else:
        note = "傾きの補正の精度が落ちます"
    return f"マーカーが {usable} 枚しか見つかりません（{note}）"


def hook_position(
    scene: Scene, view: rectify.WallView, wall_layout: rectify.WallLayout | None, report: dict
) -> tuple[float, float]:
    """構図の基準にするフックの位置（正面図の画素）。較正値があればそれ、無ければ見つけた位置。"""
    if not (wall_layout and wall_layout.hook_mm):
        return scene.hook
    hook = tuple(view.to_view([wall_layout.hook_mm])[0])
    moved = float(np.hypot(*np.subtract(scene.hook, hook))) * view.mm_per_px
    report["hook_detected_offset_mm"] = round(moved, 1)
    if moved > HOOK_MOVED_MM:
        report["warnings"].append(
            f"見つけたフックが較正の位置から {moved:.0f} mm 離れています"
            "（フックやマーカーを動かしたなら、較正し直してください）"
        )
    return hook


def process_upload(
    src: Path,
    out_dir: Path,
    progress: Progress | None = None,
    composition: framing.Layout | None = None,
    wall_path: Path | None = DEFAULT_WALL,
    look: Finish | None = DEFAULT_LOOK,
) -> dict:
    """look が None なら仕上げ（背景の明るさと色味、シャープ）をかけない。"""
    composition = composition or framing.Layout()
    wall_layout = rectify.WallLayout.load(wall_path)
    started = time.time()
    steps = [
        "読み込み",
        "色の補正",
        "傾きの補正",
        "服とフックを探す",
        "マーカーを消す",
        "仕上げと書き出し",
    ]
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
    bgr, quads = correct_color(rgb, source, white, detect, report)
    del rgb

    step(2)
    view, mm = make_view(detect, wall_layout)
    if note := marker_warning(view, wall_layout):
        warnings.append(note)
    report["wall"] = view.report()
    wall = np.clip(view.warp(bgr), 0, 255)
    valid = view.valid(bgr.shape)
    del bgr

    step(3)
    scene = analyze(wall, valid, view, quads, wall_layout)
    erase, foreground, bbox = scene.erase, scene.foreground, scene.bbox
    hook = hook_position(scene, view, wall_layout, report)
    (x0, y0, cw, ch), scale = framing.crop_box(hook, mm, composition, bbox)
    if scale > 1:
        warnings.append(
            f"服が枠（幅 {composition.width_mm:.0f} mm）からはみ出すので、この商品だけ "
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
        erase = [np.pad(m, ((top, bottom), (left, right))) for m in erase]
        x0, y0 = x0 + left, y0 + top
    report["crop"] = {
        "hook_px": [round(hook[0], 1), round(hook[1], 1)],
        "hook_from": "較正値" if wall_layout and wall_layout.hook_mm else "検出",
        "garment_px": list(bbox),
        "scale": round(scale, 3),
        "box_px": [x0, y0, cw, ch],
        "width_mm": round(cw * mm, 1),
        "height_mm": round(ch * mm, 1),
    }

    step(4)
    filler, report["removed"] = cleanup.remove_objects(wall, erase, valid, foreground, mm)
    missing = np.zeros(valid.shape, bool)
    missing[y0 : y0 + ch, x0 : x0 + cw] = ~valid[y0 : y0 + ch, x0 : x0 + cw]
    share = missing.sum() / (cw * ch)
    if share > 0:
        filler.fill(missing, keep_protected=False)
    if share > 0.005:
        warnings.append(f"切り出しの {share:.1%} が写っていなかったので、壁で埋めました")

    step(5)
    final = fit(wall[y0 : y0 + ch, x0 : x0 + cw] / 255, min(composition.side, FINAL_SIDE))
    if look is not None:
        subject = foreground[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
        subject = cv2.resize(subject, final.shape[1::-1], interpolation=cv2.INTER_AREA) > 0.1
        final, report["finish"] = finish(final, subject, cw * mm / final.shape[1], look)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jpeg(out_dir / "preview.jpg", fit(final, PREVIEW_SIDE), 85)
    report["final_size"] = [final.shape[1], final.shape[0]]
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_jpeg(out_dir / "final.jpg", final, 95)  # 最後に書く（final.jpg があれば完成）
    step(len(steps))
    return report
