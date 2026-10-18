"""服（とハンガー・フック）を見つけ、商品画像の構図（縦 2:3）で切り出す範囲を決める。

構図は 85-store の商品画像に合わせる: 壁のフックが上の端に少し掛かり、服はフックの真下で左右の
中央。縮尺は実寸（壁面座標の mm）で固定し、どの商品も同じ大きさに写るようにする。

服の見つけ方: 壁を Lab の滑らかな面（2 次曲面）で当てはめ、そこからの差で前景を決める。
服が壁に落とす影は「少し暗く、色はほぼ同じ」なので、次のどれかに当たるものだけを前景にする。
- 色が違う（Δa*b* が大きい）: 色のある服
- 壁より明るい: 白い服、金属のフックの光ったところ
- 壁よりはっきり暗い（影より暗い）: 黒・紺の服
"""

from dataclasses import dataclass

import cv2
import numpy as np

CHROMA_DE = 10.0  # 色の差（Δa*b*）がこれより大きければ前景
BRIGHTER_L = 8.0  # 壁よりこれだけ明るければ前景（ΔL*）
DARKER_L = 25.0  # 壁よりこれだけ暗ければ前景（影はこれより明るい）
FIT_STEP = 8  # 壁の面を当てはめるときの間引き


@dataclass
class Layout:
    width_mm: float = 900.0  # 切り出す幅（実寸）。どの商品も、まずこの縮尺で切り出す
    aspect: float = 3 / 2  # 高さ / 幅
    hook_cut_mm: float = 40.0  # フックの上端から、上の端までを下げる量（フックが端に掛かる）
    bottom_margin: float = 0.05  # 服の裾の下に、少なくとも高さのこの割合を空ける
    side_margin: float = 0.05  # 服の左右に、少なくとも幅のこの割合を空ける
    side: int = 2048  # 書き出す長辺の上限


def _design(x: np.ndarray, y: np.ndarray, w: int, h: int) -> np.ndarray:
    u, v = x / w - 0.5, y / h - 0.5
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def foreground_mask(
    bgr: np.ndarray, valid: np.ndarray, exclude: np.ndarray | None = None
) -> np.ndarray:
    """壁ではない画素（服・ハンガー・フック・影）。exclude（マーカーやカード）は前景にしない。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(np.clip(bgr / 255, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)
    usable = valid if exclude is None else valid & ~exclude
    ys, xs = np.mgrid[0:h:FIT_STEP, 0:w:FIT_STEP]
    keep = usable[ys, xs]
    samples = lab[ys, xs][keep]
    a = _design(xs[keep].astype(np.float64), ys[keep].astype(np.float64), w, h)
    # 服が大きく写っていても壁に合うよう、外れた画素を除いて当てはめ直す
    inliers = np.ones(len(samples), bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(a[inliers], samples[inliers], rcond=None)
        resid = np.linalg.norm(a @ coef - samples, axis=1)
        inliers = resid < max(3.0 * np.median(resid[inliers]), 2.0)
    # 面は滑らかなので、間引いた格子で求めて引き伸ばす
    grid = _design(xs.astype(np.float64), ys.astype(np.float64), w, h) @ coef
    background = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    diff = cv2.GaussianBlur(lab, (0, 0), 2) - background
    chroma = np.hypot(diff[..., 1], diff[..., 2])
    mask = (chroma > CHROMA_DE) | (diff[..., 0] > BRIGHTER_L) | (diff[..., 0] < -DARKER_L)
    mask &= valid
    if exclude is not None:
        mask &= ~exclude
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask > 0


def garment_component(foreground: np.ndarray) -> np.ndarray:
    """前景のうち一番大きいもの（服）。中の穴（明るいところで拾えなかった部分）は埋める。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    if n < 2:
        raise ValueError("服が見つかりません（壁との差が小さすぎます）")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    garment = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(garment, contours, -1, 1, thickness=cv2.FILLED)
    return garment > 0


def find_hook(
    foreground: np.ndarray, garment: np.ndarray, mm_per_px: float, edges: np.ndarray | None = None
) -> tuple[float, float]:
    """壁のフックの上端 (x, y)。

    服の上端の中心から上 250 mm・左右 60 mm の範囲で、一番上にある塊（ある程度の大きさのもの）を
    フックとみなす。ハンガーの針金は細くて拾えないことがあるので、服とつながっていなくてよい。
    edges（輪郭のくっきりした画素）を渡すと、それも塊に含める。見つからなければ服の上端を返す。
    """
    px = 1 / mm_per_px
    ys, xs = np.nonzero(garment)
    top = int(ys.min())
    cx = float(np.median(xs[ys < top + 30 * px]))
    x0, x1 = int(max(cx - 60 * px, 0)), int(min(cx + 60 * px, foreground.shape[1]))
    y0 = int(max(top - 250 * px, 0))
    window = (foreground[y0:top, x0:x1] & ~garment[y0:top, x0:x1]).astype(np.uint8)
    if edges is not None:  # 金具の灰色の部分は壁と色が近いので、輪郭でも拾う
        window |= cv2.dilate(edges[y0:top, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(window)
    blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 40 * px]
    if not blobs:
        return cx, float(top)
    hook = min(blobs, key=lambda i: stats[i, cv2.CC_STAT_TOP])
    hy = y0 + stats[hook, cv2.CC_STAT_TOP]
    hx = x0 + stats[hook, cv2.CC_STAT_LEFT] + stats[hook, cv2.CC_STAT_WIDTH] / 2
    return float(hx), float(hy)


def crop_box(
    hook: tuple[float, float],
    mm_per_px: float,
    layout: Layout,
    garment: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int, int, int], float]:
    """((x0, y0, 幅, 高さ), 縮尺の倍率)。正面図の画素。画像の外に出ることもある（呼ぶ側で埋める）。

    フックは常に左右の中央・上の端から同じ割合の位置に置く（ショップで見たときにハンガーの位置が
    揃う）。服（garment = x0, y0, x1, y1）が決まった大きさに収まらないとき（ロングコートなど）は、
    フックの位置の割合を保ったまま、収まるところまで引く（倍率 > 1）。
    """
    px = 1 / mm_per_px
    width = layout.width_mm * px
    height = width * layout.aspect
    cut = layout.hook_cut_mm * px
    scale = 1.0
    if garment is not None:
        gx0, _, gx1, gy1 = garment
        # 裾: 上の端（フック + cut·s）から裾 + 余白までが高さ·s に収まる倍率
        need_down = (gy1 - hook[1]) / (height * (1 - layout.bottom_margin) - cut)
        # 左右: フックからの遠い方の端 + 余白が、幅·s の半分に収まる倍率
        reach = max(hook[0] - gx0, gx1 - hook[0])
        need_side = reach / (width * (0.5 - layout.side_margin))
        scale = max(1.0, need_down, need_side)
    w, h, c = width * scale, height * scale, cut * scale
    box = (int(round(hook[0] - w / 2)), int(round(hook[1] + c)), int(round(w)), int(round(h)))
    return box, scale


def sharp_edges(bgr: np.ndarray, mm_per_px: float, threshold: float = 30.0) -> np.ndarray:
    """輪郭のくっきりした物（フックやハンガーの金具）の画素。

    スマホの JPEG は輪郭が強調されていて、壁の細かい凹凸も強い輪郭になる。そこで、ある程度の長さ
    （20 mm 以上）につながった輪郭だけを残す。
    """
    gray = cv2.GaussianBlur(bgr.mean(axis=2).astype(np.float32), (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edges = (grad > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = span >= 20 / mm_per_px
    keep[0] = False
    return keep[labels]
