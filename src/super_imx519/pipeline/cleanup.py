"""正面図から、4 隅のマーカーの板とカラーカードを消して、壁で埋める。

埋め方（服の画素には手を加えない）:
- 色: 消す範囲の外側の帯から、壁の色を 2 次曲面で当てはめる（照明のむらに沿う）
- 質感: 近くの壁から小さなブロックを集めて重ね、細かい凹凸とむら（σ 20 px より細かい成分）を足す。
  ブロックの重なりは、分散が変わらないよう重みを正規化してなじませる
- 影: 板やカードには厚みがあり、壁に影が落ちる。物の縁から 40 mm 以内で、当てはめた壁より少し暗い
  （2〜30 段）所を影とみなして一緒に消す。もっと暗いもの・輪郭のくっきりしたもの（フックや
  ハンガー）・服とその周りは、影とみなさず書き換えない

画像は BGR の float32（0〜255、sRGB）。
"""

import cv2
import numpy as np

from .framing import sharp_edges

EDGE_MM = 12.0  # 板の縁の分だけ、物より広めに消す
SHADOW_MM = 40.0  # 影を探す範囲（物の縁から）
SHADOW_LEVEL = (2.0, 30.0)  # 周りの壁よりこの範囲で暗ければ影
RING_MM = (4.0, 30.0)  # 壁の色を当てはめる帯（消す範囲の外側）
PROTECT_MM = 6.0  # 服・ハンガーの周りで書き換えない幅
SIGMA = 20.0  # これより細かい成分を「質感」とみなす（画素）
BLOCK, OVERLAP = 96, 32  # 質感を借りるブロックの一辺と重なり（画素）
KERNEL = np.ones((3, 3), np.uint8)


def _dilate(mask: np.ndarray, px: float) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), KERNEL, iterations=max(int(px), 1)) > 0


def _poly_mask(shape: tuple[int, ...], poly: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 1)
    return mask > 0


class WallFiller:
    """壁の色と質感で、指定した範囲を埋める。"""

    def __init__(
        self,
        image: np.ndarray,
        valid: np.ndarray,
        nonwall: np.ndarray,
        protect: np.ndarray,
        mm_per_px: float,
        seed: int = 0,
    ):
        self.image, self.valid, self.protect = image, valid, protect
        self.unfilled = np.zeros(
            image.shape[:2], bool
        )  # まだ消していない物（壁の色の推定に使わない）
        self.px = 1 / mm_per_px
        h, w = image.shape[:2]
        self.yy, self.xx = np.mgrid[0:h, 0:w]
        self.gray = cv2.GaussianBlur(image.mean(axis=2), (0, 0), 4)
        # 質感 = 壁の画素だけでぼかした面との差（壁でない所がぼかしに混ざって偏らないように）
        wall = (~_dilate(nonwall, 6)).astype(np.float32)
        base = cv2.GaussianBlur(image * wall[..., None], (0, 0), SIGMA)
        base /= np.maximum(cv2.GaussianBlur(wall, (0, 0), SIGMA), 1e-3)[..., None]
        self.texture = image - base
        # 質感を借りてよいブロック: 壁でない所から十分に離れ、強い縁の無い所
        blocked = _dilate(nonwall, 2 * SIGMA) | (np.abs(self.texture).max(axis=2) > 20)
        integral = cv2.integral((~blocked).astype(np.uint8))
        by, bx = np.mgrid[0 : h - BLOCK : 16, 0 : w - BLOCK : 16]
        inside = (
            integral[by + BLOCK, bx + BLOCK]
            - integral[by, bx + BLOCK]
            - integral[by + BLOCK, bx]
            + integral[by, bx]
        )
        full = inside == BLOCK * BLOCK
        self.sources = np.stack([by[full], bx[full]], axis=1)
        self.rng = np.random.default_rng(seed)
        ramp = np.minimum(1, (np.arange(BLOCK) + 0.5) / OVERLAP)
        ramp = np.minimum(ramp, ramp[::-1])
        self.window = np.outer(ramp, ramp)[..., None].astype(np.float32)

    def fit(self, mask: np.ndarray, where: np.ndarray, degree: int = 2):
        """mask の外側の帯から壁の色を当てはめ、where の画素での値を返す。

        帯が片側しか無いと 2 次曲面は外へ伸ばしたときに暴れるので、当てはめた値が帯の色の範囲を
        超えるなら、1 次の面、平均の色の順に単純にする。
        """
        ring = _dilate(mask, RING_MM[1] * self.px) & ~_dilate(mask, RING_MM[0] * self.px)
        ring &= self.valid & ~self.protect & ~self.unfilled
        ring_px = self.image[ring]
        if len(ring_px) < 50:
            raise ValueError("消す範囲の周りに壁が写っていません")
        ok = np.abs(ring_px - np.median(ring_px, axis=0)).max(axis=1) < 25
        if ok.sum() < 50:  # 帯の明るさのむらが大きいときは、外れを除かずに当てはめる
            ok[:] = True
        ry, rx, values = self.yy[ring][ok], self.xx[ring][ok], ring_px[ok].astype(np.float64)
        cy, cx = ry.mean(), rx.mean()
        lo, hi = np.percentile(values, 1, axis=0) - 6, np.percentile(values, 99, axis=0) + 6
        fy, fx = np.nonzero(where)

        def design(x, y, d):
            u, v = (x - cx) / 100, (y - cy) / 100
            cols = (
                [np.ones_like(u)]
                + ([u, v] if d >= 1 else [])
                + ([u * u, u * v, v * v] if d >= 2 else [])
            )
            return np.stack(cols, axis=1)

        for d in range(degree, -1, -1):
            coef, *_ = np.linalg.lstsq(design(rx, ry, d), values, rcond=None)
            out = design(fx.astype(np.float64), fy.astype(np.float64), d) @ coef
            if d == 0 or ((out >= lo).all() and (out <= hi).all()):
                break
        return fy, fx, np.clip(out, lo, hi).astype(np.float32)

    def synth_texture(self, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
        """範囲を覆う質感を、近くの壁のブロックを重ねて作る。"""
        if len(self.sources) == 0:
            return np.zeros((len(fy), 3), np.float32)
        y0, x0 = fy.min() - OVERLAP, fx.min() - OVERLAP
        hh, ww = fy.max() - y0 + BLOCK, fx.max() - x0 + BLOCK
        acc = np.zeros((hh, ww, 3), np.float32)
        acc_w2 = np.zeros((hh, ww, 1), np.float32)
        # 近いブロックほど、同じ照明・同じ壁の質感なので、近い 60 個から選ぶ
        d = np.hypot(self.sources[:, 0] - fy.mean(), self.sources[:, 1] - fx.mean())
        nearest = self.sources[np.argsort(d)[:60]]
        step = BLOCK - OVERLAP
        for ty in range(0, hh - BLOCK + 1, step):
            for tx in range(0, ww - BLOCK + 1, step):
                sy, sx = nearest[self.rng.integers(len(nearest))]
                block = self.texture[sy : sy + BLOCK, sx : sx + BLOCK]
                acc[ty : ty + BLOCK, tx : tx + BLOCK] += self.window * block
                acc_w2[ty : ty + BLOCK, tx : tx + BLOCK] += self.window**2
        synth = acc / np.sqrt(np.maximum(acc_w2, 1e-6))
        return synth[fy - y0, fx - x0]

    def fill(self, mask: np.ndarray, keep_protected: bool = True, degree: int = 2) -> int:
        """mask を埋める。縁は元の画像となめらかにつなぐ。埋めた画素数を返す。"""
        inner = _dilate(mask, RING_MM[0] * self.px)
        if keep_protected:
            inner &= mask | ~self.protect
        fy, fx, smooth = self.fit(mask, inner, degree)
        fill = smooth + self.synth_texture(fy, fx)
        alpha = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 2.0)[fy, fx][:, None]
        alpha = np.where(mask[fy, fx][:, None], 1.0, alpha)
        self.image[fy, fx] = alpha * fill + (1 - alpha) * self.image[fy, fx]
        self.unfilled &= ~inner
        return len(fy)

    def with_shadow(self, mask: np.ndarray, others: np.ndarray) -> np.ndarray:
        """mask に、つながっている影を足す。"""
        near = _dilate(mask, SHADOW_MM * self.px) & self.valid & ~self.protect & ~others
        for _ in range(2):
            fy, fx, fit = self.fit(mask, near)
            diff = self.gray[fy, fx] - fit.mean(axis=1)
            dark = np.zeros(mask.shape, np.uint8)
            dark[fy, fx] = (diff < -SHADOW_LEVEL[0]) & (diff > -SHADOW_LEVEL[1])
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, KERNEL, iterations=2)
            _, labels = cv2.connectedComponents(dark | mask.astype(np.uint8))
            touching = np.unique(labels[mask])
            mask = np.isin(labels, touching[touching > 0])
            mask = (
                cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL, iterations=6) > 0
            )
        return mask


def protect_mask(
    image: np.ndarray, foreground: np.ndarray, objects: np.ndarray, mm_per_px: float
) -> np.ndarray:
    """書き換えてはいけない所。

    服・ハンガー・フックとその周りと、輪郭のくっきりした物（消す物そのものは除く）。
    """
    px = 1 / mm_per_px
    edges = sharp_edges(image, mm_per_px) & ~_dilate(objects, 8)
    return _dilate(foreground & ~objects, PROTECT_MM * px) | _dilate(edges, 3 * px)


def remove_objects(
    image: np.ndarray,
    polys: list[np.ndarray],
    valid: np.ndarray,
    foreground: np.ndarray,
    mm_per_px: float,
) -> tuple[WallFiller, list[dict]]:
    """polys（板・カードの外形。正面図の画素）を影ごと消す。image は書き換える。"""
    objects = np.zeros(image.shape[:2], bool)
    masks = [_poly_mask(image.shape, p) for p in polys]
    for m in masks:
        objects |= m
    protect = protect_mask(image, foreground, objects, mm_per_px)
    nonwall = objects | foreground | ~valid
    filler = WallFiller(image, valid, nonwall, protect, mm_per_px)
    filler.unfilled = objects.copy()
    report = []
    for i, mask in enumerate(masks):
        others = np.zeros_like(objects)
        for j, m in enumerate(masks):
            if j != i:
                others |= m
        grown = filler.with_shadow(mask, others)
        n = filler.fill(grown)
        report.append({"filled_px": n, "shadow_px": int((grown & ~mask).sum())})
    return filler, report
