"""複数の写真から、壁の配置（マーカーの位置と傾き、フックの位置）を較正する。

同じ壁を、位置や角度を変えて数枚撮る。未知数は、写真ごとのホモグラフィ（8 × 枚数）と、全部の写真で
共通のマーカーの配置（ID が最小のもの以外の中心 x, y と傾き）。Levenberg–Marquardt で同時に解く。

- 原点は ID が最小のマーカーの中心。向きは、全マーカーの傾きの平均が 0 になるようにとる
  （板を水平に貼ったつもりの平均を水平とみなす）。縮尺はマーカーの一辺（75 mm）で決まる
- フックは、較正した配置で正面図にした各写真で見つけ、壁面上の位置の中央値をとる
- 写真ごとの残差（mm）と、マーカーの位置・フックの位置のばらつきを report に出す
"""

import cv2
import numpy as np

from .rectify import WallLayout, _fit, least_squares, normalizer, project, square


def fit_layout(views: list[dict[int, np.ndarray]]) -> tuple[WallLayout, dict]:
    """views は写真ごとの「マーカー ID → 4 角（画素）」。2 枚以上の写真に写ったマーカーを使う。"""
    counts: dict[int, int] = {}
    for found in views:
        for i in found:
            counts[i] = counts.get(i, 0) + 1
    ids = sorted(i for i, n in counts.items() if n >= 2)
    views = [{i: q for i, q in found.items() if i in ids} for found in views]
    views = [found for found in views if found]
    if len(views) < 2 or len(ids) < 2:
        raise ValueError("較正には、マーカーが 2 枚以上写った写真が 2 枚以上要ります")
    ref, others = ids[0], ids[1:]

    # 初期値: 写真ごとに単独で解き（傾きは 0 とみなす）、マーカーの中心を写真のあいだで平均する
    norms, qs, h0, centers = [], [], [], {i: [] for i in ids}
    for found in views:
        vid = sorted(found)
        pts = np.vstack([found[i] for i in vid])
        norm = normalizer(pts)
        q = cv2.perspectiveTransform(pts[None], norm)[0]
        H, targets = _fit(pts, len(vid))
        c = targets.reshape(-1, 4, 2).mean(axis=1)
        if ref in found:
            for i, center in zip(vid, c - c[vid.index(ref)], strict=True):
                centers[i].append(center)
        Hn = H @ np.linalg.inv(norm)
        norms.append(norm)
        qs.append((vid, q))
        h0.append((Hn / Hn[2, 2]).ravel()[:8])
    init = {i: np.mean(centers[i], axis=0) if centers[i] else np.zeros(2) for i in ids}
    # 原点のマーカーが写っていない写真は、初期値で原点をずらしておく
    for v, found in enumerate(views):
        if ref not in found:
            vid = sorted(found)
            h = np.append(h0[v], 1.0).reshape(3, 3)
            shift = np.eye(3)
            proj = project(h0[v], qs[v][1]).reshape(-1, 4, 2).mean(axis=1)
            shift[:2, 2] = init[vid[0]] - proj[0]
            h = shift @ h
            h0[v] = (h / h[2, 2]).ravel()[:8]

    n_views = len(views)
    params = np.concatenate([np.concatenate(h0)] + [np.append(init[i], 0.0) for i in others])

    def unpack(p: np.ndarray) -> dict[int, tuple[float, float, float]]:
        layout = {ref: (0.0, 0.0, 0.0)}
        tail = p[8 * n_views :].reshape(-1, 3)
        for i, (x, y, angle) in zip(others, tail, strict=True):
            layout[i] = (x, y, angle)
        return layout

    def residual(p: np.ndarray) -> np.ndarray:
        layout = unpack(p)
        out = []
        for v, (vid, q) in enumerate(qs):
            targets = np.vstack([square(layout[i][:2], layout[i][2]) for i in vid])
            out.append((project(p[8 * v : 8 * v + 8], q) - targets).ravel())
        return np.concatenate(out)

    params = least_squares(residual, params, iterations=200)
    markers = unpack(params)

    # 向き: 傾きの平均が 0 になるよう、全体を回す（原点のマーカーの中心のまわり）
    mean_angle = float(np.mean([m[2] for m in markers.values()]))
    t = np.radians(-mean_angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    markers = {
        i: (*(rot @ np.array([x, y])), angle - mean_angle) for i, (x, y, angle) in markers.items()
    }

    per_view = residual(params).reshape(-1, 2)
    errors, start = [], 0
    for vid, _ in qs:
        n = 4 * len(vid)
        e = np.linalg.norm(per_view[start : start + n], axis=1)
        errors.append({"markers": vid, "mean_mm": round(float(e.mean()), 3),
                       "max_mm": round(float(e.max()), 3)})  # fmt: skip
        start += n
    # 写真ごとに単独で解いたときの中心との差（配置のばらつきの目安）
    spread = {
        str(i): round(float(np.linalg.norm(np.std(centers[i], axis=0))), 2)
        for i in ids
        if len(centers[i]) >= 2
    }
    report = {"views": errors, "single_view_center_spread_mm": spread}
    layout = WallLayout({i: tuple(float(v) for v in m) for i, m in markers.items()})
    return layout, report
