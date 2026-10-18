"""4 隅の ArUco マーカーで、写真を壁の正面図（壁面座標系、mm）に直す。

マーカーは一辺 75 mm の正方形（super_imx519.markers）。壁面上の位置は未知数として、ホモグラフィと
同時に推定する（各マーカーが正方形で、4 枚とも同じ向きに貼られている、という条件だけを使う）。
壁面上の位置を実測して記録したら、それを固定値にできる（DESIGN.md「日ごとの配置ズレ補正」）。

正面図の座標は、マーカー ID が最小のものの中心を原点にした mm を、mm_per_px で画素にしたもの。
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..markers import DICTIONARY, MARKER_MM

MM_PER_PX = 0.5
_HALF = MARKER_MM / 2
_SQUARE = np.array([[-_HALF, -_HALF], [_HALF, -_HALF], [_HALF, _HALF], [-_HALF, _HALF]])


@dataclass
class WallView:
    matrix: np.ndarray  # 元の画像の画素 → 正面図の画素（3 × 3）
    size: tuple[int, int]  # 正面図の (幅, 高さ)。元の画像の全体が収まる
    mm_per_px: float
    markers: dict[int, np.ndarray] = field(default_factory=dict)  # ID → 正面図での 4 角
    residual_mm: tuple[float, float] = (0.0, 0.0)  # 平均, 最大

    def warp(self, image: np.ndarray, interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        return cv2.warpPerspective(image, self.matrix, self.size, flags=interpolation)

    def valid(self, source_shape: tuple[int, ...]) -> np.ndarray:
        """正面図のうち、元の画像が写っている画素（縁の補間のにじみの分だけ内側）。"""
        ones = np.full(source_shape[:2], 255, np.uint8)
        mask = cv2.warpPerspective(ones, self.matrix, self.size, flags=cv2.INTER_NEAREST)
        return cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0

    def report(self) -> dict:
        centers = {i: q.mean(axis=0) * self.mm_per_px for i, q in self.markers.items()}
        origin = centers[min(centers)]
        return {
            "marker_ids": sorted(self.markers),
            "marker_centers_mm": {
                str(i): [round(float(v), 1) for v in c - origin] for i, c in sorted(centers.items())
            },
            "residual_mm": [round(v, 2) for v in self.residual_mm],
            "mm_per_px": self.mm_per_px,
        }


def detect_markers(gray: np.ndarray) -> dict[int, np.ndarray]:
    """マーカー ID → 4 角（左上から時計回り、画素）。同じ ID が複数あれば最初のもの。"""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    found: dict[int, np.ndarray] = {}
    for quad, marker_id in zip(corners, [] if ids is None else ids.ravel(), strict=True):
        found.setdefault(int(marker_id), quad[0].astype(np.float64))
    return found


def _fit(pts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """画像の角（4n × 2）→ 壁面の mm のホモグラフィと、各マーカーの壁面上の角（4n × 2）。

    未知数はホモグラフィ（8）と、1 枚目を原点にした各マーカーの中心（2(n-1)）。交互に推定して
    初期値を作り、Levenberg–Marquardt で同時に詰める（交互の推定だけでは収束が遅い）。
    画像の座標は、重心を原点・平均の距離を √2 にそろえてから解く（数値を安定させる）。
    """
    mean = pts.mean(axis=0)
    k = np.sqrt(2) / np.linalg.norm(pts - mean, axis=1).mean()
    norm = np.array([[k, 0, -k * mean[0]], [0, k, -k * mean[1]], [0, 0, 1]])
    q = cv2.perspectiveTransform(pts[None], norm)[0]

    H, _ = cv2.findHomography(q[:4], _SQUARE)
    for _ in range(20):
        centers = cv2.perspectiveTransform(q[None], H)[0].reshape(-1, 4, 2).mean(axis=1)
        centers -= centers[0]
        H, _ = cv2.findHomography(q, (centers[:, None, :] + _SQUARE[None]).reshape(-1, 2))
    params = np.concatenate([(H / H[2, 2]).ravel()[:8], centers[1:].ravel()])

    def residual(p: np.ndarray) -> np.ndarray:
        h = np.append(p[:8], 1.0).reshape(3, 3)
        c = np.vstack([[0.0, 0.0], p[8:].reshape(-1, 2)])
        proj = q @ h[:, :2].T + h[:, 2]
        wall = proj[:, :2] / proj[:, 2:]
        return (wall - (c[:, None, :] + _SQUARE[None]).reshape(-1, 2)).ravel()

    lam = 1e-3
    r = residual(params)
    for _ in range(100):
        eps = 1e-7 * np.maximum(np.abs(params), 1e-3)
        jac = np.stack(
            [(residual(params + e) - r) / e[i] for i, e in enumerate(np.diag(eps))], axis=1
        )
        a, g = jac.T @ jac, jac.T @ r
        step = np.linalg.solve(a + lam * np.diag(np.diag(a)), -g)
        trial = residual(params + step)
        if trial @ trial < r @ r:
            params, r, lam = params + step, trial, lam / 3
            if np.abs(step).max() < 1e-10:
                break
        else:
            lam *= 5
            if lam > 1e8:
                break
    h = np.append(params[:8], 1.0).reshape(3, 3) @ norm
    c = np.vstack([[0.0, 0.0], params[8:].reshape(-1, 2)])
    return h, (c[:, None, :] + _SQUARE[None]).reshape(-1, 2)


def wall_view(gray: np.ndarray, mm_per_px: float = MM_PER_PX) -> WallView:
    """マーカーから正面図への変換を求める。マーカーが 1 枚も無ければ ValueError。"""
    found = detect_markers(gray)
    if not found:
        raise ValueError(
            "マーカーが写っていません（4 隅の ArUco マーカーを画面に入れて撮ってください）"
        )
    ids = sorted(found)
    pts = np.vstack([found[i] for i in ids])

    H, targets = _fit(pts, len(ids))
    residual = np.linalg.norm(cv2.perspectiveTransform(pts[None], H)[0] - targets, axis=1)

    # 元の画像の全体が収まるように、mm → 正面図の画素を決める
    h, w = gray.shape[:2]
    frame = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float64)
    bounds = cv2.perspectiveTransform(frame, H)[0]
    x0, y0 = bounds.min(axis=0)
    x1, y1 = bounds.max(axis=0)
    k = 1 / mm_per_px
    to_px = np.array([[k, 0, -x0 * k], [0, k, -y0 * k], [0, 0, 1]])
    matrix = to_px @ H
    size = (int(np.ceil((x1 - x0) * k)), int(np.ceil((y1 - y0) * k)))
    markers = {i: cv2.perspectiveTransform(found[i][None], matrix)[0] for i in ids}
    return WallView(
        matrix, size, mm_per_px, markers, (float(residual.mean()), float(residual.max()))
    )
