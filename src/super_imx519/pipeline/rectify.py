"""4 隅の ArUco マーカーで、写真を壁の正面図（壁面座標系、mm）に直す。

マーカーは一辺 75 mm の正方形（super_imx519.markers）。

- 較正値（WallLayout。calibrate.py で複数の写真から求める）があれば、マーカーの角の壁面上の位置は
  既知なので、写っているマーカー（1 枚でもよい）からホモグラフィを求める
- 無ければ、壁面上の位置を未知数として、ホモグラフィと同時に推定する（各マーカーが正方形で、
  4 枚とも同じ向きに貼られている、という条件だけを使う）

正面図の座標は、壁面座標（mm）を mm_per_px で画素にし、元の画像の全体が収まるようにずらしたもの。
較正値が無いときの壁面座標は、マーカー ID が最小のものの中心が原点。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..markers import DICTIONARY, MARKER_MM

MM_PER_PX = 0.5
_HALF = MARKER_MM / 2
_SQUARE = np.array([[-_HALF, -_HALF], [_HALF, -_HALF], [_HALF, _HALF], [-_HALF, _HALF]])


def rotation(angle_deg: float) -> np.ndarray:
    """2 × 2 の回転行列（画像の座標では時計回り）。"""
    t = np.radians(angle_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def square(center: tuple[float, float], angle_deg: float = 0.0) -> np.ndarray:
    """マーカーの 4 角（壁面の mm。左上から時計回り）。angle は時計回りの傾き。"""
    return np.asarray(center, np.float64) + _SQUARE @ rotation(angle_deg).T


@dataclass
class WallLayout:
    """壁の配置の較正値。マーカーの中心（mm）と傾き（度）、フックの上端（mm）。"""

    markers: dict[int, tuple[float, float, float]]  # ID → (x, y, 傾き)
    hook_mm: tuple[float, float] | None = None
    info: dict = field(default_factory=dict)  # 較正に使った写真・残差など（記録用）

    def corners(self, marker_id: int) -> np.ndarray:
        x, y, angle = self.markers[marker_id]
        return square((x, y), angle)

    def to_json(self) -> dict:
        return {
            "markers": {str(i): [round(v, 3) for v in m] for i, m in sorted(self.markers.items())},
            "hook_mm": None if self.hook_mm is None else [round(v, 2) for v in self.hook_mm],
            "marker_mm": MARKER_MM,
            "info": self.info,
        }

    @classmethod
    def from_json(cls, data: dict) -> "WallLayout":
        markers = {int(i): tuple(float(v) for v in m) for i, m in data["markers"].items()}
        hook = data.get("hook_mm")
        return cls(markers, tuple(hook) if hook else None, data.get("info", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | None) -> "WallLayout | None":
        if path is None or not Path(path).exists():
            return None
        return cls.from_json(json.loads(Path(path).read_text()))


@dataclass
class WallView:
    matrix: np.ndarray  # 元の画像の画素 → 正面図の画素（3 × 3）
    size: tuple[int, int]  # 正面図の (幅, 高さ)。元の画像の全体が収まる
    mm_per_px: float
    markers: dict[int, np.ndarray] = field(default_factory=dict)  # ID → 正面図での 4 角
    residual_mm: tuple[float, float] = (0.0, 0.0)  # 平均, 最大
    mm_to_px: np.ndarray = field(default_factory=lambda: np.eye(3))  # 壁面の mm → 正面図の画素
    layout: bool = False  # 較正値を使ったか

    def to_view(self, points_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_mm, np.float64).reshape(1, -1, 2)
        return cv2.perspectiveTransform(pts, self.mm_to_px)[0]

    def to_mm(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, np.float64).reshape(1, -1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.mm_to_px))[0]

    def warp(self, image: np.ndarray, interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        return cv2.warpPerspective(image, self.matrix, self.size, flags=interpolation)

    def valid(self, source_shape: tuple[int, ...]) -> np.ndarray:
        """正面図のうち、元の画像が写っている画素（縁の補間のにじみの分だけ内側）。"""
        ones = np.full(source_shape[:2], 255, np.uint8)
        mask = cv2.warpPerspective(ones, self.matrix, self.size, flags=cv2.INTER_NEAREST)
        return cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0

    def report(self) -> dict:
        centers = {i: self.to_mm(q).mean(axis=0) for i, q in self.markers.items()}
        return {
            "marker_ids": sorted(self.markers),
            "marker_centers_mm": {
                str(i): [round(float(v), 1) for v in c] for i, c in sorted(centers.items())
            },
            "residual_mm": [round(v, 2) for v in self.residual_mm],
            "mm_per_px": self.mm_per_px,
            "layout": self.layout,
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


def normalizer(pts: np.ndarray) -> np.ndarray:
    """重心を原点・平均の距離を √2 にそろえる変換（ホモグラフィを数値的に安定して解くため）。"""
    mean = pts.mean(axis=0)
    k = np.sqrt(2) / np.linalg.norm(pts - mean, axis=1).mean()
    return np.array([[k, 0, -k * mean[0]], [0, k, -k * mean[1]], [0, 0, 1]])


def to_h8(h: np.ndarray) -> np.ndarray:
    """3 × 3 のホモグラフィを、最後の要素を 1 にそろえた先頭 8 要素にする（解くときの未知数）。"""
    return (h / h[2, 2]).ravel()[:8]


def from_h8(h8: np.ndarray) -> np.ndarray:
    return np.append(h8, 1.0).reshape(3, 3)


def project(h8: np.ndarray, q: np.ndarray) -> np.ndarray:
    """ホモグラフィ（to_h8 の 8 要素）で点を写す。"""
    h = from_h8(h8)
    proj = q @ h[:, :2].T + h[:, 2]
    return proj[:, :2] / proj[:, 2:]


def least_squares(residual, params: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Levenberg–Marquardt（ヤコビアンは差分）。未知数が数十個の小さな問題用。"""
    lam = 1e-3
    r = residual(params)
    for _ in range(iterations):
        eps = 1e-7 * np.maximum(np.abs(params), 1e-3)
        jac = np.empty((len(r), len(params)))
        for i in range(len(params)):
            step = np.zeros_like(params)
            step[i] = eps[i]
            jac[:, i] = (residual(params + step) - r) / eps[i]
        a, g = jac.T @ jac, jac.T @ r
        while True:
            step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -g)
            trial = residual(params + step)
            if trial @ trial < r @ r:
                params, r, lam = params + step, trial, max(lam / 3, 1e-12)
                break
            lam *= 5
            if lam > 1e8:
                return params
        if np.abs(step).max() < 1e-10:
            break
    return params


def fit_free(pts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """画像の角（4n × 2）→ 壁面の mm のホモグラフィと、各マーカーの壁面上の角（4n × 2）。

    未知数はホモグラフィ（8）と、1 枚目を原点にした各マーカーの中心（2(n-1)）。交互に推定して
    初期値を作り、Levenberg–Marquardt で同時に詰める（交互の推定だけでは収束が遅い）。
    """
    norm = normalizer(pts)
    q = cv2.perspectiveTransform(pts[None], norm)[0]
    H, _ = cv2.findHomography(q[:4], _SQUARE)
    for _ in range(20):
        centers = cv2.perspectiveTransform(q[None], H)[0].reshape(-1, 4, 2).mean(axis=1)
        centers -= centers[0]
        H, _ = cv2.findHomography(q, (centers[:, None, :] + _SQUARE[None]).reshape(-1, 2))
    params = np.concatenate([to_h8(H), centers[1:].ravel()])

    def targets(p: np.ndarray) -> np.ndarray:
        c = np.vstack([[0.0, 0.0], p[8:].reshape(-1, 2)])
        return (c[:, None, :] + _SQUARE[None]).reshape(-1, 2)

    params = least_squares(lambda p: (project(p[:8], q) - targets(p)).ravel(), params)
    return from_h8(params[:8]) @ norm, targets(params)


def wall_view(
    gray: np.ndarray, mm_per_px: float = MM_PER_PX, layout: WallLayout | None = None
) -> WallView:
    """マーカーから正面図への変換を求める。使えるマーカーが 1 枚も無ければ ValueError。"""
    found = detect_markers(gray)
    if layout is not None:
        found = {i: q for i, q in found.items() if i in layout.markers}
    if not found:
        raise ValueError(
            "マーカーが写っていません（4 隅の ArUco マーカーを画面に入れて撮ってください）"
        )
    ids = sorted(found)
    pts = np.vstack([found[i] for i in ids])

    if layout is not None:
        targets = np.vstack([layout.corners(i) for i in ids])
        H, _ = cv2.findHomography(pts, targets, 0)  # 最小二乗（OpenCV が LM で詰める）
    else:
        H, targets = fit_free(pts, len(ids))
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
        matrix,
        size,
        mm_per_px,
        markers,
        (float(residual.mean()), float(residual.max())),
        to_px,
        layout is not None,
    )
