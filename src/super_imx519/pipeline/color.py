"""24 色カード（ColorChecker Classic 互換）で色を合わせる。

1. cv2.mcc でカードを見つけ、パッチの中央 60% の中央値を測る（縁と、互換品の番号の印字を避ける）
2. 補正をリニアな sRGB の値で求める
   - リニア（DNG を現像したもの）: グレー 4 枚（neutral 8〜3.5）でチャネルごとの倍率（WB と露出）
   - 表示用（JPEG など、トーンカーブ済み）: グレー 6 枚で、チャネルごとに lin = a·v^g を当てる
   - どちらも、そのあとにグレーを保つ 3 × 3 の行列（各列の和が 1）を最小二乗で求める
   - リニアは、最後に明るさのカーブ（tone）: 補正したグレー 5 枚（白〜neutral 3.5）の輝度を基準の
     輝度に合わせる。iPhone の ProRAW は明るい所が圧縮されていて、倍率 1 つでは白が 0.66 倍・
     neutral 8 が 0.89 倍・中間が 1.17 倍になり、全体が薄く（中間が明るく、白が暗く）出た
     （2026-11-22。露出の違う 5 枚で同じ形なので、同じ商品のほかの写真には基準の写真のカーブを
     使える）。カーブは輝度にかけ、RGB の比は保つ（色相を変えない）。**apply には含めない**:
     傾きを直し、服を探し、マーカーを消すまでは今までの明るさの画像で行い（しきい値がその明るさに
     合わせてある）、仕上げの前（finish・closeup）で apply_tone をかける
3. 基準は OpenCV が持つ ColorChecker Classic の値（cv2.ccm）。互換品の色ずれは補正の誤差になる

画像は RGB の float（0〜1）で扱う。補正の出力はリニアな sRGB。
"""

from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
]  # fmt: skip
GRAYS = [19, 20, 21, 22]  # neutral 8〜3.5。白は飛びやすく、黒はノイズとフレアが乗りやすい
TONE_GRAYS = [22, 21, 20, 19, 18]  # 暗い順。明るさのカーブを通す（黒はフレアで浮くので使わない）
LUMA = np.array([0.2126, 0.7152, 0.0722])
ALL_GRAYS = [18, 19, 20, 21, 22, 23]
PATCH_INNER = 0.6


def to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((np.maximum(srgb, 0) + 0.055) / 1.055) ** 2.4)


# 色の濃い明るいもの（濃いオレンジなど）は、明るさは収まっていても、カードで合わせると 1 つの
# チャネル（赤）だけが 1 を超える。チャネルごとに切ると、そのチャネルの濃淡が消え、色相もずれて
# 白っぽく飛んで見えた（2026-11-21、River のオレンジのジャケット・パンツ）。一番大きいチャネルが
# KNEE を超える画素は、その値を 1 に向けて滑らかに丸め、画素の RGB を同じ割合で縮める（色の比 =
# 色相と彩度を保ち、明るさだけを下げる。順序は崩れないので、生地の濃淡は残る）
KNEE = 0.9


def fit_peak(linear: np.ndarray, knee: float = KNEE) -> np.ndarray:
    """リニアな RGB（最後の軸が色）の、1 を超えるところを色の比を保って 1 以下に収める。"""
    linear = np.maximum(np.asarray(linear, np.float32), 0)
    peak = linear.max(axis=-1)
    over = peak > knee
    if not over.any():
        return linear
    room = 1 - knee
    fitted = knee + room * np.tanh((peak[over] - knee) / room)
    out = linear.copy()
    out[over] *= (fitted / peak[over])[:, None]
    return out


def pchip(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """単調な 3 次エルミート補間（Fritsch–Carlson）。値の順序を崩さない。"""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2 / (1 / d[:-1] + 1 / d[1:]), 0)
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(x, t) - 1, 0, len(h) - 1)
    s = (t - x[i]) / h[i]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]


def gray_tone(gained: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    """明るさのカーブの点（n × 2。補正したあとの輝度 → 基準の輝度。暗い順）。

    gained は補正（倍率と行列）をかけたパッチの値（24 × 3）。輝度が明るい順に並んでいない点
    （飛んだ白など）は外す。2 点に満たなければ None（カーブはかけない）。
    """
    x, y = gained[TONE_GRAYS] @ LUMA, ref[TONE_GRAYS] @ LUMA
    keep = [0]
    for i in range(1, len(x)):
        if x[i] > x[keep[-1]] * 1.02:
            keep.append(i)
    if len(keep) < 2 or x[keep[0]] <= 0:
        return None
    return np.stack([x[keep], y[keep]], axis=1)


def apply_tone(linear: np.ndarray, tone: np.ndarray | None) -> np.ndarray:
    """輝度に明るさのカーブをかける（RGB の比は保つ）。点の外側は、端の点の倍率のまま延ばす。"""
    if tone is None:
        return linear
    tone = np.asarray(tone, np.float64)
    lx, ly = np.log(tone[:, 0]), np.log(tone[:, 1])
    lum = np.maximum(np.asarray(linear, np.float32) @ LUMA.astype(np.float32), 1e-9)
    t = np.log(lum).astype(np.float64)
    out = pchip(lx, ly, np.clip(t, lx[0], lx[-1])) + (t - np.clip(t, lx[0], lx[-1]))
    return linear * np.exp(out - t).astype(np.float32)[..., None]


def to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(srgb, np.float32), 0, 1).reshape(-1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = np.asarray(lab1, np.float64).T
    l2, a2, b2 = np.asarray(lab2, np.float64).T
    cm = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cm**7 / (cm**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dl, dc = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dhh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    lm, cpm, hs = (l1 + l2) / 2, (c1p + c2p) / 2, h1p + h2p
    hm = np.where(
        c1p * c2p == 0,
        hs,
        np.where(
            np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)
        ),
    )
    t = (
        1
        - 0.17 * np.cos(np.radians(hm - 30))
        + 0.24 * np.cos(np.radians(2 * hm))
        + 0.32 * np.cos(np.radians(3 * hm + 6))
        - 0.20 * np.cos(np.radians(4 * hm - 63))
    )
    sl = 1 + 0.015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cpm, 1 + 0.015 * cpm * t
    rt = (
        -2
        * np.sqrt(cpm**7 / (cpm**7 + 25**7))
        * np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2))))
    )
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh))


@cache
def reference_linear() -> np.ndarray:
    """ColorChecker Classic の基準値（リニアな sRGB、24 × 3）。OpenCV の収録値（D50 → sRGB）。"""
    # 基準値だけが欲しいので、飽和しない適当な入力で模型を作って取り出す
    dummy = np.full((24, 1, 3), 0.5, np.float64)
    model = cv2.ccm.ColorCorrectionModel(dummy, cv2.ccm.COLORCHECKER_MACBETH)
    model.setColorSpace(cv2.ccm.COLOR_SPACE_SRGB)
    model.compute()
    ref = np.array(model.getRefLinearRGB(), np.float64).reshape(24, 3)
    ref.flags.writeable = False
    return ref


def find_card(bgr8: np.ndarray) -> np.ndarray | None:
    """カードのパッチの 4 角（24 × 4 × 2、画素）。見つからなければ None。"""
    detector = cv2.mcc.CCheckerDetector.create()
    detector.setColorChartType(cv2.mcc.MCC24)
    if not detector.process(bgr8):
        return None
    checker = detector.getBestColorChecker()
    return np.array(checker.getColorCharts(), np.float64).reshape(24, 4, 2)


def sample_patches(rgb: np.ndarray, quads: np.ndarray) -> np.ndarray:
    values = []
    for quad in quads:
        center = quad.mean(axis=0)
        inner = np.round(center + PATCH_INNER * (quad - center)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(inner)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, inner - [x, y], 1)
        values.append(np.median(rgb[y : y + h, x : x + w][mask > 0], axis=0))
    return np.array(values, np.float64)


def _gray_preserving_matrix(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """行ベクトル @ M。各列の和を 1 にして、グレー（R = G = B）をグレーのまま保つ。"""
    m = np.zeros((3, 3))
    a = np.stack([src[:, 0] - src[:, 2], src[:, 1] - src[:, 2]], axis=1)
    for j in range(3):
        (m0, m1), *_ = np.linalg.lstsq(a, ref[:, j] - src[:, 2], rcond=None)
        m[:, j] = [m0, m1, 1 - m0 - m1]
    return m


@dataclass
class CardCorrection:
    """測ったパッチから求めた補正。apply は入力（linear か display）→ リニアな sRGB。"""

    source: str  # "linear" / "display"
    gain: np.ndarray  # linear: チャネルごとの倍率。display: lin = gain · v^gamma の gain
    gamma: np.ndarray  # linear では 1
    matrix: np.ndarray
    delta_e_before: np.ndarray
    delta_e_after: np.ndarray
    # linear だけ: 明るさのカーブの点（gray_tone）。apply のあとに apply_tone でかける。
    # None ならかけない
    tone: np.ndarray | None = None

    def to_linear(self, values: np.ndarray) -> np.ndarray:
        if self.source == "linear":
            return values * self.gain
        return self.gain * np.clip(values, 1e-6, 1) ** self.gamma

    def apply(self, values: np.ndarray) -> np.ndarray:
        out = self.to_linear(values).astype(np.float32) @ self.matrix.astype(np.float32)
        return np.maximum(out, 0)

    def report(self) -> dict:
        def summary(de):
            return {
                "mean": round(float(de.mean()), 2),
                "colors": round(float(de[:18].mean()), 2),
                "grays": round(float(de[ALL_GRAYS].mean()), 2),
                "max": round(float(de.max()), 2),
            }

        return {
            "source": self.source,
            "gain": np.round(self.gain, 4).tolist(),
            "gamma": np.round(self.gamma, 4).tolist(),
            "matrix": np.round(self.matrix, 4).tolist(),
            "tone": None if self.tone is None else np.round(self.tone, 5).tolist(),
            "delta_e2000_grays": np.round(self.delta_e_after[ALL_GRAYS], 2).tolist(),
            "delta_e2000_before": summary(self.delta_e_before),
            "delta_e2000_after": summary(self.delta_e_after),
        }


def fit_card(measured: np.ndarray, source: str, tone: np.ndarray | None = None) -> CardCorrection:
    """measured はパッチの値（24 × 3、RGB）。source は "linear"（リニア）か "display"（sRGB）。

    tone を渡すと、明るさのカーブはそれを使う（同じ商品の基準の写真のもの。露出と WB だけを、
    この写真のカードで合わせる）。渡さなければ、この写真のグレーから作る。
    """
    ref = reference_linear()
    ref_lab = srgb_to_lab(to_srgb(ref))
    if source == "linear":
        gain = (measured[GRAYS] * ref[GRAYS]).sum(0) / (measured[GRAYS] ** 2).sum(0)
        gamma = np.ones(3)
        # 補正前の比較は、露出（G の倍率 1 つ）だけを合わせたもの
        before = to_srgb(measured * gain[1])
    elif source == "display":
        fits = [
            np.polyfit(
                np.log(np.clip(measured[ALL_GRAYS, c], 1e-4, 1)), np.log(ref[ALL_GRAYS, c]), 1
            )
            for c in range(3)
        ]
        gamma = np.array([f[0] for f in fits])
        gain = np.exp([f[1] for f in fits])
        before = measured
        tone = None
    else:
        raise ValueError(source)
    correction = CardCorrection(source, gain, gamma, np.eye(3), np.zeros(24), np.zeros(24))
    correction.matrix = _gray_preserving_matrix(correction.to_linear(measured), ref)
    corrected = correction.apply(measured)
    if source == "linear":
        correction.tone = gray_tone(corrected, ref) if tone is None else np.asarray(tone, float)
    correction.delta_e_before = delta_e2000(ref_lab, srgb_to_lab(before))
    correction.delta_e_after = delta_e2000(
        ref_lab, srgb_to_lab(to_srgb(apply_tone(corrected, correction.tone)))
    )
    return correction
