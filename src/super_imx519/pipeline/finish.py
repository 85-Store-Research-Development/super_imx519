"""商品画像の仕上げ（カラーカードで色を合わせたあとにかける）。

売り場で並んだときの統一感のため、背景（壁）の明るさと色味を、既存の商品画像に揃える。
服の色は、カードで合わせた実物の色からなるべく動かさない（決定。2026-10-18。DESIGN.md
「iPhone カメラモード」）。

1. 明るさ: 壁の明るさ（L*）を測り、それが wall_l になるトーンカーブを全体にかける。中間はほぼ同じ
   倍率で持ち上げ、明るいところは 100 に向けて丸める（白い服を飛ばさない）。カーブは輝度だけに
   かけ、RGB の比を保つ（色相を変えない）。収まらない色は、同じ輝度の灰色に寄せて収める
2. 背景の色味: 服・ハンガー・フック以外の壁を、wall_ab（a*, b*）に寄せる。服のすぐ外側（2 mm）
   から寄せ始め、境目は細くぼかす（間を空けすぎると、元の色の帯が服の縁ににじんで見える）
3. シャープ: 書き出す大きさで、輝度に軽くアンシャープマスクをかける（生地の質感を見せる）

画像は BGR の float32（0〜1、sRGB）。
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .color import to_linear, to_srgb


@dataclass
class Finish:
    # 既存の商品画像（ユーザーが示したカーディガン）の壁を測った値（2026-10-18）
    wall_l: float = 94.0
    wall_ab: tuple[float, float] = (-2.7, -2.4)
    max_gain: float = 1.4  # 壁が暗すぎる写真でも、持ち上げはここまで
    subject_margin_mm: float = 2.0  # 服からこれだけ外側から、背景の色味を変え始める
    feather_mm: float = 1.5  # 背景の色味の境目のぼかし
    sharpen: float = 0.35  # アンシャープマスクの強さ
    sharpen_sigma: float = 1.0


def _pchip(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
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


def tone_curve(wall_l: float, target: float, max_gain: float) -> tuple[np.ndarray, np.ndarray]:
    """L*（0〜100）→ L* のカーブ（1001 点の表）。壁を target に、0 と 100 は動かさない。"""
    gain = float(np.clip(target / max(wall_l, 1.0), 1 / max_gain, max_gain))
    knee = min(wall_l * gain, 99.0)
    xs = np.array([0.0, 10.0, 0.6 * wall_l, wall_l, 100.0])
    ys = np.array([0.0, 10.0 * gain, 0.6 * wall_l * gain, knee, 100.0])
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    xs, ys = xs[keep], np.maximum.accumulate(ys[keep])
    grid = np.linspace(0, 100, 1001)
    return grid, np.clip(_pchip(xs, ys, grid), 0, 100)


def _lightness(y: np.ndarray) -> np.ndarray:
    f = np.where(y > (6 / 29) ** 3, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def _luminance(lightness: np.ndarray) -> np.ndarray:
    f = (lightness + 16) / 116
    return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))


def finish(
    image: np.ndarray, subject: np.ndarray, mm_per_px: float, params: Finish
) -> tuple[np.ndarray, dict]:
    """image は書き出す大きさの BGR（0〜1、sRGB）、subject は服・ハンガー・フックの画素。"""
    px = 1 / mm_per_px
    near = cv2.dilate(subject.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=max(int(params.subject_margin_mm * px), 1)) > 0  # fmt: skip
    background = ~near
    lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2Lab)
    if background.sum() < 1000:
        return image, {"skipped": "背景が少なすぎます"}
    wall_before = np.median(lab[background], axis=0)

    # 1. 明るさ（輝度にだけカーブをかけ、RGB の比を保つ）
    grid, curve = tone_curve(float(wall_before[0]), params.wall_l, params.max_gain)
    linear = to_linear(np.clip(image, 0, 1)).astype(np.float32)
    y = linear @ np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR の順
    new_y = _luminance(np.interp(_lightness(y), grid, curve)).astype(np.float32)
    out = linear * (new_y / np.maximum(y, 1e-6))[..., None]
    peak = out.max(axis=2)
    over = peak > 1
    if over.any():  # 収まらない色は、同じ輝度の灰色に寄せる
        k = (1 - new_y[over]) / np.maximum(peak[over] - new_y[over], 1e-6)
        out[over] = new_y[over][:, None] + (out[over] - new_y[over][:, None]) * k[:, None]
    srgb = to_srgb(out).astype(np.float32)

    # 2. 背景の色味（服のすぐ外側から。境目は細くぼかす）
    lab = cv2.cvtColor(srgb, cv2.COLOR_BGR2Lab)
    wall_mid = np.median(lab[background], axis=0)
    shift = np.array(params.wall_ab, np.float32) - wall_mid[1:]
    weight = cv2.GaussianBlur(background.astype(np.float32), (0, 0), params.feather_mm * px)
    weight = np.where(near, 0, weight)
    lab[..., 1:] += weight[..., None] * shift
    srgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)

    # 3. シャープ（輝度だけ）
    if params.sharpen > 0:
        lab = cv2.cvtColor(srgb, cv2.COLOR_BGR2Lab)
        lum = lab[..., 0]
        lab[..., 0] = np.clip(
            lum + params.sharpen * (lum - cv2.GaussianBlur(lum, (0, 0), params.sharpen_sigma)),
            0,
            100,
        )
        srgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)

    wall_after = np.median(cv2.cvtColor(srgb, cv2.COLOR_BGR2Lab)[background], axis=0)
    report = {
        "wall_lab_before": np.round(wall_before, 1).tolist(),
        "wall_lab_after": np.round(wall_after, 1).tolist(),
        "gain_at_wall": round(float(params.wall_l / max(wall_before[0], 1)), 3),
        "out_of_gamut_px": int(over.sum()),
    }
    return srgb, report
