"""商品画像の仕上げ（カラーカードで色を合わせたあとにかける）。

売り場で並んだときの統一感のため、背景（壁）の明るさと色味を既存の商品画像に揃え、全体を引き締める。
服の色は、カードで合わせた実物の色からなるべく動かさない（決定。2026-10-18。DESIGN.md
「iPhone カメラモード」）。

0. 背景のむら（シェーディング補正。2026-11-09）: 照明ムラ（光源からの距離による明暗の傾き、窓の光
   による色の傾き）と周辺減光で、背景が場所ごとに明るさも色もむらになった。壁（服とその影から
   flatten_exclude_mm より離れた所）のリニアな RGB の対数に、チャネルごとに滑らかな面（3 次の
   多項式）を当て、画像全体をその面で割って平らにする。服も同じ光を受けているので同じ割合で割る
   （服の明暗の傾きも取れる）。割る倍率は flatten_max_gain の範囲に収める
1. 明るさとレベル補正: 輝度だけにトーンカーブをかけ、RGB の比を保つ（色相を変えない）。収まらない
   色は、同じ輝度の灰色に寄せて収める
   - 壁の明るさ（L*）を wall_l にする。壁より明るいところは 100 に向けて丸める
   - 黒点: 画像の一番暗いところ（0.1% 点）を black_l まで下げ、壁までをほぼ直線で引き伸ばす（黒を
     締め、中間も引き締める）。0.1% 点が max_black より明るい写真（暗いところの無い白い服など）では
     動かさない（無いはずの黒に向けて引き伸ばすと、コントラストが強すぎた）
   - 明るい服: 壁を 94 に上げると、壁より明るい服が 94〜100 に押し込められて濃淡がつぶれ、白っぽく
     見えた。服の明るいところ（服の中の 95% 点）が bright_max に収まり、壁から服までの傾きが
     bright_slope 以上残るよう、壁の目標を min_wall_l まで下げる
2. 背景の色味: 壁を wall_ab（a*, b*。無彩色）に寄せる。服の内側（縁から edge_mm より内）には
   寄せない。服の縁の前後 edge_mm は、その画素の色が壁に近いほど強く寄せる（服の範囲の判定が
   壁に少しはみ出していても、元の色の帯が服の縁ににじんで見えないように）
3. シャープ: 書き出す大きさで、輝度に軽くアンシャープマスクをかける（生地の質感を見せる）

画像は BGR の float32（0〜1、sRGB）。
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .color import to_linear, to_srgb


@dataclass
class Finish:
    # 明るさは既存の商品画像（ユーザーが示したカーディガン）の壁を測った値（2026-10-18）。
    # 色味は無彩色（2026-11-07。既存の画像の a* −2.7・b* −2.4 では、ストアの無彩色の地色と
    # 並ぶと青く見えた）
    wall_l: float = 94.0
    wall_ab: tuple[float, float] = (0.0, 0.0)
    max_gain: float = 1.4  # 壁が暗すぎる写真でも、持ち上げはここまで
    black_l: float = 3.0  # 一番暗いところ（0.1% 点）を、この L* まで下げる（レベル補正の黒点）
    black_percentile: float = 0.1
    max_black: float = 20.0  # 0.1% 点がこれより明るい写真では、黒点を動かさない
    # 明るい服: 服の明るいところ（服の中の 95% 点）が bright_max に収まり、壁から服までの傾きが
    # bright_slope 以上残るよう、壁の目標を min_wall_l まで下げる（服の濃淡を残すのを優先する）
    bright_percentile: float = 95.0
    bright_max: float = 98.0
    bright_slope: float = 0.75
    min_wall_l: float = 86.0
    edge_mm: float = 6.0  # 服の縁の前後で、色が壁に近いかで背景の色味のかけ方を決める幅
    wall_ab_tolerance: float = 8.0  # 壁の色との差（Δa*b*）がこれ以上なら、壁ではないとみなす
    sharpen: float = 0.35  # アンシャープマスクの強さ
    sharpen_sigma: float = 1.0
    flatten: bool = True  # 背景のむらを平らにする（シェーディング補正）
    flatten_degree: int = 3
    flatten_exclude_mm: float = 40.0  # 服からこれ以内は、服の影が落ちるので面の当てはめに使わない
    flatten_max_gain: float = 1.6  # 割る倍率の上限（と下限 1 / これ）


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


def tone_curve(
    wall_l: float,
    target: float,
    max_gain: float,
    black_in: float = 0.0,
    black_out: float = 0.0,
    bright_in: float | None = None,
    bright_slope: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """L*（0〜100）→ L* のカーブ（1001 点の表）。

    black_in → black_out、壁 → target とし、100 は動かさない。黒点と壁のあいだはほぼ直線
    （レベル補正）。壁の持ち上げは max_gain 倍まで。bright_in（服の明るいところ）を渡すと、
    壁からそこまでの傾きを bright_slope 以上に保つ（明るい服の濃淡をつぶさない）。
    """
    gain = float(np.clip(target / max(wall_l, 1.0), 1 / max_gain, max_gain))
    knee = min(wall_l * gain, 99.0)
    black_in = float(np.clip(black_in, 0.0, wall_l * 0.5))
    black_out = float(min(black_out, black_in * gain))
    mid = black_in + 0.6 * (wall_l - black_in)
    slope = (knee - black_out) / (wall_l - black_in)
    xs = [0.0, black_in, mid, wall_l]
    ys = [0.0, black_out, black_out + slope * (mid - black_in), knee]
    if bright_in is not None and wall_l + 1 < bright_in < 99:
        xs.append(bright_in)
        ys.append(min(knee + bright_slope * (bright_in - wall_l), 99.5))
    xs, ys = np.array(xs + [100.0]), np.array(ys + [100.0])
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    xs, ys = xs[keep], np.maximum.accumulate(ys[keep])
    grid = np.linspace(0, 100, 1001)
    return grid, np.clip(_pchip(xs, ys, grid), 0, 100)


def wall_target(params: "Finish", wall_l: float, bright_in: float) -> float:
    """壁の目標の明るさ。明るい服は、服の明るいところが bright_max に収まり、壁から服までの傾きが
    bright_slope 以上残るよう、wall_l から min_wall_l まで下げる。"""
    if bright_in <= wall_l:
        return params.wall_l
    room = params.bright_max - params.bright_slope * (bright_in - wall_l)
    return float(np.clip(room, params.min_wall_l, params.wall_l))


def _lightness(y: np.ndarray) -> np.ndarray:
    f = np.where(y > (6 / 29) ** 3, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def _luminance(lightness: np.ndarray) -> np.ndarray:
    f = (lightness + 16) / 116
    return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))


def subject_bands(
    subject: np.ndarray, mm_per_px: float, edge_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """服の内側（縁から edge_mm より内）と、服から edge_mm 以内（服を含む）。"""
    kernel = np.ones((3, 3), np.uint8)
    steps = max(int(edge_mm / mm_per_px), 1)
    core = cv2.erode(subject.astype(np.uint8), kernel, iterations=steps) > 0
    near = cv2.dilate(subject.astype(np.uint8), kernel, iterations=steps) > 0
    return core, near


def tint_background(
    srgb: np.ndarray, subject: np.ndarray, mm_per_px: float, params: Finish
) -> np.ndarray:
    """壁の色味を wall_ab に寄せる。服の内側には寄せず、縁は色が壁に近い画素ほど強く寄せる。"""
    core, near = subject_bands(subject, mm_per_px, params.edge_mm)
    lab = cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_BGR2Lab)
    wall_mid = np.median(lab[~near], axis=0)
    shift = np.array(params.wall_ab, np.float32) - wall_mid[1:]
    likeness = np.clip(
        1 - np.linalg.norm(lab[..., 1:] - wall_mid[1:], axis=2) / params.wall_ab_tolerance, 0, 1
    )
    weight = np.where(near, likeness, 1.0).astype(np.float32)
    weight[core] = 0
    weight = cv2.GaussianBlur(weight, (0, 0), 1.0)
    lab[..., 1:] += weight[..., None] * shift
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)


def _poly_terms(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def flatten_background(
    image: np.ndarray, subject: np.ndarray, mm_per_px: float, params: Finish
) -> tuple[np.ndarray, dict]:
    """背景のむら（照明ムラ・周辺減光・色の傾き）を、壁に当てた滑らかな面で割って平らにする。"""
    h, w = image.shape[:2]
    exclude = cv2.dilate(
        subject.astype(np.uint8), np.ones((3, 3), np.uint8),
        iterations=max(int(params.flatten_exclude_mm / mm_per_px), 1),
    ) > 0  # fmt: skip
    step = max(min(h, w) // 200, 1)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    use = ~exclude[ys, xs]
    if use.sum() < 500:
        return image, {"skipped": "壁が少なすぎます"}
    linear = to_linear(np.clip(image, 0, 1)).astype(np.float32)
    small = cv2.resize(linear, (xs.shape[1], xs.shape[0]), interpolation=cv2.INTER_AREA)
    logs = np.log(np.maximum(small[use], 1e-4))
    terms = _poly_terms(xs[use] / w - 0.5, ys[use] / h - 0.5, params.flatten_degree)
    keep = np.ones(len(logs), bool)
    for _ in range(3):  # 外れ（フックの影・汚れなど）を除いて当て直す
        coef, *_ = np.linalg.lstsq(terms[keep], logs[keep], rcond=None)
        resid = np.abs(terms @ coef - logs).max(axis=1)
        keep = resid < max(3 * np.median(resid[keep]), 0.02)
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step]
    model = _poly_terms(grid_x / w - 0.5, grid_y / h - 0.5, params.flatten_degree) @ coef
    level = np.median(logs[keep], axis=0)
    gain = np.exp(np.clip(level - model, -np.log(params.flatten_max_gain),
                          np.log(params.flatten_max_gain))).astype(np.float32)  # fmt: skip
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    out = to_srgb(linear * gain).astype(np.float32)
    lum = model @ np.array([0.0722, 0.7152, 0.2126])  # BGR の順（対数の重み付けで近似）
    # 服の所の倍率（BGR の中央値）。アップに同じ仕上げをかけるとき（apply_look）に使う
    subject_gain = np.median(gain[subject], axis=0) if subject.any() else np.ones(3)
    return out, {
        "light_range": round(float(np.exp(lum.max() - lum.min())), 3),
        "wall_fit_px": int(keep.sum()),
        "subject_gain": np.round(subject_gain, 4).tolist(),
    }


def finish(
    image: np.ndarray, subject: np.ndarray, mm_per_px: float, params: Finish
) -> tuple[np.ndarray, dict]:
    """image は書き出す大きさの BGR（0〜1、sRGB）、subject は服・ハンガー・フックの画素。"""
    core, near = subject_bands(subject, mm_per_px, params.edge_mm)
    background = ~near
    if background.sum() < 1000:
        return image, {"skipped": "背景が少なすぎます"}
    flat = None
    if params.flatten:  # 0. 背景のむら
        image, flat = flatten_background(image, subject, mm_per_px, params)
    lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2Lab)
    wall_before = np.median(lab[background], axis=0)

    # 1. 明るさとレベル補正（輝度にだけカーブをかけ、RGB の比を保つ）
    linear = to_linear(np.clip(image, 0, 1)).astype(np.float32)
    y = linear @ np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR の順
    lightness = _lightness(y)
    black_in = float(np.percentile(lightness, params.black_percentile))
    levels = black_in <= params.max_black
    wall_l = float(wall_before[0])
    bright_in = (
        float(np.percentile(lightness[core], params.bright_percentile)) if core.any() else 0.0
    )
    target = wall_target(params, wall_l, bright_in)
    grid, curve = tone_curve(
        wall_l,
        target,
        params.max_gain,
        black_in if levels else 0.0,
        params.black_l if levels else 0.0,
        bright_in if bright_in > wall_l else None,
        params.bright_slope,
    )
    srgb, over = _apply_curve(linear, y, lightness, grid, curve)

    # 2. 背景の色味（服の内側には寄せない。縁は、色が壁に近いほど強く）
    srgb = tint_background(srgb, subject, mm_per_px, params)

    # 3. シャープ（輝度だけ）
    srgb = _sharpen(srgb, params)

    wall_after = np.median(cv2.cvtColor(srgb, cv2.COLOR_BGR2Lab)[background], axis=0)
    report = {
        "wall_lab_before": np.round(wall_before, 1).tolist(),
        "wall_lab_after": np.round(wall_after, 1).tolist(),
        "gain_at_wall": round(float(params.wall_l / max(wall_before[0], 1)), 3),
        "black_point_l": round(black_in, 1),
        "levels": bool(levels),
        "garment_bright_l": round(bright_in, 1),
        "wall_target_l": round(target, 1),
        "out_of_gamut_px": int(over),
        "flatten": flat,
    }
    return srgb, report


def _apply_curve(
    linear: np.ndarray, y: np.ndarray, lightness: np.ndarray, grid: np.ndarray, curve: np.ndarray
) -> tuple[np.ndarray, int]:
    """輝度にだけカーブをかける（RGB の比を保つ）。収まらない色は、同じ輝度の灰色に寄せる。"""
    new_y = _luminance(np.interp(lightness, grid, curve)).astype(np.float32)
    out = linear * (new_y / np.maximum(y, 1e-6))[..., None]
    peak = out.max(axis=2)
    over = peak > 1
    if over.any():
        k = (1 - new_y[over]) / np.maximum(peak[over] - new_y[over], 1e-6)
        out[over] = new_y[over][:, None] + (out[over] - new_y[over][:, None]) * k[:, None]
    return to_srgb(out).astype(np.float32), int(over.sum())


def _sharpen(srgb: np.ndarray, params: Finish) -> np.ndarray:
    if params.sharpen <= 0:
        return srgb
    lab = cv2.cvtColor(srgb, cv2.COLOR_BGR2Lab)
    lum = lab[..., 0]
    lab[..., 0] = np.clip(
        lum + params.sharpen * (lum - cv2.GaussianBlur(lum, (0, 0), params.sharpen_sigma)),
        0,
        100,
    )
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)


def apply_look(image: np.ndarray, look: dict, params: Finish) -> np.ndarray:
    """同じ商品の正面にかけた仕上げ（finish の report）の明るさのカーブとシャープを、壁の無い写真
    （アップ。closeup.py）にもかける。背景の色味・むら取りは壁がある写真のものなので、かけない。

    image は BGR（0〜1、sRGB）。正面と同じ露出・色にそろえてあること。
    """
    wall_l = float(look["wall_lab_before"][0])
    levels = bool(look.get("levels"))
    bright_in = float(look.get("garment_bright_l") or 0.0)
    grid, curve = tone_curve(
        wall_l,
        float(look["wall_target_l"]),
        params.max_gain,
        float(look["black_point_l"]) if levels else 0.0,
        params.black_l if levels else 0.0,
        bright_in if bright_in > wall_l else None,
        params.bright_slope,
    )
    linear = to_linear(np.clip(image, 0, 1)).astype(np.float32)
    # 正面では、背景のむら取りで服の所もこの倍率で割られている（明るさをそろえる）
    flat = look.get("flatten") or {}
    if flat.get("subject_gain"):
        linear = linear * np.array(flat["subject_gain"], np.float32)
    y = linear @ np.array([0.0722, 0.7152, 0.2126], np.float32)
    srgb, _ = _apply_curve(linear, y, _lightness(y), grid, curve)
    return _sharpen(srgb, params)
