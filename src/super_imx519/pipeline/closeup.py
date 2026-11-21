"""アップ（タグ・生地・プリントなど）の写真から、商品画像を作る（cms.py が呼ぶ）。

アップの写真にはカラーカードも壁のマーカーも写っていないので、形は直さず、色だけを合わせる。
同じ商品の正面（か背面）の写真をカードで補正したときの値（reference）を受け取り、
明るさ（露出。exposure.py）をそろえてから同じ補正を当てる。続けて撮った写真は照明が同じなので、
正面と色がそろう（2026-11-14 に、同じ服の 6 枚で確かめた）。

reference が無い・露出が分からないときは、その写真だけで明るさを合わせる（色は iPhone の
ホワイトバランスのまま）。report の warnings に書く。
"""

import json
import time
from pathlib import Path

import numpy as np

from . import color, exposure
from .finish import Finish, apply_look
from .phone import load
from .process import FINAL_SIDE, PREVIEW_SIDE, fit, write_jpeg


def reference_from(report: dict, scale: float | None) -> dict | None:
    """正面・背面の report（phone.process_upload）から、アップに当てる補正の値を取り出す。"""
    c = report.get("color")
    if not c or c.get("source") != "linear" or not scale:
        return None
    ref = {"gain": c["gain"], "gamma": c["gamma"], "matrix": c["matrix"], "exposure_scale": scale}
    # 正面にかけた仕上げ（明るさのカーブ）も同じにする
    finish = report.get("finish")
    if isinstance(finish, dict) and "wall_target_l" in finish:
        ref["look"] = finish
    return ref


def correct(
    rgb: np.ndarray, scale: float | None, reference: dict | None, floor_stops: float = 2.0
) -> np.ndarray:
    """リニアな RGB（0〜1。DNG を現像したもの）→ 補正したリニアな sRGB。"""
    ref_scale = reference.get("exposure_scale") if reference else None
    correction = color.CardCorrection(
        "linear",
        np.array(reference["gain"], float),
        np.array(reference["gamma"], float),
        np.array(reference["matrix"], float),
        np.zeros(24),
        np.zeros(24),
    )
    values = rgb * (ref_scale / scale)
    out = correction.apply(values.reshape(-1, 3)).reshape(rgb.shape)
    return out * brightness_factor(out, floor_stops)


# 明るさの決め方: 正面と同じ露出（実際の明るさ）を基本にするが、襟の奥のタグのように影になった所は
# 実際の明るさだと真っ暗になる（iPhone は露出を大きく持ち上げて撮っている。2026-11-14 の #499 で、
# 正面の 1/150 の明るさになった）。明るい所（上位 0.5%）を、自動で合わせたとき（AUTO_BRIGHT）の
# floor_stops 段下から自動の明るさまでの間に収める。全体を同じ倍率で変えるので、色の比は変わらない。
# タグは字が読めるよう 1 段（TAG_FLOOR_STOPS）、生地などは正面と同じ濃さを優先して 2 段まで
# 暗くてよい（白い所の無い生地だけのアップを 1 段にすると、紺のプリントが白っぽく持ち上がった）
AUTO_BRIGHT = 0.9
TAG_FLOOR_STOPS = 1.0
TAG_LABEL = "タグのアップ"


def brightness_factor(linear: np.ndarray, floor_stops: float = 2.0) -> float:
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    bright = float(np.percentile(luminance, 99.5))
    if bright <= 1e-6:
        return 1.0
    return float(np.clip(bright, AUTO_BRIGHT / 2**floor_stops, AUTO_BRIGHT) / bright)


def process_closeup(
    src: Path, out_dir: Path, reference: dict | None = None, label: str | None = None
) -> dict:
    started = time.time()
    report: dict = {"source_file": src.name, "kind": "closeup", "warnings": []}
    rgb, source = load(src)
    report["source"] = source
    report["source_size"] = [rgb.shape[1], rgb.shape[0]]
    scale = exposure.exposure_scale(src) if source == "linear" else None
    report["exposure_scale"] = scale
    if source == "linear" and scale and reference and reference.get("exposure_scale"):
        floor = TAG_FLOOR_STOPS if label == TAG_LABEL else 2.0
        linear = correct(rgb, scale, reference, floor)
        report["color"] = {
            "from": "reference",
            "ratio": round(reference["exposure_scale"] / scale, 4),
        }
    else:
        if source != "linear":
            report["warnings"].append("RAW ではないので、色は補正していません")
        elif not reference:
            report["warnings"].append(
                "同じ商品の正面のカラーカードが無いので、色は補正していません"
            )
        else:
            report["warnings"].append("露出が分からないので、色は補正していません")
        # 明るさだけ、その写真の明るい所（99.5 パーセンタイル）で合わせる
        linear = (
            rgb / max(float(np.percentile(rgb, 99.5)), 1e-6)
            if source == "linear"
            else color.to_linear(rgb)
        )
        report["color"] = None
    srgb = color.to_srgb(color.fit_peak(linear))[..., ::-1]  # BGR
    if report["color"] and reference.get("look"):
        srgb = apply_look(np.ascontiguousarray(srgb, np.float32), reference["look"], Finish())
        report["look"] = "reference"
    final = fit(srgb, FINAL_SIDE)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jpeg(out_dir / "preview.jpg", fit(final, PREVIEW_SIDE), 85)
    report["final_size"] = [final.shape[1], final.shape[0]]
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_jpeg(out_dir / "final.jpg", final, 95)
    return report
