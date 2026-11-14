"""壁の正面図（実寸。mm/px）の服の形から、着丈と身幅（おおよそ）を測る（phone.py が使う）。

ハンガーに掛けた服を正面から撮った写真で測るので、平置きの採寸とは違う:
- 着丈: ハンガーの真下の、服のいちばん上（襟の上）から、中央の裾まで。前の襟は背中の首の
  付け根より高く、シャツの裾は中央が長いので、平置きの着丈より長めに出る
- 身幅: 服の下半分（上から 60〜90%）の、中央を含む続きの幅の中央値。吊るした服は袖が胴の横に
  重なるので、袖の分だけ広めに出る（おおよその値）
- 肩幅・袖丈は測らない（肩が下がり袖が胴に重なるので、形だけでは肩先・袖口を決められない。
  2026-11-14 に 6 着で試して外れた）

測った値はまだ手で測った値と比べていない（OFFSET_CM で直す）。CMS は report に残すだけで、
採寸の欄には入れない。
"""

import cv2
import numpy as np

THIN_MM = 12  # これより細い所（ハンガーの針金など）は服に数えない
NECK_HALF_MM = 20  # 服のいちばん上を探す、ハンガーの真下の幅（片側）
HEM_HALF_MM = 60  # 裾を探す、中央の幅（片側）
WIDTH_ROWS = (0.6, 0.9)  # 身幅を測る高さ（服の上からの割合）
# 手で測った値とのずれ（cm）。較正したら入れる
OFFSET_CM = {"着丈": 0.0, "身幅": 0.0}


def body_mask(garment: np.ndarray, mm: float) -> np.ndarray:
    k = max(3, round(THIN_MM / mm) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    body = cv2.morphologyEx(garment.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    if n <= 1:
        return np.zeros_like(garment, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def run_width(row: np.ndarray, x: int) -> int:
    """row の x を含む続き（True）の幅（画素）。x が False なら 0。"""
    if not row[x]:
        return 0
    left = x
    while left > 0 and row[left - 1]:
        left -= 1
    right = x
    while right < len(row) - 1 and row[right + 1]:
        right += 1
    return right - left + 1


def measure(garment: np.ndarray, hook: tuple[float, float], mm: float) -> dict | None:
    """着丈・身幅（cm、小数 1 桁）と、測った位置（正面図の画素）。測れなければ None。"""
    body = body_mask(garment, mm)
    if not body.any():
        return None
    h, w = body.shape
    cols = np.arange(w)
    has = body.any(axis=0)
    top = np.where(has, body.argmax(axis=0), -1)
    bottom = np.where(has, h - 1 - body[::-1].argmax(axis=0), -1)
    cx = int(round(hook[0]))
    near = has & (np.abs(cols - cx) <= NECK_HALF_MM / mm)
    center = has & (np.abs(cols - cx) <= HEM_HALF_MM / mm)
    if not near.any() or not center.any() or not 0 <= cx < w:
        return None
    neck = int(top[near].min())
    hem = int(np.median(bottom[center]))
    if hem <= neck:
        return None
    length_cm = (hem - neck) * mm / 10
    widths = [
        run_width(body[y], cx)
        for y in range(
            int(neck + WIDTH_ROWS[0] * (hem - neck)),
            int(neck + WIDTH_ROWS[1] * (hem - neck)),
            max(1, round(5 / mm)),
        )
    ]
    widths = [v for v in widths if v > 0]
    width_cm = float(np.median(widths)) * mm / 10 if widths else None
    values = {
        "着丈": round(length_cm + OFFSET_CM["着丈"], 1),
        "身幅": round(width_cm + OFFSET_CM["身幅"], 1) if width_cm else None,
    }
    return {
        "values": values,
        "calibrated": any(OFFSET_CM.values()),
        "points_px": {"neck": [cx, neck], "hem": [cx, hem]},
    }
