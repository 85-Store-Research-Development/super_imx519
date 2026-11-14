"""DNG の露出（シャッター速度・ISO・F 値・BaselineExposure）から、写真どうしの明るさの比を出す。

同じ商品を続けて撮った写真は、照明は同じでも iPhone が 1 枚ごとに露出を変える。正面の写真の
カラーカードで求めた補正をアップの写真にも当てるには、先に明るさをそろえる（cms.py・closeup.py）。

- read_dng（rawpy）の値は、センサーに入った光 × 露出時間 × ISO ÷ F 値² に比例する
- iPhone の ProRAW は、1 枚ごとに BaselineExposure（明るさの持ち上げ。EV）が付いている。
  rawpy はこれを掛けないが、値はすでにこの分だけ持ち上げてある（2026-11-14 に、同じ服の正面・
  背面・アップ 4 枚で、2^-BaselineExposure を掛けると明るさがそろうことを確かめた）
"""

import struct
from pathlib import Path

EXPOSURE_TIME, F_NUMBER, ISO, EXIF_IFD, BASELINE_EXPOSURE = 33434, 33437, 34855, 34665, 50730
# TIFF の型ごとの大きさ（バイト）
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
HEAD_BYTES = 4 * 1024 * 1024  # タグは先頭にある（本体の画素は読まない）


def read_tags(path: Path) -> dict[int, float]:
    """DNG（TIFF）の IFD0 と EXIF の IFD から、数のタグを読む（最初の値だけ）。"""
    with open(path, "rb") as f:
        data = f.read(HEAD_BYTES)
    if data[:2] not in (b"II", b"MM"):
        raise ValueError(f"TIFF（DNG）ではありません: {path.name}")
    end = "<" if data[:2] == b"II" else ">"

    def ifd(offset: int) -> dict[int, float]:
        out: dict[int, float] = {}
        (count,) = struct.unpack_from(end + "H", data, offset)
        for i in range(count):
            tag, typ, n, value = struct.unpack_from(end + "HHI4s", data, offset + 2 + 12 * i)
            size = TYPE_SIZE.get(typ, 1) * n
            if size > 4:
                (pointer,) = struct.unpack(end + "I", value)
                value = data[pointer : pointer + size]
            if typ in (5, 10) and len(value) >= 8:
                a, b = struct.unpack_from(end + ("II" if typ == 5 else "ii"), value)
                out[tag] = a / b if b else 0.0
            elif typ == 3:
                out[tag] = struct.unpack_from(end + "H", value)[0]
            elif typ == 4:
                out[tag] = struct.unpack_from(end + "I", value)[0]
        return out

    (first,) = struct.unpack_from(end + "I", data, 4)
    tags = ifd(first)
    if EXIF_IFD in tags:
        tags.update(ifd(int(tags[EXIF_IFD])))
    return tags


def exposure_scale(path: Path) -> float | None:
    """値の明るさの相対的な大きさ。2 枚を同じ明るさにするには、値 × (基準の scale ÷ この scale)。

    露出の情報が無ければ None（明るさはそろえられない）。
    """
    tags = read_tags(path)
    shutter, iso, f_number = tags.get(EXPOSURE_TIME), tags.get(ISO), tags.get(F_NUMBER)
    if not shutter or not iso or not f_number:
        return None
    return shutter * iso / f_number**2 * 2 ** -tags.get(BASELINE_EXPOSURE, 0.0)
