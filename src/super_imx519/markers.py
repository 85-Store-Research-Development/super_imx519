"""日ごとの配置ズレ補正に使う、4 隅の ArUco マーカー（DESIGN.md の「日ごとの配置ズレ補正」）。

10 cm 角の板に貼る前提で、A4 1 枚に 4 枚を 2 × 2 で並べた印刷用の PDF を作る。
1 枚は一辺 100 mm で、中央に一辺 75 mm のマーカー（6 × 6 マス。1 マス 12.5 mm）を置き、
周りに 1 マス分（12.5 mm）の白い余白を残す。隣どうしは切り取り線を共有するので、
カッターで縦 1 本・横 1 本を切れば 4 枚に分かれる。

PDF は外部のライブラリを使わずに書く。マーカーは塗りつぶしの矩形だけで描くので、
拡大縮小しても輪郭がぼけない。
"""

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
# ID と貼る位置。cv2.aruco が返す角の順（左上から時計回り）に合わせる
POSITIONS = {0: "TOP-LEFT", 1: "TOP-RIGHT", 2: "BOTTOM-RIGHT", 3: "BOTTOM-LEFT"}

BOARD_MM = 100.0  # 貼る板の一辺
MARKER_MM = 75.0  # 黒い枠を含むマーカーの一辺
MODULES = 6  # 4 × 4 のデータ + 1 マスの黒枠

A4_MM = (210.0, 297.0)
GRID_TOP_MM = 45.0  # 用紙の上端から、2 × 2 の上端まで
SCALE_BAR_MM = 50.0

PT_PER_MM = 72 / 25.4


def marker_modules(marker_id: int) -> np.ndarray:
    """マーカーのマスを (MODULES, MODULES) の bool で返す。True が黒。"""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, MODULES)
    return image == 0


def _pt(mm: float) -> str:
    return f"{mm * PT_PER_MM:.3f}"


class _Page:
    """mm 単位・用紙の左上が原点（下向きが y）で描き、PDF の座標に直す。"""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def _y(self, y_mm: float) -> float:
        return A4_MM[1] - y_mm

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.ops.append(f"{_pt(x)} {_pt(self._y(y + h))} {_pt(w)} {_pt(h)} re")

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(f"{_pt(x0)} {_pt(self._y(y0))} m {_pt(x1)} {_pt(self._y(y1))} l S")

    def text(self, x: float, y: float, size_pt: float, s: str) -> None:
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append(f"BT /F1 {size_pt} Tf {_pt(x)} {_pt(self._y(y))} Td ({s}) Tj ET")


def _draw_board(page: _Page, x: float, y: float, marker_id: int) -> None:
    margin = (BOARD_MM - MARKER_MM) / 2
    cell = MARKER_MM / MODULES
    # 黒いマスを 1 つのパスにまとめて塗る。マスごとに塗ると、表示で継ぎ目の線が出ることがある
    page.ops.append("0 g")
    for row, bits in enumerate(marker_modules(marker_id)):
        for col, black in enumerate(bits):
            if black:
                page.rect(x + margin + col * cell, y + margin + row * cell, cell, cell)
    page.ops.append("f")
    # 向きと ID。マーカーから 7 mm 離した薄い文字なので、検出には響かない
    page.ops.append("0.55 g")
    page.text(x + 4, y + 6, 7, f"ID {marker_id}  {POSITIONS[marker_id]}  ^ UP")


def page_content() -> bytes:
    page = _Page()
    width = A4_MM[0]
    left = (width - 2 * BOARD_MM) / 2
    top = GRID_TOP_MM
    right, bottom = left + 2 * BOARD_MM, top + 2 * BOARD_MM

    page.ops.append("0 g")
    page.text(left, 15, 11, "ArUco DICT_4X4_50  /  board 100 mm  /  marker 75 mm")
    page.text(left, 22, 9, "Print at 100% (actual size), no fit-to-page. Matte paper.")
    page.text(left, 28, 9, "Cut along the lines: 1 vertical + 1 horizontal -> 4 pieces.")

    # 用紙の上でも、壁に貼る位置と同じ並びにする
    for marker_id, (row, col) in zip((0, 1, 3, 2), ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        _draw_board(page, left + col * BOARD_MM, top + row * BOARD_MM, marker_id)

    # 切り取り線。外周は用紙の端まで伸ばさず、内側の 2 本は余白まで伸ばして定規を当てやすくする
    page.ops.append("0.6 G 0.3 w")
    cx, cy = left + BOARD_MM, top + BOARD_MM
    for y in (top, cy, bottom):
        page.line(left, y, right, y)
    for x in (left, cx, right):
        page.line(x, top, x, bottom)
    page.ops.append("[2 2] 0 d")
    page.line(cx, top - 10, cx, top)
    page.line(cx, bottom, cx, bottom + 10)
    page.line(left - 4, cy, left, cy)
    page.line(right, cy, right + 4, cy)
    page.ops.append("[] 0 d")

    # 縮尺の確認用。印刷したら定規で測る
    bar_y = bottom + 22
    page.ops.append("0 g")
    page.rect(left, bar_y, SCALE_BAR_MM, 2)
    page.ops.append("f")
    page.text(left + SCALE_BAR_MM + 4, bar_y + 2, 9, f"<- must be {SCALE_BAR_MM:.0f} mm")
    page.text(left, bar_y + 9, 9, "Measure the black square of each marker and record it.")
    return "\n".join(page.ops).encode("ascii")


def markers_pdf() -> bytes:
    """A4 1 ページに 4 枚を並べた PDF を返す。"""
    content = page_content()
    w, h = _pt(A4_MM[0]), _pt(A4_MM[1])
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
