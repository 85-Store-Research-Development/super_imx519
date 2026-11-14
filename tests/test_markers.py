import cv2
import numpy as np

from super_imx519 import markers


def _render(marker_id: int, px_per_module: int = 20) -> np.ndarray:
    """板 1 枚分（マーカー + 1 マスの白い余白）を画像にする。"""
    modules = markers.marker_modules(marker_id)
    image = np.where(modules, 0, 255).astype(np.uint8)
    image = cv2.resize(
        image, None, fx=px_per_module, fy=px_per_module, interpolation=cv2.INTER_NEAREST
    )
    margin = round((markers.BOARD_MM - markers.MARKER_MM) / 2 / markers.MARKER_MM * image.shape[0])
    return cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def test_markers_are_detected_with_their_ids():
    dictionary = cv2.aruco.getPredefinedDictionary(markers.DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    for marker_id in markers.POSITIONS:
        _, ids, _ = detector.detectMarkers(_render(marker_id))
        assert ids is not None and ids.ravel().tolist() == [marker_id]


def test_board_layout_fits_a4():
    margin = (markers.BOARD_MM - markers.MARKER_MM) / 2
    assert margin == markers.MARKER_MM / markers.MODULES  # 白い余白は 1 マス分
    assert 2 * markers.BOARD_MM < markers.A4_MM[0]
    assert markers.GRID_TOP_MM + 2 * markers.BOARD_MM + 40 < markers.A4_MM[1]


def test_pdf_is_one_a4_page():
    pdf = markers.markers_pdf()
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/Count 1" in pdf
    assert b"/MediaBox [0 0 595.276 841.890]" in pdf
