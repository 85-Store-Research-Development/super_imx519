import json

import cv2
import numpy as np
import pytest

from super_imx519 import markers
from super_imx519.pipeline import color, framing, rectify
from super_imx519.pipeline.phone import process_upload

MM = 0.5  # 合成する壁の縮尺（mm/px）


def _board(marker_id: int) -> np.ndarray:
    """10 cm 角の板（白い余白 + マーカー）を BGR で。"""
    side = round(markers.BOARD_MM / MM)
    board = np.full((side, side, 3), 245, np.uint8)
    cell = markers.MARKER_MM / markers.MODULES / MM
    margin = round((markers.BOARD_MM - markers.MARKER_MM) / 2 / MM)
    modules = markers.marker_modules(marker_id)
    for r, row in enumerate(modules):
        for c, black in enumerate(row):
            if black:
                y0, x0 = margin + round(r * cell), margin + round(c * cell)
                board[y0 : margin + round((r + 1) * cell), x0 : margin + round((c + 1) * cell)] = 25
    return board


def _card() -> np.ndarray:
    """24 色カード（基準値どおりの色）。黒い台紙に 6 × 4 のパッチ。"""
    patch, gap, border = 40, 10, 16
    w, h = 6 * patch + 5 * gap + 2 * border, 4 * patch + 3 * gap + 2 * border
    card = np.full((h, w, 3), 20, np.uint8)
    srgb = color.to_srgb(color.reference_linear())
    for i, rgb in enumerate(srgb):
        r, c = divmod(i, 6)
        y0, x0 = border + r * (patch + gap), border + c * (patch + gap)
        card[y0 : y0 + patch, x0 : x0 + patch] = np.round(rgb[::-1] * 255)
    return card


def _scene(garment_h_mm: float = 900.0) -> tuple[np.ndarray, dict]:
    """壁の正面図（0.5 mm/px）: 4 隅のマーカー、カード、フック、服（オレンジの長方形）。"""
    rng = np.random.default_rng(1)
    w, h = round(1400 / MM), round(1900 / MM)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 180 + 15 * (xx / w) - 10 * (yy / h)  # 照明のむら
    wall = np.stack([base, base + 4, base + 6], axis=2) + rng.normal(0, 1.5, (h, w, 3))
    wall = np.clip(wall, 0, 255).astype(np.uint8)
    hook = (round(700 / MM), round(150 / MM))  # フックの上端（x, y）
    places = {0: (380, 300), 1: (1020, 300), 2: (380, 1350), 3: (1020, 1350)}
    for marker_id, (cx, cy) in places.items():
        b = _board(marker_id)
        y0, x0 = round(cy / MM) - b.shape[0] // 2, round(cx / MM) - b.shape[1] // 2
        wall[y0 : y0 + b.shape[0], x0 : x0 + b.shape[1]] = b
    card = _card()
    wall[90 : 90 + card.shape[0], 1500 : 1500 + card.shape[1]] = card  # フックの右、服より上
    # フック（金属の縦の板）とハンガーの首、服
    cv2.rectangle(wall, (hook[0] - 8, hook[1]), (hook[0] + 8, hook[1] + 120), (150, 150, 155), -1)
    cv2.line(wall, (hook[0], hook[1] + 120), (hook[0], hook[1] + 220), (120, 120, 125), 6)
    top = hook[1] + 220
    garment = (hook[0] - 280, top, hook[0] + 280, top + round(garment_h_mm / MM))
    cv2.rectangle(wall, garment[:2], garment[2:], (90, 160, 240), -1)
    return wall, {"hook": hook, "garment": garment}


def _photo(wall: np.ndarray) -> np.ndarray:
    """斜めから撮ったように歪める（左右の傾き・少しの回転）。"""
    h, w = wall.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 80], [w - 60, 30], [w - 20, h - 10], [60, h - 120]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(wall, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))


def test_wall_view_recovers_markers_as_squares():
    wall, _ = _scene()
    view = rectify.wall_view(cv2.cvtColor(_photo(wall), cv2.COLOR_BGR2GRAY))
    assert sorted(view.markers) == [0, 1, 2, 3]
    assert view.residual_mm[0] < 0.3 and view.residual_mm[1] < 0.6
    for quad in view.markers.values():
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1) * view.mm_per_px
        assert sides == pytest.approx([markers.MARKER_MM] * 4, abs=1.0)
    centers = view.report()["marker_centers_mm"]
    assert centers["1"] == pytest.approx([640, 0], abs=1)
    assert centers["2"] == pytest.approx([0, 1050], abs=1)


def test_card_correction_recovers_reference_colors():
    ref = color.reference_linear()
    # リニア: 色かぶり（チャネルごとの倍率）と、少しの色の混ざり
    mix = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.0, 0.1, 0.9]])
    measured = (ref @ mix) * [0.5, 0.42, 0.3]
    c = color.fit_card(measured, "linear")
    assert c.delta_e_after.mean() < 1.0 < c.delta_e_before.mean()
    # 表示用: チャネルごとのガンマとかぶり
    shown = (np.clip(ref, 0, 1) * [0.8, 0.9, 1.0]) ** (1 / np.array([2.0, 2.2, 2.4]))
    c = color.fit_card(shown, "display")
    assert c.delta_e_after.mean() < 1.5 < c.delta_e_before.mean()


def test_crop_keeps_hook_position_and_zooms_out_for_long_garments():
    layout = framing.Layout(width_mm=900, hook_cut_mm=40)
    hook = (1000.0, 200.0)
    (x0, y0, w, h), scale = framing.crop_box(hook, MM, layout, (800, 450, 1200, 2200))
    assert scale == 1.0 and (w, h) == (1800, 2700)
    assert x0 + w / 2 == pytest.approx(hook[0]) and y0 == 280
    # ロングコート: 裾が 1350 mm を超える → 引く。フックの位置の割合（中央・上から cut/h）は同じ
    (x0, y0, w, h), scale = framing.crop_box(hook, MM, layout, (800, 450, 1200, 3400))
    assert scale > 1
    assert y0 + h >= 3400 + 0.05 * h - 1
    assert x0 + w / 2 == pytest.approx(hook[0], abs=1)
    assert (y0 - hook[1]) / h == pytest.approx(80 / 2700, rel=0.01)
    assert w / h == pytest.approx(2 / 3, rel=0.01)


@pytest.mark.parametrize("garment_h_mm", [900.0, 1500.0])
def test_process_upload_makes_clean_portrait(tmp_path, garment_h_mm):
    wall, truth = _scene(garment_h_mm)
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), _photo(wall), [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = tmp_path / "out"
    seen = []
    report = process_upload(src, out, lambda d, t, label: seen.append(label), wall_path=None)
    final = cv2.imread(str(out / "final.jpg"))
    assert final.shape[0] == 2048 and final.shape[1] / final.shape[0] == pytest.approx(
        2 / 3, abs=0.01
    )
    assert (out / "preview.jpg").exists() and seen[-1] == "完成"
    assert json.loads((out / "report.json").read_text())["final_size"] == [final.shape[1], 2048]
    # マーカーもカードも残っていない
    assert rectify.detect_markers(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)) == {}
    assert color.find_card(final) is None
    # カードは基準どおりの色で描いてあるので、補正後もほぼ基準のまま
    assert report["color"]["delta_e2000_after"]["mean"] < 2.0
    # フックは左右の中央、上の端の近く
    crop = report["crop"]
    assert crop["hook_px"][0] == pytest.approx(crop["box_px"][0] + crop["box_px"][2] / 2, abs=2)
    if garment_h_mm > 1300:
        assert crop["scale"] > 1 and any("引いて" in w for w in report["warnings"])
    else:
        assert crop["scale"] == 1.0 and crop["width_mm"] == pytest.approx(900, abs=1)


def test_fit_layout_recovers_tilted_boards_from_several_views():
    from super_imx519.pipeline.calibrate import fit_layout

    truth = {0: (0, 0, 0.8), 1: (612, -1, -0.1), 2: (-4, 1066, -1.4), 3: (617, 1074, 0.7)}
    layout_true = rectify.WallLayout(truth)
    rng = np.random.default_rng(3)
    views = []
    for _ in range(5):
        # 壁面の mm → 画像の画素。写真ごとに位置と傾きを変える
        src = np.float32([[-300, -300], [900, -300], [900, 1400], [-300, 1400]])
        jitter = rng.uniform(-120, 120, (4, 2)).astype(np.float32)
        dst = np.float32([[200, 300], [2800, 250], [2850, 3900], [150, 3950]]) + jitter
        h = cv2.getPerspectiveTransform(src, dst)
        views.append(
            {
                i: cv2.perspectiveTransform(layout_true.corners(i)[None], h)[0]
                + rng.normal(0, 0.3, (4, 2))
                for i in truth
            }
        )
    layout, report = fit_layout(views)
    # 向きは傾きの平均を 0 にとるので、正解も同じようにそろえて比べる
    mean = np.mean([m[2] for m in truth.values()])
    for i, (x, y, angle) in truth.items():
        t = np.radians(-mean)
        ex = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ [x, y]
        # 角の位置の揺れ 0.3 px で、1 m 離れたマーカーが 1 mm 程度ずれる
        assert layout.markers[i][:2] == pytest.approx(ex, abs=1.5)
        assert layout.markers[i][2] == pytest.approx(angle - mean, abs=0.15)
    assert all(v["mean_mm"] < 0.3 for v in report["views"])


def test_wall_view_with_layout_works_when_bottom_markers_are_hidden():
    wall, _ = _scene()
    centers = {0: (0, 0), 1: (640, 0), 2: (0, 1050), 3: (640, 1050)}
    layout = rectify.WallLayout({i: (x, y, 0.0) for i, (x, y) in centers.items()})
    full = rectify.wall_view(cv2.cvtColor(_photo(wall), cv2.COLOR_BGR2GRAY), layout=layout)
    assert full.layout and full.residual_mm[0] < 0.5
    # 下の 2 枚を壁の色で塗りつぶす（ロングコートで隠れたとき）
    hidden = wall.copy()
    for cx, cy in ((380, 1350), (1020, 1350)):
        x, y = round(cx / MM), round(cy / MM)
        hidden[y - 110 : y + 110, x - 110 : x + 110] = 185
    one = rectify.wall_view(cv2.cvtColor(_photo(hidden), cv2.COLOR_BGR2GRAY), layout=layout)
    assert sorted(one.markers) == [0, 1]
    # 隠れたマーカーの位置（壁面の mm）を、写真の上で正しく指せる
    probe = np.array([centers[3]], np.float64)
    a = cv2.perspectiveTransform(full.to_view(probe)[None], np.linalg.inv(full.matrix))[0]
    b = cv2.perspectiveTransform(one.to_view(probe)[None], np.linalg.inv(one.matrix))[0]
    assert np.linalg.norm(a - b) < 6  # 画素（約 3 mm）。上の 2 枚から 1 m 下まで延ばしても収まる
