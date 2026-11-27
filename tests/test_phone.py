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


def _compressed(ref: np.ndarray) -> np.ndarray:
    """明るい所が圧縮された測定値（iPhone の ProRAW のように、明るいグレーの差が詰まる）。"""
    y = ref @ color.LUMA
    return ref * (np.minimum(y, 0.3 + 0.35 * (y - 0.3).clip(0)) / y)[:, None] * 0.05


def test_gray_tone_matches_the_card_grays_and_keeps_the_color_ratio():
    ref = color.reference_linear()
    c = color.fit_card(_compressed(ref), "linear")
    out = color.apply_tone(c.apply(_compressed(ref)), c.tone)
    y = out @ color.LUMA
    # 白〜neutral 3.5 の明るさが、基準どおりになる（倍率だけでは白が暗く、中間が明るい）
    assert y[color.TONE_GRAYS] == pytest.approx((ref @ color.LUMA)[color.TONE_GRAYS], rel=0.02)
    plain = c.apply(_compressed(ref)) @ color.LUMA
    assert plain[18] < 0.8 * (ref @ color.LUMA)[18] and plain[21] > 1.05 * (ref @ color.LUMA)[21]
    # RGB の比（色相と彩度）は変えない
    before = c.apply(_compressed(ref))[6]
    assert out[6] / out[6].sum() == pytest.approx(before / before.sum(), abs=1e-4)
    assert c.report()["tone"] is not None


def test_fit_card_uses_the_reference_tone_and_its_own_exposure():
    ref = color.reference_linear()
    first = color.fit_card(_compressed(ref), "linear")
    # 同じ商品の別の写真: 露出が 3 倍、少し色かぶり。カーブは基準のものを使う
    other = color.fit_card(_compressed(ref) * [3.0, 3.3, 2.7], "linear", first.tone)
    assert other.tone == pytest.approx(first.tone)
    assert other.gain == pytest.approx(first.gain / [3.0, 3.3, 2.7], rel=1e-6)
    out = color.apply_tone(other.apply(_compressed(ref) * [3.0, 3.3, 2.7]), other.tone)
    assert (out @ color.LUMA)[color.TONE_GRAYS] == pytest.approx(
        (ref @ color.LUMA)[color.TONE_GRAYS], rel=0.02
    )


def test_linear_card_without_compression_gets_an_identity_tone():
    ref = color.reference_linear()
    c = color.fit_card(ref * 0.2, "linear")
    x = np.linspace(0.02, 1.2, 30)[:, None] * np.ones(3)
    assert color.apply_tone(x, c.tone) == pytest.approx(x, rel=0.01)
    assert color.apply_tone(x, None) is x


def test_closeup_uses_its_own_card_when_one_is_in_the_photo(tmp_path):
    from super_imx519.pipeline import closeup

    card = cv2.resize(_card(), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    photo = np.full((card.shape[0] + 400, card.shape[1] + 400, 3), 150, np.uint8)
    photo[200 : 200 + card.shape[0], 200 : 200 + card.shape[1]] = card
    tinted = np.clip(photo.astype(np.float32) * [0.75, 0.9, 1.0], 0, 255).astype(
        np.uint8
    )  # 黄かぶり
    src = tmp_path / "up.png"
    cv2.imwrite(str(src), tinted)
    report = closeup.process_closeup(src, tmp_path / "out", None, "タグのアップ")
    assert report["color"]["from"] == "card" and report["warnings"] == []
    out = cv2.imread(str(tmp_path / "out" / "final.jpg")).astype(np.float32)
    wall = out[20:150, 20:150].reshape(-1, 3).mean(0)
    assert wall.max() - wall.min() < 6  # かぶりが取れて、灰色の地が無彩色になる
    # カードが無い写真は、基準が無ければ補正せず、そのことを残す
    cv2.imwrite(str(src), np.clip(photo * 0 + [110, 135, 150], 0, 255).astype(np.uint8))
    report = closeup.process_closeup(src, tmp_path / "out2", None, None)
    assert report["color"] is None and report["warnings"]


def test_crop_is_fixed_unless_the_garment_overflows():
    layout = framing.Layout(width_mm=900, hook_cut_mm=40, margin=0.03)
    hook = (1000.0, 200.0)
    # 収まる服: 決まった枠（フックが左右の中央、上の端から 40 mm 下）
    box, scale = framing.crop_box(hook, MM, layout, (800, 450, 1200, 2200))
    assert scale == 1.0 and box == (100, 280, 1800, 2700)
    assert framing.overflow(box, (800, 450, 1200, 2200)) == []
    # ロングコート: はみ出すときだけ引く。フックの位置の割合と縦横比は同じで、裾の下に余白が残る
    long = (800, 450, 1200, 3400)
    assert framing.overflow(box, long) == ["下"]
    (x0, y0, w, h), scale = framing.crop_box(hook, MM, layout, long)
    assert scale > 1 and framing.overflow((x0, y0, w, h), long) == []
    assert y0 + h - 3400 == pytest.approx(0.03 * h, abs=2)
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
    # 収まる服は決まった枠、はみ出す服だけ引く
    if garment_h_mm > 1300:
        assert crop["scale"] > 1 and any("引いて" in w for w in report["warnings"])
    else:
        assert crop["scale"] == 1.0
        assert crop["width_mm"] == pytest.approx(framing.Layout().width_mm, abs=1)


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


def test_boards_partly_covered_by_the_hem_are_located_and_erased(tmp_path):
    """裾が下の 2 枚の板に掛かっていても、柄で位置を求め、見えている所だけを消す。"""
    wall, _ = _scene()
    navy = (90, 40, 30)
    legs_mm = [(300, 1100, 1100, 1160), (300, 1160, 450, 1330), (950, 1160, 1100, 1330)]
    for x0, y0, x1, y1 in legs_mm:  # 服とつながった紺の脚。板の上の 3 割を覆う
        cv2.rectangle(wall, (round(x0 / MM), round(y0 / MM)), (round(x1 / MM), round(y1 / MM)),
                      navy, -1)  # fmt: skip
    layout = rectify.WallLayout(
        {0: (0, 0, 0.0), 1: (640, 0, 0.0), 2: (0, 1050, 0.0), 3: (640, 1050, 0.0)},
        hook_mm=(320.0, -150.0),
    )
    layout.save(tmp_path / "wall.json")
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), _photo(wall), [cv2.IMWRITE_JPEG_QUALITY, 95])
    report = process_upload(src, tmp_path / "out", wall_path=tmp_path / "wall.json", look=None)
    assert sorted(report["wall"]["marker_ids"]) == [0, 1]
    assert sorted(report["wall"]["located_ids"]) == ["2", "3"]
    assert not any("マーカーが" in w for w in report["warnings"])
    final = cv2.imread(str(tmp_path / "out" / "final.jpg")).astype(int)
    b, r = final[..., 0], final[..., 2]
    kept = ((b - r) > 40) & (r < 70)  # 紺
    # 印刷の黒（無彩色の暗い所）。服の色の境目の混ざりは除く
    near_cloth = cv2.dilate(kept.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ink = (final.max(axis=2) < 80) & (np.abs(b - r) < 20) & ~near_cloth
    assert ink.sum() < 30
    # 紺の脚（板に掛かった所も）は残る
    px_per_mm = final.shape[0] / report["crop"]["height_mm"]
    expected = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in legs_mm) * px_per_mm**2
    assert kept.sum() > 0.95 * expected


def test_finish_brightens_the_wall_and_keeps_the_garment_hue():
    from super_imx519.pipeline.finish import Finish, finish

    # 灰色の壁（L* 約 78、少し黄み）に、紫の服
    lab = np.zeros((600, 400, 3), np.float32)
    lab[...] = (78, 1.5, 4.3)
    lab[150:450, 100:300] = (30, 40, -60)
    image = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
    subject = np.zeros((600, 400), bool)
    subject[150:450, 100:300] = True
    look = Finish(sharpen=0)
    out, report = finish(image, subject, 1.0, look)
    after = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2Lab)
    wall = after[20:100, 20:100].reshape(-1, 3).mean(axis=0)
    assert wall == pytest.approx([look.wall_l, *look.wall_ab], abs=0.6)
    before = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2Lab)[300, 200]
    garment = after[300, 200]
    hue = lambda v: np.degrees(np.arctan2(v[2], v[1]))  # noqa: E731
    assert abs(hue(garment) - hue(before)) < 3  # 色相はほぼそのまま
    # 服にはカーブをかけない（カードで合わせた明るさのまま）
    assert garment[0] == pytest.approx(before[0], abs=0.5)


def test_finish_lowers_the_wall_target_for_light_garments():
    from super_imx519.pipeline.finish import Finish, wall_target

    look = Finish()
    assert wall_target(look, 77.0, 30.0) == look.wall_l  # 暗い服: 壁は 94
    light = wall_target(look, 77.0, 89.0)  # 明るい服: 服の濃淡を残すため、壁を下げる
    assert look.min_wall_l <= light < look.wall_l
    assert light + look.bright_slope * (89.0 - 77.0) <= look.bright_max + 1e-6


def test_finish_flattens_uneven_lighting_and_color_on_the_wall():
    """左が暗く黄色い壁（照明ムラ）を平らにし、服にも同じ割合をかける。"""
    from super_imx519.pipeline.color import to_linear, to_srgb
    from super_imx519.pipeline.finish import Finish, finish

    h, w = 900, 600
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.55 + 0.35 * xx / w  # 左ほど暗い
    tint = np.stack([1 - 0.15 * (1 - xx / w), np.ones_like(xx), np.ones_like(xx)], -1)  # 左ほど黄色
    albedo = np.full((h, w, 3), 0.8, np.float32)
    subject = np.zeros((h, w), bool)
    subject[250:700, 150:450] = True
    albedo[subject] = (0.05, 0.1, 0.3)  # BGR。茶色っぽい服
    image = to_srgb(albedo * light[..., None] * tint).astype(np.float32)
    out, report = finish(image, subject, 1.0, Finish(sharpen=0))
    lab = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2Lab)
    wall = ~cv2.dilate(subject.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=50).astype(
        bool
    )
    left, right = wall & (xx < 100), wall & (xx > 500)
    assert abs(lab[left].mean(0)[0] - lab[right].mean(0)[0]) < 1.0
    assert abs(lab[left].mean(0)[2] - lab[right].mean(0)[2]) < 1.0
    # 服も左右で同じ色になる（同じ光を受けているので）
    g = to_linear(out)[subject]
    gx = np.nonzero(subject)[1]
    assert np.abs(g[gx < 200].mean(0) / g[gx > 400].mean(0) - 1).max() < 0.03
    assert report["flatten"]["light_range"] > 1.4
    # 服の明るさ・色は変えない（むら取りの基準は服の所。カードで合わせた値のまま）
    assert report["flatten"]["subject_gain"] == pytest.approx([1, 1, 1], abs=0.01)
    middle = subject & (np.abs(xx - 300) < 20)
    assert to_linear(out)[middle].mean(0) == pytest.approx(
        to_linear(image)[middle].mean(0), rel=0.03
    )


def test_finish_applies_the_card_tone_to_the_garment_only():
    from super_imx519.pipeline.color import to_linear
    from super_imx519.pipeline.finish import Finish, finish

    lab = np.zeros((600, 400, 3), np.float32)
    lab[...] = (78, 0, 0)
    lab[150:450, 100:300] = (50, 30, 30)
    lab[20:60, 300:380] = (70, 0, 0)  # 壁に残った板の影（前景には入るが、服ではない）
    image = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
    garment = np.zeros((600, 400), bool)
    garment[150:450, 100:300] = True
    subject = garment.copy()
    subject[20:60, 300:380] = True
    tone = [[0.05, 0.04], [0.5, 0.4]]  # 全体を 0.8 倍
    look = Finish(sharpen=0, flatten=False)
    out, report = finish(image, subject, 1.0, look, tone, garment)
    assert report["garment_kept"] is True
    ratio = to_linear(out)[300, 200] / to_linear(image)[300, 200]
    assert ratio == pytest.approx([0.8, 0.8, 0.8], abs=0.02)  # 服はカードのカーブだけ
    after = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2Lab)
    assert after[500, 50, 0] == pytest.approx(look.wall_l, abs=0.6)  # 壁は今までどおり
    assert after[40, 340, 0] > 85  # 板の影の残りは、服として残さず壁として明るくする


def test_fit_peak_keeps_the_color_ratio_and_the_order():
    # 赤だけが 1 を超える濃いオレンジ（リニア RGB）。切ると比が崩れるが、fit_peak は比を保つ
    x = np.array([[1.2, 0.35, 0.1], [1.5, 0.45, 0.12], [0.5, 0.5, 0.5]], np.float32)
    y = color.fit_peak(x)
    assert y.max() <= 1
    assert np.allclose(y[0] / y[0, 0], x[0] / x[0, 0], atol=1e-5)
    assert y[1, 0] > y[0, 0]  # 明るい方が明るいまま
    assert np.allclose(y[2], x[2])  # knee より暗い所は変えない


def test_finish_darkens_only_a_vivid_bright_garment_to_keep_its_texture():
    from super_imx519.pipeline.color import to_srgb
    from super_imx519.pipeline.finish import Finish, apply_look, finish

    h, w = 600, 400
    linear = np.full((h, w, 3), 0.55, np.float32)  # 灰色の壁（BGR）
    subject = np.zeros((h, w), bool)
    subject[150:450, 100:300] = True
    rng = np.random.default_rng(0)
    texture = 1 + 0.15 * rng.standard_normal((300, 200, 1)).astype(np.float32)
    orange = np.array([0.05, 0.3, 0.95], np.float32)  # BGR。濃いオレンジ
    linear[subject] = (orange * texture).reshape(-1, 3)
    image = to_srgb(linear).astype(np.float32)
    look = Finish(sharpen=0)
    out, report = finish(image, subject, 1.0, look)
    assert report["garment_gain"] < 1
    after = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2Lab)
    assert after[20:100, 20:100, 0].mean() == pytest.approx(look.wall_l, abs=0.6)  # 壁はそのまま
    red = out[200:400, 150:250, 2]
    assert (red >= 254 / 255).mean() < 0.05  # 赤が張り付かない
    assert red.std() > 0.03  # 生地の濃淡が残る
    # アップにも同じ倍率がかかる
    crop = image[200:400, 150:250]
    closeup = apply_look(np.ascontiguousarray(crop), report, look)
    assert (closeup[..., 2] >= 254 / 255).mean() < 0.05


def test_finish_garment_scale_changes_only_the_garment():
    from super_imx519.pipeline.finish import Finish, finish

    lab = np.zeros((600, 400, 3), np.float32)
    lab[...] = (78, 1.5, 4.3)
    lab[150:450, 100:300] = (45, 10, 20)
    image = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
    subject = np.zeros((600, 400), bool)
    subject[150:450, 100:300] = True
    base, r1 = finish(image, subject, 1.0, Finish(sharpen=0))
    dark, r2 = finish(image, subject, 1.0, Finish(sharpen=0, garment_scale=0.8))
    assert r2["garment_gain"] == pytest.approx(r1["garment_gain"] * 0.8)
    assert np.allclose(dark[20:100, 20:100], base[20:100, 20:100], atol=1e-3)  # 壁は同じ
    assert dark[300, 200].mean() < base[300, 200].mean()


def test_garments_close_to_the_wall_color_are_treated_like_the_wall():
    """白やグレーの服は壁と同じに扱う（服と壁でかけ方を分けると、判定のずれが白い抜けになる）。"""
    from super_imx519.pipeline.color import to_linear
    from super_imx519.pipeline.finish import Finish, apply_look, finish

    def scene(garment_lab):
        lab = np.zeros((600, 400, 3), np.float32)
        lab[...] = (70, 0, 0)
        lab[150:450, 100:300] = garment_lab
        garment = np.zeros((600, 400), bool)
        garment[150:450, 100:300] = True
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1), garment

    look = Finish(sharpen=0, flatten=False)
    tone = [[0.05, 0.04], [0.5, 0.4]]  # 服にかけるカードのカーブ（0.8 倍）
    ratios = {}
    for name, garment_lab in (("navy", (25, 5, -30)), ("gray", (62, 0, 0))):
        image, garment = scene(garment_lab)
        out, report = finish(image, garment, 1.0, look, tone, garment)
        ratios[name] = float((to_linear(out)[300, 200] / to_linear(image)[300, 200]).mean())
        # アップ（服の範囲が分からない）も、同じ規則になる
        up = apply_look(image, report, look, tone)
        assert float(
            (to_linear(up)[300, 200] / to_linear(image)[300, 200]).mean()
        ) == pytest.approx(ratios[name], rel=0.05)
    assert ratios["navy"] == pytest.approx(0.8, abs=0.02)  # 壁と違う色の服: カードのカーブだけ
    assert ratios["gray"] > 1.3  # 壁に近いグレー: 壁と同じカーブで明るくなる（今までどおり）


def test_gray_tone_uses_black_and_drops_a_steep_white():
    """黒も通す（フレアで浮いた黒を戻す）。白がカードの所で詰まっていても、明るい所を広げすぎない。"""
    ref = color.reference_linear()
    y = ref @ color.LUMA
    measured = ref.copy()
    measured[23] = ref[23] * 1.8  # フレアで浮いた黒
    measured[18] = ref[18] * (y[19] * 1.05 / y[18])  # 白が neutral 8 のすぐ上まで詰まった
    c = color.fit_card(measured * 0.1, "linear")
    tone = np.asarray(c.tone)
    assert len(tone) == 5  # 黒〜neutral 8（白は外す）
    out = color.apply_tone(c.apply(measured * 0.1), c.tone) @ color.LUMA
    assert out[23] == pytest.approx(y[23], rel=0.05)
    # neutral 8 より明るい所は、neutral 8 の倍率のまま（白を基準まで持ち上げない）
    assert out[18] / out[19] == pytest.approx(1.05, rel=0.03)
