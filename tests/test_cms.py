import struct
from pathlib import Path

import numpy as np

from super_imx519 import cms
from super_imx519.pipeline import closeup, exposure, measure


def _tiff(path: Path, shutter=(1, 60), iso=500, f_number=(9, 5), baseline=(2, 1)) -> Path:
    """露出のタグだけを持つ小さな TIFF（リトルエンディアン）。"""
    # IFD0: BaselineExposure（SRATIONAL）と EXIF の IFD の位置
    ifd0_at, exif_at, data_at = 8, 8 + 2 + 12 * 2 + 4, 200
    out = bytearray(b"II*\x00" + struct.pack("<I", ifd0_at))
    out += struct.pack("<H", 2)
    out += struct.pack("<HHII", exposure.EXIF_IFD, 4, 1, exif_at)
    out += struct.pack("<HHII", exposure.BASELINE_EXPOSURE, 10, 1, data_at)
    out += struct.pack("<I", 0)
    out += struct.pack("<H", 3)
    out += struct.pack("<HHII", exposure.EXPOSURE_TIME, 5, 1, data_at + 8)
    out += struct.pack("<HHII", exposure.F_NUMBER, 5, 1, data_at + 16)
    out += struct.pack("<HHHH", exposure.ISO, 3, 1, 0) + struct.pack("<H", iso) + b"\x00\x00"
    out += struct.pack("<I", 0)
    out += b"\x00" * (data_at - len(out))
    out += (
        struct.pack("<ii", *baseline) + struct.pack("<II", *shutter) + struct.pack("<II", *f_number)
    )
    path.write_bytes(bytes(out))
    return path


def test_exposure_scale_reads_tags(tmp_path):
    scale = exposure.exposure_scale(_tiff(tmp_path / "a.dng"))
    assert abs(scale - (1 / 60) * 500 / 1.8**2 * 2**-2) < 1e-9


def test_exposure_ratio_between_two_shots(tmp_path):
    a = exposure.exposure_scale(_tiff(tmp_path / "a.dng", shutter=(1, 60), baseline=(2, 1)))
    b = exposure.exposure_scale(_tiff(tmp_path / "b.dng", shutter=(1, 30), baseline=(3, 1)))
    # b は露出時間が 2 倍・持ち上げが 1 EV 多い → 値は同じ明るさ
    assert abs(a / b - 1) < 1e-9


def test_closeup_correct_scales_and_applies_reference():
    reference = {
        "gain": [2, 2, 2],
        "gamma": [1, 1, 1],
        "matrix": np.eye(3).tolist(),
        "exposure_scale": 1.0,
    }
    rgb = np.full((2, 2, 3), 0.8, np.float32)
    out = closeup.correct(rgb, 4.0, reference)  # 4 倍の露出で撮った → 1/4 にしてから倍率 2
    assert np.allclose(out, 0.4)


def test_reference_from_needs_linear_card_and_exposure():
    report = {
        "color": {
            "source": "linear",
            "gain": [1, 1, 1],
            "gamma": [1, 1, 1],
            "matrix": [[1, 0, 0]] * 3,
        }
    }
    assert closeup.reference_from(report, None) is None
    assert closeup.reference_from({"color": None}, 1.0) is None
    ref = closeup.reference_from({**report, "finish": {"wall_target_l": 94}}, 2.0)
    assert ref["exposure_scale"] == 2.0 and ref["look"] == {"wall_target_l": 94}


def test_reference_carries_the_gray_tone_and_correct_leaves_it_to_the_finish():
    color_report = {
        "source": "linear",
        "gain": [2, 2, 2],
        "gamma": [1, 1, 1],
        "matrix": np.eye(3).tolist(),
        "tone": [[0.1, 0.05], [0.8, 0.4]],  # 全体を半分に
    }
    ref = closeup.reference_from({"color": color_report}, 1.0)
    assert ref["tone"] == color_report["tone"]
    # correct はカーブをかけない（仕上げでかける）: 倍率 2 → 0.4
    rgb = np.full((2, 2, 3), 0.2, np.float32)
    assert np.allclose(closeup.correct(rgb, 1.0, ref), 0.4, atol=1e-3)


def test_reference_is_the_earliest_front_with_a_card():
    def source(i, role, taken, card=True, status="done"):
        report = {"color": {"source": "linear", "gain": [i] * 3, "gamma": [1] * 3,
                            "matrix": np.eye(3).tolist()} if card else None,
                  "exposure_scale": 1.0}  # fmt: skip
        return {"id": i, "recipe": "wall", "role": role, "status": status, "report": report,
                "capture": {"takenAt": taken}, "filename": f"IMG_{i}.DNG"}  # fmt: skip

    sources = [
        source(1, "back", "2026-11-21T02:00:00Z"),
        source(2, "front", "2026-11-21T02:05:00Z"),
        source(3, "front", "2026-11-21T02:01:00Z"),
        source(4, "front", "2026-11-21T01:59:00Z", card=False),  # カードを検出できなかった正面
        source(5, "front", "2026-11-21T01:00:00Z", status="pending"),
    ]
    assert cms.reference_of(sources)["gain"] == [3, 3, 3]  # 撮影時刻が最初の、カードのある正面
    assert cms.reference_of([sources[0], sources[3]])["gain"] == [1, 1, 1]  # 正面に無ければ背面
    assert cms.reference_of([sources[3]]) is None


def test_measure_length_and_width_of_a_box():
    mm = 0.5
    garment = np.zeros((2000, 2000), bool)
    garment[200:1600, 600:1400] = True  # 高さ 700 mm・幅 400 mm
    m = measure.measure(garment, (1000.0, 150.0), mm)
    assert m["values"] == {"着丈": 70.0, "身幅": 40.0}


class FakeCms:
    def __init__(self, docs, sources=None):
        self.docs = docs
        self.sources = sources or [{**d, "status": "pending"} for d in docs]
        self.claimed, self.results, self.errors = [], {}, {}

    def queue(self, limit=100):
        return self.docs

    def product_sources(self, product):
        return [s for s in self.sources if s["product"] == product]

    def claim(self, i):
        self.claimed.append(i)
        return True

    def download(self, url, dest):
        dest.write_bytes(b"x")

    def result(self, i, jpeg, report):
        self.results[i] = report

    def error(self, i, message):
        self.errors[i] = message


def _doc(i, recipe, role, product=1):
    return {
        "id": i,
        "product": product,
        "recipe": recipe,
        "role": role,
        "filename": f"IMG_{i}.DNG",
        "url": f"/f/{i}",
    }


def _fake_processing(monkeypatch, calls):
    def wall(src, out, tone=None):
        calls.append(("wall", src.name, tone))
        out.mkdir(parents=True, exist_ok=True)
        (out / "final.jpg").write_bytes(b"j")
        return {
            "color": {
                "source": "linear",
                "gain": [1, 1, 1],
                "gamma": [1, 1, 1],
                "matrix": np.eye(3).tolist(),
                "tone": [[0.1, 0.1], [0.5, 0.6]],
            },
            "warnings": [],
        }

    def close(src, out, reference, label=None):
        calls.append(("closeup", src.name, bool(reference)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "final.jpg").write_bytes(b"j")
        return {"warnings": []}

    monkeypatch.setattr(cms.phone, "process_upload", wall)
    monkeypatch.setattr(cms.closeup, "process_closeup", close)
    monkeypatch.setattr(cms.exposure, "exposure_scale", lambda path: 1.0)


def test_front_is_processed_before_closeups_and_gives_them_the_reference(monkeypatch):
    calls = []
    _fake_processing(monkeypatch, calls)
    fake = FakeCms(
        [_doc(3, "develop", "detail"), _doc(2, "wall", "back"), _doc(1, "wall", "front")]
    )
    assert cms.run_once(fake, log=lambda s: None) == 3
    assert [c[:2] for c in calls] == [
        ("wall", "IMG_1.DNG"),
        ("wall", "IMG_2.DNG"),
        ("closeup", "IMG_3.DNG"),
    ]
    assert calls[2][2] is True  # 正面の補正を使った
    # 基準（最初の正面）は自分のカードでカーブを作り、背面は基準のカーブを使う
    assert calls[0][2] is None and calls[1][2] == [[0.1, 0.1], [0.5, 0.6]]


def test_closeups_wait_while_another_worker_processes_the_front(monkeypatch):
    calls = []
    _fake_processing(monkeypatch, calls)
    closeup_doc = _doc(3, "develop", "detail")
    fake = FakeCms(
        [closeup_doc],
        sources=[
            {**_doc(1, "wall", "front"), "status": "processing"},
            {**closeup_doc, "status": "pending"},
        ],
    )
    assert cms.run_once(fake, log=lambda s: None) == 0
    assert fake.claimed == []


def test_closeups_use_a_finished_front_from_an_earlier_run(monkeypatch):
    calls = []
    _fake_processing(monkeypatch, calls)
    front = {
        **_doc(1, "wall", "front"),
        "status": "done",
        "report": {
            "exposure_scale": 1.0,
            "color": {
                "source": "linear",
                "gain": [1, 1, 1],
                "gamma": [1, 1, 1],
                "matrix": np.eye(3).tolist(),
            },
        },
    }
    closeup_doc = _doc(3, "develop", "detail")
    fake = FakeCms([closeup_doc], sources=[front, {**closeup_doc, "status": "pending"}])
    assert cms.run_once(fake, log=lambda s: None) == 1
    assert calls == [("closeup", "IMG_3.DNG", True)]


def test_failure_is_reported_to_the_cms(monkeypatch):
    calls = []
    _fake_processing(monkeypatch, calls)

    def broken(src, out, reference, label=None):
        raise ValueError("読めません")

    monkeypatch.setattr(cms.closeup, "process_closeup", broken)
    fake = FakeCms([_doc(3, "develop", "detail")])
    assert cms.run_once(fake, log=lambda s: None) == 0
    assert "読めません" in fake.errors[3]


def test_dark_closeups_are_lifted_but_not_brighter_than_auto():
    dark = np.full((10, 10, 3), 0.01, np.float32)
    # タグは自動の 1 段下まで、生地などは 2 段下まで持ち上げる
    assert abs(closeup.brightness_factor(dark, 1.0) * 0.01 - closeup.AUTO_BRIGHT / 2) < 1e-6
    assert abs(closeup.brightness_factor(dark, 2.0) * 0.01 - closeup.AUTO_BRIGHT / 4) < 1e-6
    # 明るすぎるものは自動の明るさまで下げ、その間は変えない
    assert (
        closeup.brightness_factor(np.full((4, 4, 3), 2.0, np.float32)) == closeup.AUTO_BRIGHT / 2.0
    )
    assert closeup.brightness_factor(np.full((4, 4, 3), 0.5, np.float32)) == 1.0
