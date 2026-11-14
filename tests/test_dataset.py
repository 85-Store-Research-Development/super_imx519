import json

import numpy as np

from super_imx519 import dataset


class View:
    mm_per_px = 0.5
    markers = {0: np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)}
    residual_mm = (0.3, 0.8)
    matrix = np.eye(3)
    mm_to_px = np.diag([2.0, 2.0, 1.0])

    def to_mm(self, points):
        return np.asarray(points) * self.mm_per_px


def test_save_scene_crops_around_the_garment(tmp_path):
    wall = np.full((1000, 1000, 3), 200.0)
    garment = np.zeros((1000, 1000), bool)
    garment[400:800, 300:700] = True
    dataset.save_scene(
        tmp_path, wall, garment, View(), (500.0, 380.0), (300, 400, 699, 799), {"role": "front"}
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    # 服の範囲（とハンガー）から 100 mm（200 px）広げて切り出す
    assert meta["crop_px"] == [100, 180, 800, 820]
    assert meta["hook_px"] == [400.0, 200.0]
    assert meta["garment_bbox_px"] == [200, 220, 599, 619]
    assert meta["markers_mm"] == {"0": [2.5, 2.5]}
    assert meta["source_to_crop"][0][2] == -100 and meta["source_to_crop"][1][2] == -180
    assert meta["role"] == "front"
    assert (tmp_path / "wall.jpg").exists() and (tmp_path / "mask.png").exists()


def test_measured_values_from_cms_table():
    product = {
        "measurementTable": {
            "points": ["着丈", "身幅", "肩幅", "袖丈"],
            "rows": [{"size": "", "values": [72, 55.5, None, 61]}],
        }
    }
    assert dataset.measured_values(product) == {"着丈": 72.0, "身幅": 55.5, "袖丈": 61.0}
    # サイズ違いの表・空は使わない
    assert (
        dataset.measured_values(
            {
                "measurementTable": {
                    "points": ["着丈"],
                    "rows": [{"size": "M", "values": [70]}, {"size": "L", "values": [72]}],
                }
            }
        )
        is None
    )
    assert dataset.measured_values({}) is None
