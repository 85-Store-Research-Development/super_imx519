import numpy as np

from super_imx519 import dataset

TABLE = {
    "points": ["着丈", "身幅", "肩幅", "袖丈", "メモ"],
    "rows": [{"size": "", "values": [72, 55.5, None, 61, 1]}],
}


def test_measured_values_from_cms_table():
    assert dataset.measured_values({"measurementTable": TABLE}) == {
        "着丈": 72.0,
        "身幅": 55.5,
        "袖丈": 61.0,
    }
    # サイズ違いの表・空は使わない
    sizes = {
        "points": ["着丈"],
        "rows": [{"size": "M", "values": [70]}, {"size": "L", "values": [72]}],
    }
    assert dataset.measured_values({"measurementTable": sizes}) is None
    assert dataset.measured_values({}) is None


def test_rows_are_the_corrected_wall_photos():
    product = {"id": 497, "title": "シャツ", "productType": "Shirts", "measurementTable": TABLE}
    report = {
        "wall": {"mm_per_px": 0.5, "residual_mm": [0.5, 0.9], "matrix": np.eye(3).tolist()},
        "measurements": {"values": {"着丈": 90.2}},
        "crop": {"hook_px": [1, 2]},
    }
    sources = [
        {"id": 10, "recipe": "wall", "status": "done", "role": "front", "report": report},
        {"id": 11, "recipe": "develop", "status": "done", "role": "detail", "report": {}},
        {"id": 12, "recipe": "wall", "status": "pending", "role": "back"},
    ]
    rows = dataset.rows_for(product, sources)
    assert [r["source"] for r in rows] == [10]
    assert rows[0]["measured"]["着丈"] == 72.0 and rows[0]["auto"] == {"着丈": 90.2}
    assert rows[0]["has_matrix"] is True and rows[0]["mm_per_px"] == 0.5


def test_matrix_shift():
    a = np.eye(3)
    b = np.array([[1, 0, 3], [0, 1, 0], [0, 0, 1]], float)  # 正面図で 3 px 右
    assert abs(dataset.matrix_shift_px(a.tolist(), b, (100, 100)) - 3) < 1e-9
    assert dataset.matrix_shift_px(a.tolist(), a, (100, 100)) == 0
