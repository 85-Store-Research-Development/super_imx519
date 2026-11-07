"""同じ壁を位置や角度を変えて撮った写真から、壁の配置（マーカーとフック）を較正する。

1. 写真ごとにマーカーを見つけ、配置（マーカーの位置と傾き）をまとめて解く（pipeline/calibrate.py）
2. 較正した配置で各写真を正面図にし、フックを見つけて、壁面上の位置の中央値をとる
3. レンズの歪みが残っていないかを、較正した配置を的にしたカメラ較正で確かめる（記録だけ）
4. --out に書く。スタジオ（iPhone カメラモード）は ~/data/super_imx519/wall.json を読む

フックやマーカーを動かしたら、撮り直して較正し直す。

usage:
    uv run python scripts/calibrate_wall.py photos/*.JPG [--out ~/data/super_imx519/wall.json]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from super_imx519.pipeline import calibrate, color, phone, rectify


def lens_check(views: list[dict], layout: rectify.WallLayout, size: tuple[int, int]) -> dict:
    """配置を的にしたカメラ較正。

    歪みを 0 に固定したときと、k1・k2 を解いたときの再投影誤差を比べる。
    """
    obj, img = [], []
    for found in views:
        ids = sorted(i for i in found if i in layout.markers)
        if len(ids) < 2:
            continue
        pts = np.vstack([layout.corners(i) for i in ids])
        obj.append(np.hstack([pts, np.zeros((len(pts), 1))]).astype(np.float32))
        img.append(np.vstack([found[i] for i in ids]).astype(np.float32))
    w, h = size
    # iPhone の 1 倍のカメラは、35 mm 判の 24〜26 mm 相当（長辺 36 mm に対して）
    f = max(w, h) * 25 / 36
    k0 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    base = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    out = {}
    for name, flags in (
        ("no_distortion", base | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2),
        ("k1_k2", base),
    ):
        rms, k, dist, _, _ = cv2.calibrateCamera(
            obj, img, (w, h), k0.copy(), np.zeros(5), flags=flags
        )
        out[name] = {
            "rms_px": round(float(rms), 3),
            "focal_px": round(float(k[0, 0]), 1),
            "k1": round(float(dist.ravel()[0]), 5),
            "k2": round(float(dist.ravel()[1]), 5),
        }
    return out


def photo_name(path: Path) -> str:
    """スタジオに送られた写真（uploads/<id>/original.*）は、送ったときのファイル名で記録する。"""
    meta = path.parent / "upload.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("filename", path.name)
    return path.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=phone.DEFAULT_WALL)
    ap.add_argument("--dry-run", action="store_true", help="書かずに結果だけ出す")
    args = ap.parse_args()

    photos, views, details = [], [], []
    for path in args.photos:
        rgb, source = phone.load(path)
        detect, _ = phone.detection_image(rgb, source)
        found = rectify.detect_markers(cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY))
        print(f"{photo_name(path)}: マーカー {sorted(found)}")
        if len(found) >= 2:
            photos.append(path)
            views.append(found)
            details.append((detect, source))
    layout, report = calibrate.fit_layout(views)

    # フック: 較正した配置で正面図にして見つける
    hooks = []
    for path, (detect, _) in zip(photos, details, strict=True):
        view, _ = phone.make_view(detect, layout)
        image = np.clip(view.warp(detect.astype(np.float32)), 0, 255)
        scene = phone.analyze(image, view.valid(detect.shape), view, color.find_card(detect))
        hook = view.to_mm([scene.hook])[0]
        hooks.append(hook)
        name = photo_name(path)
        print(f"{name}: フックの上端 {hook.round(1)} mm（残差 {view.residual_mm[0]:.2f} mm）")
    hooks = np.array(hooks)
    layout.hook_mm = tuple(float(v) for v in np.median(hooks, axis=0))
    # ばらつきは中央値からの偏差の中央値（服の首を見誤った写真が混じっても膨らまない）
    report["hook_spread_mm"] = np.round(
        np.median(np.abs(hooks - np.median(hooks, axis=0)), axis=0), 2
    ).tolist()
    report["hook_outliers"] = [
        photo_name(p)
        for p, h in zip(photos, hooks, strict=True)
        if np.linalg.norm(h - np.array(layout.hook_mm)) > phone.HOOK_MOVED_MM
    ]

    size = details[0][0].shape[1::-1]
    if all(d[0].shape == details[0][0].shape for d in details):
        report["lens"] = lens_check(views, layout, size)

    layout.info = {
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "photos": [photo_name(p) for p in photos],
        **report,
    }
    print(json.dumps(layout.to_json(), indent=1, ensure_ascii=False))
    if not args.dry_run:
        layout.save(args.out.expanduser())
        print(f"書きました: {args.out}")


if __name__ == "__main__":
    main()
