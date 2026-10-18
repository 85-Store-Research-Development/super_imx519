"""4 隅の ArUco マーカーを、A4 1 枚に 10 cm 角 × 4 で並べた印刷用の PDF に書き出す。

仕様は super_imx519.markers。印刷は「実際のサイズ（100%）」で、用紙に合わせる縮小をしない。

usage:
    uv run python scripts/make_markers.py --out ~/data/super_imx519/markers_a4.pdf
"""

import argparse
from pathlib import Path

from super_imx519.markers import markers_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("markers_a4.pdf"))
    args = parser.parse_args()
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(markers_pdf())
    print(out)


if __name__ == "__main__":
    main()
