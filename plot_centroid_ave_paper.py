"""
Paper-style overlay plot of averaged, centered QD centroid-shift curves.

Differences from plot_all_centroid_ave.py:
- X label: "True spot position[mm]" (x is shifted so y=0 at x=0, then shown in mm)
- Y label: "Voltage ratio[-]"
- No title
- Legend labels use the last 4-digit code from filenames (e.g., 1200), not defocus mm

Reads CSVs from `csv_files/`, processes each:
  - average value1..4 per identical y
  - center window: median(y) ± 300 (µm)
  - compute normalized horizontal centroid shift
  - shift x so zero-crossing is at 0
  - convert x from µm to mm for display

Outputs a combined figure to `plots/centroid_ave/all_centroid_ave_paper.png` by default.
Use --show to display the figure after saving, and optionally pass specific 4-digit codes
to restrict which CSVs are plotted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from plot_qd_data import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    read_qd_csv,
)
from plot_all_centroid_ave import (
    compute_centered_avg_series,
    compute_x_intercept,
)


def find_csv_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def extract_last4_digits(stem: str) -> int | None:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    last4 = digits[-4:]
    try:
        return int(last4)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay averaged centered centroid-shift (paper style)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing CSV files (default: csv_files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: plots)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving",
    )
    parser.add_argument(
        "codes",
        nargs="*",
        help="Optional 4-digit codes to include (e.g., 1200 1400). If omitted, plot all.",
    )

    args = parser.parse_args()

    csv_files = find_csv_files(args.input_dir)
    if not csv_files:
        print(f"No CSV files found in: {args.input_dir.resolve()}")
        return

    centroid_ave_dir = args.output_dir / "centroid_ave"
    centroid_ave_dir.mkdir(parents=True, exist_ok=True)

    if args.codes:
        codes_set = set(str(c)[-4:].zfill(4) for c in args.codes if str(c).isdigit())
        suffix = "-".join(sorted(codes_set))
        out_path = centroid_ave_dir / f"all_centroid_ave_paper_{suffix}.png"
    else:
        codes_set = None
        out_path = centroid_ave_dir / "all_centroid_ave_paper.png"

    fig, ax = plt.subplots(figsize=(12.0, 8.0), constrained_layout=True)

    plotted_any = False
    for csv_path in csv_files:
        try:
            df = read_qd_csv(csv_path)
        except Exception as exc:
            print(f"Skipping {csv_path.name}: failed to read ({exc})")
            continue

        x_centered_um, shift, center_y, (wmin, wmax) = compute_centered_avg_series(df)
        if x_centered_um.size == 0:
            print(f"Skipping {csv_path.name}: no data in window around median")
            continue

        stem = csv_path.stem
        last4 = extract_last4_digits(stem)
        if codes_set:
            last4_str = str(last4).rjust(4, "0") if last4 is not None else None
            if last4_str not in codes_set:
                continue
        label = (str(last4).rjust(4, "0") if last4 is not None else stem)

        x0_um = compute_x_intercept(x_centered_um, shift)
        x_shifted_um = x_centered_um - x0_um
        x_shifted_mm = x_shifted_um / 1000.0

        ax.plot(x_shifted_mm, shift, marker="o", markersize=3.0, linewidth=1.2, label=label)
        plotted_any = True

    ax.set_xlabel("True spot position[mm]")
    ax.set_ylabel("Voltage ratio[-]")
    ax.grid(True, linestyle=":", alpha=0.6)
    if plotted_any:
        ax.legend(title="ID(last4)", ncol=4, fontsize=9)
    else:
        print("No valid series to plot.")

    fig.savefig(out_path, dpi=220)
    print(f"Saved: {out_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


