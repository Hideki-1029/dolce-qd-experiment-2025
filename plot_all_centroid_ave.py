"""
Overlay averaged centered centroid-shift curves for all CSVs.

For each CSV under `csv_files/`, this script:
- Reads the data using `read_qd_csv`
- Groups by identical `y_axis` and averages `value1..4`
- Computes center = median(y_axis) and window = center ± 300
- Filters to that window, computes normalized horizontal centroid shift
- Horizontally offsets each series so the (linearly) interpolated zero-crossing is at x=0
- Plots all series on a single, large figure

Output:
- Saves a combined figure to `plots/centroid_ave/all_centroid_ave.png`
- Legend labels show the last 4 digits from the filename stem (e.g., 1531, 0014)

Optionally, use `--show` to display the figure interactively after saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_qd_data import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    read_qd_csv,
    compute_horizontal_centroid_shift,
)


def find_csv_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def extract_last4_digits_label(stem: str) -> str:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits[-4:].rjust(4, "0")
    return stem


def compute_centered_avg_series(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
    grouped = (
        df.groupby("y_axis", as_index=False)[["value1", "value2", "value3", "value4"]]
        .mean()
        .sort_values("y_axis")
        .reset_index(drop=True)
    )

    center_y = float(np.nanmedian(grouped["y_axis"].to_numpy()))
    window_min = center_y - 300.0
    window_max = center_y + 300.0
    df_win = grouped[(grouped["y_axis"] >= window_min) & (grouped["y_axis"] <= window_max)]
    if df_win.empty:
        return np.array([]), np.array([]), center_y, (window_min, window_max)

    y_values, shift = compute_horizontal_centroid_shift(df_win)
    x_centered = y_values - center_y
    return x_centered, shift, center_y, (window_min, window_max)


def compute_x_intercept(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute x0 where the linearly interpolated series crosses y=0.

    Strategy:
    - Prefer a segment with a sign change and solve exactly on that segment.
    - If no sign change exists, use the two points with the smallest |y| and
      solve for x at y=0 via their connecting line (extrapolation if needed).
    Returns x0 (may lie outside the sample range if extrapolated).
    """
    n = int(x_values.size)
    if n == 0:
        return 0.0

    # Fast path: any exact zero
    zero_indices = np.where(y_values == 0.0)[0]
    if zero_indices.size > 0:
        return float(x_values[int(zero_indices[0])])

    # Look for sign change across adjacent pairs
    signs = np.sign(y_values)
    for i in range(n - 1):
        y1 = float(y_values[i])
        y2 = float(y_values[i + 1])
        if y1 == y2:
            continue
        if (y1 < 0.0 and y2 > 0.0) or (y1 > 0.0 and y2 < 0.0):
            x1 = float(x_values[i])
            x2 = float(x_values[i + 1])
            # Linear interpolation to y=0
            t = -y1 / (y2 - y1)
            return x1 + t * (x2 - x1)

    # Fallback: take two points with smallest |y| and fit a line between them
    idx_sorted = np.argsort(np.abs(y_values))
    i1 = int(idx_sorted[0])
    i2 = int(idx_sorted[1]) if n >= 2 else i1
    x1, y1 = float(x_values[i1]), float(y_values[i1])
    x2, y2 = float(x_values[i2]), float(y_values[i2])
    if y2 == y1:
        return x1  # horizontal line near zero; choose x of closest point
    return x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay averaged centered centroid-shift for all CSVs")
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

    args = parser.parse_args()

    csv_files = find_csv_files(args.input_dir)
    if not csv_files:
        print(f"No CSV files found in: {args.input_dir.resolve()}")
        return

    centroid_ave_dir = args.output_dir / "centroid_ave"
    centroid_ave_dir.mkdir(parents=True, exist_ok=True)
    out_path = centroid_ave_dir / "all_centroid_ave.png"

    fig, ax = plt.subplots(figsize=(12.0, 8.0), constrained_layout=True)

    plotted_any = False
    for csv_path in csv_files:
        try:
            df = read_qd_csv(csv_path)
        except Exception as exc:
            print(f"Skipping {csv_path.name}: failed to read ({exc})")
            continue

        x_centered, shift, center_y, (wmin, wmax) = compute_centered_avg_series(df)
        if x_centered.size == 0:
            print(f"Skipping {csv_path.name}: no data in window around median")
            continue

        stem = csv_path.stem
        label = extract_last4_digits_label(stem)
        x0 = compute_x_intercept(x_centered, shift)
        x_shifted = x_centered - x0
        ax.plot(x_shifted, shift, marker="o", markersize=3.0, linewidth=1.2, label=label)
        plotted_any = True

    ax.set_xlabel("y_axis - center (per file, median)")
    ax.set_ylabel("normalized horizontal centroid shift (avg)")
    ax.set_title("Centroid shift (avg per y; centered at median ± 300; x-shifted to cross y=0 at x=0)")
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


