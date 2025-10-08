"""
Plot QD sensor CSV data.

This script reads all CSV files under `csv_files/` relative to the project root.
For each CSV, it expects the following columns (Excel-style A..E mapping shown):

- A: y-axis  (header may be `y-axis` or `y_axis`)
- B: value1
- C: value2
- D: value3
- E: value4

Two figures are generated per CSV (plus an optional centered third plot):
1) value1..value4 vs y-axis (all lines on a single axes)
2) ((value2+value4) - (value1+value3)) / (value1+value2+value3+value4) vs y-axis

Images are saved to `plots/<basename>_values.png` and
`plots/<basename>_centroid_shift.png`. Additionally, a centered-window plot
filtered to 2700..3300 and re-centered at 3000 is saved as
`plots/<basename>_centroid_shift_centered.png`. An averaged version
grouped by identical y values is also saved as
`plots/<basename>_centroid_shift_centered_avg.png`. Optionally show with --show.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("csv_files")
DEFAULT_OUTPUT_DIR = Path("plots")


def read_qd_csv(csv_path: Path) -> pd.DataFrame:
    """Read a QD CSV file and return a DataFrame with expected columns.

    The function is resilient to BOM and minor header variations. If headers are
    missing, it falls back to positional columns A..E -> y,value1..value4.
    """

    # Try with headers first (typical Excel-exported CSV includes headers)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Fallback encoding often used on Windows
        df = pd.read_csv(csv_path, encoding="cp932")

    # Normalize column names for robustness
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    candidate_y_names = ["y-axis", "y_axis", "y", "yaxis"]
    required_value_names = ["value1", "value2", "value3", "value4"]

    has_headers = any(name in df.columns for name in candidate_y_names) and all(
        name in df.columns for name in required_value_names
    )

    if not has_headers:
        # Re-read without header; map by position
        try:
            df = pd.read_csv(
                csv_path,
                header=None,
                encoding="utf-8-sig",
                usecols=[0, 1, 2, 3, 4],
                names=["y_axis", "value1", "value2", "value3", "value4"],
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                csv_path,
                header=None,
                encoding="cp932",
                usecols=[0, 1, 2, 3, 4],
                names=["y_axis", "value1", "value2", "value3", "value4"],
            )
    else:
        # Build a standardized DataFrame with canonical column names
        y_col = next((c for c in candidate_y_names if c in df.columns), None)
        assert y_col is not None
        df = df[[y_col, *required_value_names]].rename(columns={y_col: "y_axis"})

    # Ensure numeric types; coerce invalid entries to NaN
    for col in ["y_axis", "value1", "value2", "value3", "value4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are entirely NaN across the measurement columns
    df = df.dropna(subset=["y_axis", "value1", "value2", "value3", "value4"], how="any")

    # Sort by y for cleaner lines in case data were collected out-of-order
    df = df.sort_values("y_axis").reset_index(drop=True)
    return df


def compute_horizontal_centroid_shift(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized horizontal centroid shift.

    shift = ((v2 + v4) - (v1 + v3)) / (v1 + v2 + v3 + v4)
    """
    y_values = df["y_axis"].to_numpy()
    v1 = df["value1"].to_numpy()
    v2 = df["value2"].to_numpy()
    v3 = df["value3"].to_numpy()
    v4 = df["value4"].to_numpy()

    numerator = (v2 + v4) - (v1 + v3)
    denominator = v1 + v2 + v3 + v4

    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(denominator != 0, numerator / denominator, np.nan)

    return y_values, shift


def plot_values(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(df["y_axis"], df["value1"], marker="o", label="value1")
    ax.plot(df["y_axis"], df["value2"], marker="o", label="value2")
    ax.plot(df["y_axis"], df["value3"], marker="o", label="value3")
    ax.plot(df["y_axis"], df["value4"], marker="o", label="value4")
    ax.set_xlabel("y_axis")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_centroid_shift(df: pd.DataFrame, title: str, output_path: Path) -> None:
    y_values, shift = compute_horizontal_centroid_shift(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(y_values, shift, marker="o", color="tab:red", label="((v2+v4)-(v1+v3))/sum")
    ax.set_xlabel("y_axis")
    ax.set_ylabel("normalized horizontal centroid shift")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_centroid_shift_centered(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    center_y: float = 3000.0,
    window_min: float = 2700.0,
    window_max: float = 3300.0,
) -> None:
    """Plot centroid shift within a y window, with x shifted so center_y -> 0.

    Keeps only rows with window_min <= y_axis <= window_max and uses
    (y_axis - center_y) on the x-axis.
    """
    df_win = df[(df["y_axis"] >= window_min) & (df["y_axis"] <= window_max)].copy()
    if df_win.empty:
        # Nothing in range; skip saving an empty plot
        return
    y_values, shift = compute_horizontal_centroid_shift(df_win)
    x_centered = y_values - center_y

    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    ax.plot(x_centered, shift, marker="o", color="tab:purple", label="centered shift")
    ax.set_xlabel(f"y_axis - {int(center_y)}")
    ax.set_ylabel("normalized horizontal centroid shift")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_centroid_shift_centered_avg(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    center_y: float = 3000.0,
    window_min: float = 2700.0,
    window_max: float = 3300.0,
) -> None:
    """Average value1..4 by identical y, then plot centered-window shift.

    Steps:
    - group by y_axis and average value1..4
    - compute centroid shift on the averaged values
    - filter to [window_min, window_max]
    - plot with x = (y - center_y)
    """
    grouped = (
        df.groupby("y_axis", as_index=False)[["value1", "value2", "value3", "value4"]]
        .mean()
        .sort_values("y_axis")
        .reset_index(drop=True)
    )

    df_win = grouped[(grouped["y_axis"] >= window_min) & (grouped["y_axis"] <= window_max)]
    if df_win.empty:
        return

    y_values, shift = compute_horizontal_centroid_shift(df_win)
    x_centered = y_values - center_y

    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    ax.plot(x_centered, shift, marker="o", color="tab:green", label="centered avg shift")
    ax.set_xlabel(f"y_axis - {int(center_y)}")
    ax.set_ylabel("normalized horizontal centroid shift (avg)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_file(csv_path: Path, output_dir: Path, show: bool) -> None:
    df = read_qd_csv(csv_path)

    base = csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = output_dir / f"{base}_values.png"
    fig2_path = output_dir / f"{base}_centroid_shift.png"
    fig3_path = output_dir / f"{base}_centroid_shift_centered.png"
    fig4_path = output_dir / f"{base}_centroid_shift_centered_avg.png"

    plot_values(df, f"{base} - value1..4 vs y_axis", fig1_path)
    plot_centroid_shift(df, f"{base} - centroid shift vs y_axis", fig2_path)
    plot_centroid_shift_centered(
        df,
        f"{base} - centroid shift (centered at 3000; 2700..3300)",
        fig3_path,
    )
    plot_centroid_shift_centered_avg(
        df,
        f"{base} - centroid shift (centered; averaged per y)",
        fig4_path,
    )

    if show:
        # If --show is set, display the last rendered figure for quick look.
        # Re-render lightweight combined display.
        fig, axes = plt.subplots(4, 1, figsize=(8.0, 14.0), constrained_layout=True)
        axes[0].plot(df["y_axis"], df["value1"], marker="o", label="value1")
        axes[0].plot(df["y_axis"], df["value2"], marker="o", label="value2")
        axes[0].plot(df["y_axis"], df["value3"], marker="o", label="value3")
        axes[0].plot(df["y_axis"], df["value4"], marker="o", label="value4")
        axes[0].set_title(f"{base} - values")
        axes[0].set_xlabel("y_axis")
        axes[0].set_ylabel("value")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        y_values, shift = compute_horizontal_centroid_shift(df)
        axes[1].plot(y_values, shift, marker="o", color="tab:red", label="centroid shift")
        axes[1].set_title(f"{base} - centroid shift")
        axes[1].set_xlabel("y_axis")
        axes[1].set_ylabel("normalized shift")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        # Third subplot: centered window
        df_win = df[(df["y_axis"] >= 2700) & (df["y_axis"] <= 3300)].copy()
        if not df_win.empty:
            y_values_c, shift_c = compute_horizontal_centroid_shift(df_win)
            axes[2].plot(y_values_c - 3000.0, shift_c, marker="o", color="tab:purple", label="centered (2700..3300)")
            axes[2].set_title(f"{base} - centroid shift (centered at 3000)")
            axes[2].set_xlabel("y_axis - 3000")
            axes[2].set_ylabel("normalized shift")
            axes[2].grid(True, linestyle=":", alpha=0.6)
            axes[2].legend()

        # Fourth subplot: centered window on averaged-by-y
        grouped = (
            df.groupby("y_axis", as_index=False)[["value1", "value2", "value3", "value4"]]
            .mean()
            .sort_values("y_axis")
            .reset_index(drop=True)
        )
        df_win_avg = grouped[(grouped["y_axis"] >= 2700) & (grouped["y_axis"] <= 3300)]
        if not df_win_avg.empty:
            y_values_a, shift_a = compute_horizontal_centroid_shift(df_win_avg)
            axes[3].plot(y_values_a - 3000.0, shift_a, marker="o", color="tab:green", label="centered avg (per y)")
            axes[3].set_title(f"{base} - centered shift (avg per y)")
            axes[3].set_xlabel("y_axis - 3000")
            axes[3].set_ylabel("normalized shift (avg)")
            axes[3].grid(True, linestyle=":", alpha=0.6)
            axes[3].legend()
        plt.show()


def find_csv_files(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QD sensor CSV data")
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
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving",
    )

    args = parser.parse_args()

    csv_files = find_csv_files(args.input_dir)
    if not csv_files:
        print(f"No CSV files found in: {args.input_dir.resolve()}")
        return

    for csv_path in csv_files:
        print(f"Processing {csv_path} ...")
        process_file(csv_path, args.output_dir, args.show)
    print(f"Done. Plots saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()


