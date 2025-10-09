"""
Compare measured vs reference centroid-shift for a specified defocus.

Measured:
- Reads one CSV from `csv_files/` whose last 4 digits match focus+defocus(mm)*100
- Averages by identical y, centers by median(y), filters to ±300 µm
- Computes normalized horizontal centroid shift, then x-shifts so y=0 at x=0

Reference:
- Loads `reference_csv/voltage_raito_-185dBm_test.csv`
- X axis is in mm; converted to µm
- Column selected by defocus (e.g., f0, f2, f4, ...)
- X-shifted so y=0 at x=0

Output:
- Saves to `plots/compare/def_+x.xxmm.png`
- Optionally shows figure with --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_qd_data import read_qd_csv, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
from plot_all_centroid_ave import compute_centered_avg_series, compute_x_intercept


REFERENCE_CSV_DEFAULT = Path("reference_csv") / "voltage_raito_-185dBm_test.csv"


def find_csv_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def extract_last4_digits(stem: str) -> Optional[int]:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits[-4:])
    except ValueError:
        return None


def select_measured_csv_for_defocus(input_dir: Path, focus_um4: int, defocus_mm: float) -> Optional[Path]:
    """Select measured CSV whose last-4 digits equal focus+defocus(mm)*100.

    focus_um4: focus position as 4-digit integer (e.g., 1200 for 12.00 mm)
    defocus_mm: desired defocus in mm (may be negative)
    """
    target_code = int(round(focus_um4 + defocus_mm * 100.0))
    target_str = f"{target_code:04d}"
    for p in find_csv_files(input_dir):
        last4 = extract_last4_digits(p.stem)
        if last4 is not None and f"{last4:04d}" == target_str:
            return p
    return None


def load_reference_series(reference_csv: Path, defocus_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load reference curve (x in µm, y ratio) for the given defocus.

    The CSV is expected to have a first column for the position in mm and
    subsequent columns named like 'voltage_ratio[-](de=f0 mm)', '...f2 mm', etc.
    """
    df = pd.read_csv(reference_csv)
    # Detect the position column (in mm)
    x_mm_col = next((c for c in df.columns if "mm" in str(c) and "true" in str(c).lower()), df.columns[0])

    # Find the matching defocus column by integer mm value (e.g., f0, f2)
    target_int = int(round(defocus_mm))
    available_columns = list(df.columns[1:])
    chosen_col = None
    for col in available_columns:
        text = str(col)
        # Look for pattern f<number> (mm)
        # Accept both 'f0 mm' and 'f0mm'
        for signless in [f"f{abs(target_int)} ", f"f{abs(target_int)}m", f"f{abs(target_int)}_"]:
            if signless in text.replace(".", "").replace("-", "").replace("(", " ").replace(")", " "):
                chosen_col = col
                break
        if chosen_col is not None:
            break
    # Fallback: try simple contains 'f{int} '
    if chosen_col is None:
        for col in available_columns:
            if f"f{abs(target_int)}" in str(col):
                chosen_col = col
                break
    if chosen_col is None:
        raise ValueError(f"No reference column found for defocus ~ {target_int} mm in {reference_csv.name}")

    x_um = df[x_mm_col].to_numpy(dtype=float) * 1000.0
    y = df[chosen_col].to_numpy(dtype=float)
    return x_um, y


def plot_comparison(
    measured_csv: Path,
    reference_csv: Path,
    focus_um4: int,
    defocus_mm: float,
    output_path: Path,
    show: bool,
) -> None:
    # Measured series
    df_meas = read_qd_csv(measured_csv)
    x_meas_centered, y_meas, center_y, _ = compute_centered_avg_series(df_meas)
    x0_meas = compute_x_intercept(x_meas_centered, y_meas)
    x_meas = x_meas_centered - x0_meas

    # Reference series
    x_ref_um, y_ref = load_reference_series(reference_csv, defocus_mm)
    x0_ref = compute_x_intercept(x_ref_um, y_ref)
    x_ref = x_ref_um - x0_ref

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.plot(x_ref, y_ref, color="tab:blue", linewidth=2.2, label=f"reference (def={defocus_mm:+.2f}mm)")
    ax.plot(x_meas, y_meas, color="tab:orange", linewidth=1.8, marker="o", markersize=3.0, label=f"measured (def={defocus_mm:+.2f}mm)")
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("y_axis - center [µm] (x-shifted so y=0 at x=0)")
    ax.set_ylabel("normalized horizontal centroid shift")
    ax.set_title("Measured vs Reference (averaged & centered)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Saved: {output_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare measured vs reference for one or more defocus values")
    parser.add_argument(
        "--defocus-mm",
        type=float,
        nargs="+",
        default=[0.0],
        help="One or more defocus values in mm (e.g., 0 2 4)",
    )
    parser.add_argument("--focus", type=int, default=1200, help="Focus position as 4-digit int (e.g., 1200 -> 12.00 mm)")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Measured CSV directory (default: csv_files)")
    parser.add_argument("--reference-csv", type=Path, default=REFERENCE_CSV_DEFAULT, help="Reference CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base output directory (default: plots)")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving")

    args = parser.parse_args()

    out_dir = args.output_dir / "compare"

    any_plotted = False
    for def_mm in args.defocus_mm:
        measured_csv = select_measured_csv_for_defocus(args.input_dir, int(args.focus), float(def_mm))
        if measured_csv is None:
            print(
                f"Skip: No measured CSV for focus={int(args.focus)} and defocus={def_mm:+.2f} mm in {args.input_dir}."
            )
            continue

        sign = "+" if def_mm >= 0 else "-"
        out_name = f"def_{sign}{abs(def_mm):.2f}mm.png".replace("+", "+")
        out_path = out_dir / out_name

        plot_comparison(
            measured_csv=measured_csv,
            reference_csv=args.reference_csv,
            focus_um4=int(args.focus),
            defocus_mm=float(def_mm),
            output_path=out_path,
            show=bool(args.show),
        )
        any_plotted = True

    if not any_plotted:
        print("No plots generated.")


if __name__ == "__main__":
    main()


