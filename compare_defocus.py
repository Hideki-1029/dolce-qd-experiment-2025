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
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

from plot_qd_data import read_qd_csv, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
from plot_all_centroid_ave import compute_centered_avg_series, compute_x_intercept


# Reference CSV selection (choose at top-level)
# Available presets; add new entries as needed
REFERENCE_CSV_OPTIONS = {
    "185dBm_test": Path("reference_csv") / "voltage_raito_-185dBm_test.csv",
    "227_v2": Path("reference_csv") / "voltage_raito_227_v2.csv",
}
# Change this key to switch the default reference dataset
REFERENCE_SELECTION = "227_v2"
REFERENCE_CSV_DEFAULT = REFERENCE_CSV_OPTIONS[REFERENCE_SELECTION]
FLATSPOT_CSV_DEFAULT = Path("reference_csv") / "voltage_ratio_flat_spot.csv"


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


def autodetect_flatspot_excel(reference_dir: Path) -> Optional[Path]:
    # Excel support removed by request
    return None


def load_flatspot_series_from_csv(csv_path: Path, defocus_mm: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load flat-spot series where the first column is voltage ratio (y),
    and the defocus-specific columns contain the true position in mm (x).

    Expected header examples:
      - 'voltage_ratio[-]' (first column)
      - 'true_position(de=f4 mm)', 'true_position(de=f8 mm)', ... (x in mm)

    Some rows may contain strings like 'non'; they are ignored via NaN masking.
    """
    target_int = int(round(abs(float(defocus_mm))))
    if target_int == 0:
        return None

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # y (voltage ratio) column: prefer header containing 'voltage', else fall back to the first column
    y_col = next((c for c in df.columns if "voltage" in str(c).lower()), df.columns[0])

    # x (position in mm) column for the target defocus, chosen via regex
    available_columns = list(df.columns[1:])
    chosen_x_col = None
    pattern = re.compile(rf"f\s*{abs(target_int)}(\s|mm|\)|$)", re.IGNORECASE)
    for col in available_columns:
        text = str(col)
        if pattern.search(text):
            chosen_x_col = col
            break
    if chosen_x_col is None:
        print(f"Info: No flat-spot column found for def~{target_int} mm in {csv_path.name}")
        return None

    y_series = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    x_mm = pd.to_numeric(df[chosen_x_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_mm) & np.isfinite(y_series)
    x_mm = x_mm[mask]
    y = y_series[mask]
    if x_mm.size == 0:
        return None
    return x_mm, y


def plot_comparison(
    measured_csv: Path,
    reference_csv: Path,
    flatspot_csv: Optional[Path],
    focus_um4: int,
    defocus_mm: float,
    output_path: Path,
    show: bool,
    plot_measured: bool = True,
    plot_reference: bool = True,
    plot_flatspot: bool = True,
    report_metrics: bool = False,
    metrics_window_mm: float = 0.05,
    metrics_csv: Optional[Path] = None,
    focal_mm: float = 75.0,
    beam_diameter_mm: float = 2.27,
) -> dict:
    # Prepare data conditionally per flags
    x_meas = None
    y_meas = None
    if plot_measured:
        df_meas = read_qd_csv(measured_csv)
        x_meas_centered, y_meas_calc, center_y, _ = compute_centered_avg_series(df_meas)
        x0_meas = compute_x_intercept(x_meas_centered, y_meas_calc)
        x_meas = x_meas_centered - x0_meas
        y_meas = y_meas_calc

    x_ref_um_shifted = None
    y_ref = None
    if plot_reference:
        x_ref_um, y_ref_calc = load_reference_series(reference_csv, defocus_mm)
        x0_ref = compute_x_intercept(x_ref_um, y_ref_calc)
        x_ref_um_shifted = x_ref_um - x0_ref
        y_ref = y_ref_calc

    # Optional: Flat-spot reference from CSV (only for non-zero defocus)
    x_flat_mm_shifted: Optional[np.ndarray] = None
    y_flat: Optional[np.ndarray] = None
    if plot_flatspot and abs(defocus_mm) > 1e-6:
        flat: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if flatspot_csv is not None:
            flat = load_flatspot_series_from_csv(flatspot_csv, defocus_mm)
        if flat is not None:
            x_flat_mm, y_flat_arr = flat
            try:
                x0_flat = compute_x_intercept(x_flat_mm, y_flat_arr)
            except Exception:
                x0_flat = 0.0
            x_flat_mm_shifted = x_flat_mm - x0_flat
            y_flat = y_flat_arr
        else:
            if flatspot_csv is not None:
                print(f"Info: flat-spot data not found in {flatspot_csv.name} for def~{defocus_mm:.2f} mm.")

    # Compute and optionally report metrics when both measured and reference exist
    metrics_text_lines: List[str] = []
    if (
        report_metrics
        and x_meas is not None and y_meas is not None
        and x_ref_um_shifted is not None and y_ref is not None
    ):
        x_meas_mm = x_meas / 1000.0
        x_ref_mm = x_ref_um_shifted / 1000.0
        meas_mask = np.isfinite(x_meas_mm) & (np.abs(x_meas_mm) <= float(metrics_window_mm)) & np.isfinite(y_meas)
        ref_mask = np.isfinite(x_ref_mm) & (np.abs(x_ref_mm) <= float(metrics_window_mm)) & np.isfinite(y_ref)
        # Align sample domains by interpolation of reference onto measured x within window
        try:
            # Linear regression with zero intercept: slope = sum(x*y)/sum(x^2)
            def slope_zero_intercept(x: np.ndarray, y: np.ndarray) -> float:
                denom = float(np.sum(x * x))
                return float(np.sum(x * y) / denom) if denom > 0 else float('nan')

            xw_meas = x_meas_mm[meas_mask]
            yw_meas = y_meas[meas_mask]
            xw_ref = x_ref_mm[ref_mask]
            yw_ref = y_ref[ref_mask]
            # Interpolate reference y at measured x points to form pairs
            if xw_meas.size >= 3 and xw_ref.size >= 3:
                yref_on_meas = np.interp(xw_meas, xw_ref, yw_ref)
                slope_meas = slope_zero_intercept(xw_meas, yw_meas)
                slope_ref = slope_zero_intercept(xw_meas, yref_on_meas)
                slope_ratio = slope_meas / slope_ref if np.isfinite(slope_meas) and np.isfinite(slope_ref) and slope_ref != 0 else float('nan')
                # Gain that best maps reference->measured in LS sense within window
                denom_g = float(np.sum(yref_on_meas * yref_on_meas))
                gain = float(np.sum(yref_on_meas * yw_meas) / denom_g) if denom_g > 0 else float('nan')
                residual = yw_meas - gain * yref_on_meas
                rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size > 0 else float('nan')
                # Correlation (Pearson) within window
                if yref_on_meas.size > 1 and np.std(yref_on_meas) > 0 and np.std(yw_meas) > 0:
                    corr = float(np.corrcoef(yref_on_meas, yw_meas)[0, 1])
                else:
                    corr = float('nan')

                # Geometric spot diameter on QD at defocus d: ds = (D/f) * |d|
                ds_mm = float(abs(defocus_mm)) * float(beam_diameter_mm) / float(focal_mm) if focal_mm > 0 else float('nan')

                # K estimates from slopes and ds: K ≈ 1 / (ds * slope)
                def estimate_k(ds: float, slope: float) -> float:
                    return float(1.0 / (ds * slope)) if (np.isfinite(ds) and ds > 0 and np.isfinite(slope) and slope != 0) else float('nan')

                k_meas = estimate_k(ds_mm, slope_meas)
                k_ref = estimate_k(ds_mm, slope_ref)

                metrics_text_lines = [
                    f"window=±{metrics_window_mm:.3f} mm",
                    f"ds={ds_mm:.4f} mm",
                    f"K_meas≈{k_meas:.3f}",
                    f"K_ref≈{k_ref:.3f}",
                    f"slope_meas={slope_meas:.4f} per mm",
                    f"slope_ref={slope_ref:.4f} per mm",
                    f"slope_ratio(meas/ref)={slope_ratio:.3f}",
                    f"gain(ref→meas)={gain:.3f}",
                    f"rmse={rmse:.4f}",
                    f"corr={corr:.3f}",
                ]

                # Append to CSV if requested
                if metrics_csv is not None:
                    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
                    write_header = not metrics_csv.exists()
                    with open(metrics_csv, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow([
                                "defocus_mm",
                                "reference_key",
                                "focal_mm",
                                "beam_diameter_mm",
                                "window_mm",
                                "ds_mm",
                                "slope_meas_per_mm",
                                "slope_ref_per_mm",
                                "slope_ratio",
                                "gain",
                                "rmse",
                                "corr",
                                "K_meas",
                                "K_ref",
                            ])
                        writer.writerow([
                            f"{defocus_mm:.2f}",
                            REFERENCE_SELECTION,
                            f"{float(focal_mm):.3f}",
                            f"{float(beam_diameter_mm):.3f}",
                            f"{metrics_window_mm:.3f}",
                            f"{ds_mm:.6f}",
                            f"{slope_meas:.6f}",
                            f"{slope_ref:.6f}",
                            f"{slope_ratio:.6f}",
                            f"{gain:.6f}",
                            f"{rmse:.6f}",
                            f"{corr:.6f}",
                            f"{k_meas:.6f}",
                            f"{k_ref:.6f}",
                        ])
            else:
                metrics_text_lines = [f"window=±{metrics_window_mm:.3f} mm", "insufficient samples for metrics"]
        except Exception as e:
            metrics_text_lines = [f"metrics error: {e}"]

    # Helper for RMSE within window using interpolation onto measured x
    def compute_windowed_rmse(
        x_meas_mm_arr: np.ndarray,
        y_meas_arr: np.ndarray,
        x_ref_mm_arr: np.ndarray,
        y_ref_arr: np.ndarray,
        window_half_mm: float,
    ) -> Tuple[float, int]:
        if (
            x_meas_mm_arr is None
            or y_meas_arr is None
            or x_ref_mm_arr is None
            or y_ref_arr is None
        ):
            return float('nan'), 0
        mask_meas = (
            np.isfinite(x_meas_mm_arr)
            & np.isfinite(y_meas_arr)
            & (np.abs(x_meas_mm_arr) <= float(window_half_mm))
        )
        if not np.any(mask_meas):
            return float('nan'), 0
        x_in = x_meas_mm_arr[mask_meas]
        y_in = y_meas_arr[mask_meas]
        # Limit to ref domain to avoid extrapolation artifacts
        xmin = float(np.nanmin(x_ref_mm_arr))
        xmax = float(np.nanmax(x_ref_mm_arr))
        mask_domain = (x_in >= xmin) & (x_in <= xmax)
        x_in = x_in[mask_domain]
        y_in = y_in[mask_domain]
        if x_in.size < 2:
            return float('nan'), int(x_in.size)
        y_ref_interp = np.interp(x_in, x_ref_mm_arr, y_ref_arr)
        residual = y_in - y_ref_interp
        rmse_val = float(np.sqrt(np.mean(residual * residual)))
        return rmse_val, int(x_in.size)

    # Compute ds and P0 (linear FOV half-width) in mm
    ds_mm_for_defocus = float(abs(defocus_mm)) * float(beam_diameter_mm) / float(focal_mm) if float(focal_mm) > 0 else float('nan')
    p0_mm = float(math.pi / 8.0) * ds_mm_for_defocus if np.isfinite(ds_mm_for_defocus) else float('nan')

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    # Convert x-axes to mm for display
    if x_ref_um_shifted is not None and y_ref is not None:
        ax.plot(x_ref_um_shifted / 1000.0, y_ref, color="tab:blue", linewidth=2.2, label=f"reference (def={defocus_mm:+.2f}mm)")
    if x_flat_mm_shifted is not None and y_flat is not None:
        ax.plot(x_flat_mm_shifted, y_flat, color="tab:green", linewidth=1.8, label=f"reference_flatspot (def={defocus_mm:+.2f}mm)")
    if x_meas is not None and y_meas is not None:
        ax.plot(x_meas / 1000.0, y_meas, color="tab:orange", linewidth=1.8, marker="o", markersize=3.0, label=f"measured (def={defocus_mm:+.2f}mm)")
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("y_axis - center [mm] (x-shifted so y=0 at x=0)")
    ax.set_ylabel("normalized horizontal centroid shift")
    ax.set_title("Measured vs Reference (averaged & centered)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    if report_metrics and len(metrics_text_lines) > 0:
        ax.text(
            0.98,
            0.02,
            "\n".join(metrics_text_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Saved: {output_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # RMSE within ±P0 for reference and flat-spot
    rmse_ref = float('nan')
    n_ref = 0
    rmse_flat = float('nan')
    n_flat = 0
    try:
        if (
            x_meas is not None
            and y_meas is not None
            and x_ref_um_shifted is not None
            and y_ref is not None
            and np.isfinite(p0_mm)
            and p0_mm > 0
        ):
            rmse_ref, n_ref = compute_windowed_rmse(
                x_meas / 1000.0,
                y_meas,
                x_ref_um_shifted / 1000.0,
                y_ref,
                p0_mm,
            )
        if (
            x_meas is not None
            and y_meas is not None
            and x_flat_mm_shifted is not None
            and y_flat is not None
            and np.isfinite(p0_mm)
            and p0_mm > 0
        ):
            rmse_flat, n_flat = compute_windowed_rmse(
                x_meas / 1000.0,
                y_meas,
                x_flat_mm_shifted,
                y_flat,
                p0_mm,
            )
    except Exception:
        pass

    return {
        "defocus_mm": float(defocus_mm),
        "ds_mm": float(ds_mm_for_defocus),
        "P0_mm": float(p0_mm),
        "rmse_meas_vs_ref": float(rmse_ref),
        "rmse_meas_vs_flat": float(rmse_flat),
        "n_ref": int(n_ref),
        "n_flat": int(n_flat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare measured vs reference for one or more defocus values")
    parser.add_argument(
        "--defocus-mm",
        type=float,
        nargs="+",
        default=[0, 2, 4, 6, 8, 10],
        help="One or more defocus values in mm (default: 0 2 4 6 8 10)",
    )
    parser.add_argument("--focus", type=int, default=1200, help="Focus position as 4-digit int (e.g., 1200 -> 12.00 mm)")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Measured CSV directory (default: csv_files)")
    parser.add_argument("--reference-csv", type=Path, default=REFERENCE_CSV_DEFAULT, help="Reference CSV path")
    parser.add_argument("--flatspot-csv", type=Path, default=None, help="Flat-spot CSV path (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base output directory (default: plots)")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving")
    parser.add_argument(
        "--series",
        nargs="+",
        choices=["measured", "reference", "flatspot"],
        default=["measured", "reference", "flatspot"],
        help="Which series to plot (choose one or more)",
    )
    parser.add_argument(
        "--report-metrics",
        action="store_true",
        help="Compute and print agreement metrics near origin (also annotate plot)",
    )
    parser.add_argument(
        "--metrics-window-mm",
        type=float,
        default=0.05,
        help="Half window size around origin in mm for metrics (default: 0.05)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV file to append per-defocus metrics",
    )
    parser.add_argument(
        "--focal-mm",
        type=float,
        default=75.0,
        help="Lens focal length in mm (default: 75.0)",
    )
    parser.add_argument(
        "--beam-diameter-mm",
        type=float,
        default=2.27,
        help="Diameter of collimated beam before lens in mm (default: 2.27)",
    )

    args = parser.parse_args()

    out_dir = args.output_dir / "compare"

    # Determine flat-spot Excel path if not provided
    flatspot_csv_path: Optional[Path] = args.flatspot_csv
    if flatspot_csv_path is None:
        candidate = FLATSPOT_CSV_DEFAULT
        if candidate.exists():
            flatspot_csv_path = candidate
            print(f"Using flat-spot CSV: {flatspot_csv_path}")

    # Determine which series to draw
    plot_measured = "measured" in args.series
    plot_reference = "reference" in args.series
    plot_flatspot = "flatspot" in args.series

    any_plotted = False
    # Prepare metrics CSV writer (always generate a concise run summary CSV)
    run_metrics_csv = args.output_dir / "compare" / "rmse_summary.csv"
    run_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not run_metrics_csv.exists()
    csv_file_handle = open(run_metrics_csv, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file_handle)
    if write_header:
        writer.writerow([
            "defocus_mm",
            "reference_key",
            "focal_mm",
            "beam_diameter_mm",
            "ds_mm",
            "P0_mm",
            "rmse_meas_vs_ref",
            "rmse_meas_vs_flat",
            "n_ref",
            "n_flat",
            "output_png",
        ])
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

        metrics = plot_comparison(
            measured_csv=measured_csv,
            reference_csv=args.reference_csv,
            flatspot_csv=flatspot_csv_path,
            focus_um4=int(args.focus),
            defocus_mm=float(def_mm),
            output_path=out_path,
            show=bool(args.show),
            plot_measured=bool(plot_measured),
            plot_reference=bool(plot_reference),
            plot_flatspot=bool(plot_flatspot),
            report_metrics=bool(args.report_metrics),
            metrics_window_mm=float(args.metrics_window_mm),
            metrics_csv=args.metrics_csv,
            focal_mm=float(args.focal_mm),
            beam_diameter_mm=float(args.beam_diameter_mm),
        )
        any_plotted = True

        # Print concise line and write to summary CSV
        print(
            f"def={metrics['defocus_mm']:+.2f} mm | ds={metrics['ds_mm']:.4f} mm | P0={metrics['P0_mm']:.4f} mm | "
            f"RMSE(ref)={metrics['rmse_meas_vs_ref']:.4f} | RMSE(flat)={metrics['rmse_meas_vs_flat']:.4f}"
        )
        writer.writerow([
            f"{metrics['defocus_mm']:.2f}",
            REFERENCE_SELECTION,
            f"{float(args.focal_mm):.3f}",
            f"{float(args.beam_diameter_mm):.3f}",
            f"{metrics['ds_mm']:.6f}",
            f"{metrics['P0_mm']:.6f}",
            f"{metrics['rmse_meas_vs_ref']:.6f}",
            f"{metrics['rmse_meas_vs_flat']:.6f}",
            int(metrics['n_ref']),
            int(metrics['n_flat']),
            str(out_path),
        ])

    csv_file_handle.close()
    if not any_plotted:
        print("No plots generated.")


if __name__ == "__main__":
    main()


