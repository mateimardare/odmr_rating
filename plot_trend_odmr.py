"""
Temperature-dependence plots for PL5 / PL6 emitters.

Expects a dataframe (e.g. loaded from all_PL_peaks_all_samples.csv) with
columns:
    area, anneal_temp_C, measurement_folder, date, time,
    measurement_number, emitter, emitter_type, measurement_type,
    dip, x0, x0_u, fwhm, fwhm_u, amplitude, amplitude_u,
    baseline, contrast, contrast_u

PL5 rows: 1 row per measurement (single dip).
PL6 rows: 2 rows per measurement (one per "dip"), sharing the same
          measurement_folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

SAT_FIT = 0
HILL_FIT = 0
PL_OUTPUT_FD = "PL_analysis"

pl_out_fd = Path(PL_OUTPUT_FD)
pl_out_fd.mkdir(parents=True, exist_ok=True)

# ============================================================
# GENERIC SCATTER + TRENDLINE HELPER
# ============================================================

def sat_fit(x, I_sat, P_sat, c):
    return I_sat * x / (P_sat + x) + c * x


def Hill_fit(x, L, K, n):
    return L * x**n/(K**n + x**n)


def Hill_fit_from200(x, L, K, n):
    return L * (x-200)**n/(K**n + (x-200)**n)


def fit_saturation(power, counts):
    p0 = [
        counts.max(),
        np.median(power),
        0.0,
        # counts.min(),
    ]
    popt, pcov = curve_fit(sat_fit, power, counts, p0=p0)
    return popt, pcov


def _finalize_and_save(
    fig,
    ax,
    title,
    ylabel,
    out_path,
):
    ax.set_title(title)
    ax.set_xlabel("Anneal temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


def _scatter_with_trend(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None,
    label: str,
    color: str,
    trend_type: str
):
    """Plot measurements with optional error bars and a linear trendline."""

    if y_err is not None:
        ax.errorbar(
            x,
            y,
            yerr=y_err,
            fmt="o",
            markersize=6,
            capsize=4,
            alpha=0.7,
            label=label,
            color=color
        )
    else:
        ax.scatter(
            x,
            y,
            s=35,
            alpha=0.7,
            label=label,
            color=color
        )

    if len(x) >= 3:
        try:
            if trend_type == "shift" and HILL_FIT==1:    
                popt, pcov = curve_fit(Hill_fit, x, y, [1.342579, 360.5261, 5])

                L, K, n = popt

                # Smooth curve for plotting
                x_fit = np.linspace(
                    x.min(),
                    x.max(),
                    300,
                )
                y_fit = Hill_fit(x_fit, *popt)
                y_exp = Hill_fit(x, *popt)
                '''
                # Chi-squared
                chi2 = np.sum(((y - y_exp) / y_err)**2)

                # Degrees of freedom
                dof = len(y) - len(popt)

                # Reduced chi-squared
                chi2_red = chi2 / dof
                if chi2_red>0.8 and chi2_red<1.2:
                    print(60*"=")

                print("Chi² =", chi2)
                print("Reduced Chi² =", chi2_red)
                '''

            else: raise ValueError

        except (RuntimeError, ValueError):
            # Fall back to linear fit if saturation fit fails
            slope, intercept = np.polyfit(
                x,
                y,
                1,
            )

            x_fit = np.linspace(
                x.min(),
                x.max(),
                100,
            )

            y_fit = slope * x_fit + intercept

    ax.plot(
        x_fit,
        y_fit,
        linestyle="--",
        alpha=0.8,
        color=color
    )


def _scatter_with_trend_from0(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None,
    label: str,
    color: str,
    trend_type: str
):
    """Plot measurements with optional error bars and a linear trendline."""

    if y_err is not None:
        ax.errorbar(
            x,
            y,
            yerr=y_err,
            fmt="o",
            markersize=6,
            capsize=4,
            alpha=0.7,
            label=label,
            color=color
        )
    else:
        ax.scatter(
            x,
            y,
            s=35,
            alpha=0.7,
            label=label,
            color=color
        )

    if len(x) >= 3:
        try:    
            popt, pcov = curve_fit(Hill_fit, x, y, [1.342579, 360.5261, 5])

            L, K, n = popt

            # Smooth curve for plotting
            x_fit = np.linspace(
                0,
                x.max(),
                300,
            )
            y_fit = Hill_fit(x_fit, *popt)

        except (RuntimeError, ValueError) as e:
            print(f"Exception type: {type(e).__name__}")
            print(f"Error message: {e}")
            exit(1)

    ax.plot(
        x_fit,
        y_fit,
        linestyle="--",
        alpha=0.8,
        color=color
    )


def _scatter_with_trend_from200(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None,
    label: str,
    color: str,
    trend_type: str
):
    """Plot measurements with optional error bars and a linear trendline."""

    if y_err is not None:
        ax.errorbar(
            x,
            y,
            yerr=y_err,
            fmt="o",
            markersize=6,
            capsize=4,
            alpha=0.7,
            label=label,
            color=color
        )
    else:
        ax.scatter(
            x,
            y,
            s=35,
            alpha=0.7,
            label=label,
            color=color
        )

    if len(x) >= 3:
        try:    
            popt, pcov = curve_fit(Hill_fit_from200, x, y, [1.342579, 360.5261, 5])

            L, K, n = popt
            print("Fitted parameters:")
            for i, value in enumerate(popt):
                print(f"  Parameter {i}: {value:.6g}")

            # Smooth curve for plotting
            x_fit = np.linspace(
                200,
                x.max(),
                300,
            )
            y_fit = Hill_fit_from200(x_fit, *popt)

            ax.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    alpha=0.8,
                    color=color
                )
            
        except (RuntimeError, ValueError):
            print(ValueError)

def _plot_value_vs_temperature_by_area(
    data: pd.DataFrame,
    value_col: str,
    title_prefix: str,
    ylabel: str,
    filename_prefix: str,
    output_dir: Path,
    empty_msg: str,
    error_bar_label: str | None = None,
) -> None:

    if data.empty:
        print(empty_msg)
        return

    for area, area_df in data.groupby("area"):

        fig, ax = plt.subplots(figsize=(10, 7))

        # Get Matplotlib's default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (emitter_id, emitter_df) in enumerate(
            area_df.groupby("emitter")
        ):
            color = colors[i % len(colors)]

            emitter_df = emitter_df.sort_values(
                "anneal_temp_C"
            )

            x = emitter_df["anneal_temp_C"].to_numpy()
            y = emitter_df[value_col].to_numpy()

            if error_bar_label is not None:
                y_err = emitter_df[
                    error_bar_label
                ].to_numpy()
            else:
                y_err = None

            _scatter_with_trend(
                ax,
                x,
                y,
                y_err,
                label=f"Emitter {emitter_id}",
                color=color,
                trend_type="shift"
            )

        _finalize_and_save(
            fig,
            ax,
            title=f"{title_prefix} — area {area}",
            ylabel=ylabel,
            out_path=(
                output_dir
                / f"{filename_prefix}_area{area}.png"
            ),
        )

# ============================================================
# PL5: PEAK POSITION VS TEMPERATURE
# ============================================================

def plot_PL5_peak_vs_temperature(
    df: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:
    """
    PL5 has a single dip per measurement. Plot x0 vs anneal_temp_C
    for each emitter, with a linear trendline.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl5 = df[df["emitter_type"] == "PL5"]

    _plot_value_vs_temperature_by_area(
        pl5,
        value_col="x0",
        title_prefix="PL5 — peak position vs temperature",
        ylabel="Peak position (x0)",
        filename_prefix="PL5_peak_position_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL5 rows found — skipping PL5 plot.",
        error_bar_label="x0_u"
    )


def plot_PL5_peakshift_vs_temperature(
    df: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:
    """
    PL5 has a single dip per measurement. Plot x0 vs anneal_temp_C
    for each emitter, with a linear trendline.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl5 = (
        df[df["emitter_type"] == "PL5"]
        .sort_values(["area", "emitter", "anneal_temp_C"])
        .copy()
    )

    if pl5.empty:
        print("No PL5 rows found — skipping PL5 plot.")
        return

    # Shift x0 relative to the first temperature of each emitter
    pl5["x0"] = (
        pl5["x0"]
        - pl5.groupby(["area", "emitter"])["x0"].transform("first")
    )

    pl5["x0_u"] = np.sqrt(
            pl5["x0_u"]**2
            + pl5.groupby(["area", "emitter"])["x0"].transform("first")**2
        )

    pl5 = pl5[pl5.groupby(["area", "emitter"]).cumcount() > 0].copy()

    _plot_value_vs_temperature_by_area(
        pl5,
        value_col="x0",
        title_prefix="PL5 — peak SHIFT position vs temperature",
        ylabel="Peak SHIFT position (x0)",
        filename_prefix="PL5_shift_peak_position_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL5 rows found — skipping PL5 plot.",
        error_bar_label="x0_u"
    )

# ============================================================
# PL6: PAIR UP THE TWO DIPS PER MEASUREMENT
# ============================================================

def _pair_PL6_dips(df: pd.DataFrame) -> pd.DataFrame:
    """
    PL6 usually has 2 rows per measurement (one per dip), sharing the
    same measurement_folder. Collapse each pair into a single row with
    mean_x0 and diff_x0 (splitting between the two dips).

    If only 1 dip was fit for a given measurement, that single x0 is
    used as mean_x0 and diff_x0 is left as NaN (no splitting can be
    computed from one peak).
    """

    pl6 = df[df["emitter_type"] == "PL6"]

    if pl6.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["area", "anneal_temp_C", "emitter", "measurement_folder"]

    for key, g in pl6.groupby(group_cols):

        if len(g) > 2:
            print(
                f"WARNING: expected at most 2 dips for {key}, "
                f"found {len(g)} — skipping"
            )
            continue

        g = g.sort_values("dip")

        x0_vals = g["x0"].to_numpy()
        x0_u_vals = g["x0_u"].to_numpy()

        if len(x0_vals) == 2:
            mean_x0 = x0_vals.mean()
            diff_x0 = abs(x0_vals[1] - x0_vals[0])

        else:
            # Only one dip available — plot that single peak as the
            # "mean", but there's no splitting to compute.
            print(
                f"NOTE: only 1 dip for {key}, "
                f"using it as a single peak (no splitting)"
            )
            mean_x0 = x0_vals[0]
            diff_x0 = np.nan

        rows.append({
            "area": key[0],
            "anneal_temp_C": key[1],
            "emitter": key[2],
            "measurement_folder": key[3],
            "mean_x0": mean_x0,
            "mean_x0_u":x0_u_vals[0],
            "diff_x0": diff_x0,
        })

    return pd.DataFrame(rows)


# ============================================================
# PL6: MEAN PEAK POSITION VS TEMPERATURE
# ============================================================

def plot_PL6_mean_vs_temperature(
    pl6_pairs: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_value_vs_temperature_by_area(
        pl6_pairs,
        value_col="mean_x0",
        title_prefix="PL6 — mean peak position vs temperature",
        ylabel="Mean peak position (x0)",
        filename_prefix="PL6_mean_peak_position_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL6 pairs found — skipping PL6 mean plot.",
        error_bar_label="mean_x0_u",
    )


def plot_PL6_peakshift_vs_temperature(
    pl6: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:
    """
    PL6 has a single dip per measurement. Plot x0 vs anneal_temp_C
    for each emitter, with a linear trendline.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # pl6 = (
    #     df[df["emitter_type"] == "PL6"]
    #     .sort_values(["area", "emitter", "anneal_temp_C"])
    #     .copy()
    # )

    # if pl6.empty:
    #     print("No PL6 rows found — skipping PL6 plot.")
    #     return

    # Shift x0 relative to the first temperature of each emitter
    pl6["mean_x0"] = (
        pl6["mean_x0"]
        - pl6.groupby(["area", "emitter"])["mean_x0"].transform("first")
    )

    pl6["mean_x0_u"] = np.sqrt(
            pl6["mean_x0_u"]**2
            + pl6.groupby(["area", "emitter"])["mean_x0_u"].transform("first")**2
        )
    
    # pl6 = pl6.groupby(["area", "emitter"], group_keys=False).apply(lambda g: g.iloc[1:]).reset_index(drop=True)
    pl6 = pl6[pl6.groupby(["area", "emitter"]).cumcount() > 0].copy()

    _plot_value_vs_temperature_by_area(
        pl6,
        value_col="mean_x0",
        title_prefix="PL6 — peak SHIFT position vs temperature",
        ylabel="Peak SHIFT position (x0)",
        filename_prefix="PL6_shift_peak_position_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL6 rows found — skipping PL6 plot.",
        error_bar_label="mean_x0_u"
    )



# ============================================================
# PL6: PEAK SPLITTING VS TEMPERATURE
# ============================================================

def plot_PL6_splitting_vs_temperature(
    pl6_pairs: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Splitting only makes sense for measurements that had 2 dips
    pl6_pairs = pl6_pairs.dropna(subset=["diff_x0"])

    _plot_value_vs_temperature_by_area(
        pl6_pairs,
        value_col="diff_x0",
        title_prefix="PL6 — peak splitting vs temperature",
        ylabel="Peak splitting |x0_2 - x0_1|",
        filename_prefix="PL6_splitting_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL6 pairs found — skipping PL6 splitting plot.",
    )


def plot_fwhm_vs_temperature(
    df: pd.DataFrame,
    output_dir: str | Path = "plots_by_type",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl6 = df[df["emitter_type"] == "PL6"]

    _plot_value_vs_temperature_by_area(
        pl6,
        value_col="fwhm",
        title_prefix="PL6 — FWHM vs temperature",
        ylabel="FWHM",
        filename_prefix="PL6_FWHM_vs_temperature",
        output_dir=output_dir,
        empty_msg="No PL5 rows found — skipping FWHM plot.",
    )

    
# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv("all_PL_peaks_all_samples.csv")

    # plot_PL5_peak_vs_temperature(df)
    plot_PL5_peakshift_vs_temperature(df)
    # plot_fwhm_vs_temperature(df)
    pl6_pairs = _pair_PL6_dips(df)

    # plot_PL6_mean_vs_temperature(pl6_pairs)
    plot_PL6_peakshift_vs_temperature(pl6_pairs)

    # plot_PL6_splitting_vs_temperature(pl6_pairs)