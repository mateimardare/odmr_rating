from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from parameterFittingV0 import paramFITclass
import re

BASE_PATH = Path("./measurements/")  
TEMPLATE = "T220-2_Oimplant_1E11_annealed_area{sample}"
MEAS_TYPE = "cwODMR"
RESULTS_PATH = Path("./results_odmr/")

MAP_PATH = Path("sat_g2_odmr_maps")         # folder holding the emitter relabeling maps
OUT_PATH = Path("comparison_plots")         # where comparison plots get saved
SHIFT_CSV_PATH = Path("peak_shift_450.csv")
SAMPLE_IDS = ["45", "55"]                   # the samples/areas you have

BEFORE_TEMPLATE = "T220-2_Oimplant_1E11_annealed_area{sample}"
AFTER_TEMPLATE = "T220-2_Oimplant_1E11_annealed_area{sample}_450degC"

MEAS_TYPE = "cwODMR"                        # which measurement type to fit/plot

# map csv has no header; column order is [after_emitter, before_emitter] -- flip if needed
MAP_COLUMNS = ["after_emitter", "before_emitter"]
PEAK_MATCH_TOLERANCE = 3


SAMPLE_DIR_PATTERN = re.compile(r"area(\d+)_(\d+)degC")

X_KEY_CANDIDATES = ["fscanRange", "frequency", "freq", "x", "Frequency"]
Y_KEY_CANDIDATES = ["CountData", "counts", "signal", "y", "Counts"]

def load_map(sample_id: str) -> pd.DataFrame:
    """Load the before<->after emitter relabeling map for one sample.

    Expects a file matching '*_<sample_id>.csv' inside MAP_PATH.
    """
    matches = sorted(MAP_PATH.glob(f"*_{sample_id}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No map csv found for sample '{sample_id}' in {MAP_PATH}")
    if len(matches) > 1:
        print(f"WARNING: multiple map files match sample '{sample_id}', using {matches[0].name}")
    df = pd.read_csv(matches[0], header=None, names=MAP_COLUMNS)
    df["after_emitter"] = df["after_emitter"].astype(str)
    df["before_emitter"] = df["before_emitter"].astype(str)
    return df


def find_all_sample_dirs(base_path: Path = BASE_PATH) -> list[dict]:
    """Find every top-level measurement folder under base_path and extract
    its area number and anneal temperature."""
    results = []
    for p in sorted(base_path.iterdir()):
        if not p.is_dir():
            continue
        m = SAMPLE_DIR_PATTERN.search(p.name)
        if not m:
            print(f"WARNING: folder name doesn't match area/temp pattern, skipping: {p.name}")
            continue
        results.append({
            "area": m.group(1),
            "anneal_temp_C": int(m.group(2)),
            "path": p,
        })
    return results


def sample_dir(sample_id: str) -> Path:
    return BASE_PATH / TEMPLATE.format(sample=sample_id)


def find_emitter_folder(base_dir: Path, emitter: str, meas_type: str = MEAS_TYPE) -> Path:
    """Find the pillar measurement folder for a given emitter label inside base_dir."""
    candidates = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            info = parse_sample_name(p)
        except ValueError:
            continue
        if info["emitter"] != str(emitter):
            continue
        if meas_type and meas_type not in info["types"]:
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No folder found in {base_dir} for emitter '{emitter}', type '{meas_type}'"
        )
    if len(candidates) > 1:
        print(f"WARNING: multiple folders match emitter '{emitter}' in {base_dir}: {candidates}")
    return candidates[0]


def parse_folder_name(name)-> dict:
    if isinstance(name, Path):
        name = name.stem

    parts = name.split("_")
    sample = "_".join(parts[0:3])
    area = parts[4]
    temp =parts[5]

    return {"sample": sample, "area": area, "temperature": temp}


def parse_sample_name(name) -> dict:
    """Parse a pillar-measurement folder name (or Path/stem) into its parts.

    Expected pattern: YYYY_MM_DD_HH_MM_SS_<number>_<type1>_<type2>_..._<emitter>
    The emitter (last underscore-separated token) is returned as a string.
    """
    if isinstance(name, Path):
        name = name.stem if name.suffix else name.name

    parts = name.split("_")
    if len(parts) < 8:
        raise ValueError(
            f"Sample name '{name}' does not match the expected "
            "YYYY_MM_DD_HH_MM_SS_number_type1_type2_..._emitter pattern"
        )

    date = "_".join(parts[0:3])
    time = "_".join(parts[3:6])
    number = parts[6]
    emitter = parts[-1]
    types = parts[7:-1]

    return {
        "sample_name": name,
        "date": date,
        "time": time,
        "number": number,
        "types": types,
        "emitter": emitter,
    }


def _pick_data_file(path: Path, meas_type: str) -> Path:
    h5_files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5/.hdf5 file found in {path}"
        )

    # Prefer the actual measurement file
    candidates = [
        p for p in h5_files
        if meas_type.lower() in p.name.lower()
    ]

    if candidates:
        return sorted(candidates)[-1]

    # Fall back to files that aren't obvious utility files
    candidates = [
        p for p in h5_files
        if not any(
            word in p.name.lower()
            for word in ("optimizer", "confocal")
        )
    ]

    if candidates:
        return sorted(candidates)[-1]

    raise FileNotFoundError(
        f"No suitable {meas_type} data file found in {path}"
    )


def _first_matching_key(f: h5py.File, candidates: list[str]) -> str:
    for key in candidates:
        if key in f:
            return key
    # fall back: search one level into groups too
    for key in candidates:
        for group_name in f:
            if isinstance(f[group_name], h5py.Group) and key in f[group_name]:
                return f"{group_name}/{key}"
    raise KeyError(candidates)


def load_odmr(path: Path, meas_type: str = MEAS_TYPE) -> dict:
    """Load ODMR frequency-scan data (x, y) from the right .h5/.hdf5 file
    inside `path` (a folder may contain more than one .hdf5 file)."""
    h5_file = _pick_data_file(path, meas_type)
    with h5py.File(h5_file, "r") as f:
        try:
            x_key = _first_matching_key(f, X_KEY_CANDIDATES)
            y_key = _first_matching_key(f, Y_KEY_CANDIDATES)
        except KeyError as missing:
            available = []
            f.visititems(lambda n, o: available.append(n) if isinstance(o, h5py.Dataset) else None)
            raise KeyError(
                f"None of {missing.args[0]} found in {h5_file}. "
                f"Available datasets: {available}. "
                "Run inspect_h5(path) on this folder and add the real key "
                "name to X_KEY_CANDIDATES/Y_KEY_CANDIDATES at the top of the file."
            )
        return {
            "x": np.asarray(f[x_key][:]),
            "y": np.asarray(f[y_key][:]),
        }


def fit_odmr(x: np.ndarray, y: np.ndarray):
    """Fit an ODMR dip spectrum (handles 1 or more dips automatically)."""
    fitter = paramFITclass()
    fitres, xplot, yplot, _yplot_guess = fitter.fitLorentzian(x, y, printresults=False)
    return fitres, xplot, yplot


def extract_peak_params(fitres: dict) -> pd.DataFrame:
    """Turn a fitres dict (one or more dips) into a per-dip DataFrame with
    position, width (FWHM), amplitude, and contrast."""
    x0 = np.atleast_1d(fitres["x0"])
    x0_u = np.atleast_1d(fitres["x0_u"])

    A = np.atleast_1d(fitres["A"])
    A_u = np.atleast_1d(fitres["A_u"])

    gamma = np.atleast_1d(fitres["gamma"])
    gamma_u = np.atleast_1d(fitres["gamma_u"])
    fwhm = 2 * gamma
    fwhm_u = 2 * gamma_u

    B = np.atleast_1d(fitres["B"])
    n_dips = len(x0)
    # B may be a single global baseline shared across dips, or one per dip
    B_full = B if len(B) == n_dips else np.full(n_dips, B[0])

    contrast = A / B_full
    contrast_u = A_u / B_full  # assumes negligible uncertainty on B

    return pd.DataFrame({
        "dip": np.arange(n_dips),
        "x0": x0,
        "x0_u": x0_u,
        "fwhm": fwhm,
        "fwhm_u": fwhm_u,
        "amplitude": A,
        "amplitude_u": A_u,
        "baseline": B_full,
        "contrast": contrast,
        "contrast_u": contrast_u,
    })


def export_all_peaks_in_folder(
    folder: str | Path,
    meas_type: str = MEAS_TYPE,
    csv_path: Path = Path("all_peaks.csv"),
):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    all_dfs = []

    for measurement_folder in sorted(folder.iterdir()):
        if not measurement_folder.is_dir():
            continue
        try:
            info = parse_sample_name(measurement_folder)
        except ValueError:
            continue
        if meas_type not in info["types"]:
            continue

        try:
            data = load_odmr(measurement_folder, meas_type)
            fitres, xplot, yplot = fit_odmr(data["x"], data["y"])

            df = extract_peak_params(fitres)
            df.insert(0, "measurement_folder", measurement_folder.name)
            df.insert(1, "date", info["date"])
            df.insert(2, "time", info["time"])
            df.insert(3, "measurement_number", info["number"])
            df.insert(4, "emitter", info["emitter"])
            df.insert(5, "measurement_type", meas_type)

            all_dfs.append(df)
            print(f"OK: {measurement_folder.name} → {len(df)} peak(s)")

        except Exception as e:
            print(f"WARNING: {measurement_folder.name} failed: {e}")

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    result.to_csv(csv_path, index=False)
    print(f"\nWrote {len(result)} peaks to:\n{csv_path}")

    return result


def export_odmr_shift(sample_id: str, emitter) -> pd.DataFrame:
    emitter = str(emitter)
    folder = find_emitter_folder(sample_dir(sample_id), emitter)
    data_odmr = load_odmr(folder)
    fitres, xplot, yplot = fit_odmr(data_odmr["x"], data_odmr["y"])

    df = extract_peak_params(fitres)
    df.insert(0, "emitter", emitter)
    df.insert(0, "sample_id", sample_id)

    # optional: raw-data sanity checks, unrelated to the fit params themselves
    n_data = len(data_odmr["x"])
    n_edge = n_data // 10
    n_bgd = 2 * n_edge
    df["background_raw"] = (np.sum(data_odmr["y"][:n_edge]) + np.sum(data_odmr["y"][-n_edge:])) / n_bgd
    df["min_odmr_raw"] = np.min(data_odmr["y"])
    df["min_fit"] = np.min(yplot)

    return df


def export_all_peaks_all_samples(
    base_path: Path = BASE_PATH,
    meas_type: str = MEAS_TYPE,
    csv_path: Path = Path("all_peaks_all_samples.csv"),
) -> pd.DataFrame:
    sample_dirs = find_all_sample_dirs(base_path)
    if not sample_dirs:
        raise FileNotFoundError(f"No folders matching area/temp pattern found in {base_path}")

    all_dfs = []
    for entry in sample_dirs:
        area, temp, folder = entry["area"], entry["anneal_temp_C"], entry["path"]
        print(f"\n=== Area {area}, {temp}degC ({folder.name}) ===")

        export_path = RESULTS_PATH / f"all_peaks_area{area}_{temp}degC.csv"
        df = export_all_peaks_in_folder(
            folder,
            meas_type=meas_type,
            csv_path=export_path,
        )
        if not df.empty:
            df.insert(0, "anneal_temp_C", temp)
            df.insert(0, "area", area)
            all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    result.to_csv(csv_path, index=False)
    print(f"\n=== Wrote {len(result)} total peaks across {len(sample_dirs)} folder(s) to:\n{csv_path}")
    return result


def plot_emitter_temperatures_ODMR(
    emitter,
    meas_folder: str | Path,
):
    """
    Plot the ODMR spectrum of one emitter for every temperature.
    """
    meas_folder = Path(meas_folder)

    fig, ax = plt.subplots(figsize=(7, 5))

    for temp_folder in sorted(meas_folder.iterdir()):
        if not temp_folder.is_dir():
            continue

        try:
            info = parse_folder_name(temp_folder)
            temperature = info["temperature"]
            emitter_folder = find_emitter_folder(
                temp_folder,
                emitter,
                MEAS_TYPE,
            )

            data = load_odmr(
                emitter_folder,
                MEAS_TYPE
            )

            fitres, xplot, yplot = fit_odmr(data["x"], data["y"])

            baseline = fitres["B"]

            x = data["x"]
            y = data["y"] / baseline


            ax.plot(
                x,
                y,
                "o-",
                ms=4,
                alpha=0.5,
                label=temperature,
            )

        except Exception as e:
            print(
                f"WARNING: emitter {emitter}, "
                f"{temp_folder.name} failed: {e}"
            )

    ax.set(
        xlabel="Frequency",
        ylabel="Counts",
        title=f"Emitter {emitter}",
    )

    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_all_temperatures_ODMR(
    meas_folder: str | Path,
    start_folder: str | Path,
):
    """
    Plot the ODMR spectra across all temperatures for every emitter.
    """
    meas_folder = Path(meas_folder)
    start_folder = Path(start_folder)

    emitters = []

    for p in sorted(start_folder.iterdir()):
        if not p.is_dir():
            continue

        try:
            info = parse_sample_name(p)
            
            if MEAS_TYPE in info["types"]:
                emitters.append(info["emitter"])

        except ValueError:
            continue

    out_dir = Path("all_temp_ODMR")
    out_dir.mkdir(parents=True, exist_ok=True)

    for emitter in emitters:
        try:
            fig, ax = plot_emitter_temperatures_ODMR(
                emitter,
                meas_folder,
            )

            fig.savefig(
                out_dir / f"emitter{emitter}.png",
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

        except Exception as e:
            print(
                f"WARNING: emitter {emitter} failed: {e}"
            )


def plot_measurement(
    folder: str | Path,
    meas_type: str = MEAS_TYPE,
    *,
    info: dict | None = None,
    save: bool = True,
    out_dir: str | Path | None = None,
):
    folder = Path(folder)
    info = info or parse_sample_name(folder)
    data = load_odmr(folder, meas_type)
    fit, xplot, yplot = fit_odmr(data["x"], data["y"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(data["x"], data["y"], "o-", ms=4, alpha=.5, label="measurement")
    ax.plot(xplot, yplot, "-", label="Lorentzian fit")
    ax.set(
        xlabel="Frequency",
        ylabel="Counts",
        title=f"{info['date']} {info['time']} — emitter {info['emitter']}",
    )
    ax.legend()
    fig.tight_layout()

    if save:
        out_dir = Path(out_dir or OUT_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / f"{folder.name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return {
        "folder": folder,
        "info": info,
        "data": data,
        "fit": fit,
        "figure": fig,
    }


def plot_measurements_in_folder(
    folder: str | Path,
    meas_type: str = MEAS_TYPE,
    *,
    measurement_range="all",
    save: bool = True,
):
    folder = Path(folder)

    measurements = []
    for p in sorted(folder.iterdir()):
        if not p.is_dir():
            continue
        try:
            info = parse_sample_name(p)
            if not meas_type or meas_type in info["types"]:
                measurements.append((p, info))
        except ValueError:
            pass

    if not measurements:
        raise FileNotFoundError(
            f"No '{meas_type}' measurements found in {folder}"
        )

    # Select measurements
    if measurement_range != "all":
        measurements = measurements[measurement_range]

    out_dir = OUT_PATH / folder.name / meas_type if save else None

    results = {}
    for p, info in measurements:
        try:
            results[p.name] = plot_measurement(
                p,
                meas_type,
                info=info,
                save=save,
                out_dir=out_dir,
            )
            print(f"OK: {p.name}")
        except Exception as e:
            print(f"WARNING: {p.name}: {e}")

    return results


if __name__ == "__main__":
    plot_all_temperatures_ODMR(Path("measurements/area45"), Path("measurements/area45/T220-2_Oimplant_1E11_annealed_area45_0degC"))

