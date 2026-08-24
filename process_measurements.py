from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no display needed, we're saving files
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit
import numpy as np

from sample_utils import parse_sample_name

# measurements/<batch_name>/<sample_dirs...>
# each batch (e.g. "T220-20_Oimplant_1E11_annealed_area56") gets its own
# mirrored output folder: processed/<batch_name>/sat_g2_odmr/<sample>.png
MEASUREMENTS_ROOT = Path("measurements")
PROCESSED_ROOT = Path("processed")


def load_sat(path):
    with h5py.File(path, "r") as f:
        return {
            "x": f["power_range"][:],
            "y": f["CountData"][:],
        }


def load_g2(path):
    with h5py.File(path, "r") as f:
        return {
            "x": f["Xdata"][:],
            "y": f["Ydata"][:],
        }


def load_odmr(path):
    with h5py.File(path, "r") as f:
        return {
            "x": f["fscanRange"][:],      # replace if another dataset is frequency
            "y": f["CountData"][:],
        }


def sat_fit(x, I_sat, P_sat, c):
    return I_sat * x / (P_sat + x) + c * x


def fit_saturation(power, counts):
    p0 = [
        counts.max(),
        np.median(power),
        0.0,
        # counts.min(),
    ]
    popt, pcov = curve_fit(sat_fit, power, counts, p0=p0)
    return popt, pcov


#Lorentz
def peak(f, A, gamma, f_peak):
    return np.divide(A*np.power((gamma/2),2),np.power(np.subtract(f,f_peak),2)+np.power((gamma/2),2))


def odmr_dips_fit(f, *fit_params_guess):
    # the number of fit params determines the number of dips to be fitted
    # fit params is an array of the form
    # [A_1, A_2, ..., gamma_1, gamma_2, ..., f_peak_1, f_peak_2, ...]
    # where A is a guess on the peak amplitude,
    # Gamma is a guess on the peak FWHM
    # f_peak is a guess on the frequency location of the peak

    peak_num = int((len(fit_params_guess)-1)/3)

    A = fit_params_guess[0:peak_num]

    f_peak = fit_params_guess[peak_num:2*peak_num]

    gamma = fit_params_guess[2*peak_num:3*peak_num]

    y_0 = fit_params_guess[3*peak_num]

    fit_curve = np.zeros(len(f))+y_0
    for i in range(0, peak_num):
        fit_curve = fit_curve - peak(f, A[i], gamma[i], f_peak[i])

    return fit_curve


def plot_measurements(name, sat, g2, odmr, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    if sat is not None:
        try:
            popt, pcov = fit_saturation(sat["x"], sat["y"])
            I_sat, P_sat, c = popt

            ax[0].scatter(sat["x"], sat["y"], label="Data")

            x_fit = np.linspace(sat["x"].min(), sat["x"].max(), 500)

            ax[0].plot(x_fit, sat_fit(x_fit, *popt), label="Fit")
            ax[0].plot(x_fit, c*x_fit + I_sat, "--", label="Linear")
            ax[0].plot(x_fit, sat_fit(x_fit, *popt) - (c*x_fit))
            ax[0].plot(x_fit, c*x_fit, "--", label="Linear")

            ax[0].set_title("Saturation")
            ax[0].set_xlabel("Power (mW)")
            ax[0].set_ylabel("Counts")
            ax[0].legend()
        except RuntimeError:
            # fit failed to converge - still show raw data
            ax[0].scatter(sat["x"], sat["y"], label="Data")

        ax[0].set_title("Saturation")
        ax[0].set_xlabel("Power (mW)")
        ax[0].set_ylabel("Counts")
        ax[0].legend()

    if g2 is not None:
        ax[1].plot(g2["x"], g2["y"])
        ax[1].set_title(r"$g^{(2)}$")
        ax[1].set_xlabel("Delay (ns)")
        ax[1].set_ylabel("Coincidences")

    if odmr is not None:
        ax[2].plot(odmr["x"], odmr["y"])
        ax[2].set_title("CW ODMR")
        ax[2].set_xlabel("Frequency")
        ax[2].set_ylabel("Counts")

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_batch(batch_dir: Path) -> int:
    """Process every sample subfolder inside one batch folder.

    e.g. measurements/T220-20_Oimplant_1E11_annealed_area56/<sample_dirs>
      -> processed/T220-20_Oimplant_1E11_annealed_area56/sat_g2_odmr/<sample>.png
    """
    out_dir = PROCESSED_ROOT / batch_dir.name / "sat_g2_odmr"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0

    for sample in sorted(batch_dir.iterdir()):
        if not sample.is_dir():
            continue

        try:
            parse_sample_name(sample.name)
        except ValueError as e:
            print(f"  Skipping '{sample.name}': {e}")
            continue

        sat = g2 = odmr = None

        sat_file = next(sample.glob("*satMeas*.hdf5"), None)
        if sat_file:
            sat = load_sat(sat_file)

        g2_file = next(sample.glob("*g2v0*.hdf5"), None)
        if g2_file:
            g2 = load_g2(g2_file)

        odmr_file = next(sample.glob("*cwODMR*.hdf5"), None)
        if odmr_file:
            odmr = load_odmr(odmr_file)

        if sat is None and g2 is None and odmr is None:
            print(f"  Skipping '{sample.name}': no matching hdf5 files found")
            continue

        save_path = out_dir / f"{sample.name}.png"
        plot_measurements(sample.name, sat, g2, odmr, save_path)
        n_saved += 1
        print(f"  Saved {save_path}")

    return n_saved


def main():
    if not MEASUREMENTS_ROOT.exists():
        print(f"'{MEASUREMENTS_ROOT}/' not found.")
        return

    total_saved = 0
    total_batches = 0

    for batch_dir in sorted(MEASUREMENTS_ROOT.iterdir()):
        if not batch_dir.is_dir():
            continue

        
        print(f"Batch: {batch_dir.name}")
        total_saved += process_batch(batch_dir)
        total_batches += 1

    print(f"\nDone. {total_saved} figure(s) saved across {total_batches} batch(es) under '{PROCESSED_ROOT}/'.")


if __name__ == "__main__":
    main()