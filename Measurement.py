"""
Measurement.py
===============

An extensible framework for loading, fitting, and plotting emitter
measurements (saturation curves, g(2) autocorrelation, CW-ODMR, ...) stored
as HDF5 files inside per-sample measurement folders.

Two things are pluggable, independently of each other:

1. MEASUREMENT TYPES -- what kind of data is this, how do I load/plot it.
   Subclass `Measurement` and register it with `@Measurement.register(...)`.

2. FIT ROUTINES -- what model do I fit the data to.
   Write a plain function and register it with `@fit_registry.register(...)`.

A given measurement type can have several fit routines available for it
(e.g. try an automatic multi-peak Lorentzian fit, and fall back to a
manual-guess version if peak-finding fails), and a brand new fit routine
can be added without touching any Measurement subclass at all. Likewise a
brand new measurement type can be added without touching any fit routine.

Quick start
-----------
    from pathlib import Path
    from Measurement import load_sample

    # Load every registered measurement type found in one sample folder:
    sample = load_sample(Path("measurements/batch1/sample01"))
    sample["odmr"].fit()          # uses that type's default fit routine
    sample["odmr"].plot()
    sample.plot_grid(save_path=Path("out/sample01.png"))   # all panels at once

    # Or work with a single file/measurement directly:
    from Measurement import ODMRMeasurement
    m = ODMRMeasurement.load(Path(".../..._cwODMR_....hdf5"))
    m.fit(routine="lorentzian_dips", n_peaks=2)
    fig, ax = m.plot()
    print(m.peak_table())

Adding a new measurement type
------------------------------
    @Measurement.register("t1", file_glob="*T1Meas*.hdf5")
    class T1Measurement(Measurement):
        x_dataset = "delay"          # HDF5 dataset name(s) for x/y -- a
        y_dataset = "CountData"      # single name, or a list of candidate
        x_label = "Delay (us)"       # names to try in order (first found
        y_label = "Counts"           # wins), handy when different
        default_fit_routine = "exp_decay"   # acquisition scripts saved
                                             # slightly different key names.

That's it. `load_sample`, batch-processing scripts, glob-based discovery,
etc. all pick it up automatically because it's registered -- nothing else
in this file needs to change.

Adding a new fit routine
-------------------------
    @fit_registry.register("exp_decay")
    def fit_exp_decay(x, y, *, p0=None, **kwargs):
        def model(x, A, tau, c):
            return A * np.exp(-x / tau) + c
        p0 = p0 or [y.max() - y.min(), np.ptp(x) / 3, y.min()]
        popt, pcov = curve_fit(model, x, y, p0=p0)
        return FitResult(model=model, popt=popt, pcov=pcov,
                          param_names=["A", "tau", "c"])

Any Measurement (not just the new T1 type) can now call
`.fit(routine="exp_decay")` if the model happens to make sense for it too.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Optional, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    import pandas as pd
except ImportError:  # pandas is only needed for ODMRMeasurement.peak_table()
    pd = None


# ===========================================================================
# Fit routines: a small registry, deliberately decoupled from Measurement
# types. Any routine can be used by any measurement type that hands it
# compatible x/y data -- the routine doesn't know or care what physical
# quantity it's fitting.
# ===========================================================================

@dataclass
class FitResult:
    """Generic container for a fit result, independent of the model used."""

    model: Callable
    popt: np.ndarray
    pcov: np.ndarray
    param_names: list[str]
    routine_name: str = ""
    extra: dict = field(default_factory=dict)  # model-specific extras, e.g. n_peaks

    @property
    def perr(self) -> np.ndarray:
        """1-sigma parameter uncertainties from the covariance matrix."""
        return np.sqrt(np.diag(self.pcov))

    def params(self) -> dict:
        """{name: (value, uncertainty)} for every fit parameter."""
        return {n: (v, e) for n, v, e in zip(self.param_names, self.popt, self.perr)}

    def curve(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted model on new x values."""
        return self.model(x, *self.popt)

    def __repr__(self):
        params_str = ", ".join(f"{n}={v:.4g}\u00b1{e:.2g}" for n, (v, e) in self.params().items())
        return f"FitResult({self.routine_name}: {params_str})"


class FitRegistry:
    """Maps a routine name -> a `fit(x, y, **kwargs) -> FitResult` function."""

    def __init__(self):
        self._routines: dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable) -> Callable:
            if name in self._routines:
                warnings.warn(f"Overwriting existing fit routine '{name}'")
            self._routines[name] = func
            return func
        return decorator

    def get(self, name: str) -> Callable:
        try:
            return self._routines[name]
        except KeyError:
            raise KeyError(
                f"No fit routine registered as '{name}'. Available: {sorted(self._routines)}"
            )

    def __contains__(self, name):
        return name in self._routines

    def __iter__(self):
        return iter(self._routines)


fit_registry = FitRegistry()


# ===========================================================================
# Measurement types: a small registry mapping a short type name (e.g. "odmr")
# to the Measurement subclass responsible for loading/plotting it.
# ===========================================================================

class MeasurementRegistry:
    def __init__(self):
        self._types: dict[str, type["Measurement"]] = {}

    def register(self, type_name: str, cls: type["Measurement"]):
        self._types[type_name] = cls

    def get(self, type_name: str) -> type["Measurement"]:
        try:
            return self._types[type_name]
        except KeyError:
            raise KeyError(
                f"No measurement type registered as '{type_name}'. "
                f"Available: {sorted(self._types)}"
            )

    def items(self):
        return self._types.items()

    def __iter__(self):
        return iter(self._types)


measurement_registry = MeasurementRegistry()


class Measurement:
    """
    Base class for one measurement (one x/y trace loaded from one HDF5 file).

    Subclass and decorate with `@Measurement.register(type_name, file_glob=...)`
    to add a new measurement type. Required class attributes on the subclass:

        x_dataset / y_dataset : str | list[str]
            HDF5 dataset name holding x/y data. Can be a list of candidate
            names (first one found in the file wins) -- useful when
            different acquisition-software versions used different key
            names for the same thing.

    Optional class attributes:

        x_label, y_label     : str          -- axis labels for plotting
        default_fit_routine  : str | None   -- name in `fit_registry` used
                                                by `.fit()` when no routine
                                                is passed explicitly
    """

    # subclasses fill these in (either directly, or via `.register(...)`)
    type_name: ClassVar[str] = ""
    file_glob: ClassVar[str] = ""
    x_dataset: ClassVar[Union[str, list[str]]] = "x"
    y_dataset: ClassVar[Union[str, list[str]]] = "y"
    x_label: ClassVar[str] = "x"
    y_label: ClassVar[str] = "y"
    default_fit_routine: ClassVar[Optional[str]] = None

    def __init__(self, x: np.ndarray, y: np.ndarray, *, source: Optional[Path] = None, name: str = ""):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.source = source
        self.name = name or (source.parent.name if source else self.type_name)
        self.fit_result: Optional[FitResult] = None

    # -- registration --------------------------------------------------
    @classmethod
    def register(cls, type_name: str, *, file_glob: str):
        """Class decorator: register a Measurement subclass under `type_name`.

        `file_glob` is how this type's file is recognised inside a sample
        folder, e.g. "*cwODMR*.hdf5" -- same convention as the glob calls
        already used in process_measurements.py.
        """
        def decorator(subclass):
            subclass.type_name = type_name
            subclass.file_glob = file_glob
            measurement_registry.register(type_name, subclass)
            return subclass
        return decorator

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _read_dataset(f: h5py.File, key_or_candidates: Union[str, list[str]]) -> np.ndarray:
        candidates = [key_or_candidates] if isinstance(key_or_candidates, str) else list(key_or_candidates)
        for key in candidates:
            if key in f:
                return np.asarray(f[key][:])
            for group_name in f:
                if isinstance(f[group_name], h5py.Group) and key in f[group_name]:
                    return np.asarray(f[group_name][key][:])
        available = []
        f.visititems(lambda n, o: available.append(n) if isinstance(o, h5py.Dataset) else None)
        raise KeyError(
            f"None of {candidates} found in {getattr(f, 'filename', '<file>')}. "
            f"Available datasets: {available}"
        )

    @classmethod
    def load(cls, path: Path) -> "Measurement":
        """Load this measurement type's x/y data from a specific HDF5 file."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            x = cls._read_dataset(f, cls.x_dataset)
            y = cls._read_dataset(f, cls.y_dataset)
        return cls(x, y, source=path)

    @classmethod
    def find_in_folder(cls, folder: Path) -> Optional[Path]:
        """Return the first file inside `folder` matching this type's glob."""
        return next(Path(folder).glob(cls.file_glob), None)

    @classmethod
    def load_from_folder(cls, folder: Path) -> Optional["Measurement"]:
        """Find & load this type's file inside `folder`; None if absent."""
        path = cls.find_in_folder(folder)
        return cls.load(path) if path else None

    # -- fitting ------------------------------------------------------------
    def fit(self, routine: Optional[str] = None, **kwargs) -> FitResult:
        """Fit self.y vs self.x with a registered routine (by name).

        Falls back to `self.default_fit_routine` if `routine` isn't given.
        Extra kwargs (e.g. `p0=`, `n_peaks=`) are passed straight through to
        the routine.
        """
        routine = routine or self.default_fit_routine
        if routine is None:
            raise ValueError(
                f"No fit routine given and {type(self).__name__} has no "
                "default_fit_routine set."
            )
        fit_func = fit_registry.get(routine)
        result = fit_func(self.x, self.y, **kwargs)
        result.routine_name = routine
        self.fit_result = result
        return result

    # -- plotting -------------------------------------------------------------
    def plot(self, ax=None, *, show_fit: bool = True, **data_kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        style = {"marker": "o", "linestyle": "-", "ms": 4, "alpha": 0.6, "label": "data"}
        style.update(data_kwargs)
        ax.plot(self.x, self.y, **style)

        if show_fit and self.fit_result is not None:
            x_fit = np.linspace(self.x.min(), self.x.max(), 500)
            ax.plot(x_fit, self.fit_result.curve(x_fit), "-", lw=2,
                     label=f"fit ({self.fit_result.routine_name})")

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.name)
        ax.legend()
        return fig, ax

    def __repr__(self):
        fit_str = f", fit={self.fit_result.routine_name}" if self.fit_result else ""
        return f"<{type(self).__name__} '{self.name}' n={len(self.x)}{fit_str}>"


# ===========================================================================
# Sample: every measurement type found for one sample folder, keyed by
# type_name. New Measurement subclasses show up here automatically -- no
# changes needed here when a new type is registered elsewhere.
# ===========================================================================

class Sample:
    """All measurements found in one sample folder, keyed by type_name."""

    def __init__(self, folder: Path, measurements: dict[str, Measurement]):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.measurements = measurements

    def __getitem__(self, type_name: str) -> Measurement:
        return self.measurements[type_name]

    def __contains__(self, type_name: str) -> bool:
        return type_name in self.measurements

    def __iter__(self):
        return iter(self.measurements.values())

    def __len__(self):
        return len(self.measurements)

    def __repr__(self):
        return f"<Sample '{self.name}': {list(self.measurements)}>"

    def fit_all(self, routines: Optional[dict[str, str]] = None, **kwargs) -> dict[str, FitResult]:
        """Fit every measurement found, using each type's default routine
        unless overridden via `routines={"odmr": "lorentzian_dips", ...}`.
        Failures are warned about, not raised, so one bad fit doesn't stop
        the rest of the batch.
        """
        routines = routines or {}
        results = {}
        for type_name, m in self.measurements.items():
            try:
                results[type_name] = m.fit(routine=routines.get(type_name), **kwargs)
            except Exception as e:
                warnings.warn(f"Fit failed for '{type_name}' in sample '{self.name}': {e}")
        return results

    def plot_grid(self, save_path: Optional[Path] = None, *, fit: bool = True):
        """One row, one panel per measurement type found -- generalises the
        fixed 1x3 sat/g2/odmr layout in process_measurements.py to however
        many measurement types actually exist for this sample.
        """
        n = len(self.measurements)
        if n == 0:
            raise ValueError(f"No measurements to plot for sample '{self.name}'")

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        axes = np.atleast_1d(axes)

        for ax, (type_name, m) in zip(axes, self.measurements.items()):
            if fit and m.fit_result is None and m.default_fit_routine is not None:
                try:
                    m.fit()
                except Exception as e:
                    warnings.warn(f"Fit failed for '{type_name}' in sample '{self.name}': {e}")
            m.plot(ax=ax)

        fig.suptitle(self.name)
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

        return fig, axes


def load_sample(folder: Path, types: Optional[list[str]] = None) -> Sample:
    """Scan `folder` for every registered measurement type (or a subset
    named in `types`) and load whichever files are found. Unregistered /
    unmatched types are silently skipped, same as the old
    process_measurements.py behaviour of leaving a panel blank.
    """
    folder = Path(folder)
    types = types or list(measurement_registry)
    found = {}
    for type_name in types:
        cls = measurement_registry.get(type_name)
        m = cls.load_from_folder(folder)
        if m is not None:
            found[type_name] = m
    return Sample(folder, found)


# ===========================================================================
# Built-in fit routines
# ===========================================================================

@fit_registry.register("saturation")
def fit_saturation(x: np.ndarray, y: np.ndarray, *, p0=None, **kwargs) -> FitResult:
    """I_sat * P / (P_sat + P) + c * P  -- standard emitter saturation curve
    with a linear background term (same model as process_measurements.py).
    """
    def model(power, I_sat, P_sat, c):
        return I_sat * power / (P_sat + power) + c * power

    p0 = p0 or [y.max(), np.median(x), 0.0]
    popt, pcov = curve_fit(model, x, y, p0=p0)
    return FitResult(model=model, popt=popt, pcov=pcov, param_names=["I_sat", "P_sat", "c"])


@fit_registry.register("g2_antibunching")
def fit_g2_antibunching(x: np.ndarray, y: np.ndarray, *, p0=None, **kwargs) -> FitResult:
    """c * (1 - (1 - g2_0) * exp(-|t| / tau)) -- single-exponential
    antibunching dip, the simplest standard g(2) model.
    """
    def model(t, g2_0, tau, c):
        return c * (1 - (1 - g2_0) * np.exp(-np.abs(t) / tau))

    p0 = p0 or [max(y.min() / max(y.max(), 1e-9), 0.0), np.ptp(x) / 10 or 1.0, y.max()]
    popt, pcov = curve_fit(model, x, y, p0=p0)
    return FitResult(model=model, popt=popt, pcov=pcov, param_names=["g2_0", "tau", "c"])


def _lorentzian_dips(f: np.ndarray, y0: float, *shape_params: float) -> np.ndarray:
    """Flat baseline y0 minus a sum of Lorentzian dips.
    shape_params = [A_1..A_n, gamma_1..gamma_n, f0_1..f0_n]
    (A = dip depth, gamma = FWHM, f0 = dip center)
    """
    n = len(shape_params) // 3
    A, gamma, f0 = shape_params[:n], shape_params[n:2 * n], shape_params[2 * n:3 * n]
    y = np.full_like(f, y0, dtype=float)
    for Ai, gi, fi in zip(A, gamma, f0):
        y = y - Ai * (gi / 2) ** 2 / ((f - fi) ** 2 + (gi / 2) ** 2)
    return y


@fit_registry.register("lorentzian_dips")
def fit_lorentzian_dips(x: np.ndarray, y: np.ndarray, *, n_peaks: Optional[int] = None,
                         p0: Optional[list] = None, **kwargs) -> FitResult:
    """Fit `y` as a flat baseline minus `n_peaks` Lorentzian dips (i.e. a
    CW-ODMR spectrum). If `p0`/`n_peaks` aren't given, dips are found
    automatically with `scipy.signal.find_peaks` on the inverted,
    baseline-subtracted signal -- no manual initial guesses required for
    the common case, but you can still pass `n_peaks=2` or a full `p0` to
    override the automatic guess.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if p0 is None:
        y0_guess = np.median(y)
        inverted = y0_guess - y  # dips become peaks

        # Estimate the noise floor from point-to-point fluctuations (robust
        # to the dips themselves via the median), and require a peak to
        # rise well above that before counting it as a real feature -- a
        # fixed fraction of the signal range (e.g. 5%) is too easily
        # swamped by noise on a shallow/noisy spectrum.
        noise = 1.4826 * np.median(np.abs(np.diff(inverted)))
        prominence = max(8 * noise, 0.05 * max(inverted.max(), 1e-9))
        min_distance = max(1, len(x) // 50)  # avoid double-counting one wide dip
        peak_idx, props = find_peaks(inverted, prominence=prominence, distance=min_distance)

        if n_peaks is not None and len(peak_idx) > n_peaks:
            order = np.argsort(props["prominences"])[::-1][:n_peaks]
            peak_idx = np.sort(peak_idx[order])
        if len(peak_idx) == 0:
            peak_idx = np.array([np.argmax(inverted)])

        n = n_peaks or len(peak_idx)
        # pad with duplicated guesses if find_peaks found fewer than requested
        if len(peak_idx) < n:
            peak_idx = np.resize(peak_idx, n)

        A0 = inverted[peak_idx]
        f0_0 = x[peak_idx]
        gamma0 = np.full(n, (x.max() - x.min()) / (10 * max(n, 1)))
        p0 = [y0_guess, *A0, *gamma0, *f0_0]
    else:
        n = (len(p0) - 1) // 3

    def model(f, *params):
        return _lorentzian_dips(f, params[0], *params[1:])

    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)

    names = (
        ["y0"]
        + [f"A_{i}" for i in range(n)]
        + [f"gamma_{i}" for i in range(n)]
        + [f"f0_{i}" for i in range(n)]
    )
    return FitResult(model=model, popt=popt, pcov=pcov, param_names=names, extra={"n_peaks": n})


# ===========================================================================
# Built-in measurement types
# ===========================================================================

@Measurement.register("sat", file_glob="*satMeas*.hdf5")
class SaturationMeasurement(Measurement):
    x_dataset = "power_range"
    y_dataset = "CountData"
    x_label = "Power (mW)"
    y_label = "Counts"
    default_fit_routine = "saturation"


@Measurement.register("g2", file_glob="*g2v0*.hdf5")
class G2Measurement(Measurement):
    x_dataset = "Xdata"
    y_dataset = "Ydata"
    x_label = "Delay (ns)"
    y_label = "Coincidences"
    default_fit_routine = "g2_antibunching"


@Measurement.register("odmr", file_glob="*cwODMR*.hdf5")
class ODMRMeasurement(Measurement):
    # lists of candidate key names -- different acquisition-script versions
    # in this project have used different names for the same dataset
    x_dataset = ["fscanRange", "frequency", "freq", "Frequency", "x"]
    y_dataset = ["CountData", "counts", "signal", "Counts", "y"]
    x_label = "Frequency"
    y_label = "Counts"
    default_fit_routine = "lorentzian_dips"

    def peak_table(self):
        """Turn the current fit_result into a per-dip table with position,
        FWHM, amplitude and contrast (amplitude / baseline) -- same
        quantities odmr_all_data.py extracted from paramFITclass fits.
        Requires pandas; requires `.fit()` to have been called already.
        """
        if pd is None:
            raise ImportError("pandas is required for peak_table(); pip install pandas")
        if self.fit_result is None:
            raise ValueError("Call .fit() before .peak_table()")

        n = self.fit_result.extra.get("n_peaks")
        if n is None:
            raise ValueError(
                "fit_result has no 'n_peaks' -- peak_table() only supports "
                "results from the 'lorentzian_dips' routine"
            )

        params = self.fit_result.params()
        y0, y0_u = params["y0"]

        rows = []
        for i in range(n):
            A, A_u = params[f"A_{i}"]
            gamma, gamma_u = params[f"gamma_{i}"]
            f0, f0_u = params[f"f0_{i}"]
            rows.append({
                "dip": i,
                "f0": f0, "f0_u": f0_u,
                "fwhm": gamma, "fwhm_u": gamma_u,
                "amplitude": A, "amplitude_u": A_u,
                "baseline": y0, "baseline_u": y0_u,
                "contrast": A / y0 if y0 else np.nan,
            })
        return pd.DataFrame(rows)


# ===========================================================================
if __name__ == "__main__":
    '''
    import sys

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
        sample = load_sample(folder)
        print(sample)
        sample.fit_all()
        for type_name, m in sample.measurements.items():
            print(f"  {type_name}: {m.fit_result}")
        sample.plot_grid(save_path=Path("sample_preview.png"))
        print("Saved sample_preview.png")
    else:
        print(__doc__)
    '''
    folder_path = r"measurements\area55\T220-2_Oimplant_1E11_annealed_area55_450degC\2026_08_06_12_50_52_946126_cwODMR_noSAT_nog2_0"
    out_plot = "test.png"
    folder = Path(folder_path)
    sample = load_sample(folder)
    print(sample)
    sample.fit_all()
    sample.plot_grid(save_path=Path(out_plot))