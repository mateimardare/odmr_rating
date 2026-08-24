"""
Compare ODMR fits for the same physical pillar before/after annealing.

LAYOUT THIS ASSUMES (edit the CONFIG block if it doesn't match):

1. There is one "before" folder and one "after" (300 degC) folder per
   sample, e.g.:
       T220-2_Oimplant_1E11_annealed_area45           <- before, sample 45
       T220-2_Oimplant_1E11_annealed_area45_300degC    <- after,  sample 45
       T220-2_Oimplant_1E11_annealed_area55           <- before, sample 55
       T220-2_Oimplant_1E11_annealed_area55_300degC    <- after,  sample 55
   Both live under BASE_PATH.

2. Inside each of those folders are the individual per-pillar measurement
   sub-folders, named:
       YYYY_MM_DD_HH_MM_SS_<number>_<type1>_<type2>_..._<emitter>
   e.g. 2026_08_01_21_05_33_539824_cwODMR_SAT_g2_0
   each containing one .h5/.hdf5 file with datasets "fscanRange" (x) and
   "CountData" (y). Adjust `load_odmr` if your keys differ.

3. There is ONE map csv per sample (matching '*_45.csv', '*_55.csv', ...)
   in MAP_PATH, no header, where:
       column 0 -> emitter label used in the AFTER (300degC) scan
       column 1 -> emitter label used in the BEFORE scan (the original
                   numbering from the first scan)
   i.e. the map directly pairs "before" <-> "after" emitter labels for the
   same physical pillar. If your columns are swapped, flip MAP_COLUMNS
   below.

If any of this is off, tell me what to change.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import sample_utils
import sample_utils_2
from parameterFittingV0 import paramFITclass

TARGET_AREA = "45"


if __name__ == "__main__":
    
    # peaks = sample_utils_2.extract_emitter_peaks(
    #     f"peaks_area{TARGET_AREA}.csv"
    # )
    # sample_utils_2.analyze_peak_shifts(f"peaks_area{TARGET_AREA}.csv", f"peak_shift_analysis_area{TARGET_AREA}.csv")

    # for subfolder in sample_utils.BASE_PATH.iterdir():
    #     if subfolder.is_dir():
    #         # plot_all_measurements_in_folder(subfolder)
    #         # peaks = sample_utils.export_all_peaks_in_folder(
    #         #     subfolder,
    #         #     meas_type="cwODMR",
    #         #     csv_path=Path(f"processed/{subfolder.name}_peaks.csv")
    #         # )
    #         print(subfolder)
    sample_utils.plot_all_measurements_in_folder(sample_utils.BASE_PATH / "T220_20_Oimplant_area55_0degC")