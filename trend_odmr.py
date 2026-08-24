from odmr_all_data import *

# ============================================================
# EMITTER DEFINITIONS
# ============================================================

PL6_EMITTER_LIST = {
    55: [23, 95, 43, 49, 63, 68, 74, 79, 83, 111, 112, 114],
    45: [95, 68, 36, 12, 62, 71, 90,  ]
}

PL5_EMITTER_LIST = {
    55: [55, 61, 88, 86, 33, 6, 87],
    45: [21, 33, 38, 39, 41, 46]
}


# ============================================================
# AUXILIARY DATAFRAME
# ============================================================

def _df_aux(
    fitres,
    measurement_folder,
    info,
    meas_type,
    emitter_type,
):

    df = extract_peak_params(fitres)

    df.insert(0, "measurement_folder", measurement_folder.name)
    df.insert(1, "date", info["date"])
    df.insert(2, "time", info["time"])
    df.insert(3, "measurement_number", info["number"])
    df.insert(4, "emitter", info["emitter"])
    df.insert(5, "emitter_type", emitter_type)
    df.insert(6, "measurement_type", meas_type)

    return df


# ============================================================
# NORMALIZED ODMR
# ============================================================

def normal_odmr(fd: str):
    """
    Normalize ODMR data to the fitted baseline.

    Adds 20 artificial baseline points:
    10 before and 10 after the measured spectrum.
    """

    data = load_odmr(
        fd,
        meas_type=MEAS_TYPE
    )

    fitres, xplot, yplot = fit_odmr(
        data["x"],
        data["y"]
    )

    fit_params = extract_peak_params(fitres)

    baseline = fit_params["baseline"]

    # Normalize to baseline
    norm_cnts = data["y"] / baseline

    # Frequency spacing
    dx = np.mean(np.diff(data["x"]))

    # Artificial points before
    x_left = (
        data["x"][0]
        - np.arange(10, 0, -1) * dx
    )

    # Artificial points after
    x_right = (
        data["x"][-1]
        + np.arange(1, 11) * dx
    )

    y_left = np.ones(10)
    y_right = np.ones(10)

    # Combine
    x_norm = np.concatenate([
        x_left,
        data["x"],
        x_right
    ])

    y_norm = np.concatenate([
        y_left,
        norm_cnts,
        y_right
    ])

    return x_norm, y_norm

# ============================================================
# FITTING
# ============================================================

def fit_PL5(x: np.ndarray, y: np.ndarray):
    """Fit PL5 ODMR spectrum."""

    fitter = paramFITclass()

    startparam = {}
    startparam['B'] = np.amax(y)
    
    if x[np.argmin(y)] in range(1342, 1355):
        startparam['x0'] = x[np.argmin(y)]
        startparam['A'] = np.amax(y) - np.amin(y)
    else: 
        startparam['x0'] = 1350
        startparam['A'] = y[len(y)/2]

    tmp = np.sum(y < (startparam['B'] - startparam['A'] / 2))
    startparam['gamma'] = 0.5 * np.absolute((x[-1] - x[0]) / len(x)) * tmp
    if tmp < 1:
        tmp = 1


    fitres, xplot, yplot, _yplot_guess = (
        fitter.fitLorentzianSingleDip(
            x,
            y,
            printresults=False,
            startparam = startparam
        )
    )

    return fitres, xplot, yplot


def fit_PL6(x: np.ndarray, y: np.ndarray):
    """Fit PL6 ODMR spectrum."""

    fitter = paramFITclass()

    fitres, xplot, yplot, _yplot_guess = (
        fitter.fitLorentzian(
            x,
            y,
            printresults=False
        )
    )

    return fitres, xplot, yplot


# ============================================================
# ANALYZE ONE SAMPLE FOLDER
# ============================================================

def analyze_PL(
    folder,
    area,
    meas_type: str = MEAS_TYPE,
) -> pd.DataFrame:

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

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
            # Area comes from the parent sample folder
            area = int(area)

            # Emitter comes from the individual measurement folder
            emitter = int(info["emitter"])

            data = load_odmr(
                measurement_folder,
                meas_type
            )

            # ------------------------------------------------
            # PL5
            # ------------------------------------------------

            if emitter in PL5_EMITTER_LIST.get(area, []):

                fitres, xplot, yplot = fit_PL5(
                    data["x"],
                    data["y"]
                )
                if fitres["x0"]>1352:
                    continue

                df = _df_aux(
                    fitres,
                    measurement_folder,
                    info,
                    meas_type,
                    "PL5"
                )

                all_dfs.append(df)

                print(
                    f"OK PL5: area {area}, "
                    f"emitter {emitter}: "
                    f"{measurement_folder.name}"
                )

            # ------------------------------------------------
            # PL6
            # ------------------------------------------------

            elif emitter in PL6_EMITTER_LIST.get(area, []):

                fitres, xplot, yplot = fit_PL6(
                    data["x"],
                    data["y"]
                )

                df = _df_aux(
                    fitres,
                    measurement_folder,
                    info,
                    meas_type,
                    "PL6"
                )

                all_dfs.append(df)

                print(
                    f"OK PL6: area {area}, "
                    f"emitter {emitter}: "
                    f"{measurement_folder.name}"
                )

            # ------------------------------------------------
            # Not in either list
            # ------------------------------------------------

            else:

                print(
                    f"SKIP: area {area}, "
                    f"emitter {emitter} "
                    f"not in PL5/PL6 list"
                )

        except Exception as e:

            print(
                f"WARNING: {measurement_folder.name} "
                f"failed: {e}"
            )

    if all_dfs:

        return pd.concat(
            all_dfs,
            ignore_index=True
        )

    return pd.DataFrame()


# ============================================================
# ANALYZE ALL AREAS / TEMPERATURES
# ============================================================

def export_all_PL_peaks(
    base_path: Path = BASE_PATH,
    meas_type: str = MEAS_TYPE,
    csv_path: Path = Path(
        "all_PL_peaks_all_samples.csv"
    ),
) -> pd.DataFrame:

    sample_dirs = find_all_sample_dirs(
        base_path
    )

    if not sample_dirs:
        raise FileNotFoundError(
            f"No folders matching area/temp pattern "
            f"found in {base_path}"
        )

    all_dfs = []

    for entry in sample_dirs:

        area = int(entry["area"])
        temp = entry["anneal_temp_C"]
        folder = entry["path"]

        print(
            f"\n=== Area {area}, "
            f"{temp} degC "
            f"({folder.name}) ==="
        )
        
        df = analyze_PL(
            folder,
            area,
            meas_type=meas_type
        )

        if not df.empty:

            # Add experimental conditions
            df.insert(
                0,
                "area",
                area
            )

            df.insert(
                1,
                "anneal_temp_C",
                temp
            )

            all_dfs.append(df)

    if all_dfs:

        result = pd.concat(
            all_dfs,
            ignore_index=True
        )

    else:

        result = pd.DataFrame()

    result.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\n=== Wrote {len(result)} total peaks "
        f"across {len(sample_dirs)} folder(s) "
        f"to:\n{csv_path}"
    )

    return result



# ============================================================
# SIMPLE TEMPERATURE PLOT
# ============================================================
if __name__ == "__main__":
    df = export_all_PL_peaks()
    pl5 = df[df["emitter_type"] == "PL5"]

    pl6 = df[df["emitter_type"] == "PL6"]

    df = pd.read_csv("all_PL_peaks_all_samples.csv")
    print(df.columns.to_list())
