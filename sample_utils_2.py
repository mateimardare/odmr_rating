import pandas as pd
from pathlib import Path

# Default max allowed frequency separation (same units as the CSV's peak
# frequencies) for treating a peak at temperature N as "the same peak" as
# one at temperature N+1. Tune this to your linewidths/typical shift size.
DEFAULT_MATCH_TOLERANCE = 6.0


def _match_peaks_across_temperatures(peaks_by_temp, temperatures, tolerance):
    """
    Chain peaks for a single emitter across temperatures by nearest
    frequency instead of by row order.

    Parameters
    ----------
    peaks_by_temp : dict
        {temperature: [freq1, freq2, ...]} -- all peaks seen for this
        emitter at that temperature, in the order they appeared in the CSV.

    temperatures : list
        Ordered list of temperature labels (e.g. ["0C", "300C", ...]).

    tolerance : float
        Max frequency distance to consider two peaks at consecutive-in-time
        temperatures "the same peak".

    Returns
    -------
    list of dict
        Each dict is {temperature: frequency} for one continuous peak
        "lineage", covering only the temperatures at which that peak was
        matched (gaps are simply absent keys).
    """

    # Each open lineage tracks its most recently seen frequency (for
    # matching against the next temperature) and the values collected
    # so far.
    lineages = []

    for temp in temperatures:

        freqs = peaks_by_temp.get(temp, [])

        if not freqs:
            continue

        remaining = list(enumerate(freqs))

        # --------------------------------------------------------------
        # Build all (distance, lineage_index, freq_index) candidate pairs
        # between currently-open lineages and this temperature's peaks,
        # then greedily assign closest pairs first. This mirrors the
        # nearest-neighbor matching already used in odmr_compare.match_peaks.
        # --------------------------------------------------------------

        candidates = []

        for li, lineage in enumerate(lineages):

            if lineage["last_freq"] is None:
                continue

            for fi, f in remaining:
                candidates.append(
                    (abs(f - lineage["last_freq"]), li, fi, f)
                )

        candidates.sort(key=lambda c: c[0])

        used_lineages = set()
        used_freqs = set()

        for dist, li, fi, f in candidates:

            if li in used_lineages or fi in used_freqs:
                continue

            if dist > tolerance:
                continue

            lineages[li]["values"][temp] = f
            lineages[li]["last_freq"] = f

            used_lineages.add(li)
            used_freqs.add(fi)

        # --------------------------------------------------------------
        # Any peak at this temperature that wasn't matched to an existing
        # lineage starts a brand new one (e.g. a peak that only appears
        # after annealing).
        # --------------------------------------------------------------

        for fi, f in remaining:

            if fi in used_freqs:
                continue

            lineages.append({
                "last_freq": f,
                "values": {temp: f},
                "first_temp_index": temperatures.index(temp),
            })

    # Sort lineages by (first temperature they appear at, frequency) so
    # peak_number is stable and roughly low-to-high frequency within a
    # temperature, similar in spirit to the old row-order numbering.
    lineages.sort(
        key=lambda l: (
            l.get("first_temp_index", min(
                temperatures.index(t) for t in l["values"]
            )),
            next(iter(l["values"].values())),
        )
    )

    return [l["values"] for l in lineages]


def analyze_peak_shifts(csv_path, output_path=None, match_tolerance=DEFAULT_MATCH_TOLERANCE):
    """
    Analyze an ODMR peak CSV with the following structure:

        0C,,300C,,375C,,450C,,525C,,600C,
        emitter,peak_frequency,...

    The later temperatures contain:
        emitter
        peak_frequency

    Multiple peaks for the same emitter are preserved and are matched
    across temperatures by nearest frequency (within `match_tolerance`),
    NOT by their row position in the CSV -- a peak that disappears,
    a new peak that appears, or peaks that swap frequency order between
    temperatures are all handled correctly this way.

    Differences are calculated between consecutive temperatures
    when the peak exists at both temperatures.

    Trend:
        increasing
        decreasing
        constant
        non-monotonic
        incomplete
    """

    csv_path = Path(csv_path)

    # ============================================================
    # READ CSV RAW
    # ============================================================

    raw = pd.read_csv(
        csv_path,
        header=None
    )

    temperature_row = raw.iloc[0]
    header_row = raw.iloc[1]

    # Actual measurement data
    df = raw.iloc[2:].reset_index(drop=True)

    # ============================================================
    # FIND TEMPERATURE GROUPS
    # ============================================================

    temperature_groups = []

    for i in range(0, len(raw.columns), 2):

        temp = temperature_row.iloc[i]

        # Empty temperature column
        if pd.isna(temp):
            continue

        temp = str(temp).strip()

        # Need a pair of columns
        if i + 1 >= len(raw.columns):
            continue

        emitter_name = str(
            header_row.iloc[i]
        ).strip()

        peak_name = str(
            header_row.iloc[i + 1]
        ).strip()

        # --------------------------------------------------------
        # Determine what type of data this pair contains
        # --------------------------------------------------------

        if (
            emitter_name == "emitter"
            and peak_name == "x0"
        ):
            pair_type = "measurement"

        else:
            # Unknown/empty pair
            continue

        temperature_groups.append({
            "temperature": temp,
            "emitter_col": i,
            "peak_col": i + 1,
            "type": pair_type
        })

    print("\nTemperature groups found:")

    for group in temperature_groups:

        print(
            f"{group['temperature']}: "
            f"columns "
            f"{group['emitter_col']}/"
            f"{group['peak_col']} "
            f"({group['type']})"
        )

    temperatures = [
        group["temperature"]
        for group in temperature_groups
    ]

    # Remove duplicate temperatures if necessary, preserving order
    temperatures = list(dict.fromkeys(temperatures))

    # ============================================================
    # EXTRACT PEAKS: emitter -> temperature -> [frequencies]
    # ============================================================

    # emitter -> {temp: [freq, freq, ...]}
    emitter_temp_freqs = {}

    for group in temperature_groups:

        temp = group["temperature"]

        emitter_col = group["emitter_col"]
        peak_col = group["peak_col"]

        for _, row in df.iterrows():

            emitter = row.iloc[emitter_col]
            peak = row.iloc[peak_col]

            # ----------------------------------------------------
            # Skip missing emitter
            # ----------------------------------------------------

            if pd.isna(emitter):
                continue

            # ----------------------------------------------------
            # Skip missing peak
            # ----------------------------------------------------

            if pd.isna(peak):
                continue

            # ----------------------------------------------------
            # Clean emitter
            # ----------------------------------------------------

            try:
                emitter = int(float(emitter))

            except (ValueError, TypeError):
                continue

            # ----------------------------------------------------
            # Clean peak
            # ----------------------------------------------------

            try:
                peak = float(peak)

            except (ValueError, TypeError):
                continue

            emitter_temp_freqs.setdefault(
                emitter, {}
            ).setdefault(
                temp, []
            ).append(peak)

    # ============================================================
    # MATCH PEAKS ACROSS TEMPERATURES (per emitter, by frequency)
    # ============================================================

    result_rows = []

    for emitter in sorted(emitter_temp_freqs.keys()):

        peaks_by_temp = emitter_temp_freqs[emitter]

        lineages = _match_peaks_across_temperatures(
            peaks_by_temp,
            temperatures,
            match_tolerance
        )

        for peak_number, values in enumerate(lineages, start=1):

            row = {
                "emitter": emitter,
                "peak_number": peak_number
            }

            # ------------------------------------------------
            # Add peak frequencies
            # ------------------------------------------------

            for temp in temperatures:
                row[f"peak_{temp}"] = values.get(temp, float("nan"))

            # ------------------------------------------------
            # Consecutive differences
            # ------------------------------------------------

            differences = []

            for temp1, temp2 in zip(temperatures[:-1], temperatures[1:]):

                p1 = values.get(temp1)
                p2 = values.get(temp2)

                if p1 is not None and p2 is not None:

                    difference = p2 - p1
                    row[f"diff_{temp1}_{temp2}"] = difference
                    differences.append(difference)

                else:

                    row[f"diff_{temp1}_{temp2}"] = float("nan")

            # ------------------------------------------------
            # First-to-last difference
            # ------------------------------------------------

            available = [
                values[temp]
                for temp in temperatures
                if temp in values
            ]

            if len(available) >= 2:
                row["diff_first_last"] = available[-1] - available[0]
            else:
                row["diff_first_last"] = float("nan")

            # ------------------------------------------------
            # TREND
            # ------------------------------------------------

            number_of_steps = len(available) - 1

            if (
                number_of_steps > 0
                and len(differences) == number_of_steps
            ):

                if all(d > 0 for d in differences):
                    row["trend"] = "increasing"
                elif all(d < 0 for d in differences):
                    row["trend"] = "decreasing"
                elif all(d == 0 for d in differences):
                    row["trend"] = "constant"
                else:
                    row["trend"] = "non-monotonic"

            else:
                row["trend"] = "incomplete"

            result_rows.append(row)

    # ============================================================
    # CREATE DATAFRAME
    # ============================================================

    result = pd.DataFrame(result_rows)

    if result.empty:
        raise ValueError("No peaks were found in the CSV.")

    # ============================================================
    # SORT
    # ============================================================

    result = result.sort_values(["emitter", "peak_number"])

    # ============================================================
    # SAVE
    # ============================================================

    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_analyzed.csv")

    result.to_csv(output_path, index=False)

    print(f"\nSaved {len(result)} peaks to:")
    print(output_path)

    return result


def extract_emitter_peaks(csv_path):
    """
    Read the peak CSV and collect all frequencies for each emitter.

    The CSV is expected to have:

        0C,,300C,,375C,,450C,,525C,,600C,
        after_emitter,peak_before,emitter,peak_frequency,...

    Returns
    -------
    dict
        {
            emitter: [
                {
                    "temperature": ...,
                    "frequency": ...,
                    "source": ...
                },
                ...
            ]
        }
    """

    csv_path = Path(csv_path)

    raw = pd.read_csv(
        csv_path,
        header=None
    )

    # ------------------------------------------------------------
    # First row = temperature
    # Second row = column names
    # ------------------------------------------------------------

    temperature_row = raw.iloc[0]
    header_row = raw.iloc[1]

    data = raw.iloc[2:].reset_index(drop=True)

    emitter_peaks = {}

    # ------------------------------------------------------------
    # Go through pairs of columns
    # ------------------------------------------------------------

    for i in range(0, len(raw.columns), 2):

        temp = temperature_row.iloc[i]

        if pd.isna(temp):
            continue

        temp = str(temp).strip()

        # Need a pair of columns (guards against a trailing odd column,
        # matching the same check already used in analyze_peak_shifts).
        if i + 1 >= len(raw.columns):
            continue

        emitter_name = str(
            header_row.iloc[i]
        ).strip()

        frequency_name = str(
            header_row.iloc[i + 1]
        ).strip()

        # --------------------------------------------------------
        # Special case:
        #
        # 0C has:
        # after_emitter, peak_before
        # --------------------------------------------------------

        if (
            emitter_name == "after_emitter"
            and frequency_name == "peak_before"
        ):
            source = "before"

        elif (
            emitter_name == "emitter"
            and frequency_name == "peak_frequency"
        ):
            source = "peak"

        else:
            continue

        # --------------------------------------------------------
        # Extract rows
        # --------------------------------------------------------

        for _, row in data.iterrows():

            emitter = row.iloc[i]
            frequency = row.iloc[i + 1]

            if pd.isna(emitter):
                continue

            if pd.isna(frequency):
                continue

            try:
                emitter = int(float(emitter))
                frequency = float(frequency)
            except (ValueError, TypeError):
                continue

            if emitter not in emitter_peaks:
                emitter_peaks[emitter] = []

            emitter_peaks[emitter].append({
                "temperature": temp,
                "frequency": frequency,
                "source": source
            })

    return emitter_peaks


def group_emitter_peaks(
    emitter_peaks,
    threshold=6.0
):
    """
    Sort peaks for each emitter and group peaks whose
    consecutive frequency difference is <= threshold.

    NOTE: grouping is transitive along consecutive gaps only -- the
    first and last peak in a long group can end up farther apart than
    `threshold` if there's a chain of small hops between them. If you
    need a hard cap on a group's total span, check
    (group[-1]["frequency"] - group[0]["frequency"]) after grouping.

    Parameters
    ----------
    emitter_peaks : dict
        Output from extract_emitter_peaks()

    threshold : float
        Maximum allowed separation between neighboring
        sorted peaks.

    Returns
    -------
    dict
        {
            emitter: [
                [peak1, peak2, ...],
                [peak1, peak2, ...],
                ...
            ]
        }
    """

    grouped = {}

    for emitter, peaks in emitter_peaks.items():

        # Sort by frequency
        sorted_peaks = sorted(
            peaks,
            key=lambda x: x["frequency"]
        )

        groups = []

        for peak in sorted_peaks:

            frequency = peak["frequency"]

            # First peak
            if not groups:
                groups.append([peak])
                continue

            # Last peak in current group
            previous_frequency = groups[-1][-1]["frequency"]

            # ----------------------------------------------------
            # If close enough, add to current group
            # ----------------------------------------------------

            if frequency - previous_frequency <= threshold:

                groups[-1].append(peak)

            # ----------------------------------------------------
            # Otherwise start a new group
            # ----------------------------------------------------

            else:

                groups.append([peak])

        grouped[emitter] = groups

    return grouped


def export_grouped_peaks(
    grouped_peaks,
    output_path
):
    """
    Export grouped emitter peaks to CSV.

    One row = one peak.

    Each peak gets:
        emitter
        group
        frequency
        temperature
        source
    """

    rows = []

    for emitter, groups in grouped_peaks.items():

        for group_number, group in enumerate(groups, start=1):

            for peak in group:

                rows.append({
                    "emitter": emitter,
                    "group": group_number,
                    "temperature": peak["temperature"],
                    "frequency": peak["frequency"],
                    "source": peak["source"],
                })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["emitter", "group", "frequency"]
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved {len(result)} peaks to {output_path}"
    )

    return result

