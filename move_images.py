# this is just AI vibe coding to be treated as is

from pathlib import Path
import pandas as pd
import re
import shutil


# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = Path("comparison_plots")
MAP_PATH = Path("sat_g2_odmr_maps")

# Change these if your CSV has different column names/order
MAP_COLUMNS = [
    "before_emitter",
    "after_emitter",
]


# ============================================================
# LOAD EMITTER MAPPING
# ============================================================

def load_map(sample_id: str) -> pd.DataFrame:
    """Load the before -> after emitter relabeling map for one sample.

    Expects a file matching '*_<sample_id>_*.csv' inside MAP_PATH.
    """

    matches = sorted(MAP_PATH.glob(f"*_area{sample_id}_*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No map csv found for sample '{sample_id}' in {MAP_PATH}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple map files match sample '{sample_id}', "
            f"using {matches[0].name}"
        )

    df = pd.read_csv(
        matches[0],
        header=None,
        names=MAP_COLUMNS
    )

    df["after_emitter"] = df["after_emitter"].astype(str)
    df["before_emitter"] = df["before_emitter"].astype(str)

    return df


# ============================================================
# GET AREA
# ============================================================

def get_area(folder_name: str) -> str | None:
    """Extract area from a folder name.

    Examples:
        ..._area45
        ..._area45_300degC

    Both return '45'.
    """

    match = re.search(r"_area(\d+)(?:_|$)", folder_name)

    if match:
        return match.group(1)

    return None


# ============================================================
# GET EMITTER FROM IMAGE NAME
# ============================================================

def get_emitter(image_name: str) -> str | None:
    """Extract emitter number from the end of the PNG filename.

    Example:
        ..._cwODMR_SAT_g2_344.png

    returns:
        '344'
    """

    stem = Path(image_name).stem

    match = re.search(r"_(\d+)$", stem)

    if match:
        return match.group(1)

    return None


# ============================================================
# GET SAMPLE ID
# ============================================================

def get_sample_id(folder_name: str) -> str:
    """Extract the sample ID used for finding the mapping CSV.

    Adjust this function if your actual sample IDs have a
    different format.

    Example:
        T220-2_Oimplant_1E11_annealed_area45

    Here the sample ID is assumed to be:
        T220-2
    """

    # Example: T220-2_Oimplant_...
    match = re.match(r"([^_]+)", folder_name)

    if match:
        return match.group(1)

    raise ValueError(
        f"Could not determine sample ID from folder '{folder_name}'"
    )


# ============================================================
# PROCESS
# ============================================================

for sample_folder in BASE_PATH.iterdir():

    if not sample_folder.is_dir():
        continue

    cwodmr = sample_folder / "cwODMR"

    if not cwodmr.exists():
        continue

    folder_name = sample_folder.name

    # --------------------------------------------------------
    # Determine area
    # --------------------------------------------------------

    area = get_area(folder_name)

    if area is None:
        print(
            f"WARNING: Could not determine area from folder: "
            f"{folder_name}"
        )
        continue

    # --------------------------------------------------------
    # Determine whether this is a BEFORE sample
    # or AFTER/temperature sample
    # --------------------------------------------------------

    is_before = "degC" not in folder_name

    # --------------------------------------------------------
    # Load mapping only for BEFORE samples
    # --------------------------------------------------------

    emitter_map = None

    if is_before:

        sample_id = get_sample_id(folder_name)

        try:
            emitter_map = load_map(sample_id)

        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

        # Make a simple lookup dictionary:
        # before emitter -> after emitter
        emitter_map = dict(
            zip(
                emitter_map["before_emitter"],
                emitter_map["after_emitter"]
            )
        )

        print(
            f"\nBEFORE sample: {folder_name}"
        )
        print(
            f"Sample ID: {sample_id}"
        )
        print(
            f"Area: {area}"
        )
        print(
            f"Emitter map: {emitter_map}"
        )

    else:
        print(
            f"\nAFTER sample: {folder_name}"
        )
        print(
            f"Area: {area}"
        )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image in cwodmr.glob("*.png"):

        original_emitter = get_emitter(image.name)

        if original_emitter is None:
            print(
                f"WARNING: Could not determine emitter from "
                f"{image.name}"
            )
            continue

        # ----------------------------------------------------
        # BEFORE SAMPLE:
        # convert before emitter -> after emitter
        # ----------------------------------------------------

        if is_before:

            if original_emitter not in emitter_map:
                print(
                    f"WARNING: No mapping found for emitter "
                    f"{original_emitter} in {folder_name}"
                )
                continue

            emitter = emitter_map[original_emitter]

        # ----------------------------------------------------
        # AFTER SAMPLE:
        # emitter stays unchanged
        # ----------------------------------------------------

        else:
            emitter = original_emitter

        # ----------------------------------------------------
        # Create:
        #
        # comparison_plots/
        #     area_45/
        #         emitter_344/
        #             ...
        #
        # ----------------------------------------------------

        destination_dir = (
            BASE_PATH
            / f"area_{area}"
            / f"emitter_{emitter}"
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Keep complete sample/temperature information
        new_name = f"{folder_name}_{image.name}"

        destination = destination_dir / new_name

        # ----------------------------------------------------
        # MOVE
        # ----------------------------------------------------

        shutil.move(
            str(image),
            str(destination)
        )

        print(
            f"{image.name} -> "
            f"area_{area} / "
            f"emitter_{emitter}"
        )