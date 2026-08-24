from pathlib import Path
import re
import shutil


# ============================================================
# SETTINGS
# ============================================================

SOURCE_PATH = Path(r"measurements")
OUTPUT_PATH = Path(r"SELECTED_MEASUREMENTS")


PL6_EMITTER_LIST = {
    55: [23, 95, 43, 49, 63, 68, 74, 79, 83, 111, 112, 114],
    45: [95, 68, 36, 12, 62, 71, 90],
}

PL5_EMITTER_LIST = {
    55: [55, 61, 88, 86, 33, 6, 87],
    45: [21, 33, 38, 39, 41, 46],
}


# ============================================================
# GET AREA
# ============================================================

def get_area(folder_name):
    """
    Extract area from folder name.

    Example:
        T220-2_Oimplant_1E11_annealed_area45_0degC

    returns:
        45
    """

    match = re.search(r"area(\d+)", folder_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# GET TEMPERATURE
# ============================================================

def get_temperature(folder_name):
    """
    Extract temperature from folder name.

    Example:
        T220-2_Oimplant_1E11_annealed_area45_300degC

    returns:
        300
    """

    match = re.search(r"_(\d+)degC", folder_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# GET EMITTER
# ============================================================

def get_emitter(folder_name):
    """
    Extract emitter number from the end of the measurement
    folder name.

    Example:
        2026_08_12_17_28_19_726667_cwODMR_noSAT_nog2_115

    returns:
        115
    """

    match = re.search(r"_(\d+)$", folder_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# PROCESS
# ============================================================

for area_folder in SOURCE_PATH.iterdir():

    if not area_folder.is_dir():
        continue

    # --------------------------------------------------------
    # Get area
    # --------------------------------------------------------

    area_match = re.search(r"area(\d+)", area_folder.name)

    if not area_match:
        print(f"WARNING: Could not determine area from {area_folder}")
        continue

    area = int(area_match.group(1))

    print(f"\n{'=' * 60}")
    print(f"AREA {area}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # Loop through temperature folders
    # --------------------------------------------------------

    for temperature_folder in area_folder.iterdir():

        if not temperature_folder.is_dir():
            continue

        temperature = get_temperature(temperature_folder.name)

        if temperature is None:
            print(
                f"WARNING: Could not determine temperature from "
                f"{temperature_folder.name}"
            )
            continue

        print(f"\nTemperature: {temperature} degC")

        # ----------------------------------------------------
        # Loop through measurement folders
        # ----------------------------------------------------

        for measurement_folder in temperature_folder.iterdir():

            if not measurement_folder.is_dir():
                continue

            emitter = get_emitter(measurement_folder.name)

            if emitter is None:
                print(
                    f"WARNING: Could not determine emitter from "
                    f"{measurement_folder.name}"
                )
                continue

            # ------------------------------------------------
            # Check emitter lists
            # ------------------------------------------------

            if emitter in PL5_EMITTER_LIST.get(area, []):
                measurement_type = "PL5"

            elif emitter in PL6_EMITTER_LIST.get(area, []):
                measurement_type = "PL6"

            else:
                # Not a selected emitter
                continue

            # ------------------------------------------------
            # Create destination
            #
            # PL5/
            #   AREA45/
            #       300degC/
            #           measurement_folder/
            # ------------------------------------------------

            destination = (
                OUTPUT_PATH
                / measurement_type
                / f"AREA{area}"
                / f"{temperature}degC"
                / measurement_folder.name
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Copy entire measurement folder
            # ------------------------------------------------

            if destination.exists():
                print(
                    f"Already exists, skipping: "
                    f"{measurement_folder.name}"
                )
                continue

            shutil.copytree(
                measurement_folder,
                destination
            )

            print(
                f"COPIED | "
                f"{measurement_type} | "
                f"AREA{area} | "
                f"{temperature}degC | "
                f"emitter {emitter}"
            )


print("\nDone!")