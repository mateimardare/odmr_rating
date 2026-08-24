"""
Rating game.

Shows each image found under processed/<batch>/sat_g2_odmr/ one at a time
(across every batch). Press:

    RIGHT arrow  (or  g)  -> GOOD  (label = 1)
    LEFT  arrow  (or  b)  -> BAD   (label = 0)
    q                      -> quit (progress is saved, resume later)

Results are appended to ratings.csv as they're made, with columns:
    batch, sample_name, emitter, label

Already-rated images are skipped automatically if you run this again,
so you can stop and resume whenever you like.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from odmr_all_data import parse_sample_name

PROCESSED_ROOT = Path("processed")
RATINGS_FILE = Path("ratings.csv")
FIELDNAMES = ["batch", "sample_name", "emitter", "label"]


def find_images():
    # processed/<batch>/sat_g2_odmr/<sample>.png
    return sorted(PROCESSED_ROOT.glob("*/sat_g2_odmr/*.png"))


def batch_of(img_path: Path) -> str:
    # .../<batch>/sat_g2_odmr/<sample>.png -> <batch>
    return img_path.parent.parent.name

GOOD_KEYS = {"right", "g"}
BAD_KEYS = {"left", "b"}
QUIT_KEYS = {"q", "escape"}


def load_already_rated() -> set:
    """Returns a set of (batch, sample_name) tuples already rated."""
    rated = set()
    if RATINGS_FILE.exists():
        with open(RATINGS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rated.add((row["batch"], row["sample_name"]))
    return rated


def append_rating(row: dict, needs_header: bool):
    with open(RATINGS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def run_game():
    images = find_images()
    if not images:
        print(f"No images found under '{PROCESSED_ROOT}/*/sat_g2_odmr/'. Run process_measurements.py first.")
        return

    rated = load_already_rated()
    todo = [img for img in images if (batch_of(img), img.stem) not in rated]

    if not todo:
        print("Everything has already been rated. Check ratings.csv.")
        return

    needs_header = (not RATINGS_FILE.exists()) or RATINGS_FILE.stat().st_size == 0
    state = {"i": 0, "needs_header": needs_header}

    print(f"{len(todo)} image(s) to rate.")
    print("RIGHT / g = GOOD (1)   LEFT / b = BAD (0)   q = quit\n")

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(bottom=0.05, top=0.90)

    def show_current():
        ax.clear()
        img_path = todo[state["i"]]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"[{state['i'] + 1}/{len(todo)}] {batch_of(img_path)} / {img_path.stem}\n"
            "RIGHT/g = good     LEFT/b = bad     q = quit"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        if key in QUIT_KEYS:
            print("Quitting. Progress saved - rerun this script to resume.")
            plt.close(fig)
            return

        if key not in GOOD_KEYS and key not in BAD_KEYS:
            return

        label = 1 if key in GOOD_KEYS else 0
        img_path = todo[state["i"]]
        meta = parse_sample_name(img_path)
        batch = batch_of(img_path)

        append_rating(
            {
                "batch": batch,
                "sample_name": meta["sample_name"],
                "emitter": meta["emitter"],
                "label": label,
            },
            needs_header=state["needs_header"],
        )
        state["needs_header"] = False

        print(f"{batch} / {meta['sample_name']} (emitter {meta['emitter']}) -> {label}")

        state["i"] += 1
        if state["i"] >= len(todo):
            print("\nAll done! Ratings saved to ratings.csv")
            plt.close(fig)
            return

        show_current()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_current()
    plt.show()


if __name__ == "__main__":
    run_game()