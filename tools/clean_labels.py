"""Clean the raw hand labels into the final ground truth.

During labelling, frames where TrackNet tracked something other than the
shuttle (white shoes, a shuttle from the adjacent court) were marked with
hitter = "unknown". They are not hits, so they are removed here and
documented in labels/notes.md instead.

Clips were labelled on 720p copies; they are renamed to the original
video names so labels match the end-to-end pipeline output. The 720p
copies keep every frame, so frame numbers are unchanged.

    python -m tools.clean_labels
"""
from pathlib import Path

import pandas as pd

LABELS = Path("labels")
CLIP_NAMES = {"amogh_720": "amogh_far_23", "v03_720": "video3_03", "v27_720": "video3_27"}


def main() -> None:
    gt = pd.read_csv(LABELS / "hits_gt.csv")
    before = len(gt)
    gt = gt[gt["hitter"] != "unknown"].copy()
    gt["clip"] = gt["clip"].replace(CLIP_NAMES)
    gt = gt.sort_values(["clip", "frame"])
    gt.to_csv(LABELS / "hits_gt.csv", index=False)
    print(f"hits_gt.csv: {before} rows -> {len(gt)} hits")
    print(gt.groupby("clip").size().to_string())

    cand = pd.read_csv(LABELS / "candidates.csv")
    cand["clip"] = cand["clip"].replace(CLIP_NAMES)
    cand.to_csv(LABELS / "candidates.csv", index=False)


if __name__ == "__main__":
    main()