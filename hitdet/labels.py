"""Ground-truth hit labels and candidate lists.

Both are small CSV files, one row per hit, shared across clips:

    labels/hits_gt.csv      clip,frame,hitter    (hitter: near | far | unknown)
    labels/candidates.csv   clip,frame

``clip`` is the video file name without extension (e.g. ``v03_720``).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

HITTERS = ("near", "far", "unknown")
LABEL_FIELDS = ("clip", "frame", "hitter")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_clip_labels(path: str | Path, clip: str) -> dict[int, str]:
    """Return ``{frame: hitter}`` for one clip (empty if none yet)."""
    labels: dict[int, str] = {}
    for row in _read_rows(Path(path)):
        if row["clip"] == clip:
            hitter = row.get("hitter") or "unknown"
            labels[int(row["frame"])] = hitter if hitter in HITTERS else "unknown"
    return labels


def save_clip_labels(path: str | Path, clip: str, labels: dict[int, str]) -> None:
    """Replace one clip's labels in the shared file, keeping other clips.

    Writes to a temporary file first and then renames it, so a crash
    mid-write never corrupts existing labels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r for r in _read_rows(path) if r["clip"] != clip]
    rows += [
        {"clip": clip, "frame": str(frame), "hitter": hitter}
        for frame, hitter in labels.items()
    ]
    rows.sort(key=lambda r: (r["clip"], int(r["frame"])))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_candidates(path: str | Path, clip: str) -> list[int]:
    """Return the sorted candidate hit frames for one clip."""
    return sorted(
        int(row["frame"]) for row in _read_rows(Path(path)) if row["clip"] == clip
    )
