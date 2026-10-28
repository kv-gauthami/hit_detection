"""Loading and cleaning shuttle tracks produced by TrackNetV3.

TrackNetV3 writes one row per frame with columns ``Frame, Visibility, X, Y``
(X = Y = 0 when the shuttle is not detected). Its raw output has three
kinds of error that we observed on our clips:

1. Spikes: single frames that jump to another bright object (a second
   shuttle on the adjacent court, a player's white shoes) and back.
2. Static detections: the model locks onto a still object (shoes, a
   shuttle lying on the floor) for many frames.
3. Short isolated fragments: a few detections surrounded by gaps.

``clean_track`` removes all three and bridges short, physically
consistent gaps. Thresholds scale with the shuttle's own speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DetectorConfig, frames


def load_tracknet_csv(path: str | Path) -> pd.DataFrame:
    """Load a shuttle track CSV written by TrackNetV3.

    Returns a DataFrame indexed by ``frame`` with float columns ``x`` and
    ``y`` (NaN when the shuttle was not detected).
    """
    df = pd.read_csv(path)
    missing = {"Frame", "Visibility", "X", "Y"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    out = pd.DataFrame(
        {"x": df["X"].astype(float), "y": df["Y"].astype(float)},
        index=pd.Index(df["Frame"].astype(int), name="frame"),
    )
    out.loc[df["Visibility"].to_numpy() != 1, ["x", "y"]] = np.nan
    # Guarantee one row per frame so frame index == row position.
    full = pd.RangeIndex(out.index.min(), out.index.max() + 1, name="frame")
    return out.reindex(full)


def step_lengths(track: pd.DataFrame) -> pd.Series:
    """Distance moved between consecutive frames (NaN across gaps)."""
    return np.hypot(track["x"].diff(), track["y"].diff())


def typical_step(track: pd.DataFrame) -> float:
    """Median per-frame movement of the shuttle in this clip (pixels)."""
    step = step_lengths(track).median()
    return float(step) if np.isfinite(step) and step > 0 else 1.0


def _remove_spikes(track: pd.DataFrame, fps: float, cfg: DetectorConfig) -> pd.DataFrame:
    """Remove frames far from the rolling median of their neighbours.

    Pass 1 uses the clip-wide typical step and removes gross jumps.
    Pass 2 uses the local speed (now uncontaminated by those jumps), so
    fast genuine motion, e.g. the tip of a smash, is not mistaken for a spike.
    """
    t = track.copy()
    win = frames(cfg.outlier_window_s, fps, minimum=3) | 1  # odd
    unit = typical_step(t)

    for local in (False, True):
        med = t.rolling(win, center=True, min_periods=3).median()
        dev = np.hypot(t["x"] - med["x"], t["y"] - med["y"])
        if local:
            speed = (step_lengths(t).rolling(win + 2, center=True, min_periods=3)
                     .median().fillna(unit).clip(lower=0.25 * unit))
        else:
            speed = unit
        thr = cfg.outlier_k * speed + cfg.outlier_floor_rel * unit
        t.loc[dev > thr, ["x", "y"]] = np.nan
    return t


def _fill_short_gaps(track: pd.DataFrame, fps: float, cfg: DetectorConfig) -> pd.DataFrame:
    """Linearly bridge short gaps whose two ends are consistent.

    A gap is bridged only if the jump across it is no faster than the
    shuttle can plausibly move, so a gap never joins two unrelated objects.
    """
    t = track.copy()
    xy = t[["x", "y"]].to_numpy(copy=True)
    valid = ~np.isnan(xy[:, 0])
    max_gap = frames(cfg.max_gap_fill_s, fps)
    max_speed = cfg.outlier_k * typical_step(t)

    i, n = 0, len(t)
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            a, b = xy[i - 1], xy[j]
            if np.hypot(*(b - a)) / (j - i + 1) <= max_speed:
                steps = np.arange(1, j - i + 1)[:, None] / (j - i + 1)
                xy[i:j] = a + (b - a) * steps
        i = j
    t[["x", "y"]] = xy
    return t


def _remove_static(track: pd.DataFrame, fps: float, cfg: DetectorConfig) -> pd.DataFrame:
    """Remove detections that stay (almost) still for a long time.

    A flying shuttle is never still for ``static_window_s``; even at the
    apex of a high clear it keeps drifting. Still detections are objects
    such as shoes or a shuttle lying on the floor.
    """
    t = track.copy()
    win = frames(cfg.static_window_s, fps, minimum=3) | 1
    roll = t.rolling(win, center=True, min_periods=5)
    span = np.hypot(roll.max()["x"] - roll.min()["x"], roll.max()["y"] - roll.min()["y"])
    t.loc[span < cfg.static_range_rel * typical_step(t), ["x", "y"]] = np.nan
    return t


def _remove_fragments(track: pd.DataFrame, fps: float, cfg: DetectorConfig) -> pd.DataFrame:
    """Remove runs of detections shorter than ``min_fragment_s``."""
    t = track.copy()
    missing = t["x"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    t.loc[~missing & (run_len < frames(cfg.min_fragment_s, fps)), ["x", "y"]] = np.nan
    return t


def clean_track(track: pd.DataFrame, fps: float, cfg: DetectorConfig | None = None) -> pd.DataFrame:
    """Return a cleaned copy of a raw TrackNet track (same index, NaN = no shuttle)."""
    # cfg = cfg or DetectorConfig()
    # t = _remove_spikes(track, fps, cfg)
    cfg = cfg or DetectorConfig()
    t = track.copy()
    t.iloc[: frames(cfg.startup_trim_s, fps, minimum=0)] = np.nan
    t = _remove_spikes(t, fps, cfg)
    t = _fill_short_gaps(t, fps, cfg)
    t = _remove_static(t, fps, cfg)
    t = _remove_fragments(t, fps, cfg)
    return t
