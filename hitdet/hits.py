"""Hit detection from a cleaned shuttle track.

Definition used
---------------
A hit is the frame where a racket changes the shuttle's motion. Between
hits the shuttle follows a smooth, decelerating arc. At a hit, two things
happen together:

1. its direction of travel turns sharply (turn angle), and
2. it leaves faster than it was flying around that moment (the racket
   injects energy), measured relative to the local flight speed.

Turns without a speed-up (the apex of a high shot, the shuttle landing or
being handled before the serve, tracking drift) are rejected. The first
strong launch in the clip is the serve; turns before it are ignored.

Every threshold is relative (angles, speed ratios) or in seconds; see
``config.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DetectorConfig, frames
from .track import step_lengths, typical_step

FEATURE_COLUMNS = [
    "turn_deg", "speed_in", "speed_out", "local_speed", "out_rel", "in_rel", "follow_rel",
]


def trajectory_features(track: pd.DataFrame, fps: float, cfg: DetectorConfig) -> pd.DataFrame:
    """Per-frame turn angle and speeds of a cleaned track.

    For frame i with half-window k:
      v_in  = p[i]   - p[i-k]
      v_out = p[i+k] - p[i]
      turn_deg  = angle between v_in and v_out (0 = straight on, 180 = reversal)
      speed_*   = |v| / k  in pixels per frame
      out_rel   = speed_out / local median flight speed
                  (max over the frame and the next ``contact_slack_s``)
      follow_rel = median speed over the next ``follow_s`` / clip's typical speed
    """
    k = frames(cfg.turn_half_window_s, fps)
    xy = track[["x", "y"]].to_numpy()
    n = len(xy)

    v_in = np.full_like(xy, np.nan)
    v_out = np.full_like(xy, np.nan)
    v_in[k:] = xy[k:] - xy[:-k]
    v_out[:-k] = xy[k:] - xy[:-k]

    s_in = np.linalg.norm(v_in, axis=1)
    s_out = np.linalg.norm(v_out, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", v_in, v_out) / (s_in * s_out)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    win = frames(cfg.local_speed_window_s, fps) | 1
    local = (step_lengths(track).rolling(win, center=True, min_periods=5)
             .median().to_numpy())

    feats = pd.DataFrame(
        {
            "turn_deg": turn,
            "speed_in": s_in / k,
            "speed_out": s_out / k,
            "local_speed": local,
        },
        index=track.index,
    )
    out_rel = feats["speed_out"] / feats["local_speed"]
    slack = frames(cfg.contact_slack_s, fps, minimum=0)
    feats["out_rel"] = out_rel[::-1].rolling(slack + 1, min_periods=1).max()[::-1]
    feats["in_rel"] = feats["speed_in"] / feats["local_speed"]

    follow = frames(cfg.follow_s, fps)
    step = step_lengths(track)
    # median of steps i+1 .. i+follow (forward-looking window)
    fwd = step[::-1].rolling(follow, min_periods=3).median()[::-1].shift(-1)
    feats["follow_rel"] = fwd / typical_step(track)
    assert len(feats) == n
    return feats


def _is_hit(f: pd.DataFrame, cfg: DetectorConfig) -> pd.Series:
    sharp = (f["turn_deg"] >= cfg.min_turn_deg) & (f["out_rel"] >= cfg.min_out_speed_rel)
    driven = (f["turn_deg"] >= cfg.weak_turn_deg) & (f["out_rel"] >= cfg.strong_out_speed_rel)
    keeps_flying = f["follow_rel"] >= cfg.min_follow_speed_rel
    return (sharp | driven) & keeps_flying


def _score(f: pd.DataFrame) -> pd.Series:
    """Confidence used to choose between nearby candidates (0..1)."""
    return (f["turn_deg"] / 180.0) * (f["out_rel"].clip(upper=4.0) / 4.0)


def detect_hits(
    track: pd.DataFrame, fps: float, cfg: DetectorConfig | None = None,
    return_features: bool = False,
):
    """Detect hit frames in a cleaned track.

    Returns a DataFrame with columns
    ``frame, time_s, turn_deg, out_rel, score`` sorted by frame
    (and the per-frame feature table if ``return_features``).
    """
    cfg = cfg or DetectorConfig()
    f = trajectory_features(track, fps, cfg)

    # # 1. Candidates: local maxima of the turn angle.
    # peak_win = frames(cfg.peak_window_s, fps) | 1
    # local_max = f["turn_deg"].rolling(peak_win, center=True, min_periods=1).max()
    # cand = f[(f["turn_deg"] == local_max) & f["turn_deg"].notna()].copy()

    # # 2. Keep candidates that look like a racket hit.
    # cand = cand[_is_hit(cand, cfg)]
    # cand["score"] = _score(cand)

    # # 3. Rally start: the serve is the first strong launch.
    # launches = cand.index[cand["out_rel"] >= cfg.serve_out_speed_rel]
    # if len(launches):
    #     cand = cand[cand.index >= launches[0]]

        # 1-2. Candidates that look like a racket hit:
    #   a) the sharpest turn in its neighbourhood, with a speed-up, or
    #   b) the sharpest speed-up in its neighbourhood, even without a turn.
    peak_win = frames(cfg.peak_window_s, fps) | 1

    def is_peak(s: pd.Series) -> pd.Series:
        return s.notna() & (s == s.rolling(peak_win, center=True, min_periods=1).max())

    accel = f["speed_out"] / f["speed_in"]
    turn_hit = is_peak(f["turn_deg"]) & _is_hit(f, cfg)
    speedup_hit = (is_peak(accel) & (accel >= cfg.accel_ratio)
                   & (f["out_rel"] >= cfg.strong_out_speed_rel)
                   & (f["follow_rel"] >= cfg.min_follow_speed_rel))
    cand = f[turn_hit | speedup_hit].copy()
    cand["score"] = _score(cand).fillna(0.0)

    # 3. Rally start: the first moment the shuttle flies at its typical
    #    speed. Turns before that are handling before the serve.
    if cfg.rally_start_window_s > 0:
        w = frames(cfg.rally_start_window_s, fps)
        speed = step_lengths(track).rolling(w, min_periods=3).median()
        flying = speed.index[speed >= typical_step(track)]
        if len(flying):
            cand = cand[cand.index >= flying[0] - w]

    # 4. Temporal non-maximum suppression: at most one hit per min_hit_gap_s.
    gap = frames(cfg.min_hit_gap_s, fps)
    kept: list[int] = []
    for frame in cand.sort_values("score", ascending=False).index:
        if all(abs(frame - k) >= gap for k in kept):
            kept.append(frame)

    hits = cand.loc[sorted(kept), ["turn_deg", "out_rel", "score"]].copy()
    hits.insert(0, "time_s", hits.index / fps)
    hits = hits.reset_index().rename(columns={"index": "frame"})
    hits["frame"] = hits["frame"].astype(int)
    return (hits, f) if return_features else hits
