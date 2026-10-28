"""All tunable parameters of the hit detector, in one place.

Design rule: no parameter is in raw pixels or raw frames.
  * Durations are in seconds and converted with the clip's own fps.
  * Distances and speeds are relative to the shuttle's own speed in the
    clip (global median step, or local median speed around a frame), so
    the same values work for any resolution, zoom or camera distance.
  * Angles are unitless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DetectorConfig:
    # ---- track cleaning ------------------------------------------------
    startup_trim_s: float = 0.1       # TrackNet's first frames are a start-up artifact (same coords in every video)
    outlier_window_s: float = 0.25    # rolling-median window for spike detection
    outlier_k: float = 6.0            # spike if deviation > k * speed + outlier_floor_rel * speed
    outlier_floor_rel: float = 1.2    # floor so slow segments are not over-pruned
    max_gap_fill_s: float = 0.10      # bridge gaps up to this long, only if endpoints agree
    static_window_s: float = 0.5      # a detection is static if it barely moves over this window
    static_range_rel: float = 1.8     # ... i.e. its range is below this * typical step
    min_fragment_s: float = 0.25      # drop isolated track fragments shorter than this

    # ---- hit candidates ------------------------------------------------
    turn_half_window_s: float = 0.067     # direction measured over +/- this (2 frames at 30 fps)
    local_speed_window_s: float = 1.0     # window for the local flight-speed reference
    peak_window_s: float = 0.2            # a candidate must be the sharpest turn in this window

    # ---- hit decision --------------------------------------------------
    # A racket hit = sharp direction change AND energy injected
    # (outgoing speed at least the local flight speed). Turns without a
    # speed-up are apexes, landings, drift or the shuttle in the hand.
    min_turn_deg: float = 60.0
    min_out_speed_rel: float = 1.0
    # A gentler turn is still a hit if the speed-up is strong (e.g. drives
    # and net kills that continue roughly the same way).
    weak_turn_deg: float = 35.0
    strong_out_speed_rel: float = 2.0
    # Contact usually falls between two frames at 30 fps, so the speed-up
    # may only show one frame after the sharpest turn.
    contact_slack_s: float = 0.034
    # After a hit the shuttle keeps flying: its median speed over the next
    # ``follow_s`` must be at least this fraction of the clip's typical
    # speed. Rejects landings, where the shuttle stops on the floor.
    follow_s: float = 0.3
    min_follow_speed_rel: float = 0.5
    # A hit seen along the camera's line of sight (e.g. from a high broadcast
    # camera) barely changes direction on screen, only speed. Such a frame is
    # also a hit if the shuttle leaves this many times faster than it arrived
    # (and out_rel >= strong_out_speed_rel, and it keeps flying).
    accel_ratio: float = 2.5

    # ---- rally structure -----------------------------------------------
    # serve_out_speed_rel: float = 2.0  # the serve is the first strong launch; earlier turns are pre-serve handling
    # The rally starts when the shuttle first flies at its typical speed for
    # this long; turns before that are handling before the serve (0 = off).
    rally_start_window_s: float = 0.2
    min_hit_gap_s: float = 0.30       # two hits can't be closer than this (fastest exchanges ~0.4 s)
    min_track_coverage: float = 0.3   # warn if the shuttle is tracked in fewer frames than this

    def to_dict(self) -> dict:
        return asdict(self)


def frames(seconds: float, fps: float, minimum: int = 1) -> int:
    """Convert a duration to a whole number of frames for this clip."""
    return max(minimum, int(round(seconds * fps)))
