"""Evaluate detected hits against hand labels.

    python -m tools.evaluate                                  # uses outputs/hits.csv
    python -m tools.evaluate --tol 1 2 3                      # several tolerances
    python -m tools.evaluate --robustness --track-dir outputs/tracknet

``--robustness`` re-runs the detector from the cached tracks with each
threshold varied over a wide range and with each component removed
(ablation), to show the result does not hinge on tuned values.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from hitdet.config import DetectorConfig
from hitdet.evaluate import MatchResult, match_events
from hitdet.hits import detect_hits
from hitdet.labels import load_clip_labels
from hitdet.track import clean_track, load_tracknet_csv
from hitdet.tracknet import track_csv_path

SWEEPS = {
    "min_turn_deg": [40, 50, 60, 70, 80, 90],
    "min_out_speed_rel": [0.7, 0.8, 0.9, 1.0, 1.2, 1.4],
    "strong_out_speed_rel": [1.5, 2.0, 2.5, 3.0],
    "min_follow_speed_rel": [0.3, 0.4, 0.5, 0.6, 0.7],
    "outlier_k": [4, 5, 6, 8, 10],
    "static_range_rel": [1.2, 1.5, 1.8, 2.2, 2.6],
    "min_hit_gap_s": [0.2, 0.25, 0.3, 0.35],
    "accel_ratio": [2.0, 2.5, 3.0, 4.0],
}


def ablations(base: DetectorConfig) -> dict[str, tuple[DetectorConfig, bool]]:
    """name -> (config, clean_track?)"""
    return {
        "full method": (base, True),
        "no track cleaning": (base, False),
        "turn angle only (no speed rules)": (replace(
            base, min_out_speed_rel=0, strong_out_speed_rel=0, weak_turn_deg=999,
            min_follow_speed_rel=0, accel_ratio=float("inf")), True),
        "no follow-through rule": (replace(base, min_follow_speed_rel=0), True),
        "no contact slack": (replace(base, contact_slack_s=0), True),
        # "no serve gate": (replace(base, serve_out_speed_rel=0), True),
        "no rally-start rule": (replace(base, rally_start_window_s=0), True),
        "no speed-up branch": (replace(base, accel_ratio=float("inf")), True),
        "no weak-turn branch": (replace(base, weak_turn_deg=999), True),
    }


def evaluate_table(pred: pd.DataFrame, labels: Path, clips: list[str], tol: int) -> tuple[pd.DataFrame, MatchResult]:
    rows, total = [], None
    for clip in clips:
        gt = sorted(load_clip_labels(labels, clip))
        r = match_events(pred.loc[pred["clip"] == clip, "frame"].tolist(), gt, tol)
        total = r if total is None else total + r
        rows.append(_row(clip, r))
    rows.append(_row("ALL", total))
    return pd.DataFrame(rows), total


def _row(name: str, r: MatchResult) -> dict:
    return {
        "clip": name, "hits": r.tp + r.fn, "detected": r.tp + r.fp, "TP": r.tp, "FP": r.fp,
        "FN": r.fn, "precision": round(r.precision, 3), "recall": round(r.recall, 3),
        "F1": round(r.f1, 3), "mean_abs_err_frames": round(r.mean_abs_error, 2),
        "false_pos": r.false_pos, "missed": r.missed,
    }


def run_from_tracks(track_dir: Path, labels: Path, clips: list[str], fps: dict[str, float],
                    cfg: DetectorConfig, clean: bool, tol: int) -> MatchResult:
    total = None
    for clip in clips:
        track = load_tracknet_csv(track_csv_path(track_dir, clip))
        if clean:
            track = clean_track(track, fps[clip], cfg)
        hits = detect_hits(track, fps[clip], cfg)
        r = match_events(hits["frame"].tolist(), sorted(load_clip_labels(labels, clip)), tol)
        total = r if total is None else total + r
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, default=Path("outputs/hits.csv"))
    p.add_argument("--labels", type=Path, default=Path("labels/hits_gt.csv"))
    p.add_argument("--tol", type=int, nargs="+", default=[2], help="match tolerance in frames")
    p.add_argument("--robustness", action="store_true", help="run sensitivity sweep and ablation")
    p.add_argument("--track-dir", type=Path, default=Path("outputs/tracknet"))
    p.add_argument("--fps", type=float, default=29.97, help="fps used for --robustness (clips are ~30 fps)")
    args = p.parse_args()

    clips = sorted(pd.read_csv(args.labels)["clip"].unique())
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)

    fps = {c: args.fps for c in clips}
    if args.pred.exists():
        pred = pd.read_csv(args.pred)
        # Score only the clips this prediction file covers.
        clips = [c for c in clips if c in set(pred["clip"])]
        # Each clip's fps, recovered from its frame / time_s columns.
        timed = pred[pred["time_s"] > 0]
        fps.update((timed["frame"] / timed["time_s"]).groupby(timed["clip"]).median().to_dict())
        for tol in args.tol:
            table, _ = evaluate_table(pred, args.labels, clips, tol)
            print(f"\n=== Detection vs labels, tolerance +/-{tol} frames ===")
            print(table.to_string(index=False))
    else:
        print(f"{args.pred} not found; run detect_hits.py first (skipping main table)")

    if args.robustness:
        base, tol = DetectorConfig(), args.tol[0]
        # fps = {c: args.fps for c in clips}
        print(f"\n=== Sensitivity: F1 when one threshold is changed (default in brackets), tol +/-{tol} ===")
        for name, values in SWEEPS.items():
            default = getattr(base, name)
            cells = []
            for v in values:
                f1 = run_from_tracks(args.track_dir, args.labels, clips, fps,
                                     replace(base, **{name: v}), True, tol).f1
                cells.append(f"[{v}]={f1:.2f}" if v == default else f"{v}={f1:.2f}")
            print(f"{name:22s} " + "  ".join(cells))

        print(f"\n=== Ablation, tol +/-{tol} ===")
        rows = [_row(name, run_from_tracks(args.track_dir, args.labels, clips, fps, cfg, clean, tol))
                for name, (cfg, clean) in ablations(base).items()]
        print(pd.DataFrame(rows).drop(columns=["false_pos", "missed"]).to_string(index=False))


if __name__ == "__main__":
    main()
