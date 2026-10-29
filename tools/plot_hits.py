"""Plot the shuttle track with detected and labelled hits (one PNG per clip).

    python -m tools.plot_hits --track-dir outputs/tracknet --out outputs/plots

Top panel: shuttle height in the image over time, raw vs cleaned track.
Bottom panel: turn angle and outgoing speed ratio, the two signals the
detector thresholds. Vertical lines: labels (green) and detections (red).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.ticker import MultipleLocator 
from hitdet.config import DetectorConfig  # noqa: E402
from hitdet.hits import detect_hits  # noqa: E402
from hitdet.labels import load_clip_labels  # noqa: E402
from hitdet.track import clean_track, load_tracknet_csv  # noqa: E402
from hitdet.tracknet import TRACK_SUFFIX  # noqa: E402


def plot_clip(clip: str, track_csv: Path, labels: Path, fps: float, cfg: DetectorConfig, out: Path) -> None:
    raw = load_tracknet_csv(track_csv)
    clean = clean_track(raw, fps, cfg)
    hits, feats = detect_hits(clean, fps, cfg, return_features=True)
    gt = sorted(load_clip_labels(labels, clip)) if labels.exists() else []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(raw.index, -raw["y"], ".", ms=3, color="0.75", label="TrackNet raw")
    ax1.plot(clean.index, -clean["y"], "-", lw=1.4, color="#1f5fa8", label="cleaned")
    ax1.set_ylabel("shuttle height in image (px)")
    ax1.set_title(f"{clip}: {len(hits)} hits detected, {len(gt)} labelled")

    ax2.plot(feats.index, feats["turn_deg"], lw=1, color="#7a4fb0", label="turn angle (deg)")
    ax2b = ax2.twinx()
    ax2b.plot(feats.index, feats["out_rel"], lw=1, color="#d08a1e", label="out speed / local speed")
    ax2b.axhline(cfg.min_out_speed_rel, ls=":", color="#d08a1e", lw=1)
    ax2.axhline(cfg.min_turn_deg, ls=":", color="#7a4fb0", lw=1)
    ax2.set_ylabel("turn angle (deg)")
    ax2b.set_ylabel("speed ratio")
    ax2b.set_ylim(0, 6)
    ax2.set_xlabel("frame")
    ax2.xaxis.set_major_locator(MultipleLocator(20))

    for ax in (ax1, ax2):
        for f in gt:
            ax.axvline(f, color="#2e9e4f", lw=2.5, alpha=0.35)
        for f in hits["frame"]:
            ax.axvline(f, color="#c8372d", lw=1, ls="--")
    ax1.plot([], [], color="#2e9e4f", lw=2.5, alpha=0.5, label="labelled hit")
    ax1.plot([], [], color="#c8372d", lw=1, ls="--", label="detected hit")
    ax1.legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{clip}_hits.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track-dir", type=Path, default=Path("outputs/tracknet"))
    p.add_argument("--labels", type=Path, default=Path("labels/hits_gt.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/plots"))
    p.add_argument("--fps", type=float, default=29.97)
    args = p.parse_args()
    cfg = DetectorConfig()
    for csv in sorted(args.track_dir.glob(f"*{TRACK_SUFFIX}")):
        clip = csv.name[: -len(TRACK_SUFFIX)]
        plot_clip(clip, csv, args.labels, args.fps, cfg, args.out)
        print(f"saved {args.out / f'{clip}_hits.png'}")


if __name__ == "__main__":
    main()
