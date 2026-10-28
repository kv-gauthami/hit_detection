"""Detect shuttle hit frames in badminton rally clips.

Video in, CSV of hit frames out:

    python detect_hits.py data/videos/                 # every .mp4 in a folder
    python detect_hits.py clip1.mp4 clip2.mp4 -o outputs/hits.csv

Re-use existing TrackNet tracks (skips the slow tracking step):

    python detect_hits.py data/videos/ --track-dir outputs/tracknet

Output CSV columns:
    clip      video file name without extension
    hit_no    1-based index of the hit within the rally (1 = serve)
    frame     0-based frame index in the original video
    time_s    frame / fps
    turn_deg, out_rel, score   evidence for the hit (see hitdet/hits.py)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from hitdet.config import DetectorConfig
from hitdet.hits import detect_hits
from hitdet.track import clean_track, load_tracknet_csv
from hitdet.tracknet import run_tracknet, track_csv_path
from hitdet.video import FrameReader

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def find_videos(inputs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for p in inputs:
        if p.is_dir():
            videos += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        else:
            raise ValueError(f"Not a video or folder: {p}")
    if not videos:
        raise SystemExit("No videos found.")
    return videos


def process(video: Path, args: argparse.Namespace, cfg: DetectorConfig) -> pd.DataFrame:
    with FrameReader(video) as reader:
        fps = reader.info.fps

    t0 = time.perf_counter()
    if args.track_dir and track_csv_path(args.track_dir, video.stem).exists():
        track_csv = track_csv_path(args.track_dir, video.stem)
    else:
        track_csv = run_tracknet(video, args.work_dir, args.tracknet_dir, args.batch_size)
    t1 = time.perf_counter()

    # track = clean_track(load_tracknet_csv(track_csv), fps, cfg)
    raw = load_tracknet_csv(track_csv)
    coverage = raw["x"].notna().mean()
    if coverage < cfg.min_track_coverage:
        print(f"\nWARNING {video.name}: shuttle tracked in only {coverage:.0%} of frames,\n "
              "\nso hits will be missed. Likely causes: low resolution, heavy compression \n"
              "\nor a camera angle unlike TrackNet's training data.\n")
    track = clean_track(raw, fps, cfg)
    hits = detect_hits(track, fps, cfg)
    t2 = time.perf_counter()

    hits.insert(0, "hit_no", range(1, len(hits) + 1))
    hits.insert(0, "clip", video.stem)
    print(f"{video.name}: {len(hits)} hits | tracking {t1 - t0:.1f}s, "
          f"hit detection {(t2 - t1) * 1000:.0f} ms | frames {hits['frame'].tolist()}")
    return hits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="video files and/or folders of videos")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/hits.csv"))
    p.add_argument("--track-dir", type=Path, help="folder with existing <clip>_shuttle.csv files to reuse")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/tracknet"),
                   help="where new TrackNet tracks are written")
    p.add_argument("--tracknet-dir", type=Path, default=Path("third_party/TrackNetV3"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--config", type=Path, help="optional JSON file overriding DetectorConfig fields")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DetectorConfig(**json.loads(args.config.read_text())) if args.config else DetectorConfig()

    results = [process(v, args, cfg) for v in find_videos(args.inputs)]
    out = pd.concat(results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.round({"time_s": 3, "turn_deg": 1, "out_rel": 2, "score": 3}).to_csv(args.output, index=False)
    print(f"Wrote {len(out)} hits from {len(results)} clip(s) to {args.output}")


if __name__ == "__main__":
    main()
