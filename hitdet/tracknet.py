"""Thin wrapper around TrackNetV3's ``predict.py`` (shuttle tracking).

TrackNetV3 (https://github.com/qaz812345/TrackNetV3) is used unchanged
with its published weights. This module:

1. makes a 1280x720 copy of the clip (TrackNet resizes to 512x288
   internally, so full HD only costs memory; on our clips 720p and 1080p
   tracks were identical), keeping every frame so frame numbers match
   the original video;
2. runs ``predict.py`` in streaming mode (``--large_video``), which keeps
   memory constant regardless of clip length;
3. saves the track as ``<clip>_shuttle.csv`` and returns its path
   (TrackNet itself names it ``*_ball.csv``, a leftover from tennis).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2

TRACK_HEIGHT = 720
TRACK_SUFFIX = "_shuttle.csv"


def track_csv_path(track_dir: Path, clip: str) -> Path:
    """Where the shuttle track of ``clip`` is stored."""
    return Path(track_dir) / f"{clip}{TRACK_SUFFIX}"


def _resize_with_ffmpeg(src: Path, dst: Path, height: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale=-2:{height}", "-an", str(dst)],
        check=True,
    )


def _resize_with_opencv(src: Path, dst: Path, height: int) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (int(round(w * height / h / 2)) * 2, height)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    out.release()


def prepare_video(video: Path, work_dir: Path, height: int = TRACK_HEIGHT) -> Path:
    """Return a copy of ``video`` scaled to ``height`` (or the original if already small)."""
    cap = cv2.VideoCapture(str(video))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_h <= height:
        return video

    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / f"{video.stem}.mp4"
    if not dst.exists():
        if shutil.which("ffmpeg"):
            _resize_with_ffmpeg(video, dst, height)
        else:
            _resize_with_opencv(video, dst, height)
    return dst


def run_tracknet(
    video: Path,
    out_dir: Path,
    tracknet_dir: Path,
    batch_size: int = 4,
    overwrite: bool = False,
) -> Path:
    """Track the shuttle in ``video`` and return the TrackNet CSV path.

    Reuses an existing CSV unless ``overwrite`` is set.
    """
    video, out_dir, tracknet_dir = Path(video), Path(out_dir), Path(tracknet_dir).resolve()
    csv_path = track_csv_path(out_dir, video.stem)
    if csv_path.exists() and not overwrite:
        return csv_path

    ckpt = tracknet_dir / "ckpts"
    for name in ("TrackNet_best.pt", "InpaintNet_best.pt", ):
        if not (ckpt / name).exists():
            raise FileNotFoundError(f"Missing TrackNetV3 weights: {ckpt / name} (see README)")

    small = prepare_video(video, out_dir / "_resized").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "predict.py",
        "--video_file", str(small),
        "--tracknet_file", "ckpts/TrackNet_best.pt",
        "--inpaintnet_file", "ckpts/InpaintNet_best.pt",
        "--save_dir", str(out_dir.resolve()),
        "--large_video",
        "--batch_size", str(batch_size),
    ]
    subprocess.run(cmd, cwd=tracknet_dir, check=True)

    produced = out_dir / f"{small.stem}_ball.csv"  # TrackNet's own output name
    if produced != csv_path:
        produced.replace(csv_path)
    return csv_path
