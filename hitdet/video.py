"""Frame-accurate random access to a video file.

Frame indices are 0-based and follow decode order, which matches the
``Frame`` column written by TrackNetV3's ``predict.py``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata of a video file."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class FrameReader:
    """Random-access frame reader with a small LRU cache.

    Stepping forward by one frame reads the next frame directly, which is
    fast and exact. Any other jump seeks with ``CAP_PROP_POS_FRAMES``.
    Recently viewed frames are cached so stepping backwards is instant.
    """

    def __init__(self, path: str | Path, cache_size: int = 64) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.info = VideoInfo(
            path=self.path,
            fps=fps if fps and fps > 0 else 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        # Containers sometimes over-report the frame count; this is lowered
        # if decoding fails near the end.
        self.frame_count = self.info.frame_count

        self._next_idx = 0  # index of the frame the next cap.read() returns
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def get(self, idx: int) -> np.ndarray:
        """Return frame ``idx`` as a BGR image."""
        if not 0 <= idx < self.frame_count:
            raise IndexError(f"Frame {idx} out of range [0, {self.frame_count})")

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        if idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame = self._cap.read()
        if not ok:
            # Treat a failed decode as the true end of the video.
            self.frame_count = idx
            self._next_idx = idx
            raise IndexError(f"Could not decode frame {idx}; video ends at {idx - 1}")

        self._next_idx = idx + 1
        self._cache[idx] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
