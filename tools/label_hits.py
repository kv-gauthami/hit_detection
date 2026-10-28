"""Interactive tool for labelling shuttle hit frames in a rally clip.

Usage (from the repo root):

    python -m tools.label_hits data/videos/v03_720.mp4 \
        --track outputs/tracknet/v03_720_ball.csv

Labels are saved automatically after every change to labels/hits_gt.csv.

Keys
----
  ->  / d        next frame            <-  / a        previous frame
  up  / w        forward 10 frames     down / s       back 10 frames
  n / p          next / previous candidate
  ] / [          next / previous labelled hit
  space          play / pause
  1              mark hit by NEAR player
  2              mark hit by FAR player
  h              mark hit (hitter unknown), or remove an existing mark
  x / backspace  remove mark on this frame
  q / esc        quit
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hitdet.labels import load_candidates, load_clip_labels, save_clip_labels
from hitdet.track import load_tracknet_csv
from hitdet.video import FrameReader

WINDOW = "label_hits"
TRAIL_LEN = 12  # frames of shuttle trail drawn behind the current position

# BGR colours
WHITE, YELLOW, RED, GREEN = (255, 255, 255), (0, 215, 255), (60, 60, 230), (90, 200, 90)
HITTER_COLOURS = {"near": RED, "far": (230, 120, 50), "unknown": (200, 80, 200)}

# Arrow-key codes returned by cv2.waitKeyEx differ by OS (macOS, Linux, Windows).
KEY_RIGHT = {63235, 65363, 2555904, ord("d")}
KEY_LEFT = {63234, 65361, 2424832, ord("a")}
KEY_UP = {63232, 65362, 2490368, ord("w")}
KEY_DOWN = {63233, 65364, 2621440, ord("s")}
KEY_DELETE = {ord("x"), 8, 127}
KEY_QUIT = {ord("q"), 27}


def _next_in(items: list[int], current: int) -> int | None:
    i = bisect_right(items, current)
    return items[i] if i < len(items) else None


def _prev_in(items: list[int], current: int) -> int | None:
    i = bisect_left(items, current)
    return items[i - 1] if i > 0 else None


class HitLabeller:
    """Holds the labelling state and draws the annotated view."""

    def __init__(
        self,
        reader: FrameReader,
        clip: str,
        labels_path: Path,
        candidates: list[int],
        track: pd.DataFrame | None,
        display_width: int,
    ) -> None:
        self.reader = reader
        self.clip = clip
        self.labels_path = labels_path
        self.candidates = candidates
        self.track = track
        self.labels = load_clip_labels(labels_path, clip)
        self.idx = 0
        self.playing = False
        self.scale = min(1.0, display_width / reader.info.width)

    # ---- state changes -------------------------------------------------
    def goto(self, idx: int | None) -> None:
        if idx is not None:
            self.idx = int(np.clip(idx, 0, self.reader.frame_count - 1))

    def mark(self, hitter: str) -> None:
        self.labels[self.idx] = hitter
        self._save(f"marked frame {self.idx} ({hitter})")

    def unmark(self) -> None:
        if self.labels.pop(self.idx, None) is not None:
            self._save(f"removed frame {self.idx}")

    def toggle(self) -> None:
        self.unmark() if self.idx in self.labels else self.mark("unknown")

    def _save(self, msg: str) -> None:
        save_clip_labels(self.labels_path, self.clip, self.labels)
        print(f"[{self.clip}] {msg} -> {len(self.labels)} hits saved")

    # ---- drawing -------------------------------------------------------
    def render(self, frame: np.ndarray) -> np.ndarray:
        view = cv2.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale < 1 else frame.copy()
        if self.track is not None:
            self._draw_trail(view)
        self._draw_panel(view)
        self._draw_timeline(view)
        return view

    def _draw_trail(self, view: np.ndarray) -> None:
        window = self.track.loc[max(0, self.idx - TRAIL_LEN): self.idx].dropna()
        pts = (window[["x", "y"]].to_numpy() * self.scale).astype(int)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            cv2.line(view, (x0, y0), (x1, y1), YELLOW, 1, cv2.LINE_AA)
        if len(window) and window.index[-1] == self.idx:
            cv2.circle(view, tuple(pts[-1]), 6, YELLOW, 2, cv2.LINE_AA)

    def _draw_panel(self, view: np.ndarray) -> None:
        fps = self.reader.info.fps
        lines = [
            (f"{self.clip}   frame {self.idx} / {self.reader.frame_count - 1}"
             f"   t={self.idx / fps:.2f}s", WHITE),
            (f"hits labelled: {len(self.labels)}   candidates: {len(self.candidates)}", WHITE),
        ]
        if self.idx in self.labels:
            hitter = self.labels[self.idx]
            lines.append((f"HIT ({hitter})", HITTER_COLOURS[hitter]))
        if self.idx in self.candidates:
            lines.append(("candidate", YELLOW))
        if self.playing:
            lines.append(("playing", GREEN))

        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (560, 18 + 26 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)
        for i, (text, colour) in enumerate(lines):
            cv2.putText(view, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 1, cv2.LINE_AA)

    def _draw_timeline(self, view: np.ndarray) -> None:
        h, w = view.shape[:2]
        top, bottom = h - 22, h - 6
        cv2.rectangle(view, (0, top - 4), (w, h), (0, 0, 0), -1)
        n = max(1, self.reader.frame_count - 1)
        to_x = lambda f: int(f / n * (w - 1))  # noqa: E731
        for f in self.candidates:
            cv2.line(view, (to_x(f), top), (to_x(f), bottom), YELLOW, 1)
        for f, hitter in self.labels.items():
            cv2.line(view, (to_x(f), top), (to_x(f), bottom), HITTER_COLOURS[hitter], 3)
        cv2.line(view, (to_x(self.idx), top - 4), (to_x(self.idx), h), WHITE, 2)

    # ---- main loop -----------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        delay = max(1, int(1000 / self.reader.info.fps))
        while True:
            try:
                frame = self.reader.get(self.idx)
            except IndexError:  # container over-reported the frame count
                self.goto(self.reader.frame_count - 1)
                self.playing = False
                continue

            cv2.imshow(WINDOW, self.render(frame))
            key = cv2.waitKeyEx(delay if self.playing else 0)

            if key == -1:  # no key pressed while playing
                if self.idx >= self.reader.frame_count - 1:
                    self.playing = False
                self.goto(self.idx + 1)
                continue

            self.playing = False if key != ord(" ") else not self.playing
            if key in KEY_QUIT:
                break
            elif key in KEY_RIGHT:
                self.goto(self.idx + 1)
            elif key in KEY_LEFT:
                self.goto(self.idx - 1)
            elif key in KEY_UP:
                self.goto(self.idx + 10)
            elif key in KEY_DOWN:
                self.goto(self.idx - 10)
            elif key == ord("n"):
                self.goto(_next_in(self.candidates, self.idx))
            elif key == ord("p"):
                self.goto(_prev_in(self.candidates, self.idx))
            elif key == ord("]"):
                self.goto(_next_in(sorted(self.labels), self.idx))
            elif key == ord("["):
                self.goto(_prev_in(sorted(self.labels), self.idx))
            elif key == ord("1"):
                self.mark("near")
            elif key == ord("2"):
                self.mark("far")
            elif key == ord("h"):
                self.toggle()
            elif key in KEY_DELETE:
                self.unmark()

        cv2.destroyAllWindows()
        print(f"[{self.clip}] done: {len(self.labels)} hits in {self.labels_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label shuttle hit frames in a rally clip.")
    p.add_argument("video", type=Path, help="video to label (use the same file TrackNet ran on)")
    p.add_argument("--clip", help="clip name used in the CSVs (default: video file name)")
    p.add_argument("--labels", type=Path, default=Path("labels/hits_gt.csv"),
                   help="ground-truth file, shared across clips")
    p.add_argument("--candidates", type=Path, default=Path("labels/candidates.csv"),
                   help="optional candidate frames to jump between with n/p")
    p.add_argument("--track", type=Path, help="optional TrackNet *_ball.csv to draw the shuttle trail")
    p.add_argument("--display-width", type=int, default=1280, help="max window width in pixels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip = args.clip or args.video.stem
    candidates = load_candidates(args.candidates, clip)
    track = load_tracknet_csv(args.track) if args.track else None

    with FrameReader(args.video) as reader:
        print(f"[{clip}] {reader.frame_count} frames at {reader.info.fps:.2f} fps, "
              f"{len(candidates)} candidates")
        HitLabeller(reader, clip, args.labels, candidates, track, args.display_width).run()


if __name__ == "__main__":
    main()
