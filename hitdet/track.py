"""Loading shuttle tracks produced by TrackNetV3.

TrackNetV3 writes one row per frame with columns
``Frame, Visibility, X, Y``. Invisible frames have X = Y = 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_tracknet_csv(path: str | Path) -> pd.DataFrame:
    """Load a TrackNetV3 ``*_ball.csv`` file.

    Returns a DataFrame indexed by ``frame`` with columns
    ``visible`` (bool), ``x`` and ``y`` (float pixels, NaN when invisible).
    """
    df = pd.read_csv(path)
    expected = {"Frame", "Visibility", "X", "Y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "visible": df["Visibility"].astype(int) == 1,
            "x": df["X"].astype(float),
            "y": df["Y"].astype(float),
        },
        index=pd.Index(df["Frame"].astype(int), name="frame"),
    )
    out.loc[~out["visible"], ["x", "y"]] = np.nan
    return out
