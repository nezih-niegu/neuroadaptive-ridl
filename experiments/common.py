"""Shared data loading for the Frontiers extension experiments.

Reuses the thesis conventions (data_processing.load_images, 0.1 s timestamp
rounding, 420 s trim, state = {Flow, Optimality}, action = 9-D pose+force).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ridl.data_processing import load_images  # noqa: E402

# Raw data (not distributed): <RIDL_DATA>/{LGAF,Stream,ROS,NIC}
DATA = os.environ.get("RIDL_DATA", os.path.join(ROOT, "data", "VR-NVR"))
CACHE = os.environ.get("RIDL_CACHE", os.path.join(ROOT, "cache", "unet64"))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")

SUBJECTS = list(range(10))
CONDS = ["VR", "NVR"]
SEGMENTS = ["Reference", "Task1", "Task2"]
ACT = ["x", "y", "z", "roll", "pitch", "yaw", "fx", "fy", "fz"]
JERK_SIGNALS = ["y", "fx", "fy", "fz", "roll", "pitch", "yaw"]  # thesis Sec. 4.5


def hand(s):
    return "LH" if s == 4 else "RH"


def seg_name(s, cond, seg):
    return f"subject{s}-{hand(s)}-{cond}_{seg}"


def stream_dir(s, cond):
    return os.path.join(DATA, "Stream", f"subject{s}-{hand(s)}-{cond}")


def load_lgaf(s, cond, seg):
    return pd.read_csv(os.path.join(DATA, "LGAF", seg_name(s, cond, seg) + ".csv"))


def optimality(df):
    """Thesis jerk-based optimality O_t in [0,1] (Sec. 4.5)."""
    sig = df[JERK_SIGNALS].to_numpy(float)
    cj = np.abs(np.diff(sig, axis=0)).mean(1)
    rng = cj.max() - cj.min()
    norm = np.zeros_like(cj) if np.isclose(rng, 0) else (cj - cj.min()) / rng
    return np.concatenate([[1.0], 1 - norm])


def merged_segment(s, cond, seg, trim=420.0):
    """Kinematics + flow merged with endoscope frames (thesis testing pipeline)."""
    d = load_lgaf(s, cond, seg)
    img = load_images(stream_dir(s, cond))
    d["timestamp"] = d["timestamp"].astype(float).round(1)
    img["timestamp"] = img["timestamp"].astype(float).round(1)
    m = (d.merge(img, on="timestamp", how="left")
          .dropna(subset=["image_path"] + ACT)
          .drop_duplicates("timestamp")
          .reset_index(drop=True))
    m = m[m.timestamp <= m.timestamp.iloc[0] + trim].reset_index(drop=True)
    m["Flow"] = m["F"].to_numpy(float)
    m["Optimality"] = optimality(m)
    return m
