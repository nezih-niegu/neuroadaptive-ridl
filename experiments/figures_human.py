"""Human-data figures (thesis visual language: RGB flow/optimality encoding)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from common import FIGS, RESULTS, load_lgaf, optimality

BLUE, ORANGE, GRAY, INK, INK2 = "#257de0", "#f47230", "#9a9a9a", "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def rgb_traj(ax, d, title):
    """Thesis Fig. 4.6 encoding: colour = (1-F, F, O), F min-max scaled per panel."""
    p = d[["x", "y", "z"]].to_numpy(float) * 1000
    F = d.F.to_numpy(float)
    Fn = (np.argsort(np.argsort(F)) + 0.5) / len(F)      # within-segment percentile of CFS
    O = optimality(d)
    O = (np.argsort(np.argsort(O)) + 0.5) / len(O)       # percentile of jerk-optimality
    col = np.stack([1 - Fn, Fn, O], 1).clip(0, 1)
    ax.plot(*p.T, color="#444444", lw=0.5, alpha=0.6)
    ax.scatter(*p.T, c=col, s=7, depthshade=False, edgecolors="none")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("x (mm)", labelpad=-2); ax.set_ylabel("y (mm)", labelpad=-2); ax.set_zlabel("z (mm)", labelpad=-2)
    ax.tick_params(labelsize=6, pad=-2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.97, 1))


def fig_rgb_trajectories(subjects=(0, 8)):
    fig = plt.figure(figsize=(7.2, 3.4 * len(subjects)))
    k = 1
    for s in subjects:
        for c in ("VR", "NVR"):
            for g in ("Task1",):
                ax = fig.add_subplot(len(subjects), 2, k, projection="3d"); k += 1
                rgb_traj(ax, load_lgaf(s, c, g), f"Subject {s} · {g} · {c}")
    fig.text(0.5, 0.0, "Colour = (R, G, B) = (1 − F̂, F̂, Ô), percentile-scaled within segment:  green = high CFS, red = low CFS, "
             "bluer = smoother (higher jerk-optimality)", ha="center", fontsize=8, color=INK2)
    save(fig, "fig4a_rgb_trajectories")


def fig_paired():
    subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
    tests = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
    panels = [("cfs", "CFS (β/α × L_oddball/L_rest)"), ("cfs_cv", "CFS variability (CV)"),
              ("optimality", "Jerk optimality O"), ("ldlj", "Smoothness (LDLJ)"),
              ("micro_deviation", "Micro-deviation (mm)"), ("path_length", "Path length (m)")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    axs = axs.ravel()
    for ax, (m, lab) in zip(axs, panels):
        w = subj.pivot(index="subject", columns="cond", values=m)
        if m == "micro_deviation":
            w = w * 1000
        for s, r in w.iterrows():
            ax.plot([0, 1], [r.NVR, r.VR], color=GRAY, lw=0.8, alpha=0.7, zorder=1)
        ax.scatter(np.zeros(len(w)), w.NVR, s=18, color=ORANGE, zorder=2, edgecolor="white", lw=0.8, label="2D / no haptics")
        ax.scatter(np.ones(len(w)), w.VR, s=18, color=BLUE, zorder=2, edgecolor="white", lw=0.8, label="VR + haptics")
        for x, col in ((0, "NVR"), (1, "VR")):
            ax.plot([x - 0.18, x + 0.18], [w[col].mean()] * 2, color=INK, lw=2)
        t = tests.loc[m]
        ax.set_title(f"{lab}\n$d_z$={t.dz:.2f}, p={t.p_t:.2f}", fontsize=8, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["2D", "VR"]); ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=2)
    save(fig, "fig_human_paired")


def fig_cfs_quality():
    win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))
    mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    for c, col in (("NVR", ORANGE), ("VR", BLUE)):
        w = win[win.cond == c]
        ax.scatter(w.cfs, w.opt, s=7, color=col, alpha=0.45, edgecolor="none",
                   label="VR + haptics" if c == "VR" else "2D / no haptics")
    for s, w in win.groupby("subject"):  # within-subject fits
        if w.cfs.std() > 0:
            b = np.polyfit(w.cfs, w.opt, 1)
            xs = np.linspace(*w.cfs.quantile([0.05, 0.95]), 10)
            ax.plot(xs, np.polyval(b, xs), color=INK, lw=0.7, alpha=0.5)
    lo, hi = win.cfs.quantile([0.01, 0.99])
    ax.set_xlim(lo - 0.1, hi + 0.1)
    o = mm["mixed"]["opt"]; rs = mm["mixed"]["opt_robustness"]["random_slope"]
    ax.set_xlabel("Windowed CFS (5 s)"); ax.set_ylabel("Jerk optimality O (window mean)")
    ax.set_title(f"Random intercept: β = {o['beta']:.2f}, p = {o['p']:.3f}\n+ random slope: β = {rs['beta']:.2f}, p = {rs['p']:.2f}", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    ax.grid(color="#eeeeee", lw=0.6)
    # B: median split
    ax = axs[1]
    hb = win.assign(hi=win.groupby(["subject", "cond"]).cfs.transform(lambda v: v > v.median()))
    t = hb.groupby(["subject", "hi"]).opt.mean().unstack()
    for s, r in t.iterrows():
        ax.plot([0, 1], [r[False], r[True]], color=GRAY, lw=0.8)
    ax.scatter(np.zeros(len(t)), t[False], s=18, color="#c9c9c9", edgecolor=INK2, lw=0.6, zorder=3)
    ax.scatter(np.ones(len(t)), t[True], s=18, color="#00b894", edgecolor="white", lw=0.6, zorder=3)
    sp = mm["median_split"]["opt"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["≤ median CFS", "> median CFS"]); ax.set_xlim(-0.5, 1.5)
    ax.set_title(f"Within-subject split\n$d_z$ = {sp['dz']:.2f}, p = {sp['p']:.2f}", fontsize=8.5, color=INK)
    ax.set_ylabel("Jerk optimality O"); ax.grid(axis="y", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    save(fig, "fig4b_cfs_quality")


def fig_confusion():
    """Thesis Fig. 6.1 restyled (numbers from the thesis LGAF-CNN run)."""
    cm = np.array([[9292, 3917, 5141], [802, 27153, 6079], [1916, 8009, 25544]])
    cls = ["rest", "oddball", "msit"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rown = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax); cb.set_label("Row-normalised recall")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=cls, yticklabels=cls,
           xlabel="Predicted label", ylabel="True label")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({rown[i, j]:.0%})", ha="center", va="center", fontsize=7,
                    color="white" if rown[i, j] > 0.5 else INK)
    ax.spines[:].set_visible(False)
    save(fig, "fig_lgaf_confusion")


if __name__ == "__main__":
    fig_rgb_trajectories()
    fig_paired()
    fig_cfs_quality()
    fig_confusion()
    print("ok")
