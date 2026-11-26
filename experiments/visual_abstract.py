"""Graphical abstract: pipeline schematic + real-data findings.

Renders docs/visual_abstract.{png,pdf} (and a copy in figures/). The RIDL panel
shows the ablation if results/ablation_summary.csv exists, otherwise a
schematic of the hierarchical agent only.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGS, RESULTS, ROOT

INK, INK2, GRID, PANEL = "#191919", "#5f5f5f", "#e9e9e9", "#f6f7f9"
BLUE, ORANGE, AQUA, VIOLET = "#257de0", "#f47230", "#00b894", "#6a4ce2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#bdbdbd",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})


def box(fig, x, y, w, h, title, body, accent):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=PANEL, ec="#d9dce1", lw=1))
    ax.add_patch(FancyBboxPatch((0.01, 0.84), 0.98, 0.14, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=accent, ec="none"))
    ax.text(0.5, 0.91, title, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.text(0.06, 0.74, body, ha="left", va="top", color=INK, fontsize=8, linespacing=1.45)
    return ax


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=14, lw=1.6, color="#9aa0a6"))


def main():
    subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
    tests = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
    mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
    win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))

    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    fig.text(0.03, 0.955, "Is immersive VR–haptic surgical teleoperation cognitively transparent?",
             fontsize=17, weight="bold", color=INK)
    fig.text(0.03, 0.915, "Wearable-EEG Cognitive Flow Score (CFS) and a hierarchical diffusion-RL agent (RIDL) "
             "on dVRK teleoperation · within-subject pilot, N = 10", fontsize=10.5, color=INK2)

    # ---------------- pipeline row
    y, h, w = 0.665, 0.205, 0.215
    xs = [0.03, 0.272, 0.514, 0.756]
    box(fig, xs[0], y, w, h, "1  Embodied teleoperation",
        "dVRK / AMBF simulator\n• VR: Meta Quest 3 stereo +\n   Novint Falcon force feedback\n• 2D: monitor, no haptics\n"
        "Reference · Task 1 · Task 2", BLUE)
    box(fig, xs[1], y, w, h, "2  Wearable EEG → CFS",
        "8-ch dry Enobio, 500 Hz\nLGAF-CNN (Gramian angular\nfield) engagement decoder\n"
        r"$\mathrm{CFS}(t)=\frac{V_\beta}{V_\alpha}\cdot\frac{L_{oddball}(t)}{L_{rest}(t)}$", VIOLET)
    box(fig, xs[2], y, w, h, "3  Multimodal demonstrations",
        "endoscope frames (64×64)\n+ tool pose & force (10 Hz)\n+ CFS(t), jerk optimality\n"
        "→ 1 s action chunks", AQUA)
    box(fig, xs[3], y, w, h, "4  Hierarchical RIDL agent",
        "L1 skill prior: behaviour cloning\n    frozen Stable-Diffusion UNet\n    + Transformer kinematics\n"
        "L2 refinement: PPO on a\n    CFS-weighted reward\nsubject-by-subject curriculum", ORANGE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(fig, a + w + 0.003, b - 0.003, y + h / 2)

    # ---------------- findings row
    fy, fh = 0.085, 0.43
    fig.text(0.03, 0.585, "Findings", fontsize=12, weight="bold", color=INK)

    # A: equivalence forest
    ax = fig.add_axes([0.125, fy, 0.20, fh])
    rows = [("cfs", "Engagement (CFS)"), ("optimality", "Jerk optimality"), ("ldlj", "Smoothness (LDLJ)"),
            ("micro_deviation", "Micro-deviation"), ("path_efficiency", "Path efficiency")]
    ax.axvspan(-10, 10, color="#e6f4ee", lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, (m, lab) in enumerate(rows[::-1]):
        r = tests.loc[m]
        base = abs(r.mean_nvr)
        lo, hi, mid = 100 * r.ci90_lo / base, 100 * r.ci90_hi / base, 100 * r["diff"] / base
        eq = r.p_tost_10pct < 0.05
        col = AQUA if eq else (BLUE if lo > 0 or hi < 0 else INK2)
        ax.plot([lo, hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(mid, i, "o", color=col, ms=6, mec="white", mew=1)
        tag = "equivalent" if eq else ("VR higher (uncorr.)" if lo > 0 else "inconclusive")
        ax.text(-95, i - 0.32, tag, va="center", ha="right", fontsize=6.8, color=col, clip_on=False)
    ax.set_xlim(-90, 75); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows[::-1]], fontsize=8, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    ax.set_xlabel("VR − 2D, % of 2D mean (90% CI)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("A  VR vs 2D: equivalence", loc="left", fontsize=10, weight="bold", color=INK)
    ax.text(12, len(rows) - 0.5, "±10% band", ha="left", fontsize=7, color="#2f8f6b")

    # B: paired CFS
    ax = fig.add_axes([0.41, fy, 0.15, fh])
    w_ = subj.pivot(index="subject", columns="cond", values="cfs")
    for _, r in w_.iterrows():
        ax.plot([0, 1], [r.NVR, r.VR], color="#c4c4c4", lw=0.9)
    ax.scatter(np.zeros(len(w_)), w_.NVR, s=26, color=ORANGE, edgecolor="white", zorder=3)
    ax.scatter(np.ones(len(w_)), w_.VR, s=26, color=BLUE, edgecolor="white", zorder=3)
    t = tests.loc["cfs"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2D", "VR + haptics"]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Mean CFS per participant"); ax.grid(axis="y", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("B  Engagement preserved\n", loc="left", fontsize=10, weight="bold", color=INK)
    ax.text(0.0, 1.0, f"$d_z$ = {t.dz:.2f}, p = {t.p_t:.2f}, BF$_{{01}}$ = {t.BF01:.1f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=INK2)

    # C: CFS -> optimality
    ax = fig.add_axes([0.625, fy, 0.16, fh])
    lo, hi = win.cfs.quantile([0.02, 0.98])
    ww = win[(win.cfs >= lo) & (win.cfs <= hi)]
    ax.scatter(ww.cfs, ww.opt, s=5, color=VIOLET, alpha=0.25, edgecolor="none")
    for _, g in ww.groupby("subject"):
        if g.cfs.std() > 0:
            b = np.polyfit(g.cfs, g.opt, 1); x_ = np.linspace(*g.cfs.quantile([0.1, 0.9]), 5)
            ax.plot(x_, np.polyval(b, x_), color=INK, lw=0.8, alpha=0.55)
    o = mm["mixed"]["opt"]
    ax.set_xlabel("Windowed CFS (5 s)"); ax.set_ylabel("Jerk optimality"); ax.grid(color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("C  Engagement tracks control\n", loc="left", fontsize=10, weight="bold", color=INK)
    rs = mm["mixed"]["opt_robustness"]["random_slope"]
    ax.text(0.0, 1.0, f"β = {o['beta']:.2f}, p = {o['p']:.3f}; random-slope p = {rs['p']:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK2)

    # D: RIDL ablation (if available)
    ax = fig.add_axes([0.835, fy, 0.135, fh])
    abl = os.path.join(RESULTS, "ablation_summary.csv")
    ax.set_title("D  RIDL ablation", loc="left", fontsize=10, weight="bold", color=INK)
    if os.path.exists(abl):
        s = pd.read_csv(abl, index_col=0)
        keep = [v for v in ["RIDL (full)", "BC only", "PPO from scratch", "Raw pixels (no SD prior)",
                            "w/o flow-weighted reward", "Hold-position baseline"] if v in s.index]
        s = s.loc[keep]
        yy = np.arange(len(s))[::-1]
        cols = [ORANGE if v == "RIDL (full)" else ("#c9c9c9" if "baseline" in v else "#f6b48f") for v in s.index]
        ax.barh(yy, s.err_mm, xerr=s.err_mm_ci, color=cols, height=0.6, error_kw=dict(lw=0.8, ecolor=INK2))
        ax.set_yticks(yy); ax.set_yticklabels([v.replace(" (", "\n(") for v in s.index], fontsize=6.8)
        ax.set_xlabel("Held-out error (mm)"); ax.grid(axis="x", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")
        ax.text(0.02, 0.85, "Subject-wise 5-fold CV\n× VR / 2D demonstrations\n\nablates: imitation prior,\nPPO refinement, SD prior,\n"
                "CFS reward & input,\nsubject curriculum\n\n(results pending)", va="top", fontsize=8, color=INK2)

    fig.text(0.03, 0.012, "VR always preceded 2D (order not counterbalanced); N = 10 pilot. "
             "Statistics: paired t / Wilcoxon, TOST (±10%), JZS Bayes factors, linear mixed models.",
             fontsize=7.5, color=INK2)

    out = os.path.join(ROOT, "docs")
    os.makedirs(out, exist_ok=True)
    for d in (out, FIGS):
        fig.savefig(os.path.join(d, "visual_abstract.png"), dpi=200)
    fig.savefig(os.path.join(out, "visual_abstract.pdf"))
    print("ok")


if __name__ == "__main__":
    main()
