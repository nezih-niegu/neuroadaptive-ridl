"""RIDL figures: vis2-style trajectory panels (+GIF), ablation bars, learning
curves, and VR/NVR cross-domain matrix."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from common import FIGS, RESULTS

INK, INK2, GRID = "#191919", "#666666", "#eeeeee"
SERIES = ["#257de0", "#f47230", "#00b894"]


def save(fig, name, dark=False):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", dpi=300,
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def tfile(variant, src, fold):
    return os.path.join(RESULTS, "trajectories",
                        f"{variant.replace('/', '_').replace(' ', '_')}__{src}__f{fold}.npz")


# --------------------------------------------------------------- vis2 style
def _dark(ax, is3d=False):
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for a in ("x", "y") + (("z",) if is3d else ()):
        getattr(ax, f"{a}axis").label.set_color("white")
    if is3d:
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor((0.45, 0.45, 0.45, 1))
            a._axinfo["grid"]["color"] = (1, 1, 1, 0.25)
    else:
        for sp in ax.spines.values():
            sp.set_color("#888888")


def vis2_panel(fig, gs_row, seg_key, ridl, ppo, title, steps=None):
    tr = ridl[seg_key + "__true"]; pr = ridl[seg_key + "__pred"]; pp = ppo[seg_key + "__pred"]
    er = ridl[seg_key + "__err"]; ep = ppo[seg_key + "__err"]
    n = len(tr) if steps is None else steps
    ax3 = fig.add_subplot(gs_row[0], projection="3d"); _dark(ax3, True)
    ax2 = fig.add_subplot(gs_row[1]); _dark(ax2)
    for arr, col, lab in ((tr, "white", "TRUE"), (pp, "red", "PPO (scratch)"), (pr, "#3f7bff", "RIDL")):
        ax3.plot(*(arr[:n] * 1000).T, color=col, lw=1.4, label=lab)
    ax3.set_xlabel("X (mm)"); ax3.set_ylabel("Y (mm)"); ax3.set_zlabel("Z (mm)")
    ax3.legend(facecolor="black", labelcolor="white", fontsize=6, loc="upper left")
    ax3.set_title(title, color="white", fontsize=8)
    k = np.arange(n)
    ax2.plot(k, ep[:n], color="red", lw=1.2, label=f"ERR_PPO  (mean {ep.mean():.1f} mm)")
    ax2.plot(k, er[:n], color="#3f7bff", lw=1.4, label=f"ERR_RIDL (mean {er.mean():.1f} mm)")
    ax2.set_xlabel("Step (0.1 s)"); ax2.set_ylabel("‖x̂ − x‖ (mm)")
    ax2.legend(facecolor="black", labelcolor="white", fontsize=6)
    return ax3, ax2


def fig_vis2(src="VR", fold=4, subject=None):
    ridl = np.load(tfile("RIDL (full)", src, fold)); ppo = np.load(tfile("PPO from scratch", src, fold))
    keys = sorted({k.rsplit("__", 1)[0] for k in ridl.files})
    if subject is not None:
        keys = [k for k in keys if k.startswith(f"subject{subject}-")]
    keys = [k for k in keys if "Task1" in k]  # VR and NVR, Task1, both held-out subjects
    fig = plt.figure(figsize=(8, 3.0 * len(keys)), facecolor="black")
    gs = fig.add_gridspec(len(keys), 2, width_ratios=[1.1, 1])
    for i, k in enumerate(keys):
        vis2_panel(fig, (gs[i, 0], gs[i, 1]), k, ridl, ppo, f"{k.replace('_', ' · ')}  (trained on {src}, held-out)")
    fig.tight_layout()
    save(fig, f"fig5_vis2_{src}_fold{fold}")
    return keys


def gif_vis2(src="VR", fold=4, key=None, every=3):
    ridl = np.load(tfile("RIDL (full)", src, fold)); ppo = np.load(tfile("PPO from scratch", src, fold))
    key = key or sorted({k.rsplit("__", 1)[0] for k in ridl.files})[0]
    tr, pr, pp = (a[key + s] * 1000 for a, s in ((ridl, "__true"), (ridl, "__pred"), (ppo, "__pred")))
    er, ep = ridl[key + "__err"], ppo[key + "__err"]
    fig = plt.figure(figsize=(10, 5), facecolor="black")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d"); ax2 = fig.add_subplot(1, 2, 2)
    _dark(ax3, True); _dark(ax2)
    allp = np.concatenate([tr, pr, pp])
    ax3.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax3.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax3.set_zlim(allp[:, 2].min(), allp[:, 2].max())
    ax2.set_xlim(0, len(er)); ax2.set_ylim(0, max(er.max(), ep.max()) * 1.05)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z"); ax2.set_xlabel("Step"); ax2.set_ylabel("Error (mm)")
    lt, = ax3.plot([], [], [], color="white", lw=2, label="TRUE")
    lp, = ax3.plot([], [], [], color="red", lw=2, label="PPO")
    lr, = ax3.plot([], [], [], color="#3f7bff", lw=2, label="RIDL")
    ep_l, = ax2.plot([], [], color="red", lw=2, label="ERR_PPO")
    er_l, = ax2.plot([], [], color="#3f7bff", lw=2, label="ERR_RIDL")
    ax3.legend(facecolor="black", labelcolor="white"); ax2.legend(facecolor="black", labelcolor="white")
    ax3.set_title(key, color="white", fontsize=9)

    def upd(i):
        n = i * every + 1
        for ln, a in ((lt, tr), (lp, pp), (lr, pr)):
            ln.set_data(a[:n, 0], a[:n, 1]); ln.set_3d_properties(a[:n, 2])
        ep_l.set_data(np.arange(n), ep[:n]); er_l.set_data(np.arange(n), er[:n])
        return lt, lp, lr, ep_l, er_l

    anim = FuncAnimation(fig, upd, frames=len(er) // every, blit=False)
    anim.save(os.path.join(FIGS, f"anim_vis2_{key}_{src}.gif"), writer=PillowWriter(fps=12))
    plt.close(fig)


# --------------------------------------------------------------- ablation summary
ORDER = ["RIDL (full)", "PPO from scratch", "BC only", "w/o SD-UNet (no vision)",
         "Raw pixels (no SD prior)", "w/o flow-weighted reward", "w/o CFS state input",
         "Pooled (no curriculum)", "Constant-velocity baseline", "Hold-position baseline", "Mean-pose baseline"]


def summarize(df):
    """Per run (variant, source, fold, seed): mean over held-out segments; then
    mean ± 95% CI over runs."""
    run = df.groupby(["variant", "source", "fold", "seed"]).mean(numeric_only=True).reset_index()
    g = run.groupby("variant")
    out = g[["err_mm", "E_traj", "sigma_r", "nmse", "pred_jerk_mm", "env_steps"]].mean()
    n = g.size()
    for m in ("err_mm", "E_traj", "nmse"):
        out[m + "_ci"] = 1.96 * g[m].std() / np.sqrt(n)
    out["n_runs"] = n
    return out.reindex([o for o in ORDER if o in out.index]), run


def fig_ablation(tag="ablation"):
    df = pd.read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    s, run = summarize(df)
    s.to_csv(os.path.join(RESULTS, f"{tag}_summary.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axs[0]
    y = np.arange(len(s))[::-1]
    cols = ["#257de0" if v == "RIDL (full)" else ("#c9c9c9" if "baseline" in v else "#8fb5e8") for v in s.index]
    ax.barh(y, s.err_mm, xerr=s.err_mm_ci, color=cols, height=0.62, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for yi, v, e in zip(y, s.err_mm, s.err_mm_ci):
        ax.text(v + e + 0.3, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=7.5)
    ax.set_xlabel("Held-out position error (mm, ↓)")
    ax.grid(axis="x", color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Ablation (5 subject-folds × seeds, both demo sources)", fontsize=8.5)
    # paired deltas vs RIDL (same fold/seed/source)
    ax = axs[1]
    piv = run.pivot_table(index=["source", "fold", "seed"], columns="variant", values="err_mm")
    vs = [v for v in s.index if v not in ("RIDL (full)",) and v in piv]
    d = [(piv[v] - piv["RIDL (full)"]).dropna() for v in vs]
    yy = np.arange(len(vs))[::-1]
    for yi, dv in zip(yy, d):
        ax.scatter(dv, np.full(len(dv), yi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(dv)),
                   s=8, color="#257de0", alpha=0.6, edgecolor="none")
        ax.plot([dv.mean()] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([])
    ax.set_xlabel("Δ error vs RIDL (mm; >0 = worse)")
    ax.grid(axis="x", color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Paired difference per run", fontsize=8.5)
    fig.tight_layout()
    save(fig, f"fig_ablation_{tag}")
    return s


def fig_crossdomain(tag="ablation", variant="RIDL (full)"):
    df = pd.read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    d = df[df.variant == variant].groupby(["source", "cond"]).err_mm.agg(["mean", "std"]).reset_index()
    M = d.pivot(index="source", columns="cond", values="mean").loc[["VR", "NVR"], ["VR", "NVR"]]
    S = d.pivot(index="source", columns="cond", values="std").loc[["VR", "NVR"], ["VR", "NVR"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    im = ax.imshow(M.values, cmap="Blues_r")
    fig.colorbar(im, ax=ax).set_label("Position error (mm)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["VR test", "2D test"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["VR-trained", "2D-trained"])
    lo, hi = M.values.min(), M.values.max()
    for i in range(2):
        for j in range(2):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.1f}\n±{S.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if (v - lo) / (hi - lo + 1e-9) < 0.5 else INK)
    ax.spines[:].set_visible(False)
    ax.set_title(f"{variant}: train × test condition", fontsize=8.5)
    save(fig, f"fig_crossdomain_{tag}")
    return M


def fig_curves(tag="curves"):
    f = os.path.join(RESULTS, f"{tag}.json")
    if not os.path.exists(f):
        return None
    C = json.load(open(f))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (name, runs), col in zip(C.items(), SERIES):
        L = min(len(r) for r in runs)
        st = np.array([[p["steps"] for p in r[:L]] for r in runs]).mean(0)
        y = np.array([[p["test_nmse"] for p in r[:L]] for r in runs])
        m, ci = y.mean(0), 1.96 * y.std(0) / np.sqrt(len(y))
        ax.plot(st / 1000, m, color=col, lw=2, label=name)
        ax.fill_between(st / 1000, m - ci, m + ci, color=col, alpha=0.18, lw=0)
    ax.set_xlabel("PPO environment steps (×1000)"); ax.set_ylabel("Held-out normalised MSE (↓)")
    ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig_learning_curves")


if __name__ == "__main__":
    print(fig_ablation())
    print(fig_crossdomain())
    fig_curves()
    for src in ("VR", "NVR"):
        fig_vis2(src, 4)
    gif_vis2("VR", 4)
