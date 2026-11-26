"""Human-side analyses for the Frontiers extension (real data only).

1. CFS (flow score F from LGAF-CNN) VR vs NVR, paired across N=10 subjects.
2. Kinematic execution-quality metrics VR vs NVR (paired, Holm-corrected).
3. Linear mixed-effects model: windowed execution quality ~ CFS + condition
   + segment + (1|subject).
4. Above- vs below-median CFS windows (Fig. 4B of the draft).
Outputs CSV/JSON in results/.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.signal import savgol_filter
from statsmodels.stats.multitest import multipletests

from common import CONDS, RESULTS, SUBJECTS, load_lgaf, optimality

TASKS = ["Task1", "Task2"]
FS = 10.0  # LGAF/ROS logs are resampled at 10 Hz (0.1 s timestamps)
WIN = 50   # 5 s windows for the mixed model


def ldlj(pos, fs=FS):
    """Log dimensionless jerk (Balasubramanian et al. 2015); higher (less negative) = smoother."""
    if len(pos) < 8:
        return np.nan
    dt = 1.0 / fs
    vel = np.gradient(pos, dt, axis=0)
    jerk = np.gradient(np.gradient(vel, dt, axis=0), dt, axis=0)
    T = len(pos) * dt
    vpeak = np.linalg.norm(vel, axis=1).max()
    if vpeak <= 0:
        return np.nan
    integ = np.sum(np.linalg.norm(jerk, axis=1) ** 2) * dt
    return -np.log(integ * T ** 3 / vpeak ** 2)


def submovement_efficiency(p, fs=FS, min_arc=1e-3):
    """Path efficiency: chord/arc of each submovement (segmented at speed minima),
    arc-length weighted. 1 = perfectly straight point-to-point reaches."""
    sp = np.linalg.norm(np.gradient(savgol_filter(p, 7, 2, axis=0), axis=0), axis=1) * fs
    thr = 0.2 * np.median(sp[sp > 0]) if (sp > 0).any() else 0
    moving = sp > thr
    eff, arcs, active = [], [], moving.mean()
    i = 0
    while i < len(p):
        if moving[i]:
            j = i
            while j < len(p) and moving[j]:
                j += 1
            seg = p[i:j + 1]
            arc = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
            if arc > min_arc:
                eff.append(np.linalg.norm(seg[-1] - seg[0]) / arc); arcs.append(arc)
            i = j
        else:
            i += 1
    pe = np.average(eff, weights=arcs) if eff else np.nan
    return pe, len(eff), active


def kin_metrics(d):
    p = d[["x", "y", "z"]].to_numpy(float)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path = step.sum()
    smooth = savgol_filter(p, 11, 3, axis=0)  # ~1 s window
    micro = np.sqrt(np.mean(np.sum((p - smooth) ** 2, 1)))
    f = d[["fx", "fy", "fz"]].to_numpy(float)
    return dict(
        path_length=path,
        mean_speed=path / (len(p) / FS),
        straightness=np.linalg.norm(p[-1] - p[0]) / path if path > 0 else np.nan,
        ldlj=ldlj(p),
        micro_deviation=micro,  # RMS deviation from 1 s Savitzky-Golay path (draft "tremor index")
        force_rms=np.sqrt(np.mean(np.sum(f ** 2, 1))),
        optimality=optimality(d).mean(),
        path_efficiency=submovement_efficiency(p)[0],
        n_submovements=submovement_efficiency(p)[1] / (len(p) / FS) * 60,  # per minute
        active_time_frac=submovement_efficiency(p)[2],
    )


def cohen_dz(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / diff.std(ddof=1)


def paired(a, b, sesoi_dz=0.5):
    import pingouin as pg
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    ci90 = diff.mean() + np.array([-1, 1]) * stats.t.ppf(0.95, n - 1) * se
    bound = sesoi_dz * diff.std(ddof=1)  # equivalence bound = 0.5 SD of differences
    p_tost = max(stats.t.sf((diff.mean() + bound) / se, n - 1), stats.t.cdf((diff.mean() - bound) / se, n - 1))
    bf10 = float(pg.bayesfactor_ttest(t.statistic, n, paired=True))
    # relative bound: +-10% of the 2D mean
    rb = 0.10 * abs(np.mean(b))
    p_tost_rel = max(stats.t.sf((diff.mean() + rb) / se, n - 1), stats.t.cdf((diff.mean() - rb) / se, n - 1))
    return dict(diff=diff.mean(), ci90_lo=ci90[0], ci90_hi=ci90[1], p_tost_dz05=p_tost,
                p_tost_10pct=p_tost_rel, BF01=1 / bf10,mean_vr=np.mean(a), sd_vr=np.std(a, ddof=1), mean_nvr=np.mean(b),
                sd_nvr=np.std(b, ddof=1), t=t.statistic, p_t=t.pvalue,
                W=w.statistic, p_w=w.pvalue, dz=cohen_dz(a, b),
                n_vr_better=None)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows, wins = [], []
    for s in SUBJECTS:
        ref_f = {c: load_lgaf(s, c, "Reference")["F"].mean() for c in CONDS}
        for c in CONDS:
            for g in TASKS:
                d = load_lgaf(s, c, g)
                r = dict(subject=s, cond=c, seg=g, cfs=d.F.mean(), cfs_sd=d.F.std(),
                         cfs_cv=d.F.std() / d.F.mean(),
                         E=(d.Vb / d.Va).mean(), task_ratio=d.Task_Ratio.mean(),
                         cfs_rel=d.F.mean() / ref_f[c])
                r.update(kin_metrics(d))
                rows.append(r)
                O = optimality(d)
                for w0 in range(0, len(d) - WIN + 1, WIN):
                    dw = d.iloc[w0:w0 + WIN]
                    wins.append(dict(subject=s, cond=c, seg=g, cfs=dw.F.mean(),
                                     opt=O[w0:w0 + WIN].mean(),
                                     ldlj=ldlj(dw[["x", "y", "z"]].to_numpy(float)),
                                     micro=kin_metrics(dw)["micro_deviation"]))
    seg = pd.DataFrame(rows)
    seg.to_csv(os.path.join(RESULTS, "human_segments.csv"), index=False)
    win = pd.DataFrame(wins).dropna()
    win.to_csv(os.path.join(RESULTS, "human_windows.csv"), index=False)

    # --- subject-level paired tests (mean over Task1+Task2) ---
    subj = seg.groupby(["subject", "cond"]).mean(numeric_only=True).reset_index()
    subj.to_csv(os.path.join(RESULTS, "human_subject_means.csv"), index=False)
    metrics = ["cfs", "cfs_cv", "E", "task_ratio", "cfs_rel", "path_length", "mean_speed",
               "straightness", "ldlj", "micro_deviation", "force_rms", "optimality",
               "path_efficiency", "n_submovements", "active_time_frac"]
    out = []
    for m in metrics:
        a = subj[subj.cond == "VR"].sort_values("subject")[m].to_numpy()
        b = subj[subj.cond == "NVR"].sort_values("subject")[m].to_numpy()
        r = paired(a, b)
        r["metric"] = m
        r["n_vr_higher"] = int((a > b).sum())
        out.append(r)
    tests = pd.DataFrame(out)
    kin = tests.metric.isin(metrics[5:])
    tests.loc[kin, "p_holm"] = multipletests(tests.loc[kin, "p_t"], method="holm")[1]
    tests.loc[~kin, "p_holm"] = multipletests(tests.loc[~kin, "p_t"], method="holm")[1]
    tests.to_csv(os.path.join(RESULTS, "human_paired_tests.csv"), index=False)

    # --- mixed-effects: quality ~ CFS (z) + condition + segment + (1|subject) ---
    win["cfs_z"] = (win.cfs - win.cfs.mean()) / win.cfs.std()
    mm = {}
    for y in ["opt", "ldlj", "micro"]:
        win[y + "_z"] = (win[y] - win[y].mean()) / win[y].std()
        f = smf.mixedlm(f"{y}_z ~ cfs_z + C(cond) + C(seg)", win, groups=win["subject"]).fit(reml=True)
        mm[y] = dict(beta=f.params["cfs_z"], se=f.bse["cfs_z"], p=f.pvalues["cfs_z"],
                     beta_vr=f.params.get("C(cond)[T.VR]"), p_vr=f.pvalues.get("C(cond)[T.VR]"),
                     n_windows=len(win))
        # interaction model
        fi = smf.mixedlm(f"{y}_z ~ cfs_z * C(cond) + C(seg)", win, groups=win["subject"]).fit(reml=True)
        mm[y]["beta_inter"] = fi.params["cfs_z:C(cond)[T.VR]"]
        mm[y]["p_inter"] = fi.pvalues["cfs_z:C(cond)[T.VR]"]

    # --- robustness: random slopes, outlier trimming, within-subject ranks ---
    lo, hi = win.cfs.quantile([0.01, 0.99])
    rob = {}
    for name, d in [("random_slope", win),
                    ("random_slope_trim1_99", win[(win.cfs >= lo) & (win.cfs <= hi)]),
                    ("random_slope_rank", win.assign(cfs=win.groupby("subject").cfs.rank(pct=True)))]:
        d = d.copy()
        d["cz"] = (d.cfs - d.cfs.mean()) / d.cfs.std()
        d["oz"] = (d.opt - d.opt.mean()) / d.opt.std()
        f = smf.mixedlm("oz ~ cz + C(cond) + C(seg)", d, groups=d["subject"], re_formula="~cz").fit()
        rob[name] = dict(beta=f.params["cz"], se=f.bse["cz"], p=f.pvalues["cz"], n=len(d))
    mm["opt_robustness"] = rob

    # --- above vs below within-subject median CFS windows ---
    win["hi"] = win.groupby(["subject", "cond"]).cfs.transform(lambda v: v > v.median())
    hb = win.groupby(["subject", "hi"])[["opt", "ldlj", "micro"]].mean().unstack()
    split = {}
    for y in ["opt", "ldlj", "micro"]:
        a, b = hb[(y, True)].to_numpy(), hb[(y, False)].to_numpy()
        split[y] = dict(hi=a.mean(), lo=b.mean(), t=stats.ttest_rel(a, b).statistic,
                        p=stats.ttest_rel(a, b).pvalue, dz=cohen_dz(a, b))

    # --- engagement variability ---
    res = dict(mixed=mm, median_split=split,
               cfs_overall={c: dict(mean=subj[subj.cond == c].cfs.mean(),
                                    sd=subj[subj.cond == c].cfs.std()) for c in CONDS})
    json.dump(res, open(os.path.join(RESULTS, "human_models.json"), "w"), indent=2, default=float)
    pd.set_option("display.width", 200)
    print(tests[["metric", "mean_vr", "sd_vr", "mean_nvr", "sd_nvr", "t", "p_t", "p_w", "dz", "n_vr_higher", "p_holm"]].round(4))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
