"""Write every reported number to paper/results_macros.tex so the
manuscript never contains hand-typed statistics."""
import json
import os

import numpy as np
import pandas as pd

from common import PAPER, RESULTS

OUT = os.path.join(PAPER, "results_macros.tex")
M = {}


def p(v):
    return "<0.001" if v < 0.001 else f"={v:.3f}" if v < 0.01 else f"={v:.2f}"


def put(name, val):
    M[name] = val


# ---------------- human data
t = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
for m, key, fmt in [("cfs", "Cfs", "{:.2f}"), ("cfs_cv", "CfsCv", "{:.2f}"), ("optimality", "Opt", "{:.3f}"),
                    ("ldlj", "Ldlj", "{:.2f}"), ("micro_deviation", "Micro", "{:.2f}"),
                    ("path_efficiency", "PathEff", "{:.2f}"), ("path_length", "PathLen", "{:.2f}"),
                    ("active_time_frac", "Active", "{:.2f}"), ("n_submovements", "Subm", "{:.1f}")]:
    r = t.loc[m]
    sc = 1000 if m == "micro_deviation" else 1
    put(f"h{key}VR", fmt.format(r.mean_vr * sc)); put(f"h{key}VRsd", fmt.format(r.sd_vr * sc))
    put(f"h{key}NVR", fmt.format(r.mean_nvr * sc)); put(f"h{key}NVRsd", fmt.format(r.sd_nvr * sc))
    put(f"h{key}T", f"{r.t:.2f}"); put(f"h{key}P", p(r.p_t)); put(f"h{key}Pholm", p(r.p_holm))
    put(f"h{key}Dz", f"{r.dz:.2f}"); put(f"h{key}BF", f"{r.BF01:.2f}")
    put(f"h{key}CIlo", fmt.format(r.ci90_lo * sc)); put(f"h{key}CIhi", fmt.format(r.ci90_hi * sc))
    put(f"h{key}Tost", p(r.p_tost_10pct)); put(f"h{key}TostDz", p(r.p_tost_dz05))
    put(f"h{key}Nvr", str(int(r.n_vr_higher)))
r = t.loc["cfs"]
put("hCfsCIloPct", f"{100 * r.ci90_lo / r.mean_nvr:.0f}"); put("hCfsCIhiPct", f"{100 * r.ci90_hi / r.mean_nvr:.0f}")

mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
o = mm["mixed"]["opt"]
put("mmBeta", f"{o['beta']:.2f}"); put("mmSE", f"{o['se']:.3f}"); put("mmP", p(o["p"])); put("mmN", str(o["n_windows"]))
put("mmInterP", p(o["p_inter"]))
for k, nm in [("random_slope", "Rs"), ("random_slope_trim1_99", "Trim"), ("random_slope_rank", "Rank")]:
    q = mm["mixed"]["opt_robustness"][k]
    put(f"mm{nm}Beta", f"{q['beta']:.2f}"); put(f"mm{nm}SE", f"{q['se']:.3f}"); put(f"mm{nm}P", p(q["p"]))
for y, nm in [("ldlj", "Ldlj"), ("micro", "Micro")]:
    put(f"mm{nm}Beta", f"{mm['mixed'][y]['beta']:.2f}"); put(f"mm{nm}P", p(mm["mixed"][y]["p"]))
sp = mm["median_split"]["opt"]
win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))
win["hi"] = win.groupby(["subject", "cond"]).cfs.transform(lambda v: v > v.median())
hb = win.groupby(["subject", "hi"]).opt.mean().unstack()
put("msNup", str(int((hb[True] > hb[False]).sum())))
put("msHi", f"{sp['hi']:.3f}"); put("msLo", f"{sp['lo']:.3f}"); put("msP", p(sp["p"])); put("msDz", f"{sp['dz']:.2f}")

# ---------------- RIDL ablation (if available)
abl = os.path.join(RESULTS, "ablation_summary.csv")
if os.path.exists(abl):
    s = pd.read_csv(abl, index_col=0)
    names = {"RIDL (full)": "Ridl", "PPO from scratch": "Ppo", "BC only": "Bc",
             "w/o SD-UNet (no vision)": "NoVis", "Raw pixels (no SD prior)": "Pix",
             "w/o flow-weighted reward": "NoFlowR", "w/o CFS state input": "NoCfsIn",
             "Pooled (no curriculum)": "Pooled", "Hold-position baseline": "Hold",
             "Constant-velocity baseline": "CV"}
    for v, k in names.items():
        if v in s.index:
            put(f"a{k}Err", f"{s.loc[v, 'err_mm']:.2f}"); put(f"a{k}Ci", f"{s.loc[v, 'err_mm_ci']:.2f}")
            put(f"a{k}Etraj", f"{s.loc[v, 'E_traj']:.2f}"); put(f"a{k}Sig", f"{s.loc[v, 'sigma_r']:.2f}")
            put(f"a{k}Nmse", f"{s.loc[v, 'nmse']:.3f}"); put(f"a{k}Steps", f"{s.loc[v, 'env_steps']:,.0f}")
            put(f"a{k}N", str(int(s.loc[v, "n_runs"])))
    # paired run-level tests vs RIDL and VR-vs-NVR demo source equivalence
    d = pd.read_csv(os.path.join(RESULTS, "ablation.csv"))
    run = d.groupby(["variant", "source", "fold", "seed"]).err_mm.mean().unstack("variant")
    from scipy import stats
    for v, k in names.items():
        if v in run and v != "RIDL (full)":
            x = (run[v] - run["RIDL (full)"]).dropna()
            if len(x) > 2:
                put(f"d{k}", f"{x.mean():+.2f}"); put(f"d{k}P", p(stats.wilcoxon(x).pvalue))
    r = d[d.variant == "RIDL (full)"].groupby(["source", "fold", "seed"]).err_mm.mean().unstack("source")
    diff = (r["VR"] - r["NVR"]).dropna()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    bound = 0.10 * r["NVR"].mean()
    pt = max(stats.t.sf((diff.mean() + bound) / se, len(diff) - 1), stats.t.cdf((diff.mean() - bound) / se, len(diff) - 1))
    put("srcVR", f"{r['VR'].mean():.2f}"); put("srcNVR", f"{r['NVR'].mean():.2f}")
    put("srcDiff", f"{diff.mean():+.2f}"); put("srcP", p(stats.ttest_1samp(diff, 0).pvalue)); put("srcTost", p(pt))
    put("srcCIlo", f"{diff.mean() - stats.t.ppf(.95, len(diff) - 1) * se:+.2f}")
    put("srcCIhi", f"{diff.mean() + stats.t.ppf(.95, len(diff) - 1) * se:+.2f}")
cs = os.path.join(RESULTS, "curves_summary.json")
if os.path.exists(cs):
    c = json.load(open(cs))
    put("cRidlFinal", f"{c['ridl_final']:.3f}"); put("cPpoFinal", f"{c['ppo_final']:.3f}")
    put("cRidlFirst", f"{c['ridl_after_first_update']:.3f}"); put("cSteps", f"{c['total_steps']:,}")
    put("cPpoReach", f"{100 * c['ppo_reached_frac']:.0f}")

tc = os.path.join(RESULTS, "thesis_ckpts.csv")
if os.path.exists(tc):
    t_ = pd.read_csv(tc)
    g = t_.groupby("model")
    for mname, k in (("Thesis BC (imitation)", "tBc"), ("Thesis RIDL (BC→PPO)", "tRidl")):
        if mname in g.groups:
            x = g.get_group(mname)
            put(f"{k}Err", f"{x.err_mm.mean():.1f}"); put(f"{k}Nmse", f"{x.nmse.mean():.2f}")
    put("tBase", f"{t_.baseline_mean_pose_mm.mean():.1f}")

EXPECTED = [f"a{k}{m}" for k in ["Ridl", "Ppo", "Bc", "NoVis", "Pix", "NoFlowR", "NoCfsIn", "Pooled", "Hold", "CV"]
            for m in ["Err", "Ci", "Etraj", "Sig", "Nmse", "Steps", "N"]]
EXPECTED += [f"d{k}{m}" for k in ["Ppo", "Bc", "NoVis", "Pix", "NoFlowR", "NoCfsIn", "Pooled", "Hold", "CV"] for m in ["", "P"]]
EXPECTED += ["srcVR", "srcNVR", "srcDiff", "srcP", "srcTost", "srcCIlo", "srcCIhi",
             "tBcErr", "tBcNmse", "tRidlErr", "tRidlNmse", "tBase", "cRidlFinal", "cPpoFinal", "cRidlFirst", "cSteps", "cPpoReach"]
for k in EXPECTED:
    M.setdefault(k, "\\textbf{TBD}")

with open(OUT, "w") as f:
    f.write("% AUTO-GENERATED by experiments/make_macros.py -- do not edit by hand\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros -> {OUT}")

# ---------------- tables
SUBJ_HEAD = r"""\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{2}{c}{\textbf{CFS}} & \multicolumn{2}{c}{\textbf{Path efficiency}} & \multicolumn{2}{c}{\textbf{Optimality $O$}} & \multicolumn{2}{c}{\textbf{LDLJ}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Subject} & VR & 2D & VR & 2D & VR & 2D & VR & 2D \\
\midrule
"""
ABL_HEAD = r"""\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{Error (mm)} & $\bm{E_{\mathrm{traj}}}$ & $\bm{\sigma_r}$ & \textbf{NMSE} & \textbf{Steps} \\
\midrule
"""
TAIL = "\n\\bottomrule\n\\end{tabular}\n"
subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
w = subj.pivot(index="subject", columns="cond", values=["cfs", "path_efficiency", "optimality", "ldlj"])
lines = []
for s_, r in w.iterrows():
    lines.append(f"sub{s_} & {r[('cfs','VR')]:.2f} & {r[('cfs','NVR')]:.2f} & {r[('path_efficiency','VR')]:.2f} & "
                 f"{r[('path_efficiency','NVR')]:.2f} & {r[('optimality','VR')]:.3f} & {r[('optimality','NVR')]:.3f} & "
                 f"{r[('ldlj','VR')]:.2f} & {r[('ldlj','NVR')]:.2f} \\\\")
m_, sd_ = w.mean(), w.std()
lines.append("\\midrule")
lines.append("Mean (SD) & " + " & ".join(
    f"{m_[(a, c)]:.{k}f} ({sd_[(a, c)]:.{k}f})" for a, k in (("cfs", 2), ("path_efficiency", 2), ("optimality", 3), ("ldlj", 2))
    for c in ("VR", "NVR")) + " \\\\")
open(os.path.join(PAPER, "table_subjects.tex"), "w").write(SUBJ_HEAD + "\n".join(lines) + TAIL)

rows = ["\\multicolumn{6}{c}{\\TODO{ablation results pending (run\\_all.sh)}} \\\\"]
if os.path.exists(abl):
    rows = []
    for v in s.index:
        steps = "--" if s.loc[v, "env_steps"] == 0 else f"{s.loc[v, 'env_steps']:,.0f}"
        rows.append(f"{v} & {s.loc[v,'err_mm']:.2f} $\\pm$ {s.loc[v,'err_mm_ci']:.2f} & {s.loc[v,'E_traj']:.2f} & "
                    f"{s.loc[v,'sigma_r']:.2f} & {s.loc[v,'nmse']:.3f} & {steps} \\\\")
open(os.path.join(PAPER, "table_ablation.tex"), "w").write(ABL_HEAD + "\n".join(rows) + TAIL)
print("tables written")
