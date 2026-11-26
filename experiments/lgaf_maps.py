"""LGAF maps of the decision-making network (thesis Fig. 3.2 nodes covered by the montage).

For each node, the channel-averaged alpha (8-12 Hz) and beta (13-30 Hz) power envelopes
are resampled to the 10 Hz CFS timeline and cut into 64-step (6.4 s) windows. Each window
is mapped with the LGAF transform of thesis Eqs. 3.9-3.14:
    x_hat = 2 (x - min~(x)) / (max~(x) - min~(x)) - 1      (LogSumExp-smoothed min/max)
    phi   = arccos(x_hat)
    GADF_ij = sin(W1 phi^alpha_i - W2 phi^beta_j + b)
The trained LGAF weights were not archived, so W1 = W2 = 1, b = 0 (identity initialisation).
Diagonal sin(phi^a_t - phi^b_t) > 0  <=>  beta relatively stronger than alpha (engagement).
Windows are split at each participant's median CFS (per condition) and averaged.
Outputs results/lgaf_maps.npz, results/lgaf_diag_stats.csv.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

from common import CONDS, DATA, RESULTS, SUBJECTS, load_lgaf

CHANNELS = ["O1", "Oz", "O2", "CP1", "CP2", "C3", "C4", "Fz"]
NODES = {  # thesis Fig. 3.2 node -> recorded electrodes
    "Core decision & goal maintenance (Fz)": ["Fz"],
    "Motor planning / output (C3, C4)": ["C3", "C4"],
    "Parietal attention (CP1, CP2)": ["CP1", "CP2"],
    "Visual input (O1, Oz, O2)": ["O1", "Oz", "O2"],
}
FS, N, ALPHA_LSE = 500, 64, 4.0
TASKS = ["Task1", "Task2"]


def easy_file(s, c, task):
    f = glob.glob(os.path.join(DATA, "NIC", f"Subject-{s}", f"*_subject{s}-*-{c}_{task}.easy"))
    return f[0] if f else None


def band_env(x, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=0), axis=0))


def lse_norm(x):
    z = (x - x.mean()) / (x.std() + 1e-12)
    mx = np.log(np.exp(ALPHA_LSE * z).sum()) / ALPHA_LSE
    mn = -np.log(np.exp(-ALPHA_LSE * z).sum()) / ALPHA_LSE
    return np.clip(2 * (z - mn) / (mx - mn) - 1, -1, 1)


def gadf(a, b, w1=1.0, w2=1.0, bias=0.0):
    pa, pb = np.arccos(lse_norm(a)), np.arccos(lse_norm(b))
    return np.sin(w1 * pa[:, None] - w2 * pb[None, :] + bias)


def main():
    maps = {}   # (cond, node, hi) -> list of per-subject mean maps
    rows = []
    for s in SUBJECTS:
        for c in CONDS:
            per = {(node, hi): [] for node in NODES for hi in (True, False)}
            for task in TASKS:
                f = easy_file(s, c, task)
                if f is None:
                    continue
                raw = np.loadtxt(f)
                eeg, ts = raw[:, :8] / 1000.0, raw[:, 12] / 1000.0      # nV -> uV, ms -> s
                lg = load_lgaf(s, c, task)
                t10 = lg.timestamp.to_numpy(float)
                keep = (t10 >= ts[0]) & (t10 <= ts[-1])
                t10, F = t10[keep], lg.F.to_numpy(float)[keep]
                if len(t10) < N:
                    continue
                envs = {}
                for node, chans in NODES.items():
                    idx = [CHANNELS.index(ch) for ch in chans]
                    x = eeg[:, idx].mean(1)
                    envs[node] = [np.interp(t10, ts, band_env(x, *band)) for band in ((8, 12), (13, 30))]
                wins = [(w, F[w:w + N].mean()) for w in range(0, len(t10) - N + 1, N)]
                med = np.median([m for _, m in wins])
                for w, fm in wins:
                    for node in NODES:
                        a, b = envs[node][0][w:w + N], envs[node][1][w:w + N]
                        per[(node, fm > med)].append(gadf(a, b))
            for (node, hi), L in per.items():
                if L:
                    m = np.mean(L, 0)
                    maps.setdefault((c, node, hi), []).append(m)
                    rows.append(dict(subject=s, cond=c, node=node, high_cfs=hi, n_windows=len(L),
                                     diag=np.diag(m).mean(), mean=m.mean()))
            print(f"subject {s} {c}: {sum(len(v) for v in per.values()) // len(NODES)} windows", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "lgaf_subject_windows.csv"), index=False)
    out = []
    for c in CONDS:
        for node in NODES:
            p = df[(df.cond == c) & (df.node == node)].pivot(index="subject", columns="high_cfs", values="diag").dropna()
            t = stats.ttest_rel(p[True], p[False])
            d = (p[True] - p[False])
            out.append(dict(cond=c, node=node, diag_high=p[True].mean(), diag_low=p[False].mean(),
                            diff=d.mean(), t=t.statistic, p=t.pvalue, dz=d.mean() / d.std(ddof=1), n=len(p)))
    st = pd.DataFrame(out)
    st.to_csv(os.path.join(RESULTS, "lgaf_diag_stats.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, "lgaf_maps.npz"),
                        **{f"{c}|{node}|{'high' if hi else 'low'}": np.mean(v, 0) for (c, node, hi), v in maps.items()},
                        **{f"{c}|{node}|n": len(maps.get((c, node, True), [])) for c in CONDS for node in NODES})
    print(st.round(3).to_string())


if __name__ == "__main__":
    main()
