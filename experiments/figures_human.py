"""Human-data and setup figures, all in Plotly (thesis-notebook visual language).

fig1_setup              composite of thesis artwork (photo, system diagram, EEG montage)
fig_human_paired        participant-level VR vs 2D metrics
fig3_ridl               composite of thesis RIDL diagrams
fig4a_rgb_trajectories  visualsThesis.ipynb 3D trajectory: RGB = (1-F, F, O), "Reward:" labels
fig4b_cfs_quality       windowed CFS vs jerk optimality + within-subject median split
fig_lgaf_confusion      LGAF-CNN confusion matrix (Blues, counts)
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import FIGS, RESULTS, load_lgaf, optimality
from pstyle import GRAY, GREEN, NVR, VR, add_image, save, style

ART = os.path.join(FIGS, "thesis_art")


def fig_setup():
    fig = go.Figure()
    add_image(fig, os.path.join(ART, "setup_photo.png"), 0.0, 0.55, 0.47, 0.42, "A")
    add_image(fig, os.path.join(ART, "eeg_montage.png"), 0.07, 0.0, 0.33, 0.50, "C")
    add_image(fig, os.path.join(ART, "system_diagram.png"), 0.49, 0.0, 0.51, 0.97, "B")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    style(fig, width=1500, height=930).update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    save(fig, "fig1_setup")


def fig_ridl():
    fig = go.Figure()
    add_image(fig, os.path.join(ART, "ridl_ambf.png"), 0.0, 0.0, 0.56, 0.94, "A")
    add_image(fig, os.path.join(ART, "ridl_training_loop.png"), 0.58, 0.0, 0.42, 0.94, "B")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    style(fig, width=1600, height=680).update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    save(fig, "fig3_ridl")


def _stretch(v, lo=2, hi=98):
    a, b = np.percentile(v, [lo, hi])
    return np.clip((v - a) / (b - a + 1e-12), 0, 1)


def _spread(P, k):
    """Farthest-point sampling: k well-separated indices of the path P (n x 3)."""
    idx = [int(np.argmax(P[:, 2]))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return sorted(idx)


def fig_rgb_trajectories(subjects=(0, 8), task="Task1", n_labels=5):
    """visualsThesis.ipynb cell 2 (thesis Fig. 4.6) with the real CFS."""
    panels = [(s, c) for s in subjects for c in ("VR", "NVR")]
    fig = make_subplots(rows=len(subjects), cols=2, specs=[[{"type": "scene"}] * 2] * len(subjects),
                        subplot_titles=[f"Subject {s} · {task} · {'VR + haptics' if c == 'VR' else '2D / no haptics'}"
                                        for s, c in panels], horizontal_spacing=0.0, vertical_spacing=0.04)
    for k, (s, c) in enumerate(panels):
        d = load_lgaf(s, c, task)
        x, y, z = (d[a].to_numpy(float) * 1000 for a in ("x", "y", "z"))
        F = d.F.to_numpy(float)
        Fn, On = _stretch(F), _stretch(optimality(d))
        rgb = [f"rgb({int(255 * (1 - f))},{int(255 * f)},{int(255 * o)})" for f, o in zip(Fn, On)]
        R = np.clip(F / F.mean(), 0, 2) / 2 + 0.1 * optimality(d)   # reward the demonstration itself receives
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers+lines", marker=dict(size=3, color=rgb, opacity=0.85),
                                   line=dict(color="black", width=1), showlegend=False), row=k // 2 + 1, col=k % 2 + 1)
        fig.layout[f"scene{k + 1 if k else ''}"].update(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
            annotations=[dict(x=x[i], y=y[i], z=z[i], text=f"Reward: {R[i]:.2f}", showarrow=True, arrowhead=1,
                              ax=30, ay=-30, font=dict(size=10)) for i in _spread(np.c_[x, y, z], n_labels)])
    style(fig, "Weighted Real-Time Trajectory Visualization · colour = (1−F, F, O): green = high CFS, "
               "red = low CFS, bluer = smoother", width=1400, height=560 * len(subjects))
    fig.update_layout(margin=dict(l=0, r=0, t=80, b=0))
    save(fig, "fig4a_rgb_trajectories", scale=2)


def fig_paired():
    subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
    tests = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
    panels = [("cfs", "Cognitive Flow Score"), ("cfs_cv", "CFS variability (CV)"), ("optimality", "Jerk optimality"),
              ("ldlj", "Smoothness (LDLJ)"), ("micro_deviation", "Micro-deviation (mm)"), ("path_efficiency", "Path efficiency")]
    titles = [f"{lab}<br><sup>d<sub>z</sub> = {tests.loc[m].dz:.2f}, p = {tests.loc[m].p_t:.2f}</sup>" for m, lab in panels]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, vertical_spacing=0.16, horizontal_spacing=0.08)
    for k, (m, _) in enumerate(panels):
        r, c = k // 3 + 1, k % 3 + 1
        w = subj.pivot(index="subject", columns="cond", values=m) * (1000 if m == "micro_deviation" else 1)
        for _, row in w.iterrows():
            fig.add_trace(go.Scatter(x=["2D", "VR"], y=[row.NVR, row.VR], mode="lines", line=dict(color=GRAY, width=1),
                                     showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=["2D"] * len(w), y=w.NVR, mode="markers", marker=dict(color=NVR, size=9),
                                 name="2D / no haptics", showlegend=k == 0), row=r, col=c)
        fig.add_trace(go.Scatter(x=["VR"] * len(w), y=w.VR, mode="markers", marker=dict(color=VR, size=9),
                                 name="VR + haptics", showlegend=k == 0), row=r, col=c)
        fig.add_trace(go.Scatter(x=["2D", "VR"], y=[w.NVR.mean(), w.VR.mean()], mode="lines+markers",
                                 line=dict(color="black", width=4), marker=dict(size=12, color="black"),
                                 name="mean", showlegend=k == 0), row=r, col=c)
    style(fig, "Participant-level Metrics: VR + Haptics vs 2D", width=1300, height=850)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08))
    save(fig, "fig_human_paired")


def fig_cfs_quality(n_bins=5):
    """(A) Binned within-participant relationship: CFS and optimality are centred on each participant's block
    (interface x task) means, windows are binned into CFS quintiles within each participant, and the per-bin
    optimality deviation is averaged per participant, then mean +/- 95% CI across participants (VR and 2D pooled).
    (B) Per-participant standardized slopes (95% CI) with the pooled random-slope estimate."""
    from scipy import stats
    win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))
    mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
    o, rs = mm["mixed"]["opt"], mm["mixed"]["opt_robustness"]["random_slope"]
    g = ["subject", "cond", "seg"]
    win["cfs_w"] = win.cfs - win.groupby(g).cfs.transform("mean")      # CFS relative to the participant's own block mean
    win["dopt"] = win.opt - win.groupby(g).opt.transform("mean")
    win["bin"] = win.groupby("subject").cfs_w.transform(lambda v: pd.qcut(v.rank(method="first"), n_bins, labels=False))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12, subplot_titles=[
        f"Optimality across each participant's CFS range<br><sup>mixed model: β = {o['beta']:.2f} SD/SD, p = {o['p']:.3f}; "
        f"no CFS × interface interaction</sup>",
        f"Coupling per participant<br><sup>standardized slope ± 95% CI; pooled random-slope β = {rs['beta']:.2f}, p = {rs['p']:.2f}</sup>"])
    xs = np.arange(1, n_bins + 1)
    p = win.groupby(["subject", "bin"]).dopt.mean().unstack()
    m, h = p.mean().values, (p.sem() * stats.t.ppf(0.975, len(p) - 1)).values
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[m + h, (m - h)[::-1]], fill="toself", fillcolor="rgba(0,204,150,0.18)", mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=m, mode="lines+markers", name="mean ± 95% CI across participants (VR and 2D pooled)",
                             line=dict(color=GREEN, width=4), marker=dict(size=12, color=GREEN)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=1, col=1)
    fig.update_xaxes(tickvals=xs, ticktext=["lowest", "2", "3", "4", "highest"][:n_bins] if n_bins == 5 else None,
                     title_text="CFS quintile (within participant)", row=1, col=1)
    fig.update_yaxes(title_text="Δ jerk optimality vs. participant mean", row=1, col=1)
    # per-participant slopes on z-scored, within-participant-centred variables
    d = win.copy()
    for v in ("cfs", "opt"):
        d[v + "_c"] = d[v] - d.groupby(g)[v].transform("mean")
        d[v + "_c"] /= d[v].std()
    rows = []
    for sid, w in d.groupby("subject"):
        r = stats.linregress(w.cfs_c, w.opt_c)
        hw = r.stderr * stats.t.ppf(0.975, len(w) - 2)
        rows.append((sid, r.slope, r.slope - hw, r.slope + hw))
    S = pd.DataFrame(rows, columns=["subject", "b", "lo", "hi"]).sort_values("b")
    labels = [f"P{int(s) + 1}" for s in S.subject]
    cols = [GREEN if lo > 0 else ("#EF553B" if hi < 0 else GRAY) for lo, hi in zip(S.lo, S.hi)]
    fig.add_trace(go.Scatter(x=S.b, y=labels, mode="markers", marker=dict(size=11, color=cols), showlegend=False,
                             error_x=dict(type="data", symmetric=False, array=S.hi - S.b, arrayminus=S.b - S.lo,
                                          thickness=1.6, width=0, color="#7f7f7f")), row=1, col=2)
    hw = 1.96 * rs.get("se", 0)
    fig.add_trace(go.Scatter(x=[rs["beta"]], y=["pooled"], mode="markers", marker=dict(symbol="diamond", size=18, color="black"),
                             error_x=dict(type="data", array=[hw], thickness=2.2, width=0, color="black"),
                             showlegend=False), row=1, col=2)
    fig.add_vline(x=0, line=dict(color="gray", dash="dot", width=1), row=1, col=2)
    fig.update_yaxes(categoryorder="array", categoryarray=["pooled"] + labels, row=1, col=2)
    fig.update_xaxes(title_text="Standardized slope (SD optimality per SD CFS)", row=1, col=2)
    style(fig, width=1400, height=600).update_layout(legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"))
    save(fig, "fig4b_cfs_quality")


def fig_confusion():
    """visualsThesis.ipynb cell 9 (thesis Fig. 6.1): Blues, raw counts."""
    cm = np.array([[9292, 3917, 5141], [802, 27153, 6079], [1916, 8009, 25544]])
    cls = ["rest", "oddball", "msit"]
    fig = go.Figure(go.Heatmap(z=cm, x=cls, y=cls, colorscale="Blues", colorbar=dict(title="Count"),
                               text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                               textfont=dict(size=16)))
    fig.update_yaxes(autorange="reversed", title="True Label"); fig.update_xaxes(title="Predicted Label", tickangle=-45)
    style(fig, "Confusion Matrix", width=650, height=560)
    save(fig, "fig_lgaf_confusion")




def fig_lgaf_maps():
    """LGAF (GADF) maps of the decision-making network nodes, high vs low CFS windows."""
    from lgaf_maps import NODES
    Z = np.load(os.path.join(RESULTS, "lgaf_maps.npz"))
    st = pd.read_csv(os.path.join(RESULTS, "lgaf_diag_stats.csv"))
    cols = [("VR", "high"), ("VR", "low"), ("VR", "diff"), ("NVR", "high"), ("NVR", "low"), ("NVR", "diff")]
    head = {"high": "high CFS", "low": "low CFS", "diff": "high − low"}
    titles = [f"{'VR + haptics' if c == 'VR' else '2D'} · {head[k]}" for c, k in cols]
    fig = make_subplots(rows=len(NODES), cols=6, subplot_titles=titles + [""] * (6 * (len(NODES) - 1)),
                        horizontal_spacing=0.012, vertical_spacing=0.035)
    vmax = max(np.abs(Z[f"{c}|{n}|{k}"]).max() for c in ("VR", "NVR") for n in NODES for k in ("high", "low"))
    dmax = max(np.abs(Z[f"{c}|{n}|high"] - Z[f"{c}|{n}|low"]).max() for c in ("VR", "NVR") for n in NODES)
    for r, node in enumerate(NODES, start=1):
        for j, (c, k) in enumerate(cols, start=1):
            m = Z[f"{c}|{node}|high"] - Z[f"{c}|{node}|low"] if k == "diff" else Z[f"{c}|{node}|{k}"]
            fig.add_trace(go.Heatmap(z=m, coloraxis="coloraxis2" if k == "diff" else "coloraxis"), row=r, col=j)
            if k == "diff":
                q = st[(st.cond == c) & (st.node == node)].iloc[0]
                fig.add_annotation(x=0.5, y=-0.06, xref=f"x{(r - 1) * 6 + j} domain", yref=f"y{(r - 1) * 6 + j} domain",
                                   text=f"diag Δ = {q['diff']:+.3f}, p = {q.p:.2f}", showarrow=False, font=dict(size=11))
        fig.add_annotation(x=-0.006, y=1 - (r - 0.5) / len(NODES), xref="paper", yref="paper", xanchor="right",
                           text=f"<b>{node.split(' (')[0]}</b><br>({node.split(' (')[1]}", showarrow=False,
                           font=dict(size=13), align="right")
    fig.update_xaxes(showticklabels=False, showgrid=False); fig.update_yaxes(showticklabels=False, showgrid=False,
                                                                             autorange="reversed", scaleanchor=None)
    style(fig, "LGAF maps of the decision-making network · GADF<sub>ij</sub> = sin(φ<sup>α</sup><sub>i</sub> − "
               "φ<sup>β</sup><sub>j</sub>), 64 × 6.4 s windows", width=1700, height=1150)
    fig.update_layout(margin=dict(l=230, r=40, t=110, b=40),
                      coloraxis=dict(colorscale="RdBu_r", cmin=-vmax, cmax=vmax,
                                     colorbar=dict(title="GADF", x=1.0, len=0.45, y=0.77)),
                      coloraxis2=dict(colorscale="PuOr_r", cmin=-dmax, cmax=dmax,
                                      colorbar=dict(title="Δ (high − low)", x=1.0, len=0.45, y=0.25)))
    save(fig, "fig_lgaf_maps", scale=2)


if __name__ == "__main__":
    fig_setup()
    fig_ridl()
    fig_rgb_trajectories()
    fig_paired()
    fig_cfs_quality()
    fig_confusion()
    fig_lgaf_maps()
    print("ok")
