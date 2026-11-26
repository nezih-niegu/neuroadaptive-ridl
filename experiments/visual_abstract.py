"""Graphical abstract in Plotly: pipeline schematic + real-data findings.
Renders docs/visual_abstract.{png,pdf} (and figures/visual_abstract.png). Panel D shows
the RIDL ablation once results/ablation_summary.csv exists."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import FIGS, RESULTS, ROOT
from pstyle import FONT, GRAY, GREEN, NVR, VR

INK, INK2 = "#191919", "#5f5f5f"
BOX = [("1  Embodied teleoperation", "#636EFA",
        "dVRK / AMBF simulator<br>• VR: Meta Quest 3 stereo +<br>   Novint Falcon force feedback<br>"
        "• 2D: monitor, no haptics<br>Reference · Task 1 · Task 2"),
       ("2  Wearable EEG → CFS", "#AB63FA",
        "8-ch dry Enobio, 500 Hz<br>LGAF-CNN (Gramian angular<br>field) engagement decoder<br>"
        "CFS(t) = V<sub>β</sub>/V<sub>α</sub> · L<sub>oddball</sub>(t)/L<sub>rest</sub>(t)"),
       ("3  Multimodal demonstrations", "#00CC96",
        "endoscope frames (64×64)<br>+ tool pose & force (10 Hz)<br>+ CFS(t), jerk optimality<br>→ 1 s action chunks"),
       ("4  Hierarchical RIDL agent", "#FFA15A",
        "L1 skill prior: behaviour cloning<br>   frozen Stable-Diffusion UNet<br>   + Transformer kinematics<br>"
        "L2 refinement: PPO on a<br>   CFS-weighted reward<br>subject-by-subject curriculum")]


def main():
    subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
    tests = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
    mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
    win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))
    abl = os.path.join(RESULTS, "ablation_summary.csv")
    has_abl = os.path.exists(abl)

    tc, o_, rs_ = tests.loc["cfs"], mm["mixed"]["opt"], mm["mixed"]["opt_robustness"]["random_slope"]
    fig = make_subplots(rows=1, cols=4, column_widths=[0.31, 0.19, 0.25, 0.25], horizontal_spacing=0.075,
                        subplot_titles=[
        "<b>A  VR vs 2D: equivalence</b><br><sup>VR − 2D as % of the 2D mean, 90% CI; green band = ±10%</sup>",
        f"<b>B  Engagement preserved</b><br><sup>d<sub>z</sub> = {tc.dz:.2f}, p = {tc.p_t:.2f}, BF<sub>01</sub> = {tc.BF01:.1f}</sup>",
        f"<b>C  Engagement tracks control</b><br><sup>β = {o_['beta']:.2f}, p = {o_['p']:.3f}; random-slope p = {rs_['p']:.2f}</sup>",
        "<b>D  RIDL ablation</b><br><sup>held-out position error, mean ± 95% CI</sup>"])
    fig.update_layout(width=1600, height=900, paper_bgcolor="white", font=FONT, template="plotly",
                      margin=dict(l=40, r=30, t=30, b=40), showlegend=False)
    for ax in ("yaxis", "yaxis2", "yaxis3", "yaxis4"):
        fig.layout[ax].domain = [0.06, 0.50]
    for a in fig.layout.annotations:
        a.update(y=0.535, font=dict(size=16, color=INK), xanchor="center", yanchor="bottom")

    # ---- title + pipeline row (paper coordinates)
    fig.add_annotation(x=0.0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text="<b>Is immersive VR–haptic surgical teleoperation cognitively transparent?</b>",
                       font=dict(size=26, color=INK))
    fig.add_annotation(x=0.0, y=0.945, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text="Wearable-EEG Cognitive Flow Score (CFS) and a hierarchical diffusion-RL agent (RIDL) "
                            "on dVRK teleoperation · within-subject pilot, N = 10", font=dict(size=15, color=INK2))
    w, gap, y0, y1 = 0.228, 0.029, 0.685, 0.885
    for i, (title, col, body) in enumerate(BOX):
        x0 = i * (w + gap)
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x0 + w, y0=y0, y1=y1,
                      fillcolor="#f6f7f9", line=dict(color="#d9dce1"))
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x0 + w, y0=y1 - 0.045, y1=y1,
                      fillcolor=col, line=dict(color=col))
        fig.add_annotation(x=x0 + w / 2, y=y1 - 0.0225, xref="paper", yref="paper", text=f"<b>{title}</b>",
                           xanchor="center", yanchor="middle", showarrow=False, font=dict(size=15, color="white"))
        fig.add_annotation(x=x0 + 0.012, y=y1 - 0.06, xref="paper", yref="paper", text=body, align="left",
                           xanchor="left", yanchor="top", showarrow=False, font=dict(size=13.5, color=INK))
        if i < 3:
            fig.add_annotation(x=x0 + w + gap - 0.003, y=(y0 + y1) / 2, ax=-int((gap - 0.008) * 1600), ay=0,
                               xref="paper", yref="paper", axref="pixel", ayref="pixel", text="", showarrow=True,
                               arrowhead=2, arrowsize=1.4, arrowwidth=2, arrowcolor="#9aa0a6")
    fig.add_annotation(x=0.0, y=0.635, xref="paper", yref="paper", xanchor="left", showarrow=False,
                       text="<b>Findings</b>", font=dict(size=19, color=INK))

    # ---- A: equivalence forest (90% CI of VR-2D, % of 2D mean)
    rows = [("cfs", "Engagement (CFS)"), ("optimality", "Jerk optimality"), ("ldlj", "Smoothness (LDLJ)"),
            ("micro_deviation", "Micro-deviation"), ("path_efficiency", "Path efficiency")][::-1]
    fig.add_vrect(x0=-10, x1=10, fillcolor="rgba(0,204,150,0.22)", line_width=0, layer="below", row=1, col=1)
    for lab_i, (m, lab) in enumerate(rows):
        r = tests.loc[m]; base = abs(r.mean_nvr)
        lo, hi, mid = 100 * r.ci90_lo / base, 100 * r.ci90_hi / base, 100 * r["diff"] / base
        eq = r.p_tost_10pct < 0.05
        col = GREEN if eq else (VR if lo > 0 or hi < 0 else GRAY)
        tag = "equivalent" if eq else ("VR higher (uncorr.)" if lo > 0 else "inconclusive")
        fig.add_trace(go.Scatter(x=[mid], y=[lab], mode="markers", marker=dict(size=12, color=col),
                                 error_x=dict(type="data", symmetric=False, array=[hi - mid], arrayminus=[mid - lo],
                                              color=col, thickness=3, width=0)), row=1, col=1)
        fig.add_annotation(x=74, y=lab, text=tag, showarrow=False, xanchor="right", yshift=-14,
                           font=dict(size=11, color=col), row=1, col=1)
    fig.add_vline(x=0, line_color=INK2, line_width=1, row=1, col=1)
    fig.update_xaxes(range=[-90, 76], title_text="% of 2D mean", row=1, col=1)

    # ---- B: paired CFS
    wv = subj.pivot(index="subject", columns="cond", values="cfs")
    for _, r in wv.iterrows():
        fig.add_trace(go.Scatter(x=["2D", "VR + haptics"], y=[r.NVR, r.VR], mode="lines",
                                 line=dict(color="#c4c4c4", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=["2D"] * len(wv), y=wv.NVR, mode="markers", marker=dict(color=NVR, size=10)), row=1, col=2)
    fig.add_trace(go.Scatter(x=["VR + haptics"] * len(wv), y=wv.VR, mode="markers", marker=dict(color=VR, size=10)),
                  row=1, col=2)
    fig.update_yaxes(title_text="Mean CFS per participant", row=1, col=2)

    # ---- C: CFS -> optimality
    lo, hi = win.cfs.quantile([0.02, 0.98]); ww = win[(win.cfs >= lo) & (win.cfs <= hi)]
    fig.add_trace(go.Scatter(x=ww.cfs, y=ww.opt, mode="markers", marker=dict(color="#AB63FA", size=4, opacity=0.3)),
                  row=1, col=3)
    for _, g in ww.groupby("subject"):
        b = np.polyfit(g.cfs, g.opt, 1); x_ = np.linspace(*g.cfs.quantile([0.1, 0.9]), 5)
        fig.add_trace(go.Scatter(x=x_, y=np.polyval(b, x_), mode="lines", line=dict(color=INK, width=1), opacity=0.6),
                      row=1, col=3)
    o, rs = mm["mixed"]["opt"], mm["mixed"]["opt_robustness"]["random_slope"]
    fig.update_xaxes(title_text="Windowed CFS (5 s)", row=1, col=3)
    fig.update_yaxes(title_text="Jerk optimality", row=1, col=3)

    # ---- D: ablation (if available)
    if has_abl:
        s = pd.read_csv(abl, index_col=0)
        keep = [v for v in ["Hold-position baseline", "w/o flow-weighted reward", "Raw pixels (no SD prior)",
                            "PPO from scratch", "BC only", "RIDL (full)"] if v in s.index]
        s = s.loc[keep]
        fig.add_trace(go.Bar(y=list(s.index), x=s.err_mm, orientation="h",
                             marker_color=["#FFA15A" if v == "RIDL (full)" else (GRAY if "baseline" in v else "#ffd0ab")
                                           for v in s.index],
                             error_x=dict(type="data", array=s.err_mm_ci, color=INK2)), row=1, col=4)
        fig.update_xaxes(title_text="Held-out error (mm)", row=1, col=4)
    else:
        fig.update_xaxes(visible=False, row=1, col=4); fig.update_yaxes(visible=False, row=1, col=4)
        d4 = fig.layout.xaxis4.domain
        fig.add_annotation(x=(d4[0] + d4[1]) / 2, y=0.28, xref="paper", yref="paper", showarrow=False, align="left",
                           text="Subject-wise 5-fold CV<br>× VR / 2D demonstrations<br><br>ablates: imitation prior,<br>"
                                "PPO refinement, SD prior,<br>CFS reward & input,<br>subject curriculum<br><br>(results pending)",
                           font=dict(size=13, color=INK2))
    fig.add_annotation(x=0.0, y=-0.035, xref="paper", yref="paper", xanchor="left", showarrow=False,
                       text="VR always preceded 2D (order not counterbalanced); N = 10 pilot. Statistics: paired t / "
                            "Wilcoxon, TOST (±10%), JZS Bayes factors, linear mixed models.", font=dict(size=11.5, color=INK2))

    out = os.path.join(ROOT, "docs"); os.makedirs(out, exist_ok=True)
    fig.write_image(os.path.join(out, "visual_abstract.png"), scale=2)
    fig.write_image(os.path.join(out, "visual_abstract.pdf"))
    fig.write_image(os.path.join(FIGS, "visual_abstract.png"), scale=2)
    print("ok")


if __name__ == "__main__":
    main()
