"""Graphical abstract in Plotly: neuroscience rationale + pipeline + real-data findings.
Renders docs/visual_abstract.{png,pdf} (and figures/visual_abstract.png). Panel D shows
the RIDL ablation once results/ablation_summary.csv exists."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import FIGS, RESULTS, ROOT
from fig_decision_circuit import POS, RECORDED
from pstyle import FONT, GRAY, GREEN, NVR, VR, img_uri

INK, INK2 = "#191919", "#5f5f5f"
ART = os.path.join(FIGS, "thesis_art")
W, H = 1600, 1350
# circuit roles of the recorded electrodes (thesis Sec. 3.6)
ROLE = {"Fz": ("#E45756", "decision · goal maintenance"), "C3": ("#54A24B", "motor output"),
        "C4": ("#54A24B", "motor output"), "CP1": ("#2a9d8f", "parietal attention"),
        "CP2": ("#2a9d8f", "parietal attention"), "O1": ("#8c8c8c", "visual input"),
        "Oz": ("#8c8c8c", "visual input"), "O2": ("#8c8c8c", "visual input")}
STRIP = [("#636EFA", "<b>Teleoperation</b>  dVRK/AMBF · VR (Quest 3 + Falcon haptics) vs 2D monitor"),
         ("#00CC96", "<b>Demonstrations</b>  endoscope 64×64 + pose/force 10 Hz + CFS(t)"),
         ("#FFA15A", "<b>RIDL L1</b>  behaviour cloning: frozen SD-UNet + Transformer"),
         ("#EF553B", "<b>RIDL L2</b>  PPO, reward weighted by CFS · subject curriculum")]


def text(fig, x, y, s, size=13, color=INK, xa="left", ya="top", **kw):
    fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=s, showarrow=False, xanchor=xa, yanchor=ya,
                       align=kw.pop("align", "left"), font=dict(size=size, color=color), **kw)


def image(fig, path, x0, yc, w, h=None):
    uri, (iw, ih) = img_uri(path)
    h = w * (W - 70) * ih / iw / (H - 90); y1 = yc + h / 2
    fig.add_layout_image(dict(source=uri, xref="paper", yref="paper", x=x0, y=y1, sizex=w, sizey=h,
                              xanchor="left", yanchor="top", sizing="contain", layer="above"))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x0 + w, y0=y1 - h, y1=y1,
                  line=dict(color="#d9dce1", width=1))


def arrow(fig, x0, x1, y, label=None):
    fig.add_annotation(x=x1, y=y, ax=-(x1 - x0) * (W - 70), ay=0, xref="paper", yref="paper", axref="pixel",
                       ayref="pixel", text="",
                       showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2, arrowcolor="#9aa0a6")
    if label:
        text(fig, (x0 + x1) / 2, y + 0.008, label, size=11.5, color=INK2, xa="center", ya="bottom", align="center")


def section(fig, y, s):
    text(fig, 0.0, y, f"<b>{s}</b>", size=19, ya="bottom")
    fig.add_shape(type="line", xref="paper", yref="paper", x0=0, x1=1, y0=y - 0.004, y1=y - 0.004,
                  line=dict(color="#d9dce1", width=1))


def scalp(fig, dom_x, dom_y):
    """Decision-making network on the recorded montage, with the perception→decision→action flow."""
    fig.update_layout(xaxis5=dict(domain=dom_x, anchor="y5", range=[-1.25, 2.55], visible=False),
                      yaxis5=dict(domain=dom_y, anchor="x5", range=[-1.2, 1.22], visible=False,
                                  scaleanchor="x5", scaleratio=1))
    kw = dict(xref="x5", yref="y5")
    fig.add_shape(type="rect", xref="x5 domain", yref="y5 domain", x0=0, x1=1, y0=0, y1=1, fillcolor="white",
                  line_width=0, layer="below")
    fig.add_shape(type="circle", x0=-1.08, x1=1.08, y0=-1.08, y1=1.08, line=dict(color="#5f6b7a", width=2),
                  fillcolor="#fbfbfc", layer="below", **kw)
    fig.add_shape(type="path", path="M -0.12 1.07 L 0 1.2 L 0.12 1.07", line=dict(color="#5f6b7a", width=2), **kw)
    flow = [("Oz", (0, -0.27)), ((0, -0.27), "Fz"), ("Fz", "C3"), ("Fz", "C4")]
    for a, b in flow:
        (xa, ya) = POS[a] if isinstance(a, str) else a
        (xb, yb) = POS[b] if isinstance(b, str) else b
        d = np.hypot(xb - xa, yb - ya); ux, uy = (xb - xa) / d, (yb - ya) / d
        s0, s1 = (0.13 if isinstance(a, str) else 0.0), 0.14
        fig.add_annotation(x=xb - ux * s1, y=yb - uy * s1, ax=xa + ux * s0, ay=ya + uy * s0, axref="x5", ayref="y5",
                           text="", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5,
                           arrowcolor="rgba(228,87,86,0.55)", **kw)
    for e, (x, y) in POS.items():
        if e in RECORDED:
            col = ROLE[e][0]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{e}</b>"], xaxis="x5", yaxis="y5",
                                     textfont=dict(size=10, color="white"), hoverinfo="skip",
                                     marker=dict(size=30, color=col, line=dict(color=INK, width=1.5))))
        else:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", xaxis="x5", yaxis="y5", hoverinfo="skip",
                                     marker=dict(size=9, color="#e6e8eb", line=dict(color="#c3c7cd", width=1))))
    for (x, y, col, s) in [(1.1, 0.62, "#E45756", "<b>Fz</b>  decision ·<br>goal maintenance"),
                           (1.1, 0.06, "#54A24B", "<b>C3/C4</b><br>motor output"),
                           (1.1, -0.42, "#2a9d8f", "<b>CP1/CP2</b>  parietal<br>attention, evidence"),
                           (1.1, -0.95, "#8c8c8c", "<b>O1/Oz/O2</b>  visual<br>input (endoscope)")]:
        fig.add_annotation(x=x + 0.08, y=y, text=s, showarrow=False, xanchor="left", yanchor="middle", align="left",
                           font=dict(size=11.5, color=col), **kw)


def main():
    subj = pd.read_csv(os.path.join(RESULTS, "human_subject_means.csv"))
    tests = pd.read_csv(os.path.join(RESULTS, "human_paired_tests.csv")).set_index("metric")
    mm = json.load(open(os.path.join(RESULTS, "human_models.json")))
    win = pd.read_csv(os.path.join(RESULTS, "human_windows.csv"))
    lg = pd.read_csv(os.path.join(RESULTS, "lgaf_diag_stats.csv"))
    abl = os.path.join(RESULTS, "ablation_summary.csv")
    has_abl = os.path.exists(abl)
    tc, o_, rs_ = tests.loc["cfs"], mm["mixed"]["opt"], mm["mixed"]["opt_robustness"]["random_slope"]
    teq = {m: tests.loc[m].p_tost_10pct for m in ("optimality", "ldlj")}

    FY = (0.05, 0.36)      # findings plot band
    fig = make_subplots(rows=1, cols=4, column_widths=[0.31, 0.19, 0.25, 0.25], horizontal_spacing=0.075,
                        subplot_titles=[
        "<b>A  VR vs 2D: behaviour</b><br><sup>VR − 2D as % of the 2D mean, 90% CI; green band = ±10%</sup>",
        f"<b>B  Engagement not reduced</b><br><sup>d<sub>z</sub> = {tc.dz:.2f}, p = {tc.p_t:.2f}, BF<sub>01</sub> = {tc.BF01:.1f}</sup>",
        f"<b>C  Engagement tracks control</b><br><sup>β = {o_['beta']:.2f}, p = {o_['p']:.3f}; random-slope p = {rs_['p']:.2f}</sup>",
        "<b>D  RIDL ablation</b><br><sup>held-out position error, mean ± 95% CI</sup>"])
    fig.update_layout(width=W, height=H, paper_bgcolor="white", plot_bgcolor="#E5ECF6", font=FONT, template="plotly",
                      margin=dict(l=40, r=30, t=30, b=60), showlegend=False)
    for ax in ("yaxis", "yaxis2", "yaxis3", "yaxis4"):
        fig.layout[ax].domain = list(FY)
    for a in fig.layout.annotations:
        a.update(y=FY[1] + 0.012, font=dict(size=15.5, color=INK), xanchor="center", yanchor="bottom")

    # ---------------- title
    text(fig, 0.0, 1.0, "<b>Does immersive VR–haptic surgical teleoperation leave the brain's decision-making "
                        "network undisturbed?</b>", size=25)
    text(fig, 0.0, 0.968, "Wearable EEG reads engagement from the decision network (Cognitive Flow Score, CFS); "
                          "a hierarchical diffusion-RL agent (RIDL) learns from CFS-weighted demonstrations · "
                          "within-subject pilot, N = 10", size=14.5, color=INK2)

    # ---------------- neuroscience row
    section(fig, 0.925, "1  Neuroscience rationale")
    ny0, ny1 = 0.585, 0.905
    text(fig, 0.0, ny1, "<b>a  The decision-making network, as sampled</b>", size=14.5)
    text(fig, 0.0, ny1 - 0.022, "perception → evidence → decision → action (thesis model)", size=12, color=INK2)
    scalp(fig, [0.0, 0.30], [ny0, ny1 - 0.045])

    x0 = 0.335
    text(fig, x0, ny1, "<b>b  From EEG to an engagement signal</b>", size=14.5)
    text(fig, x0, ny1 - 0.022, "engaged cortex: β (13–30 Hz) ↑, α (8–12 Hz) idling ↓", size=12, color=INK2)
    iy1, ih = ny1 - 0.06, 0.17
    yc = iy1 - ih / 2
    image(fig, os.path.join(ART, "data_eeg.png"), x0, yc, 0.125)
    arrow(fig, x0 + 0.128, x0 + 0.163, yc, "α/β<br>envelopes")
    image(fig, os.path.join(ART, "data_lgaf.png"), x0 + 0.166, yc, 0.085)
    arrow(fig, x0 + 0.254, x0 + 0.289, yc, "LGAF-<br>CNN")
    image(fig, os.path.join(ART, "data_cfs.png"), x0 + 0.292, yc, 0.11)
    cy = yc - 0.07
    text(fig, x0 + 0.0625, cy, "8-ch dry EEG, 500 Hz<br>over the network nodes", size=11.5, color=INK2,
         xa="center", align="center")
    text(fig, x0 + 0.2085, cy, "Gramian angular field:<br>α–β phase coupling<br>sin(φ<sub>α,i</sub> − φ<sub>β,j</sub>)",
         size=11.5, color=INK2, xa="center", align="center")
    text(fig, x0 + 0.347, cy, "CFS(t) = V<sub>β</sub>/V<sub>α</sub> · L<sub>oddball</sub>/L<sub>rest</sub><br>"
                              "active processing × attentive<br>(oddball-like) vs resting state", size=11.5,
         color=INK2, xa="center", align="center")

    hx0, hx1 = 0.765, 1.0
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=hx0, x1=hx1, y0=ny0 + 0.005, y1=ny1 + 0.002,
                  fillcolor="#f6f7f9", line=dict(color="#d9dce1"))
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=hx0, x1=hx1, y0=ny1 - 0.036, y1=ny1 + 0.002,
                  fillcolor="#AB63FA", line=dict(color="#AB63FA"))
    text(fig, (hx0 + hx1) / 2, ny1 - 0.017, "<b>c  Cognitive-transparency test</b>", size=14.5, color="white",
         xa="center", ya="middle")
    mx = lg.dz.abs().max()
    text(fig, hx0 + 0.01, ny1 - 0.048,
         "<b>H:</b> if VR + haptics added cognitive cost,<br>engagement of the decision network<br>"
         "(β/α, CFS) would drop vs a 2D monitor.<br><br>"
         f"• CFS <b>not reduced</b> in VR (BF<sub>01</sub> = {tc.BF01:.1f};<br>   equivalence not yet established)<br>"
         f"• movement quality <b>equivalent</b> (±10%,<br>   TOST p &lt; {max(teq.values()) if max(teq.values()) > 0.001 else 0.001:g})<br>"
         f"• CFS <b>tracks motor control</b> (β = {o_['beta']:.2f})<br>"
         f"• LGAF node maps: no node-specific<br>   effect (all |d<sub>z</sub>| ≤ {mx:.2f}, pilot power)<br><br>"
         "<b>→</b> RIDL: balanced error/volatility profile,<br>   beats PPO from scratch on average;<br>   limited by 16 demonstrations per fold",
         size=13.2)

    # ---------------- pipeline strip
    section(fig, 0.535, "2  Neuroadaptive learning pipeline")
    sw, sg, sy0, sy1 = 0.232, 0.024, 0.455, 0.515
    for i, (col, s) in enumerate(STRIP):
        xa = i * (sw + sg)
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=xa, x1=xa + sw, y0=sy0, y1=sy1,
                      fillcolor="#f6f7f9", line=dict(color=col, width=2))
        text(fig, xa + sw / 2, (sy0 + sy1) / 2, s.replace("  ", "<br>"), size=12.3, xa="center", ya="middle",
             align="center")
        if i < 3:
            arrow(fig, xa + sw + 0.003, xa + sw + sg - 0.003, (sy0 + sy1) / 2)

    section(fig, 0.425, "3  Findings")

    # ---------------- A: equivalence forest (90% CI of VR-2D, % of 2D mean)
    rows = [("cfs", "Engagement (CFS)"), ("optimality", "Jerk optimality"), ("ldlj", "Smoothness (LDLJ)"),
            ("micro_deviation", "Micro-deviation"), ("path_efficiency", "Path efficiency")][::-1]
    fig.add_vrect(x0=-10, x1=10, fillcolor="rgba(0,204,150,0.22)", line_width=0, layer="below", row=1, col=1)
    for m, lab in rows:
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

    # ---------------- B: paired CFS
    wv = subj.pivot(index="subject", columns="cond", values="cfs")
    for _, r in wv.iterrows():
        fig.add_trace(go.Scatter(x=["2D", "VR + haptics"], y=[r.NVR, r.VR], mode="lines",
                                 line=dict(color="#c4c4c4", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=["2D"] * len(wv), y=wv.NVR, mode="markers", marker=dict(color=NVR, size=10)), row=1, col=2)
    fig.add_trace(go.Scatter(x=["VR + haptics"] * len(wv), y=wv.VR, mode="markers", marker=dict(color=VR, size=10)),
                  row=1, col=2)
    fig.update_yaxes(title_text="Mean CFS per participant", row=1, col=2)

    # ---------------- C: CFS -> optimality, one slope per participant (forest plot)
    from scipy import stats as st_
    gk = ["subject", "cond", "seg"]; wc = win.copy()
    for v in ("cfs", "opt"):
        wc[v + "_c"] = (wc[v] - wc.groupby(gk)[v].transform("mean")) / wc[v].std()
    sl = []
    for sid, g in wc.groupby("subject"):
        r = st_.linregress(g.cfs_c, g.opt_c); h = r.stderr * st_.t.ppf(0.975, len(g) - 2)
        sl.append((f"P{int(sid) + 1}", r.slope, r.slope - h, r.slope + h))
    sl = pd.DataFrame(sl, columns=["p", "b", "lo", "hi"]).sort_values("b")
    fig.add_trace(go.Scatter(x=sl.b, y=sl.p, mode="markers",
                             marker=dict(size=9, color=[GREEN if l > 0 else GRAY for l in sl.lo]),
                             error_x=dict(type="data", symmetric=False, array=sl.hi - sl.b, arrayminus=sl.b - sl.lo,
                                          thickness=1.4, width=0, color="#8c8c8c")), row=1, col=3)
    fig.add_trace(go.Scatter(x=[rs_["beta"]], y=["pooled"], mode="markers", marker=dict(symbol="diamond", size=15, color=INK),
                             error_x=dict(type="data", array=[1.96 * rs_["se"]], thickness=2, width=0, color=INK)),
                  row=1, col=3)
    fig.add_vline(x=0, line=dict(color="gray", dash="dot", width=1), row=1, col=3)
    fig.update_yaxes(categoryorder="array", categoryarray=["pooled"] + list(sl.p), row=1, col=3)
    fig.update_xaxes(title_text="CFS → optimality slope (SD/SD)", row=1, col=3)

    # ---------------- D: ablation (if available)
    if has_abl:
        s = pd.read_csv(abl, index_col=0)
        keep = [v for v in ["Hold-position baseline", "w/o flow-weighted reward", "w/o CFS state input",
                            "Raw pixels (no SD prior)", "w/o SD-UNet (no vision)", "PPO from scratch", "BC only", "RIDL (full)"] if v in s.index]
        s = s.loc[keep]
        fig.add_trace(go.Bar(y=list(s.index), x=s.err_mm, orientation="h",
                             marker_color=["#FFA15A" if v == "RIDL (full)" else (GRAY if "baseline" in v else "#ffd0ab")
                                           for v in s.index],
                             error_x=dict(type="data", array=s.err_mm_ci, color=INK2)), row=1, col=4)
        fig.update_xaxes(title_text="Held-out error (mm)", row=1, col=4)
    else:
        fig.update_xaxes(visible=False, row=1, col=4); fig.update_yaxes(visible=False, row=1, col=4)
        d4 = fig.layout.xaxis4.domain
        text(fig, (d4[0] + d4[1]) / 2, 0.29, "Subject-wise 5-fold CV<br>× VR / 2D demonstrations<br><br>ablates: "
             "imitation prior,<br>PPO refinement, SD prior,<br>CFS reward & input,<br>subject curriculum<br><br>"
             "(results pending)", size=13, color=INK2, xa="center")
    text(fig, 0.0, -0.012, "VR always preceded 2D (order not counterbalanced); N = 10 pilot. Statistics: paired t / "
                           "Wilcoxon, TOST (±10%), JZS Bayes factors, linear mixed models. Brain–electrode mapping "
                           "follows the thesis decision-circuit model (Sec. 3.6).", size=11.5, color=INK2)

    out = os.path.join(ROOT, "docs"); os.makedirs(out, exist_ok=True)
    fig.write_image(os.path.join(out, "visual_abstract.png"), scale=2)
    fig.write_image(os.path.join(out, "visual_abstract.pdf"))
    fig.write_image(os.path.join(FIGS, "visual_abstract.png"), scale=2)
    print("ok")


if __name__ == "__main__":
    main()
