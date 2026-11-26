"""RIDL figures, all in Plotly:
  fig5_vis2_<src>_fold<k>   ThesisVis2.ipynb style: black background, 3D TRUE (white) / PPO (red) /
                            RIDL (blue) trajectories + per-step error, held-out participants
  anim_vis2_*.gif           the same, animated (Plotly frames rendered with Kaleido)
  fig_ablation_<tag>        held-out error per variant + paired differences vs RIDL
  fig_crossdomain_<tag>     training-interface x test-interface error (Blues heatmap)
  fig_learning_curves       held-out NMSE vs PPO steps, RIDL vs PPO from scratch
Also writes results/<tag>_summary.csv (read by make_macros.py).
"""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import FIGS, RESULTS
from pstyle import GRAY, NVR, VR, save, style

TRUE_C, PPO_C, RIDL_C = "white", "red", "#3f7bff"


def tfile(variant, src, fold):
    return os.path.join(RESULTS, "trajectories", f"{variant.replace('/', '_').replace(' ', '_')}__{src}__f{fold}.npz")


# ------------------------------------------------------------------ vis2 (black)
def _black(fig, n_rows):
    fig.update_layout(paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white", size=13),
                      legend=dict(bgcolor="black", font=dict(color="white")))
    fig.update_xaxes(color="white", gridcolor="#333", zerolinecolor="#555")
    fig.update_yaxes(color="white", gridcolor="#333", zerolinecolor="#555")
    for k in range(n_rows):
        sc = f"scene{k + 1 if k else ''}"
        ax = dict(backgroundcolor="#777777", gridcolor="#aaaaaa", color="white", showbackground=True)
        fig.layout[sc].update(xaxis=dict(ax, title="X (mm)"), yaxis=dict(ax, title="Y (mm)"),
                              zaxis=dict(ax, title="Z (mm)"), bgcolor="black")


def _vis2_traces(fig, row, key, ridl, ppo, n=None, legend=True):
    tr, pr, pp = (a[key + s] * 1000 for a, s in ((ridl, "__true"), (ridl, "__pred"), (ppo, "__pred")))
    er, ep = ridl[key + "__err"], ppo[key + "__err"]
    n = len(tr) if n is None else n
    for arr, col, name in ((tr, TRUE_C, "TRUE"), (pp, PPO_C, "PPO"), (pr, RIDL_C, "RIDL")):
        fig.add_trace(go.Scatter3d(x=arr[:n, 0], y=arr[:n, 1], z=arr[:n, 2], mode="lines", line=dict(color=col, width=5),
                                   name=name, showlegend=legend), row=row, col=1)
    k = np.arange(n)
    fig.add_trace(go.Scatter(x=k, y=ep[:n], mode="lines", line=dict(color=PPO_C, width=2),
                             name="ERR_PPO", showlegend=legend), row=row, col=2)
    fig.add_trace(go.Scatter(x=k, y=er[:n], mode="lines", line=dict(color=RIDL_C, width=2.5),
                             name="ERR_RIDL", showlegend=legend), row=row, col=2)
    fig.update_xaxes(title_text="Step (0.1 s)", row=row, col=2)
    fig.update_yaxes(title_text="‖x̂ − x‖ (mm)", row=row, col=2)


def fig_vis2(src="VR", fold=4, ridl_v="RIDL (full)", ppo_v="PPO from scratch", task="Task1"):
    ridl, ppo = np.load(tfile(ridl_v, src, fold)), np.load(tfile(ppo_v, src, fold))
    keys = sorted({k.rsplit("__", 1)[0] for k in ridl.files if task in k})
    fig = make_subplots(rows=len(keys), cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.06,
                        vertical_spacing=0.08, specs=[[{"type": "scene"}, {"type": "xy"}]] * len(keys),
                        subplot_titles=[t for k in keys for t in (
                            k.replace("_", " · ") + " (held-out)",
                            f"Error · PPO {ppo[k + '__err'].mean():.1f} mm · RIDL {ridl[k + '__err'].mean():.1f} mm")])
    for i, k in enumerate(keys):
        _vis2_traces(fig, i + 1, k, ridl, ppo, legend=i == 0)
    style(fig, f"RIDL vs PPO on held-out participants (trained on {src} demonstrations)", width=1400,
          height=520 * len(keys))
    _black(fig, len(keys))
    fig.update_annotations(font=dict(color="white"))
    save(fig, f"fig5_vis2_{src}_fold{fold}", scale=2)
    return keys


def gif_vis2(src="VR", fold=4, ridl_v="RIDL (full)", ppo_v="PPO from scratch", n_frames=45, key=None):
    from PIL import Image
    ridl, ppo = np.load(tfile(ridl_v, src, fold)), np.load(tfile(ppo_v, src, fold))
    key = key or sorted({k.rsplit("__", 1)[0] for k in ridl.files if "Task1" in k})[0]
    N = len(ridl[key + "__err"])
    allp = np.concatenate([ridl[key + "__true"], ridl[key + "__pred"], ppo[key + "__pred"]]) * 1000
    emax = max(ridl[key + "__err"].max(), ppo[key + "__err"].max()) * 1.05
    frames = []
    for n in np.linspace(5, N, n_frames).astype(int):
        fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], specs=[[{"type": "scene"}, {"type": "xy"}]])
        _vis2_traces(fig, 1, key, ridl, ppo, n=n)
        style(fig, f"{key.replace('_', ' · ')} · mean error PPO {ppo[key + '__err'].mean():.1f} mm, "
                   f"RIDL {ridl[key + '__err'].mean():.1f} mm", width=1000, height=500)
        _black(fig, 1)
        fig.layout.scene.update(xaxis_range=[allp[:, 0].min(), allp[:, 0].max()],
                                yaxis_range=[allp[:, 1].min(), allp[:, 1].max()],
                                zaxis_range=[allp[:, 2].min(), allp[:, 2].max()], aspectmode="cube")
        fig.update_xaxes(range=[0, N], row=1, col=2); fig.update_yaxes(range=[0, emax], row=1, col=2)
        frames.append(Image.open(io.BytesIO(fig.to_image(format="png"))).convert("RGB"))
    frames[0].save(os.path.join(FIGS, f"anim_vis2_{key}_{src}.gif"), save_all=True, append_images=frames[1:],
                   duration=120, loop=0)


# ------------------------------------------------------------------ ablation summary
ORDER = ["RIDL (full)", "PPO from scratch", "BC only", "w/o SD-UNet (no vision)", "Raw pixels (no SD prior)",
         "w/o flow-weighted reward", "w/o CFS state input", "Pooled (no curriculum)",
         "Constant-velocity baseline", "Hold-position baseline", "Mean-pose baseline"]


def summarize(df):
    run = df.groupby(["variant", "source", "fold", "seed"]).mean(numeric_only=True).reset_index()
    g = run.groupby("variant")
    out = g[["err_mm", "E_traj", "sigma_r", "nmse", "pred_jerk_mm", "env_steps"]].mean()
    n = g.size()
    for m in ("err_mm", "E_traj", "nmse"):
        out[m + "_ci"] = 1.96 * g[m].std().fillna(0) / np.sqrt(n)
    out["n_runs"] = n
    return out.reindex([o for o in ORDER if o in out.index]), run


def fig_ablation(tag="ablation"):
    df = pd.read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    s, run = summarize(df)
    s.to_csv(os.path.join(RESULTS, f"{tag}_summary.csv"))
    piv = run.pivot_table(index=["source", "fold", "seed"], columns="variant", values="err_mm")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=["RIDL Ablation: Held-out Error (mm)", "Paired Difference vs RIDL (mm)"])
    names = list(s.index)[::-1]
    cols = [VR if v == "RIDL (full)" else (GRAY if "baseline" in v else "#a3abf8") for v in names]
    fig.add_trace(go.Bar(y=names, x=s.err_mm[names], orientation="h", marker_color=cols,
                         error_x=dict(type="data", array=s.err_mm_ci[names], color="#444"),
                         text=[f"{v:.2f}" for v in s.err_mm[names]], textposition="outside", showlegend=False),
                  row=1, col=1)
    if "RIDL (full)" in piv:
        for v in names:
            if v != "RIDL (full)" and v in piv:
                dv = (piv[v] - piv["RIDL (full)"]).dropna()
                fig.add_trace(go.Box(x=dv, y=[v] * len(dv), orientation="h", boxpoints="all", jitter=0.4,
                                     marker=dict(color=VR, size=5), line=dict(color="#444"), showlegend=False),
                              row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    style(fig, width=1400, height=620).update_layout(margin=dict(l=230))
    save(fig, f"fig_ablation_{tag}")
    return s


def fig_crossdomain(tag="ablation", variant="RIDL (full)"):
    df = pd.read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    d = df[df.variant == variant].groupby(["source", "cond"]).err_mm.agg(["mean", "std"]).reset_index()
    M = d.pivot(index="source", columns="cond", values="mean").loc[["VR", "NVR"], ["VR", "NVR"]]
    S = d.pivot(index="source", columns="cond", values="std").loc[["VR", "NVR"], ["VR", "NVR"]]
    fig = go.Figure(go.Heatmap(z=M.values, x=["VR test", "2D test"], y=["VR-trained", "2D-trained"], colorscale="Blues",
                               colorbar=dict(title="Error (mm)"),
                               text=[[f"{M.values[i, j]:.2f} ± {S.values[i, j]:.2f}" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}", textfont=dict(size=18)))
    fig.update_yaxes(autorange="reversed", title="Training demonstrations"); fig.update_xaxes(title="Test condition")
    style(fig, f"Train × Test Condition ({variant})", width=650, height=520)
    save(fig, f"fig_crossdomain_{tag}")
    return M


def fig_curves(tag="curves"):
    f = os.path.join(RESULTS, f"{tag}.json")
    if not os.path.exists(f):
        return None
    C = json.load(open(f))
    fig = go.Figure()
    for (name, runs), col, fill in zip(C.items(), [VR, "#EF553B"], ["rgba(99,110,250,0.2)", "rgba(239,85,59,0.2)"]):
        L = min(len(r) for r in runs)
        st = np.array([[p["steps"] for p in r[:L]] for r in runs]).mean(0) / 1000
        y = np.array([[p["test_nmse"] for p in r[:L]] for r in runs])
        m, ci = y.mean(0), 1.96 * y.std(0) / np.sqrt(len(y))
        fig.add_trace(go.Scatter(x=np.r_[st, st[::-1]], y=np.r_[m + ci, (m - ci)[::-1]], fill="toself", fillcolor=fill,
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=st, y=m, mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=5),
                                 name=name))
    fig.update_xaxes(title="PPO environment steps (×1000)"); fig.update_yaxes(title="Held-out normalised MSE")
    style(fig, "Held-out NMSE Across PPO Steps", width=1000, height=600)
    save(fig, "fig_learning_curves")


# ------------------------------------------------------------------ Kalman residuals
def _kalman(z, q=1e-5, r=0.1):
    x, P, out = z[0], 1.0, []
    for v in z:
        P += q; K = P / (P + r); x += K * (v - x); P *= (1 - K); out.append(x)
    return np.array(out)


def fig_kalman(src="VR", fold=4, ridl_v="RIDL (full)", ppo_v="PPO from scratch", tag="ablation", subject=None):
    """Raw residual r_t = ||x_hat - x||, Kalman-smoothed r~_t (thesis Sec. 4.5.2) and the
    E_traj / sigma_r plane over all runs."""
    ridl, ppo = np.load(tfile(ridl_v, src, fold)), np.load(tfile(ppo_v, src, fold))
    keys = sorted({k.rsplit("__", 1)[0] for k in ridl.files if "Task1" in k})
    subj = subject or keys[0].split("-")[0]
    keys = [k for k in keys if k.startswith(subj + "-")][:2]
    fig = make_subplots(rows=2, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.08, vertical_spacing=0.12,
                        specs=[[{}, {"rowspan": 2}], [{}, None]],
                        subplot_titles=[keys[0].replace("_", " · ") + " (held-out)", "Error vs volatility (all runs)",
                                        keys[1].replace("_", " · ") + " (held-out)" if len(keys) > 1 else ""])
    for r, k in enumerate(keys, start=1):
        for arr, col, name, fill in ((ppo, "#EF553B", "PPO from scratch", "rgba(239,85,59,0.15)"),
                                     (ridl, VR, "RIDL", "rgba(99,110,250,0.18)")):
            e = arr[k + "__err"]; kf = _kalman(e); sd = np.sqrt(((kf - kf.mean()) ** 2).mean()); t = np.arange(len(e))
            fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[kf + sd, (kf - sd).clip(0)[::-1]], fill="toself",
                                     fillcolor=fill, line=dict(width=0), hoverinfo="skip", showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=t, y=e, mode="lines", line=dict(color=col, width=1), opacity=0.35,
                                     name=f"{name}: raw r<sub>t</sub>", showlegend=r == 1), row=r, col=1)
            fig.add_trace(go.Scatter(x=t, y=kf, mode="lines", line=dict(color=col, width=3),
                                     name=f"{name}: Kalman r̃<sub>t</sub> (E<sub>traj</sub> {kf.mean():.1f} mm, "
                                          f"σ<sub>r</sub> {sd:.1f})" if r == 1 else f"{name} r̃", showlegend=r == 1),
                          row=r, col=1)
        fig.update_yaxes(title_text="residual (mm)", row=r, col=1)
    fig.update_xaxes(title_text="Step (0.1 s)", row=2, col=1)
    f = os.path.join(RESULTS, f"{tag}.csv")
    if os.path.exists(f):
        run = pd.read_csv(f).groupby(["variant", "source", "fold", "seed"])[["E_traj", "sigma_r"]].mean().reset_index()
        palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#9e9e9e",
                   "#555555", "#bbbbbb"]
        for v, col in zip([o for o in ORDER if o in set(run.variant)], palette):
            d = run[run.variant == v]
            fig.add_trace(go.Scatter(x=d.E_traj, y=d.sigma_r, mode="markers", name=v,
                                     marker=dict(color=col, size=9 if v == "RIDL (full)" else 6,
                                                 symbol="diamond" if "baseline" in v else "circle",
                                                 line=dict(width=1, color="white")), legendgroup="plane"), row=1, col=2)
        fig.update_xaxes(title_text="E<sub>traj</sub> (mm)", row=1, col=2)
        fig.update_yaxes(title_text="σ<sub>r</sub> (mm)", row=1, col=2)
    style(fig, "Kalman-filtered residuals on held-out participants", width=1500, height=820)
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.0, font=dict(size=11)))
    save(fig, f"fig_kalman_{src}_fold{fold}")


if __name__ == "__main__":
    print(fig_ablation())
    print(fig_crossdomain())
    fig_curves()
    for src in ("VR", "NVR"):
        fig_vis2(src, 4)
    fig_kalman("VR", 4)
    gif_vis2("VR", 4)
