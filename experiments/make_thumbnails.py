"""Real-data thumbnails embedded in the architecture diagram (subject 0, VR, Task 1)."""
import glob
import os

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt

from common import CACHE, DATA, FIGS, merged_segment, optimality, seg_name
from lgaf_maps import CHANNELS, FS, band_env, gadf

OUT = os.path.join(FIGS, "thesis_art")
S, C, T = 0, "VR", "Task1"


def thumb(fig, name, w, h):
    fig.update_layout(width=w, height=h, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="white",
                      plot_bgcolor="white", showlegend=False)
    fig.write_image(os.path.join(OUT, f"data_{name}.png"), scale=3)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = merged_segment(S, C, T)
    i = len(m) // 3
    frame = cv2.cvtColor(cv2.imread(m.image_path.iloc[i]), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(OUT, "data_frame.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, "data_frame64.png"),
                cv2.cvtColor(cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))

    # kinematics: position (mm) + force magnitude over 30 s
    k = m.iloc[:300]
    t = k.timestamp - k.timestamp.iloc[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for a, col in zip("xyz", ("#636EFA", "#EF553B", "#00CC96")):
        fig.add_trace(go.Scatter(x=t, y=(k[a] - k[a].mean()) * 1000, line=dict(color=col, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.linalg.norm(k[["fx", "fy", "fz"]].to_numpy(float), axis=1),
                             line=dict(color="#FFA15A", width=1.5)), row=2, col=1)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    thumb(fig, "kinematics", 300, 170)

    # jerk optimality
    O = optimality(m)[:300]
    fig = go.Figure(go.Scatter(x=t, y=O, line=dict(color="#FFA15A", width=1.5), fill="tozeroy",
                               fillcolor="rgba(255,161,90,0.25)"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, range=[0.6, 1.02])
    thumb(fig, "optimality", 300, 110)

    # EEG 8 channels, 3 s, 1-40 Hz
    f = glob.glob(os.path.join(DATA, "NIC", f"Subject-{S}", f"*_subject{S}-*-{C}_{T}.easy"))[0]
    raw = np.loadtxt(f, max_rows=FS * 40)
    x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=FS, output="sos"), raw[:, :8] / 1000.0, axis=0)[FS * 20:FS * 23]
    fig = go.Figure()
    for j in range(8):
        sig = x[:, j] / (np.std(x[:, j]) * 6)
        fig.add_trace(go.Scatter(x=np.arange(len(sig)) / FS, y=sig - j, line=dict(color="#AB63FA", width=1)))
        fig.add_annotation(x=-0.02, y=-j, text=CHANNELS[j], xref="x", yref="y", showarrow=False, xanchor="right",
                           font=dict(size=9, color="#444"))
    fig.update_xaxes(visible=False, range=[-0.35, 3]); fig.update_yaxes(visible=False)
    thumb(fig, "eeg", 300, 190)

    # LGAF map (Fz) for one 6.4 s window on the CFS timeline
    ts = raw[:, 12] / 1000.0
    t10 = m.timestamp.to_numpy(float)
    t10 = t10[(t10 >= ts[0]) & (t10 <= ts[-1])][:64]
    fz = raw[:, CHANNELS.index("Fz")] / 1000.0
    a, b = (np.interp(t10, ts, band_env(fz, *bd)) for bd in ((8, 12), (13, 30)))
    fig = go.Figure(go.Heatmap(z=gadf(a, b), colorscale="RdBu_r", zmid=0, showscale=False))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
    thumb(fig, "lgaf", 170, 170)

    # CFS(t)
    F = m.F.to_numpy(float)[:300]
    fig = go.Figure(go.Scatter(x=t, y=F, line=dict(color="#00CC96", width=1.6), fill="tozeroy",
                               fillcolor="rgba(0,204,150,0.2)"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, range=[0, np.percentile(F, 99) * 1.1])
    thumb(fig, "cfs", 300, 100)

    # frozen SD-UNet feature maps (4 x 64 x 64) of the same frame, from the cache
    z = np.load(os.path.join(CACHE, seg_name(S, C, T) + ".npz"))
    feat = z["feats"][i].astype(np.float32).reshape(4, 64, 64)
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.03, vertical_spacing=0.03)
    for c in range(4):
        fig.add_trace(go.Heatmap(z=feat[c], colorscale="Viridis", showscale=False), row=c // 2 + 1, col=c % 2 + 1)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, autorange="reversed")
    thumb(fig, "unet", 200, 200)
    print("thumbnails ->", OUT)


if __name__ == "__main__":
    main()
