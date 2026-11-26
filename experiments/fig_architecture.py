"""Multimodal integration & RIDL architecture diagram (Plotly) -> figures/fig3_architecture.{png,pdf}."""
import os

import plotly.graph_objects as go

from common import FIGS

from pstyle import img_uri, save

ART = os.path.join(FIGS, "thesis_art")

VID, KIN, EEG, DER, POL, LRN, INK = "#636EFA", "#FFA15A", "#AB63FA", "#00CC96", "#19A7CE", "#EF553B", "#1f2a37"


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def rbox(fig, x0, y0, x1, y1, color, title=None, body=None, fill=0.12, dash=None, tsize=16, bsize=14, r=1.2):
    path = (f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} "
            f"L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z")
    fig.add_shape(type="path", path=path, fillcolor=_rgba(color, fill), line=dict(color=color, width=2, dash=dash),
                  layer="below")
    if title:
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 0.4, text=f"<b>{title}</b>", showarrow=False, yanchor="top",
                           font=dict(size=tsize, color=INK))
    if body:
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 - (1.2 if title else 0), text=body, showarrow=False,
                           font=dict(size=bsize, color=INK), align="center")


def arrow(fig, x0, y0, x1, y1, color=INK, label=None, width=2, dash=False, lpos=0.5, lshift=(0, 10), lsize=11):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                       arrowhead=2, arrowsize=1.2, arrowwidth=width, arrowcolor=color, opacity=0.65 if dash else 1)
    if label:
        fig.add_annotation(x=x0 + (x1 - x0) * lpos, y=y0 + (y1 - y0) * lpos, text=label, showarrow=False,
                           xshift=lshift[0], yshift=lshift[1], font=dict(size=lsize, color=color),
                           bgcolor="rgba(255,255,255,0.85)")


def line(fig, pts, color, dash="dot", width=2):
    xs, ys = zip(*pts)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, dash=dash, width=width),
                             hoverinfo="skip", showlegend=False))


def img(fig, name, x0, y0, x1, y1):
    uri, _ = img_uri(os.path.join(ART, f"data_{name}.png"))
    fig.add_layout_image(dict(source=uri, xref="x", yref="y", x=x0, y=y1, sizex=x1 - x0, sizey=y1 - y0,
                              xanchor="left", yanchor="top", sizing="contain", layer="above"))


def note(fig, x, y, text, size=12, color=INK, **kw):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=size, color=color), **kw)


def main():
    fig = go.Figure()
    fig.update_xaxes(range=[0, 200], visible=False); fig.update_yaxes(range=[0, 104], visible=False)
    for x, t in ((20, "1 · Sensing (raw data)"), (55, "2 · Per-stream processing"), (107, "3 · RIDL policy (RIDLPolicy)"),
                 (166, "4 · Hierarchical learning & evaluation")):
        note(fig, x, 102.5, f"<b>{t}</b>", size=17)

    # ---- trainee
    rbox(fig, 1, 6, 7, 96, INK, fill=0.05)
    fig.add_annotation(x=4, y=51, text="<b>Trainee</b><br>VR + haptics or 2D", textangle=-90, showarrow=False,
                       font=dict(size=14, color=INK))

    # ---- 1 sensing, with data
    rbox(fig, 10, 73, 34, 96, VID, "Endoscope video", fill=0.08)
    img(fig, "frame", 11, 77.3, 33, 92.6)
    note(fig, 22, 75, "dVRK stereo camera → Quest 3 / monitor", size=11)
    rbox(fig, 10, 48, 34, 70, KIN, "Haptic teleoperation", fill=0.08)
    img(fig, "kinematics", 11, 52.2, 33, 66.8)
    note(fig, 22, 50, "Falcon ↔ ROS: pose x,y,z (top) · |f| (bottom)", size=11)
    rbox(fig, 10, 22, 34, 45, EEG, "Wearable EEG", fill=0.08)
    img(fig, "eeg", 11, 25.8, 33, 41.8)
    note(fig, 22, 24, "Enobio 8-ch dry, 500 Hz (3 s, 1–40 Hz)", size=11)
    rbox(fig, 10, 12, 34, 19, EEG, None, "UWIST & ARSQ questionnaires", fill=0.06, bsize=13)
    arrow(fig, 10, 84.5, 7, 84.5, VID); note(fig, 8.5, 86.3, "view", size=11, color=VID)
    arrow(fig, 7, 61, 10, 61, KIN); note(fig, 8.5, 62.8, "hand", size=11, color=KIN)
    arrow(fig, 10, 56, 7, 56, KIN); note(fig, 8.5, 54.2, "force", size=11, color=KIN)
    arrow(fig, 7, 33.5, 10, 33.5, EEG)

    # ---- 2 processing, with data
    rbox(fig, 38, 75, 72, 94, VID, "Frame stream", fill=0.08)
    img(fig, "frame64", 39, 76.5, 51, 91.3)
    note(fig, 61.5, 83.5, "timestamped left frames<br>resized to 64×64<br>change detection")
    rbox(fig, 38, 50, 72, 70, KIN, "Kinematic stream", fill=0.08)
    img(fig, "optimality", 39, 52.5, 57, 63.5)
    note(fig, 48, 55, "jerk optimality O<sub>t</sub>", size=11)
    note(fig, 64.5, 59, "merge on 0.1 s<br>O<sub>t</sub> = 1 − ĉ<sub>t</sub><br>Δa<sub>t</sub> history<br>a<sub>t</sub> ∈ ℝ<sup>9</sup>")
    rbox(fig, 38, 26, 72, 46, EEG, "LGAF engagement decoder", fill=0.08)
    img(fig, "lgaf", 39, 27.5, 51, 42)
    note(fig, 61.5, 35, "α/β envelopes → GADF<sub>ij</sub><br>= sin(φ<sup>α</sup><sub>i</sub> − φ<sup>β</sup><sub>j</sub>)<br>"
                       "depthwise-sep. CNN<br>→ L<sub>oddball</sub>, L<sub>rest</sub><br>(Nencki–Symfonia, 71%)", size=11.5)
    rbox(fig, 38, 14, 60, 21, EEG, None, "virtualised V<sub>β</sub> / V<sub>α</sub>", fill=0.06, bsize=13)
    rbox(fig, 38, 0.5, 72, 11, DER, None, None, fill=0.18)
    note(fig, 45.5, 5.8, "<b>CFS(t)</b><br>V<sub>β</sub>/V<sub>α</sub> · L<sub>odd</sub>/L<sub>rest</sub>", size=12)
    img(fig, "cfs", 53, 1.5, 71, 10)
    for y, c in ((84.5, VID), (59, KIN), (33.5, EEG), (15.5, EEG)):
        arrow(fig, 34, y, 38, y if y > 20 else 17.5, c)
    arrow(fig, 67, 26, 67, 11, EEG); note(fig, 69.3, 18.5, "L<sub>odd</sub>/L<sub>rest</sub>", size=10, color=EEG,
                                           textangle=-90)
    arrow(fig, 49, 14, 49, 11, EEG)

    # ---- sync bar
    rbox(fig, 76, 1, 81, 95, DER, fill=0.25)
    fig.add_annotation(x=78.5, y=48, text="<b>time-aligned demonstration stream @ 10 Hz</b>", textangle=-90,
                       showarrow=False, font=dict(size=13, color=INK))
    arrow(fig, 72, 84.5, 76, 84.5, VID, "I<sub>t</sub>", lshift=(0, 10))
    arrow(fig, 72, 60, 76, 60, KIN, "a<sub>t</sub>, O<sub>t</sub>", lshift=(0, 10))
    arrow(fig, 72, 5.5, 76, 5.5, DER, "CFS<sub>t</sub>", lshift=(0, 10))

    # ---- 3 RIDL policy
    rbox(fig, 85, 8, 129, 96, POL, fill=0.04, dash="dash")
    rbox(fig, 88, 71, 127, 93, VID, "Visual encoder (frozen SD-UNet)", fill=0.08)
    note(fig, 100.5, 80.5, "SD v1.5 UNet, t = 0, null text<br>3×64×64 → 4×64×64<br>→ 16 384 → Linear → 512", size=12)
    img(fig, "unet", 113.5, 72.5, 126.5, 89.5)
    rbox(fig, 88, 43, 127, 65, KIN, "Kinematic–engagement encoder",
         "11 ch × 8 steps: CFS, O, Δa (9)<br>Conv1D 1×1 → +sinusoidal PE<br>Transformer 4 L · 8 H · ff 256<br>"
         "mean-pool → 512", fill=0.08, bsize=13)
    arrow(fig, 81, 82, 88, 82, VID, "3×64×64", lshift=(0, 10))
    arrow(fig, 81, 54, 88, 54, KIN, "11×8", lshift=(0, 10))
    fig.add_shape(type="circle", x0=104.5, y0=33, x1=110.5, y1=39, line=dict(color=POL, width=2), fillcolor="white")
    note(fig, 107.5, 36, "<b>+</b>", size=22, color=POL)
    line(fig, [(127, 76), (128.3, 76), (128.3, 36)], VID, dash="solid")
    arrow(fig, 128.3, 36, 110.5, 36, VID, "512", lshift=(0, 10))
    arrow(fig, 107.5, 43, 107.5, 39, KIN, "512", lshift=(16, 0))
    rbox(fig, 93, 22, 122, 30, POL, None, "MLP 512 → 256 (GELU, dropout)", fill=0.15, bsize=13)
    arrow(fig, 107.5, 33, 107.5, 30, POL)
    rbox(fig, 87, 10, 105, 19, POL, None, "<b>Policy head</b><br>𝒩(μ ∈ ℝ<sup>9</sup>, σ)", fill=0.15, bsize=13)
    rbox(fig, 109, 10, 127, 19, POL, None, "<b>Value head</b><br>V(s<sub>t</sub>)", fill=0.15, bsize=13)
    arrow(fig, 101, 22, 96, 19, POL); arrow(fig, 114, 22, 118, 19, POL)

    # ---- 4 learning & evaluation
    rbox(fig, 135, 76, 198, 95, LRN, "Level 1 · Skill prior (behaviour cloning)",
         "â<sub>t</sub> = Δpose over the next 1 s (action chunk)<br>min ‖μ(s<sub>t</sub>) − a*<sub>t</sub>‖² on the demonstrations",
         bsize=13)
    rbox(fig, 135, 50, 198, 71, LRN, "Level 2 · Refinement (PPO)",
         "R<sub>t</sub> = A · <b>w(CFS<sub>t</sub>)</b> · exp(−‖â<sub>t</sub> − a*<sub>t</sub>‖² / 2σ²d) + B · smoothness<br>"
         "clip ε = 0.2 · GAE γ = 0.99, λ = 0.95", bsize=13)
    rbox(fig, 135, 33, 198, 45, LRN, "Subject curriculum",
         "participant 0 → 1 → … → 7, weights carried forward<br>(levels 1 + 2 repeated per participant)", fill=0.07, bsize=13)
    rbox(fig, 135, 10, 198, 28, DER, "Held-out evaluation (participants 8–9)",
         "residual r<sub>t</sub> = ‖x̂<sub>t+10</sub> − x<sub>t+10</sub>‖ → Kalman r̃<sub>t</sub><br>"
         "E<sub>traj</sub>, σ<sub>r</sub>; VR ↔ 2D cross-domain", fill=0.15, bsize=13)
    line(fig, [(96, 10), (96, 5.5), (132.5, 5.5), (132.5, 85.5)], POL, dash="solid")
    arrow(fig, 132.5, 85.5, 135, 85.5, POL); arrow(fig, 132.5, 63, 135, 63, POL); arrow(fig, 132.5, 19, 135, 19, POL)
    fig.add_annotation(x=132.5, y=40, text="â<sub>t</sub> ~ 𝒩(μ, σ)", textangle=-90, showarrow=False,
                       font=dict(size=12, color=POL), bgcolor="white")
    line(fig, [(127, 14.5), (130.6, 14.5), (130.6, 56)], POL, dash="dot")
    arrow(fig, 130.6, 56, 135, 56, POL)
    fig.add_annotation(x=130.6, y=28, text="V(s<sub>t</sub>) critic", textangle=-90, showarrow=False,
                       font=dict(size=11, color=POL), bgcolor="white")
    arrow(fig, 166.5, 76, 166.5, 71, LRN, "init", lshift=(18, 0))
    arrow(fig, 166.5, 50, 166.5, 45, LRN)
    line(fig, [(81, 97.8), (166, 97.8)], KIN, dash="dot")
    arrow(fig, 166, 97.8, 166, 95, KIN, "a*<sub>t</sub> (operator demonstration)", lpos=0.0, lshift=(-80, 8))
    line(fig, [(81, 2.5), (199.4, 2.5), (199.4, 60.5)], DER, dash="dot")
    arrow(fig, 199.4, 60.5, 198, 60.5, DER)
    fig.add_annotation(x=199.6, y=38, text="w(CFS<sub>t</sub>) = ½ clip(F<sub>t</sub>/F̄, 0, 2)", textangle=-90,
                       showarrow=False, font=dict(size=11, color=DER))
    line(fig, [(150, 10), (150, 0.2), (4, 0.2), (4, 6)], LRN, dash="dash", width=1.5)
    note(fig, 112, 0.2, "neuroadaptive loop (future): CFS-driven haptic assistance / difficulty → trainee", size=11,
         color=LRN, bgcolor="white")

    fig.update_layout(width=2000, height=1250, paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Arial, Helvetica, sans-serif"),
                      showlegend=False)
    save(fig, "fig3_architecture", scale=2)


if __name__ == "__main__":
    main()
