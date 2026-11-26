"""Multimodal integration & RIDL architecture diagram (Plotly) -> figures/fig3_architecture.{png,pdf}."""
import plotly.graph_objects as go

from pstyle import save

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


def main():
    fig = go.Figure()
    fig.update_xaxes(range=[0, 200], visible=False); fig.update_yaxes(range=[0, 104], visible=False)

    # column headers
    for x, t in ((18, "1 · Sensing"), (55, "2 · Per-stream processing"), (106, "3 · RIDL policy (RIDLPolicy)"),
                 (170, "4 · Hierarchical learning & evaluation")):
        fig.add_annotation(x=x, y=103, text=f"<b>{t}</b>", showarrow=False, font=dict(size=17, color=INK))

    # ---- 1 sensing
    rbox(fig, 1, 8, 7, 96, INK, fill=0.05)
    fig.add_annotation(x=4, y=52, text="<b>Trainee</b><br>VR + haptics<br>or 2D", textangle=-90, showarrow=False,
                       font=dict(size=14, color=INK))
    rbox(fig, 10, 76, 33, 95, VID, "Endoscope video",
         "AMBF dVRK stereo camera<br>→ Meta Quest 3 (VR)<br>or 2D monitor")
    rbox(fig, 10, 50, 33, 70, KIN, "Haptic teleoperation",
         "Novint Falcon ↔ ROS<br>multi_psm_control<br>pose x,y,z,roll,pitch,yaw<br>forces f<sub>x</sub>,f<sub>y</sub>,f<sub>z</sub>")
    rbox(fig, 10, 22, 33, 44, EEG, "Wearable EEG",
         "Enobio 8-ch dry, 500 Hz<br>O1 Oz O2 · CP1 CP2<br>C3 C4 · Fz")
    rbox(fig, 10, 10, 33, 19, EEG, None, "UWIST & ARSQ questionnaires", fill=0.06, bsize=12)
    arrow(fig, 10, 85.5, 7, 85.5, VID)                                          # trainee views the scene
    fig.add_annotation(x=8.5, y=87.3, text="view", showarrow=False, font=dict(size=11, color=VID))
    arrow(fig, 7, 62, 10, 62, KIN); arrow(fig, 7, 33, 10, 33, EEG)
    arrow(fig, 10, 56, 7, 56, KIN, width=2)                                     # force feedback to the hand
    fig.add_annotation(x=8.5, y=63.8, text="hand", showarrow=False, font=dict(size=11, color=KIN))
    fig.add_annotation(x=8.5, y=54.2, text="force", showarrow=False, font=dict(size=11, color=KIN))

    # ---- 2 processing
    rbox(fig, 38, 78, 72, 93, VID, "Frame stream", "timestamped left frames<br>resize 64×64, change detection")
    rbox(fig, 38, 51, 72, 70, KIN, "Kinematic stream",
         "merge on 0.1 s timestamps, 420 s trim<br>jerk optimality O<sub>t</sub> = 1 − ĉ<sub>t</sub><br>"
         "displacement history Δa<sub>t</sub>")
    rbox(fig, 38, 24, 72, 46, EEG, "LGAF engagement decoder",
         "α/β envelopes → LGAF map<br>GADF<sub>ij</sub> = sin(W<sub>1</sub>φ<sup>α</sup><sub>i</sub> − W<sub>2</sub>φ<sup>β</sup><sub>j</sub> + b)<br>"
         "→ depthwise-separable CNN<br>(Nencki–Symfonia, 71% acc.) → L<sub>oddball</sub>, L<sub>rest</sub>")
    rbox(fig, 38, 10, 61, 19, EEG, None, "virtualised band power<br>V<sub>β</sub> / V<sub>α</sub>", fill=0.06)
    arrow(fig, 33, 85.5, 38, 85.5, VID); arrow(fig, 33, 60, 38, 60, KIN); arrow(fig, 33, 33, 38, 33, EEG)
    arrow(fig, 33, 14.5, 38, 14.5, EEG)
    rbox(fig, 42, 0.5, 72, 8.5, DER, None,
         "<b>CFS(t)</b> = V<sub>β</sub>/V<sub>α</sub> · L<sub>oddball</sub>(t)/L<sub>rest</sub>(t)", fill=0.2, bsize=13)
    arrow(fig, 67, 24, 67, 8.5, EEG, "L<sub>oddball</sub>/L<sub>rest</sub>", lpos=0.5, lshift=(0, 0))
    arrow(fig, 49.5, 10, 49.5, 8.5, EEG)

    # synchronisation bar
    rbox(fig, 76, 1, 81, 95, DER, fill=0.25)
    fig.add_annotation(x=78.5, y=48, text="<b>time-aligned demonstration stream @ 10 Hz</b>", textangle=-90,
                       showarrow=False, font=dict(size=13, color=INK))
    arrow(fig, 72, 85.5, 76, 85.5, VID, "I<sub>t</sub>", lshift=(0, 10))
    arrow(fig, 72, 60, 76, 60, KIN, "a<sub>t</sub>, O<sub>t</sub>", lshift=(0, 10))
    arrow(fig, 72, 4.5, 76, 4.5, DER)

    # ---- 3 RIDL policy
    rbox(fig, 85, 8, 128, 96, POL, fill=0.05, dash="dash")
    rbox(fig, 88, 72, 125, 93, VID, "Visual encoder (frozen)",
         "Stable-Diffusion v1.5 UNet, t = 0, null text<br>3×64×64 (+0 ch) → 4×64×64 → 16 384<br>"
         "Linear 16 384 → 512 (trainable)")
    rbox(fig, 88, 43, 125, 66, KIN, "Kinematic–engagement encoder",
         "11 channels × 8 steps: CFS, O, Δa (9)<br>Conv1D 1×1 → +sinusoidal PE<br>"
         "Transformer 4 layers · 8 heads · ff 256<br>mean-pool → 512")
    arrow(fig, 81, 85.5, 88, 85.5, VID, "3×64×64", lshift=(0, 10))
    arrow(fig, 81, 55, 88, 55, KIN, "11×8", lshift=(0, 10))
    fig.add_shape(type="circle", x0=103.5, y0=33, x1=109.5, y1=39, line=dict(color=POL, width=2),
                  fillcolor="white")
    fig.add_annotation(x=106.5, y=36, text="<b>+</b>", showarrow=False, font=dict(size=22, color=POL))
    line(fig, [(125, 80), (127, 80), (127, 36)], VID, dash="solid")
    arrow(fig, 127, 36, 109.5, 36, VID, "512", lpos=0.5, lshift=(0, 10))
    arrow(fig, 106.5, 43, 106.5, 39, KIN, "512", lpos=0.5, lshift=(16, 0))
    rbox(fig, 92, 22, 121, 30, POL, None, "MLP 512 → 256 (GELU, dropout)", fill=0.15)
    arrow(fig, 106.5, 33, 106.5, 30, POL)
    rbox(fig, 87, 10, 105, 19, POL, None, "<b>Policy head</b><br>𝒩(μ ∈ ℝ<sup>9</sup>, σ)", fill=0.15)
    rbox(fig, 108, 10, 126, 19, POL, None, "<b>Value head</b><br>V(s<sub>t</sub>)", fill=0.15)
    arrow(fig, 100, 22, 96, 19, POL); arrow(fig, 113, 22, 117, 19, POL)

    # ---- 4 learning & evaluation
    rbox(fig, 134, 76, 197, 95, LRN, "Level 1 · Skill prior (behaviour cloning)",
         "â<sub>t</sub> = Δpose over the next 1 s (action chunk)<br>min ‖μ(s<sub>t</sub>) − a*<sub>t</sub>‖² on the demonstrations")
    rbox(fig, 134, 50, 197, 71, LRN, "Level 2 · Refinement (PPO)",
         "R<sub>t</sub> = A · <b>w(CFS<sub>t</sub>)</b> · exp(−‖â<sub>t</sub> − a*<sub>t</sub>‖² / 2σ²d) + B · smoothness<br>"
         "clip ε = 0.2 · GAE γ = 0.99, λ = 0.95")
    rbox(fig, 134, 33, 197, 45, LRN, "Subject curriculum",
         "participant 0 → 1 → … → 7, weights carried forward<br>(levels 1 + 2 repeated per participant)", fill=0.07)
    rbox(fig, 134, 10, 197, 28, DER, "Held-out evaluation (participants 8–9)",
         "residual r<sub>t</sub> = ‖x̂<sub>t+10</sub> − x<sub>t+10</sub>‖ → Kalman r̃<sub>t</sub><br>"
         "E<sub>traj</sub>, σ<sub>r</sub>; VR ↔ 2D cross-domain", fill=0.15)
    line(fig, [(96, 10), (96, 5.5), (131, 5.5), (131, 85.5)], POL, dash="solid")    # policy output (mu, sigma)
    arrow(fig, 131, 85.5, 134, 85.5, POL); arrow(fig, 131, 63, 134, 63, POL); arrow(fig, 131, 19, 134, 19, POL)
    fig.add_annotation(x=131, y=40, text="â<sub>t</sub> ~ 𝒩(μ, σ)", textangle=-90, showarrow=False,
                       font=dict(size=12, color=POL), bgcolor="white")
    line(fig, [(126, 14.5), (129.3, 14.5), (129.3, 56)], POL, dash="dot")        # critic only feeds PPO
    arrow(fig, 129.3, 56, 134, 56, POL)
    fig.add_annotation(x=129.3, y=30, text="V(s<sub>t</sub>) critic", textangle=-90, showarrow=False,
                       font=dict(size=11, color=POL), bgcolor="white")
    arrow(fig, 165.5, 76, 165.5, 71, LRN, "init", lshift=(18, 0))
    arrow(fig, 165.5, 50, 165.5, 45, LRN)
    # demonstrations a* and CFS weight into the learner
    line(fig, [(81, 97.5), (165, 97.5)], KIN, dash="dot")
    arrow(fig, 165, 97.5, 165, 95, KIN, "a*<sub>t</sub> (operator demonstration)", lpos=0.0, lshift=(-75, 8))
    line(fig, [(81, 2.5), (199, 2.5), (199, 60.5)], DER, dash="dot")
    arrow(fig, 199, 60.5, 197, 60.5, DER)
    fig.add_annotation(x=199.5, y=40, text="w(CFS<sub>t</sub>) = ½ clip(F<sub>t</sub>/F̄, 0, 2)", textangle=-90,
                       showarrow=False, font=dict(size=11, color=DER))
    # neuroadaptive loop (future)
    line(fig, [(150, 10), (150, 0.3), (4, 0.3), (4, 8)], LRN, dash="dash", width=1.5)
    fig.add_annotation(x=110, y=0.3, text="neuroadaptive loop (future): CFS-driven haptic assistance / difficulty → trainee",
                       showarrow=False, font=dict(size=11, color=LRN), bgcolor="white", yshift=0)

    fig.update_layout(width=2000, height=1100, paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Arial, Helvetica, sans-serif"),
                      showlegend=False)
    save(fig, "fig3_architecture", scale=2)


if __name__ == "__main__":
    main()
