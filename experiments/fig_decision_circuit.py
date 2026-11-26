"""Decision-making brain circuit (thesis Fig. 3.2 / Sec. 3.6) in Plotly -> figures/fig_decision_circuit.{png,pdf}.
A: circuit model (compact blocks + electrode loci), B: scalp loci vs recorded montage, C: RIDL analogue."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pstyle import save

RED, BLUE, VIOL, GREEN, YEL, ORG, GRAYD, INK = "#E45756", "#4C78A8", "#9D6BD8", "#54A24B", "#ECC94B", "#F58518", "#8c8c8c", "#1f2a37"
RECORDED = {"O1", "Oz", "O2", "CP1", "CP2", "C3", "C4", "Fz"}
POS = {"Fp1": (-.31, .95), "Fp2": (.31, .95), "F7": (-.81, .59), "F3": (-.4, .55), "Fz": (0, .5), "F4": (.4, .55),
       "F8": (.81, .59), "FC1": (-.22, .27), "FCz": (0, .25), "FC2": (.22, .27), "T7": (-1, 0), "C3": (-.5, 0),
       "Cz": (0, 0), "C4": (.5, 0), "T8": (1, 0), "CP1": (-.22, -.27), "CP2": (.22, -.27), "P7": (-.81, -.59),
       "P3": (-.4, -.55), "Pz": (0, -.5), "P4": (.4, -.55), "P8": (.81, -.59), "O1": (-.31, -.95), "Oz": (0, -1),
       "O2": (.31, -.95)}
NODES = {  # name: (x, y, colour, loci, group)
    "Decision": (14, 54, RED, "Fz, FCz"), "Maintenance": (38, 54, BLUE, "Fz"), "Reward": (64, 54, YEL, "FCz, FP1/2"),
    "BG, STN": (88, 54, GRAYD, "deep"), "Motor": (14, 28, GREEN, "C3/4"), "Control": (38, 28, VIOL, "FC1/2"),
    "VTA": (64, 28, ORG, "deep · dopamine"), "LC": (64, 7, ORG, "deep · noradrenaline"),
    "dlPFC": (38, 80, "#72B7B2", "F3/4"), "vmPFC": (64, 80, "#72B7B2", "FP1/2"), "Parietal": (88, 80, "#72B7B2", "P3/4, Pz"),
}


def _rgba(h, a):
    h = h.lstrip("#"); return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


def main():
    fig = make_subplots(rows=1, cols=3, column_widths=[0.5, 0.24, 0.26], horizontal_spacing=0.03,
                        subplot_titles=["<b>A</b>  Decision-making circuit (compact blocks, electrode loci)",
                                        "<b>B</b>  Loci on the scalp", "<b>C</b>  Computational analogue in RIDL"])
    # ---------------- A: circuit
    for (x0, y0, x1, y1, lab) in ((26, 68, 100, 92, "Moderators · attention"), (2, 17, 50, 66, "Core & motor"),
                                  (52, -2, 76, 37, "Modulatory")):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#9aa0a6", dash="dash", width=1.5),
                      fillcolor="rgba(0,0,0,0.02)", layer="below", xref="x", yref="y")
        fig.add_annotation(x=x0 + 1, y=y1 - 1, text=f"<i>{lab}</i>", showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=12, color="#5f6368"), xref="x", yref="y")
    edges = [("Decision", "Maintenance", "", "solid"), ("Maintenance", "Control", "", "solid"),
             ("Decision", "Motor", "", "solid"), ("Control", "Motor", "", "solid"),
             ("Maintenance", "Reward", "value / cost", "solid"), ("Control", "VTA", "exploit / explore", "solid"),
             ("Reward", "BG, STN", "", "solid"), ("BG, STN", "VTA", "", "solid"), ("VTA", "Reward", "", "dot"),
             ("LC", "Control", "", "dot"), ("LC", "VTA", "", "dot"),
             ("dlPFC", "Maintenance", "", "dash"), ("vmPFC", "Reward", "", "dash"), ("Parietal", "BG, STN", "", "dash")]

    def clip(xa, ya, xb, yb, hw=10.5, hh=6.0):
        """point where the segment from box-centre a towards b leaves box a"""
        dx, dy = xb - xa, yb - ya
        t = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
        return xa + dx * t, ya + dy * t
    for a, b, lab, dash in edges:
        (xa, ya, *_), (xb, yb, *_) = NODES[a], NODES[b]
        col = ORG if a in ("LC", "VTA") and dash == "dot" else "#6b7280"
        sx, sy = clip(xa, ya, xb, yb); ex, ey = clip(xb, yb, xa, ya)
        fig.add_annotation(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2 if dash != "solid" else 0, arrowwidth=2, arrowcolor=col, text="")
        if lab:
            fig.add_annotation(x=(sx + ex) / 2, y=(sy + ey) / 2, text=f"<i>{lab}</i>", showarrow=False, xref="x",
                               yref="y", yshift=12, font=dict(size=12, color=INK), bgcolor="white")
    for name, (x, y, col, loci) in NODES.items():
        rec = any(e in RECORDED for e in loci.replace("/", ",").replace("3", "3").split(", ")) or loci in ("Fz",)
        rec = name in ("Decision", "Maintenance", "Motor", "Parietal")
        fig.add_shape(type="rect", x0=x - 10.5, y0=y - 6, x1=x + 10.5, y1=y + 6, xref="x", yref="y",
                      fillcolor=_rgba(col, 0.28), line=dict(color=col, width=3.5 if rec else 1.5))
        fig.add_annotation(x=x, y=y, text=f"<b>{name}</b><br>({loci})", showarrow=False, xref="x", yref="y",
                           font=dict(size=13, color=INK))
    fig.add_annotation(x=97, y=6, xref="x", yref="y", xanchor="right", showarrow=False, align="right",
                       text="thick border = node sampled in this study<br>(Parietal via CP1/CP2)",
                       font=dict(size=11, color="#5f6368"))
    fig.update_xaxes(range=[0, 100], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-3, 95], visible=False, row=1, col=1)

    # ---------------- B: scalp map
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=1.12 * np.cos(t), y=1.12 * np.sin(t), mode="lines", line=dict(color="#6b7280", width=2),
                             showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-0.12, 0, 0.12], y=[1.1, 1.25, 1.1], mode="lines", line=dict(color="#6b7280", width=2),
                             showlegend=False), row=1, col=2)
    node_col = {"Fz": RED, "FCz": RED, "FC1": VIOL, "FC2": VIOL, "C3": GREEN, "C4": GREEN, "Fp1": YEL, "Fp2": YEL,
                "F3": "#72B7B2", "F4": "#72B7B2", "P3": "#72B7B2", "P4": "#72B7B2", "Pz": "#72B7B2",
                "CP1": "#72B7B2", "CP2": "#72B7B2", "O1": "#bdbdbd", "Oz": "#bdbdbd", "O2": "#bdbdbd"}
    for e, (x, y) in POS.items():
        col = node_col.get(e, "white")
        rec = e in RECORDED
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{e}</b>" if rec else e],
                                 textposition="middle center", textfont=dict(size=10 if rec else 9, color=INK),
                                 marker=dict(size=34 if rec else 26, color=col,
                                             line=dict(color=INK if rec else "#bdbdbd", width=3 if rec else 1)),
                                 showlegend=False), row=1, col=2)
    fig.add_annotation(x=0, y=-1.32, xref="x2", yref="y2", showarrow=False, font=dict(size=11, color="#5f6368"),
                       text="bold ring = recorded (Enobio 8-ch); colour = circuit node")
    fig.update_xaxes(range=[-1.3, 1.3], visible=False, row=1, col=2)
    fig.update_yaxes(range=[-1.45, 1.35], visible=False, scaleanchor="x2", row=1, col=2)

    # ---------------- C: analogue
    rows = [(RED, "Decision / control (PFC)", "policy head 𝒩(μ, σ)"),
            (BLUE, "Goal maintenance", "Transformer state (8 steps)"),
            (YEL, "Reward, BG (value / cost)", "value head V(s): critic"),
            (ORG, "VTA dopamine (RPE)", "PPO advantage A<sub>t</sub>"),
            (ORG, "LC (exploit / explore)", "policy σ, entropy bonus"),
            ("#72B7B2", "Attention moderators", "CFS(t): input + reward weight"),
            (GREEN, "Motor output (C3/4)", "Δpose action chunk")]
    for i, (col, brain, ridl) in enumerate(rows):
        y = 92 - i * 13.2
        fig.add_shape(type="rect", x0=0, y0=y - 5.5, x1=46, y1=y + 5.5, xref="x3", yref="y3",
                      fillcolor=_rgba(col, 0.28), line=dict(color=col, width=1.5))
        fig.add_annotation(x=23, y=y, text=brain, showarrow=False, xref="x3", yref="y3", font=dict(size=13, color=INK))
        fig.add_annotation(x=55, y=y, ax=47, ay=y, xref="x3", yref="y3", axref="x3", ayref="y3", text="",
                           showarrow=True, arrowhead=2, arrowcolor="#6b7280", arrowwidth=1.5)
        fig.add_shape(type="rect", x0=56, y0=y - 5.5, x1=100, y1=y + 5.5, xref="x3", yref="y3",
                      fillcolor="rgba(25,167,206,0.10)", line=dict(color="#19A7CE", width=1.2))
        fig.add_annotation(x=78, y=y, text=ridl, showarrow=False, xref="x3", yref="y3", font=dict(size=13, color=INK))
    fig.add_annotation(x=50, y=-3, xref="x3", yref="y3", showarrow=False, font=dict(size=10.5, color="#5f6368"),
                       text="brain (thesis §3.6)  →  RIDL component (analogy, not a claim of mechanism)")
    fig.update_xaxes(range=[0, 100], visible=False, row=1, col=3)
    fig.update_yaxes(range=[-6, 100], visible=False, row=1, col=3)
    fig.update_layout(width=2100, height=760, paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Arial, Helvetica, sans-serif"))
    for a in fig.layout.annotations[:3]:
        a.update(font=dict(size=16, color=INK))
    save(fig, "fig_decision_circuit", scale=2)


if __name__ == "__main__":
    main()
