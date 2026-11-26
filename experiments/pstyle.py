"""Shared Plotly look for every figure (thesis-notebook style: Plotly default
template, white paper) plus static export through Kaleido."""
import base64
import io
import os

import plotly.graph_objects as go
import plotly.io as pio

from common import FIGS

pio.templates.default = "plotly"
VR, NVR, GREEN, GRAY = "#636EFA", "#EF553B", "#00CC96", "#9e9e9e"   # Plotly default cycle
FONT = dict(family="Arial, Helvetica, sans-serif", size=14, color="#2a3f5f")


def style(fig, title=None, width=1200, height=700):
    fig.update_layout(template="plotly", font=FONT, paper_bgcolor="white", width=width, height=height,
                      margin=dict(l=60, r=30, t=90 if title else 40, b=60))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    return fig


def save(fig, name, scale=3):
    os.makedirs(FIGS, exist_ok=True)
    fig.write_image(os.path.join(FIGS, f"{name}.png"), scale=scale)
    fig.write_image(os.path.join(FIGS, f"{name}.pdf"))


def img_uri(path):
    """PNG/JPEG file -> data URI for fig.add_layout_image."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), im.size


def add_image(fig, path, x0, y0, w, h, label=None):
    """Place an image inside paper-coordinate box (x0, y0 = bottom-left; w, h fractions)."""
    uri, _ = img_uri(path)
    fig.add_layout_image(dict(source=uri, xref="paper", yref="paper", x=x0, y=y0 + h, sizex=w, sizey=h,
                              xanchor="left", yanchor="top", sizing="contain", layer="below"))
    if label:
        fig.add_annotation(x=x0, y=y0 + h, xref="paper", yref="paper", text=f"<b>{label}</b>", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=22, color="black"))
