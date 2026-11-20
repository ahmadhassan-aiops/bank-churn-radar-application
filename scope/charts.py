"""Plotly figures in the console style: white scope backgrounds, teal rings, recessive grids, mono numbers.

The radar scope places every customer by two facts: distance from the centre is the churn risk (the centre is
certain churn; the call zone is the inner disc inside the threshold ring) and bearing, clockwise from north, is the
number of card transactions in the last 12 months (the strongest single signal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scope import theme as t

CONFIG = {"displayModeBar": False, "scrollZoom": False}
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"  # system fonts: measured correctly at first draw
MONO = "Consolas, Menlo, monospace"
CT_MIN, CT_MAX = 10, 140  # transaction counts in the data run from 10 to 139


def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=14, t=14, b=8), paper_bgcolor=t.PANEL, plot_bgcolor=t.PANEL,
        font=dict(family=FONT, size=13, color=t.MUTED), showlegend=legend,
        legend=dict(orientation="v", yanchor="top", y=0.99, x=0.01, xanchor="left", bgcolor="rgba(255,255,255,.85)", title_text="",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=t.SCOPE, font=dict(family=FONT, color=t.INK, size=13)),
    )
    fig.update_xaxes(automargin=True, showgrid=False, linecolor=t.LINE, ticks="outside", tickcolor=t.LINE, zeroline=False,
                     tickfont=dict(family=MONO, size=11), title_font=dict(color=t.MUTED, size=12))
    fig.update_yaxes(automargin=True, gridcolor=t.GRID, linecolor=t.LINE, zeroline=False, tickfont=dict(family=MONO, size=11),
                     title_font=dict(color=t.MUTED, size=12))
    return fig


def radius(p) -> np.ndarray:
    """Distance from the centre: 1 - sqrt(p), so risk 0 sits on the rim and the threshold ring falls mid-scope."""
    return 1 - np.sqrt(np.clip(np.asarray(p, dtype=float), 0, 1))


def bearing(ct) -> np.ndarray:
    return 15 + np.clip((np.asarray(ct, dtype=float) - CT_MIN) / (CT_MAX - CT_MIN), 0, 1) * 330


def radar(p: np.ndarray, ct: np.ndarray, threshold: float, outcome: np.ndarray | None = None,
          me: tuple[float, float] | None = None, height: int = 470, ids: np.ndarray | None = None) -> go.Figure:
    """The scope. `outcome` colours dots by what really happened (test set); otherwise by risk band.
    `me` = (probability, transactions) of the customer being tracked."""
    rt, rw = float(radius(threshold)), float(radius(threshold / 2))
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=[rt], base=[0], theta=[180], width=[360], marker=dict(color=t.BAND["Flag"]["soft"], line_width=0),
                              hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Barpolar(r=[rw - rt], base=[rt], theta=[180], width=[360], marker=dict(color=t.BAND["Watch"]["soft"], line_width=0),
                              hoverinfo="skip", showlegend=False))
    for r, color, dash in ((rt, t.BAND["Flag"]["color"], "solid"), (rw, t.BAND["Watch"]["color"], "dot")):
        th = np.linspace(0, 360, 181)
        fig.add_trace(go.Scatterpolar(r=np.full_like(th, r), theta=th, mode="lines", line=dict(color=color, width=1.6, dash=dash),
                                      hoverinfo="skip", showlegend=False))
    rr, bb = radius(p), bearing(ct)
    ident = ids if ids is not None else np.arange(len(p))
    custom = np.c_[ident, p, ct]
    tip = "Customer %{customdata[0]}<br>Churn risk %{customdata[1]:.0%}<br>%{customdata[2]:.0f} transactions<extra></extra>"
    if outcome is not None:
        for mask, name, color, size in ((outcome == 0, "Stayed", t.STAYED, 5), (outcome == 1, "Churned", t.CHURNED, 6)):
            fig.add_trace(go.Scatterpolar(r=rr[mask], theta=bb[mask], mode="markers", name=name,
                                          marker=dict(size=size, color=color, opacity=.55, line_width=0),
                                          customdata=custom[mask], hovertemplate=f"{name}<br>" + tip))
    else:
        band = np.where(p >= threshold, "Flag", np.where(p >= threshold / 2, "Watch", "Low"))
        for b in ("Low", "Watch", "Flag"):
            mask = band == b
            fig.add_trace(go.Scatterpolar(r=rr[mask], theta=bb[mask], mode="markers", name=t.BAND[b]["word"],
                                          marker=dict(size=6, color=t.BAND[b]["color"], opacity=.6, line_width=0),
                                          customdata=custom[mask], hovertemplate=tip))
    if me is not None:
        mp, mct = me
        b = "Flag" if mp >= threshold else "Watch" if mp >= threshold / 2 else "Low"
        fig.add_trace(go.Scatterpolar(r=radius([mp]), theta=bearing([mct]), mode="markers+text", name="This customer",
                                      marker=dict(size=20, color=t.BAND[b]["color"], line=dict(color="#ffffff", width=3), symbol="circle"),
                                      text=["  THIS CUSTOMER"], textposition="middle right",
                                      textfont=dict(family=MONO, size=12, color=t.INK),
                                      hovertemplate=f"This customer<br>Churn risk {mp:.0%}<br>{mct:.0f} transactions<extra></extra>"))
    ticks = [0.5, threshold, threshold / 2, 0.01]
    counts = [20, 40, 60, 80, 100, 120, 140]
    style(fig, height, legend=True)
    fig.update_layout(
        margin=dict(l=36, r=36, t=24, b=24),
        legend=dict(orientation="v", y=1.02, x=-0.02, xanchor="left", yanchor="top", font=dict(size=12)),
        polar=dict(
            bgcolor="#fbfdfc", hole=0,
            radialaxis=dict(range=[0, 1], tickvals=radius(ticks).tolist(), ticktext=[f"{v * 100:g}%" for v in ticks],
                            angle=82, tickangle=82, gridcolor=t.RING, linecolor=t.RING, tickfont=dict(family=MONO, size=10, color=t.MUTED),
                            showline=False),
            angularaxis=dict(rotation=90, direction="clockwise", tickvals=bearing(counts).tolist(), ticktext=[str(c) for c in counts],
                             gridcolor=t.GRID, linecolor=t.RING, tickfont=dict(family=MONO, size=10, color=t.MUTED)),
        ),
    )
    return fig


def importance(labels: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=t.SCOPE, cornerradius=2),
                           text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False,
                           textfont=dict(family=MONO, size=11, color=t.INK),
                           hovertemplate="<b>%{y}</b><br>Average impact %{x:.2f} log-odds<extra></extra>"))
    style(fig, 28 * len(labels) + 60)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(family=FONT, size=13, color=t.INK))
    fig.update_xaxes(showgrid=True, gridcolor=t.GRID, title="Average impact on the churn score (mean |SHAP|, log-odds)",
                     range=[0, max(values) * 1.15])
    fig.update_layout(bargap=.35)
    return fig


def tradeoff(curve: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for col, color, dash in (("Recall", t.SCOPE, "solid"), ("Precision", "#b26a00", "dash")):
        fig.add_trace(go.Scatter(x=curve["threshold"], y=curve[col], name=col, mode="lines", line=dict(color=color, width=2.4, dash=dash),
                                 hovertemplate=f"{col} %{{y:.1%}}<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=t.INK, width=1.4, dash="dot"))
    fig.add_annotation(x=threshold, y=1.05, yref="paper", text=f"call line {threshold:.2f}", showarrow=False, xanchor="left", xshift=4,
                       font=dict(family=MONO, size=11, color=t.INK))
    style(fig, 340, legend=True)
    fig.update_layout(hovermode="x unified", legend=dict(y=0.03, yanchor="bottom"))
    fig.update_xaxes(title="Call line (flag customers at or above this churn probability)", range=[0, 1])
    fig.update_yaxes(title="Held-out customers", tickformat=".0%", range=[0, 1.03])
    return fig


def calibration(predicted: list[float], observed: list[float], counts: list[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=t.RING, width=1.6, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="This model", line=dict(color=t.SCOPE, width=2.2),
                             marker=dict(size=10, color=t.SCOPE, line=dict(color="#ffffff", width=2)), customdata=counts,
                             hovertemplate="Predicted %{x:.0%}<br>Actually churned %{y:.0%}<br>%{customdata:,} customers<extra></extra>"))
    style(fig, 340, legend=True)
    fig.update_xaxes(title="Average predicted churn probability", tickformat=".0%", range=[0, 1], showgrid=True, gridcolor=t.GRID)
    fig.update_yaxes(title="Share that actually churned", tickformat=".0%", range=[0, 1])
    return fig
