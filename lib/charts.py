"""
Chart styling and builders (Plotly), in the same style as the analysis apps.

Colour rules: one colour per single-series chart; red/blue only for "raises / lowers the risk" (always
labelled in words too); a one-hue ramp for the confusion matrix; at most two series per chart, legend shown.
No pies, no dual axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.theme import ACCENT, LOWERS, QUIET, RAISES, SEQUENTIAL, SERIES_2

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
CONFIG = {"displayModeBar": False, "scrollZoom": False}


def style(fig: go.Figure, *, height: int = 360, title: str | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13)),
        bargap=0.3,
    )
    fig.update_xaxes(automargin=True, showgrid=False, linecolor=AXIS, ticks="", tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2))
    fig.update_yaxes(automargin=True, gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2))
    return fig


def drivers(labels: list[str], values: list[float], details: list[str], *, height: int | None = None) -> go.Figure:
    """Diverging bars of SHAP contributions: right (red) raises the churn risk, left (blue) lowers it."""
    colors = [RAISES if v > 0 else LOWERS for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        customdata=np.array([details, ["Raises the risk" if v > 0 else "Lowers the risk" for v in values]]).T,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]} (%{x:+.2f} log-odds)<extra></extra>",
    ))
    style(fig, height=height or 34 * len(labels) + 70)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2, size=13))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=INK_2, zerolinewidth=1,
                     title="← lowers · raises risk →")
    return fig


def importance(labels: list[str], values: list[float]) -> go.Figure:
    """Mean |SHAP| per feature, single series, largest first."""
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=ACCENT, line=dict(width=0), cornerradius=4),
        hovertemplate="<b>%{y}</b><br>Average impact %{x:.2f} log-odds<extra></extra>",
    ))
    style(fig, height=30 * len(labels) + 70)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title="Average impact on the churn score (mean |SHAP|, log-odds)")
    return fig


def tradeoff(curve: pd.DataFrame, threshold: float) -> go.Figure:
    """Recall and precision against the decision threshold, with the chosen threshold marked."""
    fig = go.Figure()
    for col, color, dash in (("Recall", ACCENT, "solid"), ("Precision", SERIES_2, "solid")):
        fig.add_trace(go.Scatter(
            x=curve["threshold"], y=curve[col], name=col, mode="lines", line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"{col} %{{y:.1%}}<extra></extra>",
        ))
    fig.add_vline(x=threshold, line=dict(color=INK, width=1.5, dash="dash"))
    fig.add_annotation(x=threshold, y=1.04, yref="paper", text=f"threshold {threshold:.2f}", showarrow=False,
                       font=dict(color=INK, size=12), xanchor="left", xshift=4)
    style(fig, height=360, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Decision threshold (flag customers at or above this churn probability)", range=[0, 1],
                     showspikes=True, spikecolor=AXIS, spikethickness=1, spikedash="solid")
    fig.update_yaxes(title="Test set", tickformat=".0%", range=[0, 1.02])
    return fig


def confusion(tn: int, fp: int, fn: int, tp: int) -> go.Figure:
    """2x2 confusion matrix, one-hue ramp, counts and plain-language labels in each cell."""
    z = [[tn, fp], [fn, tp]]
    total = tn + fp + fn + tp
    names = [["Stayed, not flagged", "Stayed, flagged"], ["Churned, missed", "Churned, caught"]]
    shade = np.log1p(z) / np.log1p(max(tn, fp, fn, tp))
    fig = go.Figure(go.Heatmap(
        z=shade, x=["Not flagged", "Flagged"], y=["Stayed", "Churned"], zmin=0, zmax=1,
        colorscale=SEQUENTIAL, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
    ))
    for r, row in enumerate(["Stayed", "Churned"]):
        for c, col in enumerate(["Not flagged", "Flagged"]):
            fig.add_annotation(x=col, y=row, showarrow=False, font=dict(size=13, color="#ffffff" if shade[r][c] > 0.6 else INK),
                               text=f"<b>{z[r][c]:,}</b><br>{names[r][c]}<br>{z[r][c] / total:.1%}")
    style(fig, height=330)
    fig.update_yaxes(autorange="reversed", showgrid=False, title="What happened", tickfont=dict(color=INK_2))
    fig.update_xaxes(side="top", title="What the model said", tickfont=dict(color=INK_2))
    return fig


def risk_histogram(proba: np.ndarray, threshold: float, churned: np.ndarray | None = None) -> go.Figure:
    """Distribution of churn probabilities; stacked by outcome when it is known."""
    bins = dict(start=0, end=1.0001, size=0.05)
    fig = go.Figure()
    if churned is None:
        fig.add_trace(go.Histogram(x=proba, xbins=bins, marker=dict(color=ACCENT, line=dict(color=SURFACE, width=2)),
                                   hovertemplate="%{x}: %{y:,} customers<extra></extra>", name="Customers"))
    else:
        for mask, name, color in ((churned == 0, "Stayed", LOWERS), (churned == 1, "Churned", RAISES)):
            fig.add_trace(go.Histogram(x=proba[mask], xbins=bins, name=name,
                                       marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                                       hovertemplate=f"{name}: %{{y:,}} customers<extra></extra>"))
        fig.update_layout(barmode="group", bargroupgap=0.05)  # side by side: stacking misleads on a log axis
    fig.add_vline(x=threshold, line=dict(color=INK, width=1.5, dash="dash"))
    fig.add_annotation(x=threshold, y=1.04, yref="paper", text=f"flag at {threshold:.0%}", showarrow=False,
                       font=dict(color=INK, size=12), xanchor="left", xshift=4)
    style(fig, height=320, legend=churned is not None)
    fig.update_layout(bargap=0.08 if churned is not None else 0)
    fig.update_xaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Customers (log scale)", type="log", tickvals=[1, 10, 100, 1000], ticktext=["1", "10", "100", "1,000"])
    return fig


def calibration(predicted: list[float], observed: list[float], counts: list[int]) -> go.Figure:
    """Predicted against observed churn rate per probability bin, with the perfect-calibration diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                             line=dict(color=AXIS, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=predicted, y=observed, mode="lines+markers", name="This model",
        line=dict(color=ACCENT, width=2), marker=dict(size=9, color=ACCENT, line=dict(color=SURFACE, width=2)),
        customdata=counts,
        hovertemplate="Predicted %{x:.0%}<br>Actually churned %{y:.0%}<br>%{customdata:,} customers<extra></extra>",
    ))
    style(fig, height=340, legend=True)
    fig.update_xaxes(title="Average predicted churn probability", tickformat=".0%", range=[0, 1], showgrid=True,
                     gridcolor=GRID)
    fig.update_yaxes(title="Share that actually churned", tickformat=".0%", range=[0, 1])
    return fig


def gauge(p: float, threshold: float, color: str, label: str) -> go.Figure:
    """Half-circle gauge of the churn probability: Low / Watch / Flag zones, threshold marked, number in the middle."""
    fig = go.Figure(go.Indicator(
        mode="gauge", value=p * 100,
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, 100], tickvals=[0, threshold * 100, 50, 100],
                      ticktext=["0%", f"{threshold:.0%}", "50%", "100%"], tickfont=dict(size=12, color=INK_2),
                      tickwidth=1, tickcolor=AXIS),
            bar=dict(color=color, thickness=0.32),
            bgcolor=SURFACE, borderwidth=0,
            steps=[dict(range=[0, threshold * 50], color="#e9f5ee"),
                   dict(range=[threshold * 50, threshold * 100], color="#fdf6e7"),
                   dict(range=[threshold * 100, 100], color="#fbeceb")],
            threshold=dict(line=dict(color=INK, width=3), thickness=0.9, value=threshold * 100),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.add_annotation(x=0.5, y=0.1, xref="paper", yref="paper", text=f"<b>{label}</b>", showarrow=False,
                       font=dict(size=40, color=INK, family=FONT))
    fig.add_annotation(x=0.5, y=-0.08, xref="paper", yref="paper", text="chance of churn", showarrow=False,
                       font=dict(size=13, color=INK_2, family=FONT))
    fig.update_layout(height=230, margin=dict(l=40, r=44, t=18, b=24), paper_bgcolor=SURFACE, font=dict(family=FONT))
    return fig
