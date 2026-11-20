"""Churn Radar components: a light air-traffic-control console. Modules (bordered panels) carry green corner brackets
and a header with a status LED and a mono module code; readouts are instrument tiles; customers are flight-strip style
strips; signals are diverging bars around a centre line; actions are numbered instructions."""

from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

from scope import theme as t
from scope.site import CODE_URL, DATA_CREDIT, LEARN_URL, NOTEBOOK_URL

_BR = "linear-gradient(var(--scope), var(--scope))"
BRACKETS = ", ".join(f"{_BR} {pos} / {w} {h} no-repeat" for pos in ("top left", "top right", "bottom left", "bottom right")
                     for w, h in (("16px", "2px"), ("2px", "16px")))

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap');
:root {{ --bg:{t.BG}; --panel:{t.PANEL}; --line:{t.LINE}; --ink:{t.INK}; --muted:{t.MUTED}; --faint:{t.FAINT};
         --scope:{t.SCOPE}; --scope-dark:{t.SCOPE_DARK}; --scope-soft:{t.SCOPE_SOFT};
         --head:{t.HEAD}; --body:{t.BODY}; --mono:{t.MONO}; }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--body); color: var(--ink); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: var(--bg); }}
[data-testid="stAppViewContainer"] {{ background-image: radial-gradient(rgba(46,125,50,.07) 1px, transparent 1px);
    background-size: 22px 22px; }}
.block-container {{ padding-top: 1.2rem; max-width: 1260px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.msr {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }}

/* Modules: white, thin border, green corner brackets */
[data-testid="stVerticalBlockBorderWrapper"] {{ border: 1px solid var(--line) !important; border-radius: 2px !important;
    background: {BRACKETS}, var(--panel) !important; }}

/* Console bar */
.cbar {{ display: flex; align-items: center; gap: .8rem; }}
.cbar img {{ width: 2.7rem; height: 2.7rem; }}
.cbar .nm {{ font-family: var(--head); font-weight: 700; font-size: 1.75rem; letter-spacing: .08em; line-height: 1; margin: 0; }}
.cbar .nm b {{ color: var(--scope); }}
.cbar .sub {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .06em; color: var(--muted); margin: .25rem 0 0; text-transform: uppercase; }}
[data-testid="stPageLink"] a {{ background: var(--panel); border: 1px solid var(--line); border-radius: 2px; padding: .32rem .5rem; justify-content: flex-start; }}
[data-testid="stPageLink"] a p {{ font-family: var(--head); font-size: 1.02rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; white-space: nowrap; }}
[data-testid="stPageLink"] a:hover {{ border-color: var(--scope); background: var(--panel); }}
.status {{ display: flex; flex-wrap: wrap; gap: 0; margin: .7rem 0 1.4rem; border: 1px solid var(--line); background: var(--panel);
           font-family: var(--mono); font-size: .74rem; color: var(--muted); }}
.status span {{ padding: .38rem .8rem; border-right: 1px solid var(--line); white-space: nowrap; }}
.status span:last-child {{ border-right: none; }}
.status b {{ color: var(--ink); font-weight: 600; }}
.status .on::before {{ content: ""; display: inline-block; width: .5rem; height: .5rem; border-radius: 50%; background: var(--scope);
                      box-shadow: 0 0 0 3px var(--scope-soft); margin-right: .45rem; vertical-align: 1px; }}

/* Page head */
.mode {{ font-family: var(--mono); font-size: .76rem; letter-spacing: .1em; color: var(--scope-dark); margin: 0 0 .25rem; text-transform: uppercase; }}
.title {{ font-family: var(--head); font-weight: 700; font-size: 2.45rem; line-height: 1.05; margin: 0; color: var(--ink); }}
.lead {{ color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 840px; margin: .5rem 0 1.1rem; }}
.lead b {{ color: var(--ink); font-weight: 600; }}

/* Module header */
.mh {{ display: flex; align-items: center; gap: .6rem; margin: -.15rem 0 .2rem; }}
.mh .led {{ width: .55rem; height: .55rem; border-radius: 50%; background: var(--c, var(--scope)); box-shadow: 0 0 0 3px var(--s, var(--scope-soft)); flex: none; }}
.mh .code {{ font-family: var(--mono); font-size: .7rem; letter-spacing: .08em; color: var(--scope-dark); border: 1px solid var(--line);
             padding: .08rem .4rem; border-radius: 2px; flex: none; text-transform: uppercase; }}
.mh .mt {{ font-family: var(--head); font-weight: 700; font-size: 1.32rem; letter-spacing: .02em; margin: 0; line-height: 1.2; }}
.ms {{ font-size: .88rem; color: var(--muted); margin: 0 0 .7rem; padding-bottom: .55rem; border-bottom: 1px dashed var(--line); line-height: 1.5; }}

/* Readout tiles */
.ro {{ display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: .6rem; margin: 0 0 1rem; }}
@media (max-width: 820px) {{ .ro {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.ro > div {{ background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--c, var(--scope)); padding: .55rem .8rem; }}
.ro .l {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); margin: 0; }}
.ro .v {{ font-family: var(--mono); font-size: 1.55rem; font-weight: 600; margin: .15rem 0 .05rem; color: var(--ink); line-height: 1.15; }}
.ro .n {{ font-size: .8rem; color: var(--faint); margin: 0; line-height: 1.35; }}

/* Big status readout next to the scope */
.lock {{ display: grid; grid-template-columns: auto 1fr; gap: 1.1rem; align-items: center; border: 1px solid var(--line); border-left: 4px solid var(--c);
         background: var(--s); padding: .7rem 1rem; margin: .3rem 0 .2rem; }}
.lock .p {{ font-family: var(--mono); font-size: 2.6rem; font-weight: 600; color: var(--c); line-height: 1; margin: 0; }}
.lock .pl {{ font-family: var(--mono); font-size: .66rem; letter-spacing: .08em; color: var(--muted); text-transform: uppercase; margin: .3rem 0 0; }}
.lock .w {{ display: inline-flex; align-items: center; gap: .35rem; font-family: var(--head); font-weight: 700; font-size: 1.25rem;
            letter-spacing: .06em; text-transform: uppercase; color: var(--c); margin: 0; }}
.lock .w .msr {{ font-size: 1.3rem; }}
.lock .t {{ font-size: .93rem; line-height: 1.5; margin: .2rem 0 0; color: var(--ink); }}
.ladder {{ display: flex; gap: .4rem; flex-wrap: wrap; margin: .55rem 0 0; font-family: var(--mono); font-size: .72rem; }}
.ladder span {{ border: 1px solid var(--line); padding: .15rem .45rem; color: var(--muted); background: var(--panel); }}
.ladder span.on {{ border-color: var(--c); color: var(--c); font-weight: 600; }}
.facts {{ display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: .84rem; color: var(--muted); margin: .6rem 0 0; }}
.facts b {{ color: var(--ink); font-family: var(--mono); font-weight: 500; }}

/* Signals (SHAP rows): name and values left, diverging bar around a centre line right */
.sig, .sig-key {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: .8rem; align-items: center; padding: .42rem 0;
        border-bottom: 1px dashed var(--line); }}
.sig:last-of-type {{ border-bottom: none; }}
.sig .n {{ font-weight: 600; font-size: .93rem; margin: 0; }}
.sig .v {{ font-family: var(--mono); font-size: .76rem; color: var(--muted); margin: .1rem 0 0; }}
.sig .v b {{ color: var(--ink); font-weight: 600; }}
.bar {{ position: relative; height: 18px; }}
.bar::before {{ content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: var(--ink); opacity: .5; }}
.bar i {{ position: absolute; top: 3px; height: 12px; border-radius: 2px; background: var(--c); }}
.bar em {{ position: absolute; top: 0; font-style: normal; font-family: var(--mono); font-size: .7rem; color: var(--c); line-height: 18px; white-space: nowrap; }}
.sig-key {{ font-family: var(--mono); font-size: .68rem; color: var(--muted); margin: .1rem 0 .2rem; letter-spacing: .04em; text-transform: uppercase; }}
.sig-key div {{ display: flex; justify-content: space-between; }}

/* Instructions (actions) */
.ins {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: .7rem; }}
.in {{ border: 1px solid var(--line); background: var(--panel); padding: .7rem .85rem; position: relative; }}
.in .k {{ font-family: var(--mono); font-size: .68rem; letter-spacing: .08em; color: var(--scope-dark); margin: 0 0 .3rem; text-transform: uppercase;
          display: flex; align-items: center; gap: .4rem; }}
.in .k .msr {{ font-size: 1.1rem; color: var(--scope); }}
.in .h {{ font-family: var(--head); font-weight: 700; font-size: 1.18rem; margin: 0 0 .2rem; line-height: 1.2; }}
.in .y {{ font-family: var(--mono); font-size: .72rem; color: var(--muted); margin: 0 0 .35rem; }}
.in .b {{ font-size: .9rem; line-height: 1.5; margin: 0; }}

/* Flight strips */
.bay {{ border: 1px solid var(--line); background: #f6f8f7; padding: .5rem; }}
.bay-h {{ display: flex; justify-content: space-between; align-items: baseline; font-family: var(--head); font-weight: 700; font-size: 1.1rem;
          letter-spacing: .06em; text-transform: uppercase; color: var(--c); margin: 0 0 .45rem; padding: 0 .1rem; }}
.bay-h .cnt {{ font-family: var(--mono); font-size: .74rem; color: var(--muted); letter-spacing: 0; text-transform: none; font-weight: 500; }}
.strip {{ display: grid; grid-template-columns: 6px 5.4rem 1fr auto; align-items: stretch; background: var(--s); border: 1px solid var(--line);
          margin: 0 0 .35rem; font-size: .8rem; }}
.strip .c {{ background: var(--c); }}
.strip .id {{ padding: .3rem .45rem; border-right: 1px dashed var(--line); font-family: var(--mono); font-weight: 600; font-size: .78rem; }}
.strip .id small {{ display: block; font-weight: 400; color: var(--muted); font-size: .64rem; white-space: nowrap; }}
.strip .why {{ padding: .3rem .5rem; line-height: 1.3; min-width: 0; }}
.strip .why small {{ display: block; color: var(--muted); font-family: var(--mono); font-size: .66rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.strip .p {{ padding: .3rem .5rem; border-left: 1px dashed var(--line); font-family: var(--mono); font-weight: 600; font-size: .95rem; color: var(--c);
             display: flex; flex-direction: column; justify-content: center; text-align: right; }}
.strip .p small {{ font-size: .62rem; color: var(--muted); font-weight: 500; }}
.more {{ font-family: var(--mono); font-size: .72rem; color: var(--muted); text-align: center; margin: .2rem 0 0; }}

/* Natural frequency line, notes */
.freq {{ font-size: .98rem; line-height: 1.65; border-left: 3px solid var(--scope); background: var(--scope-soft); padding: .6rem .9rem; margin: .3rem 0 .6rem; }}
.freq b {{ font-family: var(--mono); font-weight: 600; }}
.note {{ font-size: .9rem; line-height: 1.6; color: var(--ink); background: #f6f8f7; border: 1px dashed var(--line); padding: .55rem .8rem; margin: .4rem 0 .2rem; }}
.note p {{ margin: 0 0 .35rem; }} .note p:last-child {{ margin: 0; }}
.cap {{ font-size: .8rem; color: var(--faint); margin: .2rem 0 0; line-height: 1.5; }}
.foot {{ margin: 2.2rem 0 .4rem; padding-top: .8rem; border-top: 1px solid var(--line); font-size: .8rem; color: var(--muted); line-height: 1.7; }}
.foot a {{ color: var(--scope-dark); font-weight: 600; }}
.mono {{ font-family: var(--mono); }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def icon(name: str) -> str:
    return f'<span class="msr" aria-hidden="true">{name}</span>'


def esc(s) -> str:
    return html.escape(str(s))


def pct(p: float) -> str:
    if p < 0.01:
        return "&lt;1%"
    if p > 0.99:
        return "&gt;99%"
    return f"{p:.0%}"


def page(mode: str, title: str, lead: str) -> None:
    st.html(f'<p class="mode">{esc(mode)}</p><h1 class="title">{esc(title)}</h1><p class="lead">{lead}</p>')


def status_line(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<span class="{"on" if i == 0 else ""}">{esc(a)} <b>{b}</b></span>' for i, (a, b) in enumerate(items))
    st.html(f'<div class="status">{cells}</div>')


@contextmanager
def module(code: str, title: str, sub: str = "", band: str | None = None):
    """A console module: bordered panel with corner brackets, a status LED, a mono code and a title."""
    c = f'--c:{t.BAND[band]["color"]};--s:{t.BAND[band]["soft"]}' if band else ""
    with st.container(border=True):
        st.html(f'<div class="mh" style="{c}"><span class="led"></span><span class="code">{esc(code)}</span>'
                f'<p class="mt">{esc(title)}</p></div>' + (f'<p class="ms">{sub}</p>' if sub else '<p class="ms"></p>'))
        yield


def readouts(items: list[tuple[str, str, str]] | list[tuple[str, str, str, str]]) -> None:
    cells = []
    for it in items:
        label, value, note = it[:3]
        color = f' style="--c:{it[3]}"' if len(it) > 3 else ""
        cells.append(f'<div{color}><p class="l">{esc(label)}</p><p class="v">{value}</p><p class="n">{note}</p></div>')
    st.html(f'<div class="ro" style="--n:{len(items)}">{"".join(cells)}</div>')


def lock(p: float, band: str, text: str, rungs: list[tuple[str, str]], facts: list[tuple[str, str]]) -> None:
    b = t.BAND[band]
    ladder = "".join(f'<span class="{"on" if name == band else ""}">{t.BAND[name]["word"].upper()} {esc(rng)}</span>'
                     for name, rng in rungs)
    fx = "".join(f"<span>{esc(a)} <b>{v}</b></span>" for a, v in facts)
    st.html(f'<div class="lock" style="--c:{b["color"]};--s:{b["soft"]}"><div><p class="p">{pct(p)}</p>'
            f'<p class="pl">chance of churn</p></div><div><p class="w">{icon(b["icon"])}{b["word"]}</p>'
            f'<p class="t">{text}</p></div></div><div class="ladder" style="--c:{b["color"]}">{ladder}</div>'
            f'<div class="facts">{fx}</div>')


def signals(rows: list[dict]) -> None:
    """rows: name, value, typical, shap. Bars scaled to the strongest; red right raises, green left lowers."""
    top = max([abs(r["shap"]) for r in rows] + [1e-9])
    out = ['<div class="sig-key"><span></span><div><span>&larr; lowers risk</span><span>raises risk &rarr;</span></div></div>']
    for r in rows:
        w = abs(r["shap"]) / top * 46
        up = r["shap"] > 0
        c = t.RAISES if up else t.LOWERS
        pos = f"left:50%;width:{w:.1f}%" if up else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        lab = (f"left:calc(50% + {w:.1f}% + 6px)" if up else f"right:calc(50% + {w:.1f}% + 6px)")
        typ = f" · typical stayer <b>{esc(r['typical'])}</b>" if r.get("typical") else ""
        out.append(f'<div class="sig"><div><p class="n">{esc(r["name"])}</p><p class="v">this customer <b>{esc(r["value"])}</b>{typ}</p>'
                   f'</div><div class="bar" style="--c:{c}"><i style="{pos}"></i><em style="{lab}">{r["shap"]:+.2f}</em></div></div>')
    st.html("".join(out))


def instructions(items: list[tuple[str, str, str, str]]) -> None:
    """items: (material icon, title, why line, body)."""
    cards = "".join(f'<div class="in"><p class="k">{icon(ic)}Instruction {n}</p><p class="h">{esc(title)}</p>'
                    + (f'<p class="y">{esc(why)}</p>' if why else "") + f'<p class="b">{esc(body)}</p></div>'
                    for n, (ic, title, why, body) in enumerate(items, 1))
    st.html(f'<div class="ins">{cards}</div>')


def strip(ident: str, sub: str, why: str, detail: str, p: float, band: str, tail: str = "") -> str:
    b = t.BAND[band]
    return (f'<div class="strip" style="--c:{b["color"]};--s:{b["soft"]}"><div class="c"></div>'
            f'<div class="id">{esc(ident)}<small>{esc(sub)}</small></div><div class="why">{esc(why)}<small>{esc(detail)}</small></div>'
            f'<div class="p">{pct(p)}<small>{esc(tail)}</small></div></div>')


def bay(band: str, count: int, strips: list[str], shown: int) -> None:
    b = t.BAND[band]
    more = f'<p class="more">+ {count - shown:,} more in the list below</p>' if count > shown else ""
    empty = '<p class="more">No customers in this bay</p>' if count == 0 else ""
    st.html(f'<div class="bay" style="--c:{b["color"]}"><p class="bay-h"><span>{icon(b["icon"])} {b["word"]}</span><span class="cnt">{count:,} customers</span></p>'
            f'{"".join(strips)}{empty}{more}</div>')


def detection(tp: int, fp: int, fn: int, tn: int) -> None:
    """2x2 outcome grid in console words: rows = what happened, columns = what the model said."""
    total = tp + fp + fn + tn
    cell = lambda n, word, sub, c, s: (f'<div style="border:1px solid var(--line);border-top:3px solid {c};background:{s};padding:.6rem .8rem">'  # noqa: E731
                                       f'<p class="mono" style="font-size:1.6rem;font-weight:600;margin:0;color:{c}">{n:,}</p>'
                                       f'<p style="font-weight:700;margin:.1rem 0 0">{word}</p><p class="cap" style="margin:0">{sub} · '
                                       f'{n / total:.1%}</p></div>')
    flag, low, watch = t.BAND["Flag"], t.BAND["Low"], t.BAND["Watch"]
    st.html('<div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:.5rem;align-items:stretch">'
            '<div></div><p class="mode" style="margin:0">Model said: call</p><p class="mode" style="margin:0">Model said: clear</p>'
            '<p class="mode" style="margin:0;align-self:center">Churned</p>'
            + cell(tp, "Churners reached", "called and did leave", low["color"], low["soft"])
            + cell(fn, "Churners missed", "left without a call", flag["color"], flag["soft"])
            + '<p class="mode" style="margin:0;align-self:center">Stayed</p>'
            + cell(fp, "Extra calls", "called but would have stayed", watch["color"], watch["soft"])
            + cell(tn, "Correctly left alone", "not called and stayed", t.MUTED, "#f6f8f7") + "</div>")


def freq(text: str) -> None:
    st.html(f'<div class="freq">{text}</div>')


def note(*paras: str) -> None:
    st.html('<div class="note">' + "".join(f"<p>{p}</p>" for p in paras) + "</div>")


def cap(text: str) -> None:
    st.html(f'<p class="cap">{text}</p>')


def footer() -> None:
    st.html(f'<div class="foot">Model and app by <b>Ahmad Hassan</b>, AI Engineer · <a href="{NOTEBOOK_URL}" target="_blank">'
            f'Notebook</a> · <a href="{CODE_URL}" target="_blank">App code</a> · <a href="{LEARN_URL}" target="_blank">'
            f"Learn with Ahmad</a><br>{DATA_CREDIT}</div>")
