"""Churn Radar colours and fonts: a light air-traffic-control console. Pale console grey, white modules with green
corner brackets, a radar scope, flight-strip style customer strips. Status colours (clear / watch / call) always come
with a word and an icon."""

BG = "#eef1f0"          # console
PANEL = "#ffffff"
LINE = "#c6d1cd"
INK = "#14201c"
MUTED = "#4c5a55"
FAINT = "#6b7873"
GRID = "#e3e9e7"

SCOPE = "#2e7d32"       # radar green: rings, brackets, accents (5.1:1 on white)
SCOPE_DARK = "#1f5c23"
SCOPE_SOFT = "#e9f4e9"
RING = "#c2dcc3"

# Risk bands (the model's names) as console states
BAND = {
    "Low": {"word": "Clear", "color": "#2e7d32", "soft": "#e9f4e9", "icon": "check_circle"},
    "Watch": {"word": "Watch", "color": "#b26a00", "soft": "#fff4e0", "icon": "visibility"},
    "Flag": {"word": "Call", "color": "#c0392b", "soft": "#fdecea", "icon": "call"},
}
RAISES, LOWERS = "#c0392b", "#2e7d32"   # SHAP: raises / lowers the churn risk (always labelled in words too)
STAYED, CHURNED = "#8fa39c", "#c0392b"  # test-set outcome dots

HEAD = "'Rajdhani', 'Segoe UI', sans-serif"
BODY = "'Source Sans 3', 'Segoe UI', sans-serif"
MONO = "'Azeret Mono', 'Consolas', monospace"
