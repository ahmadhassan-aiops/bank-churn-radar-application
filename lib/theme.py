"""This app's colours and icon. Same family look as the analysis apps, with its own accent."""

ACCENT = "#2459c4"      # charts and highlights (6.2:1 on the page background)
ACCENT_INK = "#1d4aa8"  # text in the accent colour (7.9:1)
SERIES_2 = "#eb6834"    # second series (precision next to recall)
ICON = "credit_card"    # Material Symbols name for the hero badge
HERO_MOTIF = ""         # extra CSS for the hero band, if any

# Risk bands are status colours: always shown with a label and an icon, never colour alone
RISK = {
    "Low": {"label": "Low risk", "color": "#0f7a3a", "ink": "#0f7a3a", "soft": "#e9f5ee", "icon": "verified"},
    "Watch": {"label": "Watch", "color": "#b77800", "ink": "#8a5a00", "soft": "#fdf6e7", "icon": "visibility"},
    "Flag": {"label": "Flag for a call", "color": "#c4382b", "ink": "#a32b20", "soft": "#fbeceb", "icon": "warning"},
}
RAISES = "#c4382b"   # SHAP bar that raises the churn risk
LOWERS = "#2459c4"   # SHAP bar that lowers it


def _mix(color: str, other: str, t: float) -> str:
    """Blend `color` towards `other` by t (0 = color, 1 = other)."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


ACCENT_SOFT = _mix(ACCENT, "#ffffff", 0.92)   # callout and hero backgrounds
ACCENT_TINT = _mix(ACCENT, "#ffffff", 0.72)   # borders and decoration
QUIET = _mix(ACCENT, "#e4e4e0", 0.78)         # de-emphasised marks
SEQUENTIAL = [
    [0.0, _mix(ACCENT, "#ffffff", 0.95)], [0.35, _mix(ACCENT, "#ffffff", 0.6)], [0.7, ACCENT],
    [1.0, _mix(ACCENT, "#000000", 0.45)],
]
