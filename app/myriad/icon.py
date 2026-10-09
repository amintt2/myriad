"""The Myriad icon: many small points (a sunflower / Vogel spiral) forming one disc, warm at the centre
(the answer) and cool at the edge (the peers). One geometry, two renderings: SVG (UI, sources of the
packaging icons) and Pillow (tray icon, .png/.ico/.icns)."""
from __future__ import annotations

import math

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
N_POINTS = 89
STOPS = ((0.0, (255, 214, 110)), (0.35, (255, 106, 160)), (0.7, (124, 108, 255)), (1.0, (58, 214, 255)))
BG_TOP, BG_BOTTOM = (20, 24, 48), (8, 10, 20)


def _mix(t: float) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(a + (b - a) * u) for a, b in zip(c0, c1))
    return STOPS[-1][1]


def points(n: int = N_POINTS) -> list[tuple[float, float, float, tuple[int, int, int]]]:
    """(x, y, radius, rgb) in a unit square centred on (0.5, 0.5)."""
    out = []
    outer = 0.36
    for i in range(n):
        f = math.sqrt((i + 0.5) / n)
        a = i * GOLDEN_ANGLE
        r = outer * f
        dot = 0.012 + 0.020 * (1 - f) ** 0.8
        out.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a), dot, _mix(f)))
    return out


def svg(size: int = 1024, background: bool = True) -> str:
    s = size
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s} {s}" width="{s}" height="{s}">']
    if background:
        parts.append('<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
                     f'<stop offset="0" stop-color="rgb{BG_TOP}"/><stop offset="1" stop-color="rgb{BG_BOTTOM}"/>'
                     '</linearGradient></defs>')
        parts.append(f'<rect x="{s * 0.04:.1f}" y="{s * 0.04:.1f}" width="{s * 0.92:.1f}" height="{s * 0.92:.1f}" '
                     f'rx="{s * 0.22:.1f}" fill="url(#bg)"/>')
    for x, y, r, c in points():
        parts.append(f'<circle cx="{x * s:.1f}" cy="{y * s:.1f}" r="{r * s:.1f}" fill="rgb{c}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_icon(size: int = 256, background: bool = True):
    """A Pillow RGBA image (drawn 4x larger, then downscaled for smooth edges)."""
    from PIL import Image, ImageDraw

    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if background:
        grad = Image.new("RGBA", (1, S))
        for yy in range(S):
            u = yy / max(1, S - 1)
            grad.putpixel((0, yy), tuple(round(a + (b - a) * u) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))
        grad = grad.resize((S, S))
        mask = Image.new("L", (S, S), 0)
        m = round(S * 0.04)
        ImageDraw.Draw(mask).rounded_rectangle((m, m, S - m, S - m), radius=round(S * 0.22), fill=255)
        img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    small = size <= 48  # at tray sizes, fewer and bigger dots read better
    for x, y, r, c in points(34 if small else N_POINTS):
        rr = (r * (1.9 if small else 1.0)) * S
        d.ellipse((x * S - rr, y * S - rr, x * S + rr, y * S + rr), fill=c + (255,))
    return img.resize((size, size), Image.LANCZOS)
