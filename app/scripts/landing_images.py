"""Optimised copies of the app screenshots (docs/screenshot-*.png) for the landing page
(myriad/landing/img/*.webp), plus the social preview image (og.jpg, 1200x630). Needs Pillow:

    uv run --extra desktop python scripts/landing_images.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

APP = Path(__file__).resolve().parent.parent
DOCS = APP / "docs"
OUT = APP / "myriad" / "landing" / "img"
SHOTS = ["dashboard", "chat", "agents", "wizard-models", "peers", "dashboard-light-en"]
WIDTH = 1200


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SHOTS:
        im = Image.open(DOCS / f"screenshot-{name}.png").convert("RGB")
        if im.width > WIDTH:
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
        dst = OUT / f"{name}.webp"
        im.save(dst, "WEBP", quality=80, method=6)
        print(f"{dst.relative_to(APP)}: {im.width}x{im.height}, {dst.stat().st_size // 1024} kB")
    # Social preview: the dashboard, cropped to 1200x630 around its top.
    im = Image.open(DOCS / "screenshot-dashboard.png").convert("RGB")
    scale = 1200 / im.width
    im = im.resize((1200, round(im.height * scale)), Image.LANCZOS).crop((0, 0, 1200, 630))
    dst = OUT / "og.jpg"
    im.save(dst, "JPEG", quality=82, optimize=True, progressive=True)
    print(f"{dst.relative_to(APP)}: 1200x630, {dst.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main()
