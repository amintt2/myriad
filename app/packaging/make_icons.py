"""Regenerate the Myriad icons from myriad/icon.py (the outputs are committed, CI does not need this).

    uv run --extra desktop python packaging/make_icons.py
"""
from __future__ import annotations

from pathlib import Path

from myriad.icon import render_icon, svg

HERE = Path(__file__).resolve().parent
OUT = HERE / "icons"
WEB = HERE.parent / "myriad" / "web"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "myriad.svg").write_text(svg(1024), encoding="utf-8")
    (WEB / "icon.svg").write_text(svg(256), encoding="utf-8")
    big = render_icon(1024)
    big.save(OUT / "myriad-1024.png")
    for s in (512, 256, 128, 64, 32):
        render_icon(s).save(OUT / f"myriad-{s}.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = {s: render_icon(s) for s in sizes}
    imgs[256].save(OUT / "myriad.ico", sizes=[(s, s) for s in sizes], append_images=[imgs[s] for s in sizes[:-1]])
    big.save(OUT / "myriad.icns")
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
