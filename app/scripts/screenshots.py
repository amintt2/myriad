"""Headless screenshots of the interface (app/docs/*.png), with Chrome or Edge driven over the DevTools
protocol (no Playwright, only `websockets` which the app already needs).

    uv run python scripts/demo_swarm.py --port 8471 --wizard-port 8472     # in another terminal
    uv run python scripts/screenshots.py --dashboard http://127.0.0.1:8471/ --wizard http://127.0.0.1:8472/
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

OUT = Path(__file__).resolve().parent.parent / "docs"
BROWSERS = [
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def find_browser() -> str:
    for b in BROWSERS:
        if b and (Path(b).exists() or shutil.which(b)):
            return b if Path(b).exists() else shutil.which(b)
    raise SystemExit("Chrome / Edge introuvable (variable CHROME pour l'indiquer)")


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def goto(self, url: str, wait: float = 2.0):
        self.n_nav = getattr(self, "n_nav", 0) + 1  # a new query string: always a full page load
        base, _, frag = url.partition("#")
        url = f"{base}{'&' if '?' in base else '?'}v={self.n_nav}" + (f"#{frag}" if frag else "")
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def shot(self, path: Path):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(r["data"]))
        print("wrote", path)


MOBILE_UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/126.0.0.0 Mobile Safari/537.36")


async def landing(p: Page, url: str) -> None:
    """The public landing page served by a tracker (FR, dark): desktop hero, full page, phone."""
    async def open_page(wait: float):
        await p.goto(url, 0.6)
        await p.js("localStorage.setItem('myriad.site.theme','dark'); localStorage.setItem('myriad.site.lang','fr'); true")
        await p.goto(url, wait)  # the murmuration gathers into the disc ~4-8 s after the page opens

    async def reveal_all():  # scroll down once so that every section has played its entrance
        h = await p.js("document.documentElement.scrollHeight")
        for y in range(0, int(h), 500):
            await p.js(f"window.scrollTo(0, {y}); true")
            await asyncio.sleep(0.12)
        await asyncio.sleep(1.6)
        await p.js("window.scrollTo(0, 0); true")
        await asyncio.sleep(0.4)

    await p.call("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
    await open_page(5.6)
    await p.shot(OUT / "landing-desktop.png")
    await reveal_all()
    h = await p.js("document.querySelector('.top').style.position = 'absolute'; "  # sticky header: once, on top
                   "document.querySelector('.top').style.width = '100%'; document.documentElement.scrollHeight")
    r = await p.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                     clip={"x": 0, "y": 0, "width": 1440, "height": h, "scale": 0.5})
    (OUT / "landing-full.png").write_bytes(base64.b64decode(r["data"]))
    print("wrote", OUT / "landing-full.png")

    await p.call("Emulation.setUserAgentOverride", userAgent=MOBILE_UA)
    await p.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
    await open_page(5.6)
    await p.shot(OUT / "landing-mobile.png")
    await p.js("window.scrollTo(0, document.getElementById('reseau').offsetTop - 70); true")
    await asyncio.sleep(1.5)
    await p.shot(OUT / "landing-mobile-network.png")


async def run(args) -> None:
    OUT.mkdir(exist_ok=True)
    port = 9333
    profile = tempfile.mkdtemp(prefix="myriad-shots-")
    proc = subprocess.Popen([find_browser(), "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--hide-scrollbars", "--no-first-run",
                             "--window-size=1440,900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        target = None
        for _ in range(100):
            try:
                tabs = httpx.get(f"http://127.0.0.1:{port}/json/list", timeout=1).json()
                target = next(t for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.1)
        if target is None:
            raise SystemExit("le navigateur sans tête n'a pas démarré")
        async with connect(target["webSocketDebuggerUrl"], max_size=64 << 20) as ws:
            p = Page(ws)
            await p.call("Page.enable")
            await p.call("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=args.scale,
                         mobile=False)

            async def prefs(base: str, theme: str, lang: str):
                await p.goto(base, 0.8)
                await p.js(f"localStorage.setItem('myriad.theme','{theme}'); localStorage.setItem('myriad.lang','{lang}')")

            d = args.dashboard

            async def agents(name: str, live: str | None):
                """The Agents view running the demo plan: one shot while it runs, one when it is done."""
                await p.goto(d + "#agents", 1.5)
                await p.js("document.getElementById('ag-demo').click(); true")
                await asyncio.sleep(2.2)
                await p.js("window.scrollTo(0, document.getElementById('ag-out').offsetTop - 24); true")
                if live:
                    await asyncio.sleep(0.3)
                    await p.shot(OUT / live)
                for _ in range(60):  # until the run is over
                    if await p.js("!!(window.MyriadAgents && window.MyriadAgents.state.result)"):
                        break
                    await asyncio.sleep(0.5)
                await asyncio.sleep(0.8)
                await p.js("window.scrollTo(0, document.getElementById('ag-out').offsetTop - 24); true")
                await asyncio.sleep(0.3)
                await p.shot(OUT / name)

            if args.landing:
                await landing(p, args.landing)
                return
            if args.only_agents:
                await prefs(d, "dark", "fr")
                await agents("screenshot-agents.png", "screenshot-agents-live.png")
                await prefs(d, "light", "en")
                await agents("screenshot-agents-light-en.png", None)
                return
            await prefs(d, "dark", "fr")
            await p.goto(d + "#dashboard", args.settle)
            await p.shot(OUT / "screenshot-dashboard.png")
            await p.goto(d + "#chat", 1.5)
            await p.js("document.querySelector('#examples button').click();"
                       "document.getElementById('chat').requestSubmit(); true")
            # the conversation: the question, the answer and its "how the swarm decided" panel, opened
            show_turn = ("const d = document.querySelector('.turn:last-of-type .decision'); if (d) d.open = true;"
                         "const t = document.querySelector('.turn:last-of-type');"
                         "if (t) window.scrollTo(0, t.getBoundingClientRect().top + window.scrollY - 24); true")
            await asyncio.sleep(0.5)  # while the peers are still thinking
            await p.js(show_turn)
            await asyncio.sleep(0.2)
            await p.shot(OUT / "screenshot-chat-live.png")
            await asyncio.sleep(6)
            await p.js(show_turn)
            await asyncio.sleep(1.0)
            await p.shot(OUT / "screenshot-chat.png")
            await p.goto(d + "#peers", 2.5)
            await p.shot(OUT / "screenshot-peers.png")
            await agents("screenshot-agents.png", "screenshot-agents-live.png")
            await prefs(d, "light", "en")
            await p.goto(d + "#dashboard", args.settle)
            await p.shot(OUT / "screenshot-dashboard-light-en.png")
            await p.goto(d + "#about", 1.5)
            await p.shot(OUT / "screenshot-about-light-en.png")
            await agents("screenshot-agents-light-en.png", None)
            if args.wizard:
                w = args.wizard
                await prefs(w, "dark", "fr")
                await p.goto(w, 3.5)
                await p.shot(OUT / "screenshot-wizard-welcome.png")
                await p.js("document.getElementById('wz-next').click(); true")
                await asyncio.sleep(1.2)
                await p.shot(OUT / "screenshot-wizard-hardware.png")
                await p.js("document.getElementById('wz-next').click(); true")
                await asyncio.sleep(1.2)
                await p.shot(OUT / "screenshot-wizard-models.png")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard", default="http://127.0.0.1:8471/")
    ap.add_argument("--wizard", default="http://127.0.0.1:8472/")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--settle", type=float, default=7.0, help="seconds to let the live map fill")
    ap.add_argument("--only-agents", action="store_true", help="only the Agents view (screenshot-agents*.png)")
    ap.add_argument("--landing", metavar="URL", help="only the landing page served by this tracker (landing-*.png)")
    asyncio.run(run(ap.parse_args()))
