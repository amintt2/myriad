"""The public landing page served by the tracker (myriad/landing.py, myriad/landing/)."""
from __future__ import annotations

import re

import httpx
import pytest

from myriad import landing
from myriad.tracker import Tracker


def client(**kw) -> httpx.AsyncClient:
    tracker = Tracker(db_path=":memory:", **kw)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=tracker.app), base_url="http://tracker.test")


def assert_secure(r: httpx.Response) -> None:
    csp = r.headers["content-security-policy"]
    for part in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self' https://api.github.com",
                 "frame-ancestors 'none'", "base-uri 'none'"):
        assert part in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


async def test_index_served_with_security_headers():
    async with client() as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert_secure(r)
    assert r.headers["cache-control"] == "no-cache"
    html = r.text
    assert "__V__" not in html
    assert f"?v={landing.bundle().version}" in html
    assert "<script>" not in html and not re.search(r"\son\w+=", html) and "style=" not in html  # CSP-clean
    assert 'lang="fr"' in html and "Une myriade de petits modèles, une seule réponse" in html


async def test_every_linked_asset_is_served():
    async with client() as c:
        html = (await c.get("/")).text
        css = (await c.get("/static/landing/landing.css")).text
        paths = set(re.findall(r'(?:src|href|data-full)="(/static/landing/[^"]+)"', html))
        paths |= {"/static/landing/" + u for u in re.findall(r'url\("([^"]+)"\)', css)}
        assert len(paths) > 10
        for p in sorted(paths):
            r = await c.get(p)
            assert r.status_code == 200, p
            assert_secure(r)
            assert r.headers["cache-control"] == landing.IMMUTABLE, p  # all carry the current ?v=hash
            assert r.content


async def test_asset_types_and_cache():
    v = landing.bundle().version
    async with client() as c:
        for name, ctype in (("landing.css", "text/css"), ("landing.js", "text/javascript"), ("icon.svg", "image/svg+xml"),
                            ("img/dashboard.webp", "image/webp"), ("fonts/inter-latin-wght-normal.woff2", "font/woff2")):
            r = await c.get(f"/static/landing/{name}")
            assert r.status_code == 200 and r.headers["content-type"].startswith(ctype), name
            assert r.headers["cache-control"] == landing.SHORT
            r = await c.get(f"/static/landing/{name}?v={v}")
            assert r.headers["cache-control"] == landing.IMMUTABLE
        r = await c.get("/static/landing/landing.css?v=stale")
        assert r.headers["cache-control"] == landing.SHORT
        assert "__V__" not in (await c.get("/static/landing/landing.css")).text


async def test_etag_head_and_missing_files():
    async with client() as c:
        r = await c.get("/")
        r2 = await c.get("/", headers={"If-None-Match": r.headers["etag"]})
        assert r2.status_code == 304 and not r2.content
        h = await c.head("/")
        assert h.status_code == 200 and h.headers["content-length"] == str(len(r.content)) and not h.content
        for bad in ("/static/landing/nope.js", "/static/landing/../tracker.py", "/static/landing/%2e%2e/tracker.py",
                    "/static/landing/index.html", "/static/landing/", "/static/landing/img"):
            r = await c.get(bad)
            assert r.status_code == 404, bad
        robots = await c.get("/robots.txt")
        assert robots.status_code == 200 and "Disallow: /v1/" in robots.text


async def test_v1_routes_unaffected():
    async with client() as c:
        h = await c.get("/v1/health")
        assert h.status_code == 200 and h.json()["ok"] is True
        assert "content-security-policy" not in h.headers  # the API keeps its own headers
        s = await c.get("/v1/stats")
        assert s.status_code == 200 and "nodes_online" in s.json()
        assert (await c.get("/v1/peers")).json() == {"peers": []}
        assert (await c.get("/v1/nope")).status_code == 404
    paths = [getattr(r, "path", "") for r in Tracker(db_path=":memory:").app.routes]
    assert "/v1/ws" in paths and "/" in paths
    assert not any(p.startswith("/v1") for p in paths if p in ("/", landing.PREFIX + "{name:path}", "/robots.txt"))


async def test_landing_can_be_disabled():
    async with client(landing=False) as c:
        assert (await c.get("/")).status_code == 404
        assert (await c.get("/static/landing/landing.css")).status_code == 404
        assert (await c.get("/v1/health")).status_code == 200


@pytest.mark.parametrize("argv,expected", [(["tracker"], True), (["tracker", "--no-landing"], False)])
def test_cli_flag(monkeypatch, argv, expected):
    import myriad.tracker as tmod
    from myriad import cli

    seen = {}
    monkeypatch.setattr(tmod, "run", lambda **kw: seen.update(kw))
    assert cli.main(argv) == 0
    assert seen["landing"] is expected
