"""Public landing page of the network, served by the tracker: GET / and GET /static/landing/<file>.

The page is static (myriad/landing/: index.html, CSS, JS, images); the fonts and the icon are the app's
own (myriad/web/), served under the same prefix. Files are read once, on the first request, and only the
files found there with a known type can be served (no path is ever joined with user input).

Caching: every text asset carries the placeholder __V__, replaced by a short hash of all the assets; the
page links its assets as `?v=<hash>`. A request with the current hash is cached for a year (immutable),
anything else for an hour, and the page itself is revalidated on every visit (ETag).
Security headers: a strict Content-Security-Policy (no inline script or style; the only foreign origin
is the GitHub API, to find the download links of the latest release), nosniff, no referrer, no framing.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.responses import Response

PREFIX = "/static/landing/"
GITHUB_API = "https://api.github.com"
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
       f"connect-src 'self' {GITHUB_API}; manifest-src 'self'; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), browsing-topics=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
TEXT = (".html", ".css", ".js", ".svg", ".txt")
PLACEHOLDER = b"__V__"
IMMUTABLE = "public, max-age=31536000, immutable"
SHORT = "public, max-age=3600"
INDEX_CACHE = "no-cache"
ROBOTS = b"User-agent: *\nAllow: /\nDisallow: /v1/\n"
# Shared with the app's interface instead of copied: fonts (SIL OFL) and the icon.
SHARED = {"icon.svg": ("web", "icon.svg"),
          "fonts/inter-latin-wght-normal.woff2": ("web", "fonts", "inter-latin-wght-normal.woff2"),
          "fonts/inter-latin-ext-wght-normal.woff2": ("web", "fonts", "inter-latin-ext-wght-normal.woff2"),
          "fonts/jetbrains-mono-latin-wght-normal.woff2": ("web", "fonts", "jetbrains-mono-latin-wght-normal.woff2")}


@dataclass(frozen=True)
class Bundle:
    version: str
    files: dict  # name -> (body, content type, etag)

    def get(self, name: str):
        return self.files.get(name)


def _walk(node, prefix: str = ""):
    for child in node.iterdir():
        name = f"{prefix}{child.name}"
        if child.is_dir():
            yield from _walk(child, name + "/")
        elif child.is_file():
            yield name, child


@functools.cache
def bundle() -> Bundle:
    """All the landing files in memory (read once; a few hundred kB)."""
    root = resources.files("myriad")
    raw: dict[str, bytes] = {}
    for name, f in _walk(root.joinpath("landing")):
        if "." + name.rsplit(".", 1)[-1].lower() in TYPES and not name.split("/")[-1].startswith("."):
            raw[name] = f.read_bytes()
    for name, parts in SHARED.items():
        f = root.joinpath(*parts)
        if f.is_file():
            raw[name] = f.read_bytes()
    h = hashlib.sha256()
    for name in sorted(raw):
        h.update(name.encode() + b"\0" + hashlib.sha256(raw[name]).digest())
    version = h.hexdigest()[:12]
    files = {}
    for name, body in raw.items():
        ext = "." + name.rsplit(".", 1)[-1].lower()
        if ext in TEXT:
            body = body.replace(PLACEHOLDER, version.encode())
        files[name] = (body, TYPES[ext], '"' + hashlib.sha256(body).hexdigest()[:20] + '"')
    return Bundle(version, files)


def _not_modified(request: Request, etag: str) -> bool:
    given = request.headers.get("if-none-match", "")
    return any(tag.strip().removeprefix("W/") == etag for tag in given.split(",")) if given else False


def _respond(request: Request, entry, cache: str) -> Response:
    body, media, etag = entry
    headers = {**SECURITY_HEADERS, "Cache-Control": cache, "ETag": etag}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    if request.method == "HEAD":
        return Response(status_code=200, headers={**headers, "Content-Length": str(len(body))}, media_type=media)
    return Response(body, media_type=media, headers=headers)


def _missing() -> Response:
    return Response(b"introuvable\n", status_code=404, media_type="text/plain; charset=utf-8",
                    headers={**SECURITY_HEADERS, "Cache-Control": "no-store"})


def install(app: FastAPI) -> None:
    """Serve the landing page at / and its files under /static/landing/ (never under /v1/)."""

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def landing_index(request: Request):
        entry = bundle().get("index.html")
        return _missing() if entry is None else _respond(request, entry, INDEX_CACHE)

    @app.api_route(PREFIX + "{name:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def landing_asset(name: str, request: Request):
        b = bundle()
        entry = b.get(name) if name != "index.html" else None
        if entry is None:
            return _missing()
        return _respond(request, entry, IMMUTABLE if request.query_params.get("v") == b.version else SHORT)

    @app.api_route("/robots.txt", methods=["GET", "HEAD"], include_in_schema=False)
    async def robots(request: Request):
        return _respond(request, (ROBOTS, TYPES[".txt"], '"robots-1"'), SHORT)
