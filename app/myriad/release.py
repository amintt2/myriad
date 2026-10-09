"""Releases of the Myriad app: strict version parsing and comparison, SHA256SUMS parsing, and the
tracker's release watcher.

The tracker polls GitHub ONCE for the whole network (every `poll_s`, conditional requests with ETag,
backoff on errors and rate limits) and tells the nodes the latest version: in the Welcome frame, by an
UpdateAvailable push frame and on GET /v1/version. Nodes therefore never poll the GitHub API themselves
(60 unauthenticated requests per hour and per IP address), except as a last resort when the tracker is
unreachable (updater.py).

What the tracker says is only a HINT that a version exists: a client builds the download URLs itself
from its pinned repository and the validated version, and checks the SHA-256 against the release's own
SHA256SUMS.txt (updater.py). The tracker is never trusted for what to download."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable

import httpx

log = logging.getLogger("myriad.release")

DEFAULT_RELEASE_REPO = "amintt2/myriad"
REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})/[A-Za-z0-9_.](?:[A-Za-z0-9_.-]{0,99})$")
# MAJOR.MINOR.PATCH only, no leading zeros, no pre-release or build suffix (pre-releases are ignored).
VERSION_RE = re.compile(r"^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$")
SUMS_NAME = "SHA256SUMS.txt"
SUMS_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *]?([A-Za-z0-9][A-Za-z0-9._+-]{0,199})$")
ASSET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$")
MAX_SUMS_BYTES = 256 << 10
MAX_API_BYTES = 4 << 20
MAX_ASSETS = 64
USER_AGENT = "Myriad release watcher"


def parse_version(v) -> tuple[int, int, int] | None:
    """'1.2.3' -> (1, 2, 3); anything else (prefix 'v', suffix, leading zeros, not a str) -> None."""
    if not isinstance(v, str) or len(v) > 32:
        return None
    m = VERSION_RE.fullmatch(v)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def is_newer(candidate, current) -> bool:
    """True only if both versions are valid and `candidate` is strictly greater (never a downgrade)."""
    a, b = parse_version(candidate), parse_version(current)
    return a is not None and b is not None and a > b


def version_from_tag(tag) -> str | None:
    """'v1.2.3' -> '1.2.3' (a tag without the 'v' is accepted too); None if malformed."""
    if not isinstance(tag, str):
        return None
    v = tag[1:] if tag.startswith("v") else tag
    return v if parse_version(v) else None


def parse_sums(text: str) -> dict[str, str]:
    """`sha256sum` / `shasum -a 256` output -> {file name: lowercase hex digest}. Malformed lines are
    skipped; a name listed twice with different digests is dropped (ambiguous)."""
    out: dict[str, str] = {}
    bad: set[str] = set()
    for line in text.splitlines():
        m = SUMS_LINE.fullmatch(line.strip())
        if not m:
            continue
        digest, name = m[1].lower(), m[2]
        if name in out and out[name] != digest:
            bad.add(name)
        out[name] = digest
    for name in bad:
        out.pop(name, None)
    return out


def parse_release(data) -> dict | None:
    """A GitHub release (GET /repos/{repo}/releases/latest) -> the cached summary, or None for a draft,
    a pre-release or a malformed answer."""
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    tag = data.get("tag_name")
    version = version_from_tag(tag)
    if version is None:
        return None
    assets: dict[str, dict] = {}
    for a in (data.get("assets") or [])[:MAX_ASSETS] if isinstance(data.get("assets"), list) else []:
        if not isinstance(a, dict):
            continue
        name, url, size = a.get("name"), a.get("browser_download_url"), a.get("size")
        if not isinstance(name, str) or not ASSET_NAME.fullmatch(name) or not isinstance(url, str):
            continue
        if not url.startswith("https://github.com/"):
            continue
        assets[name] = {"url": url[:500], "size": size if isinstance(size, int) and size >= 0 else None}
    published = data.get("published_at")
    html_url = data.get("html_url")
    return {"version": version, "tag": tag, "published_at": published if isinstance(published, str) else None,
            "html_url": html_url if isinstance(html_url, str) and html_url.startswith("https://github.com/") else None,
            "assets": assets, "sha256": {}}


async def get_capped(client: httpx.AsyncClient, url: str, cap: int, headers: dict | None = None) -> httpx.Response:
    """GET with a body size cap; raises ValueError beyond it. Returns a fully read response."""
    async with client.stream("GET", url, headers=headers or {}) as r:
        length = r.headers.get("content-length")
        if length and length.isdigit() and int(length) > cap:
            raise ValueError(f"réponse trop grande ({length} octets)")
        chunks, n = [], 0
        async for chunk in r.aiter_bytes():
            n += len(chunk)
            if n > cap:
                raise ValueError("réponse trop grande")
            chunks.append(chunk)
        headers_out = [(k, v) for k, v in r.headers.multi_items() if k.lower() not in ("content-encoding", "content-length")]
        return httpx.Response(r.status_code, headers=headers_out, content=b"".join(chunks), request=r.request)


class ReleaseWatcher:
    """Tracker side: polls the latest release of `repo` on GitHub and keeps a summary of it.

    One conditional request (If-None-Match) every `poll_s`: an unchanged release costs a 304, which
    GitHub does not count against the rate limit. Errors back off exponentially (up to `max_backoff_s`);
    a rate-limit answer waits for the reset time GitHub announces. Nothing here may stop the tracker."""

    def __init__(self, repo: str = DEFAULT_RELEASE_REPO, poll_s: float = 600.0, http: httpx.AsyncClient | None = None,
                 on_change: Callable[[dict], None] | None = None, max_backoff_s: float = 6 * 3600.0,
                 api_base: str = "https://api.github.com", download_base: str = "https://github.com"):
        if not isinstance(repo, str) or not REPO_RE.fullmatch(repo):
            raise ValueError(f"dépôt invalide : {repo!r} (attendu : propriétaire/nom)")
        self.repo, self.poll_s, self.max_backoff_s = repo, max(30.0, float(poll_s)), max_backoff_s
        self.api_url = f"{api_base}/repos/{repo}/releases/latest"
        self.download_base = download_base
        self._http, self._own_http = http, http is None
        self.on_change = on_change
        self.latest: dict | None = None
        self.etag: str | None = None
        self.checked_at: float | None = None  # wall time of the last successful answer (200 or 304)
        self.error: str | None = None
        self.failures = 0
        self._task: asyncio.Task | None = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=True,
                                           headers={"User-Agent": USER_AGENT})
        return self._http

    async def poll_once(self) -> float | None:
        """One check. Returns how long to wait before the next one when GitHub asks for it (rate
        limit), else None. Raises on network or format errors (the loop backs off)."""
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": USER_AGENT}
        if self.etag and self.latest is not None and self.latest.get("sha256"):
            headers["If-None-Match"] = self.etag
        http = self._client()
        r = await get_capped(http, self.api_url, MAX_API_BYTES, headers)
        if r.status_code == 304:
            self.checked_at, self.error = time.time(), None
            return None
        if r.status_code in (403, 429):
            wait = _rate_limit_wait(r.headers)
            raise RateLimited(f"GitHub : HTTP {r.status_code} (limite de requêtes)", wait)
        if r.status_code == 404:  # no published release yet
            self.checked_at, self.error = time.time(), None
            return None
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
        info = parse_release(r.json())
        if info is None:
            raise ValueError("dernière version GitHub inexploitable (brouillon, pré-version ou format inattendu)")
        if SUMS_NAME in info["assets"]:
            # Built from the repository and the tag, not from the URL in the answer.
            url = f"{self.download_base}/{self.repo}/releases/download/{info['tag']}/{SUMS_NAME}"
            s = await get_capped(http, url, MAX_SUMS_BYTES, {"User-Agent": USER_AGENT})
            if s.status_code == 200:
                info["sha256"] = {k: v for k, v in parse_sums(s.text).items() if k in info["assets"]}
        # Without the checksums, ask again in full next time (no If-None-Match).
        self.etag = r.headers.get("etag") if info["sha256"] else None
        changed = self.latest is None or self.latest.get("version") != info["version"]
        self.latest, self.checked_at, self.error = info, time.time(), None
        if changed:
            log.info("latest Myriad release: %s", info["version"])
            if self.on_change is not None:
                try:
                    self.on_change(info)
                except Exception:  # never let a listener break the watcher
                    log.exception("release listener failed")
        return None

    async def run(self) -> None:
        while True:
            wait = self.poll_s
            try:
                asked = await self.poll_once()
                self.failures = 0
                if asked:
                    wait = max(wait, asked)
            except asyncio.CancelledError:
                raise
            except RateLimited as e:
                self.failures += 1
                self.error = str(e)
                wait = min(self.max_backoff_s, max(e.wait_s or 0.0, self.poll_s * 2 ** min(self.failures, 10)))
                log.warning("release check: %s; next try in %.0f s", e, wait)
            except Exception as e:  # network, HTTP, JSON: back off, never crash the tracker
                self.failures += 1
                self.error = f"{type(e).__name__}: {e}"[:300]
                wait = min(self.max_backoff_s, self.poll_s * 2 ** min(self.failures, 10))
                log.warning("release check failed (%s); next try in %.0f s", self.error, wait)
            await asyncio.sleep(wait)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="release-watcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._own_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    def public(self) -> dict:
        """The GET /v1/version body."""
        return {"repo": self.repo, "latest": self.latest, "checked_at": self.checked_at}


class RateLimited(RuntimeError):
    def __init__(self, msg: str, wait_s: float | None):
        super().__init__(msg)
        self.wait_s = wait_s


def _rate_limit_wait(headers) -> float | None:
    ra = headers.get("retry-after")
    if ra and ra.isdigit():
        return float(ra)
    reset = headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        return max(0.0, float(reset) - time.time()) + 5.0
    return None
