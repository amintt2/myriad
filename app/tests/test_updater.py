"""App updates: version parsing, platform assets, pinned download URLs whatever the tracker says, SHA-256
checks, the tracker's release watcher (/v1/version, Welcome field, UpdateAvailable push) with a mocked
GitHub, backward compatibility with nodes that did not ask for the feature, install-kind detection,
applying an update, the UI endpoints and the CLI. No real network."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from myriad import release, updater
from myriad.config import Config
from myriad.crypto import Identity
from myriad.node import NodeClient
from myriad.protocol import UpdateAvailable, Welcome, dump_frame, parse_frame
from myriad.release import ReleaseWatcher, is_newer, parse_release, parse_sums, parse_version
from myriad.tracker import Tracker, requested_features
from myriad.updater import InstallKind, Updater, UpdateError, detect_install

from .conftest import start_swarm, wait_until

REPO = "amintt2/myriad"


# ---------------------------------------------------------------------------------------------------
# versions, checksums

def test_version_parsing_and_comparison():
    assert parse_version("1.2.3") == (1, 2, 3) and parse_version("0.10.0") == (0, 10, 0)
    for bad in ("v1.2.3", "1.2", "1.2.3.4", "01.2.3", "1.2.3-rc1", "1.2.3+b", " 1.2.3", "1.2.3\n", "", None, 123,
                "1.2.x", "../1.2.3", "1.2.3/../../x", "9" * 40):
        assert parse_version(bad) is None, bad
    assert is_newer("0.10.0", "0.9.9") and is_newer("1.0.0", "0.99.99") and is_newer("0.1.1", "0.1.0")
    assert not is_newer("0.1.0", "0.1.0") and not is_newer("0.0.9", "0.1.0")  # never a downgrade
    assert not is_newer("2.0.0-rc1", "0.1.0") and not is_newer("garbage", "0.1.0") and not is_newer("1.0.0", "x")


def test_parse_sums():
    a, b = "a" * 64, "B" * 64
    text = (f"{a}  Myriad-Setup-0.2.0.exe\n{b} *Myriad-0.2.0-arm64.dmg\nnot a line\n{a}  ../evil\n"
            f"{a}  dup.bin\n{b}  dup.bin\n")
    sums = parse_sums(text)
    assert sums == {"Myriad-Setup-0.2.0.exe": a, "Myriad-0.2.0-arm64.dmg": b.lower()}  # ambiguous name dropped


def test_parse_release_ignores_drafts_prereleases_and_foreign_urls():
    base = {"tag_name": "v0.2.0", "published_at": "2026-10-01T00:00:00Z", "html_url": "https://github.com/x/y",
            "assets": [{"name": "Myriad-Setup-0.2.0.exe", "size": 5,
                        "browser_download_url": "https://github.com/amintt2/myriad/releases/download/v0.2.0/Myriad-Setup-0.2.0.exe"},
                       {"name": "x.exe", "size": 1, "browser_download_url": "https://evil.example/x.exe"},
                       {"name": "../bad", "size": 1, "browser_download_url": "https://github.com/a"}]}
    info = parse_release(base)
    assert info["version"] == "0.2.0" and list(info["assets"]) == ["Myriad-Setup-0.2.0.exe"]
    assert parse_release({**base, "draft": True}) is None and parse_release({**base, "prerelease": True}) is None
    assert parse_release({**base, "tag_name": "v0.2.0-beta"}) is None and parse_release("nope") is None


# ---------------------------------------------------------------------------------------------------
# install kinds and assets

def test_install_kind_detection_and_assets(tmp_path):
    win = tmp_path / "Programs" / "Myriad"
    win.mkdir(parents=True)
    (win / "Myriad.exe").write_bytes(b"")
    k = detect_install(frozen=True, executable=str(win / "Myriad.exe"), platform="win32", env={}, machine="AMD64")
    assert k.kind == "windows-portable"
    assert not k.can_apply and k.asset("0.2.0") is None
    (win / "unins000.exe").write_bytes(b"")
    k = detect_install(frozen=True, executable=str(win / "Myriad.exe"), platform="win32", env={}, machine="AMD64")
    assert k.kind == "windows-installer" and k.can_apply and k.asset("0.2.0") == "Myriad-Setup-0.2.0.exe"

    app = tmp_path / "Applications" / "Myriad.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    exe = str(app / "Myriad")
    k = detect_install(frozen=True, executable=exe, platform="darwin", env={}, machine="arm64", writable=lambda p: True)
    assert k.kind == "macos-app" and k.can_apply and k.asset("0.2.0") == "Myriad-0.2.0-arm64.dmg"
    assert Path(k.target).name == "Myriad.app"
    k = detect_install(frozen=True, executable=exe, platform="darwin", env={}, machine="arm64", writable=lambda p: False)
    assert k.kind == "macos-app" and not k.can_apply and k.reason
    k = detect_install(frozen=True, executable="/private/var/folders/x/AppTranslocation/ABC/d/Myriad.app/Contents/MacOS/Myriad",
                       platform="darwin", env={}, machine="arm64", writable=lambda p: True)
    assert not k.can_apply and "Applications" in k.reason

    appdir = tmp_path / "mnt" / "AppDir"
    (appdir / "usr" / "lib" / "myriad").mkdir(parents=True)
    image = tmp_path / "Myriad-0.1.0-x86_64.AppImage"
    image.write_bytes(b"old")
    env = {"APPIMAGE": str(image), "APPDIR": str(appdir)}
    k = detect_install(frozen=True, executable=str(appdir / "usr" / "lib" / "myriad" / "Myriad"), platform="linux",
                       env=env, machine="x86_64", writable=lambda p: True)
    assert k.kind == "linux-appimage" and k.can_apply and k.asset("0.2.0") == "Myriad-0.2.0-x86_64.AppImage"
    # APPIMAGE inherited from another AppImage: this Myriad is not inside its mount point
    k = detect_install(frozen=True, executable=str(tmp_path / "Myriad" / "Myriad"), platform="linux", env=env,
                       machine="x86_64", writable=lambda p: True)
    assert k.kind == "linux-archive" and not k.can_apply
    k = detect_install(frozen=True, executable="/opt/myriad/Myriad", platform="linux", env={}, machine="x86_64")
    assert k.kind == "linux-deb" and not k.can_apply and k.asset("0.2.0") is None
    k = detect_install(frozen=False)
    assert k.kind == "source" and not k.can_apply
    with pytest.raises(UpdateError):
        InstallKind("windows-installer", can_apply=True).asset("0.2.0/../../x")


# ---------------------------------------------------------------------------------------------------
# downloads: pinned URLs, verified digests

PAYLOAD = b"MZ fake installer " * 1000


class FakeNet:
    """Mock transport: the tracker (127.0.0.1:8500), GitHub release files and the GitHub API."""

    def __init__(self, version="0.2.0", payload=PAYLOAD, sums_digest=None, tracker_body=None, github_latest=None):
        self.version, self.payload = version, payload
        self.sums_digest = sums_digest or hashlib.sha256(payload).hexdigest()
        self.tracker_body = tracker_body
        self.github_latest = github_latest
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        v = self.version
        if url.endswith("/v1/version") and "127.0.0.1" in url:
            if self.tracker_body is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json=self.tracker_body)
        if url == f"https://api.github.com/repos/{REPO}/releases/latest":
            if self.github_latest is None:
                return httpx.Response(404)
            return httpx.Response(200, json={"tag_name": f"v{self.github_latest}", "assets": []})
        if url == f"https://github.com/{REPO}/releases/download/v{v}/SHA256SUMS.txt":
            return httpx.Response(200, text=f"{self.sums_digest}  Myriad-Setup-{v}.exe\n{'0' * 64}  other.zip\n")
        if url == f"https://github.com/{REPO}/releases/download/v{v}/Myriad-Setup-{v}.exe":
            return httpx.Response(200, content=self.payload)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), follow_redirects=True)


def win_kind(tmp_path) -> InstallKind:
    return InstallKind("windows-installer", str(tmp_path / "app"), True, arch="x64")


async def settle(up: Updater) -> None:
    if up._dl_task is not None:
        await asyncio.gather(up._dl_task, return_exceptions=True)


async def test_download_urls_come_from_the_pinned_repo_whatever_the_tracker_says(tmp_path):
    hostile = {"repo": REPO, "latest": {
        "version": "0.2.0", "tag": "v0.2.0/../../evil",
        "assets": {"Myriad-Setup-0.2.0.exe": {"url": "https://evil.example/payload.exe", "size": 1}},
        "sha256": {"Myriad-Setup-0.2.0.exe": hashlib.sha256(b"evil").hexdigest()},
        "html_url": "https://evil.example/notes"}}
    net = FakeNet(tracker_body=hostile)
    up = Updater(tmp_path / "home", tracker_url="http://127.0.0.1:8500", current="0.1.0", kind=win_kind(tmp_path),
                 http=net.client())
    try:
        st = await up.check(force=True)
        assert st["available"] and st["latest"] == "0.2.0" and st["source"] == "tracker"
        await settle(up)
        st = up.status()
        assert st["state"] == "ready", st
        assert up.file.read_bytes() == PAYLOAD
        assert st["notes_url"] == f"https://github.com/{REPO}/releases/tag/v0.2.0"
        downloads = [u for u in net.requests if "127.0.0.1" not in u]
        assert downloads and all(u.startswith(f"https://github.com/{REPO}/releases/download/v0.2.0/") for u in downloads)
        assert not any("evil" in u for u in net.requests)
        # malformed or hostile version hints are ignored, never turned into URLs
        for bad in ("../../etc", "0.3.0/../../x", "v0.3.0", "0.3.0-rc1", 3, None, "0.2.0 ", "1" * 50):
            assert up.offer(bad, "push") is False
        assert up.offer("0.1.5", "push") is False  # older than the known latest: no downgrade
        assert up.latest == "0.2.0"
    finally:
        await up.close()


async def test_sha_mismatch_is_rejected_and_the_file_deleted(tmp_path):
    net = FakeNet(sums_digest="f" * 64, tracker_body={"repo": REPO, "latest": {"version": "0.2.0"}})
    up = Updater(tmp_path / "home", tracker_url="http://127.0.0.1:8500", current="0.1.0", kind=win_kind(tmp_path),
                 http=net.client())
    try:
        await up.check(force=True)
        await settle(up)
        st = up.status()
        assert st["state"] == "error" and "SHA-256" in st["error"]
        assert up.file is None and st["pending"] is None
        assert not any(p.name.startswith("Myriad-Setup") for p in (tmp_path / "home" / "updates").iterdir())
        with pytest.raises(UpdateError):
            up.request_install()
        # a failed version is not retried automatically at once (no download loop)
        assert up.start_download() is False
    finally:
        await up.close()


async def test_github_is_only_a_fallback(tmp_path):
    # an older tracker (no /v1/version): GitHub is asked
    net = FakeNet(tracker_body=None, github_latest="0.3.0")
    up = Updater(tmp_path / "h", tracker_url="http://127.0.0.1:8500", current="0.1.0",
                 kind=InstallKind("source"), http=net.client())
    try:
        st = await up.check(force=True)
        assert st["latest"] == "0.3.0" and st["source"] == "github" and not st["can_apply"]
        assert any("api.github.com" in u for u in net.requests)
    finally:
        await up.close()
    # a tracker that knows: GitHub is never asked
    net = FakeNet(tracker_body={"repo": REPO, "latest": None}, github_latest="0.3.0")
    up = Updater(tmp_path / "h2", tracker_url="http://127.0.0.1:8500", current="0.1.0",
                 kind=InstallKind("source"), http=net.client())
    try:
        st = await up.check(force=True)
        assert st["state"] == "up_to_date" and not st["available"]
        assert not any("api.github.com" in u for u in net.requests)
    finally:
        await up.close()


def test_signature_hook():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    sums = b"abc  file\n"
    updater.verify_sums_signature(sums, None, key_hex="")  # no pinned key (today): nothing to check
    updater.verify_sums_signature(sums, sk.sign(sums).hex().encode(), key_hex=pk)
    for sig in (None, b"00" * 64, sk.sign(b"other").hex().encode(), b"zz"):
        with pytest.raises(UpdateError):
            updater.verify_sums_signature(sums, sig, key_hex=pk)


# ---------------------------------------------------------------------------------------------------
# applying

def test_apply_windows_runs_the_installer_silently(tmp_path):
    calls = []
    updater.apply_windows(tmp_path / "Myriad-Setup-0.2.0.exe", relaunch=True, run=calls.append)
    updater.apply_windows(tmp_path / "Myriad-Setup-0.2.0.exe", relaunch=False, run=calls.append)
    assert calls[0][1:] == ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/MYRIADRELAUNCH=1"]
    assert calls[1][1:] == ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"]
    # the relaunched app gets the same data directory back (--home), e.g. a custom one
    home = tmp_path / "data dir" / "myriad"
    updater.apply_windows(tmp_path / "s.exe", relaunch=True, home=home, run=calls.append)
    assert calls[2][-1] == f"/MYRIADHOME={home}"
    updater.apply_windows(tmp_path / "s.exe", relaunch=False, home=home, run=calls.append)
    assert not any(a.startswith("/MYRIADHOME") for a in calls[3])
    iss = (Path(__file__).parents[1] / "packaging" / "windows" / "myriad.iss").read_text(encoding="utf-8")
    assert "MYRIADRELAUNCH" in iss and "--after-update" in iss and "CloseApplications=yes" in iss
    assert "{param:MYRIADHOME|}" in iss and "--home" in iss


def test_cleanup_leftovers_spares_an_update_in_progress(tmp_path):
    bundle = tmp_path / "Myriad.app"
    bundle.mkdir()
    kind = InstallKind("macos-app", str(bundle), True, arch="arm64")
    fresh = tmp_path / ".Myriad.app.new-999999"  # being copied right now (young)
    fresh.mkdir()
    stale = tmp_path / ".Myriad.app.old-999998"
    stale.mkdir()
    old = 1_000_000_000
    os.utime(stale, (old, old))
    updater.cleanup_leftovers(kind)
    assert fresh.exists() and not stale.exists() and bundle.exists()


def test_apply_appimage_replaces_the_file_atomically(tmp_path):
    target = tmp_path / "Myriad-0.1.0-x86_64.AppImage"
    target.write_bytes(b"old")
    new = tmp_path / "updates" / "Myriad-0.2.0-x86_64.AppImage"
    new.parent.mkdir()
    new.write_bytes(b"new image")
    calls = []
    updater.apply_appimage(new, target, relaunch=True, home=tmp_path / "home", run=calls.append)
    assert target.read_bytes() == b"new image" and not list(tmp_path.glob(".*.new-*"))
    if os.name != "nt":
        assert os.access(target, os.X_OK)
    assert calls == [[str(target), "--after-update", "--home", str(tmp_path / "home")]]


def test_apply_pending_needs_a_request_and_an_intact_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(updater, "apply_windows", lambda f, relaunch, home: calls.append((f, relaunch, home)))
    up = Updater(tmp_path, current="0.1.0", kind=win_kind(tmp_path))
    f = tmp_path / "Myriad-Setup-0.2.0.exe"
    f.write_bytes(PAYLOAD)
    up.latest, up.state, up.file, up.ready_version = "0.2.0", "ready", f, "0.2.0"
    up.digest = hashlib.sha256(PAYLOAD).hexdigest()
    assert up.pending_install() is None and up.apply_pending() is False  # nobody asked
    up.install_on_quit = True
    assert up.pending_install() == "quit"
    with pytest.raises(UpdateError):
        up.request_install()  # no app to restart (on_install_request unset)
    stopped = []
    up.on_install_request = lambda: stopped.append(True)
    up.request_install()
    assert stopped and up.pending_install() == "restart"
    f.write_bytes(PAYLOAD + b"tampered")
    assert up.apply_pending() is False and up.state == "error" and not calls and not f.exists()
    f.write_bytes(PAYLOAD)
    up.state = "ready"
    assert up.apply_pending() is True and calls == [(f, True, tmp_path)]


def test_desktop_installs_only_after_the_service_stopped(tmp_path):
    from myriad import desktop

    class Thread:
        alive = True

        def is_alive(self):
            return self.alive

    class Upd:
        n = 0

        def apply_pending(self):
            self.n += 1
            return True

    class Svc:
        thread = Thread()
        app = type("A", (), {"updater": Upd()})()

    svc = Svc()
    assert desktop.finish_update(svc) is False and svc.app.updater.n == 0
    svc.thread.alive = False
    assert desktop.finish_update(svc) is True and svc.app.updater.n == 1
    lock = desktop.InstanceLock(tmp_path)
    assert lock.acquire()
    other = desktop.InstanceLock(tmp_path)
    assert desktop.acquire_after_update(other, wait_s=0.2, step=0.05) is False
    lock.release()
    assert desktop.acquire_after_update(other, wait_s=1.0, step=0.05) is True
    other.release()


def test_startup_cleanup_keeps_newer_downloads(tmp_path):
    up = Updater(tmp_path, current="0.2.0", kind=win_kind(tmp_path))
    d = tmp_path / "updates"
    d.mkdir()
    for name in ("Myriad-Setup-0.1.0.exe", "Myriad-Setup-0.2.0.exe", "Myriad-Setup-0.3.0.exe",
                 "Myriad-Setup-0.3.0.exe.part", "junk.txt"):
        (d / name).write_bytes(b"x")
    up.startup_cleanup()
    assert sorted(p.name for p in d.iterdir()) == ["Myriad-Setup-0.3.0.exe", "Myriad-Setup-0.3.0.exe.part"]


# ---------------------------------------------------------------------------------------------------
# tracker: release watcher, /v1/version, Welcome field, push frame

def github_api(versions: list[str], sums_ok: bool = True, draft: bool = False):
    """Mocked GitHub: the API answers with versions[-1] (mutable list), with ETag / 304 support."""
    calls = {"api": 0, "304": 0, "sums": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        v = versions[-1]
        if url == f"https://api.github.com/repos/{REPO}/releases/latest":
            calls["api"] += 1
            etag = f'"{v}"'
            if request.headers.get("if-none-match") == etag:
                calls["304"] += 1
                return httpx.Response(304)
            assets = [{"name": n, "size": 10, "browser_download_url": f"https://github.com/{REPO}/releases/download/v{v}/{n}"}
                      for n in (f"Myriad-Setup-{v}.exe", f"Myriad-{v}-arm64.dmg", "SHA256SUMS.txt")]
            return httpx.Response(200, headers={"ETag": etag}, json={
                "tag_name": f"v{v}", "draft": draft, "prerelease": False, "published_at": "2026-10-01T00:00:00Z",
                "html_url": f"https://github.com/{REPO}/releases/tag/v{v}", "assets": assets})
        if url == f"https://github.com/{REPO}/releases/download/v{v}/SHA256SUMS.txt":
            calls["sums"] += 1
            if not sums_ok:
                return httpx.Response(500)
            return httpx.Response(200, text=f"{'1' * 64}  Myriad-Setup-{v}.exe\n{'2' * 64}  Myriad-{v}-arm64.dmg\n")
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True), calls


async def test_release_watcher_and_version_endpoint(tmp_path):
    versions = ["0.2.0"]
    http, calls = github_api(versions)
    t = Tracker(db_path=tmp_path / "t.sqlite", release_repo=REPO, release_http=http, release_poll_s=3600)
    try:
        await t.releases.poll_once()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=t.app), base_url="http://t") as c:
            r = await c.get("/v1/version")
            assert r.status_code == 200 and "max-age" in r.headers["cache-control"]
            latest = r.json()["latest"]
            assert r.json()["repo"] == REPO and latest["version"] == "0.2.0" and latest["tag"] == "v0.2.0"
            assert latest["sha256"] == {"Myriad-Setup-0.2.0.exe": "1" * 64, "Myriad-0.2.0-arm64.dmg": "2" * 64}
            assert latest["assets"]["Myriad-Setup-0.2.0.exe"]["size"] == 10
            assert "update" in (await c.get("/v1/health")).json()["features"]
        await t.releases.poll_once()  # unchanged: a conditional request answered 304
        assert calls["304"] == 1 and calls["sums"] == 1
        versions.append("0.3.0")
        await t.releases.poll_once()
        assert t.latest_version() == "0.3.0"
    finally:
        await t.releases.stop()
        await http.aclose()
        t.ledger.close()
    # disabled (tests, private networks): /v1/version answers, with nothing
    t = Tracker(db_path=tmp_path / "t2.sqlite")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=t.app), base_url="http://t") as c:
            assert (await c.get("/v1/version")).json() == {"repo": None, "latest": None, "checked_at": None}
    finally:
        t.ledger.close()


async def test_release_watcher_errors_and_rate_limits(tmp_path):
    http, _ = github_api(["0.2.0"], draft=True)
    w = ReleaseWatcher(REPO, http=http)
    with pytest.raises(ValueError):
        await w.poll_once()  # a draft is never announced
    assert w.latest is None
    await http.aclose()

    def limited(request):
        return httpx.Response(403, headers={"x-ratelimit-reset": "9999999999", "x-ratelimit-remaining": "0"})
    w = ReleaseWatcher(REPO, http=httpx.AsyncClient(transport=httpx.MockTransport(limited)))
    with pytest.raises(release.RateLimited) as e:
        await w.poll_once()
    assert e.value.wait_s and e.value.wait_s > 3600
    await w.stop()
    with pytest.raises(ValueError):
        ReleaseWatcher("evil.example/../x")


class NoFeatureNode(NodeClient):
    """A node from before the "update" feature: it does not ask for it."""
    FEATURES = ()


async def test_handshake_push_and_old_nodes(tmp_path):
    versions = ["0.2.0"]
    http, _ = github_api(versions)
    s = await start_swarm(tmp_path, release_repo=REPO, release_http=http, release_poll_s=3600)
    try:
        await wait_until(lambda: s.tracker.latest_version() == "0.2.0")
        hints: list = []
        new = NodeClient(Identity.generate(), s.url, reconnect=False)
        new.on_update = lambda v, src: hints.append((v, src))
        old = NoFeatureNode(Identity.generate(), s.url, reconnect=False)
        old_hints: list = []
        old.on_update = lambda v, src: old_hints.append((v, src))
        seen: list = []
        old.add_observer(lambda d, f: seen.append(f) if d == "in" else None)
        for n in (new, old):
            s.tasks.append(asyncio.create_task(n.run()))
            await asyncio.wait_for(n.connected.wait(), 5)
            s.nodes[n.node_id] = n
        await wait_until(lambda: hints)
        assert hints == [("0.2.0", "welcome")]
        assert s.tracker.conns[new.node_id].features == frozenset({"update"})
        assert s.tracker.conns[old.node_id].features == frozenset()
        versions.append("0.3.0")
        await s.tracker.releases.poll_once()  # a new release appears: pushed to the nodes that asked
        await wait_until(lambda: len(hints) == 2)
        assert hints[1] == ("0.3.0", "push") and new.latest_version == "0.3.0"
        await asyncio.sleep(0.2)
        assert old_hints == [] and old.latest_version is None
        assert not any(isinstance(f, UpdateAvailable) for f in seen)
    finally:
        await s.close()
        await http.aclose()


def test_frames_stay_compatible_with_older_nodes():
    w = Welcome(node_id="a" * 32, balance=1.0)
    assert "latest_version" not in json.loads(dump_frame(w))  # an older node forbids unknown fields
    w2 = parse_frame(dump_frame(Welcome(node_id="a" * 32, balance=1.0, latest_version="0.2.0")))
    assert w2.latest_version == "0.2.0"
    assert parse_frame(dump_frame(UpdateAvailable(version="0.2.0"))).version == "0.2.0"
    with pytest.raises(ValueError):
        parse_frame(json.dumps({"t": "update", "version": "x" * 65}))
    assert requested_features("update,tags,zzz") == frozenset({"update"})
    assert requested_features(None) == frozenset() and requested_features("u" * 300) == frozenset()


async def test_node_with_a_bad_hint_still_connects(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        s.tracker.latest_version = lambda: "not-a-version"  # e.g. a hostile tracker
        up = Updater(tmp_path / "h", current="0.1.0", kind=InstallKind("source"))
        n = NodeClient(Identity.generate(), s.url, reconnect=False)
        n.on_update = up.offer
        s.tasks.append(asyncio.create_task(n.run()))
        await asyncio.wait_for(n.connected.wait(), 5)
        s.nodes["n"] = n
        assert n.latest_version == "not-a-version" and up.latest is None and up.status()["available"] is False
    finally:
        await s.close()


# ---------------------------------------------------------------------------------------------------
# UI and CLI

async def test_ui_update_endpoints(tmp_path):
    from myriad.ui import make_ui_app

    home = tmp_path / "home"
    home.mkdir()
    Config().save(home)
    up = Updater(home, current="0.1.0", kind=win_kind(tmp_path))
    restarts = []
    up.on_install_request = lambda: restarts.append(True)
    app = make_ui_app(config=Config(), home=home, token="tok", runtime=None, updater=up)
    hdr = {"X-Myriad-Token": "tok"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        st = (await c.get("/api/status")).json()["update"]
        assert st["current"] == "0.1.0" and not st["available"] and st["kind"] == "windows-installer"
        assert (await c.post("/api/update/apply")).status_code == 403  # no token
        r = await c.post("/api/update/apply", headers=hdr)
        assert r.status_code == 409 and not restarts  # nothing downloaded and verified yet
        r = await c.post("/api/update/settings", json={"auto_update": "yes"}, headers=hdr)
        assert r.status_code == 400
        r = await c.post("/api/update/settings", json={"auto_update": False, "install_on_quit": True}, headers=hdr)
        assert r.status_code == 200 and r.json()["install_on_quit"] is True
        saved = Config.load(home)
        assert saved.auto_update is False and saved.install_on_quit is True
        f = home / "updates" / "Myriad-Setup-0.2.0.exe"
        f.parent.mkdir()
        f.write_bytes(PAYLOAD)
        up.latest, up.state, up.file, up.digest = "0.2.0", "ready", f, hashlib.sha256(PAYLOAD).hexdigest()
        r = await c.post("/api/update/apply", headers=hdr)
        assert r.status_code == 200 and restarts == [True] and r.json()["update"]["pending"] == "restart"
    # a UI without an updater (tests, embedding) says so
    app = make_ui_app(config=Config(), home=home, token="tok", runtime=None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        assert (await c.get("/api/update")).status_code == 404
        assert (await c.get("/api/status")).json()["update"] is None


def test_cli_update_check(tmp_path, monkeypatch, capsys):
    from myriad import cli

    async def fake_tracker(self):
        return "99.0.0"
    monkeypatch.setattr(Updater, "_from_tracker", fake_tracker)
    assert cli.main(["--home", str(tmp_path), "update", "--check"]) == 0
    out = capsys.readouterr().out
    assert "99.0.0" in out and f"https://github.com/{REPO}/releases/tag/v99.0.0" in out

    async def none_tracker(self):
        return None
    monkeypatch.setattr(Updater, "_from_tracker", none_tracker)
    assert cli.main(["--home", str(tmp_path), "update"]) == 0
    assert "à jour" in capsys.readouterr().out
