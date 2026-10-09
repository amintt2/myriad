"""App updates, client side.

Where the latest version comes from (cheapest first):
1. the tracker: Welcome.latest_version and UpdateAvailable frames (feature "update", node.py), and
   GET {tracker}/v1/version at startup and every CHECK_INTERVAL_S;
2. the GitHub API of the pinned repository, ONLY when the tracker is unreachable or does not know
   (an older tracker without /v1/version, or one that watches no repository).

Security: the tracker is NOT trusted for what to download. Its answer is reduced to a version string,
validated by a strict regular expression (release.VERSION_RE) and accepted only if strictly newer than
the running version. The client then builds every URL itself from the PINNED repository
(`https://github.com/amintt2/myriad/releases/download/v{version}/{asset}`), picks the asset of its own
platform, downloads it over HTTPS into `<data dir>/updates/` (resumable: downloader.py) and checks its
SHA-256 against SHA256SUMS.txt fetched from the same GitHub release, never from the tracker. A mismatch
deletes the file. TODO(signature): releases are not signed yet; `SUMS_SIGNING_KEY` is the hook for a
detached ed25519 signature of SHA256SUMS.txt (see verify_sums_signature).

Applying (only after the user asked, or on quit with `install_on_quit`, and after the node, its
llama-server and the gateway were stopped):
- Windows, installed build (Inno Setup, unins000.exe next to Myriad.exe): the installer runs silently
  (/SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS) and relaunches Myriad when asked to;
- macOS .app bundle: the disk image is mounted, the new bundle copied next to the running one (ditto)
  and swapped with it, the image detached;
- Linux AppImage: the file at $APPIMAGE is replaced atomically;
- anything else (portable zip, .deb, tar.gz, sources or pip): a notification with a link, no
  self-modification.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from . import __version__
from .downloader import DownloadError, Progress, download, sha256_file
from .release import (DEFAULT_RELEASE_REPO, MAX_API_BYTES, MAX_SUMS_BYTES, SUMS_NAME, get_capped, is_newer,
                      parse_release, parse_sums, parse_version)

log = logging.getLogger("myriad.updater")

# The ONLY repository updates are downloaded from. Never taken from the tracker or the network.
PINNED_REPO = DEFAULT_RELEASE_REPO
GITHUB = "https://github.com"
GITHUB_API = "https://api.github.com"
CHECK_INTERVAL_S = 6 * 3600.0
FIRST_CHECK_DELAY_S = 45.0  # leave the node time to connect: its Welcome frame usually answers first
MIN_CHECK_GAP_S = 60.0  # a forced check (tray, UI button) is not repeated faster than this
RETRY_FAILED_S = 3600.0  # a version whose download failed is not tried again before this
NO_CHECK_ENV = "MYRIAD_UPDATE_CHECK"  # "0": no periodic network check (tests, managed machines)
USER_AGENT = f"Myriad/{__version__} updater"
INNO_ARGS = ("/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS")
INNO_RELAUNCH = "/MYRIADRELAUNCH=1"  # read by packaging/windows/myriad.iss ([Run] entry with a Check)
INNO_HOME = "/MYRIADHOME="  # data directory of the instance to relaunch (myriad.iss: --home)
LEFTOVER_MIN_AGE_S = 3600.0  # swap leftovers younger than this may belong to an update in progress

# TODO(signature): hex of the ed25519 public key that signs SHA256SUMS.txt (release.yml would publish
# SHA256SUMS.txt.sig, the hex signature of the file's bytes). None: releases are unsigned and only the
# SHA-256 check against the release's own SHA256SUMS.txt applies. Never invent a key here.
SUMS_SIGNING_KEY: str | None = None
SUMS_SIGNATURE_NAME = "SHA256SUMS.txt.sig"


class UpdateError(RuntimeError):
    pass


_FILE_VERSION = re.compile(r"^Myriad-(?:Setup-)?([0-9]+\.[0-9]+\.[0-9]+)[-.]")


# ---------------------------------------------------------------------------------------------------
# URLs: always from the pinned repository and a validated version

def release_page(version: str, repo: str = PINNED_REPO) -> str:
    _check_version(version)
    return f"{GITHUB}/{repo}/releases/tag/v{version}"


def asset_url(version: str, asset: str, repo: str = PINNED_REPO) -> str:
    _check_version(version)
    if "/" in asset or "\\" in asset or asset.startswith("."):
        raise UpdateError(f"nom de fichier invalide : {asset!r}")
    return f"{GITHUB}/{repo}/releases/download/v{version}/{asset}"


def _check_version(version) -> None:
    if parse_version(version) is None:
        raise UpdateError(f"version invalide : {version!r}")


def verify_sums_signature(sums: bytes, signature: bytes | None, key_hex: str | None = None) -> None:
    """Hook for a detached ed25519 signature of SHA256SUMS.txt. Without a pinned key (today), nothing is
    checked. With one, a missing or bad signature raises UpdateError."""
    key_hex = SUMS_SIGNING_KEY if key_hex is None else key_hex
    if not key_hex:
        return
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not signature:
        raise UpdateError("signature de SHA256SUMS.txt manquante")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex)).verify(bytes.fromhex(signature.decode().strip()), sums)
    except (InvalidSignature, ValueError, UnicodeDecodeError) as e:
        raise UpdateError("signature de SHA256SUMS.txt invalide") from e


# ---------------------------------------------------------------------------------------------------
# Install kind

@dataclass(frozen=True)
class InstallKind:
    kind: str  # windows-installer, windows-portable, macos-app, linux-appimage, linux-deb, linux-archive, source
    target: str | None = None  # what an update replaces: app folder, .app bundle, AppImage file
    can_apply: bool = False
    reason: str | None = None  # why the app cannot update itself (shown with the link)
    arch: str = "x86_64"

    def asset(self, version: str) -> str | None:
        """Name of the release file this install updates from (None: notification only)."""
        _check_version(version)
        if self.kind == "windows-installer":
            return f"Myriad-Setup-{version}.exe"
        if self.kind == "macos-app":
            return f"Myriad-{version}-{self.arch}.dmg"
        if self.kind == "linux-appimage":
            return f"Myriad-{version}-{self.arch}.AppImage"
        return None


def _mac_arch(machine: str) -> str:
    """arm64 on Apple Silicon, also for an Intel build running under Rosetta (it moves to the native one)."""
    if machine == "arm64":
        return "arm64"
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5)
        if out.stdout.strip() == "1":
            return "arm64"
    except (OSError, subprocess.SubprocessError):
        pass
    return "x86_64"


def detect_install(frozen: bool | None = None, executable: str | None = None, platform: str | None = None,
                   env: dict | None = None, machine: str | None = None,
                   writable: Callable[[Path], bool] | None = None) -> InstallKind:
    """How this copy of Myriad was installed, hence how (and whether) it can update itself."""
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    executable = executable or sys.executable
    platform = platform or sys.platform
    env = os.environ if env is None else env
    machine = machine or _platform.machine()
    writable = writable or (lambda p: os.access(p, os.W_OK))
    if not frozen:
        return InstallKind("source", reason="sources ou pip : mettre à jour avec git pull / pip install -U")
    exe = Path(executable).resolve()
    if platform == "win32":
        app_dir = exe.parent
        if (app_dir / "unins000.exe").is_file():
            return InstallKind("windows-installer", str(app_dir), True, arch="x64")
        return InstallKind("windows-portable", str(app_dir), reason="version portable : télécharger la nouvelle archive")
    if platform == "darwin":
        arch = _mac_arch(machine)
        parts = exe.parts
        if len(parts) >= 4 and parts[-2] == "MacOS" and parts[-3] == "Contents" and parts[-4].endswith(".app"):
            bundle = exe.parents[2]
            # Gatekeeper's randomized read-only copy, or still on the disk image: nothing to replace.
            if "AppTranslocation" in bundle.parts or bundle.parts[1:2] == ("Volumes",):
                return InstallKind("macos-app", str(bundle), arch=arch,
                                   reason="déplacer Myriad dans le dossier Applications pour les mises à jour automatiques")
            if not writable(bundle.parent):
                return InstallKind("macos-app", str(bundle), arch=arch,
                                   reason=f"{bundle.parent} n'est pas modifiable par cet utilisateur")
            return InstallKind("macos-app", str(bundle), True, arch=arch)
        return InstallKind("source", reason="exécutable hors d'un paquet .app")
    # Linux and other POSIX systems
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine.lower(), machine)
    appimage, appdir = env.get("APPIMAGE"), env.get("APPDIR")
    if appimage and appdir:
        try:
            inside = exe.is_relative_to(Path(appdir).resolve())
        except OSError:
            inside = False
        image = Path(appimage)
        if inside and image.is_file():
            if not writable(image.parent):
                return InstallKind("linux-appimage", str(image), arch=arch,
                                   reason=f"{image.parent} n'est pas modifiable par cet utilisateur")
            return InstallKind("linux-appimage", str(image), True, arch=arch)
    if exe.parts[1:3] == ("opt", "myriad"):  # the .deb installs to /opt/myriad (/usr/bin/myriad links to it)
        return InstallKind("linux-deb", "/opt/myriad", arch=arch,
                           reason="paquet .deb : installer le nouveau paquet (sudo apt install ./myriad_….deb)")
    return InstallKind("linux-archive", str(exe.parent), arch=arch, reason="archive : télécharger la nouvelle archive")


# ---------------------------------------------------------------------------------------------------
# Applying (synchronous: called once the app has stopped its node, llama-server and gateway)

def child_env() -> dict:
    """Environment for a process started by the frozen app: without PyInstaller's own variables (a new
    Myriad must not believe it is a child of this one) and with the user's library path restored."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("_PYI_", "_MEIPASS"))}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    if sys.platform.startswith("linux"):
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env.pop("LD_LIBRARY_PATH_ORIG")
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _relaunch_args(home: Path | None) -> list[str]:
    return ["--after-update"] + (["--home", str(home)] if home is not None else [])


def _detached(cmd: list[str]) -> None:
    kw: dict = {"env": child_env(), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, **kw)  # noqa: S603 - fixed argument list, no shell


def apply_windows(installer: Path, relaunch: bool, home: Path | None = None, run=_detached) -> None:
    """The silent installer; with `relaunch`, it starts Myriad again on the same data directory
    (myriad.iss reads /MYRIADHOME= and passes it back as --home)."""
    args = [str(installer), *INNO_ARGS]
    if relaunch:
        args.append(INNO_RELAUNCH)
        h = str(home).rstrip("\\/") if home is not None else ""
        if h and '"' not in h and not h.endswith(":"):  # a drive root would end the quoted --home badly
            args.append(f"{INNO_HOME}{h}")
    run(args)


def apply_macos(dmg: Path, bundle: Path, relaunch: bool, home: Path | None = None, sh=subprocess.run,
                run=_detached) -> None:
    mnt = Path(tempfile.mkdtemp(prefix="myriad-update-"))
    new = bundle.with_name(f".{bundle.name}.new-{os.getpid()}")
    old = bundle.with_name(f".{bundle.name}.old-{os.getpid()}")
    sh(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mnt), str(dmg)],
       check=True, capture_output=True, timeout=180)
    try:
        src = mnt / bundle.name if (mnt / bundle.name).is_dir() else mnt / "Myriad.app"
        if not src.is_dir():
            raise UpdateError("Myriad.app introuvable dans l'image disque")
        shutil.rmtree(new, ignore_errors=True)
        sh(["ditto", str(src), str(new)], check=True, capture_output=True, timeout=900)
    finally:
        try:
            sh(["hdiutil", "detach", str(mnt), "-force"], check=False, capture_output=True, timeout=60)
        finally:
            shutil.rmtree(mnt, ignore_errors=True)
    # Swap: the running process keeps its open files; the old bundle is removed at the next start
    # (cleanup_leftovers), not now, because this process may still load modules from it.
    os.rename(bundle, old)
    try:
        os.rename(new, bundle)
    except OSError:
        os.rename(old, bundle)
        raise
    if relaunch:
        run(["open", "-n", str(bundle), "--args", *_relaunch_args(home)])


def apply_appimage(new_image: Path, target: Path, relaunch: bool, home: Path | None = None, run=_detached) -> None:
    tmp = target.with_name(f".{target.name}.new-{os.getpid()}")
    shutil.copyfile(new_image, tmp)
    os.chmod(tmp, 0o755)
    os.replace(tmp, target)  # atomic on the same file system; the running image keeps its inode
    if relaunch:
        run([str(target), *_relaunch_args(home)])


def _pid_alive(pid: str) -> bool:
    """Is the process that named a swap file still running? (POSIX; the swaps are macOS / Linux only)"""
    if not pid.isdigit() or os.name == "nt":
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except OSError:  # e.g. EPERM: it exists
        return True
    return True


def cleanup_leftovers(kind: InstallKind, min_age_s: float = LEFTOVER_MIN_AGE_S) -> None:
    """Remove what a previous update left next to the app (old bundle, half-copied files). Only entries
    older than `min_age_s`: an instance on ANOTHER data directory (not held by the same lock) must not
    remove files that an update in progress is copying right now."""
    if not kind.target or kind.kind not in ("macos-app", "linux-appimage"):
        return
    target = Path(kind.target)
    now = time.time()
    try:
        for p in target.parent.glob(f".{target.name}.*-*"):
            try:
                if now - p.lstat().st_mtime < min_age_s or _pid_alive(p.name.rsplit("-", 1)[-1]):
                    continue  # (ditto may keep the source's dates: the owner's PID is checked too)
            except OSError:
                continue
            if p.name.startswith((f".{target.name}.old-", f".{target.name}.new-")):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    except OSError as e:
        log.debug("cleanup of update leftovers: %s", e)


# ---------------------------------------------------------------------------------------------------
# The updater

class Updater:
    """Learns the latest version, downloads and verifies it in the background, applies it on request.

    `offer()` is called with every hint (tracker frames, checks); `check()` asks the tracker, then GitHub
    as a fallback; `request_install()` (UI button, tray) asks the app to stop and install; the desktop
    app calls `apply_pending()` once everything is stopped."""

    def __init__(self, home: Path | None, tracker_url: Callable[[], str] | str | None = None,
                 current: str = __version__, kind: InstallKind | None = None, auto_update: bool = True,
                 install_on_quit: bool = False, http: httpx.AsyncClient | None = None, repo: str = PINNED_REPO,
                 check_s: float = CHECK_INTERVAL_S, first_check_s: float = FIRST_CHECK_DELAY_S):
        self.home = Path(home) if home is not None else None
        self._tracker_url = tracker_url
        self.current = current
        self.kind = kind or detect_install()
        self.auto_update, self.install_on_quit = bool(auto_update), bool(install_on_quit)
        self.repo = repo
        self.check_s, self.first_check_s = check_s, first_check_s
        self._http, self._own_http = http, http is None
        self.latest: str | None = None
        self.source: str | None = None
        self.state = "idle"  # idle, checking, up_to_date, available, downloading, ready, installing, error
        self.error: str | None = None
        self.progress: Progress | None = None
        self.file: Path | None = None
        self.digest: str | None = None
        self.ready_version: str | None = None
        self.checked_at: float | None = None
        self.install_request: str | None = None  # "restart" once the user asked
        self.on_install_request: Callable[[], None] | None = None
        self._dl_task: asyncio.Task | None = None
        self._dl_version: str | None = None
        self._cancel: asyncio.Event | None = None
        self._failed: dict[str, float] = {}
        self._loop_task: asyncio.Task | None = None
        self._last_check = -1e9

    # ---------- helpers ----------
    @property
    def updates_dir(self) -> Path | None:
        return self.home / "updates" if self.home is not None else None

    def tracker_url(self) -> str | None:
        u = self._tracker_url() if callable(self._tracker_url) else self._tracker_url
        return u or None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(20.0, read=60.0),
                                           headers={"User-Agent": USER_AGENT})
        return self._http

    def available(self) -> bool:
        return self.latest is not None and is_newer(self.latest, self.current)

    def status(self) -> dict:
        asset = self.kind.asset(self.latest) if self.latest else None
        return {"current": self.current, "latest": self.latest, "available": self.available(),
                "state": self.state, "error": self.error, "source": self.source,
                "progress": self.progress.to_dict() if self.progress is not None else None,
                "kind": self.kind.kind, "can_apply": bool(self.kind.can_apply and asset),
                "reason": self.kind.reason, "asset": asset,
                "notes_url": release_page(self.latest, self.repo) if self.latest else None,
                "auto_update": self.auto_update, "install_on_quit": self.install_on_quit,
                "checked_at": self.checked_at, "pending": self.pending_install()}

    # ---------- learning about versions ----------
    def offer(self, version, source: str = "tracker") -> bool:
        """A hint that `version` exists. Accepted only if valid and strictly newer than both the running
        version and the latest one known (no downgrade). Starts the background download when allowed."""
        if not isinstance(version, str) or parse_version(version) is None:
            log.info("ignored malformed version hint from %s", source)
            return False
        if not is_newer(version, self.current) or (self.latest and not is_newer(version, self.latest)):
            return False
        self.latest, self.source = version, source
        # Strictly newer than anything known: a download or a ready file is of an older version.
        self.state, self.error = "available", None
        log.info("Myriad %s is available (from %s)", version, source)
        if self.auto_update:
            self.start_download()
        return True

    async def _from_tracker(self) -> str | None:
        """The tracker's latest version (None: it knows none). Raises LookupError when it cannot tell
        (unreachable, older tracker, or no repository watched): the caller falls back to GitHub."""
        from .node import http_url
        base = self.tracker_url()
        if not base:
            raise LookupError("aucun traqueur")
        try:
            r = await get_capped(self._client(), f"{http_url(base)}/v1/version", 64 << 10)
            body = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError) as e:
            raise LookupError(f"traqueur injoignable : {e}") from e
        if not isinstance(body, dict) or body.get("repo") != self.repo:
            raise LookupError("le traqueur ne suit pas les versions de ce dépôt")
        latest = body.get("latest")
        return latest.get("version") if isinstance(latest, dict) else None

    async def _from_github(self) -> str | None:
        r = await get_capped(self._client(), f"{GITHUB_API}/repos/{self.repo}/releases/latest", MAX_API_BYTES,
                             {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        info = parse_release(r.json())
        return info["version"] if info else None

    async def check(self, force: bool = False) -> dict:
        """Ask the tracker (GitHub only if the tracker cannot tell) and offer what it says."""
        now = time.monotonic()
        if now - self._last_check < (5.0 if force else MIN_CHECK_GAP_S):  # a burst of clicks
            return self.status()
        self._last_check = now
        busy = self.state in ("downloading", "ready", "installing")
        if not busy:
            self.state = "checking"
        try:
            try:
                version, source = await self._from_tracker(), "tracker"
            except LookupError as e:
                log.info("version from the tracker unavailable (%s): asking GitHub", e)
                version, source = await self._from_github(), "github"
        except (httpx.HTTPError, ValueError) as e:
            if not busy:  # a version already known stays offered; else the failure is shown
                if self.available():
                    self.state = "available"
                else:
                    self.state, self.error = "error", f"vérification impossible : {e}"[:300]
            return self.status()
        self.checked_at = time.time()
        if version is not None:
            self.offer(version, source)
        if self.state == "checking":
            failed = self.latest is not None and self._recently_failed(self.latest)
            self.state = "available" if self.available() else "up_to_date"
            if failed:
                self.state = "error"  # keep the download error visible until the retry delay is over
            else:
                self.error = None
        if self.available() and self.auto_update and self.state == "available":
            self.start_download()  # e.g. a download that failed long ago: try again
        return self.status()

    def _recently_failed(self, version: str) -> bool:
        return time.monotonic() - self._failed.get(version, -1e9) < RETRY_FAILED_S

    # ---------- downloading ----------
    def start_download(self, force: bool = False) -> bool:
        """Start downloading the latest version in the background (False: nothing to do now)."""
        v = self.latest
        if not v or not self.available() or self.updates_dir is None or self.kind.asset(v) is None:
            return False
        if self.ready_version == v and self.file is not None and self.file.exists():
            return False
        if self._dl_task is not None and not self._dl_task.done():
            if self._dl_version == v:
                return False
            if self._cancel is not None:
                self._cancel.set()  # a newer version replaces the one being downloaded
            self._dl_task.cancel()
        if not force and self._recently_failed(v):
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._dl_version = v
        self._dl_task = loop.create_task(self._download(v), name="update-download")
        return True

    async def _fetch_sums(self, version: str) -> dict[str, str]:
        http = self._client()
        r = await get_capped(http, asset_url(version, SUMS_NAME, self.repo), MAX_SUMS_BYTES)
        if r.status_code != 200:
            raise UpdateError(f"{SUMS_NAME} introuvable pour la version {version} (HTTP {r.status_code})")
        sig = None
        if SUMS_SIGNING_KEY:
            s = await get_capped(http, asset_url(version, SUMS_SIGNATURE_NAME, self.repo), 4096)
            sig = s.content if s.status_code == 200 else None
        verify_sums_signature(r.content, sig)
        return parse_sums(r.text)

    def _clean_updates_dir(self, keep: str | None) -> None:
        """Remove every file of <data dir>/updates but `keep` (and its partial download). With keep=None
        (at start), files of a version newer than the running one are kept: no download twice."""
        d = self.updates_dir
        if d is None or not d.is_dir():
            return
        for p in d.iterdir():
            if keep and p.name in (keep, keep + ".part"):
                continue
            m = None if keep else _FILE_VERSION.search(p.name)
            if m and is_newer(m[1], self.current):
                continue
            try:
                p.unlink() if p.is_file() else shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass

    async def _download(self, version: str) -> Path | None:
        asset = self.kind.asset(version)
        self.state, self.error = "downloading", None
        self.progress = Progress(label=asset)
        self._cancel = asyncio.Event()
        try:
            sums = await self._fetch_sums(version)
            digest = sums.get(asset)
            if not digest:
                raise UpdateError(f"aucun fichier {asset} dans la version {version} (plateforme non publiée ?)")
            self.updates_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._clean_updates_dir, asset)
            dest = await download(asset_url(version, asset, self.repo), self.updates_dir / asset, sha256=digest,
                                  client=self._client(), progress=self.progress, cancel=self._cancel)
        except asyncio.CancelledError:
            if self.latest == version:
                self.state = "available"
            raise
        except (UpdateError, DownloadError, httpx.HTTPError, OSError, ValueError) as e:
            self._failed[version] = time.monotonic()
            if self.latest == version:
                self.state, self.error = "error", str(e)[:300]
            log.warning("update %s: %s", version, e)
            return None
        if self.latest != version:  # overtaken by a newer version meanwhile
            return None
        self.file, self.digest, self.ready_version = dest, digest, version
        self.state = "ready"
        log.info("update %s downloaded and verified: %s", version, dest)
        return dest

    async def wait_download(self) -> None:
        """Wait for the background download, if one runs (command line)."""
        if self._dl_task is not None:
            await asyncio.gather(self._dl_task, return_exceptions=True)

    async def download_now(self) -> dict:
        """UI button (automatic downloads off, or a retry after an error): download the latest version."""
        if not self.available():
            raise UpdateError("aucune nouvelle version à télécharger")
        if self.updates_dir is None or self.kind.asset(self.latest) is None:
            raise UpdateError(self.kind.reason or "pas de fichier de mise à jour pour cette installation")
        self.start_download(force=True)
        return self.status()

    # ---------- installing ----------
    def pending_install(self) -> str | None:
        """'restart' (the user asked), 'quit' (install_on_quit) or None."""
        if self.state != "ready" or not self.kind.can_apply or self.file is None:
            return None
        if self.install_request == "restart":
            return "restart"
        return "quit" if self.install_on_quit else None

    def request_install(self) -> None:
        """« Mettre à jour et redémarrer » : the app stops cleanly, then apply_pending() runs."""
        if not self.kind.can_apply:
            raise UpdateError(self.kind.reason or "cette installation ne se met pas à jour toute seule")
        if self.state != "ready" or self.file is None:
            raise UpdateError("la mise à jour n'est pas encore téléchargée et vérifiée")
        self.install_request = "restart"
        cb = self.on_install_request
        if cb is None:
            raise UpdateError("l'application ne peut pas redémarrer d'ici (lancez l'application de bureau)")
        cb()

    def apply_pending(self) -> bool:
        """Install the verified update if one is pending. Synchronous; call only after the node, its
        llama-server and the gateway have stopped. Returns True if the installer / swap was started."""
        mode = self.pending_install()
        if mode is None:
            return False
        if sha256_file(self.file) != self.digest:  # the file changed since it was verified
            self.state, self.error = "error", "fichier de mise à jour modifié depuis sa vérification"
            self.file.unlink(missing_ok=True)
            return False
        relaunch = mode == "restart"
        self.state = "installing"
        k = self.kind
        log.info("installing update %s (%s, relaunch=%s)", self.ready_version, k.kind, relaunch)
        if k.kind == "windows-installer":
            apply_windows(self.file, relaunch, self.home)
        elif k.kind == "macos-app":
            apply_macos(self.file, Path(k.target), relaunch, self.home)
        elif k.kind == "linux-appimage":
            apply_appimage(self.file, Path(k.target), relaunch, self.home)
            self.file.unlink(missing_ok=True)
        else:
            return False
        return True

    # ---------- lifecycle ----------
    def startup_cleanup(self) -> None:
        """At start: forget installers of versions not newer than this one, remove swap leftovers."""
        cleanup_leftovers(self.kind)
        self._clean_updates_dir(None)

    async def run(self) -> None:
        if os.environ.get(NO_CHECK_ENV, "").strip() == "0":
            return
        await asyncio.sleep(self.first_check_s)
        while True:
            try:
                await self.check(force=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # never let the updater stop the app
                log.warning("update check failed: %s", e)
            await asyncio.sleep(self.check_s)

    def start(self) -> None:
        try:
            self.startup_cleanup()
        except OSError as e:
            log.debug("update cleanup: %s", e)
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.get_running_loop().create_task(self.run(), name="update-check")

    async def close(self) -> None:
        for t in (self._loop_task, self._dl_task):
            if t is not None and not t.done():
                t.cancel()
        await asyncio.gather(*(t for t in (self._loop_task, self._dl_task) if t is not None), return_exceptions=True)
        self._loop_task = self._dl_task = None
        if self._own_http and self._http is not None:
            await self._http.aclose()
            self._http = None
