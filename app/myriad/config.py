"""Per-user configuration: directory, key file, config.json."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger("myriad.config")

APP = "myriad"  # name of the data directory
# Name of the data directory before the package was renamed (essaim -> myriad). On first start, an
# existing legacy directory is COPIED to the new one (never moved or deleted): see migrate_legacy_home.
LEGACY_APP = "essaim"
HOME_ENV = "MYRIAD_HOME"
LEGACY_HOME_ENV = "ESSAIM_HOME"  # still honoured (deprecated)
APP_NAME = "Myriad"  # product name shown to users
# The public tracker of the Myriad network (WebSocket: wss://myriad.french-web.com/v1/ws). This is the
# one place to change it: default of the configuration, of `myriad init` and of the desktop wizard.
# Until it is deployed, nodes show "tracker unreachable" and retry with backoff; the UI lets users change it.
PUBLIC_TRACKER_URL = "https://myriad.french-web.com"
DEFAULT_TRACKER = PUBLIC_TRACKER_URL
LOCAL_TRACKER = "http://127.0.0.1:8500"  # `myriad tracker` on this machine (tests, private networks)
DEFAULT_MODEL = "Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf"
# Complete model files (models/**/*.gguf) at least this large are hard-linked instead of copied when the
# legacy directory is migrated, where the file system allows it: no second copy of several gigabytes.
# They are never modified in place (a download writes `<file>.part`, then os.replace; a reinstall unlinks
# first), so sharing them is safe. Everything else is copied: partial downloads (`.part`, appended to or
# truncated when resumed), the engine (re-extracted in place), the configuration, the logs.
LINK_MIN_BYTES = 64 << 20
# Per-instance files of the desktop app (desktop.py): never copied by the migration.
_INSTANCE_FILES = ("myriad.lock", "myriad.show", "myriad.stop", "myriad.url")


def platform_dir(name: str) -> Path:
    """The platform's per-user config directory for `name` (same rules as platformdirs)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / name
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / name


def home_dir() -> Path:
    """MYRIAD_HOME (or the deprecated ESSAIM_HOME), else the platform's per-user directory `myriad`,
    created from a copy of the legacy `essaim` directory the first time (migrate_legacy_home)."""
    env = os.environ.get(HOME_ENV) or os.environ.get(LEGACY_HOME_ENV)
    if env:
        return Path(env).expanduser()
    return migrate_legacy_home(platform_dir(LEGACY_APP), platform_dir(APP))


def _lock_if_free(path: Path):
    """Lock an existing desktop instance lock file (the same lock as desktop.InstanceLock). Returns the
    open, locked file, or None if another process holds the lock."""
    f = open(path, "r+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _unlock(f) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    f.close()


def _is_complete_model(rel: Path) -> bool:
    return len(rel.parts) >= 2 and rel.parts[0] == "models" and rel.suffix.lower() == ".gguf"


def _copier(old: Path):
    """copytree's copy function: hard-link complete large models (see LINK_MIN_BYTES), copy the rest."""
    def copy(src: str, dst: str) -> str:
        if _is_complete_model(Path(src).relative_to(old)) and os.path.getsize(src) >= LINK_MIN_BYTES:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)
    return copy


def _rebase_config_paths(cfg_file: Path, old: Path, new: Path) -> None:
    """config.json stores absolute paths (model file, llama-server) that may point into the legacy
    directory: point them into the new one."""
    if not cfg_file.exists():
        return
    data = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("config.json n'est pas un objet JSON")
    changed = False
    for key in ("gguf_path", "llama_server"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            rel = Path(value).relative_to(old)
        except ValueError:
            continue
        data[key] = str(new / rel)
        changed = True
    if changed:
        cfg_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def migrate_legacy_home(old: Path, new: Path) -> Path:
    """First start after the rename: copy the legacy data directory `old` to `new` (key, configuration,
    models, engine, logs) and return the directory to use. The legacy directory is never modified or
    deleted. Nothing is done if `new` already exists or `old` does not. If a desktop instance of the
    old version holds `old`, or if the copy fails, `old` is returned and used as is: the migration is
    tried again at the next start."""
    if new.exists() or not old.is_dir():
        return new
    lock = None
    if (old / "myriad.lock").exists():
        try:
            lock = _lock_if_free(old / "myriad.lock")
        except OSError:
            lock = None
        if lock is None:
            log.warning("%s est utilisé par une instance en cours : migration vers %s remise à plus tard", old, new)
            return old
    tmp = new.with_name(f"{new.name}.migrating-{os.getpid()}")
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(old, tmp, ignore=shutil.ignore_patterns(*_INSTANCE_FILES), copy_function=_copier(old))
        _rebase_config_paths(tmp / "config.json", old, new)
        os.replace(tmp, new)
    except (OSError, ValueError) as e:  # ValueError: unreadable config.json
        shutil.rmtree(tmp, ignore_errors=True)  # only our partial copy, never `old`
        if new.exists():  # another process migrated at the same time
            return new
        log.warning("migration de %s vers %s impossible (%s) : %s reste utilisé", old, new, e, old)
        return old
    finally:
        if lock is not None:
            _unlock(lock)
    log.info("dossier de données copié de %s vers %s (l'ancien dossier est conservé)", old, new)
    return new


def key_path(home: Path | None = None) -> Path:
    return (home or home_dir()) / "node_key.pem"


def config_path(home: Path | None = None) -> Path:
    return (home or home_dir()) / "config.json"


def default_llama_server() -> str:
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    return os.environ.get("LLAMA_SERVER", exe)


@dataclass
class Config:
    tracker_url: str = DEFAULT_TRACKER
    model: str | None = None  # "repo:file.gguf"
    gguf_path: str | None = None  # local path of the downloaded GGUF
    family: str | None = None
    params_b: float | None = None
    ctx: int = 4096
    max_parallel: int = 1
    accepting: bool = True
    active_hours: str | None = None  # "8-23": accept jobs only from 08:00 to 22:59 (local time)
    n_gpu_layers: int = 999
    llama_server: str | None = None  # path of the llama-server binary (env LLAMA_SERVER otherwise)
    max_job_tokens: int = 1024
    gateway_port: int = 8400
    ui_port: int = 8401
    default_k: int = 4
    request_timeout_s: float = 120.0
    # Skill tags this node advertises (essaim/1.2), on top of its base model's capabilities: clients
    # reach it with the model name "myriad:<tag>" (e.g. "python", "review"). See routing.py.
    tags: list = field(default_factory=list)
    # Sub-agents (POST /v1/agents/run): the ONLY local commands a verification step may run, by name,
    # as argument lists (no shell), e.g. {"pytest": ["python", "-m", "pytest", "-q"]}. See agents.py.
    verify_commands: dict = field(default_factory=dict)
    # Protection settings (security.SecuritySettings): end-to-end encryption required, blocklist,
    # trusted nodes, private swarm, privacy guard, serving limits... Missing keys take their defaults.
    security: dict = field(default_factory=dict)
    # App updates (updater.py): download a new version in the background as soon as it is known (else
    # only tell), and install a downloaded update when the app quits (else only on « Mettre à jour et
    # redémarrer »). Checking costs nothing: the tracker announces new versions to every node.
    auto_update: bool = True
    install_on_quit: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def repo_id(self) -> str | None:
        return self.model.split(":", 1)[0] if self.model else None

    @property
    def gguf_file(self) -> str | None:
        return self.model.split(":", 1)[1] if self.model and ":" in self.model else None

    @classmethod
    def load(cls, home: Path | None = None) -> "Config":
        p = config_path(home)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8-sig"))  # tolerate a BOM (Windows editors)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, home: Path | None = None) -> Path:
        p = config_path(home)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
        return p


def parse_active_hours(spec: str | None) -> tuple[int, int] | None:
    """'8-23' -> (8, 23). Hours are 0..24; start > end wraps past midnight ('22-6')."""
    if not spec:
        return None
    a, b = spec.split("-", 1)
    start, end = int(a), int(b)
    if not (0 <= start <= 24 and 0 <= end <= 24):
        raise ValueError("heures hors de 0..24")
    return start, end


def within_hours(spec: str | None, hour: int) -> bool:
    rng = parse_active_hours(spec)
    if rng is None:
        return True
    start, end = rng
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
