"""E18: run the Lean checker on one file, with a wall-clock limit, a memory limit and bounded output.

One fresh `lean --json` process per file, in its own temporary directory and in the job's process group (stopping
the job stops its checks), with the Mathlib
search path computed once by e18_setup.py (`LEAN_PATH`, saved in <lean home>/e18_env.json). On Linux (the Colab
VM, where the experiment runs) the process also gets:
  * rlimits through `prlimit` (no `preexec_fn`: the caller is multi-threaded): no core file, written files <=
    OUTPUT_CAP (stdout included). No CPU-time rlimit: Lean elaborates with several threads; the wall-clock limit
    applies to the whole group;
  * no network when `unshare --net` works (probed once, as for E11's sandbox);
  * a memory watchdog: the anonymous resident memory of the process (RssAnon, so the Mathlib .olean files that
    Lean maps and shares with every other check are not counted) is read every 0.25 s and the whole group is
    killed above the limit. An address-space rlimit is not used: Lean maps the whole Mathlib library.
stdout goes to a file (bounded by RLIMIT_FSIZE) and is read back up to OUTPUT_CAP bytes.
The model-written text was screened before (essaim/lean.screen); these limits are the second line of defence, and
the VM is disposable.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

LINUX = os.name == "posix" and os.uname().sysname == "Linux"
OUTPUT_CAP = 8 << 20
POLL_S = 0.25


def lean_home() -> Path:
    return Path(os.environ.get("E18_LEAN_HOME", Path.home() / ".cache" / "e18-lean"))


def load_env(home: Path | None = None) -> dict:
    """The checker environment written by e18_setup.py: lean binary, LEAN_PATH, versions."""
    path = (home or lean_home()) / "e18_env.json"
    if not path.exists():
        raise SystemExit(f"{path} absent : lancer e18_setup.py d'abord")
    return json.loads(path.read_text(encoding="utf-8"))


_UNSHARE: list[str] | None = None
_UNSHARE_LOCK = threading.Lock()


def unshare_prefix() -> list[str]:
    global _UNSHARE
    with _UNSHARE_LOCK:
        if _UNSHARE is None:
            _UNSHARE = []
            exe = shutil.which("unshare")
            if LINUX and exe:
                for pre in ([exe, "--net", "--"], [exe, "--net", "--map-root-user", "--"]):
                    try:
                        if subprocess.run(pre + ["true"], capture_output=True, timeout=20).returncode == 0:
                            _UNSHARE = pre
                            break
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        return _UNSHARE


def children(pid: int, proc: str = "/proc") -> set[int]:
    """Children of every thread of `pid` (each /proc/<pid>/task/<tid>/children lists only that thread's children)."""
    out: set[int] = set()
    try:
        tids = os.listdir(f"{proc}/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"{proc}/{pid}/task/{tid}/children") as f:
                out.update(int(c) for c in f.read().split())
        except (OSError, ValueError):
            pass
    return out


def descendants(pid: int, proc: str = "/proc") -> list[int]:
    """`pid` and its descendants (Linux /proc), parents first, each once."""
    out, seen, todo = [], {pid}, [pid]
    while todo:
        p = todo.pop(0)
        out.append(p)
        for c in sorted(children(p, proc) - seen):
            seen.add(c)
            todo.append(c)
    return out


def rss_anon_mib(pid: int) -> float:
    """Anonymous resident memory of `pid` and its descendants in MiB (Linux; 0 if unknown)."""
    total = 0.0
    for p in descendants(pid):
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        total += int(line.split()[1]) / 1024
                        break
        except (OSError, ValueError):
            pass
    return total


class Checker:
    """Thread-safe: at most `workers` Lean processes at a time."""

    def __init__(self, env: dict | None = None, timeout_s: float = 150.0, mem_mib: int = 8192, workers: int = 4,
                 command: list[str] | None = None):
        self.env = env if env is not None else load_env()
        self.timeout_s, self.mem_mib = timeout_s, mem_mib
        self.command = command or [self.env["lean"], "--json"]
        self.slots = threading.BoundedSemaphore(workers)
        self.workers = workers
        self.isolation = {"network": "unshare --net" if unshare_prefix() else "none",
                          "memory": f"RssAnon watchdog {mem_mib} MiB" if LINUX else "none (not Linux)",
                          "wall_s": timeout_s, "rlimits": "core 0, fsize" if self._limits() else "none",
                          "output_cap": OUTPUT_CAP}

    def _env(self, tmp: str) -> dict:
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": tmp, "TMPDIR": tmp, "LANG": "C.UTF-8"}
        if self.env.get("LEAN_PATH"):
            env["LEAN_PATH"] = self.env["LEAN_PATH"]
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return env

    @staticmethod
    def _limits() -> list[str]:
        exe = shutil.which("prlimit") if LINUX else None
        return [exe, "--core=0", f"--fsize={OUTPUT_CAP}", "--"] if exe else []

    def check(self, text: str) -> dict:
        """Compile `text`; returns returncode, timeout, memory, overflow, stdout, stderr (tail), ms, peak_mib."""
        with self.slots:
            return self._run(text)

    def _run(self, text: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="e18-") as tmp:
            src = Path(tmp) / "Check.lean"
            src.write_text(text, encoding="utf-8")
            out_path, err_path = Path(tmp) / "out.json", Path(tmp) / "err.txt"
            t0 = time.perf_counter()
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                # Same process group as the job (no new session): when the launcher stops the job's group, the Lean
                # processes stop with it instead of outliving the watchdogs that bound them.
                kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
                p = subprocess.Popen(unshare_prefix() + self._limits() + self.command + [str(src)], cwd=tmp, env=self._env(tmp),
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kw)
                timeout = memory = False
                peak = 0.0
                deadline = time.monotonic() + self.timeout_s
                while p.poll() is None:
                    if LINUX:
                        peak = max(peak, rss_anon_mib(p.pid))
                        if peak > self.mem_mib:
                            memory = True
                            self._kill(p)
                            break
                    if time.monotonic() > deadline:
                        timeout = True
                        self._kill(p)
                        break
                    time.sleep(POLL_S)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._kill(p)
            ms = round((time.perf_counter() - t0) * 1000)
            raw = out_path.read_bytes()
            overflow = len(raw) >= OUTPUT_CAP or (LINUX and p.returncode == -signal.SIGXFSZ)
            return {"returncode": p.returncode, "timeout": timeout, "memory": memory, "overflow": overflow,
                    "stdout": raw[:OUTPUT_CAP].decode("utf-8", errors="replace"),
                    "stderr": err_path.read_bytes()[-2000:].decode("utf-8", errors="replace"), "ms": ms,
                    "peak_mib": round(peak)}

    @staticmethod
    def _kill(p: subprocess.Popen):
        """Kill the Lean process and its descendants (it shares the job's process group, so no killpg)."""
        if LINUX:
            for pid in reversed(descendants(p.pid)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        try:
            p.kill()
        except OSError:
            pass
