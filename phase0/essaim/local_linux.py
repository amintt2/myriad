"""Fail-closed shell environment: Landlock, seccomp and a disposable Linux PID/network namespace.

Unlike E11's Python audit hook, this boundary permits fork/exec for bash and pytest. No E11 rule is changed.
The trusted launcher installs the kernel rules before exec; only stdin/stdout/stderr survive exec.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import selectors
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    import resource

VERSION = "linux-shell-v2"
OUTPUT_CAP = 1 << 20


class IsolationError(RuntimeError):
    pass


def landlock(work: Path, runtime: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.syscall
    call.restype = ctypes.c_long
    abi = call(444, 0, 0, 1)
    if abi < 3:
        raise IsolationError("Landlock ABI >= 3 is required (including TRUNCATE)")
    rights = (1 << 15) - 1
    if abi >= 5:
        rights |= 1 << 15  # IOCTL_DEV
    read = (1 << 0) | (1 << 2) | (1 << 3)
    rules = [(work, rights & ~((1 << 6) | (1 << 11))), (runtime, read)]  # no device nodes
    python = (runtime / "bin/python").resolve()
    if python.parent != Path("/usr/bin"):
        prefix = python.parent.parent
        if prefix.parent.parts[-4:] != (".local", "share", "uv", "python") or not prefix.name.startswith("cpython-"):
            raise IsolationError("runtime Python must be system Python or a dedicated uv-managed installation")
        rules.append((prefix, read))
    rules += [(Path(p), read) for p in ("/usr/bin", "/usr/lib", "/lib", "/lib64") if Path(p).exists()]
    rules += [(Path(p), 1 << 2) for p in ("/etc/ld.so.cache", "/etc/localtime") if Path(p).exists()]
    rules += [(Path("/dev/null"), (1 << 1) | (1 << 2)), (Path("/dev/urandom"), 1 << 2)]
    attr = ctypes.create_string_buffer(struct.pack("=Q", rights))
    fd = call(444, attr, 8, 0)
    if fd < 0:
        raise IsolationError("Landlock create failed")
    try:
        for path, access in rules:
            if not path.is_dir():
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = ctypes.create_string_buffer(struct.pack("=Qi", access, pfd))
                if call(445, fd, 1, rule, 0):
                    raise IsolationError("Landlock add rule failed")
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) or call(446, fd, 0):
            raise IsolationError("Landlock restrict failed")
    finally:
        os.close(fd)
    return abi


def seccomp() -> None:
    """Deny sockets, namespace/mount escapes and cross-process introspection, also through native code."""
    if platform.machine() != "x86_64":
        raise IsolationError("seccomp syscall table requires x86_64")
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                    ("k", ctypes.c_uint32)]
    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]
    # Reject other syscall ABIs (including x32); permit ordinary shell fork/exec and local file operations.
    # Landlock does not cover metadata: ioctl can also mutate inode flags through a read-only runtime fd.
    denied = [16, 41, 42, 43, 49, 50, 53, 90, 91, 92, 93, 94, 101, 132, 133, 155, 161, 165, 166, 167, 168, 169, 175,
              176, 188, 189, 190, 197, 198, 199, 235, 246, 248, 249, 250, 259, 260, 261, 268, 272, 280, 298,
              303, 304, 308, 310, 311, 313, 321, 323, 425, 426, 427, 428, 429, 430, 431, 432, 442, 452, 463, 466, 469]
    filters = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, 0xc000003e), Filter(0x06, 0, 0, 0x80000000),
               Filter(0x20, 0, 0, 0), Filter(0x35, 0, 1, 0x40000000), Filter(0x06, 0, 0, 0x80000000)]
    # CPython uses FIOCLEX to mark its own descriptors close-on-exec; it changes no inode/device metadata.
    filters += [Filter(0x15, 0, 3, 16), Filter(0x20, 0, 0, 24), Filter(0x15, 0, 1, 0x5451),
                Filter(0x06, 0, 0, 0x7fff0000), Filter(0x20, 0, 0, 0)]
    for number in denied:
        filters += [Filter(0x15, 0, 1, number), Filter(0x06, 0, 0, 0x00050000 | 1)]  # EPERM
    filters += [Filter(0x06, 0, 0, 0x7fff0000)]
    array = (Filter * len(filters))(*filters)
    prog = Program(len(filters), array)
    if ctypes.CDLL(None, use_errno=True).prctl(22, 2, ctypes.byref(prog), 0, 0):
        raise IsolationError("seccomp install failed")


def launch() -> None:
    cfg = json.loads(sys.stdin.readline())
    work, runtime = Path(cfg["work"]), Path(cfg["runtime"])
    os.chdir(work)
    for limit, value in ((resource.RLIMIT_AS, 512 << 20), (resource.RLIMIT_CPU, cfg["cpu"]),
                         (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_NOFILE, 128),
                         (resource.RLIMIT_NPROC, 32), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(limit, (value, value))
    abi = landlock(work, runtime)
    seccomp()
    print(json.dumps({"isolation_ready": VERSION, "landlock_abi": abi, "seccomp": "on",
                      "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}), flush=True)
    os.execve("/usr/bin/bash", ["bash", "--noprofile", "--norc", "-c", cfg["cmd"]], {
        "PATH": f"{runtime}/bin:/usr/bin", "HOME": str(work), "TMPDIR": str(work), "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})


def enter(parent: int) -> None:
    """Keep the namespace supervisor tied to its controller, including abrupt controller death."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) or os.getppid() != parent:  # PR_SET_PDEATHSIG
        raise IsolationError("controller disappeared before namespace setup")
    script = str(Path(__file__).resolve())
    os.execve("/usr/bin/unshare", ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid",
              "--ipc", "--uts", "--fork", "--kill-child=KILL", "/usr/bin/python3", "-I", script, "--launch"],
              {"PATH": "/usr/bin:/bin"})


class LocalLinuxEnv:
    def __init__(self, work: Path, runtime: Path):
        if not sys.platform.startswith("linux") or platform.machine() != "x86_64":
            raise IsolationError("execute only on x86_64 Linux; Windows is orchestration only")
        self.work, self.runtime = work.resolve(), runtime.resolve()
        if not self.work.is_dir() or not (self.runtime / "bin/python").exists():
            raise IsolationError("a task directory and dedicated runtime are required")
        self.process = None
        self.closed = False
        self.isolation = None

    def run(self, cmd: str, timeout_s: float) -> tuple[int, str]:
        if self.closed or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("closed environment or invalid timeout")
        args = ["/usr/bin/python3", "-I", str(Path(__file__).resolve()), "--enter", str(os.getpid())]
        started = time.monotonic()
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True, env={"PATH": "/usr/bin:/bin"})
        self.process = proc
        output, code = bytearray(), None
        try:
            cfg = {"work": str(self.work), "runtime": str(self.runtime), "cmd": cmd,
                   "cpu": max(1, math.ceil(timeout_s))}
            proc.stdin.write((json.dumps(cfg) + "\n").encode())
            proc.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    remaining = timeout_s - (time.monotonic() - started)
                    if remaining <= 0:
                        code = 124
                        break
                    if selector.select(min(remaining, 0.1)):
                        chunk = os.read(proc.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        output.extend(chunk)
                        if len(output) > OUTPUT_CAP:
                            code = 125
                            break
            if code is not None:
                self._kill()
            else:
                try:
                    code = proc.wait(timeout=max(0.01, timeout_s - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    code = 124
                    self._kill()
        finally:
            self._kill()
            proc.wait()
            proc.stdout.close()
            self.process = None
        out = bytes(output[:OUTPUT_CAP]).decode("utf-8", errors="replace")
        first, separator, out = out.partition("\n")
        try:
            ready = json.loads(first)
        except ValueError as error:
            raise IsolationError("launcher did not establish isolation") from error
        if not separator or not isinstance(ready, dict) or ready.get("isolation_ready") != VERSION:
            raise IsolationError("missing isolation receipt")
        self.isolation = ready
        return code, out

    def _kill(self):
        if self.process is not None and self.process.poll() is None:
            # Killing unshare triggers --kill-child even when descendants have called setsid.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        self.closed = True
        self._kill()


if __name__ == "__main__" and sys.argv[1:] == ["--launch"]:
    launch()
elif __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--enter":
    enter(int(sys.argv[2]))
