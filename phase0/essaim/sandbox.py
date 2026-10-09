"""Run model-written code in a separate, limited process (E11). Model code never runs in this process.

Each call starts a fresh Python child (`python -s -B -c CHILD`) in a new temporary directory, sends it a JSON
job on stdin and reads tagged result lines back; the directory is deleted afterwards. Barriers, by platform:

  both     a Python audit hook, installed before the candidate runs, refuses: writes outside the temporary
           directory (open for writing, remove, rename, mkdir, chmod, rmtree...), new processes (subprocess,
           os.system, fork, exec, spawn, kill), sockets (create, connect, bind, DNS) and chdir. It cannot be
           removed from Python, but native code (ctypes) could bypass it: it guards against accidents, not
           against an adversary.
           A clean environment (PYTHONHASHSEED=0, one BLAS thread, HOME and TMP in the temp dir), stdin, stdout
           and stderr on the null device (results go through a private copy of the stdout pipe, each line
           tagged with a random nonce), random.seed(0) before every case, a wall-clock limit for the whole
           child (killed, with its process group or job, when exceeded).
  Linux    rlimits set by the child on itself before reading the candidate (address space, CPU seconds,
           file size, no core files), a per-case timeout (SIGALRM), and no network at all when `unshare --net`
           works on this machine (probed once; recorded in isolation()).
  Windows  a job object (memory limit for the whole process tree, killed as a whole); no per-case timeout and
           no network namespace: only the child's wall-clock limit and the audit hook. Windows is only used
           for the smoke test; the experiment runs on the Linux VM.

Three phases, one child each: `visible` (the prompt's tests, pass/fail per test), `extra` (outputs on given
inputs, as hashes of a normalised value, for functional clustering) and `hidden` (grading with the EvalPlus
tests, stops at the first failure).
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time

SANDBOX_VERSION = "sandbox-v1"
MEM_MB = {"visible": 2048, "extra": 2048, "hidden": 4096}  # some EvalPlus cases build ~1 GB outputs, twice
# (per case, whole child), seconds. Hidden cases include the reference call of some EvalPlus tests (ref_func) and
# up to ~1 GB outputs (Mbpp/255): 10 s per case; only the first failing case is ever waited for.
LIMITS = {"visible": (2.0, 20.0), "extra": (2.0, 30.0), "hidden": (10.0, 300.0)}
POSIX = os.name == "posix"

CHILD = r'''
import sys, os, json, copy, math, random, re, time, ast, hashlib
cfg = json.loads(sys.stdin.buffer.read().decode("utf-8"))
_nonce = cfg["nonce"]
_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
_null = os.open(os.devnull, os.O_RDWR)
for _fd in (0, 1, 2):
    os.dup2(_null, _fd)
sys.stdin = open(os.devnull, "r")
sys.stdout = sys.stderr = open(os.devnull, "w")


def _emit(**kw):
    _out.write(_nonce + " " + json.dumps(kw) + "\n")
    _out.flush()


try:
    import resource
    for _r, _v in ((resource.RLIMIT_AS, cfg["mem_mb"] << 20), (resource.RLIMIT_CPU, cfg["cpu_s"]),
                   (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(_r, (_v, _v))
        except (ValueError, OSError):
            pass
except ImportError:
    pass

import signal


class _Timeout(BaseException):
    pass


_ALARM = hasattr(signal, "setitimer")
if _ALARM:
    def _on_alarm(*a):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _on_alarm)


def _call(f, t):
    if _ALARM:
        signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return f()
    finally:
        if _ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)


def _err(e):
    return "Timeout" if isinstance(e, _Timeout) else type(e).__name__


_TMP = os.path.realpath(cfg["tmp"])


def _inside(p):
    try:
        if isinstance(p, int):
            return False
        p = os.fsdecode(p)
        full = os.path.realpath(p if os.path.isabs(p) else os.path.join(_TMP, p))
        return full == _TMP or full.startswith(_TMP + os.sep)
    except Exception:
        return False


_WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_PATHS = {"os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1), "os.mkdir": (0,), "os.chmod": (0,),
          "os.chown": (0,), "os.link": (0, 1), "os.symlink": (0, 1), "os.truncate": (0,), "os.utime": (0,),
          "os.chflags": (0,), "os.lchflags": (0,), "os.lchmod": (0,), "os.setxattr": (0,), "os.removexattr": (0,),
          "os.mkfifo": (0,), "os.mknod": (0,), "shutil.rmtree": (0,), "shutil.copyfile": (1,),
          "shutil.copytree": (1,), "shutil.move": (0, 1), "shutil.chown": (0,), "shutil.make_archive": (0,),
          "shutil.unpack_archive": (1,)}
_BLOCK = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg",
          "os.chdir", "os.fchdir", "os.startfile", "subprocess.Popen", "pty.spawn", "socket.", "webbrowser.open",
          "urllib.Request", "_winapi.CreateProcess", "winreg.")


def _hook(ev, args):
    if ev == "open":
        a = list(args) + [None, None, 0]
        path, mode, flags = a[0], a[1], a[2]
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _WRITE)
        if w and not _inside(path):
            raise PermissionError("sandbox: write outside the temporary directory")
    elif ev in _PATHS:
        for k in _PATHS[ev]:
            if k < len(args) and not _inside(args[k]):
                raise PermissionError("sandbox: " + ev + " outside the temporary directory")
    elif ev.startswith(_BLOCK):
        raise PermissionError("sandbox: " + ev + " is not allowed")


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _key(v):
    return json.dumps(v, sort_keys=True, default=str)


def _norm(x, d=0):
    if d > 40:
        return "<deep>"
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x if -(1 << 62) < x < (1 << 62) else "int:" + hex(x)[-64:] + ":" + str(x.bit_length())
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x in (float("inf"), float("-inf")):
            return repr(x)
        return float(format(x, ".9g")) + 0.0
    if isinstance(x, complex):
        return ["complex", _norm(x.real, d + 1), _norm(x.imag, d + 1)]
    if isinstance(x, str):
        return x if len(x) <= 4096 else "str:" + hashlib.sha1(x.encode("utf-8", "replace")).hexdigest()
    if isinstance(x, (bytes, bytearray)):
        return ["bytes", bytes(x).hex()]
    if isinstance(x, list):
        return [_norm(v, d + 1) for v in x]
    if isinstance(x, tuple):
        return ["tuple"] + [_norm(v, d + 1) for v in x]
    if isinstance(x, (set, frozenset)):
        return ["set"] + sorted((_norm(v, d + 1) for v in x), key=_key)
    if isinstance(x, dict):
        return ["dict"] + sorted(([_norm(k, d + 1), _norm(v, d + 1)] for k, v in x.items()), key=_key)
    if type(x).__module__ == "numpy" and hasattr(x, "tolist"):  # numpy scalars and arrays: plain values
        return _norm(x.tolist(), d + 1)
    return ["repr", type(x).__name__, re.sub(r" at 0x[0-9A-Fa-f]+", "", repr(x))[:1000]]


def _sig(x):
    return hashlib.sha1(_key(_norm(x)).encode("utf-8")).hexdigest()[:16]


_phase, _entry, _cs = cfg["phase"], cfg["entry"], cfg["call_s"]
_t0 = time.perf_counter()
_tns = {"__name__": "__test__"}
if _phase == "hidden":  # the dataset's test code: trusted, prepared before the candidate is loaded
    exec(compile(cfg["spec"]["prelude"], "<prelude>", "exec"), _tns)
    for _s in cfg["spec"]["setup"]:
        exec(compile(_s, "<setup>", "exec"), _tns)
    _cases = list(eval(cfg["spec"]["iter"], _tns))
    _body = compile(cfg["spec"]["body"], "<test>", "exec")
    _bind = compile(cfg["spec"]["target"] + " = __case", "<bind>", "exec")
elif _phase == "extra":
    _inputs = [ast.literal_eval(s) for s in cfg["inputs"]]
else:
    _tests = []
    for _t in cfg["tests"]:
        try:
            _tests.append(compile(_t, "<visible>", "exec"))
        except SyntaxError:
            _tests.append(None)

sys.addaudithook(_hook)
_ns = {"__name__": "__main__"}
random.seed(0)
try:
    _call(lambda: exec(compile(cfg["header"], "<header>", "exec"), _ns), _cs)
    _stub = _ns.get(_entry)
    _call(lambda: exec(compile(cfg["code"], "<candidate>", "exec"), _ns), _cs)
    _fn = _ns.get(_entry)
    _load = None if callable(_fn) and _fn is not _stub else "NoEntry"
except BaseException as e:
    _fn, _load = None, _err(e)
_emit(kind="load", err=_load)
if _load is None:
    if _phase == "visible":
        _ns["_close"] = _close
        for _i, _t in enumerate(_tests):
            random.seed(0)
            try:
                if _t is None:
                    raise SyntaxError("test")
                _call(lambda: exec(_t, _ns), _cs)
                _emit(kind="case", i=_i, ok=True, err=None)
            except BaseException as e:
                _emit(kind="case", i=_i, ok=False, err=_err(e))
    elif _phase == "extra":
        for _i, _a in enumerate(_inputs):
            random.seed(0)
            try:
                _emit(kind="case", i=_i, sig=_call(lambda: _sig(_fn(*copy.deepcopy(_a))), _cs), err=None)
            except BaseException as e:
                _emit(kind="case", i=_i, sig="!", err=_err(e))
    else:
        _tns["candidate"] = _fn
        _tns[_entry] = _fn
        _done = len(_cases)
        for _i, _c in enumerate(_cases):
            _tns["__case"] = copy.deepcopy(_c)
            exec(_bind, _tns)
            random.seed(0)
            try:
                _call(lambda: exec(_body, _tns), _cs)
            except BaseException as e:
                _emit(kind="hidden", ok=False, n_pass=_i, n=len(_cases), err=_err(e))
                _done = None
                break
        if _done is not None:
            _emit(kind="hidden", ok=True, n_pass=_done, n=_done, err=None)
_emit(kind="done", ms=round((time.perf_counter() - _t0) * 1000))
_out.close()
os._exit(0)
'''

_ISO: dict | None = None
_ISO_LOCK = threading.Lock()


def _probe_unshare() -> list[str]:
    exe = shutil.which("unshare")
    if not POSIX or not exe:
        return []
    for pre in ([exe, "--net", "--"], [exe, "--net", "--map-root-user", "--"]):
        try:
            if subprocess.run(pre + ["true"], capture_output=True, timeout=20).returncode == 0:
                return pre
        except (OSError, subprocess.TimeoutExpired):
            pass
    return []


def isolation() -> dict:
    """What actually isolates the children on this machine (recorded in the execution manifest)."""
    global _ISO
    with _ISO_LOCK:
        if _ISO is None:
            pre = _probe_unshare()
            _ISO = {"version": SANDBOX_VERSION, "platform": sys.platform, "prefix": pre,
                    "network": "unshare --net (no network namespace access)" if pre else "audit hook only",
                    "memory": f"RLIMIT_AS, MiB: {MEM_MB}" if POSIX else
                              (f"job object, MiB: {MEM_MB}" if _winjob_ok() else "none"),
                    "per_case_timeout": POSIX, "filesystem": "audit hook: writes only inside a fresh temp dir",
                    "limits_s": LIMITS}
        return _ISO


# ---------- Windows job objects (memory limit + kill the whole tree) ----------

def _winjob_ok() -> bool:
    if POSIX:
        return False
    try:
        _Job(64).close()
        return True
    except OSError:
        return False


class _Job:
    def __init__(self, mem_mb: int):
        import ctypes
        from ctypes import wintypes
        self.k32 = k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll")
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Ext(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.h = k32.CreateJobObjectW(None, None)
        if not self.h:
            raise OSError("CreateJobObject")
        info = Ext()
        info.Basic.LimitFlags = 0x200 | 0x2000  # JOB_MEMORY | KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = mem_mb << 20
        if not k32.SetInformationJobObject(self.h, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError("SetInformationJobObject")

    def adopt_and_resume(self, proc: subprocess.Popen):
        ok = bool(self.k32.AssignProcessToJobObject(self.h, int(proc._handle)))
        self.ntdll.NtResumeProcess(int(proc._handle))
        if not ok:
            raise OSError("AssignProcessToJobObject")

    def kill(self):
        self.k32.TerminateJobObject(self.h, 1)

    def close(self):
        if self.h:
            self.k32.CloseHandle(self.h)
            self.h = None


# ---------- running a child ----------

def _env(tmp: str) -> dict:
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": tmp, "TMPDIR": tmp,
           "TEMP": tmp, "TMP": tmp, "USERPROFILE": tmp, "LANG": "C.UTF-8", "PATH": os.environ.get("PATH", os.defpath)}
    if not POSIX:
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _run(job: dict, phase: str) -> tuple[list[dict], str, int]:
    """(result records, status, wall ms); status: ok, timeout (killed), crash (no final record)."""
    call_s, total_s = LIMITS[phase]
    iso = isolation()
    tmp = tempfile.mkdtemp(prefix="e11-")
    nonce = secrets.token_hex(8)
    cfg = {**job, "phase": phase, "nonce": nonce, "tmp": tmp, "call_s": call_s, "mem_mb": MEM_MB[phase],
           "cpu_s": int(total_s) + 2}
    cmd = iso["prefix"] + [sys.executable, "-s", "-B", "-c", CHILD]
    t0 = time.perf_counter()
    job_obj = None
    try:
        if POSIX:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 cwd=tmp, env=_env(tmp), start_new_session=True)
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 cwd=tmp, env=_env(tmp), creationflags=0x4 | 0x200)  # SUSPENDED | NEW_PROCESS_GROUP
            try:
                job_obj = _Job(MEM_MB[phase])
                job_obj.adopt_and_resume(p)
            except OSError:  # no job object: the process must still be resumed (wall-clock limit only)
                if job_obj is not None:
                    job_obj.close()
                    job_obj = None
                import ctypes
                ctypes.WinDLL("ntdll").NtResumeProcess(int(p._handle))
        status = "ok"
        try:
            out, _ = p.communicate(json.dumps(cfg).encode("utf-8"), timeout=total_s)
        except subprocess.TimeoutExpired:
            status = "timeout"
            _kill(p, job_obj)
            out, _ = p.communicate()
    finally:
        if job_obj is not None:
            job_obj.close()
        shutil.rmtree(tmp, ignore_errors=True)
    recs = []
    for line in out.decode("utf-8", "replace").splitlines():
        if line.startswith(nonce + " "):
            try:
                recs.append(json.loads(line[len(nonce) + 1:]))
            except ValueError:
                pass
    if status == "ok" and not any(r.get("kind") == "done" for r in recs):
        status = "crash"
    return recs, status, round((time.perf_counter() - t0) * 1000)


def _kill(p: subprocess.Popen, job_obj):
    if POSIX:
        import signal
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif job_obj is not None:
        job_obj.kill()
    try:
        p.kill()
    except OSError:
        pass


def _load(recs: list[dict], status: str) -> str | None:
    rec = next((r for r in recs if r.get("kind") == "load"), None)
    if rec is None:
        return "Timeout" if status == "timeout" else "Crash"
    return rec["err"]


def run_visible(code: str, entry: str, header: str, tests: list[str]) -> dict:
    """Pass/fail of each visible test. A test not reached (child killed) fails with Timeout or Crash."""
    recs, status, ms = _run({"code": code, "entry": entry, "header": header, "tests": tests}, "visible")
    load = _load(recs, status)
    got = {r["i"]: r for r in recs if r.get("kind") == "case"}
    miss = "Timeout" if status == "timeout" else (load or "Crash")
    cases = [{"ok": got[i]["ok"], "err": got[i]["err"]} if i in got else {"ok": False, "err": miss}
             for i in range(len(tests))]
    return {"status": status, "load": load, "cases": cases, "ms": ms}


def run_extra(code: str, entry: str, header: str, inputs: list[tuple]) -> dict:
    """Output signature on each input: 16 hex chars of a hash of the normalised value, '!' for no output
    (exception, timeout, crash, or the program does not load)."""
    recs, status, ms = _run({"code": code, "entry": entry, "header": header, "inputs": [repr(a) for a in inputs]},
                            "extra")
    got = {r["i"]: r for r in recs if r.get("kind") == "case"}
    return {"status": status, "load": _load(recs, status), "ms": ms,
            "sigs": [got[i]["sig"] if i in got else "!" for i in range(len(inputs))]}


def run_hidden(code: str, entry: str, header: str, spec: dict) -> dict:
    """Grading: passes only if every hidden case passes (first failure stops the run)."""
    recs, status, ms = _run({"code": code, "entry": entry, "header": header, "spec": spec}, "hidden")
    h = next((r for r in recs if r.get("kind") == "hidden"), None)
    load = _load(recs, status)
    if h is None:
        return {"status": status, "load": load, "pass": False, "n_pass": 0, "n": None,
                "err": load or ("Timeout" if status == "timeout" else "Crash"), "ms": ms}
    return {"status": status, "load": load, "pass": bool(h["ok"]) and status == "ok", "n_pass": h["n_pass"],
            "n": h["n"], "err": h["err"], "ms": ms}
