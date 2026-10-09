"""Run model-written code in a separate, limited process (E11), and grade it in ANOTHER, trusted process.

Model code never runs in this process, and never in a process that holds an expected value. Per program:

  candidate child   (untrusted; one per phase: `visible`, `extra`, `hidden`) receives the program and the
                    INPUTS only, calls the entry point on each, and sends back what it returned: the value
                    itself in a typed encoding (CODEC below), a digest of it when it is too large to send, or
                    the name of the exception. It never sees an expected value, a test, or the grading code.
  grader child      (trusted: dataset code only, no model code) prepares the inputs once per problem (the
                    arguments of the visible tests, the inputs of the hidden EvalPlus tests) and, per program,
                    replays the dataset's own assertions with the entry point replaced by the outputs the
                    candidate sent (`assertion(candidate(*inp), exp, atol)`, the visible `assert`s). Expected
                    values, `ref_func` and `assertion` exist only there.
An output is decoded by a strict decoder that only builds plain values (None, bool, int, float, complex, str,
bytes, list, tuple, set, frozenset, dict, numeric numpy arrays): no pickle, no object of the candidate's
classes reaches the grader. Values above NODE_BUDGET nodes travel as a digest (SHA-256 of their repr, the
top-level list or tuple element by element) and pass only if the expected value has the same digest, i.e.
the same repr (exact equality; float tolerance and int/float mixing do not apply to such huge outputs).

Barriers on the candidate child, by platform:
  both     a Python audit hook, installed before the candidate runs, refuses: writes outside a fresh temporary
           directory (open for writing, remove, rename, mkdir, chmod, rmtree...), every `dir_fd=` variant of
           these, opening a directory outside the temporary directory (so no directory descriptor exists to
           resolve a relative path elsewhere), reads outside the temporary directory and the Python
           installation (stdlib, site-packages, system libraries; never the repository, its data and results,
           or the Hugging Face cache), new processes, sockets, chdir, and ctypes loading or calling native
           functions. Native code could still bypass a Python hook; on Linux the OS rules below do not depend
           on it.
           A clean environment (PYTHONHASHSEED=0, one BLAS thread, HOME and TMP in the temp dir), stdin, stdout
           and stderr on the null device (results go through a private copy of the stdout pipe, each line
           tagged with a random nonce), random.seed(0) before every call, a wall-clock limit for the whole
           child, and at most OUT_CAP bytes read from it.
  Linux    rlimits set by the child on itself before reading the candidate (address space, CPU seconds,
           file size, no core files), a per-call timeout (SIGALRM), no network when `unshare --net` works
           (probed once), Landlock when the kernel has it (probed, recorded in isolation()): writes only in
           the temporary directory, reads only in the same places as the audit hook, enforced by the kernel.
           PR_SET_PDEATHSIG: a child dies with the process that started it.
  Windows  a job object (memory limit for the whole process tree, killed as a whole, also when this process
           dies); no per-call timeout, no network namespace and no Landlock: Windows is only used for the smoke
           test; the experiment runs on the Linux VM.
Every child is registered while it runs; it is killed and reaped on any interruption of the call that
started it, and `shutdown()` kills all of them (exec_code.py calls it on SIGTERM and Ctrl-C).
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
from pathlib import Path

SANDBOX_VERSION = "sandbox-v3"  # v3: ctypes initialised before the hook (numpy under Windows)
MEM_MB = {"visible": 2048, "extra": 2048, "hidden": 4096, "grade": 4096}  # some EvalPlus outputs reach ~1 GB
# (per call, whole child), seconds. Hidden cases: some EvalPlus outputs take seconds to build (Mbpp/255).
LIMITS = {"visible": (2.0, 20.0), "extra": (2.0, 30.0), "hidden": (10.0, 300.0), "grade": (60.0, 900.0)}
OUT_CAP = 512 << 20  # bytes read from one child at most (the child is killed beyond)
NODE_BUDGET = 1_000_000  # values with more nodes are sent as a digest
ENCODE_S = 60.0  # seconds to encode or hash one output (harness work, not counted in the call limit)
POSIX = os.name == "posix"
LINUX = sys.platform.startswith("linux")
PHASE0 = Path(__file__).resolve().parent.parent


class SandboxError(RuntimeError):
    """The trusted grader could not prepare or grade a problem (a harness problem, never a program's)."""


# ---------- typed encoding, shared by every process (exec'd in the children, imported here) ----------

CODEC = r'''
import hashlib as _hl, json as _js, re as _re


class _Big(Exception):
    pass


def _ckey(n):
    return _js.dumps(n, separators=(",", ":"), ensure_ascii=True)


def _walk(x, sig, budget, d):
    budget[0] -= 1
    if budget[0] < 0 or d > 200:
        raise _Big()
    if x is None:
        return ["N"]
    if isinstance(x, bool):
        return ["b", True if x else False]
    if isinstance(x, int):
        return ["i", hex(int.__int__(x))]
    if isinstance(x, float):
        v = float.__float__(x)
        if v != v:
            return ["f", "nan"]
        if v in (float("inf"), float("-inf")):
            return ["f", repr(v)]
        return ["f", repr(float(format(v, ".9g")) + 0.0)] if sig else ["f", float.hex(v)]
    if isinstance(x, complex):
        return ["c", _walk(x.real, sig, budget, d + 1), _walk(x.imag, sig, budget, d + 1)]
    if isinstance(x, str):
        s = str.__str__(x)
        if sig and len(s) > 4096:
            return ["H", _hl.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()]
        return ["s", s]
    if isinstance(x, bytearray):
        return ["Y", bytes(x).hex()]
    if isinstance(x, bytes):
        return ["y", bytes.hex(x)]
    if isinstance(x, list):
        return ["l", [_walk(v, sig, budget, d + 1) for v in list.__iter__(x)]]
    if isinstance(x, tuple):
        return ["t", [_walk(v, sig, budget, d + 1) for v in tuple.__iter__(x)]]
    # Sets and dicts: iteration order kept when a value travels (a program may depend on a dict's insertion
    # order, as the reference solutions of Mbpp/390, 421 and 475 do); sorted in a signature (canonical).
    if isinstance(x, frozenset):
        items = [_walk(v, sig, budget, d + 1) for v in frozenset.__iter__(x)]
        return ["F", sorted(items, key=_ckey) if sig else items]
    if isinstance(x, set):
        items = [_walk(v, sig, budget, d + 1) for v in set.__iter__(x)]
        return ["S", sorted(items, key=_ckey) if sig else items]
    if isinstance(x, dict):
        items = [[_walk(k, sig, budget, d + 1), _walk(v, sig, budget, d + 1)] for k, v in dict.items(x)]
        return ["d", sorted(items, key=lambda kv: _ckey(kv[0])) if sig else items]
    mod = type(x).__module__ or ""
    if mod == "numpy" or mod.startswith("numpy."):
        if getattr(x, "ndim", None) == 0 and hasattr(x, "item"):
            return _walk(x.item(), sig, budget, d + 1)
        if hasattr(x, "tolist") and hasattr(x, "dtype") and x.dtype.kind in "biufc":
            body = _walk(x.tolist(), sig, budget, d + 1)
            return body if sig else ["a", x.dtype.str, [int(n) for n in x.shape], body]
    try:
        r = _re.sub(r" at 0x[0-9A-Fa-f]+", "", repr(x))[:1000]
    except BaseException:
        r = "?"
    return ["r", type(x).__name__, r]


def enc_text(x, sig=False):
    """Typed JSON text of a value (exact; sig: floats to 9 significant digits, long strings hashed), or None
    when it has more than NODE_BUDGET nodes or is deeper than 200 levels."""
    try:
        return _ckey(_walk(x, sig, [NODE_BUDGET], 0))
    except (_Big, RecursionError):
        return None


def digest(x):
    """SHA-256 of repr(x), a top-level list or tuple element by element (bounded memory)."""
    h = _hl.sha256()
    if type(x) in (list, tuple):
        h.update(b"L[" if type(x) is list else b"T(")
        for e in x:
            h.update(repr(e).encode("utf-8", "surrogatepass"))
            h.update(b",")
    else:
        h.update(b"R")
        h.update(repr(x).encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def signature(x):
    t = enc_text(x, sig=True)
    if t is None:
        return "D" + digest(x)[:15]
    return _hl.sha256(t.encode("ascii")).hexdigest()[:16]


class Opaque:
    """A value of a type the codec does not carry: equal to nothing."""
    def __init__(self, tname, text):
        self.tname, self.text = tname, text

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    __hash__ = object.__hash__

    def __repr__(self):
        return "<opaque " + str(self.tname) + ">"


def _dec(n, d):
    if d > 210 or not isinstance(n, list) or not n or not isinstance(n[0], str):
        raise ValueError("codec: malformed value")
    t, a = n[0], n[1:]
    if t == "N" and not a:
        return None
    if t == "b" and len(a) == 1 and isinstance(a[0], bool):
        return a[0]
    if t == "i" and len(a) == 1 and isinstance(a[0], str) and len(a[0]) <= 1 << 20 and _re.fullmatch(r"-?0x[0-9a-f]+", a[0]):
        return int(a[0], 16)
    if t == "f" and len(a) == 1 and isinstance(a[0], str) and len(a[0]) < 64:
        return float(a[0]) if a[0] in ("nan", "inf", "-inf") else float.fromhex(a[0])
    if t == "c" and len(a) == 2:
        re_, im = _dec(a[0], d + 1), _dec(a[1], d + 1)
        if type(re_) is not float or type(im) is not float:
            raise ValueError("codec: complex")
        return complex(re_, im)
    if t == "s" and len(a) == 1 and isinstance(a[0], str):
        return a[0]
    if t in ("y", "Y") and len(a) == 1 and isinstance(a[0], str):
        b = bytes.fromhex(a[0])
        return b if t == "y" else bytearray(b)
    if t in ("l", "t", "S", "F") and len(a) == 1 and isinstance(a[0], list):
        items = [_dec(v, d + 1) for v in a[0]]
        return {"l": list, "t": tuple, "S": set, "F": frozenset}[t](items)
    if t == "d" and len(a) == 1 and isinstance(a[0], list):
        out = {}
        for kv in a[0]:
            if not isinstance(kv, list) or len(kv) != 2:
                raise ValueError("codec: dict item")
            out[_dec(kv[0], d + 1)] = _dec(kv[1], d + 1)
        return out
    if t == "a" and len(a) == 3 and isinstance(a[0], str) and isinstance(a[1], list):
        import numpy as _np
        dt = _np.dtype(a[0])
        if dt.kind not in "biufc" or not all(isinstance(k, int) and k >= 0 for k in a[1]):
            raise ValueError("codec: array")
        return _np.array(_dec(a[2], d + 1), dtype=dt).reshape(a[1])
    if t == "r" and len(a) == 2 and all(isinstance(v, str) for v in a):
        return Opaque(a[0], a[1])
    raise ValueError("codec: unknown tag")


def dec(text):
    """The value of a typed JSON text; ValueError (or TypeError: unhashable key) if malformed."""
    return _dec(_js.loads(text), 0)
'''

_codec_ns: dict = {"NODE_BUDGET": NODE_BUDGET}
exec(compile(CODEC, "<codec>", "exec"), _codec_ns)
enc_text, dec, digest, signature, Opaque = (_codec_ns[k] for k in ("enc_text", "dec", "digest", "signature", "Opaque"))

# ---------- common start of every child: die with the parent, then read the job ----------

_PROLOGUE = r'''
import sys, os
if sys.platform.startswith("linux"):
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass
import json, math, random, re, time, ast, signal
cfg = json.loads(sys.stdin.buffer.read().decode("utf-8"))
if sys.platform.startswith("linux") and cfg.get("ppid") and os.getppid() != cfg["ppid"]:  # parent died before prctl
    os._exit(70)
_nonce = cfg["nonce"]
_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
_null = os.open(os.devnull, os.O_RDWR)
for _fd in (0, 1, 2):
    os.dup2(_null, _fd)
sys.stdin = open(os.devnull, "r")
sys.stdout = sys.stderr = open(os.devnull, "w")
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
NODE_BUDGET = cfg["node_budget"]


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


_t0 = time.perf_counter()
'''

# ---------- the candidate child (untrusted code) ----------

CHILD = _PROLOGUE + CODEC + r'''
_TMP = os.path.realpath(cfg["tmp"])
_N = os.path.normcase


def _real(p):
    if isinstance(p, int):
        return None
    p = os.fsdecode(p)
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(_TMP, p))


def _under(full, root):
    full, root = _N(full), _N(root)
    return full == root or full.startswith(root.rstrip(os.sep) + os.sep)


_DENY = [os.path.realpath(p) for p in cfg["deny"]]
_ALLOW = []
for _p in [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, os.path.dirname(os.__file__)] + \
        [p for p in sys.path if p] + cfg["read_extra"]:
    _p = os.path.realpath(_p)
    # a root that contains a denied place (the repository through a .pth file...) is not a read root
    if os.path.exists(_p) and not any(_under(d, _p) for d in _DENY) and _p not in _ALLOW:
        _ALLOW.append(_p)


def _inside(p):
    try:
        full = _real(p)
        return full is not None and _under(full, _TMP)
    except Exception:
        return False


def _readable(p):
    try:
        full = _real(p)
        if full is None:
            return True  # an already open descriptor
        if _under(full, _TMP):
            return True
        if any(_under(full, d) for d in _DENY):
            return False
        return any(_under(full, r) for r in _ALLOW)
    except Exception:
        return False


def _landlock():
    """Kernel-enforced file rules (Linux >= 5.13 with Landlock enabled): writes in the temp dir only, reads in
    the temp dir and the read roots only. Returns a short status, recorded in the execution manifest."""
    if not sys.platform.startswith("linux") or not cfg.get("landlock"):
        return "off"
    try:
        import ctypes, struct
        libc = ctypes.CDLL(None, use_errno=True)
        sc = libc.syscall
        sc.restype = ctypes.c_long
        abi = sc(ctypes.c_long(444), None, ctypes.c_size_t(0), ctypes.c_uint32(1))  # create_ruleset VERSION
        if abi < 1:
            return "unavailable"
        rw = (1 << 13) - 1  # EXECUTE .. MAKE_SYM
        if abi >= 2:
            rw |= 1 << 13  # REFER
        if abi >= 3:
            rw |= 1 << 14  # TRUNCATE
        ro_dir, ro_file = (1 << 0) | (1 << 2) | (1 << 3), (1 << 0) | (1 << 2)
        attr = ctypes.create_string_buffer(struct.pack("=Q", rw))
        fd = sc(ctypes.c_long(444), attr, ctypes.c_size_t(8), ctypes.c_uint32(0))
        if fd < 0:
            return "create failed"
        rules = [(_TMP, rw, rw & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)))]
        rules += [(r, ro_dir, ro_file) for r in _ALLOW]
        rules += [(os.devnull, None, (1 << 1) | (1 << 2))]
        for path, on_dir, on_file in rules:
            try:
                pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                acc = on_dir if os.path.isdir(path) and on_dir is not None else on_file
                buf = ctypes.create_string_buffer(struct.pack("=Qi", acc, pfd))
                if sc(ctypes.c_long(445), ctypes.c_int(fd), ctypes.c_int(1), buf, ctypes.c_uint32(0)) != 0:
                    return "add_rule failed"
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            return "no_new_privs failed"
        if sc(ctypes.c_long(446), ctypes.c_int(fd), ctypes.c_uint32(0)) != 0:
            return "restrict failed"
        os.close(fd)
        return "on (ABI %d)" % abi
    except Exception as e:
        return "error " + type(e).__name__


_WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# event -> (indices of path arguments, indices of dir_fd arguments)
_PATHS = {"os.remove": ((0,), (1,)), "os.rmdir": ((0,), (1,)), "os.rename": ((0, 1), (2, 3)),
          "os.mkdir": ((0,), (2,)), "os.chmod": ((0,), (2,)), "os.chown": ((0,), (3,)),
          "os.link": ((0, 1), (2, 3)), "os.symlink": ((0, 1), (2,)), "os.truncate": ((0,), ()),
          "os.utime": ((0,), (3,)), "os.chflags": ((0,), ()), "os.lchflags": ((0,), ()), "os.lchmod": ((0,), ()),
          "os.setxattr": ((0,), ()), "os.removexattr": ((0,), ()), "os.mkfifo": ((0,), (2,)),
          "os.mknod": ((0,), (3,)), "shutil.rmtree": ((0,), (1,)), "shutil.copyfile": ((1,), ()),
          "shutil.copytree": ((1,), ()), "shutil.move": ((0, 1), ()), "shutil.chown": ((0,), ()),
          "shutil.make_archive": ((0,), ()), "shutil.unpack_archive": ((1,), ())}
_BLOCK = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg",
          "os.chdir", "os.fchdir", "os.chroot", "os.startfile", "subprocess.Popen", "pty.spawn", "socket.",
          "webbrowser.open", "urllib.Request", "_winapi.CreateProcess", "winreg.", "ctypes.dlopen", "ctypes.dlsym",
          "ctypes.call_function", "ctypes.cdata", "ctypes.addressof", "ctypes.string_at", "ctypes.wstring_at",
          "ctypes.set_errno", "ctypes.get_errno", "sys.remote_exec")


def _hook(ev, args):
    if ev == "open":
        a = list(args) + [None, None, 0]
        path, mode, flags = a[0], a[1], a[2]
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _WRITE)
        if w and not _inside(path):
            raise PermissionError("sandbox: write outside the temporary directory")
        if not w and not _readable(path):
            raise PermissionError("sandbox: read outside the allowed places")
        if not isinstance(path, int) and not _inside(path) and os.path.isdir(_real(path)):
            raise PermissionError("sandbox: no directory descriptor outside the temporary directory")
    elif ev in _PATHS:
        paths, dirfds = _PATHS[ev]
        for k in dirfds:
            if k < len(args) and args[k] not in (None, -1, -100):  # -1 (audit) / -100 (AT_FDCWD): no dir_fd
                raise PermissionError("sandbox: " + ev + " with dir_fd is not allowed")
        for k in paths:
            if k < len(args) and not _inside(args[k]):
                raise PermissionError("sandbox: " + ev + " outside the temporary directory")
    elif ev.startswith(_BLOCK):
        raise PermissionError("sandbox: " + ev + " is not allowed")


_landlock_status = _landlock()
_phase, _entry, _cs = cfg["phase"], cfg["entry"], cfg["call_s"]
_inputs = cfg["inputs"]
_emit(kind="iso", landlock=_landlock_status)
try:  # imported before the hook forbids loading native libraries: on Windows, importing ctypes (which numpy
    import ctypes  # does) loads kernel32, and `import numpy` in a program must keep working
except Exception:
    pass
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
    for _i, _t in enumerate(_inputs):
        random.seed(0)
        try:
            _a = ast.literal_eval(_t) if _phase == "extra" else dec(_t)  # fresh arguments for every call
            _r = _call(lambda: _fn(*_a), _cs)
        except BaseException as e:
            _emit(kind="case", i=_i, err=_err(e))
            if _phase == "hidden":  # grading stops at the first failure
                break
            continue
        try:
            if _phase == "extra":
                _rec = {"sig": _call(lambda: signature(_r), cfg["encode_s"])}
            else:
                _txt = _call(lambda: enc_text(_r), cfg["encode_s"])
                _rec = {"out": _txt} if _txt is not None else {"digest": _call(lambda: digest(_r), cfg["encode_s"])}
        except BaseException as e:
            _rec = {"err": "Output" + _err(e)}
        _emit(kind="case", i=_i, **_rec)
        if _phase == "hidden" and "err" in _rec:
            break
_emit(kind="done", ms=round((time.perf_counter() - _t0) * 1000))
_out.close()
os._exit(0)
'''

# ---------- the grader child (trusted: dataset code only) ----------

GRADER = _PROLOGUE + CODEC + r'''
class _Digest:
    """Stands for an output too large to send: equal to a value with the same digest."""
    def __init__(self, d):
        self.d = d

    def __eq__(self, other):
        return digest(other) == self.d

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = object.__hash__


class _CandidateError(Exception):
    pass


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _value(rec, missing):
    if rec is None:
        raise _CandidateError(missing)
    if "err" in rec:
        raise _CandidateError(rec["err"])
    if "digest" in rec:
        return _Digest(rec["digest"])
    try:
        return dec(rec["out"])
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _CandidateError("BadOutput")


def _hidden_cases(spec):
    ns = {"__name__": "__test__"}
    exec(compile(spec["prelude"], "<prelude>", "exec"), ns)
    for s in spec["setup"]:
        exec(compile(s, "<setup>", "exec"), ns)
    cases = list(eval(spec["iter"], ns))
    return ns, cases, compile(spec["target"] + " = __case", "<bind>", "exec"), compile(spec["body"], "<test>", "exec")


def _entry_calls(tree, entry):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == entry]


def _visible_ns():
    """The header (HumanEval prompt, MBPP test imports), `_close`, and the standard modules a test names
    without importing them (`sys.getsizeof` in Mbpp/596: before, the test borrowed the program's imports)."""
    import builtins, importlib
    ns = {"__name__": "__visible__"}
    exec(compile(cfg["header"], "<header>", "exec"), ns)
    ns["_close"] = _close
    for t in cfg.get("tests") or []:
        try:
            names = {n.id for n in ast.walk(ast.parse(t)) if isinstance(n, ast.Name)}
        except SyntaxError:
            continue
        for name in sorted(names):
            if name not in ns and not hasattr(builtins, name) and name in sys.stdlib_module_names:
                try:
                    ns[name] = importlib.import_module(name)
                except ImportError:
                    pass
    return ns


_mode, _entry = cfg["mode"], cfg["entry"]
_res = {}
if _mode == "prepare":
    if cfg.get("spec") is not None:
        ns, cases, bind, _ = _call(lambda: _hidden_cases(cfg["spec"]), cfg["call_s"])
        ins = []
        for c in cases:
            ns["__case"] = c
            exec(bind, ns)
            t = enc_text(ns["inp"])
            if t is None:
                raise SystemExit("hidden input too large to send")
            ins.append(t)
        _res["hidden_inputs"] = ins
    if cfg.get("tests") is not None:
        ns = _visible_ns()
        per_test, keys = [], {}
        for t in cfg["tests"]:
            try:
                calls = _entry_calls(ast.parse(t), _entry)
                mine = []
                for c in calls:
                    if c.keywords or any(_entry_calls(a, _entry) for a in c.args):
                        raise ValueError("not replayable")
                    src = "(" + ", ".join(ast.unparse(a) for a in c.args) + ("," if c.args else "") + ")"
                    k = enc_text(_call(lambda: eval(src, ns), cfg["call_s"]))
                    if k is None:
                        raise ValueError("argument too large")
                    mine.append(k)
                    keys.setdefault(k, None)
                per_test.append(mine if calls else None)
            except BaseException:
                per_test.append(None)
        _res["visible_tests"], _res["visible_inputs"] = per_test, list(keys)
    _emit(kind="prepared", **_res)
else:  # grade
    if cfg.get("visible") is not None:
        outs = cfg["visible"]["outs"]
        ns = _visible_ns()

        def _replay(*args, **kw):
            if kw:
                raise _CandidateError("NotReplayable")
            return _value(outs.get(enc_text(args)), cfg["visible"]["missing_err"])

        ns[_entry] = _replay
        cases = []
        for t, calls in zip(cfg["tests"], cfg["visible"]["per_test"]):
            random.seed(0)
            if calls is None:  # no call of the entry point that can be replayed: fails for every program
                cases.append({"ok": False, "err": "NotReplayable"})
                continue
            try:
                _call(lambda: exec(compile(t, "<visible>", "exec"), ns), cfg["call_s"])
                cases.append({"ok": True, "err": None})
            except _CandidateError as e:
                cases.append({"ok": False, "err": str(e)})
            except BaseException as e:
                cases.append({"ok": False, "err": _err(e)})
        _res["visible"] = cases
    if cfg.get("hidden") is not None:
        ns, cases, bind, body = _call(lambda: _hidden_cases(cfg["spec"]), cfg["call_s"])
        outs = {int(k): v for k, v in cfg["hidden"]["outs"].items()}
        h = {"ok": True, "n_pass": len(cases), "n": len(cases), "err": None}
        for i, c in enumerate(cases):
            ns["__case"] = c
            exec(bind, ns)
            random.seed(0)
            try:
                v = _value(outs.get(i), cfg["hidden"]["missing_err"])
                ns["candidate"] = ns[_entry] = (lambda v: lambda *a, **k: v)(v)
                _call(lambda: exec(body, ns), cfg["call_s"])
            except _CandidateError as e:
                h = {"ok": False, "n_pass": i, "n": len(cases), "err": str(e)}
                break
            except BaseException as e:
                h = {"ok": False, "n_pass": i, "n": len(cases), "err": _err(e)}
                break
        _res["hidden"] = h
    _emit(kind="graded", **_res)
_emit(kind="done", ms=round((time.perf_counter() - _t0) * 1000))
_out.close()
os._exit(0)
'''

# ---------- isolation probe ----------

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


def _deny_roots() -> list[str]:
    """Never readable by a candidate: the research data and results, the models, the Hugging Face caches."""
    home = Path.home()
    out = [PHASE0 / "data", PHASE0 / "results", PHASE0.parent / "models", home / ".cache" / "huggingface"]
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
        if os.environ.get(var):
            out.append(Path(os.environ[var]))
    if os.environ.get("XDG_CACHE_HOME"):
        out.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface")
    return [str(p) for p in out]


READ_EXTRA = ["/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/local/lib", "/etc"] if POSIX else []


def isolation() -> dict:
    """What actually isolates the children on this machine (recorded in the execution manifest)."""
    global _ISO
    with _ISO_LOCK:
        if _ISO is None:
            pre = _probe_cached()
            iso = {"version": SANDBOX_VERSION, "platform": sys.platform, "prefix": pre, "landlock": "off"}
            if LINUX:  # what the kernel accepts, measured with a real child
                recs, status, _ = _spawn(CHILD, {"code": "def f():\n    return 1", "entry": "f", "header": "",
                                                 "phase": "visible", "inputs": []}, "visible", sandboxed=True)
                iso["landlock"] = next((r["landlock"] for r in recs if r.get("kind") == "iso"), "probe failed")
            iso.update({
                "network": "unshare --net (no network namespace access)" if pre else "audit hook only",
                "memory": f"RLIMIT_AS, MiB: {MEM_MB}" if POSIX else
                          (f"job object, MiB: {MEM_MB}" if _winjob_ok() else "none"),
                "per_case_timeout": POSIX,
                "filesystem": "audit hook (writes in a fresh temp dir only, no dir_fd, reads in the Python "
                              "installation only)" + (" + Landlock " + iso["landlock"] if LINUX else ""),
                "grading": "separate trusted process; the candidate only receives inputs",
                "limits_s": {k: list(v) for k, v in LIMITS.items()}})  # as stored in JSON (resume)
            _ISO = iso
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
        if self.h:
            self.k32.TerminateJobObject(self.h, 1)

    def close(self):
        if self.h:
            self.k32.CloseHandle(self.h)
            self.h = None


# ---------- running a child: registered, bounded, always killed and reaped ----------

_LIVE: dict[subprocess.Popen, object] = {}
_LIVE_LOCK = threading.Lock()
_STOP = threading.Event()


def shutdown():
    """Start no new child, kill every running one (and its process group or job). Idempotent."""
    _STOP.set()
    with _LIVE_LOCK:
        live = list(_LIVE.items())
    for p, job in live:
        _kill(p, job)


def _env(tmp: str) -> dict:
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": tmp, "TMPDIR": tmp,
           "TEMP": tmp, "TMP": tmp, "USERPROFILE": tmp, "LANG": "C.UTF-8", "PATH": os.environ.get("PATH", os.defpath)}
    if not POSIX:
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _kill(p: subprocess.Popen, job_obj):
    if POSIX:
        import signal
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    elif job_obj is not None:
        try:
            job_obj.kill()
        except OSError:
            pass
    try:
        p.kill()
    except OSError:
        pass


def _exchange(p: subprocess.Popen, data: bytes, timeout: float, kill) -> tuple[bytes, str]:
    """Write `data` to the child's stdin and read its stdout (at most OUT_CAP bytes) until it exits or the
    deadline passes (then `kill()`). Status: ok, timeout, overflow."""
    chunks, size, over = [], [0], [False]

    def write():
        try:
            p.stdin.write(data)
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                p.stdin.close()
            except OSError:
                pass

    def read():
        while True:
            try:
                b = p.stdout.read1(1 << 16)
            except (OSError, ValueError):
                break
            if not b:
                break
            size[0] += len(b)
            if size[0] > OUT_CAP:
                over[0] = True
                break
            chunks.append(b)

    tw = threading.Thread(target=write, daemon=True)
    tr = threading.Thread(target=read, daemon=True)
    tw.start()
    tr.start()
    tr.join(timeout)
    status = "timeout" if tr.is_alive() else ("overflow" if over[0] else "ok")
    if status != "ok":
        kill()
        tr.join(10)
    return b"".join(chunks), status


def _spawn(script: str, job: dict, phase: str, sandboxed: bool) -> tuple[list[dict], str, int]:
    """Run one child; (records, status, wall ms). Status: ok, timeout, overflow (killed), crash (no final
    record). The child is killed and reaped whatever happens, including an exception in this thread."""
    if _STOP.is_set():
        raise SandboxError("sandbox arrêté")
    call_s, total_s = LIMITS[phase]
    iso_prefix = _probe_cached() if sandboxed else []
    tmp = tempfile.mkdtemp(prefix="e11-")
    nonce = secrets.token_hex(8)
    cfg = {**job, "nonce": nonce, "tmp": tmp, "call_s": call_s, "mem_mb": MEM_MB[phase], "cpu_s": int(total_s) + 2,
           "ppid": os.getpid(), "node_budget": NODE_BUDGET, "encode_s": ENCODE_S, "deny": _deny_roots(),
           "read_extra": READ_EXTRA, "landlock": LINUX}
    cmd = iso_prefix + [sys.executable, "-s", "-B", "-c", script]
    t0 = time.perf_counter()
    job_obj, p, out, status = None, None, b"", "crash"
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
        with _LIVE_LOCK:
            _LIVE[p] = job_obj
        if _STOP.is_set():  # shutdown() ran between the check above and the registration
            raise SandboxError("sandbox arrêté")
        out, status = _exchange(p, json.dumps(cfg).encode("utf-8"), total_s, lambda: _kill(p, job_obj))
        p.wait(timeout=30)
    except BaseException:
        if p is not None:
            _kill(p, job_obj)
        raise
    finally:
        if p is not None:
            try:
                if p.poll() is None:
                    _kill(p, job_obj)
                    p.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
            with _LIVE_LOCK:
                _LIVE.pop(p, None)
            for f in (p.stdin, p.stdout):
                try:
                    f.close()
                except (OSError, ValueError):
                    pass
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


_PREFIX: list[str] | None = None


def _probe_cached() -> list[str]:
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = _probe_unshare()
    return _PREFIX


# ---------- the three phases and their grading ----------

def prepare(entry: str, header: str, tests: list[str] | None, spec: dict | None) -> dict:
    """Run once per problem by the trusted grader: the inputs the candidate will receive. Keys: hidden_inputs
    (typed texts, one per hidden case), visible_tests (per visible test, the typed texts of the arguments of
    its calls of the entry point, None if it cannot be replayed), visible_inputs (the distinct ones)."""
    recs, status, _ = _spawn(GRADER, {"mode": "prepare", "entry": entry, "header": header, "tests": tests,
                                      "spec": spec}, "grade", sandboxed=False)
    r = next((x for x in recs if x.get("kind") == "prepared"), None)
    if r is None or status != "ok":
        raise SandboxError(f"préparation impossible ({status})")
    return {k: r.get(k) for k in ("hidden_inputs", "visible_tests", "visible_inputs")}


def _candidate(code: str, entry: str, header: str, phase: str, inputs: list[str]) -> tuple[str | None, dict, str, int]:
    """(load error, {input index: record}, status, ms)."""
    recs, status, ms = _spawn(CHILD, {"code": code, "entry": entry, "header": header, "phase": phase,
                                      "inputs": inputs}, phase, sandboxed=True)
    rec = next((r for r in recs if r.get("kind") == "load"), None)
    load = rec["err"] if rec is not None else ("Timeout" if status == "timeout" else "Crash")
    got = {}
    for r in recs:
        if r.get("kind") == "case" and isinstance(r.get("i"), int) and r["i"] not in got:
            got[r["i"]] = r
    return load, got, status, ms


def _rec(r: dict) -> dict:
    return {k: r[k] for k in ("out", "digest", "err") if k in r}


def _grade(entry: str, header: str, tests, spec, visible: dict | None, hidden: dict | None) -> dict:
    recs, status, _ = _spawn(GRADER, {"mode": "grade", "entry": entry, "header": header, "tests": tests,
                                      "spec": spec, "visible": visible, "hidden": hidden}, "grade", sandboxed=False)
    r = next((x for x in recs if x.get("kind") == "graded"), None)
    if r is None or status != "ok":
        raise SandboxError(f"notation impossible ({status})")
    return r


def evaluate(code: str, entry: str, header: str, tests: list[str], extra_inputs: list[tuple], spec: dict,
             prepared: dict) -> tuple[dict, dict, dict]:
    """The three phases of one program, graded by the trusted grader: (visible, extra, hidden) results."""
    v_load, v_got, v_status, v_ms = _candidate(code, entry, header, "visible", prepared["visible_inputs"])
    x = _extra(code, entry, header, extra_inputs)
    h_load, h_got, h_status, h_ms = _candidate(code, entry, header, "hidden", prepared["hidden_inputs"])
    miss_v = "Timeout" if v_status == "timeout" else (v_load or "Crash")
    miss_h = "Timeout" if h_status == "timeout" else (h_load or "Crash")
    vis_outs = {k: _rec(v_got[i]) for i, k in enumerate(prepared["visible_inputs"]) if i in v_got}
    g = {}
    if v_load is None or h_load is None:
        g = _grade(entry, header, tests, spec,
                   {"outs": vis_outs, "per_test": prepared["visible_tests"], "missing_err": miss_v} if v_load is None else None,
                   {"outs": {str(i): _rec(r) for i, r in h_got.items()}, "missing_err": miss_h} if h_load is None else None)
    if v_load is None:
        cases = g["visible"]
    else:
        cases = [{"ok": False, "err": miss_v} for _ in tests]
    v = {"status": v_status, "load": v_load, "cases": cases, "ms": v_ms}
    if h_load is None:
        hh = g["hidden"]
        h = {"status": h_status, "load": h_load, "pass": bool(hh["ok"]) and h_status == "ok", "n_pass": hh["n_pass"],
             "n": hh["n"], "err": hh["err"], "ms": h_ms}
    else:
        h = {"status": h_status, "load": h_load, "pass": False, "n_pass": 0, "n": len(prepared["hidden_inputs"]),
             "err": miss_h, "ms": h_ms}
    return v, x, h


def _extra(code: str, entry: str, header: str, inputs: list[tuple]) -> dict:
    load, got, status, ms = _candidate(code, entry, header, "extra", [repr(a) for a in inputs])
    return {"status": status, "load": load, "ms": ms,
            "sigs": [got[i]["sig"] if i in got and isinstance(got[i].get("sig"), str) else "!" for i in range(len(inputs))]}


# Convenience entry points (tests, smoke checks): each prepares its own inputs.

def run_visible(code: str, entry: str, header: str, tests: list[str]) -> dict:
    """Pass/fail of each visible test. A test not reached (child killed) fails with Timeout or Crash."""
    prep = prepare(entry, header, tests, None)
    load, got, status, ms = _candidate(code, entry, header, "visible", prep["visible_inputs"])
    miss = "Timeout" if status == "timeout" else (load or "Crash")
    if load is not None:
        return {"status": status, "load": load, "cases": [{"ok": False, "err": miss} for _ in tests], "ms": ms}
    outs = {k: _rec(got[i]) for i, k in enumerate(prep["visible_inputs"]) if i in got}
    g = _grade(entry, header, tests, None, {"outs": outs, "per_test": prep["visible_tests"], "missing_err": miss}, None)
    return {"status": status, "load": load, "cases": g["visible"], "ms": ms}


def run_extra(code: str, entry: str, header: str, inputs: list[tuple]) -> dict:
    """Output signature on each input: 16 hex chars of a hash of the typed value, '!' for no output
    (exception, timeout, crash, or the program does not load)."""
    return _extra(code, entry, header, inputs)


def run_hidden(code: str, entry: str, header: str, spec: dict) -> dict:
    """Grading: passes only if every hidden case passes (first failure stops the run)."""
    prep = prepare(entry, header, None, spec)
    load, got, status, ms = _candidate(code, entry, header, "hidden", prep["hidden_inputs"])
    miss = "Timeout" if status == "timeout" else (load or "Crash")
    if load is not None:
        return {"status": status, "load": load, "pass": False, "n_pass": 0, "n": len(prep["hidden_inputs"]),
                "err": miss, "ms": ms}
    g = _grade(entry, header, None, spec, None, {"outs": {str(i): _rec(r) for i, r in got.items()}, "missing_err": miss})
    hh = g["hidden"]
    return {"status": status, "load": load, "pass": bool(hh["ok"]) and status == "ok", "n_pass": hh["n_pass"],
            "n": hh["n"], "err": hh["err"], "ms": ms}
