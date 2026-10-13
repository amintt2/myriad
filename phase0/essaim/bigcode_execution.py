"""Separate, bounded supervisor for an untrusted Python worker inside the existing Linux sandbox.

The pipe and completion exit code reject file/stdout forgeries and premature exit(0). They are not an
inviolable attestation: arbitrary code sharing the worker can inspect its descriptors and Python frames.
"""
from __future__ import annotations

import importlib
import json
import os
import selectors
import subprocess
import sys

PROTOCOL = "bigcode-rpc-v3"
COMPLETED = 73  # Public completion marker, not a secret or proof against arbitrary native candidate code.
CAP = 1 << 20


def arguments(raw, decode):
    cfg = json.loads(raw)
    if type(cfg) is not dict or set(cfg) != {"args", "kwargs"}:
        raise ValueError("invalid input schema")
    args, kwargs = decode(cfg["args"]), decode(cfg["kwargs"])
    if type(args) is not tuple or type(kwargs) is not dict:
        raise TypeError("invalid argument containers")
    return args, kwargs


def worker(raw, fd, decode, encode):
    # Capture genuine builtins and termination/serialization functions before importing any candidate.
    def run(base=BaseException, value_error=ValueError, kind=type, subclass=issubclass,
            serialize=json.dumps, write=os.write, finish=os._exit, encoder=encode,
            importer=importlib.import_module, attr=getattr, callable_test=callable,
            protocol=PROTOCOL, completed=COMPLETED):
        phase = "arguments"
        try:
            args, kwargs = arguments(raw, decode)
            phase = "load"
            fn = attr(importer("solution"), "task_func")
            if not callable_test(fn):
                raise TypeError("task_func must be callable")
            phase = "call"
            value = fn(*args, **kwargs)
            phase = "encode"
            out = {"protocol": protocol, "phase": phase, "value": encoder(value)}
        except base as error:
            # Class ancestry, not a candidate-supplied class name or __class__ property.
            exception_kind = "value_error" if subclass(kind(error), value_error) else "other"
            out = {"protocol": protocol, "phase": phase, "exception": kind(error).__name__[:128],
                   "exception_kind": exception_kind, "message": str(error)[:4096]}
        payload = serialize(out, allow_nan=False).encode("utf-8")
        while payload:
            payload = payload[write(fd, payload):]
        finish(completed)
    run()


def supervise(raw, decode):
    # Only failures before the child exists are preparation errors on the supervisor's own channel.
    try:
        if len(raw) > CAP:
            raise ValueError("input limit")
        arguments(raw, decode)
        read_fd, write_fd = os.pipe()
        try:
            child = subprocess.Popen([sys.executable, __file__, "--worker", str(write_fd)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     close_fds=True, pass_fds=(write_fd,))
        except Exception:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
    except Exception as error:
        return {"preparation_error": type(error).__name__}
    buffers = {"response": bytearray(), "diagnostics": bytearray()}
    limited = False
    try:
        child.stdin.write(raw.encode("utf-8"))
        child.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(read_fd, selectors.EVENT_READ, "response")
            selector.register(child.stdout, selectors.EVENT_READ, "diagnostics")
            selector.register(child.stderr, selectors.EVENT_READ, "diagnostics")
            while selector.get_map():
                for key, _ in selector.select():
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(chunk)
                    if sum(map(len, buffers.values())) > CAP:
                        limited = True
                        break
                if limited:
                    child.kill()
                    break
        code = child.wait()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        os.close(read_fd)
        child.stdout.close()
        child.stderr.close()
    return {"worker_exit": code, "limited": limited,
            **{key: bytes(value[:CAP]).decode("utf-8", errors="replace") for key, value in buffers.items()}}


if __name__ == "__main__":
    # This file and codec are copied from trusted sources before any candidate process is launched.
    from _codec import decode, encode
    incoming = sys.stdin.read(CAP + 1)
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(incoming, int(sys.argv[2]), decode, encode)
    else:
        print(json.dumps(supervise(incoming, decode), allow_nan=False))
