"""Isolated public-benchmark unittest worker, without adversarial result attestation."""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import types
import unittest
from pathlib import Path

PROTOCOL = "bigcode-unittest-v1"
CAP = 1 << 20


def worker(raw: str, fd: int) -> None:
    cfg = json.loads(raw)
    diagnostics = sys.stdout
    records = []

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            records.append({"test": test.id().split(".")[-1], "complete": False})
            super().startTest(test)

        def stopTest(self, test):
            records[-1]["complete"] = True
            super().stopTest(test)

    out = {"protocol": PROTOCOL, "phase": "load", "candidate_failure": None}
    # The supervisor owns the result channel. The worker still shares it with arbitrary Python code;
    # it is transport separation, never authentication against a deliberately malicious candidate.
    try:
        try:
            import solution
            function = getattr(solution, cfg["entry_point"])
            if not callable(function):
                raise TypeError("entry point must be callable")
        except BaseException as error:
            out["candidate_failure"] = type(error).__name__
        else:
            out["phase"] = "collection"
            try:
                module = types.ModuleType("official")
                module.__dict__[cfg["entry_point"]] = function
                text = Path("_official.py").read_text(encoding="utf-8")
                exec(compile(text, "_official.py", "exec"), module.__dict__)
                suite = unittest.defaultTestLoader.loadTestsFromModule(module)
                if suite.countTestCases() != cfg["tests_expected"]:
                    raise ValueError("official test collection mismatch")
                out["phase"] = "tests"
                result = unittest.TextTestRunner(stream=diagnostics, failfast=False, resultclass=Result).run(suite)
                out.update(phase="tests", tests=result.testsRun, calls=records,
                           failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                           expected_failures=len(result.expectedFailures),
                           unexpected_successes=len(result.unexpectedSuccesses),
                           feedback=[{"test": t.id().split(".")[-1], "traceback": detail[-4000:]}
                                     for t, detail in result.failures + result.errors])
            except BaseException as error:
                key = "collection_error" if out["phase"] == "collection" else "candidate_failure"
                out[key] = type(error).__name__
    finally:
        diagnostics.flush()
    payload = json.dumps(out, allow_nan=False).encode()
    while payload:
        payload = payload[os.write(fd, payload):]
    os._exit(0)


def supervise(raw: str, *, worker_path: str | None = None) -> dict:
    read_fd, write_fd = None, None
    try:
        read_fd, write_fd = os.pipe()
        child = subprocess.Popen([sys.executable, worker_path or __file__, "--worker", str(write_fd)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 close_fds=True, pass_fds=(write_fd,))
    except Exception as error:
        if read_fd is not None:
            os.close(read_fd)
        return {"preparation_error": type(error).__name__}
    finally:
        if write_fd is not None:
            os.close(write_fd)
    buffers = {"response": bytearray(), "diagnostics": bytearray()}
    limited = False
    try:
        child.stdin.write(raw.encode())
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
            **{k: bytes(v[:CAP]).decode("utf-8", errors="replace") for k, v in buffers.items()}}


if __name__ == "__main__":
    incoming = sys.stdin.read(CAP + 1)
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(incoming, int(sys.argv[2]))
    else:
        print(json.dumps(supervise(incoming), allow_nan=False))
