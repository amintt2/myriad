"""Bound a local Colab read, including connection setup, without touching the remote campaign.

The supervisor stays outside the CLI's process group, so it can kill all descendants even when they
ignore TERM. Inherit all streams unchanged; return the CLI status, or 124 after the wall-clock deadline.
"""
import math
import os
import signal
import subprocess
import sys
import time


def killgroup(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run(cmd, seconds=90, grace=5, interrupt_term=True, monitor=None, deadline=None, **streams):
    if not all(math.isfinite(v) and v > 0 for v in (seconds, grace)):
        raise ValueError("invalid process deadline")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("invalid absolute deadline")
    deadline = min(time.monotonic() + seconds, deadline if deadline is not None else math.inf)
    if time.monotonic() >= deadline:
        return 124
    p = subprocess.Popen(cmd, start_new_session=True, **streams)
    try:
        try:
            while True:
                limit = deadline
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(cmd, seconds)
                if monitor is not None:
                    available = monitor(deadline)
                    limit = min(deadline, time.monotonic() + available)
                remaining = limit - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, seconds)
                try:
                    code = p.wait(timeout=min(1, remaining) if monitor is not None else remaining)
                    if time.monotonic() >= limit:
                        raise subprocess.TimeoutExpired(cmd, seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= limit or monitor is None:
                        raise
            killgroup(p.pid, signal.SIGKILL)  # reap descendants even after an early successful parent exit
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            killgroup(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
            killgroup(p.pid, signal.SIGKILL)  # also kill descendants that outlive the CLI
            p.wait()
            return 124
    except BaseException:
        if interrupt_term:
            killgroup(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        killgroup(p.pid, signal.SIGKILL)
        p.wait()
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(run(sys.argv[1:]))
