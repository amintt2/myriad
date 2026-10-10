"""Bound a local Colab read, including connection setup, without touching the remote campaign.

The supervisor stays outside the CLI's process group, so it can kill all descendants even when they
ignore TERM. Inherit all streams unchanged; return the CLI status, or 124 after the wall-clock deadline.
"""
import math
import os
import signal
import subprocess
import sys


def killgroup(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run(cmd, seconds=90, grace=5, interrupt_term=True, **streams):
    if not all(math.isfinite(v) and v > 0 for v in (seconds, grace)):
        raise ValueError("invalid process deadline")
    p = subprocess.Popen(cmd, start_new_session=True, **streams)
    try:
        try:
            code = p.wait(timeout=seconds)
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
