"""Bound a local Colab read, including connection setup, without touching the remote campaign.

The supervisor stays outside the CLI's process group, so it can kill all descendants even when they
ignore TERM. Inherit all streams unchanged; return the CLI status, or 124 after the wall-clock deadline.
"""
import contextlib
import contextvars
import math
import os
import signal
import subprocess
import sys
import time


WATCHERS = contextvars.ContextVar("rpc_expiry_watchers", default=())


@contextlib.contextmanager
def watch_expiry(deadline, expire):
    token = WATCHERS.set((*WATCHERS.get(), (deadline, expire)))
    try:
        yield
    finally:
        WATCHERS.reset(token)


def watched_wait(seconds):
    # Account reads and backoff drive enclosing RPC expiry without inheriting its deadline.
    watchers = WATCHERS.get()
    for _, expire in watchers:
        expire()
    now = time.monotonic()
    future = [deadline - now for deadline, _ in watchers if deadline > now]
    return min(seconds, 1, *future) if watchers else seconds


def pause(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        delay = watched_wait(end - time.monotonic())
        # Reaping may consume time; never extend the account pause or window by that grace.
        delay = min(delay, end - time.monotonic())
        if delay > 0:
            time.sleep(delay)
    watched_wait(0)


def killgroup(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run(cmd, seconds=90, grace=5, interrupt_term=True, monitor=None, deadline=None, monitor_expiry=False, **streams):
    if not all(math.isfinite(v) and v > 0 for v in (seconds, grace)):
        raise ValueError("invalid process deadline")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("invalid absolute deadline")
    deadline = min(time.monotonic() + seconds, deadline if deadline is not None else math.inf)
    if time.monotonic() >= deadline:
        return 124
    p = subprocess.Popen(cmd, start_new_session=True, **streams)
    expired = reaped = False
    def reap(term=True):
        nonlocal reaped
        if reaped:
            return
        if term:
            killgroup(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        killgroup(p.pid, signal.SIGKILL)
        p.wait()
        reaped = True
    def expire():
        nonlocal expired
        if not expired and time.monotonic() >= deadline:
            expired = True
            reap()
    try:
        try:
            while True:
                watched_wait(0)
                limit = deadline
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(cmd, seconds)
                if monitor is not None:
                    if monitor_expiry:
                        with watch_expiry(deadline, expire):
                            available = monitor(deadline, expire)
                    else:
                        available = monitor(deadline)
                    limit = min(deadline, time.monotonic() + available)
                remaining = limit - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, seconds)
                try:
                    wait = watched_wait(min(1, remaining) if monitor is not None else remaining)
                    wait = min(wait, limit - time.monotonic())
                    if wait <= 0:
                        raise subprocess.TimeoutExpired(cmd, seconds)
                    code = p.wait(timeout=wait)
                    if time.monotonic() >= limit:
                        raise subprocess.TimeoutExpired(cmd, seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= limit or monitor is None and not WATCHERS.get():
                        raise
            killgroup(p.pid, signal.SIGKILL)  # reap descendants even after an early successful parent exit
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            reap()  # also kill descendants that outlive the CLI
            return 124
    except BaseException:
        reap(interrupt_term)
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(run(sys.argv[1:]))
