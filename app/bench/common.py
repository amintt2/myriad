"""Shared helpers of the benchmarks: percentiles, worker processes driven over stdin/stdout."""
from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def pct(xs: list[float], q: float) -> float | None:
    """q-th percentile (0..100) with linear interpolation; None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def r(x, nd: int = 3):
    return None if x is None else round(x, nd)


class Worker:
    """A child process speaking one JSON line per message on stdout; commands go to its stdin."""

    def __init__(self, name: str, args: list[str], log_path: Path, cwd: Path = APP):
        self.name = name
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-u", *args], cwd=str(cwd), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8",
                                     bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.lines.put(json.loads(line))
                except ValueError:
                    pass
        self.lines.put(None)

    def expect(self, timeout: float) -> dict:
        try:
            msg = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{self.name}: no answer within {timeout:.0f} s (log: {self.log.name})") from None
        if msg is None:
            raise RuntimeError(f"{self.name} exited (code {self.proc.poll()}), see {self.log.name}")
        if msg.get("error"):
            raise RuntimeError(f"{self.name}: {msg['error']}")
        return msg

    def send(self, cmd: dict) -> None:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()

    def call(self, cmd: dict, timeout: float) -> dict:
        self.send(cmd)
        return self.expect(timeout)

    def close(self, timeout: float = 20.0) -> None:
        if self.proc.poll() is None:
            try:
                self.send({"cmd": "quit"})
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self.log.close()


def stdin_commands(loop, q) -> None:
    """Read JSON commands from stdin in a thread and hand them to the asyncio loop."""
    def pump():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except ValueError:
                continue
            loop.call_soon_threadsafe(q.put_nowait, cmd)
        loop.call_soon_threadsafe(q.put_nowait, {"cmd": "quit"})
    threading.Thread(target=pump, daemon=True).start()


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def now() -> float:
    return time.time()
