"""Exclusive E12 supervisor. Every cloud operation goes through the versioned bash wrapper."""
import argparse
import fcntl
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

from read_timeout import run
from snapshot import safe_snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_jobs import PLANS

EXPECTED = {" ".join(j) for j in PLANS["aa-1"]}
WRAPPER = Path(__file__).with_name("colab_phase0.sh").resolve()


class Failure(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def state(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema") != 1:
        raise Failure("invalid snapshot", 65)
    boot, jobs, diag = (snapshot.get(k) for k in ("bootstrap", "jobs", "diagnostics"))
    if not isinstance(boot, dict) or not isinstance(boot.get("step"), str):
        raise Failure("missing bootstrap state", 65)
    step = boot["step"]
    if step.startswith("refusé"):
        raise Failure("bootstrap refused an existing campaign", 73)
    if step.startswith("ÉCHEC"):
        raise Failure("bootstrap failed")
    if step != "lanceur en marche" or boot.get("plan") != "aa-1" or type(boot.get("pid")) is not int:
        raise Failure("bootstrap did not acknowledge aa-1", 65)
    if not isinstance(diag, dict) or type(diag.get("launcher_alive")) is not bool:
        raise Failure("missing launcher liveness", 65)
    if jobs is None:
        if not diag["launcher_alive"]:
            raise Failure("launcher exited during downloads/preflight")
        return "downloads/preflight"
    if not isinstance(jobs, dict) or set(jobs) != EXPECTED or not all(isinstance(v, str) for v in jobs.values()):
        raise Failure("incomplete or unexpected jobs state", 65)
    if any(v.startswith(("ÉCHEC", "annulé")) for v in jobs.values()):
        raise Failure("job failed or cancelled")
    if all(re.fullmatch(r"fini en [0-9]+\.[0-9]+ min", v) for v in jobs.values()):
        return "complete"
    allowed = {"prévu", "en attente de mémoire GPU", "en cours"}
    if any(v not in allowed and not re.fullmatch(r"fini en [0-9]+\.[0-9]+ min", v) for v in jobs.values()):
        raise Failure("unknown job state", 65)
    if not diag["launcher_alive"]:
        raise Failure("launcher exited with unfinished jobs")
    return "jobs"


class Supervisor:
    def __init__(self, args, receipt):
        self.a, self.receipt = args, receipt
        self.deadline = time.monotonic() + args.hours * 3600
        self.campaign_id = uuid.uuid4().hex
        self.env = {**os.environ, "DLLM_CAMPAIGN_RECEIPT": str(receipt), "DLLM_CAMPAIGN_ID": self.campaign_id,
                    "DLLM_SUPERVISED": "1"}
        self.owner = None

    def call(self, operation, seconds, structured=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise Failure("global deadline exceeded", 124)
        log(f"wrapper {operation}")
        with tempfile.TemporaryFile() as output:
            code = run(["bash", str(WRAPPER), *operation.split()], seconds=min(seconds, remaining),
                       grace=self.a.grace, env=self.env, stdout=output if structured else None)
            if code:
                raise Failure(f"wrapper {operation}: exit {code}", code)
            if structured:
                output.seek(0)
                try:
                    return json.load(output)
                except (ValueError, UnicodeError) as e:
                    raise Failure(f"wrapper {operation}: malformed JSON", 65) from e

    def session(self):
        value = self.call("sessions-json", self.a.read_seconds, True)
        if not isinstance(value, dict) or value.get("schema") != 1 or "phase0" not in value:
            raise Failure("invalid sessions response", 65)
        item = value["phase0"]
        if item is not None and (not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                               str(item.get("fingerprint")))):
            raise Failure("invalid session fingerprint", 65)
        return item

    def ownership(self):
        if self.owner is None:
            try:
                self.owner = json.loads(self.receipt.read_text())["phase0"]
            except (OSError, ValueError, KeyError) as e:
                raise Failure("allocation ownership unconfirmed; no cloud cleanup authorized", 65) from e
            if not isinstance(self.owner, dict) or not self.owner.get("fingerprint"):
                raise Failure("invalid ownership receipt", 65)
        if self.session() != self.owner:
            raise Failure("owned session absent or replaced; no cloud mutation authorized", 73)

    def observe(self):
        last = None
        for attempt in range(self.a.read_failures):
            try:
                self.ownership()
                snap = self.call("snapshot", self.a.read_seconds, True)
                if isinstance(snap, dict) and snap.get("campaign_present") and snap.get("bootstrap") is None:
                    raise Failure("preexisting campaign has no owned bootstrap", 73)
                if isinstance(snap, dict) and isinstance(snap.get("bootstrap"), dict):
                    if snap["bootstrap"].get("campaign_id") != self.campaign_id:
                        raise Failure("existing campaign bootstrap was not ours", 73)
                log(json.dumps(safe_snapshot(snap), ensure_ascii=False))
                result = state(snap)
                return result
            except Failure as e:
                log(str(e))
                if e.code not in (65, 124) and not str(e).startswith("wrapper"):
                    raise
                last = e
                if attempt + 1 < self.a.read_failures:
                    time.sleep(min(self.a.poll_seconds, max(0, self.deadline - time.monotonic())))
        raise last

    def execute(self):
        code, complete = 0, False
        try:
            if self.session() is not None:
                raise Failure("existing phase0 session refused without touching it", 73)
            self.call("up aa-1 A100", self.a.up_seconds)
            self.ownership()
            next_pull = time.monotonic() + self.a.pull_seconds
            while True:
                phase = self.observe()
                if phase == "complete":
                    complete = True
                    break
                if phase == "jobs" and time.monotonic() >= next_pull:
                    self.ownership()
                    self.call("pull", self.a.transfer_seconds)
                    next_pull = time.monotonic() + self.a.pull_seconds
                time.sleep(min(self.a.poll_seconds, max(0, self.deadline - time.monotonic())))
        except Failure as e:
            log(str(e))
            code = e.code
            if code == 73:
                self.receipt.unlink(missing_ok=True)
        except (KeyboardInterrupt, SystemExit) as e:
            code = e.code if isinstance(e, SystemExit) else 130
            log(f"signal received: exit {code}")
        except Exception as e:
            log(f"supervision error: {type(e).__name__}: {e}")
            code = 1
        finally:
            # Ignore repeated signals during bounded recovery; cleanup has its own finite budget.
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            self.deadline = time.monotonic() + self.a.cleanup_seconds
            if self.receipt.exists():
                cleanup_end = self.deadline
                self.deadline -= min(self.a.cleanup_seconds / 2, 3 * (self.a.read_seconds + self.a.grace))
                try:
                    self.ownership()
                    try:
                        snap = self.call("snapshot", self.a.read_seconds, True)
                    except Failure as e:
                        log(f"cleanup snapshot failed: {e}")
                        code = code or e.code
                        snap = None
                    if isinstance(snap, dict):
                        boot = snap.get("bootstrap")
                        foreign = boot is None and snap.get("campaign_present")
                        foreign = foreign or isinstance(boot, dict) and boot.get("campaign_id") != self.campaign_id
                        if foreign:
                            self.receipt.unlink(missing_ok=True)
                            raise Failure("preexisting campaign: no recovery or down authorized", 73)
                    self.call("pull final" if complete else "pull", self.a.transfer_seconds)
                    log("final recovery succeeded")
                except Failure as e:
                    log(f"FINAL RECOVERY FAILED: {e}; results not confirmed recovered")
                    code = code or e.code
                self.deadline = cleanup_end
                try:
                    if not self.receipt.exists():
                        raise Failure("no ownership receipt: down not authorized", 73)
                    self.ownership()
                    self.call("down", self.a.read_seconds)
                    if self.session() is not None:
                        raise Failure("down returned but phase0 is still present")
                    log("owned allocation released")
                except Failure as e:
                    log(f"FINAL DOWN FAILED: {e}")
                    code = code or e.code
        if not complete:
            code = code or 1
        log("SUCCESS: 15 jobs succeeded, final pull and down confirmed" if code == 0 else f"FAILED: exit {code}")
        return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--poll-seconds", type=float, default=60)
    ap.add_argument("--pull-seconds", type=float, default=600)
    ap.add_argument("--read-seconds", type=float, default=120)
    ap.add_argument("--up-seconds", type=float, default=2400)
    ap.add_argument("--transfer-seconds", type=float, default=900)
    ap.add_argument("--cleanup-seconds", type=float, default=1800)
    ap.add_argument("--grace", type=float, default=5)
    ap.add_argument("--read-failures", type=int, default=3)
    ap.add_argument("--lock", type=Path, default=Path("/tmp/myriad-phase0-campaign.lock"))
    a = ap.parse_args()
    if any(not math.isfinite(v) or v <= 0 for k, v in vars(a).items() if k != "lock"):
        ap.error("all time limits and retry counts must be finite and positive")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with a.lock.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another local campaign holds the lock")
            return 73
        with tempfile.TemporaryDirectory(prefix="myriad-e12-") as tmp:
            return Supervisor(a, Path(tmp) / "allocation.json").execute()


if __name__ == "__main__":
    sys.exit(main())
