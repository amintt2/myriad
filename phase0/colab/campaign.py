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
from session_json import identity, lifecycle_lock, receipt_path, save
from budget import remaining, validate_usage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_jobs import PLANS

EXPECTED = {" ".join(j) for j in PLANS["aa-1"]}
WRAPPER = Path(__file__).with_name("colab_phase0.sh").resolve()


class Failure(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


class SessionAbsent(Failure):
    def __init__(self):
        super().__init__("owned session explicitly absent; campaign failed, accounting cleanup authorized")


class AccountingFailure(Failure):
    pass


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
        self.started = time.monotonic()
        self.next_usage = 0
        self.observed_rate = 0
        self.monitoring = False
        self.campaign_id = uuid.uuid4().hex
        self.env = {**os.environ, "DLLM_CAMPAIGN_RECEIPT": str(receipt), "DLLM_CAMPAIGN_ID": self.campaign_id,
                    "DLLM_SUPERVISED": "1", "DLLM_BUDGET_UNITS": str(args.budget_units),
                    "DLLM_CLEANUP_SECONDS": str(args.cleanup_seconds), "DLLM_HOURS": str(args.hours),
                    "DLLM_MIN_BALANCE_UNITS": str(args.min_balance_units),
                    "DLLM_MAX_RATE_UNITS_HOUR": str(args.max_rate_units_hour),
                    "DLLM_ACCOUNTING_MARGIN_SECONDS": str(args.usage_seconds + args.read_seconds + args.grace * 2 + 1)}
        self.owner = None
        self.cleanup_lock_fd = None

    def call(self, operation, seconds, structured=False, deadline=None):
        deadline = min(self.deadline, deadline if deadline is not None else math.inf)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Failure("global deadline exceeded", 124)
        log(f"wrapper {operation}")
        with tempfile.TemporaryFile() as output:
            code = run(["bash", str(WRAPPER), *operation.split()], seconds=min(seconds, remaining),
                       deadline=deadline,
                       grace=self.a.grace, env=self.env, stdout=output if structured else None,
                       monitor=self.monitor if self.monitoring and operation != "usage-json" else None,
                       pass_fds=() if self.cleanup_lock_fd is None else (self.cleanup_lock_fd,))
            if code:
                raise Failure(f"wrapper {operation}: exit {code}", code)
            if structured:
                output.seek(0)
                try:
                    return json.load(output)
                except (ValueError, UnicodeError) as e:
                    raise Failure(f"wrapper {operation}: malformed JSON", 65) from e

    def monitor(self, deadline=None):
        now = time.monotonic()
        if now >= self.next_usage and self.receipt.exists():
            self.next_usage = now + self.a.usage_seconds
            saved = json.loads(self.receipt.read_text(encoding="utf-8"))
            if saved.get("status") == "owned":
                saved = self.cleanup_identity()
                try:
                    value = validate_usage(self.call("usage-json", self.a.read_seconds, True, deadline=deadline), True)
                    seconds, self.observed_rate = remaining(value, saved["budget_policy"],
                        saved["usage_before"]["balance_units"], time.monotonic() - self.started, self.observed_rate)
                except (ValueError, KeyError, Failure) as exc:
                    code = 124 if isinstance(exc, Failure) and exc.code == 124 else 65
                    raise AccountingFailure("accounting unknown or reserve exhausted; cleanup required", code) from exc
                self.deadline = min(self.deadline, time.monotonic() + seconds)
                log("account usage " + json.dumps(value, sort_keys=True))
        return self.deadline - time.monotonic()

    def session(self):
        value = self.call("sessions-json", self.a.read_seconds, True)
        if not isinstance(value, dict) or value.get("schema") != 1 or "phase0" not in value:
            raise Failure("invalid sessions response", 65)
        item = value["phase0"]
        if item is not None and (not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                               str(item.get("fingerprint")))):
            raise Failure("invalid session fingerprint", 65)
        return item

    def cleanup_identity(self):
        try:
            saved = json.loads(self.receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise Failure("allocation ownership unconfirmed; no cloud cleanup authorized", 65) from e
        if saved.get("status") == "foreign-campaign" or saved.get("campaign_id") != self.campaign_id:
            raise Failure("receipt foreign or campaign differs; no recovery authorized", 73)
        pinned = json.dumps(identity(saved), sort_keys=True)
        if "DLLM_OPERATION_IDENTITY" not in self.env:
            self.env["DLLM_OPERATION_IDENTITY"] = pinned
        elif self.env["DLLM_OPERATION_IDENTITY"] != pinned:
            raise Failure("operation receipt replaced; no recovery authorized", 73)
        return saved

    def ownership(self):
        saved = self.cleanup_identity()
        if saved.get("status") != "owned":
            raise Failure("allocation ownership unconfirmed; no recovery authorized", 65)
        if not isinstance(saved.get("phase0"), dict) or not saved["phase0"].get("fingerprint"):
            raise Failure("invalid ownership receipt", 65)
        self.owner = saved["phase0"]
        self.env["DLLM_REQUIRE_OWNER"] = "1"
        current = self.session()
        if current is None:
            raise SessionAbsent()
        if current != self.owner:
            raise Failure("owned session replaced; no cloud mutation authorized", 73)

    def mark_foreign(self):
        saved = json.loads(self.receipt.read_text(encoding="utf-8"))
        pinned = self.env.get("DLLM_OPERATION_IDENTITY")
        if (saved.get("campaign_id") == self.campaign_id
                and (pinned is None or json.dumps(identity(saved), sort_keys=True) == pinned)):
            save(self.receipt, {**saved, "status": "foreign-campaign"})

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
                if isinstance(e, AccountingFailure):
                    raise
                if e.code not in (65, 124) and not str(e).startswith("wrapper"):
                    raise
                last = e
                if attempt + 1 < self.a.read_failures:
                    time.sleep(min(self.a.poll_seconds, self.a.usage_seconds,
                                   max(0, self.deadline - time.monotonic())))
        raise last

    def execute(self):
        code, complete = 0, False
        try:
            if self.session() is not None:
                raise Failure("existing phase0 session refused without touching it", 73)
            self.monitoring = True
            self.call(f"up aa-1 {self.a.gpu}", self.a.up_seconds)
            self.ownership()
            self.next_usage = 0
            self.monitor()
            next_pull = time.monotonic() + self.a.pull_seconds
            while True:
                phase = self.observe()
                if phase == "complete":
                    complete = True
                    break
                if phase == "jobs" and time.monotonic() >= next_pull:
                    for attempt in range(self.a.read_failures):
                        try:
                            self.ownership()
                            self.call("pull", self.a.transfer_seconds)
                            break
                        except Failure as e:
                            if e.code == 73 or isinstance(e, AccountingFailure):
                                raise
                            log(f"periodic recovery pending (attempt {attempt + 1}): {e}")
                            if attempt + 1 < self.a.read_failures:
                                time.sleep(min(self.a.poll_seconds * 2 ** attempt, self.a.usage_seconds,
                                               max(0, self.deadline - time.monotonic())))
                    next_pull = time.monotonic() + self.a.pull_seconds
                time.sleep(min(self.a.poll_seconds, self.a.usage_seconds, max(0, self.deadline - time.monotonic())))
        except Failure as e:
            log(str(e))
            code = e.code
            if code == 73 and self.receipt.exists():
                self.mark_foreign()
        except (KeyboardInterrupt, SystemExit) as e:
            code = e.code if isinstance(e, SystemExit) else 130
            log(f"signal received: exit {code}")
        except Exception as e:
            log(f"supervision error: {type(e).__name__}: {e}")
            code = 1
        finally:
            self.monitoring = False
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
                            self.mark_foreign()
                            raise Failure("preexisting campaign: no recovery or down authorized", 73)
                    for attempt in range(self.a.read_failures):
                        try:
                            self.ownership()
                            self.call("pull final" if complete else "pull", self.a.transfer_seconds)
                            log("final recovery succeeded")
                            break
                        except Failure:
                            if attempt + 1 == self.a.read_failures:
                                raise
                            time.sleep(min(self.a.poll_seconds, max(0, self.deadline - time.monotonic())))
                except Failure as e:
                    log(f"FINAL RECOVERY FAILED: {e}; results not confirmed recovered")
                    code = code or e.code
                self.deadline = cleanup_end
                try:
                    if not self.receipt.exists():
                        raise Failure("no ownership receipt: down not authorized", 73)
                    self.cleanup_identity()
                    self.call("down", max(.01, self.deadline - time.monotonic()))
                    log("owned allocation released")
                except Failure as e:
                    saved = json.loads(self.receipt.read_text()) if self.receipt.exists() else {}
                    pending = saved.get("status") == "released/accounting-pending"
                    detail = "release confirmed, accounting pending" if pending else "VM may still be billed"
                    log(f"FINAL DOWN FAILED: {e}; {detail}; receipt retained at {self.receipt}; "
                        "resume with --cleanup-only")
                    code = code or e.code
        if not complete:
            code = code or 1
        log("SUCCESS: 15 jobs succeeded, final pull and down confirmed" if code == 0 else f"FAILED: exit {code}")
        return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", choices=("auto", "L4", "A100", "H100", "T4"), default="auto")
    ap.add_argument("--budget-units", type=float, default=float(os.environ.get("DLLM_BUDGET_UNITS", "nan")))
    ap.add_argument("--min-balance-units", type=float, default=float(os.environ.get("DLLM_MIN_BALANCE_UNITS", "15")))
    ap.add_argument("--max-rate-units-hour", type=float, default=5.3)
    ap.add_argument("--usage-seconds", type=float, default=30)
    ap.add_argument("--receipt", type=Path, default=receipt_path())
    ap.add_argument("--cleanup-only", action="store_true")
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
    times = (v for k, v in vars(a).items() if k not in
             ("lock", "receipt", "cleanup_only", "gpu", "budget_units", "min_balance_units",
              "max_rate_units_hour", "usage_seconds"))
    if any(not math.isfinite(v) or v <= 0 for v in times):
        ap.error("all time limits and retry counts must be finite and positive")
    if not a.cleanup_only and (not math.isfinite(a.budget_units) or a.budget_units <= 0):
        ap.error("--budget-units must explicitly reserve sufficient compute units")
    if not a.cleanup_only and (not math.isfinite(a.min_balance_units) or a.min_balance_units < 0):
        ap.error("--min-balance-units must be finite and nonnegative")
    if not a.cleanup_only and any(not math.isfinite(v) or v <= 0 for v in (a.max_rate_units_hour, a.usage_seconds)):
        ap.error("rate reservation and usage interval must be finite and positive")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with a.lock.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another local campaign holds the lock")
            return 73
        supervisor = Supervisor(a, a.receipt)
        if a.cleanup_only:
            supervisor.deadline = time.monotonic() + a.cleanup_seconds
            try:
                # Explicit adoption is local, under the same lock retained through wrapper/down.
                with lifecycle_lock(a.receipt) as receipt_lock:
                    saved = json.loads(a.receipt.read_text(encoding="utf-8"))
                    supervisor.campaign_id = saved.get("campaign_id")
                    if supervisor.campaign_id is None:
                        supervisor.env.pop("DLLM_CAMPAIGN_ID", None)
                    else:
                        supervisor.env["DLLM_CAMPAIGN_ID"] = supervisor.campaign_id
                    supervisor.env["DLLM_OPERATION_IDENTITY"] = json.dumps(identity(saved), sort_keys=True)
                    supervisor.cleanup_lock_fd = receipt_lock.fileno()
                    supervisor.env["DLLM_LIFECYCLE_LOCK_FD"] = str(supervisor.cleanup_lock_fd)
                    log("cleanup-only explicitly adopted the selected receipt under lifecycle lock")
                    supervisor.call("down", a.cleanup_seconds)
                return 0
            except (PermissionError, OSError, ValueError) as exc:
                log(f"cleanup-only adoption refused: {exc}")
                return 73 if isinstance(exc, PermissionError) else 65
            except Failure as exc:
                saved = json.loads(a.receipt.read_text()) if a.receipt.exists() else {}
                detail = ("release confirmed, accounting pending" if saved.get("status") == "released/accounting-pending"
                          else "ownership unresolved")
                log(f"CLEANUP PENDING: {exc}; {detail}; receipt retained at {a.receipt}")
                return exc.code
        if a.receipt.exists():
            log(f"persistent ownership requires --cleanup-only before allocation: {a.receipt}")
            return 73
        return supervisor.execute()


if __name__ == "__main__":
    sys.exit(main())
