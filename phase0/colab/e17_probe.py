"""One E17 Colab compatibility attempt, with in-memory cleanup authority independent of persistence."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

import session_json as lifecycle
from budget import UsageTimeout, account_deadline, preflight, read_account, remaining, validate_usage
from e17_probe_remote import FLAGS, NAMESPACES
from read_timeout import run

HERE = Path(__file__).resolve().parent


class RpcExpiredAfterAccountWait(RuntimeError):
    """The account recovered, but the enclosing RPC response is no longer admissible."""


@contextlib.contextmanager
def defer_signals():
    # A confirmed new must be recorded in memory before TERM can enter finally, even if save fails.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def cli_signals():
    # The single-threaded controller defers signals, but safe_cli must handle TERM to reap its own group.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})


def validate_result(value, request):
    if request.get("sandbox_backend", "strict") not in ("strict", "ephemeral-vm"):
        raise ValueError("unknown probe backend")
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in request.items()):
        raise ValueError("probe acknowledgement identity missing or foreign")
    if (type(value.get("schema")) is not int or value["schema"] != 1
            or value.get("origin") != "colab-exec" or value.get("namespace_flags") != FLAGS
            or value.get("verdict") not in ("compatible", "incompatible", "infrastructure-unknown")
            or not isinstance(value.get("kernel"), str)):
        raise ValueError("invalid probe acknowledgement")
    if value["verdict"] == "compatible":
        abi, smoke = value.get("landlock_abi"), value.get("smoke")
        ephemeral = request.get("sandbox_backend", "strict") == "ephemeral-vm"
        markers = value.get("colab_markers")
        ready = {"isolation_ready": "ephemeral-vm-v1" if ephemeral else "linux-shell-v2",
                 "landlock_abi": None if ephemeral else abi, "seccomp": "on",
                 "namespaces": NAMESPACES}
        if (value.get("system") != "Linux" or value.get("machine") != "x86_64"
                or type(abi) is not int or (not ephemeral and abi < 3)
                or (ephemeral and (not isinstance(markers, dict)
                    or set(markers) != {"hostname_file", "drive_binary"}
                    or any(v is not True for v in markers.values())))
                or not isinstance(value.get("namespaces"), dict)
                or set(value["namespaces"]) != set(NAMESPACES)
                or any(v is not True for v in value["namespaces"].values())
                or value.get("namespace_combined") is not True or value.get("seccomp") is not True
                or not isinstance(smoke, dict) or type(smoke.get("code")) is not int or smoke["code"] != 0
                or smoke.get("output_ok") is not True or smoke.get("isolation") != ready):
            raise ValueError("compatible verdict lacks required observations")
    return value


class Probe:
    def __init__(self, args):
        self.a = args
        self.started = time.monotonic()
        self.deadline = self.started + args.seconds
        self.campaign = uuid.uuid4().hex
        self.policy = {"budget": args.budget_units, "minimum": args.min_balance_units,
                       "hours": args.seconds / 3600, "cleanup": args.cleanup_seconds,
                       "rate": args.max_rate_units_hour, "margin": args.read_seconds + args.grace * 2 + 31}
        self.pending = self.owner = self.before = None
        self.new_confirmed = self.absent = self.closed = False
        self.foreign = False
        self.next_usage = 0
        self.rate = args.max_rate_units_hour
        self.observations = []
        self.last_usage = None
        self.last_usage_at = self.started
        self.account_recoveries = 0
        self.account_waiting = False

    def cli(self, args, monitor=False):
        recoveries = self.account_recoveries
        rpc_deadline = min(self.deadline, time.monotonic() + self.operation_seconds(args))
        available = self.deadline - time.monotonic()
        if args[0] == "usage":
            available -= self.a.grace  # Reaping belongs inside the account window, including its final read.
            if available <= 0:
                raise UsageTimeout("colab usage: exit 124")
        if available <= 0:
            raise TimeoutError("operation deadline exceeded")
        with tempfile.TemporaryFile() as output:
            code = run([sys.executable, str(HERE / "safe_cli.py"), *args],
                       seconds=min(self.operation_seconds(args), available),
                       grace=self.a.grace, deadline=rpc_deadline, stdout=output,
                       monitor=self.monitor if monitor else None,
                       monitor_expiry=monitor,
                       preexec_fn=cli_signals,
                       env={**os.environ, "DLLM_REQUIRE_OWNER": "0"},
                       pass_fds=(int(os.environ["DLLM_LIFECYCLE_LOCK_FD"]),)
                       if os.environ.get("DLLM_LIFECYCLE_LOCK_FD") else ())
            output.seek(0)
            text = output.read().decode("utf-8")
        if code:
            if args[0] == "usage" and code == 124:
                raise UsageTimeout("colab usage: exit 124")
            if code == 124 and monitor and self.account_recoveries > recoveries:
                raise RpcExpiredAfterAccountWait(f"colab {args[0]}: exit 124")
            raise RuntimeError(f"colab {args[0]}: exit {code}")
        return text

    def operation_seconds(self, args):
        return self.a.read_seconds if args[0] != "exec" else 60

    def session(self):
        return lifecycle.parse(self.cli(["sessions"]))["phase0"]

    def usage(self, allocated):
        observed_at = time.monotonic()
        value = validate_usage(lifecycle.usage(self.cli(["usage"])), allocated)
        self.observations.append(value)
        self.last_usage, self.last_usage_at = value, observed_at
        return value

    def account_window_deadline(self):
        return self.deadline

    def wait_usage(self, allocated, first_deadline=None, expire=None):
        old = self.deadline
        end = self.account_window_deadline()
        first, waiting = True, False
        if allocated:
            self.account_waiting = True
        def read(limit):
            nonlocal first, waiting
            self.deadline = min(limit, first_deadline) if first and first_deadline is not None else limit
            first = False
            try:
                return self.usage(allocated)
            except UsageTimeout:
                waiting = True
                raise
            finally:
                if expire is not None:
                    expire()
        def allowance():
            value = {**(self.last_usage or self.before), "assignments": 1,
                     "rate_units_hour": max(self.rate, (self.last_usage or self.before)["rate_units_hour"])}
            return account_deadline(value, self.last_usage_at, self.policy, self.before["balance_units"],
                                    self.started, self.rate)
        try:
            value = read_account(read, end, allowance if allocated else None, self.a.retry_seconds)
            self.account_recoveries += int(waiting)
            if allocated:
                self.account_waiting = False
            return value
        finally:
            self.deadline = old

    def monitor(self, deadline=None, expire=None):
        if self.account_waiting:
            raise ValueError("account wait interrupted or exhausted; only cleanup is allowed")
        if time.monotonic() >= self.next_usage:
            self.next_usage = time.monotonic() + self.a.usage_seconds
            old = self.deadline
            try:
                value = self.wait_usage(True, deadline, expire)
                if value["rate_units_hour"] > self.a.max_rate_units_hour:
                    raise ValueError("observed rate exceeds reserved ceiling")
                seconds, self.rate = remaining(value, self.policy, self.before["balance_units"],
                                               time.monotonic() - self.started, self.rate)
            finally:
                self.deadline = old
            self.deadline = min(self.deadline, time.monotonic() + seconds)
        return self.deadline - time.monotonic()

    def check_receipt(self):
        saved = json.loads(self.a.receipt.read_text(encoding="utf-8"))
        if (any(saved.get(k) != self.pending[k] for k in ("campaign_id", "allocation_id", "plan"))
                or saved.get("status") == "foreign-campaign"
                or saved.get("phase0") not in (None, self.owner)):
            raise PermissionError("foreign or replaced receipt: no stop authorized")

    def reconcile(self):
        if self.foreign:
            raise PermissionError("foreign campaign: no stop authorized")
        self.check_receipt()
        current = self.session()
        if current is not None and self.owner is None:
            if not self.new_confirmed or current["hardware"] != self.gpu:
                raise PermissionError("allocation ambiguous: no stop authorized")
            self.owner = current
        if current is not None and current != self.owner:
            raise PermissionError("foreign session: no stop authorized")
        return current

    def cleanup(self):
        self.deadline = time.monotonic() + self.a.cleanup_seconds
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        while time.monotonic() < self.deadline:
            try:
                current = None if self.absent else self.reconcile()
                if current is None:
                    self.absent = True
                    after = self.usage(False)  # Always required, including when saving the receipt failed.
                    lifecycle.usage_record(self.a.receipt, self.before, after, campaign_id=self.campaign,
                                           release_confirmed=True)
                    self.check_receipt()
                    self.a.receipt.unlink()
                    self.closed = True
                    return
                self.cli(["stop", "-s", "phase0"])
            except PermissionError:
                raise
            except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                print(f"cleanup pending: {type(exc).__name__}", flush=True)
            time.sleep(min(self.a.retry_seconds, max(0, self.deadline - time.monotonic())))
        self.check_receipt()
        saved = json.loads(self.a.receipt.read_text(encoding="utf-8"))
        lifecycle.save(self.a.receipt, {**saved, "cleanup_blocked": True, "release_confirmed": self.absent})
        raise TimeoutError("cleanup/accounting pending; receipt retained")

    gpu = "L4"
    plan = "e17-probe"

    def prepare(self):
        pass

    def perform(self):
        source = (HERE.parent / "essaim/local_linux.py").read_text(encoding="utf-8")
        request = {"campaign_id": self.campaign, "allocation_id": self.pending["allocation_id"],
                   "phase0": self.owner, "nonce": uuid.uuid4().hex,
                   "sandbox_backend": getattr(self.a, "sandbox_backend", "strict"),
                   "local_linux_sha256": hashlib.sha256(source.encode()).hexdigest()}
        payload = (HERE / "e17_probe_remote.py").read_text(encoding="utf-8")
        request["probe_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        ephemeral_source = None
        if getattr(self.a, "sandbox_backend", "strict") == "ephemeral-vm":
            ephemeral_source = (HERE.parent / "essaim/ephemeral_vm.py").read_text(encoding="utf-8")
            request.update(sandbox_backend="ephemeral-vm",
                           ephemeral_vm_sha256=hashlib.sha256(ephemeral_source.encode()).hexdigest())
        payload += ("\nprint('E17_PROBE ' + json.dumps("
                    f"observe({request!r}, {source!r}, {ephemeral_source!r}), sort_keys=True))\n")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as script:
            script.write(payload)
            script.flush()
            if self.reconcile() != self.owner:
                raise ValueError("owned session absent before probe")
            raw = self.cli(["exec", "-s", "phase0", "--timeout", "50", "-f", script.name], True)
        lines = raw.strip().splitlines()
        if len(lines) != 1 or not lines[0].startswith("E17_PROBE "):
            raise ValueError("missing or duplicate probe acknowledgement")
        return validate_result(json.loads(lines[0][10:]), request)

    def execute(self):
        result, error = None, None
        try:
            if self.a.receipt.exists():
                raise PermissionError("existing ownership receipt refused")
            if self.session() is not None:
                raise PermissionError("existing session refused")
            self.before = self.wait_usage(False)
            self.prepare()
            preflight(self.before, self.policy)
            self.pending = {"schema": 1, "status": "allocation-unconfirmed", "phase0": None,
                            "campaign_id": self.campaign, "allocation_id": uuid.uuid4().hex,
                            "plan": self.plan, "usage_before": self.before, "budget_policy": self.policy}
            lifecycle.save(self.a.receipt, self.pending)
            with defer_signals():
                self.cli(["new", "-s", "phase0", "--gpu", self.gpu])
                self.new_confirmed = True
            lifecycle.save(self.a.receipt, {**self.pending, "allocation_returned": True})
            if self.reconcile() is None:
                raise ValueError("new did not establish a session")
            lifecycle.save(self.a.receipt, {**self.pending, "allocation_returned": True,
                                            "status": "owned", "phase0": self.owner})
            self.monitor()
            # An impossible snapshot fails this attempt but never prevents ownership-based cleanup.
            if self.reconcile() != self.owner:
                raise ValueError("owned session absent before snapshot")
            raw = self.cli(["exec", "-s", "phase0", "--timeout", "30", "-f", str(HERE / "snapshot.py")], True)
            snapshot = json.loads(raw)
            if snapshot.get("campaign_present") is True:
                self.foreign = True
                lifecycle.save(self.a.receipt, {**self.pending, "status": "foreign-campaign", "phase0": self.owner})
                raise PermissionError("preexisting campaign: no execution or stop authorized")
            if snapshot.get("schema") != 1 or snapshot.get("campaign_present") is not False:
                raise ValueError("snapshot absent or preexisting campaign")
            result = self.perform()
        except BaseException as exc:
            error = type(exc).__name__
        finally:
            if self.pending is not None and self.a.receipt.exists():
                try:
                    self.cleanup()
                except BaseException as exc:
                    error = error or type(exc).__name__
                    try:
                        self.observations.append(lifecycle.usage(self.cli(["usage"])))
                    except (ValueError, RuntimeError, OSError, TimeoutError):
                        self.observations.append({"final_usage_unknown": True})
        report = {"schema": 1, "campaign_id": self.campaign, "verdict": result["verdict"]
                  if result and self.closed and error is None else
                  "infrastructure-unknown", "probe": result, "error": error, "absence_confirmed": self.absent,
                  "closed": self.closed, "usage": self.observations,
                  "sandbox_backend": getattr(self.a, "sandbox_backend", "strict")}
        lifecycle.save(self.a.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0 if self.closed and error is None and result["verdict"] == "compatible" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox-backend", choices=("strict", "ephemeral-vm"), default="strict")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, default=lifecycle.receipt_path())
    for name, default in (("seconds", 600), ("budget-units", 4), ("min-balance-units", 15),
                          ("cleanup-seconds", 1800), ("max-rate-units-hour", 5.3),
                          ("read-seconds", 120), ("usage-seconds", 30), ("grace", 5), ("retry-seconds", 1)):
        ap.add_argument("--" + name, type=float, default=default)
    args = ap.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in vars(args).values() if isinstance(v, float)):
        ap.error("all limits must be finite and positive")
    if args.seconds > 600:
        ap.error("preparation/probe must not exceed ten minutes")
    if args.output.resolve() in (args.receipt.resolve(), Path(str(args.receipt) + ".lock").resolve()):
        ap.error("output must differ from ownership and lock")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with lifecycle.lifecycle_lock(args.receipt):
            return Probe(args).execute()
    except (PermissionError, OSError, ValueError) as exc:
        print(f"probe refused: {type(exc).__name__}; inspect retained receipt", file=sys.stderr)
        return 73 if isinstance(exc, PermissionError) else 65


if __name__ == "__main__":
    sys.exit(main())
