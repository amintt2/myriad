"""One DEV pilot with durable ownership, account monitoring and bounded final cleanup."""
import json
import math
import os
import signal
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import budget
import session_json as lifecycle
import verif1_campaign as cloud


def technical(path, campaign_id, kind, **values):
    with Path(str(path) + ".technical.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"utc": cloud.utc(), "campaign_id": campaign_id,
                                 "event": kind, **values}, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def cleanup_diagnostic(path, campaign_id, kind, **values):
    """Diagnostic failures must not interrupt authorized release or final accounting."""
    try:
        technical(path, campaign_id, kind, **values)
    except Exception:
        pass


@contextmanager
def cleanup_watch(path, record, deadline):
    """Observe the account during long main cleanup calls; an unknown account never forbids owned stop."""
    import safe_cli
    original = safe_cli.bounded_run
    next_read = 0
    def monitor(enclosing):
        nonlocal next_read
        if time.monotonic() >= next_read:
            next_read = time.monotonic() + 30
            try:
                seconds = min(45, enclosing - time.monotonic(), deadline - time.monotonic())
                if seconds <= 0:
                    raise TimeoutError("cleanup account read deadline exceeded")
                value = lifecycle.usage(cloud.process([cloud.sys.executable, str(cloud.HERE / "safe_cli.py"),
                                                       "usage"], seconds))
                if value["assignments"] not in (0, 1):
                    raise ValueError("unexpected account assignments during cleanup")
                budget.validate_usage(value, bool(value["assignments"]))
                cleanup_diagnostic(path, record["campaign_id"], "cleanup_usage", account=value)
            except Exception as exc:
                cleanup_diagnostic(path, record["campaign_id"], "cleanup_account_unknown",
                                   error_type=type(exc).__name__)
        return deadline - time.monotonic()
    def watched(argv, **kwargs):
        return original(argv, monitor=monitor, deadline=deadline, **kwargs)
    safe_cli.bounded_run = watched
    try:
        yield
    finally:
        safe_cli.bounded_run = original


@contextmanager
def environment(values):
    previous = dict(os.environ)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


class Pilot:
    def __init__(self, args):
        self.a = args
        self.started = time.monotonic()
        self.deadline = self.started + args.hours * 3600
        self.operation_deadline = math.inf
        self.next_usage = 0
        self.rate = 0
        self.record = None
        self.monitoring = False
        self.reading_usage = False
        self.recovery_reserved = False
        self.closing_pull = False
        self.cleaning_up = False
        self.last_usage = None
        self.observed_owner = None
        self.new_evidence = None
        self.policy = {"budget": args.budget_units, "hours": args.hours, "minimum": args.min_balance_units,
                       "rate": args.max_rate_units_hour, "cleanup": args.cleanup_seconds,
                       "margin": 30 + args.read_seconds + 11}
        for name, value in self.policy.items():
            budget.number(value, name)
        if args.min_balance_units < 15:
            raise ValueError("minimum account reserve is 15 units")
        for name in ("read_seconds", "poll_seconds", "up_seconds"):
            budget.number(getattr(args, name), name)
        if args.poll_seconds > 30 or args.read_seconds > 45:
            raise ValueError("poll <= 30s and account read <= 45s required")
        self.campaign_id = uuid.uuid4().hex

    def event(self, kind, **values):
        emit = cleanup_diagnostic if self.cleaning_up else technical
        emit(self.a.receipt, self.campaign_id, kind, **values)

    def usage(self, allocated):
        self.reading_usage = True
        try:
            raw = cloud.process([cloud.sys.executable, str(cloud.HERE / "safe_cli.py"), "usage"],
                                self.a.read_seconds)
            value = lifecycle.usage(raw)
            if allocated is None:
                if value["assignments"] not in (0, 1):
                    raise ValueError("unexpected allocations during new")
                allocated = bool(value["assignments"])
            budget.validate_usage(value, allocated)
            self.event("usage", account=value)
            return value
        except BaseException:
            self.event("account_unknown")
            raise
        finally:
            self.reading_usage = False

    def monitor(self, deadline=None):
        if self.monitoring and not self.reading_usage and time.monotonic() >= self.next_usage:
            self.next_usage = time.monotonic() + 30
            enclosing = self.operation_deadline
            self.operation_deadline = min(enclosing, deadline if deadline is not None else math.inf)
            try:
                value = self.usage(True if self.record["status"] == "owned" else None)
            finally:
                self.operation_deadline = enclosing
            measured = value if value["assignments"] else {**value, "assignments": 1,
                                                         "rate_units_hour": self.policy["rate"]}
            seconds, self.rate = budget.remaining(measured, self.policy, self.record["usage_before"]["balance_units"],
                                                  time.monotonic() - self.started, self.rate)
            self.deadline = min(self.deadline, time.monotonic() + seconds)
            previous = self.last_usage or self.record["usage_before"]
            self.last_usage = value
            if self.closing_pull and (value["balance_units"] < previous["balance_units"] or
                    value["rate_units_hour"] > previous["rate_units_hour"]):
                raise ValueError("account changed during optional recovery; cleanup required")
        return self.deadline - time.monotonic()

    def allocate(self, plan):
        _, empty, _ = cloud.sessions()
        if not empty:
            raise ValueError("any existing session blocks allocation")
        before = self.usage(False)
        budget.preflight(before, self.policy)
        self.last_usage = before
        self.record = {"schema": 1, "campaign": "VERIF-1", "campaign_id": self.campaign_id,
                       "allocation_id": uuid.uuid4().hex, "status": "allocation-unconfirmed", "phase0": None,
                       "session": None, "usage_before": before, "budget_policy": self.policy,
                       "archive": str(self.a.archive.resolve()), "archive_sha256": cloud.file_sha(self.a.archive),
                       "bootstrap_sha256": plan["sources"]["phase0/colab/verif1_bootstrap.py"],
                       "hourly_rate": self.a.max_rate_units_hour, "utc": cloud.utc()}
        lifecycle.save(self.a.receipt, self.record)
        self.event("allocation_attempt", allocation_id=self.record["allocation_id"])
        self.monitoring = True
        self.next_usage = time.monotonic() + 30
        # Never replay new. A failed/ambiguous CLI is reconciled only from identity in its own output.
        try:
            cloud.cli(["new", "-s", "phase0", "--gpu", self.a.gpu])
            returned = True
            self.new_evidence = {**self.record, "allocation_returned": True}
            lifecycle.save(self.a.receipt, self.new_evidence)
            self.record = self.new_evidence
        except (RuntimeError, TimeoutError) as exc:
            returned = False
            try:
                attested = cloud.parse_sessions(getattr(exc, "stdout", str(exc)))["phase0"]
            except ValueError:
                raise ValueError("new failed without attested identity; receipt retained, no stop authorized") from exc
        if not returned:
            if attested is None or attested["hardware"] != self.a.gpu:
                raise ValueError("failed new has no allocation identity")
            self.new_evidence = {**self.record, "new_attested": attested}
            lifecycle.save(self.a.receipt, self.new_evidence)
            self.record = self.new_evidence
        for attempt in range(3):
            try:
                parsed, _, lines = cloud.sessions()
                owner = parsed["phase0"]
                if len(lines) != 1 or owner is None or owner["hardware"] != self.a.gpu:
                    raise ValueError("allocation identity unknown")
                if not returned and owner != attested:
                    raise ValueError("failed new identity differs from session")
                break
            except (RuntimeError, TimeoutError):
                if attempt == 2:
                    raise
        self.observed_owner = owner
        owned = {**self.record, "status": "owned", "phase0": owner, "session": owner}
        lifecycle.save(self.a.receipt, owned)
        self.record = owned
        os.environ.update(DLLM_REQUIRE_OWNER="1", DLLM_OPERATION_IDENTITY=json.dumps(lifecycle.identity(self.record)))
        self.event("ownership", identity=lifecycle.identity(self.record))
        self.monitoring = True
        self.next_usage = 0
        self.monitor()

    def reconcile(self, saved):
        evidence = self.new_evidence or saved
        for attempt in range(3):
            try:
                parsed, _, lines = cloud.sessions()
                owner = parsed["phase0"]
                if owner is None:
                    return  # Main cleanup confirms absence and accounts without authorizing a stop.
                if len(lines) != 1 or owner["hardware"] != self.a.gpu:
                    raise PermissionError("identity reconciliation differs; no stop authorized")
                if evidence.get("new_attested") is not None and owner != evidence["new_attested"]:
                    raise PermissionError("failed new attestation differs; no stop authorized")
                if self.observed_owner is not None and owner != self.observed_owner:
                    raise PermissionError("observed allocation identity differs; no stop authorized")
                self.observed_owner = owner
                owned = {**saved, **{k: evidence[k] for k in ("allocation_returned", "new_attested")
                                   if k in evidence}, "status": "owned", "phase0": owner, "session": owner}
                if json.loads(self.a.receipt.read_text(encoding="utf-8")) != saved:
                    raise PermissionError("reconciliation receipt replaced; no stop authorized")
                lifecycle.save(self.a.receipt, owned)
                self.record = owned
                os.environ.update(DLLM_REQUIRE_OWNER="1",
                                  DLLM_OPERATION_IDENTITY=json.dumps(lifecycle.identity(self.record)))
                self.event("ownership_reconciled", identity=lifecycle.identity(self.record))
                return
            except (RuntimeError, TimeoutError):
                if attempt == 2:
                    raise

    def cleanup(self):
        self.cleaning_up = True
        self.monitoring = False
        pilot_deadline = self.deadline
        cleanup_deadline = time.monotonic() + self.a.cleanup_seconds
        self.operation_deadline = cleanup_deadline
        self.deadline = cleanup_deadline
        # The main cleanup adapter preserves ambiguous receipts, repeats stop and confirms absence/accounting.
        with cloud.cleanup_mutex(self.a.receipt.with_suffix(".cleanup.lock")):
            saved = json.loads(self.a.receipt.read_text(encoding="utf-8"))
            phases = {"status", "phase0", "session"}
            # Only this exact volatile new proof may bridge an interrupted durable save.
            expected = [self.record] if self.new_evidence is None else [self.record, self.new_evidence]
            if saved.get("campaign_id") != self.campaign_id or (self.record is not None and
                    {k: v for k, v in saved.items() if k not in phases} not in
                    [{k: v for k, v in value.items() if k not in phases} for value in expected]):
                raise PermissionError("cleanup receipt replaced; no stop authorized")
            if self.record is not None and lifecycle.identity(saved) != lifecycle.identity(self.record):
                # Only the observed session transition of this durable allocation may differ after save interruption.
                pending, owned = (saved, self.record) if saved.get("status") == "allocation-unconfirmed" else (
                    self.record, saved)
                if (pending.get("status") != "allocation-unconfirmed" or pending.get("phase0") is not None or
                        pending.get("session") is not None or owned.get("status") != "owned" or
                        self.observed_owner is None or owned.get("phase0") != self.observed_owner or
                        owned.get("session") != self.observed_owner):
                    raise PermissionError("cleanup session receipt replaced; no stop authorized")
            self.record = saved
            evidence = self.new_evidence or saved
            if saved.get("status") == "allocation-unconfirmed" and (evidence.get("allocation_returned") is True or
                    evidence.get("new_attested") is not None):
                self.reconcile(saved)
                saved = self.record
            if (saved.get("status") == "owned" and saved.get("archive_sha256")
                    and not self.recovery_reserved):
                # A late completed writer is recovered even after TERM/up failure, before releasing its VM.
                self.deadline = min(pilot_deadline, cleanup_deadline - self.a.cleanup_seconds / 2)
                self.operation_deadline = self.deadline
                self.monitoring = True
                self.closing_pull = True
                self.next_usage = 0
                try:
                    if self.monitor(self.operation_deadline) <= 0:
                        raise TimeoutError("optional recovery deadline exhausted")
                    cloud.owned(self.a.receipt)
                    state = json.loads(cloud.cli(["exec", "-s", "phase0", "--timeout", "30"],
                                                 cloud.status_script(saved)))
                    self.event("cleanup_writer", status=state["status"])
                    if state["status"] in ("complete", "failed"):
                        self.recovery_reserved = True
                        cloud.recover(saved, self.a.output)
                except Exception as exc:
                    self.event("recovery_uncertain", error_type=type(exc).__name__)
                finally:
                    self.closing_pull = False
                    self.monitoring = False
                    self.deadline = cleanup_deadline
                    self.operation_deadline = cleanup_deadline
            cleanup_left = self.deadline - time.monotonic() - self.a.read_seconds - 5
            if cleanup_left <= 0:
                raise TimeoutError("cleanup deadline exhausted; durable receipt retained")
            os.environ["DLLM_CLEANUP_SECONDS"] = str(cleanup_left)
            with cleanup_watch(self.a.receipt, saved, self.deadline):
                lifecycle.down()
            try:
                final = self.usage(False)
            except BaseException:
                lifecycle.save(self.a.receipt, {**saved, "status": "released/accounting-pending",
                                                "release_confirmed": True})
                raise
            self.event("released", account=final)

    def execute(self):
        plan = cloud.validate_archive(self.a.archive)
        if self.a.scope != "pilot" or self.a.output is None or self.a.prepared is None:
            raise ValueError("supervised requires pilot, --prepared and a fresh output directory")
        prepared_plan = json.loads((self.a.prepared / "verif1_dev_plan.json").read_text(encoding="utf-8"))
        if prepared_plan != plan:
            raise ValueError("prepared plan differs from archive")
        if (self.a.receipt.exists() or self.a.output.exists()
                or Path(str(self.a.receipt) + ".technical.jsonl").exists()):
            raise ValueError("existing receipt/output refused; cleanup separately, never reallocate")
        self.a.receipt.parent.mkdir(parents=True, exist_ok=True)
        if self.a.dry_run:
            print(cloud.canonical({"network": False, "scope": "pilot", "policy": self.policy,
                                   "cost_ceiling": self.policy["rate"] * (self.policy["hours"] +
                                       (self.policy["cleanup"] + self.policy["margin"]) / 3600)}))
            return
        values = {"DLLM_CAMPAIGN_RECEIPT": str(self.a.receipt.resolve()), "DLLM_CAMPAIGN_ID": self.campaign_id,
                  "DLLM_SUPERVISED": "1", "DLLM_CLEANUP_SECONDS": str(self.a.cleanup_seconds)}
        # Also hold the main lifecycle lock, so a different wrapper cannot allocate concurrently.
        global_lock = lifecycle.receipt_path()
        if global_lock.exists():
            raise ValueError("main lifecycle receipt exists; cleanup must be reconciled first")
        with lifecycle.lifecycle_lock(global_lock), cloud.cleanup_mutex(self.a.receipt.with_suffix(".pilot.lock")), \
                environment(values):
            cloud.CONTROL = self
            try:
                self.allocate(plan)
                self.operation_deadline = time.monotonic() + self.a.up_seconds
                cloud.upload(self.a.archive)
                cloud.cli(["exec", "-s", "phase0", "--timeout", "60"],
                          cloud.launch("pilot", self.record["archive_sha256"], self.record["bootstrap_sha256"]))
                self.operation_deadline = math.inf
                while True:
                    self.monitor()
                    self.operation_deadline = time.monotonic() + self.a.read_seconds
                    cloud.owned(self.a.receipt)
                    status = json.loads(cloud.cli(["exec", "-s", "phase0", "--timeout", "30"],
                                                  cloud.status_script(self.record)))
                    self.operation_deadline = math.inf
                    self.event("writer", status=status["status"], launch=status.get("launch"))
                    if status["status"] in ("complete", "failed"):
                        with cloud.cleanup_mutex(self.a.receipt.with_suffix(".cleanup.lock")):
                            cloud.owned(self.a.receipt)
                            self.recovery_reserved = True
                            cloud.recover(self.record, self.a.output)
                        if status["status"] != "complete":
                            raise ValueError("pilot bootstrap/writer failed; recovered before cleanup")
                        from verif1_analyze import technical_check
                        technical_check(self.a.prepared, self.a.output / "verif-1/results", "pilot")
                        self.event("pilot_checked", labels_loaded=False)
                        break
                    if status["status"] != "active" or status.get("writer_alive") is not True:
                        raise ValueError("bootstrap absent or writer dead without completion; no infinite polling")
                    if self.monitor() <= 0:
                        raise TimeoutError("pilot deadline exceeded")
                    time.sleep(min(self.a.poll_seconds, max(0, self.monitor())))
            except BaseException as exc:
                self.event("failure", error_type=type(exc).__name__)
                raise
            finally:
                previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    if self.a.receipt.exists():
                        self.cleanup()
                finally:
                    signal.signal(signal.SIGTERM, previous)
                    cloud.CONTROL = None


def run(args):
    return Pilot(args).execute()


def cleanup_only(args):
    record = json.loads(args.receipt.read_text(encoding="utf-8"))
    if record.get("campaign") != "VERIF-1":
        raise PermissionError("not a VERIF-1 receipt")
    values = {"DLLM_CAMPAIGN_RECEIPT": str(args.receipt.resolve()), "DLLM_CAMPAIGN_ID": record["campaign_id"],
              "DLLM_OPERATION_IDENTITY": json.dumps(lifecycle.identity(record)), "DLLM_SUPERVISED": "1",
              "DLLM_CLEANUP_SECONDS": str(args.cleanup_seconds), "DLLM_REQUIRE_OWNER": "1"}
    budget.number(args.cleanup_seconds, "cleanup deadline")
    deadline = time.monotonic() + args.cleanup_seconds
    values["DLLM_CLEANUP_SECONDS"] = str(max(.01, args.cleanup_seconds - min(45, args.cleanup_seconds / 2) - 5))
    with lifecycle.lifecycle_lock(), cloud.cleanup_mutex(args.receipt.with_suffix(".cleanup.lock")), \
            environment(values):
        with cleanup_watch(args.receipt, record, deadline):
            lifecycle.down()
        try:
            seconds = min(45, deadline - time.monotonic())
            if seconds <= 0:
                raise TimeoutError("final accounting deadline exhausted")
            final = budget.validate_usage(lifecycle.usage(lifecycle.invoke(["usage"], seconds)), False)
            lifecycle.usage_record(args.receipt, record.get("usage_before"), final,
                                   campaign_id=record["campaign_id"], release_confirmed=True)
        except BaseException:
            lifecycle.save(args.receipt, {**record, "status": "released/accounting-pending", "release_confirmed": True})
            raise
