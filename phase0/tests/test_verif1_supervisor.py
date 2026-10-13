"""Offline account and lifecycle regressions for the single supervised DEV pilot."""
import argparse
import contextlib
import json
import io
import itertools
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import verif1_supervisor as pilot
import verif1_campaign as cloud
import session_json as lifecycle


def arguments(root, **changes):
    values = dict(receipt=root / "receipt.json", archive=root / "prepared.tgz", prepared=root / "prepared",
                  output=root / "recovered", hours=4, budget_units=25, min_balance_units=15,
                  max_rate_units_hour=5.3, cleanup_seconds=1800, read_seconds=45, poll_seconds=30,
                  up_seconds=2400, scope="pilot", gpu="A100", dry_run=False)
    return argparse.Namespace(**{**values, **changes})


def account(balance=59.62, rate=0, assignments=0):
    return dict(schema=1, balance_units=balance, rate_units_hour=rate, assignments=assignments)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.env = patch.dict(pilot.os.environ, {"HOME": self.home.name,
            "DLLM_CAMPAIGN_RECEIPT": str(Path(self.home.name) / "main-receipt.json")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.home.cleanup)

    def test_unknown_account_and_incoherent_reserve_duration_refused_before_new(self):
        for value, changes in ((None, {}), (account(), {"hours": 8}),
                               (account(), {"budget_units": 45}), (account(), {"min_balance_units": 14})):
            with tempfile.TemporaryDirectory() as directory, self.subTest(changes=changes):
                root = Path(directory)
                with patch.object(cloud, "sessions", return_value=({"phase0": None}, True, [])), \
                        patch.object(cloud, "cli") as cli:
                    with self.assertRaises((ValueError, TypeError)):
                        supervisor = pilot.Pilot(arguments(root, **changes))
                        with patch.object(supervisor, "usage", return_value=value):
                            supervisor.allocate({})
                    cli.assert_not_called()
                    self.assertFalse((root / "receipt.json").exists())

    def test_increased_rate_and_balance_loss_shorten_deadline_and_never_extend_it(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = pilot.Pilot(arguments(Path(directory)))
            supervisor.record = {"status": "owned", "usage_before": account()}
            supervisor.monitoring = True
            initial = supervisor.deadline
            with patch.object(supervisor, "usage", return_value=account(50, 10, 1)):
                supervisor.monitor()
            reduced = supervisor.deadline
            self.assertLess(reduced, initial)
            supervisor.next_usage = 0
            with patch.object(supervisor, "usage", return_value=account(58, 5.3, 1)):
                supervisor.monitor()
            self.assertLessEqual(supervisor.deadline, reduced)
            supervisor.next_usage = 0
            with patch.object(supervisor, "usage", side_effect=ValueError("unknown account")):
                with self.assertRaises(ValueError):
                    supervisor.monitor()

    @unittest.skipUnless(sys.platform == "linux", "durable directory fsync requires POSIX")
    def test_new_attempt_durable_and_nonzero_attested_identity_owned_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = pilot.Pilot(arguments(Path(directory)))
            line = "[phase0] allocated | Hardware: A100 | Shape: fixture | Variant: fixture"
            owner = cloud.parse_sessions(line)["phase0"]
            plan = {"sources": {"phase0/colab/verif1_bootstrap.py": "b" * 64}}
            calls = []
            def new(args):
                calls.append(args)
                saved = json.loads(supervisor.a.receipt.read_text())
                self.assertEqual(saved["status"], "allocation-unconfirmed")
                self.assertIsNone(saved["phase0"])
                raise RuntimeError(line)
            with patch.object(cloud, "sessions", side_effect=[({"phase0": None}, True, []),
                    ({"phase0": owner}, False, [line])]), patch.object(cloud, "cli", new), \
                    patch.object(cloud, "file_sha", return_value="a" * 64), \
                    patch.object(supervisor, "usage", side_effect=[account(), account(59, 5.3, 1)]), \
                    pilot.environment({}):
                supervisor.allocate(plan)
            self.assertEqual(calls, [["new", "-s", "phase0", "--gpu", "A100"]])
            self.assertEqual(json.loads(supervisor.a.receipt.read_text())["phase0"], owner)

    @unittest.skipUnless(sys.platform == "linux", "durable directory fsync requires POSIX")
    def test_nonzero_unknown_identity_preserves_pending_without_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = pilot.Pilot(arguments(Path(directory)))
            with patch.object(cloud, "sessions", return_value=({"phase0": None}, True, [])), \
                    patch.object(cloud, "cli", side_effect=RuntimeError("lost connection")) as cli, \
                    patch.object(cloud, "file_sha", return_value="a" * 64), \
                    patch.object(supervisor, "usage", return_value=account()):
                with self.assertRaisesRegex(ValueError, "without attested identity"):
                    supervisor.allocate({"sources": {"phase0/colab/verif1_bootstrap.py": "b" * 64}})
                self.assertEqual(cli.call_count, 1)
            self.assertEqual(json.loads(supervisor.a.receipt.read_text())["status"], "allocation-unconfirmed")

    @unittest.skipUnless(sys.platform == "linux", "main lifecycle cleanup requires POSIX")
    def test_stop_failure_retried_absence_and_final_account_no_new_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(schema=1, campaign="VERIF-1", campaign_id="fixture", allocation_id="allocation",
                          status="owned", phase0={"fingerprint": "ours", "hardware": "A100"},
                          usage_before=account())
            path = root / "receipt.json"
            lifecycle.save(path, record)
            owner_line = "[phase0] ours | Hardware: A100 | Shape: fixture | Variant: fixture"
            record["phase0"] = cloud.parse_sessions(owner_line)["phase0"]
            lifecycle.save(path, record)
            stop_count = 0
            def invoke(args, seconds=None):
                nonlocal stop_count
                if args[0] == "sessions":
                    return owner_line if stop_count < 2 else "[colab] No active sessions found on server."
                if args[0] == "stop":
                    stop_count += 1
                    if stop_count == 1:
                        raise RuntimeError("offline transient failure")
                    return "stopped"
                self.assertEqual(args, ["usage"])
                return "Current balance: 58.62 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
            with patch.object(lifecycle, "invoke", invoke), patch.object(lifecycle.time, "sleep"), \
                    patch.object(cloud, "cli", side_effect=lambda args: invoke(args)), \
                    contextlib.redirect_stdout(io.StringIO()):
                pilot.cleanup_only(argparse.Namespace(receipt=path, cleanup_seconds=10))
            self.assertEqual(stop_count, 2)
            self.assertFalse(path.exists())
            self.assertTrue(Path(str(path) + ".usage.jsonl").exists())

    def test_account_monitor_bounds_nested_read_and_rechecks_after_long_call(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = pilot.Pilot(arguments(Path(directory)))
            supervisor.record = {"status": "owned", "usage_before": account()}
            supervisor.monitoring = True
            bound = time.monotonic() + 1
            def usage(allocated):
                self.assertLessEqual(supervisor.operation_deadline, bound)
                return account(59, 5.3, 1)
            with patch.object(supervisor, "usage", usage):
                supervisor.monitor(bound)
            self.assertEqual(supervisor.operation_deadline, float("inf"))

    def test_cleanup_monitors_long_calls_every_30s_and_unknown_account_does_not_forbid_stop(self):
        import safe_cli
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            now = [100.0]
            def bounded(argv, **kwargs):
                self.assertEqual(argv, ["owned-stop-fixture"])
                self.assertEqual(kwargs["deadline"], 200)
                self.assertGreater(kwargs["monitor"](150), 0)
                now[0] += 31
                self.assertGreater(kwargs["monitor"](180), 0)
                return 0
            raw = "Current balance: 58.62 compute units\nUsage rate: 5.30/hr\nActive assignments: 1"
            with patch.object(pilot.time, "monotonic", side_effect=lambda: now[0]), \
                    patch.object(safe_cli, "bounded_run", bounded), \
                    patch.object(cloud, "process", side_effect=[raw, TimeoutError("account unknown")]) as process:
                with pilot.cleanup_watch(path, {"campaign_id": "fixture"}, 200):
                    self.assertEqual(safe_cli.bounded_run(["owned-stop-fixture"]), 0)
                self.assertIs(safe_cli.bounded_run, bounded)
                self.assertEqual(process.call_count, 2)
            events = [json.loads(line)["event"] for line in Path(str(path) + ".technical.jsonl").read_text().splitlines()]
            self.assertEqual(events, ["cleanup_usage", "cleanup_account_unknown"])

    @unittest.skipUnless(sys.platform == "linux", "main lifecycle cleanup requires POSIX")
    def test_absence_never_stops_and_unknown_final_account_retains_receipt(self):
        for failed in (False, "initial", "late"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(failed=failed):
                path = Path(directory) / "receipt.json"
                record = dict(schema=1, campaign="VERIF-1", campaign_id="fixture", allocation_id="allocation",
                              status="owned", phase0={"fingerprint": "ours", "hardware": "A100"},
                              usage_before=account())
                lifecycle.save(path, record)
                calls = []
                reads = 0
                def invoke(args, seconds=None):
                    nonlocal reads
                    calls.append(args[0])
                    if args[0] == "sessions":
                        return "[colab] No active sessions found on server."
                    self.assertEqual(args[0], "usage")
                    reads += 1
                    return "unknown" if failed == "initial" or (failed == "late" and reads == 2) else (
                        "Current balance: 58.62 compute units\nUsage rate: 0.00/hr\nActive assignments: 0")
                with patch.object(lifecycle, "invoke", invoke), patch.object(lifecycle.time, "sleep"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if failed:
                        with self.assertRaises((TimeoutError, ValueError)):
                            pilot.cleanup_only(argparse.Namespace(receipt=path, cleanup_seconds=10))
                        self.assertEqual(json.loads(path.read_text())["status"], "released/accounting-pending")
                    else:
                        pilot.cleanup_only(argparse.Namespace(receipt=path, cleanup_seconds=10))
                        self.assertFalse(path.exists())
                self.assertNotIn("stop", calls)
                self.assertNotIn("new", calls)

    def test_finished_attempt_cannot_allocate_again_even_after_receipt_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            args.prepared.mkdir()
            (args.prepared / "verif1_dev_plan.json").write_text('{"sources":{}}')
            Path(str(args.receipt) + ".technical.jsonl").write_text('{"event":"released"}\n')
            supervisor = pilot.Pilot(args)
            with patch.object(cloud, "validate_archive", return_value={"sources": {}}), \
                    patch.object(supervisor, "allocate") as allocate:
                with self.assertRaises(ValueError):
                    supervisor.execute()
                allocate.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "main ownership durability requires POSIX")
    def test_cleanup_reconciles_successful_new_after_failed_identity_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = pilot.Pilot(arguments(root))
            line = "[phase0] ours | Hardware: A100 | Shape: fixture | Variant: fixture"
            owner = cloud.parse_sessions(line)["phase0"]
            supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                allocation_id="fixture", status="allocation-unconfirmed", phase0=None, session=None,
                allocation_returned=True, usage_before=account())
            lifecycle.save(supervisor.a.receipt, supervisor.record)
            stopped = []
            def invoke(args, seconds=None):
                if args[0] == "sessions":
                    return "[colab] No active sessions found on server." if stopped else line
                if args[0] == "stop":
                    stopped.append(args)
                    return "stopped"
                self.assertEqual(args, ["usage"])
                return "Current balance: 58.62 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
            with pilot.environment({"DLLM_CAMPAIGN_RECEIPT": str(supervisor.a.receipt),
                    "DLLM_CAMPAIGN_ID": supervisor.campaign_id}), \
                    patch.object(cloud, "sessions", return_value=({"phase0": owner}, False, [line])), \
                    patch.object(lifecycle, "invoke", invoke), patch.object(lifecycle.time, "sleep"), \
                    patch.object(supervisor, "usage", return_value=account(58.62)), \
                    contextlib.redirect_stdout(io.StringIO()):
                supervisor.cleanup()
            self.assertEqual(stopped, [["stop", "-s", "phase0"]])
            self.assertFalse(supervisor.a.receipt.exists())

    @unittest.skipUnless(sys.platform == "linux", "final recovery uses POSIX lifecycle locks")
    def test_finally_pulls_late_completed_writer_before_down(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = pilot.Pilot(arguments(root))
            supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                allocation_id="fixture", status="owned", phase0={"fingerprint": "ours", "hardware": "A100"},
                session={"fingerprint": "ours", "hardware": "A100"}, archive_sha256="a" * 64,
                usage_before=account())
            supervisor.last_usage = account(59.62, 5.3, 1)
            lifecycle.save(supervisor.a.receipt, supervisor.record)
            calls = []
            def down():
                calls.append("down")
                supervisor.a.receipt.unlink()
            with patch.object(cloud, "owned", return_value=supervisor.record), \
                    patch.object(cloud, "cli", return_value='{"status":"complete"}'), \
                    patch.object(cloud, "recover", side_effect=lambda *a: calls.append("pull")), \
                    patch.object(lifecycle, "down", down), \
                    patch.object(supervisor, "usage", side_effect=lambda allocated:
                                 account(59.62, 5.3, 1) if allocated else account(58.62)), pilot.environment({}):
                supervisor.cleanup()
            self.assertEqual(calls, ["pull", "down"])

    @unittest.skipUnless(sys.platform == "linux", "main ownership and cleanup require POSIX")
    def test_diagnostic_oserror_never_prevents_reconciled_or_owned_release(self):
        for pending in (False, True):
            with tempfile.TemporaryDirectory() as directory, self.subTest(pending=pending):
                supervisor = pilot.Pilot(arguments(Path(directory)))
                line = "[phase0] ours | Hardware: A100 | Shape: fixture | Variant: fixture"
                owner = cloud.parse_sessions(line)["phase0"]
                supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                    allocation_id="fixture", status="allocation-unconfirmed" if pending else "owned",
                    phase0=None if pending else owner, session=None if pending else owner,
                    allocation_returned=True, archive_sha256="a" * 64, usage_before=account())
                lifecycle.save(supervisor.a.receipt, supervisor.record)
                stopped, events = [], []
                def invoke(args, seconds=None):
                    if args[0] == "sessions":
                        return "[colab] No active sessions found on server." if stopped else line
                    if args[0] == "stop":
                        stopped.append(args)
                        return "stopped"
                    self.assertEqual(args, ["usage"])
                    return "Current balance: 58.62 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
                def failed_journal(path, campaign_id, kind, **values):
                    events.append(kind)
                    raise OSError("synthetic journal unavailable")
                with pilot.environment({"DLLM_CAMPAIGN_RECEIPT": str(supervisor.a.receipt),
                        "DLLM_CAMPAIGN_ID": supervisor.campaign_id}), \
                        patch.object(cloud, "sessions", return_value=({"phase0": owner}, False, [line])), \
                        patch.object(lifecycle, "invoke", invoke), patch.object(lifecycle.time, "sleep"), \
                        patch.object(cloud, "process", side_effect=lambda *a: invoke(["usage"])), \
                        patch.object(supervisor, "monitor", return_value=1), patch.object(cloud, "owned"), \
                        patch.object(cloud, "cli", return_value='{"status":"complete"}'), \
                        patch.object(cloud, "recover", side_effect=OSError("synthetic recovery unavailable")), \
                        patch.object(pilot, "technical", side_effect=failed_journal), \
                        contextlib.redirect_stdout(io.StringIO()):
                    supervisor.cleanup()
                self.assertEqual(stopped, [["stop", "-s", "phase0"]])
                self.assertFalse(supervisor.a.receipt.exists())
                self.assertIn("recovery_uncertain", events)
                self.assertIn("released", events)
                if pending:
                    self.assertIn("ownership_reconciled", events)

    def test_cleanup_diagnostics_do_not_authorize_replaced_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = pilot.Pilot(arguments(Path(directory)))
            supervisor.record = dict(campaign_id=supervisor.campaign_id, allocation_id="ours", status="owned",
                                     phase0={"fingerprint": "ours"}, session={"fingerprint": "ours"})
            supervisor.a.receipt.write_text(json.dumps({**supervisor.record, "allocation_id": "foreign"}))
            before = supervisor.a.receipt.read_bytes()
            with patch.object(pilot, "technical", side_effect=OSError("synthetic journal unavailable")), \
                    patch.object(lifecycle, "down") as down, patch.object(cloud, "sessions") as sessions:
                with self.assertRaises(PermissionError):
                    supervisor.cleanup()
                down.assert_not_called()
                sessions.assert_not_called()
            self.assertEqual(supervisor.a.receipt.read_bytes(), before)

    def test_cleanup_watch_diagnostic_failure_keeps_stop_monitor_available(self):
        import safe_cli
        with patch.object(cloud, "process", side_effect=ValueError("synthetic unknown account")), \
                patch.object(pilot, "technical", side_effect=OSError("synthetic journal unavailable")), \
                patch.object(pilot.time, "monotonic", return_value=100), \
                patch.object(safe_cli, "bounded_run") as original:
            with pilot.cleanup_watch(Path("synthetic"), {"campaign_id": "fixture"}, 200):
                safe_cli.bounded_run(["synthetic stop"])
                monitor = original.call_args.kwargs["monitor"]
                self.assertEqual(monitor(200), 100)

    def test_closing_pull_account_changes_and_deadline_always_release(self):
        for reason in ("balance", "rate", "rate_below_max", "unknown", "budget", "floor", "deadline", "stable"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(reason=reason):
                now = [100.0]
                with patch.object(pilot.time, "monotonic", side_effect=lambda: now[0]):
                    supervisor = pilot.Pilot(arguments(Path(directory)))
                    supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                        allocation_id="fixture", status="owned", phase0={"fingerprint": "ours", "hardware": "A100"},
                        session={"fingerprint": "ours", "hardware": "A100"}, archive_sha256="a" * 64,
                        usage_before=account())
                    initial_rate = 4 if reason == "rate_below_max" else 5.3
                    supervisor.last_usage = account(58, initial_rate, 1)
                    if reason == "deadline":
                        supervisor.deadline = 125
                    reads, calls = [], []
                    def usage(allocated):
                        reads.append((now[0], allocated))
                        self.assertLessEqual(supervisor.operation_deadline, 1900)
                        if not allocated:
                            return account(58)
                        if len(reads) == 1 or reason in ("stable", "deadline"):
                            return account(58, initial_rate, 1)
                        if reason == "unknown":
                            raise ValueError("unknown account")
                        return {"balance": account(57.99, 5.3, 1), "rate": account(58, 5.4, 1),
                                "rate_below_max": account(58, 4.1, 1),
                                "budget": account(34, 5.3, 1), "floor": account(15, 5.3, 1)}[reason]
                    def recover(*args):
                        calls.append("pull")
                        self.assertTrue(supervisor.monitoring)
                        self.assertLessEqual(supervisor.operation_deadline, 1000)
                        self.assertGreater(supervisor.monitor(supervisor.operation_deadline), 0)
                        self.assertEqual(len(reads), 1)  # No extra read before the 30-second interval.
                        for _ in range(29):
                            now[0] += 30
                            if supervisor.monitor(supervisor.operation_deadline) <= 0:
                                raise TimeoutError("shared pull deadline")
                        calls.append("pull_complete")
                    def down():
                        self.assertFalse(supervisor.monitoring)
                        self.assertGreaterEqual(supervisor.deadline - now[0], 900)
                        calls.append("down")
                    with patch.object(Path, "read_text", return_value=json.dumps(supervisor.record)), \
                            patch.object(cloud, "cleanup_mutex", return_value=contextlib.nullcontext()), \
                            patch.object(cloud, "owned"), patch.object(cloud, "cli", return_value='{"status":"complete"}'), \
                            patch.object(cloud, "recover", recover), patch.object(supervisor, "usage", usage), \
                            patch.object(lifecycle, "down", down), patch.object(supervisor, "event"), \
                            patch.object(pilot, "cleanup_watch", return_value=contextlib.nullcontext()), pilot.environment({}):
                        supervisor.cleanup()
                    self.assertEqual(calls, ["pull", "pull_complete", "down"] if reason == "stable" else ["pull", "down"])
                    self.assertEqual(reads[-1][1], False)
                    self.assertEqual([t for t, allocated in reads if allocated],
                                     list(range(100, 971, 30)) if reason == "stable" else [100, 130])

    @unittest.skipUnless(sys.platform == "linux", "durable directory fsync requires POSIX")
    def test_interrupted_new_and_owner_save_reconciles_only_same_allocation_without_second_new(self):
        cases = itertools.product(("allocation_returned", "new_attested", "owned"),
                                  (None, "campaign", "allocation", "metadata", "session", "unattested"),
                                  (OSError("replacement interrupted"), SystemExit(143)), (False, True))
        for marker, refusal, failure, replaced in cases:
            with tempfile.TemporaryDirectory() as directory, self.subTest(failure=type(failure).__name__,
                                                                        replaced=replaced, marker=marker, refusal=refusal):
                supervisor = pilot.Pilot(arguments(Path(directory)))
                line = "[phase0] ours | Hardware: A100 | Shape: fixture | Variant: fixture"
                owner = cloud.parse_sessions(line)["phase0"]
                original = Path.replace
                interrupted = []
                def replace(path, target):
                    value = json.loads(path.read_text())
                    transition = value.get(marker) is not None if marker != "owned" else value["status"] == "owned"
                    if transition and not interrupted:
                        interrupted.append(True)
                        if replaced:
                            original(path, target)
                        raise failure
                    return original(path, target)
                calls = []
                def cli(argv):
                    calls.append(argv)
                    self.assertEqual(argv, ["new", "-s", "phase0", "--gpu", "A100"])
                    if marker == "new_attested":
                        raise RuntimeError(line)
                    return "allocated"
                with patch.object(cloud, "sessions", side_effect=[({"phase0": None}, True, []),
                        ({"phase0": owner}, False, [line])]), patch.object(cloud, "cli", cli), \
                        patch.object(cloud, "file_sha", return_value="a" * 64), \
                        patch.object(supervisor, "usage", return_value=account()), \
                        patch.object(Path, "replace", replace):
                    with self.assertRaises(type(failure)):
                        supervisor.allocate({"sources": {"phase0/colab/verif1_bootstrap.py": "b" * 64}})
                durable = json.loads(supervisor.a.receipt.read_text())
                self.assertEqual(bool(durable.get(marker)) if marker != "owned" else
                                 durable["status"] == "owned", replaced)
                self.assertEqual(supervisor.record["status"], "allocation-unconfirmed")
                if refusal in ("campaign", "allocation"):
                    durable[refusal + "_id"] = "foreign"
                if refusal == "metadata":
                    durable["usage_before"] = account(100)
                if refusal == "unattested":
                    supervisor.new_evidence = None
                    durable.pop("allocation_returned", None)
                    durable.pop("new_attested", None)
                    durable.update(status="allocation-unconfirmed", phase0=None, session=None)
                    supervisor.record = dict(durable)
                lifecycle.save(supervisor.a.receipt, durable)
                if refusal == "session" and marker == "allocation_returned":
                    supervisor.observed_owner = owner
                current = {**owner, "fingerprint": "foreign"} if refusal == "session" else owner
                current_line = line.replace("ours", "foreign") if refusal == "session" else line
                stopped = []
                def invoke(argv, seconds=None):
                    if argv[0] == "sessions":
                        return "[colab] No active sessions found on server." if stopped else current_line
                    if argv[0] == "stop":
                        stopped.append(argv)
                        return "stopped"
                    self.assertEqual(argv, ["usage"])
                    return "Current balance: 58.62 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
                supervisor.recovery_reserved = True
                with pilot.environment({"DLLM_CAMPAIGN_RECEIPT": str(supervisor.a.receipt),
                        "DLLM_CAMPAIGN_ID": supervisor.campaign_id}), \
                        patch.object(cloud, "sessions", return_value=({"phase0": current}, False, [current_line])), \
                        patch.object(lifecycle, "invoke", invoke), patch.object(lifecycle.time, "sleep"), \
                        patch.object(supervisor, "usage", return_value=account(58.62)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if refusal is None:
                        supervisor.cleanup()
                    else:
                        with self.assertRaises((PermissionError, ValueError)):
                            supervisor.cleanup()
                self.assertEqual(stopped, [["stop", "-s", "phase0"]] if refusal is None else [])
                self.assertEqual(len(calls), 1)
                self.assertEqual(supervisor.a.receipt.exists(), refusal is not None)

    def test_reconciliation_foreign_receipt_session_and_failed_persistence_never_authorize_stop(self):
        for reason in ("campaign", "allocation", "owned_foreign", "changed_metadata", "session", "save", "term",
                       "replaced_during_reconcile"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(reason=reason):
                supervisor = pilot.Pilot(arguments(Path(directory)))
                owner = {"fingerprint": "ours", "hardware": "A100"}
                supervisor.observed_owner = owner
                supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                    allocation_id="fixture", status="allocation-unconfirmed", phase0=None, session=None,
                    allocation_returned=True, usage_before=account())
                saved = dict(supervisor.record)
                if reason in ("campaign", "allocation"):
                    saved[reason + "_id"] = "foreign"
                if reason == "changed_metadata":
                    saved["usage_before"] = account(100)
                if reason == "owned_foreign":
                    saved.update(status="owned", phase0={**owner, "fingerprint": "foreign"},
                                 session={**owner, "fingerprint": "foreign"})
                session = {**owner, "fingerprint": "foreign"} if reason == "session" else owner
                failure = SystemExit(143) if reason == "term" else OSError("persistence failed")
                foreign = {**saved, "allocation_id": "foreign"}
                with patch.object(Path, "read_text", side_effect=[json.dumps(saved),
                        json.dumps(foreign if reason == "replaced_during_reconcile" else saved)]), \
                        patch.object(cloud, "cleanup_mutex", return_value=contextlib.nullcontext()), \
                        patch.object(cloud, "sessions", return_value=({"phase0": session}, False, ["fixture"])), \
                        patch.object(lifecycle, "save", side_effect=failure), patch.object(lifecycle, "down") as down:
                    with self.assertRaises(SystemExit if reason == "term" else
                                           OSError if reason == "save" else PermissionError):
                        supervisor.cleanup()
                    down.assert_not_called()
                    self.assertEqual(supervisor.record["status"], "allocation-unconfirmed")

    @unittest.skipUnless(sys.platform == "linux", "process supervision requires POSIX")
    def test_closing_pull_account_failure_and_deadline_kill_descendants_before_down(self):
        for reason in ("unknown", "deadline"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(reason=reason):
                root = Path(directory)
                supervisor = pilot.Pilot(arguments(root))
                supervisor.record = dict(schema=1, campaign="VERIF-1", campaign_id=supervisor.campaign_id,
                    allocation_id="fixture", status="owned", phase0={"fingerprint": "ours", "hardware": "A100"},
                    session={"fingerprint": "ours", "hardware": "A100"}, archive_sha256="a" * 64,
                    usage_before=account())
                supervisor.last_usage = account(58, 5.3, 1)
                escaped = root / "escaped"
                child = ("import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                         f"time.sleep(2);Path({str(escaped)!r}).touch()")
                command = root / "pull.py"
                command.write_text("import subprocess,sys,time\n"
                    f"subprocess.Popen([sys.executable,'-c',{child!r}])\ntime.sleep(30)\n")
                reads, calls = [], []
                def usage(allocated):
                    reads.append(allocated)
                    if not allocated:
                        return account(58)
                    if len(reads) > 1:
                        raise ValueError("unknown account")
                    return account(58, 5.3, 1)
                def recover(*args):
                    calls.append("pull")
                    if reason == "unknown":
                        supervisor.next_usage = time.monotonic() + .1
                    else:
                        supervisor.operation_deadline = min(supervisor.operation_deadline, time.monotonic() + .2)
                    cloud.process([sys.executable, str(command)], 30)
                    self.fail("pull should be interrupted")
                with patch.object(Path, "read_text", return_value=json.dumps(supervisor.record)), \
                        patch.object(cloud, "owned"), patch.object(cloud, "cli", return_value='{"status":"complete"}'), \
                        patch.object(cloud, "recover", recover), patch.object(supervisor, "usage", usage), \
                        patch.object(lifecycle, "down", side_effect=lambda: calls.append("down")), \
                        patch.object(pilot, "cleanup_watch", return_value=contextlib.nullcontext()), \
                        patch.object(cloud, "CONTROL", supervisor), pilot.environment({}):
                    supervisor.cleanup()
                self.assertEqual(calls, ["pull", "down"])
                time.sleep(2.1)
                self.assertFalse(escaped.exists())

    @unittest.skipUnless(sys.platform == "linux", "main lifecycle cleanup requires POSIX")
    def test_foreign_receipt_or_session_never_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            record = dict(schema=1, campaign="VERIF-1", campaign_id="fixture", allocation_id="allocation",
                          status="owned", phase0={"fingerprint": "ours", "hardware": "A100"})
            lifecycle.save(path, record)
            line = "[phase0] foreign | Hardware: A100 | Shape: fixture | Variant: fixture"
            with patch.object(lifecycle, "invoke", return_value=line) as invoke:
                with self.assertRaises(PermissionError):
                    pilot.cleanup_only(argparse.Namespace(receipt=path, cleanup_seconds=10))
                self.assertEqual(invoke.call_args.args[0], ["sessions"])
                self.assertEqual(invoke.call_count, 1)
            self.assertEqual(json.loads(path.read_text()), record)

    @unittest.skipUnless(sys.platform == "linux", "process supervision requires POSIX")
    def test_nested_slow_usage_cannot_outlive_enclosing_command_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe_cli.py").write_text("import sys,time\n"
                "if sys.argv[1]=='usage':\n"
                "    time.sleep(.4)\n"
                "    print('Current balance: 59.00 compute units\\nUsage rate: 5.30/hr\\nActive assignments: 1')\n"
                "else: time.sleep(5)\n")
            supervisor = pilot.Pilot(arguments(root))
            supervisor.record = {"status": "owned", "usage_before": account()}
            supervisor.monitoring = True
            start = time.monotonic()
            with patch.object(cloud, "HERE", root), patch.object(cloud, "CONTROL", supervisor):
                with self.assertRaises(TimeoutError):
                    cloud.process([sys.executable, str(root / "safe_cli.py"), "sessions"], .15)
            self.assertLess(time.monotonic() - start, 1.5)

    @unittest.skipUnless(sys.platform == "linux", "TERM and process groups require POSIX")
    def test_real_term_during_up_kills_local_descendants_and_enters_finally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            args.prepared.mkdir()
            (args.prepared / "verif1_dev_plan.json").write_text('{"sources":{}}')
            ready, escaped, cleaned = root / "ready", root / "escaped", root / "cleaned"
            child = ("import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                     f"Path({str(ready)!r}).touch();time.sleep(1);Path({str(escaped)!r}).touch()")
            command = root / "blocked.py"
            command.write_text("import subprocess,sys,time\n"
                               f"subprocess.Popen([sys.executable,'-c',{child!r}])\ntime.sleep(30)\n")
            script = f"""
import argparse,json,signal,sys
from pathlib import Path
sys.path.insert(0,{str(cloud.HERE)!r})
import verif1_campaign as cloud
import verif1_supervisor as pilot
signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
args=argparse.Namespace(**{vars(args)!r})
supervisor=pilot.Pilot(args)
cloud.validate_archive=lambda _: {{'sources':{{}}}}
def allocate(plan):
    supervisor.record={{'campaign_id':supervisor.campaign_id,'status':'owned'}}
    args.receipt.write_text(json.dumps(supervisor.record))
supervisor.allocate=allocate
supervisor.monitor=lambda deadline=None: 30
cloud.upload=lambda _: cloud.process([sys.executable,{str(command)!r}],30)
supervisor.cleanup=lambda: Path({str(cleaned)!r}).touch()
supervisor.execute()
"""
            # Paths in Namespace repr are trusted fixture paths, not candidate source.
            script = script.replace("PosixPath(", "Path(")
            controller = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline and controller.poll() is None:
                    time.sleep(.01)
                self.assertTrue(ready.exists())
                controller.terminate()
                self.assertEqual(controller.wait(timeout=10), 143)
                self.assertTrue(cleaned.exists())
                time.sleep(1.1)
                self.assertFalse(escaped.exists())
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait()

    @unittest.skipUnless(sys.platform == "linux", "pilot lifecycle uses POSIX locks")
    def test_up_failure_timeout_and_term_always_cleanup_owned_receipt(self):
        for failure in (RuntimeError("upload failed"), TimeoutError("up expired"), SystemExit(143)):
            with tempfile.TemporaryDirectory() as directory, self.subTest(failure=type(failure).__name__):
                root = Path(directory)
                args = arguments(root)
                args.prepared.mkdir()
                plan = {"sources": {}}
                (args.prepared / "verif1_dev_plan.json").write_text(json.dumps(plan))
                supervisor = pilot.Pilot(args)
                def allocate(plan):
                    supervisor.record = {"campaign_id": supervisor.campaign_id, "status": "owned"}
                    args.receipt.write_text(json.dumps(supervisor.record))
                with patch.object(cloud, "validate_archive", return_value=plan), \
                        patch.object(supervisor, "allocate", allocate), \
                        patch.object(cloud, "upload", side_effect=failure), \
                        patch.object(supervisor, "cleanup") as cleanup:
                    with self.assertRaises(type(failure)):
                        supervisor.execute()
                    cleanup.assert_called_once()
                    self.assertIsNone(cloud.CONTROL)
                self.assertTrue(args.receipt.exists())

    @unittest.skipUnless(sys.platform == "linux", "pilot lifecycle uses POSIX locks")
    def test_incomplete_writer_stops_and_complete_pulls_before_down_without_extension(self):
        for state in ("active", "not_launched", "failed", "complete"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(state=state):
                root = Path(directory)
                args = arguments(root)
                args.prepared.mkdir()
                plan = {"sources": {}}
                (args.prepared / "verif1_dev_plan.json").write_text(json.dumps(plan))
                supervisor = pilot.Pilot(args)
                calls = []
                def allocate(plan):
                    supervisor.record = {"campaign_id": supervisor.campaign_id, "archive_sha256": "a",
                                         "bootstrap_sha256": "b"}
                    args.receipt.write_text(json.dumps(supervisor.record))
                def cli(argv, script=None):
                    calls.append(argv)
                    return json.dumps({"status": state, "writer_alive": False})
                with patch.object(cloud, "validate_archive", return_value=plan), \
                        patch.object(supervisor, "allocate", allocate), \
                        patch.object(supervisor, "monitor", return_value=10), \
                        patch.object(cloud, "upload"), patch.object(cloud, "owned"), patch.object(cloud, "cli", cli), \
                        patch.object(cloud, "recover", side_effect=lambda *a: calls.append("pull")), \
                        patch("verif1_analyze.technical_check", side_effect=lambda *a: calls.append("check")), \
                        patch.object(supervisor, "cleanup", side_effect=lambda: calls.append("down")):
                    if state == "complete":
                        supervisor.execute()
                        self.assertEqual(calls[-3:], ["pull", "check", "down"])
                    else:
                        with self.assertRaises(ValueError):
                            supervisor.execute()
                        self.assertEqual(calls[-1], "down")
                        self.assertEqual("pull" in calls, state == "failed")
                    self.assertEqual(len([c for c in calls if isinstance(c, list)]), 2)


if __name__ == "__main__":
    unittest.main()
