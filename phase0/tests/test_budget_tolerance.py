"""Synthetic account clocks; no CLI, allocation or scientific workload."""
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import budget
import campaign
import e16_supervision
import e17_probe
import read_timeout
from tests.test_e17_campaign import supervisor as controller


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class AccountWait(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.monotonic = mock.patch.object(budget.time, "monotonic", side_effect=lambda: self.clock.now)
        self.sleep = mock.patch.object(budget.time, "sleep", side_effect=self.clock.sleep)
        self.monotonic.start()
        self.sleep.start()
        self.addCleanup(self.monotonic.stop)
        self.addCleanup(self.sleep.stop)

    def args(self):
        return SimpleNamespace(seconds=2000, budget_units=8, min_balance_units=15, cleanup_seconds=1800,
                               max_rate_units_hour=5.3, read_seconds=120, grace=5, usage_seconds=30,
                               retry_seconds=1, recover_seconds=900, execute=True, receipt=Path("synthetic-receipt"))

    def probes(self):
        return (e17_probe.Probe(self.args()), controller.Campaign(self.args(), {"gpu": "L4"}, {}),
                e16_supervision.E16(self.args()))

    def arm(self, probe, balance=59.62):
        probe.before = {"schema": 1, "balance_units": balance, "rate_units_hour": 0, "assignments": 0}
        probe.last_usage = probe.before
        probe.last_usage_at = 0

    def raw(self, balance=59.62, assignments=1):
        return (f"Current balance: {balance} compute units\nUsage rate: 5.30/hr\n"
                f"Active assignments: {assignments}")

    def test_rpc_is_reaped_during_account_read_and_pause(self):
        # Drive the real run/cli/monitor loops; no mock invokes the expiry callback itself.
        for kind in ("probe", "e17", "e16", "historical"):
            for retry in ((1,) if kind == "historical" else (1, 10)):
                with self.subTest(kind=kind, retry=retry), tempfile.TemporaryDirectory() as directory:
                    self.clock.now, self.clock.sleeps = 0, []
                    processes, events = [], []
                    clock, raw = self.clock, self.raw()
                    class Process:
                        def __init__(self, command, **streams):
                            self.pid = len(processes) + 1
                            self.usage = "usage" in command or "usage-json" in command
                            self.done = self.reaped = False
                            self.output = streams.get("stdout")
                            self.finish = (clock.now + 120 if self.usage and self.pid > 2 else float("inf"))
                            processes.append(self)
                        def wait(self, timeout=None):
                            if self.done:
                                self.reaped = True
                                return -15
                            if self.finish <= clock.now + timeout:
                                clock.now = self.finish
                                self.done = True
                                value = (json.dumps({"schema": 1, "balance_units": 59.62,
                                                     "rate_units_hour": 5.3, "assignments": 1})
                                         if kind == "historical" else raw)
                                self.output.write(value.encode())
                                return 0
                            clock.now += timeout
                            if self.usage and self.pid > 2 and clock.now >= 66:
                                events.append((clock.now, processes[0].done, processes[0].reaped))
                            raise subprocess.TimeoutExpired("synthetic", timeout)
                    def kill(pid, sig):
                        process = processes[pid - 1]
                        if not process.done:
                            events.append(("kill", pid, clock.now))
                            process.done = True
                    if kind == "historical":
                        args = self.args()
                        args.read_seconds = 121
                        args.hours = args.seconds / 3600
                        target = campaign.Supervisor(args, Path(directory) / "receipt")
                        target.receipt.write_text(json.dumps({"status": "owned", "campaign_id": target.campaign_id,
                            "phase0": {"fingerprint": "a" * 64}, "usage_before":
                            {"schema": 1, "balance_units": 59.62, "rate_units_hour": 0, "assignments": 0},
                            "budget_policy": {"budget": 8, "minimum": 15, "rate": 5.3,
                                              "hours": .1, "cleanup": 1800, "margin": 161}}))
                        target.monitoring = True
                        invoke = lambda: target.call("snapshot", 60, True)
                        failure = campaign.Failure
                    else:
                        target = self.probes()[("probe", "e17", "e16").index(kind)]
                        target.a.retry_seconds = retry
                        target.a.read_seconds = 121
                        self.arm(target)
                        invoke = lambda: target.cli(["exec", "--timeout", "30"], monitor=True)
                        failure = e17_probe.RpcExpiredAfterAccountWait
                    with mock.patch.object(read_timeout.subprocess, "Popen", side_effect=Process), \
                         mock.patch.object(read_timeout, "killgroup", side_effect=kill), \
                         mock.patch.object(read_timeout.signal, "SIGKILL", 9, create=True), \
                         mock.patch.object(e16_supervision.lifecycle, "usage_record"), self.assertRaises(failure):
                        invoke()
                    self.assertEqual(len(processes), 3)
                    self.assertEqual(events[0], ("kill", 2, 55))
                    self.assertIn(("kill", 1, 60), events)
                    self.assertTrue(events[-1][1] and events[-1][2], "RPC must be reaped during the second read")
                    self.assertEqual(self.clock.now, 175 + (1 if kind == "historical" else retry))
                    self.assertFalse(getattr(target, "account_unknown", False))
                    self.assertFalse(getattr(target, "account_failed", False))
                    self.assertEqual(read_timeout.WATCHERS.get(), (), "expiry scope must not leak into cleanup")

    def test_transient_timeout_then_valid_shared_e16_e17(self):
        for probe in self.probes():
            with self.subTest(probe=type(probe).__name__):
                self.clock.now, self.clock.sleeps = 0, []
                self.arm(probe)
                def read(args):
                    self.clock.now += 120 if not self.clock.sleeps else 1
                    if not self.clock.sleeps:
                        raise budget.UsageTimeout("colab usage: exit 124")
                    return self.raw()
                with mock.patch.object(probe, "cli", side_effect=read) as cli, \
                     mock.patch.object(e16_supervision.lifecycle, "usage_record"):
                    self.assertGreater(probe.monitor(), 0)
                self.assertEqual(cli.call_count, 2)
                self.assertEqual(self.clock.sleeps, [1])
                self.assertFalse(getattr(probe, "account_unknown", False))
                self.assertFalse(getattr(probe, "account_failed", False))
                self.assertEqual(probe.deadline, 2000)

    def test_window_includes_reads_and_waits_and_never_exceeds_600(self):
        limits = []
        def read(limit):
            limits.append(limit)
            self.clock.now = min(limit, self.clock.now + 120)
            raise budget.UsageTimeout("synthetic timeout")
        with self.assertRaises(budget.UsageTimeout):
            budget.read_account(read, 2000)
        self.assertEqual(self.clock.now, 600)
        self.assertEqual(self.clock.sleeps, [1, 2, 4, 8])
        self.assertEqual(limits, [600] * 5)

    def test_short_rpc_timeout_preserves_account_window(self):
        for probe in self.probes():
            with self.subTest(probe=type(probe).__name__):
                self.clock.now, self.clock.sleeps = 0, []
                self.arm(probe)
                limits = []
                def read(args):
                    limits.append(probe.deadline)
                    if len(limits) == 1:
                        self.clock.now = 55
                        raise budget.UsageTimeout("colab usage: exit 124")
                    return self.raw()
                expire = mock.Mock()
                with mock.patch.object(probe, "cli", side_effect=read), \
                     mock.patch.object(e16_supervision.lifecycle, "usage_record"):
                    self.assertGreater(probe.monitor(55, expire), 0)
                self.assertEqual(limits, [55, 600])
                self.assertEqual(self.clock.now, 56)
                self.assertEqual(self.clock.sleeps, [1])
                self.assertEqual(expire.call_count, 2)
                self.assertEqual(probe.account_recoveries, 1)
                self.assertFalse(getattr(probe, "account_unknown", False))
                self.assertFalse(getattr(probe, "account_failed", False))

    def test_global_and_late_account_response_are_fatal(self):
        for bound in (55, 300):
            with self.subTest(bound=bound):
                self.clock.now = 0
                probe = self.probes()[1]
                self.arm(probe)
                probe.deadline = bound
                def late(args):
                    self.clock.now = bound
                    return self.raw()
                with mock.patch.object(probe, "cli", side_effect=late) as cli:
                    with self.assertRaises(budget.UsageTimeout):
                        probe.monitor(10)
                    self.assertEqual(cli.call_count, 1)
                    self.assertTrue(probe.account_unknown)
                    with self.assertRaises(ValueError):
                        probe.monitor()
                    self.assertEqual(cli.call_count, 1)
                self.assertEqual(probe.deadline, bound)

    def test_status_wait_exhausts_600_or_global_or_reserved_floor(self):
        for global_end, balance, minimum, expected in ((2000, 59.62, 15, 630), (200, 59.62, 15, 200),
                                                     (2000, 19.3, 16.15, 3.15 / 5.3 * 3600 - 1961)):
            with self.subTest(global_end=global_end, balance=balance):
                self.clock.now, self.clock.sleeps = 0, []
                probe = self.probes()[1]
                self.arm(probe, balance)
                probe.deadline = global_end
                probe.policy["minimum"] = minimum
                probe.owner, probe.request = {"synthetic": True}, {"identity": "synthetic"}
                probe.next_usage = 30
                reads = []
                calls = []
                def run(command, **kwargs):
                    op = command[2]
                    calls.append(op)
                    if op == "usage":
                        reads.append(probe.deadline)
                        self.clock.now = min(probe.deadline, self.clock.now + 120)
                        return 124
                    self.clock.now = 30
                    kwargs["monitor"](55, lambda: None)
                    self.fail("an exhausted account must abort its RPC")
                with mock.patch.object(e17_probe, "run", side_effect=run), \
                     mock.patch.object(probe, "reconcile", return_value=probe.owner), \
                     mock.patch.object(controller, "decode_response") as decode:
                    with self.assertRaises((budget.UsageTimeout, ValueError)):
                        probe.rpc("status")
                    decode.assert_not_called()
                self.assertEqual(calls.count("exec"), 1, "no fresh status under an unknown account")
                self.assertGreater(len(reads), 1)
                self.assertEqual(reads[0], 55)
                self.assertAlmostEqual(self.clock.now, expected)
                self.assertTrue(probe.account_unknown)

    def test_real_status_control_flow_recovers_account_but_rejects_late_rpc_and_mutation(self):
        for operation in ("status", "start"):
            with self.subTest(operation=operation):
                self.clock.now, self.clock.sleeps = 0, []
                probe = self.probes()[1]
                self.arm(probe)
                probe.next_usage = 30
                probe.owner, probe.request = {"synthetic": True}, {"identity": "synthetic"}
                calls = []
                def run(command, **kwargs):
                    op = command[2]
                    calls.append(op)
                    if op == "usage":
                        if calls.count(op) == 1:
                            self.clock.now = 55
                            return 124
                        kwargs["stdout"].write(self.raw().encode())
                        return 0
                    if calls.count("exec") == 1:
                        self.clock.now = 30
                        kwargs["monitor"](55, lambda: None)
                        return 124
                    kwargs["stdout"].write(b"synthetic fresh status")
                    return 0
                value = {"request": probe.request, "operation": operation}
                with mock.patch.object(e17_probe, "run", side_effect=run), \
                     mock.patch.object(probe, "operation_seconds", return_value=55), \
                     mock.patch.object(probe, "reconcile", return_value=probe.owner), \
                     mock.patch.object(controller, "decode_response", return_value=value) as decode:
                    if operation == "status":
                        self.assertEqual(probe.rpc(operation), value)
                        self.assertEqual(calls.count("exec"), 2)
                        decode.assert_called_once_with("synthetic fresh status")
                    else:
                        with self.assertRaises(e17_probe.RpcExpiredAfterAccountWait):
                            probe.rpc(operation, "synthetic mutation")
                        self.assertEqual(calls.count("exec"), 1)
                        decode.assert_not_called()
                self.assertFalse(probe.account_unknown)
                self.assertGreaterEqual(calls.count("usage"), 2)

    def test_balance_and_budget_stop_wait_before_ten_minutes_with_cleanup_reserved(self):
        for balance, budget_units in ((18.1, 8), (59.62, 3.1)):
            with self.subTest(balance=balance, budget=budget_units):
                self.clock.now = 0
                probe = self.probes()[1]
                self.arm(probe, balance)
                probe.policy["budget"] = budget_units
                limits = []
                def expired(args):
                    limits.append(probe.deadline)
                    self.clock.now = min(probe.deadline, self.clock.now + 120)
                    raise budget.UsageTimeout("colab usage: exit 124")
                with mock.patch.object(probe, "cli", side_effect=expired):
                    with self.assertRaises(budget.UsageTimeout):
                        probe.monitor()
                self.assertGreater(self.clock.now, 0)
                self.assertLess(self.clock.now, 600)
                self.assertAlmostEqual(limits[0], 3.1 / 5.3 * 3600 - 1800 - 161)
                self.assertTrue(probe.account_unknown)

    def test_invalid_account_non_timeout_and_term_do_not_retry(self):
        for failure in (ValueError("invalid shape"), RuntimeError("colab usage: exit 27"), SystemExit(143)):
            with self.subTest(failure=type(failure).__name__):
                read = mock.Mock(side_effect=failure)
                with self.assertRaises(type(failure)):
                    budget.read_account(read, 2000)
                read.assert_called_once()
                self.assertEqual(self.clock.sleeps, [])
        for raw in ("invalid", self.raw(assignments=2), self.raw(balance=15)):
            probe = self.probes()[1]
            self.arm(probe)
            with mock.patch.object(probe, "cli", return_value=raw) as cli:
                with self.assertRaises(ValueError):
                    probe.monitor()
                cli.assert_called_once()
                self.assertTrue(probe.account_unknown)

    def test_absolute_floor_cannot_be_lowered(self):
        probe = e17_probe.Probe(self.args())
        self.arm(probe)
        probe.policy["minimum"] = 14
        with self.assertRaises(ValueError):
            budget.preflight(probe.before, probe.policy)
        with self.assertRaises(ValueError):
            budget.remaining({**probe.before, "assignments": 1, "rate_units_hour": 5.3}, probe.policy, 59.62, 0, 5.3)
        probe.policy.update(minimum=15, prior_reserve=1.15)
        with self.assertRaises(ValueError):
            budget.preflight(probe.before, probe.policy)

    def test_term_in_nested_account_wait_blocks_recovery(self):
        for probe in self.probes():
            with self.subTest(probe=type(probe).__name__):
                self.clock.now, self.clock.sleeps = 0, []
                self.arm(probe)
                with mock.patch.object(probe, "cli", side_effect=SystemExit(143)) as cli:
                    with self.assertRaises(SystemExit):
                        probe.monitor(55)
                    with self.assertRaises(ValueError):
                        probe.monitor()
                    cli.assert_called_once()
                self.assertTrue(probe.account_waiting)

    def test_cleanup_failure_keeps_receipt_with_explicit_blocked_state(self):
        probe = e17_probe.Probe(self.args())
        probe.a.cleanup_seconds = 3
        probe.a.receipt = mock.Mock()
        probe.a.receipt.read_text.return_value = '{"status": "owned", "phase0": "synthetic"}'
        with mock.patch.object(e17_probe.signal, "signal"), \
             mock.patch.object(probe, "reconcile", return_value="synthetic"), \
             mock.patch.object(probe, "check_receipt"), \
             mock.patch.object(probe, "cli", side_effect=budget.UsageTimeout("stop pending")) as cli, \
             mock.patch.object(e17_probe.lifecycle, "save") as save:
            with self.assertRaises(TimeoutError):
                probe.cleanup()
            self.assertEqual(cli.call_count, 3)
            self.assertEqual(save.call_args.args[1], {"status": "owned", "phase0": "synthetic",
                                                     "cleanup_blocked": True, "release_confirmed": False})
        probe.a.receipt.unlink.assert_not_called()
        self.assertFalse(probe.closed)

    def test_expired_e16_account_wait_cannot_enter_recovery(self):
        probe = e16_supervision.E16(self.args())
        self.arm(probe)
        probe.deadline = probe.science_deadline = 100
        probe.recovery_deadline = 2000
        def read(args):
            self.clock.now = probe.deadline
            raise budget.UsageTimeout("colab usage: exit 124")
        with mock.patch.object(probe, "cli", side_effect=read):
            with self.assertRaises(budget.UsageTimeout):
                probe.monitor()
        self.assertTrue(probe.account_failed)
        self.assertEqual(self.clock.now, 600)

    def test_campaign_rechecks_ownership_after_account_wait_before_scientific_rpc(self):
        probe = self.probes()[1]
        self.arm(probe)
        with mock.patch.object(probe, "cli", side_effect=[budget.UsageTimeout("expired"), self.raw()]) as cli, \
             mock.patch.object(probe, "reconcile", side_effect=PermissionError("foreign replacement")):
            with self.assertRaises(PermissionError):
                probe.rpc("start", "synthetic")
        self.assertEqual(cli.call_args_list, [mock.call(["usage"]), mock.call(["usage"])])
        self.assertEqual(self.clock.sleeps, [1])

    def test_legacy_supervisor_uses_same_wait_and_sticky_failure(self):
        args = self.args()
        args.hours = args.seconds / 3600
        supervisor = campaign.Supervisor(args, mock.Mock())
        initial = {"schema": 1, "balance_units": 59.62, "rate_units_hour": 0, "assignments": 0}
        saved = {"status": "owned", "usage_before": initial, "budget_policy": e17_probe.Probe(args).policy}
        supervisor.receipt.read_text.return_value = json.dumps(saved)
        with mock.patch.object(supervisor, "cleanup_identity", return_value=saved), \
             mock.patch.object(supervisor, "call", side_effect=[campaign.Failure("expired", 124),
                                                               {**initial, "assignments": 1, "rate_units_hour": 5.3}]):
            self.assertGreater(supervisor.monitor(), 0)
        self.assertEqual(self.clock.sleeps, [1])
        supervisor.next_usage = 0
        with mock.patch.object(supervisor, "cleanup_identity", return_value=saved), \
             mock.patch.object(supervisor, "call", side_effect=campaign.Failure("invalid JSON", 65)) as call:
            with self.assertRaises(campaign.AccountingFailure):
                supervisor.monitor()
            with self.assertRaises(campaign.AccountingFailure):
                supervisor.monitor()
            call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
