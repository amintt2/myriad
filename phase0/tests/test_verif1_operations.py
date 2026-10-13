"""Offline operational regressions; fake local CLI only, never a cloud allocation."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verif1_campaign as cloud
from verif1_prepare import SOURCE_FILES
from verif1_protocol import Ledger, file_sha, sha, utc, write_new


def receipt(root):
    record = {"campaign": "VERIF-1", "session": {"fingerprint": "ours", "hardware": "A100"},
              "control_binding": "fixture", "utc": utc(), "hourly_rate": 0,
              "archive_sha256": "a" * 64, "bootstrap_sha256": "b" * 64}
    path = root / "receipt.json"
    write_new(path, record)
    return path, record


class VerifOperationsTests(unittest.TestCase):
    def test_down_preserves_pending_and_truncated_history_and_rejects_foreign_session(self):
        for interrupted in ("pending", "truncated"):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                path, record = receipt(Path(directory))
                journal = path.with_suffix(".ledger.jsonl")
                if interrupted == "pending":
                    with self.assertRaises(SystemExit):
                        Ledger(journal, "fixture").operation("upload", {}, lambda: sys.exit(1))
                else:
                    journal.write_bytes(b'{"kind":"reserve",')
                original = journal.read_bytes()
                path.with_suffix(".lock").write_bytes(b"orphaned work lock")
                args = argparse.Namespace(action="down", receipt=path, dry_run=False)
                with patch.object(cloud, "sessions", return_value=(
                        {"phase0": {"fingerprint": "foreign", "hardware": "A100"}}, False, ["session"])), patch.object(
                        cloud, "cli") as cli:
                    with self.assertRaises(ValueError):
                        cloud.execute(args)
                    cli.assert_not_called()
                with patch.object(cloud, "sessions", return_value=({"phase0": record["session"]}, False,
                                                                  ["session"])), patch.object(
                        cloud, "cli", return_value="stopped") as cli:
                    cloud.execute(args)
                    cli.assert_called_once_with(["stop", "-s", "phase0"])
                self.assertEqual(journal.read_bytes(), original)
                self.assertEqual(len(list(path.parent.glob("receipt.down.*.ledger.jsonl"))), 1)

    def test_status_active_complete_failed_absent_and_bounded_tail_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, record = receipt(root)
            journal = path.with_suffix(".ledger.jsonl")
            journal.write_bytes(b"truncated control journal")
            path.with_suffix(".lock").write_bytes(b"orphaned")
            state, log = root / "verif1_launch.json", root / "verif1_bootstrap.log"
            log.write_text("x" * 12000 + "\n" + "\n".join(str(i) for i in range(100)), encoding="utf-8")
            real_path = Path
            def cli(args, script=None):
                self.assertEqual(args, ["exec", "-s", "phase0", "--timeout", "30"])
                out = io.StringIO()
                with patch("pathlib.Path", side_effect=lambda p: real_path(root / real_path(p).name)), \
                        contextlib.redirect_stdout(out):
                    exec(compile(script, "trusted status fixture", "exec"), {})
                return out.getvalue()
            for complete, exit_code, expected in ((False, None, "active"), (True, 0, "complete"),
                                                  (True, 1, "failed"), (None, None, "not_launched")):
                if complete is None:
                    state.unlink()
                else:
                    state.write_text(json.dumps({"scope": "pilot", "archive_sha256": record["archive_sha256"],
                                                 "complete": complete, "exit_code": exit_code}))
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                out = io.StringIO()
                with patch.object(cloud, "sessions", return_value=({"phase0": record["session"]}, False,
                                                                  ["session"])), patch.object(cloud, "cli", cli), \
                        contextlib.redirect_stdout(out):
                    cloud.execute(argparse.Namespace(action="status", receipt=path, dry_run=False))
                result = json.loads(out.getvalue())
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["log_tail"], [str(i) for i in range(80, 100)])
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            state.write_text(json.dumps({"archive_sha256": "foreign", "scope": "full", "complete": True}))
            with self.assertRaises(AssertionError):
                cli(["exec", "-s", "phase0", "--timeout", "30"], cloud.status_script(record))
            state.write_text('{"scope":')
            with self.assertRaises(json.JSONDecodeError):
                cli(["exec", "-s", "phase0", "--timeout", "30"], cloud.status_script(record))

    def test_retrieval_destinations_are_distinct_and_independent_of_partial_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, record = receipt(root)
            journal = path.with_suffix(".ledger.jsonl")
            journal.write_bytes(b"partial")
            recovered = []
            def recover(record, output):
                recovered.append(output)
                return {"scope": "pilot" if len(recovered) == 1 else "full"}
            with patch.object(cloud, "owned", return_value=record), patch.object(cloud, "recover", recover):
                for name in ("pilot", "full", "full"):
                    cloud.execute(argparse.Namespace(action="pull", receipt=path, output=root / name, dry_run=False))
            self.assertEqual(recovered, [root / "pilot", root / "full"])
            self.assertEqual(journal.read_bytes(), b"partial")
            self.assertEqual(len(list(root.glob("*.pull.ledger.jsonl"))), 2)
            # A premature pull is terminal for that destination, not a polling mechanism.
            args = argparse.Namespace(action="pull", receipt=path, output=root / "premature", dry_run=False)
            with patch.object(cloud, "owned", return_value=record), patch.object(
                    cloud, "recover", side_effect=ValueError("writer active")) as recover:
                with self.assertRaises(ValueError):
                    cloud.execute(args)
                with self.assertRaises(ValueError):
                    cloud.execute(args)
                recover.assert_called_once()

    def test_every_local_action_and_whole_transfer_have_explicit_deadlines(self):
        with patch.object(cloud, "process", return_value="ok") as process:
            for action, seconds in cloud.CLI_SECONDS.items():
                cloud.cli([action])
                self.assertEqual(process.call_args.args[1], seconds)
            cloud.cli(["exec", "--timeout", "30"], "read only")
            self.assertEqual(process.call_args.args[1:], (90, "read only"))
            with patch.object(cloud, "file_sha", return_value="hash"):
                cloud.upload(Path("archive"))
            self.assertEqual(process.call_args_list[-2].args[1], 1800)
            self.assertTrue(process.call_args_list[-2].args[0][1].endswith("transfer.py"))

    def test_extension_and_recovery_share_live_mutex_but_keep_orphan_work_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, record = receipt(Path(directory))
            with cloud.cleanup_mutex(path.with_suffix(".cleanup.lock")):
                with patch.object(cloud, "owned") as owned, patch.object(cloud, "cli") as cli:
                    for action in ("extend", "pull", "down"):
                        with self.subTest(action=action), self.assertRaises(OSError):
                            cloud.execute(argparse.Namespace(action=action, receipt=path,
                                                             output=path.parent / "pull", dry_run=False))
                    owned.assert_not_called()
                    cli.assert_not_called()
            path.with_suffix(".lock").write_text("orphaned")
            with patch.object(cloud, "owned", return_value=record), patch.object(cloud, "cli") as cli:
                with self.assertRaises(FileExistsError):
                    cloud.execute(argparse.Namespace(action="extend", receipt=path, dry_run=False))
                cli.assert_not_called()
                with patch.object(cloud, "recover", return_value={"scope": "pilot"}) as recover:
                    cloud.execute(argparse.Namespace(action="pull", receipt=path, output=path.parent / "pilot",
                                                     dry_run=False))
                    recover.assert_called_once()
            self.assertEqual(path.with_suffix(".lock").read_text(), "orphaned")

    def test_real_recovery_adapter_preserves_distinct_pilot_and_full_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, record = receipt(root)
            source = root / "source.tgz"
            payload = root / "payload"
            payload.write_text("public fixture")
            with tarfile.open(source, "w:gz") as archive:
                archive.add(payload, arcname="verif-1/results/public.json")
            for scope in ("pilot", "full"):
                def cli(args, script=None):
                    if args[0] == "exec":
                        self.assertIn("pull requires stopped writer", script)
                        return json.dumps({"sha256": file_sha(source), "scope": scope, "exit_code": 0})
                    self.assertEqual(args[0], "download")
                    Path(args[-1]).write_bytes(source.read_bytes())
                    return ""
                destination = root / scope
                with patch.object(cloud, "cli", cli):
                    result = cloud.recover(record, destination)
                self.assertEqual(result["scope"], scope)
                self.assertEqual((destination / "verif-1/results/public.json").read_text(), "public fixture")
                self.assertEqual(json.loads((destination / "verif1_recovery_receipt.json").read_text())["scope"], scope)
            with patch.object(cloud, "cli", side_effect=AssertionError("must preserve previous pull")):
                with self.assertRaises(ValueError):
                    cloud.recover(record, root / "full")

    def test_technical_pilot_check_never_opens_labels_and_rejects_changed_artifacts(self):
        import verif1_analyze as analyzer
        from verif1_analyze import GRIDS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [{"contract": {"bench": b, "id": f"synthetic/{i:02d}"}}
                         for b in analyzer.historical.BENCHES for i in range(32)]
            write_new(root / "verif1_dev_bundle.json", {"questions": questions})
            plan = {"fixture": "not scientific"}
            write_new(root / "verif1_dev_plan.json", plan)
            binding = sha({"plan": plan, "simulation": False, "sandbox": {}})
            write_new(root / "verif1_manifest.json", {"plan": plan, "binding": binding,
                                                       "simulation": False, "sandbox": {}})
            ids = [q["contract"]["bench"] + ":" + q["contract"]["id"] for q in questions]
            final = {"coverage": {"scope": "pilot", "simulation": False, "complete_structural_coverage": True,
                                  "question_ids": ids, "questions": 64, "judges_expected": 4}, "costs": {},
                     "scores": {qid: {"scores": {c: [None] * 35 for grid in GRIDS.values() for c in grid}}
                                for qid in ids}}
            ledger = Ledger(root / "verif1_ledger.jsonl", binding)
            for i, status in enumerate(("valid", "truncated", "format_error", "usage_error")):
                result = {"status": status, "raw": {"truncated": status == "truncated"}}
                ledger.operation(f"{ids[0]}:judge:{i}:Qwen", {}, lambda: result)
            # A full-only question in the same history must not contaminate pilot diagnostics.
            ledger.operation("humanevalplus:synthetic/99:judge:0:Qwen", {}, lambda: {
                "status": "context_overflow", "raw": {"truncated": False}})
            ledger.operation("final:pilot", {}, lambda: final)
            for name, value in final.items():
                write_new(root / f"verif1_pilot_{name}.json", value)
            before = {p.name: file_sha(p) for p in root.iterdir()}
            with patch.object(analyzer, "validate_bundle"), patch.object(
                    analyzer.historical, "load_bench", side_effect=AssertionError("labels closed")):
                result = analyzer.technical_check(root, root, "pilot")
                self.assertFalse(result["labels_loaded"])
                self.assertEqual(result["judge_diagnostics"], {
                    "statuses": {"valid": 1, "truncated": 1, "format_error": 1, "usage_error": 1},
                    "requests": 4, "expected": 4, "raw_truncated": 1})
                self.assertEqual(before, {p.name: file_sha(p) for p in root.iterdir()})
                (root / "verif1_pilot_coverage.json").write_text("{}")
                with self.assertRaises(ValueError):
                    analyzer.technical_check(root, root, "pilot")

    def test_official_route_is_packaged_and_legacy_body_is_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        wrapper = root / "phase0/colab/colab_phase0.sh"
        route = '  verif-1)\n    shift\n    exec python3 "$COLAB_DIR/verif1_campaign.py" "$@"\n    ;;\n'
        self.assertIn(route, wrapper.read_text(encoding="utf-8"))
        self.assertIn("phase0/colab/colab_phase0.sh", SOURCE_FILES)
        self.assertIn("phase0/colab/read_timeout.py", SOURCE_FILES)
        # E11 scientific provenance inventory never included the orchestration wrapper.
        historical = (root / "phase0/analyze_percees_code.py").read_text(encoding="utf-8")
        inventory = historical.split('"source_sha256":', 1)[1].split('"attempts":', 1)[0]
        self.assertNotIn("colab_phase0", inventory)

    @unittest.skipUnless(sys.platform == "linux", "real group supervision and wrapper require WSL/Linux")
    def test_fake_cli_timeout_kills_descendants_then_owned_down_remains_callable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, record = receipt(root)
            line = "[phase0] our-session | Hardware: A100 | Shape: fixture | Variant: fixture"
            record["session"] = cloud.parse_sessions(line)["phase0"]
            path.write_text(json.dumps(record))
            marker, flag, pidfile = root / "survived", root / "block", root / "pid"
            flag.touch()
            child = ("import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                     f"time.sleep(1.5);Path({str(marker)!r}).write_text('escaped')")
            fake = ("import sys,subprocess,time;from pathlib import Path\n"
                    f"flag=Path({str(flag)!r})\n"
                    "if sys.argv[1]=='sessions' and flag.exists():\n"
                    f"    flag.unlink();p=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
                    f"    Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(30)\n"
                    f"elif sys.argv[1]=='sessions': print({line!r})\n"
                    "elif sys.argv[1]=='stop': print('stopped our-session')\n")
            (root / "safe_cli.py").write_text(fake)
            original = cloud.supervise
            def short_grace(*args, **kwargs):
                return original(*args, grace=.05, **kwargs)
            with patch.object(cloud, "HERE", root), patch.dict(cloud.CLI_SECONDS, sessions=.4), patch.object(
                    cloud, "supervise", short_grace):
                started = time.monotonic()
                with self.assertRaises(TimeoutError):
                    cloud.cli(["sessions"])
                self.assertLess(time.monotonic() - started, 3)
                self.assertTrue(pidfile.exists())
                cloud.execute(argparse.Namespace(action="down", receipt=path, dry_run=False))
            time.sleep(1.6)
            self.assertFalse(marker.exists(), "TERM-ignoring descendant escaped supervision")

    @unittest.skipUnless(sys.platform == "linux", "wrapper requires WSL/Linux")
    def test_completed_cli_and_terminated_controller_leave_no_local_descendants(self):
        for terminate_controller in (False, True):
            with self.subTest(terminate_controller=terminate_controller), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker, ready = root / "escaped", root / "ready"
                child = ("import signal,time;from pathlib import Path;"
                         "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                         f"Path({str(ready)!r}).touch();time.sleep(1);Path({str(marker)!r}).touch()")
                fake = ("import subprocess,sys,time;from pathlib import Path\n"
                        f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
                        f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
                        + ("time.sleep(30)\n" if terminate_controller else "print('complete')\n"))
                (root / "safe_cli.py").write_text(fake)
                if terminate_controller:
                    script = (f"import sys,signal;sys.path.insert(0,{str(cloud.HERE)!r});"
                              "import verif1_campaign as c;from pathlib import Path;"
                              f"c.HERE=Path({str(root)!r});"
                              "signal.signal(signal.SIGTERM,lambda *_:sys.exit(143));c.cli(['sessions'])")
                    controller = subprocess.Popen([sys.executable, "-c", script])
                    try:
                        deadline = time.monotonic() + 5
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue(ready.exists())
                        controller.terminate()
                        self.assertEqual(controller.wait(timeout=5), 143)
                    finally:
                        if controller.poll() is None:
                            controller.kill()
                            controller.wait()
                else:
                    with patch.object(cloud, "HERE", root):
                        self.assertEqual(cloud.cli(["sessions"]).strip(), "complete")
                time.sleep(1.1)
                self.assertFalse(marker.exists(), "local descendant survived controller/CLI exit")

    @unittest.skipUnless(sys.platform == "linux", "wrapper requires WSL/Linux")
    def test_fake_transfer_has_a_whole_operation_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "transfer.py").write_text("import time;time.sleep(30)")
            original = cloud.supervise
            def short_grace(*args, **kwargs):
                return original(*args, grace=.05, **kwargs)
            with patch.object(cloud, "HERE", root), patch.object(cloud, "TRANSFER_SECONDS", .2), patch.object(
                    cloud, "supervise", short_grace), patch.object(cloud, "cli") as cli:
                started = time.monotonic()
                with self.assertRaises(TimeoutError):
                    cloud.upload(root / "unused-archive")
                self.assertLess(time.monotonic() - started, 3)
                cli.assert_not_called()  # No bootstrap upload or hidden retry after the timed-out transfer.

    @unittest.skipUnless(sys.platform == "linux", "wrapper requires WSL/Linux")
    def test_wrapper_routes_all_offline_actions_without_cloud(self):
        wrapper = Path(__file__).resolve().parents[1] / "colab/colab_phase0.sh"
        for action in ("status", "extend", "pull", "down"):
            args = ["bash", str(wrapper), "verif-1", action, "--receipt", "absent", "--dry-run"]
            if action == "pull":
                args += ["--output", "absent-output"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
            self.assertEqual(json.loads(result.stdout), {"action": action, "receipt": "absent", "network": False})


if __name__ == "__main__":
    unittest.main()
