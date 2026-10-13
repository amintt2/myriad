"""Offline regression tests for train-only selection, nested grids, verifier diagnostics and cloud ownership."""
import argparse
import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_percees_code as h
from verif1_analyze import (CONFIG, GRIDS, analyze, choose, diagnostic_bootstrap, diagnostic_statistics,
                            freeze, run_analysis, scientific_select)
from verif1_protocol import (AUTHOR_DECODER, AUTHORS, JUDGE_DECODER, Ledger, author_acceptances,
                             canonical, differential_scores, file_sha, parse_proposal, sha,
                             supported_scores, write_new)
from test_verif1_pipeline import DataEngine, DataSandbox, candidate, contract
from verif1_run import inputs_for, proposed, run_pipeline


def fixture():
    datasets, questions, vectors = {}, {}, {}
    ids, counts = [], {}
    for i in range(1000):
        qid = f"synthetic/dev/{i}"
        key = h.fold(qid), h.fold(qid, 3)
        if counts.get(key, 0) < 2:
            counts[key] = counts.get(key, 0) + 1
            ids.append(qid)
        if len(ids) == 30:
            break
    for bench in h.BENCHES:
        candidates, features, labels = {}, {}, {}
        questions[bench], vectors[bench] = {}, {}
        for qi, qid in enumerate(ids):
            candidates[qid] = []
            public = []
            for mi, model in enumerate(h.MODELS):
                for sample in range(5):
                    c = h.Candidate(qid, model, sample, f"program-{mi}-{sample}", "source", 1, 1)
                    candidates[qid].append(c)
                    features[c] = h.PublicFeatures(True, True, 1, ("a" * 16,))
                    labels[c] = (qi + mi + sample) % 3 == 0
                    if model in h.SWARM:
                        public.append({"model": model, "sample": sample, "prog": c.prog})
            questions[bench][qid] = {"candidates": public}
            vectors[bench][qid] = {
                "scores": {key: [None] * 35 for grid in GRIDS.values() for key in grid},
                "diagnostic_acceptances": {t: [{a: None for a in AUTHORS} for _ in range(35)]
                                           for t in ("1", "2")}}
        datasets[bench] = h.BenchData(bench, candidates, features, labels, {"synthetic": True})
    return datasets, questions, vectors


class VerifAnalysisTests(unittest.TestCase):
    def test_visible_priority_train_tie_and_exact_missing_baseline(self):
        models = h.SWARM[:3]
        candidates = [h.Candidate("q", m, 0, str(i), "", 0, 0) for i, m in enumerate(models)]
        features = {c: h.PublicFeatures(True, i == 0, i, ()) for i, c in enumerate(candidates)}
        weights = {models[0]: 0, models[1]: 20, models[2]: 100}
        self.assertEqual(scientific_select(candidates, features, [.1, .9, 1.0], weights), candidates[0])
        for visible in (True, False):
            features = {c: h.PublicFeatures(i != 2, visible and i == 0, i, ())
                        for i, c in enumerate(candidates)}
            self.assertEqual(scientific_select(candidates, features, [None] * 3, weights),
                             h.select(candidates, features, weights, "visible"))
        features = {c: h.PublicFeatures(True, True, 1, ()) for c in candidates}
        self.assertEqual(scientific_select(candidates, features, [.5] * 3, weights), candidates[2])

    def test_nested_all_grids_fits_exclude_validation_and_external(self):
        data, questions, vectors = fixture()
        train = {b: [q for q in d.candidates if h.fold(q) != 0] for b, d in data.items()}
        for experiment, grid in GRIDS.items():
            journal = []
            choice = choose(data, train, questions, vectors, experiment, 0, journal)
            self.assertEqual(choice, grid[0])  # All vectors absent: all configurations are identical baseline.
            trials = [r for r in journal if r["type"] == "inner_trial"]
            self.assertEqual(len(trials), len(grid) * 3)
            for trial in trials:
                for bench in h.BENCHES:
                    fitted = trial["fits"][bench]["train_ids"]
                    self.assertTrue(all(h.fold(q) != 0 and h.fold(q, 3) != trial["inner"] for q in fitted))
                    self.assertFalse(set(fitted) & set(trial["validation_ids"][bench]))

    def test_full_synthetic_analysis_smoke_and_freeze_provenance(self):
        data, questions, vectors = fixture()
        result = analyze(data, questions, vectors)
        summary = result["summary"]
        for experiment in GRIDS:
            self.assertEqual(summary["results"][experiment], summary["results"]["visible_k5"])
        self.assertEqual(len(summary["outer_choices"]), 20)
        self.assertEqual(len(summary["freeze_choices"]), 4)
        for experiment in GRIDS:
            for bank in summary["selection_diagnostics"][experiment]["by_bench"].values():
                self.assertEqual(bank["signal_absent_questions"], 30)
                self.assertEqual(bank["versus"]["visible_k5"]["repairs"], 0)
                self.assertEqual(bank["versus"]["visible_k5"]["breaks"], 0)
        # This fixture stays inside a temporary software test, never in a scientific campaign artifact.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from verif1_prepare import source_hashes
            summary.update(bindings={"code_sha256": sha(source_hashes()), "plan_sha256": "a" * 64,
                                     "data_sha256": "a" * 64}, run_ledger_sha256="b" * 64)
            binding = {"bindings": summary["bindings"], "run_ledger_sha256": summary["run_ledger_sha256"],
                       "config": CONFIG}
            ledger = Ledger(root / "verif1_analysis_ledger.jsonl", sha(binding))
            ledger.operation("analyze_dev", {}, lambda: result)
            write_new(root / "verif1_summary_dev.json", summary)
            freeze(root, root / "frozen")
            first = file_sha(root / "frozen/verif1_V06_frozen_dev.json")
            freeze(root, root / "frozen")
            self.assertEqual(file_sha(root / "frozen/verif1_V06_frozen_dev.json"), first)
            changed = json.loads((root / "verif1_summary_dev.json").read_text())
            changed["freeze_choices"]["V03"]["configuration"] = "V03:2"
            (root / "verif1_summary_dev.json").write_text(canonical(changed))
            with self.assertRaises(ValueError):
                freeze(root, root / "frozen")

    def test_diagnostic_all_cases_absent_pair_and_question_bootstrap(self):
        fixed = [candidate(a) for a in AUTHORS]
        proposals = [{"author": "Qwen", "kind": "cases", "item": {"expected": 2}, "indices": [0]},
                     {"author": "Qwen", "kind": "cases", "item": {"expected": 3}, "indices": [1]}]
        matrix = [[{"ok": True, "value": 2}, {"ok": True, "value": 3}] for _ in fixed]
        matrix[1][1]["value"] = 0
        outcome = author_acceptances(fixed, proposals, matrix, 2)
        self.assertIsNone(outcome[0]["Qwen"])
        self.assertFalse(outcome[1]["Qwen"])
        self.assertTrue(outcome[2]["Qwen"])
        self.assertIsNone(outcome[2]["Phi"])
        matrix[1][1] = {"ok": False, "coverage_loss": True}
        self.assertIsNone(author_acceptances(fixed, proposals, matrix, 2)[1]["Qwen"])
        rows = [{"id": "q1", "correct": True, "accept": [True, False]},
                {"id": "q1", "correct": False, "accept": [True, True]},
                {"id": "q2", "correct": False, "accept": [False, False]}]
        stats = diagnostic_statistics(rows)
        self.assertEqual(stats["false_reject"], 1)
        self.assertEqual(stats["false_agreement"], .5)
        self.assertEqual(stats["rho_FA"], 1)
        self.assertEqual(stats["candidates"], 3)
        self.assertEqual(stats["questions"], 2)
        self.assertEqual(diagnostic_bootstrap(rows), diagnostic_bootstrap(rows))

    def test_seven_sampling_dedup_missing_json_and_failed_smoke(self):
        engine = DataEngine()
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger", "fixture")
            responses = {a: [proposed(ledger, f"{a}:{i}", a, "contract only", 512,
                                      20261010 + i, engine, parse_proposal) for i in range(2)] for a in AUTHORS}
        self.assertEqual(len(engine.calls), 14)
        self.assertTrue(all(d == AUTHOR_DECODER for d in engine.decoders))
        self.assertEqual({c[3] for c in engine.calls}, {20261010, 20261011})
        prepared = inputs_for(contract(), responses, DataSandbox())
        self.assertEqual(len(prepared["proposals"]), 14)  # One case and one property per family, not per call.
        fixed = [candidate(a) for a in AUTHORS]
        matrix = [[{"ok": True, "value": 2}] for _ in fixed]
        matrix[1] = [{"ok": False, "coverage_loss": True}]
        self.assertIsNone(supported_scores(fixed, prepared["proposals"], matrix, 1, "cases")[0][1])
        self.assertIsNone(differential_scores(fixed, matrix, [0])[1])
        failing = DataSandbox()
        failing.smoke = lambda: (_ for _ in ()).throw(ValueError("real smoke failure"))
        engine = DataEngine()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            run_pipeline({"questions": [{"contract": contract(), "candidates": fixed}]}, {"config": {}},
                         Path(directory), engine, failing, simulation=True)
        self.assertEqual(engine.calls, [])

    def test_scientific_cli_rejects_simulation_before_label_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_new(root / "verif1_dev_bundle.json", {"snapshot": {}})
            write_new(root / "verif1_dev_plan.json", {})
            write_new(root / "verif1_manifest.json", {"simulation": True, "sandbox": {"simulation": True}})
            with patch("verif1_analyze.validate_bundle"), patch.object(
                    h, "load_bench", side_effect=AssertionError("labels must stay closed")):
                with self.assertRaises(ValueError):
                    run_analysis(root, root, root, root / "analysis")


class VerifCloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "colab/verif1_campaign.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("verif1_campaign", path)
        cls.cloud = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cloud)

    def test_launcher_syntax_offline_and_dry_down_no_network(self):
        from verif1_protocol import utc
        self.assertIsNone(self.cloud.billing({"hourly_rate": 0, "utc": utc()})["estimated_monetary"])
        compile(self.cloud.launch("pilot", "a" * 64, "b" * 64), "launcher", "exec")
        with patch.object(socket.socket, "connect", side_effect=AssertionError), patch.object(
                self.cloud, "cli", side_effect=AssertionError("network forbidden")):
            self.cloud.execute(argparse.Namespace(action="down", receipt=Path("absent"), dry_run=True))

    def test_uploadable_launcher_constructs_valid_child_and_refuses_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verif1_bootstrap.py").write_bytes(b"bootstrap fixture")
            (root / "dllm.tgz").write_bytes(b"archive fixture")
            script = self.cloud.launch("pilot", file_sha(root / "dllm.tgz"), file_sha(root / "verif1_bootstrap.py"))
            captured = []
            def popen(argv, **kwargs):
                captured.append(argv)
                state = root / "verif1_launch.json"
                record = json.loads(state.read_text())
                state.write_text(json.dumps({**record, "pid": 999}))  # Simulate the child startup acknowledgement.
                return argparse.Namespace(pid=999)
            real_path = Path
            with patch("pathlib.Path", side_effect=lambda p: real_path(root / real_path(p).name)), patch(
                    "subprocess.Popen", side_effect=popen), patch("builtins.open", create=True) as opened:
                # Launcher log is the sole builtins.open call; paths use Path.open for digest/state.
                exec(compile(script, "trusted launcher fixture", "exec"), {})
                compile(captured[0][2], "background child", "exec")
                self.assertIn("--run", captured[0][2])
                with self.assertRaises(AssertionError):
                    exec(compile(script, "trusted launcher fixture", "exec"), {})

    @unittest.skipUnless(sys.platform == "linux", "concurrent marker replacement requires POSIX")
    def test_launcher_publishes_every_state_atomically_for_concurrent_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap, archive = root / "verif1_bootstrap.py", root / "dllm.tgz"
            bootstrap.write_text("import time; time.sleep(.1); raise SystemExit(7)\n")
            archive.write_bytes(b"synthetic archive")
            marker = root / "verif1_launch.json"
            # Slow writes expose the old live-marker truncation; staged writes remain invisible to readers.
            slow = ("import json,time\nfrom pathlib import Path\n"
                    "original_write=Path.write_text; original_dump=json.dump\n"
                    "def slow_write(path,text,*args,**kwargs):\n"
                    "    if path.name!='verif1_launch.json': return original_write(path,text,*args,**kwargs)\n"
                    "    with path.open('w') as stream:\n"
                    "        stream.write(text[:len(text)//2]); stream.flush(); time.sleep(.05)\n"
                    "        return stream.write(text[len(text)//2:])\n"
                    "def slow_dump(*args,**kwargs):\n"
                    "    original_dump(*args,**kwargs); time.sleep(.05)\n"
                    "Path.write_text=slow_write; json.dump=slow_dump\n")
            script = self.cloud.launch("pilot", file_sha(archive), file_sha(bootstrap))
            script = script.replace("/content/", root.as_posix() + "/")
            script = script.replace("[sys.executable,'-c',child]", "[sys.executable,'-c',slow+child]")
            script = "slow=" + repr(slow) + "\n" + slow + script
            reads, errors, stop = [], [], threading.Event()
            def read():
                while not stop.is_set():
                    try:
                        reads.append(json.loads(marker.read_text()))
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        errors.append(exc)
                    time.sleep(.001)
            reader = threading.Thread(target=read)
            reader.start()
            try:
                launched = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
                self.assertEqual(launched.returncode, 0, launched.stdout + launched.stderr)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    current = json.loads(marker.read_text())
                    if current["complete"]:
                        break
                    time.sleep(.01)
                self.assertTrue(current["complete"])
                self.assertEqual(current["exit_code"], 7)
                self.assertEqual(current["pid"], json.loads(launched.stdout)["launched_pid"])
                relaunch = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(relaunch.returncode, 0)
            finally:
                stop.set()
                reader.join(timeout=5)
            self.assertFalse(reader.is_alive())
            self.assertFalse(errors, errors)
            self.assertTrue(any(not row["complete"] and "pid" not in row for row in reads))
            self.assertTrue(any(not row["complete"] and "pid" in row for row in reads))
            self.assertTrue(any(row["complete"] and row["exit_code"] == 7 for row in reads))
            self.assertEqual({row["archive_sha256"] for row in reads}, {file_sha(archive)})
            self.assertFalse(list(root.glob("tmp*")))

    def test_occupied_session_refused_and_fingerprint_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            write_new(receipt, {"session": {"fingerprint": "ours", "hardware": "A100"}, "campaign": "VERIF-1"})
            with patch.object(self.cloud, "sessions", return_value=({"phase0": {
                    "fingerprint": "other", "hardware": "A100"}}, False, ["one session"])):
                with self.assertRaises(ValueError):
                    self.cloud.owned(receipt)
            archive = Path(directory) / "fixture"
            archive.write_bytes(b"public")
            args = argparse.Namespace(action="up", receipt=Path(directory) / "new.json", archive=archive,
                                      scope="pilot", hourly_rate=0, dry_run=False, gpu="A100")
            with patch.object(self.cloud, "validate_archive", return_value={}), patch.object(
                    self.cloud, "sessions", return_value=({"phase0": None}, False, ["another session"])), patch.object(
                    self.cloud, "cli", side_effect=AssertionError("must not allocate")):
                with self.assertRaises(ValueError):
                    self.cloud.execute(args)

    def test_owned_down_remains_available_after_systemexit_upload_reservation(self):
        from verif1_protocol import utc
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            record = {"campaign": "VERIF-1", "session": {"fingerprint": "ours", "hardware": "A100"},
                      "control_binding": "fixture", "utc": utc(), "hourly_rate": 1.5}
            write_new(receipt, record)
            path = receipt.with_suffix(".ledger.jsonl")
            ledger = Ledger(path, "fixture")
            with self.assertRaises(SystemExit):
                ledger.operation("upload", {}, lambda: (_ for _ in ()).throw(SystemExit(1)))
            work_lock = receipt.with_suffix(".lock")
            write_new(work_lock, {"orphaned_after_kill": True})
            args = argparse.Namespace(action="down", receipt=receipt, dry_run=False)
            with patch.object(self.cloud, "owned", return_value=record), patch.object(
                    self.cloud, "cli", return_value="stopped") as cli:
                with self.cloud.cleanup_mutex(receipt.with_suffix(".cleanup.lock")):
                    with self.assertRaises(OSError):
                        self.cloud.execute(args)
                    cli.assert_not_called()
                self.cloud.execute(args)
                cli.assert_called_once_with(["stop", "-s", "phase0"])
            with self.assertRaises(ValueError):
                Ledger(path, "fixture")  # Cleanup never changes or authorizes the incomplete campaign.
            self.assertEqual(len(list(receipt.parent.glob("receipt.down.*.ledger.jsonl"))), 1)
            self.assertTrue(work_lock.exists())  # Cleanup does not steal an orphaned generation lock.

    def test_cleanup_shared_os_lock_is_released_after_process_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "cleanup.lock"
            module = Path(self.cloud.__file__).parent
            script = (f"import sys;sys.path.insert(0,{str(module)!r});"
                      "from verif1_campaign import cleanup_mutex;from pathlib import Path\n"
                      f"with cleanup_mutex(Path({str(lock)!r})):\n"
                      "    print('locked',flush=True)\n    sys.stdin.readline()\n")
            child = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaises(OSError):
                    with self.cloud.cleanup_mutex(lock):
                        self.fail("concurrent cleanup acquired the shared lock")
                child.kill()
                child.wait(timeout=10)
                with self.cloud.cleanup_mutex(lock):
                    self.assertTrue(lock.exists())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdin.close()
                child.stdout.close()
                child.stderr.close()


if __name__ == "__main__":
    unittest.main()
