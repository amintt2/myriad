"""Synthetic-only scientific gate and cluster regressions; no model or dataset execution."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aa_timing
import analyze_e12 as ae
import e12_costs as costs
import validate_e12 as ve
from essaim import answers, data, gpqa, scicode
from exec_scicode import gen_digest, jobs_for_model, jobs_for_oracle
from tests.test_aa import fake_rows, synthetic_problem


class Delivery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.probs = {sp: [{**synthetic_problem(), "split": sp}] for sp in ("dev", "test")}
        self.ident = {"dataset": "synthetic", "revision": "r", "sha256": "0" * 64, "license": "test"}
        self.delivery = {"schema": 1, "complete": True, "plan": "aa-1", "benches": ["scicode"], "files": {},
                         "identities": {}, "attempts": [{"campaign_id": "a" * 32, "interrupted": True,
                         "resumed": True, "evidence": {"attempt/log": "c" * 64}}],
                         "execution_environment": {"isolation": {"sandbox": "synthetic"}, "python": "3.12",
                                                   "numpy": "test", "scipy": "test"}}
        sources = {name: {"sha256": ve.sha((Path(ve.__file__).parent / name).read_bytes()),
                          "size": (Path(ve.__file__).parent / name).stat().st_size} for name in ve.RUNTIME}
        for name in ("e12_campaign_sources.json", "e12_campaign_sources.local.json"):
            self.write(name, {"schema": 1, "plan": "aa-1", "benches": ["scicode"], "sources": sources})
        for m in ve.MODELS:
            self.delivery["identities"][m] = {"model": m, "revision": "d" * 40, "gguf": ve.SOLO[m][1],
                "weights": {"name": ve.SOLO[m][1], "size": 100, "sha256": "e" * 64}, "engine": "synthetic",
                "backend": "llama.cpp (--jinja, OpenAI chat)"}
        for sp in ("dev", "test"):
            for m in ve.MODELS + (["oracle"] if sp == "dev" else []):
                gman, gen = None, None
                if m != "oracle":
                    name = f"aa_{m.replace('/', '__')}_t_scicode_{sp}.jsonl"
                    gman = {**self.delivery["identities"][m], "bench": "scicode", "split": sp, "data": self.ident,
                        "n": 1, "n_steps": scicode.COUNTS[sp][1], "prompt": scicode.PROMPT_VERSION,
                        "max_tokens": 2048, "temperature": 0.0, "seed": 7, "thinking": False,
                        "ctx_per_slot": 8192, "background": True, "official_commit": scicode.OFFICIAL_COMMIT,
                        "skipped_steps": sorted(scicode.SKIPPED_SHA256), "skipped_code": scicode.SKIPPED_SHA256,
                        "measurements": "e12-wall-v2"}
                    rows = [{"id": s["number"], "problem": "7", "step": scicode.step_number(s), "model": m,
                             "bench": "scicode", "text": f"```python\n{s['reference']}\n```", "extract": "fenced",
                             "finish": "stop", "reasoning": False, "n_tokens": 10, "prompt_tokens": 30,
                             "ms": 1, "wall_s": 1.0} for s in self.probs[sp][0]["steps"]]
                    gen = {r["id"]: r for r in rows}
                    self.write(name, rows)
                    self.write(name + ".meta.json", gman)
                    events = [aa_timing.header(name, gman)] + [{"kind": "call", "bench": "scicode", "id": r["id"],
                        "problem": "7", "returned": True, "error_type": None, "n_tokens": 10,
                        "prompt_tokens": 30, "wall_s": 1.0} for r in rows] + [
                        {"kind": "problem", "problem": "7", "complete": True, "resumed_steps": 0, "wall_s": 3.5},
                        {"kind": "batch", "bench": "scicode", "split": sp, "new": 3, "wall_s": 4.0}]
                    self.write(name + ".timing.jsonl", events)
                name = f"sciexec_{m.replace('/', '__')}_t_scicode_{sp}.jsonl"
                man = {**self.delivery["execution_environment"], "bench": "scicode", "split": sp, "model": m,
                       "n": 3, "data": self.ident, "h5": {"sha256": scicode.H5_SHA256, "size": 1049345865},
                       "official_commit": scicode.OFFICIAL_COMMIT, "harness": ve.HARNESS_VERSION,
                       "timeout_s": 300, "skipped_code": scicode.SKIPPED_SHA256, "oracle": m == "oracle",
                       "generation": {**gman, "rows_sha256": gen_digest(gen)} if gman else None}
                jobs = jobs_for_model(self.probs[sp], gen) if gen else jobs_for_oracle(self.probs[sp])
                rows = [{"id": rid, "problem": pid, "model": m, "ok": True, "err": None, "msg": "",
                         "status": "ok", "ms": 1, "script": ve.sha(script.encode())[:16]} for rid, pid, script in jobs]
                self.write(name, rows)
                self.write(name + ".meta.json", man)
        self.path = self.root / "delivery.json"

    def write(self, name, obj):
        path = self.root / name
        raw = ("".join(json.dumps(r) + "\n" for r in obj) if isinstance(obj, list) else json.dumps(obj)).encode()
        path.write_bytes(raw)
        self.delivery["files"][name] = ve.sha(raw)

    def pin(self):
        self.path.write_text(json.dumps(self.delivery), encoding="utf-8")
        return ve.sha(self.path.read_bytes())

    def validate(self):
        with mock.patch.object(ve, "official", return_value=(self.probs, self.ident)):
            return ve.validate(self.root, "_t", self.path, self.pin(), self.root)

    def add_gpqa(self, benches=None):
        benches = ["gpqa", "scicode"] if benches is None else benches
        assert benches == ["gpqa", "scicode"]  # Same availability as the campaign provenance writer.
        self.delivery["benches"] = benches
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(fake_rows(1)[0]))
        writer.writeheader()
        writer.writerows(fake_rows(198))
        raw = out.getvalue().encode()
        self.csv_path = self.root / "synthetic.csv"
        self.csv_path.write_bytes(raw)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(gpqa, "PINNED", {"size": len(raw), "git_blob_sha1": gpqa.git_blob_sha1(raw)}).start()
        items = gpqa.parse(raw)
        ident = {"dataset": data.REPOS["gpqa"], "revision": data.REVISIONS["gpqa"],
                 "license": data.LICENSES["gpqa"], "sha256": ve.sha(data._canonical(items))}
        for name in ("e12_campaign_sources.json", "e12_campaign_sources.local.json"):
            sources = ve.read(self.root / name)
            sources["benches"] = benches
            sources["sources"].update({n: {"size": (Path(ve.__file__).parent / n).stat().st_size,
                "sha256": ve.sha((Path(ve.__file__).parent / n).read_bytes())} for n in ve.GPQA_RUNTIME})
            self.write(name, sources)
        peer = list(ve.FAMILIES.values())[-1]
        self.write("e4_summary_t.json", {"benches": {"mmlupro": {
            "weights": {f: 1.0 for f in ve.FAMILIES},
            "p_dev": {f: .6 if m == peer else .5 for f, m in ve.FAMILIES.items()}, "best_dev": peer}}})
        for m in ve.MODELS:
            name = f"aa_{m.replace('/', '__')}_t_gpqa_all.jsonl"
            man = {**self.delivery["identities"][m], "bench": "gpqa", "split": "all", "data": ident,
                   "n": 198, "prompt": ve.MC_PROMPT, "max_tokens": 1024, "temperature": 0.0, "seed": 7,
                   "thinking": False, "ctx_per_slot": 8192, "logprobs": False, "measurements": ve.MEASUREMENTS}
            rows = [{"id": it["id"], "model": m, "bench": "gpqa", "text": "A", "answer": "A",
                     "gold": answers.gold("gpqa", it), "finish": "stop", "n_tokens": 1, "ms": 1,
                     "wall_s": 1.0, "reasoning": False} for it in items]
            self.write(name, rows)
            self.write(name + ".meta.json", man)
            self.write(name + ".timing.jsonl", [aa_timing.header(name, man)] + [
                {"kind": "call", "bench": "gpqa", "id": it["id"], "returned": True,
                 "error_type": None, "n_tokens": 1, "prompt_tokens": None, "wall_s": 1.0} for it in items] + [
                {"kind": "batch", "bench": "gpqa", "new": 198, "wall_s": 40.0}])

    def cli(self, benches, output):
        argv = ["analyze_e12.py", "--suffix", "_t", "--delivery", str(self.path), "--delivery-sha256",
                self.pin(), "--results-dir", str(self.root), "--output-dir", str(output),
                "--dataset-dir", str(self.root)]
        if benches is not None:
            argv += ["--benches", *benches]
        with mock.patch.object(gpqa, "local_path", return_value=getattr(self, "csv_path", None)), \
                mock.patch.object(gpqa, "diamond", side_effect=AssertionError("no downloader/cache")), \
                mock.patch.object(ve, "official", return_value=(self.probs, self.ident)), \
                mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            ae.main()

    def test_gpqa_report_filter_and_frozen_e4_dev(self):
        self.add_gpqa()
        output = self.root / "output"
        self.cli(["gpqa"], output)
        summary = ve.read(output / "e12_summary_t.json")
        self.assertNotIn("scicode", summary)
        self.assertEqual(summary["gpqa"]["best_peer"], list(ve.FAMILIES.values())[-1])
        self.assertEqual(summary["gpqa"]["frozen_from"], "e4 mmlupro dev")
        self.assertTrue(set(ve.MODELS) <= set(summary["gpqa"]["systems"]))
        self.assertNotIn(b"synthetic question", (output / "e12_report_t.md").read_bytes())

    def test_same_combined_delivery_supports_both_filters_together_and_auto(self):
        self.add_gpqa()
        self.pin()
        names = list(self.delivery["files"]) + ["delivery.json"]
        before = {n: ve.sha((self.root / n).read_bytes()) for n in names}
        for benches in (["gpqa"], ["scicode"], ["gpqa", "scicode"], None):
            self.cli(benches, self.root / "output")
            summary = ve.read(self.root / "output/e12_summary_t.json")
            selected = ["gpqa", "scicode"] if benches is None else benches
            self.assertEqual(summary["provenance"]["available_benches"], ["gpqa", "scicode"])
            self.assertEqual(summary["provenance"]["report_benches"], selected)
            self.assertEqual({b for b in ("gpqa", "scicode") if b in summary}, set(selected))
            self.assertEqual(before, {n: ve.sha((self.root / n).read_bytes()) for n in names})
            for name in ("e12_report_t.md", "e12_summary_t.json"):
                self.assertNotIn(b"synthetic question", (self.root / "output" / name).read_bytes())

    def test_hidden_benchmark_reference_and_protocol_are_still_validated(self):
        self.add_gpqa()
        output = self.root / "output"
        output.mkdir()
        products = [output / name for name in ("e12_report_t.md", "e12_summary_t.json", "e12_t_time.png")]
        for path in products:
            path.write_bytes(b"valid prior product")
        reference = ve.REFS[-1].replace("/", "__")
        for selected, name, field in ((["gpqa"], f"sciexec_{reference}_t_scicode_test.jsonl", "timeout_s"),
                                      (["scicode"], f"aa_{reference}_t_gpqa_all.jsonl", "max_tokens")):
            digest = self.delivery["files"].pop(name)
            with self.subTest(selected=selected, defect="missing reference"), self.assertRaises(SystemExit):
                self.cli(selected, output)
            self.delivery["files"][name] = digest
            meta_name = name + ".meta.json"
            original = ve.read(self.root / meta_name)
            self.write(meta_name, {**original, field: 1})  # Re-pin inventory: the scientific gate must reject it.
            with self.subTest(selected=selected, defect="forged protocol"), self.assertRaises(SystemExit):
                self.cli(selected, output)
            self.write(meta_name, original)
            self.assertTrue(all(p.read_bytes() == b"valid prior product" for p in products))

    def test_absent_requested_benchmark_and_duplicate_filter_preserve_products(self):
        output = self.root / "output"
        output.mkdir()
        prior = output / "e12_summary_t.json"
        prior.write_bytes(b"prior")
        for selected in (["gpqa"], ["gpqa", "scicode"], ["scicode", "scicode"]):
            with self.subTest(selected=selected), self.assertRaises(SystemExit):
                self.cli(selected, output)
        with self.assertRaises(SystemExit):
            ve.validate(self.root, "_t", self.path, self.pin(), self.root, [])
        self.assertEqual(prior.read_bytes(), b"prior")

    def test_availability_is_canonical_and_source_attestation_must_agree(self):
        self.add_gpqa()
        for available in (["gpqa"], [], ["scicode", "gpqa"], ["gpqa", "scicode", "scicode"], "scicode"):
            self.delivery["benches"] = available
            with self.subTest(available=available), self.assertRaises(SystemExit):
                self.cli(["scicode"], self.root / "output")
        self.delivery["benches"] = ["gpqa", "scicode"]
        for name in ("e12_campaign_sources.json", "e12_campaign_sources.local.json"):
            source = ve.read(self.root / name)
            self.write(name, {**source, "benches": ["scicode"]})
        with self.assertRaises(SystemExit):
            self.cli(["scicode"], self.root / "output")

    def test_hidden_bench_still_requires_all_validation_inputs(self):
        self.add_gpqa()
        csv_path = self.csv_path
        self.csv_path = None
        with self.assertRaises(SystemExit):
            self.cli(["scicode"], self.root / "output")
        self.csv_path = csv_path
        with mock.patch.object(ve, "official", side_effect=SystemExit("synthetic sources unavailable")):
            with self.assertRaises(SystemExit):
                ve.validate(self.root, "_t", self.path, self.pin(), None, ["gpqa"], csv_path)

    def test_gpqa_invalid_delivery_and_campaign_mix_refused(self):
        self.add_gpqa(["gpqa", "scicode"])
        output = self.root / "output"
        output.mkdir()
        prior = output / "e12_summary_t.json"
        prior.write_bytes(b"prior")
        self.delivery["files"].pop(f"aa_{ve.REFS[-1].replace('/', '__')}_t_gpqa_all.jsonl")
        for benches in (["gpqa", "scicode"], ["gpqa"], ["scicode"]):
            with self.assertRaises(SystemExit):
                self.cli(benches, output)
        self.assertEqual(prior.read_bytes(), b"prior")

    def test_gpqa_forged_types_protocol_timing_and_dev_frozen(self):
        self.add_gpqa()
        name = f"aa_{ve.MODELS[0].replace('/', '__')}_t_gpqa_all.jsonl"
        rows = ve.read(self.root / name, lines=True)
        for key, value in (("id", 1), ("reasoning", 1), ("gold", "Z"), ("answer", True), ("n_tokens", True)):
            changed = [dict(r) for r in rows]
            changed[0][key] = value
            self.write(name, changed)
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.cli(["gpqa"], self.root / "output")
        self.write(name, rows)
        timing_name = name + ".timing.jsonl"
        trace = ve.read(self.root / timing_name, lines=True)
        trace[0]["protocol_sha256"] = "0" * 64
        self.write(timing_name, trace)
        with self.assertRaises(SystemExit):
            self.cli(["gpqa"], self.root / "output")

    def test_determining_sources_missing_or_divergent(self):
        original = ve.read(self.root / "e12_campaign_sources.json")
        for key in ("essaim/llamacpp.py", "pyproject.toml"):
            for absent in (False, True):
                sources = json.loads(json.dumps(original))
                if absent:
                    sources["sources"].pop(key)
                else:
                    sources["sources"][key]["sha256"] = "0" * 64
                for name in ("e12_campaign_sources.json", "e12_campaign_sources.local.json"):
                    self.write(name, sources)
                with self.subTest(key=key, absent=absent), self.assertRaises(SystemExit):
                    self.validate()

    def test_batch_bounds_clean_concurrent_and_resumed(self):
        events = self.validate()["traces"]["test"][ve.MODELS[0]]
        clean = [{"interrupted": False, "resumed": False}]
        self.assertTrue(costs.accounting(events, clean, 1, 6)["duration_total_known"])
        for seconds in (0.0, 3.0):
            events[-1]["wall_s"] = seconds
            c = costs.accounting(events, clean, 1, 6)
            self.assertFalse(c["duration_total_known"])
            self.assertTrue(c["duration_contradictions"])
            self.assertIsNone(costs.energy_scenario(c["total_generation_wall_s"], 100, .25))
        events[-1]["wall_s"] = 4
        events[-2]["wall_s"] = 2
        self.assertFalse(costs.accounting(events, clean, 1, 6)["duration_total_known"])
        events[-2]["wall_s"] = 100  # Old worker belongs to an earlier attempt.
        events.append({"kind": "batch", "new": 0, "wall_s": 0.1})
        c = costs.accounting(events, clean + [{"resumed": True, "interrupted": False}], 1, 6)
        self.assertFalse(c["duration_total_known"])
        self.assertEqual(c["duration_contradictions"], [])
        concurrent = [dict(e) for e in events[:5]]
        concurrent[-2]["wall_s"] = 3.5
        concurrent += [{**e, "id": e.get("id", "") + "x", "problem": "8"} for e in concurrent[:4]]
        concurrent[-1]["wall_s"] = 3.5
        concurrent[4]["new"] = 6
        self.assertTrue(costs.accounting(concurrent, clean, 2, 2)["duration_total_known"])

    def test_gpqa_requires_admitted_csv_and_frozen_dev_provenance(self):
        self.add_gpqa()
        csv_path = self.csv_path
        self.csv_path = None
        with self.assertRaises(SystemExit):
            self.cli(["gpqa"], self.root / "output")
        self.csv_path = csv_path
        raw = csv_path.read_bytes()
        csv_path.write_bytes(raw + b"changed")
        with self.assertRaises(SystemExit):
            self.cli(["gpqa"], self.root / "output")
        csv_path.write_bytes(raw)
        self.delivery["files"].pop("e4_summary_t.json")
        with self.assertRaises(SystemExit):
            self.cli(["gpqa"], self.root / "output")
        self.write("e4_summary_t.json", {"benches": {"mmlupro": {
            "weights": {f: 1.0 for f in ve.FAMILIES}, "p_dev": {f: .5 for f in ve.FAMILIES},
            "best_dev": list(ve.FAMILIES.values())[-1]}}})
        with self.assertRaises(SystemExit):
            self.cli(["gpqa"], self.root / "output")

    def test_stale_figures_cleanup_only_after_all_products_built(self):
        output = self.root / "output"
        output.mkdir()
        managed = [output / f"e12_t_{axis}.{ext}" for axis in ("time", "cost") for ext in ("svg", "png", "pdf")]
        foreign = [output / "e12_other_time.svg", output / "foreign.pdf"]
        for path in managed + foreign:
            path.write_bytes(b"prior")
        self.delivery["complete"] = False
        with self.assertRaises(SystemExit):
            self.cli(["scicode"], output)
        self.assertTrue(all(p.read_bytes() == b"prior" for p in managed + foreign))
        self.delivery["complete"] = True
        with mock.patch.object(costs, "figures", side_effect=RuntimeError("synthetic construction failure")):
            with self.assertRaises(RuntimeError):
                self.cli(["scicode"], output)
        self.assertTrue(all(p.read_bytes() == b"prior" for p in managed + foreign))
        self.cli(["scicode"], output)
        self.assertTrue(all(not p.exists() for p in managed))
        self.assertTrue(all(p.read_bytes() == b"prior" for p in foreign))
        for path in managed:
            path.write_bytes(b"prior")

        def time_only(systems, accounting, destination, *scenario):
            names = []
            for ext in ("svg", "png", "pdf"):
                path = Path(str(destination) + f"_time.{ext}")
                path.write_bytes(b"new synthetic figure")
                names.append(path.name)
            return names

        with mock.patch.object(costs, "figures", side_effect=time_only):
            self.cli(["scicode"], output)
        for path in managed:
            if "_time." in path.name:
                self.assertEqual(path.read_bytes(), b"new synthetic figure")
            else:
                self.assertFalse(path.exists())
        self.assertTrue(all(p.read_bytes() == b"prior" for p in foreign))

    def test_complete_and_resumed_costs_unknown(self):
        v = self.validate()
        c = costs.accounting(v["traces"]["test"][ve.MODELS[0]], v["attempts"], 1, 6)
        self.assertEqual(c["observed_completion_tokens_sum"], 30)
        self.assertIsNone(c["total_completion_tokens"])
        self.assertIsNone(c["batch_amortized_seconds_per_problem"])
        self.assertEqual(len(v["data_by"]["test"]), 13)

    def test_null_error_counts_stay_unknown_in_clean_attempt(self):
        v = self.validate()
        events = v["traces"]["test"][ve.MODELS[0]]
        clean = [{"interrupted": False, "resumed": False}]
        c = costs.accounting(events, clean, 1, 6)
        self.assertTrue(c["attempt_total_known"])
        self.assertEqual(c["total_completion_tokens"], 30)
        self.assertEqual(c["batch_amortized_seconds_per_problem"], 4)
        self.assertEqual(c["observed_call_seconds_sum"], 3)
        events[0].update(returned=False, n_tokens=None, prompt_tokens=None, error_type="RuntimeError")
        c = costs.accounting(events, clean, 1, 6)
        self.assertEqual(c["observed_completion_tokens_sum"], 20)
        self.assertEqual(c["errors"], 1)
        self.assertEqual(c["unknown_completion_token_calls"], 1)
        self.assertIsNone(c["total_completion_tokens"])
        self.assertEqual(c["total_generation_wall_s"], 4)  # A null token count does not erase a measured batch.
        self.assertTrue(c["duration_total_known"])
        self.assertFalse(c["completion_token_total_known"])

    def test_inventory_and_delivery_hashes_are_not_self_repaired(self):
        pinned = self.pin()
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaises(SystemExit):
            ve.validate(self.root, "_t", self.path, pinned, self.root)
        name = f"aa_{ve.MODELS[0].replace('/', '__')}_t_scicode_test.jsonl"
        (self.root / name).write_bytes((self.root / name).read_bytes() + b" ")
        with self.assertRaises(SystemExit):
            self.validate()

    def test_missing_reference_and_incomplete(self):
        name = f"sciexec_{ve.REFS[-1].replace('/', '__')}_t_scicode_test.jsonl"
        self.delivery["files"].pop(name)
        with self.assertRaises(SystemExit):
            self.validate()
        self.delivery["complete"] = False
        with self.assertRaises(SystemExit):
            self.validate()

    def test_forged_types_ids_statuses_programs_and_timing(self):
        name = f"sciexec_{ve.MODELS[0].replace('/', '__')}_t_scicode_test.jsonl"
        original = ve.read(self.root / name, lines=True)
        for field, value in (("ok", 1), ("id", 7), ("problem", "99"), ("script", "f" * 16),
                             ("status", "none"), ("ms", True), ("err", "AssertionError")):
            with self.subTest(field=field):
                rows = [dict(r) for r in original]
                rows[0][field] = value
                self.write(name, rows)
                with self.assertRaises(SystemExit):
                    self.validate()
        self.write(name, original + [original[0]])
        with self.assertRaises(SystemExit):
            self.validate()
        self.write(name, original)
        timing_name = f"aa_{ve.MODELS[0].replace('/', '__')}_t_scicode_test.jsonl.timing.jsonl"
        trace = ve.read(self.root / timing_name, lines=True)
        trace[0]["protocol_sha256"] = "f" * 64
        self.write(timing_name, trace)
        with self.assertRaises(SystemExit):
            self.validate()

    def test_modified_provenance_and_protocol(self):
        name = "e12_campaign_sources.json"
        source = ve.read(self.root / name)
        source["sources"]["run_aa.py"]["sha256"] = "0" * 64
        self.write(name, source)
        with self.assertRaises(SystemExit):
            self.validate()
        self.write("e12_campaign_sources.local.json", source)
        with self.assertRaises(SystemExit):
            self.validate()

    def test_oracle_invalid_harness_and_test_oracle_refused(self):
        name = "sciexec_oracle_t_scicode_dev.jsonl"
        rows = ve.read(self.root / name, lines=True)
        rows[0].update(ok=False, err="AssertionError")
        self.write(name, rows)
        with self.assertRaisesRegex(SystemExit, "90"):
            self.validate()
        self.write("sciexec_oracle_t_scicode_test.jsonl", rows)
        with self.assertRaises(SystemExit):
            self.validate()

    def test_generation_protocol_boolean_and_missing_counters(self):
        name = f"aa_{ve.MODELS[0].replace('/', '__')}_t_scicode_test.jsonl"
        rows = ve.read(self.root / name, lines=True)
        rows[0].pop("n_tokens")
        self.write(name, rows)
        with self.assertRaises(SystemExit):
            self.validate()
        rows[0]["n_tokens"] = 10
        self.write(name, rows)
        meta_name = name + ".meta.json"
        meta = ve.read(self.root / meta_name)
        meta["background"] = 1
        self.write(meta_name, meta)
        with self.assertRaises(SystemExit):
            self.validate()

    def test_dataset_source_is_offline_and_hash_pinned(self):
        (self.root / "problems_dev.jsonl").write_bytes(b'{"problem_id":1}\n')
        with self.assertRaises(SystemExit):
            ve.official(self.root)

    def test_cli_complete_synthetic_delivery_is_deterministic(self):
        argv = ["analyze_e12.py", "--suffix", "_t", "--benches", "scicode", "--delivery", str(self.path),
                "--delivery-sha256", self.pin(), "--dataset-dir", str(self.root)]
        with mock.patch.object(ae, "RESULTS", self.root), \
                mock.patch.object(ve, "official", return_value=(self.probs, self.ident)), \
                mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            ae.main()
            report = (self.root / "e12_report_t.md").read_bytes()
            summary = (self.root / "e12_summary_t.json").read_bytes()
            ae.main()
        self.assertEqual(report, (self.root / "e12_report_t.md").read_bytes())
        self.assertEqual(summary, (self.root / "e12_summary_t.json").read_bytes())
        self.assertFalse(json.loads(summary)["scicode"]["figures"])
        self.assertNotIn(b"synthetic question", report)

    def test_json_duplicate_keys_and_truncation(self):
        for raw in (b'{"ok":true,"ok":false}\n', b'{"id":"7.1"}\n{"id":'):
            path = self.root / "bad.jsonl"
            path.write_bytes(raw)
            with self.assertRaises(SystemExit):
                ve.read(path, lines=True)

    def test_cli_refusal_preserves_reports(self):
        targets = [self.root / f"e12_{kind}_t.{ext}" for kind, ext in (("report", "md"), ("summary", "json"))]
        for target in targets:
            target.write_bytes(b"valid prior artifact")
        self.delivery["complete"] = False
        with mock.patch.object(ae, "RESULTS", self.root), mock.patch.object(ve, "official") as official, \
                mock.patch.object(sys, "argv", ["analyze_e12.py", "--suffix", "_t", "--benches", "scicode",
                    "--delivery", str(self.path), "--delivery-sha256", self.pin(), "--dataset-dir", str(self.root)]):
            with self.assertRaises(SystemExit):
                ae.main()
            official.assert_not_called()
        self.assertTrue(all(t.read_bytes() == b"valid prior artifact" for t in targets))

    def test_dev_only_tie_order_and_reproducibility(self):
        v = self.validate()
        # Test winners cannot change the peer selected by tied dev scores.
        for row in v["data_by"]["test"][ve.MODELS[0]]["rows"].values():
            row["ok"] = False
        with mock.patch.object(ae, "RESULTS", self.root):
            summaries = []
            for _ in range(2):
                summary = {}
                ae.sci_section("_t", summary, v)
                summaries.append(summary)
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["scicode"]["best_dev_peer"], ve.MODELS[0])


class Clusters(unittest.TestCase):
    def test_cluster_dependence_and_exact_step_weighting(self):
        groups = {"a": [True] * 20, "b": [False]}
        result = costs.cluster_interval(groups)
        self.assertAlmostEqual(result["mean"], 100 * 20 / 21)
        self.assertEqual((result["lo95"], result["hi95"]), (0, 100))
        self.assertEqual(result, costs.cluster_interval(groups))
        self.assertEqual(costs.cluster_interval(groups, groups)["hi95"], 0)

    def test_incompatible_step_oracle_is_not_chain_oracle(self):
        coverage, chains = costs.oracle_chains({"a": {"p": [True, False]}, "b": {"p": [False, True]}})
        self.assertEqual(coverage, {"p": [True, True]})
        self.assertEqual(chains, {"p": False})

    def test_figures_are_reproducible_and_unknowns_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            system = {"synthetic/model": {"steps_pct": 50}}
            known = {"synthetic/model": {"batch_amortized_seconds_per_problem": 2}}
            for sub in ("one", "two"):
                (root / sub).mkdir()
                names = costs.figures(system, known, root / sub / "mock", 100, .25)
                self.assertEqual(len(names), 6)
            for name in names:
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            self.assertEqual(costs.figures(system, {"synthetic/model": {
                "batch_amortized_seconds_per_problem": None}}, root / "unknown"), [])
            self.assertIsNone(costs.energy_scenario(None, 100, .25))


if __name__ == "__main__":
    unittest.main()
