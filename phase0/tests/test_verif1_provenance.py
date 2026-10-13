"""Synthetic preparation and strict provenance checks in the real exporter's operational selection."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verif1_prepare as prepare
from verif1_protocol import FAMILIES, MODELS, Ledger, file_sha, sha, write_new


def exported_workflow(output):
    from essaim import code
    from verif1_analyze import CONFIG, freeze
    from verif1_run import validate_bundle

    root = prepare.ROOT
    assert not (root / "docs/12_percees.md").exists()
    hashes = prepare.source_hashes()
    assert set(hashes) == set(prepare.SOURCE_FILES)
    assert hashes == {name: file_sha(root / name) for name in prepare.SOURCE_FILES}
    snapshot, cache = output / "snapshot", output / "cache"
    snapshot.mkdir(parents=True)
    cache.mkdir()
    source = "def f(x):\n    return x\n"
    program = code.prog_id(source)
    for bench, count in prepare.COUNTS.items():
        rows, execution = [], []
        generations = {model: [] for model in MODELS}
        for index in range(count):
            qid = f"HumanEval/{index}" if bench == "humanevalplus" else f"synthetic/dev/{index}"
            row = {"id": qid, "prompt": "def f(x):\n    '''Return the input.'''\n    pass\n",
                   "entry_point": "f", "test_list": ["assert f(1) == 1"], "test_imports": []}
            rows.append(row)
            execution.append({"id": qid, "prog": program, "load": None, "visible": [True],
                              "visible_err": [None], "extra": [], "status": {"visible": "ok", "extra": "ok"}})
            for model in MODELS:
                generations[model].extend({"id": qid, "model": model, "bench": bench, "prog": program,
                                           "code": source, "sample": sample, "n_tokens": 1,
                                           "prompt_tokens": 1, "ms": 1} for sample in range(5))
        def jsonl(path, values):
            path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        jsonl(cache / f"{bench}_all_s0_v2.jsonl", rows)
        jsonl(snapshot / f"codeexec_e11_colab_{bench}_dev.jsonl", execution)
        for model, values in generations.items():
            name = f"code_{model.replace('/', '__')}_colab_{bench}_dev.jsonl"
            jsonl(snapshot / name, values)
            write_new(snapshot / (name + ".meta.json"), {"split": "dev", "model": model, "bench": bench,
                      "revision": "synthetic", "gguf": "synthetic", "weights": "synthetic", "engine": "synthetic"})
    for index in range(26):
        (snapshot / f"synthetic_{index}_dev.jsonl").write_text("{}\n", encoding="utf-8")
    registry = {path.name: file_sha(path) for path in snapshot.iterdir()}
    assert len(registry) == 56
    registry_path = root / "phase0/results/percees_code_snapshot_copy.json"
    write_new(registry_path, {"source_and_snapshot_sha256": registry})
    prepared = output / "prepared"
    plan = prepare.prepare(snapshot, cache, prepared)
    bundle = json.loads((prepared / "verif1_dev_bundle.json").read_text(encoding="utf-8"))
    validate_bundle(bundle, plan)
    assert plan["sources"] == hashes
    analysis = output / "analysis"
    analysis.mkdir()
    summary = {"status": "complete_real_dev_exploratory", "config": CONFIG,
               "bindings": {"code_sha256": sha(hashes), "plan_sha256": file_sha(prepared / "verif1_dev_plan.json"),
                            "data_sha256": sha(bundle)}, "run_ledger_sha256": "a" * 64,
               "freeze_choices": {"V06": {"configuration": "synthetic", "fits": {"synthetic": {"weights": {}}}}}}
    binding = {"bindings": summary["bindings"], "run_ledger_sha256": summary["run_ledger_sha256"], "config": CONFIG}
    Ledger(analysis / "verif1_analysis_ledger.jsonl", sha(binding)).operation(
        "analyze_dev", {}, lambda: {"summary": summary})
    write_new(analysis / "verif1_summary_dev.json", summary)
    freeze(analysis, output / "frozen")
    frozen = file_sha(output / "frozen/verif1_V06_frozen_dev.json")
    freeze(analysis, output / "frozen")
    assert file_sha(output / "frozen/verif1_V06_frozen_dev.json") == frozen
    rejected = []
    for mutation in ("missing_hash", "wrong_hash", "extra_private_hash"):
        changed = copy.deepcopy(plan)
        name = prepare.SOURCE_FILES[0]
        if mutation == "missing_hash":
            changed["sources"].pop(name)
        elif mutation == "wrong_hash":
            changed["sources"][name] = "0" * 64
        else:
            changed["sources"]["docs/12_percees.md"] = "0" * 64
        try:
            validate_bundle(bundle, changed)
        except ValueError:
            rejected.append(mutation)
        else:
            raise AssertionError(mutation)
    source_path = root / prepare.SOURCE_FILES[0]
    original = source_path.read_bytes()
    try:
        source_path.write_bytes(original + b"\n# Synthetic provenance mutation.\n")
        for action in (lambda: validate_bundle(bundle, plan), lambda: freeze(analysis, output / "changed-freeze")):
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError("changed source accepted")
        rejected.append("changed_source_validation_and_freeze")
        source_path.unlink()
        try:
            prepare.source_hashes()
        except FileNotFoundError:
            rejected.append("missing_required_source")
        else:
            raise AssertionError("missing source accepted")
    finally:
        source_path.write_bytes(original)
    return {"source_hashes": hashes, "questions": len(bundle["questions"]), "frozen_sha256": frozen,
            "rejected": rejected, "synthetic_only": True}


class VerifExportProvenanceTests(unittest.TestCase):
    def test_selected_tree_prepares_validates_freezes_and_rejects_provenance_mutations(self):
        exporter = prepare.ROOT / "tools/export_public.py"
        if not exporter.exists():
            self.skipTest("private exporter absent (public tree)")
        spec = importlib.util.spec_from_file_location("verif_provenance_exporter", exporter)
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        tracked = export.git_files()
        required = set(prepare.SOURCE_FILES)
        for name in required:
            self.assertIn(name, tracked)
            self.assertTrue(export.selected(name), name)
        self.assertFalse(export.selected("docs/12_percees.md"))
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / "export"
            # Keep real selection, safe reads, rewrites and scanner; omit scientific result payloads.
            script = ("import importlib.util,sys;from pathlib import Path;"
                      "sys.path.insert(0,str(Path(sys.argv[1]).resolve().parents[1]/'phase0'));"
                      "spec=importlib.util.spec_from_file_location('exporter',sys.argv[1]);"
                      "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
                      "tracked=m.git_files();required=set(__import__('verif1_prepare').SOURCE_FILES);"
                      "m.git_files=lambda:{n:v for n,v in tracked.items() if n in required or "
                      "(n.startswith('phase0/') and n.endswith('.py') and not n.startswith('phase0/results/'))};"
                      "sys.exit(m.main([sys.argv[2],'--allow-dirty']))")
            result = subprocess.run([sys.executable, "-c", script, str(exporter), str(tree)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((tree / "docs/12_percees.md").exists())
            script = ("import sys,json;from pathlib import Path;sys.path.insert(0,str(Path.cwd()/'phase0/tests'));"
                      "from test_verif1_provenance import exported_workflow;"
                      "print(json.dumps(exported_workflow(Path(sys.argv[1])),sort_keys=True))")
            result = subprocess.run([sys.executable, "-c", script, str(Path(directory) / "workflow")],
                                    cwd=tree, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            proof = json.loads(result.stdout)
            self.assertEqual(set(proof["source_hashes"]), required)
            self.assertEqual(proof["questions"], 270)
            self.assertEqual(len(proof["rejected"]), 5)


if __name__ == "__main__":
    unittest.main()
