"""Offline campaign guards and cost attribution: no candidate code is executed."""
import ast
import contextlib
import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_code as ac
import validate_code as vc
from e11_costs import measure, observable_costs
from essaim import code

ROOT = Path(__file__).resolve().parents[2]


class Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = patch.object(code, "RESULTS", self.root)
        self.paths.start()
        self.items = [{"id": "HumanEval/0", "entry_point": "f", "prompt": "def f(x):\n    pass\n",
                       "test": "def check(candidate):\n    inputs = [(1,)]\n    results = [1]\n"
                               "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                               "        assert candidate(*inp) == exp\n"}]
        self.identity = {"sha256": "test"}
        self.gp = code.gen_path("m", "_t", "humanevalplus", "dev")
        self.ep = code.exec_path("e11", "_t", "humanevalplus", "dev")
        text = "```python\ndef f(x):\n    return x\n```"
        src, extract = code.extract_code(text, "f")
        self.gs = [{"id": "HumanEval/0", "sample": s, "model": "m", "bench": "humanevalplus",
                    "seed": code.sample_seed(s), "temperature": 0.0 if s == 0 else .8,
                    "text": text, "code": src, "prog": code.prog_id(src), "extract": extract,
                    "finish": "stop", "reasoning": False, "n_tokens": 10, "prompt_tokens": 20, "ms": 100}
                   for s in range(5)]
        self.gm = {"model": "m", "bench": "humanevalplus", "split": "dev", "n": 1, "data": self.identity,
                   "prompt": code.PROMPT_VERSION, "max_tokens": code.MAX_TOKENS, "thinking": False,
                   "ctx_per_slot": 3072, "samples": 4, "greedy": {"temperature": 0.0, "seed": 7},
                   "sampling": {**code.SAMPLING, "seeds": [1001, 1002, 1003, 1004]}, "revision": "a" * 40,
                   "gguf": "m.gguf", "weights": {"name": "m.gguf", "size": 1, "sha256": "b" * 64},
                   "engine": "test", "backend": "llama.cpp (--jinja, OpenAI chat)"}
        self.em = {"bench": "humanevalplus", "split": "dev", "n": 1, "data": self.identity,
                   "tests": code.TESTS_VERSION, "extract": code.EXTRACT_VERSION, "extra_n": 16,
                   "models": ["m"], "reference_only": False, "python": "3", "numpy": "2",
                   "isolation": {"version": "sandbox-v3", "platform": "linux", "landlock": "unavailable",
                                 "prefix": ["/usr/bin/unshare", "--net", "--"], "per_case_timeout": True}}
        self.rs = [{"id": "HumanEval/0", "prog": p, "bench": "humanevalplus", "load": None,
                    "visible": [], "visible_err": [], "extra": [], "hidden": {"pass": True, "n": 1,
                    "n_pass": 1, "err": None}, "status": dict.fromkeys(("visible", "extra", "hidden"), "ok"), "ms": 1}
                   for p in ("reference", code.prog_id(src))]
        self.reset()

    def tearDown(self):
        self.paths.stop()
        self.tmp.cleanup()

    def write(self, path, value, meta=False):
        path = path.with_name(path.name + ".meta.json") if meta else path
        path.write_text(json.dumps(value) if meta else "".join(json.dumps(r) + "\n" for r in value), encoding="utf-8")

    def reset(self):
        self.write(self.gp, self.gs)
        self.write(self.gp, self.gm, True)
        self.write(self.ep, self.rs)
        self.write(self.ep, self.em, True)

    def validate(self):
        return vc.validate_partition("humanevalplus", "dev", self.items, self.identity, "_t", "e11", ["m"])

    def test_complete_and_deterministic(self):
        a, hashes = self.validate()
        self.assertEqual(a["exec_rows"], 2)
        self.assertEqual(a["generations"]["generations"], 5)
        self.assertEqual((a, hashes), self.validate())

    def test_absent_corrupt_and_duplicate_sources(self):
        self.gp.unlink()
        with self.assertRaisesRegex(SystemExit, "absente"):
            self.validate()
        self.reset()
        self.gp.write_bytes(self.gp.read_bytes() + b'{"id":')
        with self.assertRaisesRegex(SystemExit, "corrompue"):
            self.validate()
        self.write(self.gp, self.gs + [self.gs[0]])
        with self.assertRaisesRegex(SystemExit, "dupliquées"):
            self.validate()

    def test_missing_offline_cache_never_downloads(self):
        with patch.object(vc.data, "DATA", self.root), patch.object(vc.data, "_parquets") as download:
            with self.assertRaisesRegex(SystemExit, "cache offline absent"):
                vc.cache("humanevalplus")
            download.assert_not_called()

    def test_wrong_ids_samples_extraction_and_exec_coverage(self):
        mutations = [(self.gp, self.gs[:-1]), (self.gp, [{**g, "id": "foreign"} for g in self.gs]),
                     (self.gp, [{**g, "prog": "0" * 16} for g in self.gs]), (self.ep, self.rs[1:]),
                     (self.ep, self.rs + [{**self.rs[0], "prog": "foreign"}])]
        for path, rows in mutations:
            self.reset()
            self.write(path, rows)
            with self.subTest(path=path), self.assertRaises(SystemExit):
                self.validate()

    def test_divergent_split_protocol_and_models(self):
        for path, man, field, value in ((self.gp, self.gm, "split", "test"),
                                         (self.gp, self.gm, "max_tokens", 512),
                                         (self.ep, self.em, "models", []),
                                         (self.ep, self.em, "tests", "old")):
            self.reset()
            self.write(path, {**man, field: value}, True)
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.validate()
        with self.assertRaisesRegex(SystemExit, "modèles"):
            ac.check_splits("b", {"manifests": {"a": {}}, "exec": {}}, {"manifests": {}, "exec": {}})

    def test_excluded_reference_does_not_hide_missing_program(self):
        failed = copy.deepcopy(self.rs)
        failed[0]["hidden"] = {"pass": False, "n": 1, "n_pass": 0, "err": "AssertionError"}
        self.write(self.ep, failed)
        self.assertIn("HumanEval/0", self.validate()[0]["excluded"])
        self.write(self.ep, failed[:1])
        with self.assertRaisesRegex(SystemExit, "couverture exec"):
            self.validate()

    def test_unknown_costs_and_cascade_attribution(self):
        self.assertIsNone(measure([{"ms": None}])["ms"]["total"])
        g = {"id": "q", "sample": 0, "ms": 100, "n_tokens": 10, "prompt_tokens": 20, "finish": "stop"}
        gens = {"m": [g], "r": [{**g, "ms": 900, "n_tokens": 90}]}
        costs = observable_costs(["q"], gens, {"cascade|1": [1]}, {"cascade|1": {"q": None}},
                                 {"M": "m"}, "m", "r", lambda q, m: True)
        c = costs["cascade|1"]
        self.assertEqual(c["metrics"]["ms"]["mean_per_problem"], 1000)
        self.assertEqual(c["calls"], 2)

    def test_explicit_single_family_smoke_stays_available(self):
        gp = code.gen_path("m", "_t", "humanevalplus", "test")
        ep = code.exec_path("e11", "_t", "humanevalplus", "test")
        self.write(gp, self.gs)
        self.write(gp, {**self.gm, "split": "test"}, True)
        self.write(ep, self.rs)
        self.write(ep, {**self.em, "split": "test"}, True)
        argv = ["analyze_code.py", "--suffix", "_t", "--swarm", "m", "--refs", "--benches",
                "humanevalplus", "--fit-split", "test"]
        with patch.object(sys, "argv", argv), patch.object(ac, "RESULTS", self.root), \
                patch.dict(ac.data.CODE_LOADERS, {"humanevalplus": lambda n, split: self.items}), \
                patch.object(ac.data, "dataset_identity", return_value=self.identity), \
                contextlib.redirect_stdout(io.StringIO()):
            ac.main()
        summary = json.loads((self.root / "e11_summary_t_fittest.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["fit_split"], "test")
        self.assertNotIn("validation", summary)
        self.assertEqual(summary["benches"]["humanevalplus"]["systems"]["best"], 100)


class Paper(unittest.TestCase):
    def setUp(self):
        source = ast.parse((ROOT / "paper/make_numbers.py").read_text(encoding="utf-8"))
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("put", "put_e11")]
        self.env = {"macros": {}, "re": re, "verify_e11": vc.verify_provenance, "RES": ROOT / "phase0/results",
                    "E11_FAMILIES": ac.FAMILIES, "E11_REFS": ac.REFS, "E11_DELTA": ac.DELTA}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "make_numbers.py", "exec"), self.env)
        self.summary = json.loads((ROOT / "phase0/results/e11_summary_colab.json").read_text(encoding="utf-8"))

    def test_canonical_macros(self):
        self.env["put_e11"](self.summary)
        generated = (ROOT / "paper/numbers.tex").read_text(encoding="utf-8")
        for name, value in self.env["macros"].items():
            self.assertIn(f"\\newcommand{{\\{name}}}{{{value}}}", generated)

    def test_missing_corrupt_and_partial_provenance(self):
        for mutation in (lambda e: e.update(fit_split="test"), lambda e: e["benches"].pop("mbppplus"),
                         lambda e: e["validation"].update(complete=False),
                         lambda e: e["validation"]["sources_sha256"].pop(next(iter(e["validation"]["sources_sha256"]))),
                         lambda e: e["validation"]["sources_sha256"].update(
                             {next(iter(e["validation"]["sources_sha256"])): "0" * 64}),
                         lambda e: e["benches"]["humanevalplus"].update(primary="visible|all")):
            e = copy.deepcopy(self.summary)
            mutation(e)
            with self.subTest(mutation=mutation), self.assertRaises(SystemExit):
                self.env["put_e11"](e)
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(SystemExit, "absente"):
            vc.verify_provenance(self.summary, Path(d))

    def test_absent_e11_macro_is_fatal(self):
        # Exercise the actual generator with a paper-only typo, without modifying this worktree.
        source = (ROOT / "paper/make_numbers.py").read_text(encoding="utf-8")
        suffix = source[source.index("# Every result macro used by the paper"):source.index("OUT.write_text")]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "main.tex").write_text("\\eElevenMissing", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "eElevenMissing"):
                exec(suffix, {"HERE": Path(d), "macros": {}, "re": re})


if __name__ == "__main__":
    unittest.main()
