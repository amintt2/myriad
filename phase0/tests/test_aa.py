"""E12 unit tests (no model, no GPU, no gated data, no network):  uv run python -m unittest tests.test_aa

GPQA and SciCode are exercised on tiny synthetic problems written for these tests; the SciCode run tests
start real child processes through the E11 sandbox (a few seconds in all)."""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_e12 as ae  # noqa: E402
import colab_jobs  # noqa: E402
import exec_scicode as xs  # noqa: E402
import run_aa  # noqa: E402
from essaim import answers, gpqa, scicode  # noqa: E402


def fake_rows(n: int) -> list[dict]:
    return [{"Record ID": f"rec{i:03d}", "Question": f"synthetic question {i}?", "Correct Answer": f"right{i}",
             "Incorrect Answer 1": f"wrong{i}a", "Incorrect Answer 2": f"wrong{i}b", "Incorrect Answer 3": f"wrong{i}c",
             "High-level domain": "Physics", "Subdomain": "x"} for i in range(n)]


class Gpqa(unittest.TestCase):
    def test_balanced_positions_and_content(self):
        items = gpqa.balanced_items(fake_rows(198))
        pos = [x["answer"] for x in items]
        self.assertEqual({p: pos.count(p) for p in range(4)}, {0: 50, 1: 50, 2: 49, 3: 49})
        for i, x in enumerate(items):
            self.assertEqual(len(x["options"]), 4)
            self.assertEqual(x["options"][x["answer"]], f"right{i}")
            self.assertEqual(sorted(x["options"]), sorted([f"right{i}", f"wrong{i}a", f"wrong{i}b", f"wrong{i}c"]))
        self.assertEqual(items, gpqa.balanced_items(fake_rows(198)))  # deterministic
        self.assertNotEqual([x["options"] for x in items], [x["options"] for x in gpqa.balanced_items(fake_rows(198), seed=1)])

    def test_order_of_rows_does_not_matter(self):
        rows = fake_rows(20)
        self.assertEqual(gpqa.balanced_items(rows), gpqa.balanced_items(rows[::-1]))

    def test_csv_parsing_and_missing_column(self):
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=list(fake_rows(1)[0]))
        w.writeheader()
        rows = fake_rows(3)
        rows[0]["Question"] = "multi\nline, with \"quotes\""
        w.writerows(rows)
        items = gpqa.parse(("﻿" + buf.getvalue()).encode("utf-8"))
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["question"], "multi\nline, with \"quotes\"")
        with self.assertRaises(SystemExit):
            gpqa.parse(b"Record ID,Question\nr,q\n")

    def test_pinned_file_check(self):
        self.assertEqual(gpqa.git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")
        with self.assertRaises(SystemExit):
            gpqa.check_file(b"not the pinned file")

    def test_letter_extraction_and_gold(self):
        item = {"options": ["a", "b", "c", "d"], "answer": 2}
        self.assertEqual(answers.extract("gpqa", "Reasoning...\nThe answer is (C).", item), "C")
        self.assertIsNone(answers.extract("gpqa", "I am not sure.", item))
        self.assertEqual(answers.gold("gpqa", item), "C")

    def test_result_rows_hold_no_question(self):
        # run_aa.run_gpqa writes only these fields; the question text and the options are not among them
        src = Path(run_aa.__file__).read_text(encoding="utf-8")
        row = src[src.index('out.write({"id": it["id"]'):]
        row = row[:row.index("})")]
        for banned in ('it["question"]', 'it["options"]', "prompt"):
            self.assertNotIn(banned, row)


def synthetic_problem() -> dict:
    steps = []
    for k, (name, body, test) in enumerate([
            ("double", "def double(x):\n    return 2 * x", "assert double(2) == target"),
            ("addone", "def addone(x):\n    return x + 1", "assert addone(double(3)) == target"),
            ("Acc", "class Acc:\n    def __init__(self):\n        self.v = 0", "assert Acc().v == target")], 1):
        steps.append({"number": f"7.{k}", "description": f"Step {k} description.", "background": f"Background {k}.",
                      "header": body.splitlines()[0] + ("" if body.startswith("class") else " ..."),
                      "return_line": "Return the value.", "tests": [test], "reference": body})
    return {"id": "7", "split": "dev", "name": "p", "description": "Main problem text.", "background": "bg",
            "io": "", "dependencies": "import numpy as np", "steps": steps}


class SciCodeText(unittest.TestCase):
    def setUp(self):
        self.p = synthetic_problem()

    def test_test_data_without_reference_and_dev_oracle(self):
        p = self.p
        raw = {"problem_id": p["id"], "problem_name": p["name"], "problem_description_main": p["description"],
               "problem_background_main": p["background"], "problem_io": p["io"],
               "required_dependencies": p["dependencies"], "sub_steps": [
                   {"step_number": s["number"], "step_description_prompt": s["description"],
                    "step_background": s["background"], "function_header": s["header"],
                    "return_line": s["return_line"], "test_cases": s["tests"]} for s in p["steps"]]}
        test = scicode._problem(raw, "test")
        self.assertTrue(all(s["reference"] is None for s in test["steps"]))
        rows = {s["number"]: {"text": f"```python\n{s['reference']}\n```"} for s in p["steps"]}
        self.assertEqual(len(xs.jobs_for_model([test], rows)), 3)
        with self.assertRaisesRegex(ValueError, "reference code"):
            xs.jobs_for_oracle([test])
        for r, s in zip(raw["sub_steps"], p["steps"]):
            r["ground_truth_code"] = s["reference"]
        self.assertEqual(len(xs.jobs_for_oracle([scicode._problem(raw, "dev")])), 3)
        raw["sub_steps"][0].pop("ground_truth_code")
        with self.assertRaisesRegex(ValueError, "reference code"):
            xs.jobs_for_oracle([scicode._problem(raw, "dev")])

    def test_all_official_skipped_code_and_modified_file(self):
        for pid, k, header in (("13", 6, "class Maxwell:"), ("62", 1, "class EnlargedBlock:"),
                               ("76", 3, "def generate_dna(N, PWM):")):
            p = {"id": pid, "steps": [{"number": f"{pid}.{k}", "header": header, "reference": None}]}
            code = scicode.chain_code(p, 0, None)
            self.assertIn(header.split("(")[0], code)
            compile(code, "<skipped>", "exec")
            with tempfile.TemporaryDirectory() as d, mock.patch.object(scicode, "SKIPPED_DIR", Path(d)):
                with self.assertRaises(FileNotFoundError):
                    scicode.chain_code(p, 0, None)
                (Path(d) / f"{pid}.{k}.txt").write_text("modified", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    scicode.chain_code(p, 0, None)

    def test_names_and_ids(self):
        self.assertEqual([scicode.def_name(s["header"]) for s in self.p["steps"]], ["double", "addone", "Acc"])
        self.assertEqual([scicode.step_number(s) for s in self.p["steps"]], [1, 2, 3])
        self.assertEqual(scicode.def_name("async def go(x):"), "go")
        with self.assertRaises(ValueError):
            scicode.def_name("x = 1")

    def test_skipped_steps(self):
        p = {"id": "13", "steps": [{"number": "13.6"}]}
        self.assertTrue(scicode.is_skipped(p, p["steps"][0]))
        self.assertFalse(scicode.is_skipped(self.p, self.p["steps"][0]))
        self.assertIn("class EnlargedBlock:", scicode.chain_code(
            {"id": "62", "steps": [{"number": "62.1", "header": "class EnlargedBlock:"}]}, 0, "ignored"))

    def test_prompt_carries_the_chain(self):
        first = scicode.prompt(self.p, 0, [])
        self.assertIn("Main problem text.", first)
        self.assertNotIn("PREVIOUS STEPS", first)
        self.assertIn("Background 1.", first)
        second = scicode.prompt(self.p, 1, ["def double(x):\n    return 2 * x"])
        self.assertIn("PREVIOUS STEPS AND THEIR CODE", second)
        self.assertIn("return 2 * x", second)
        self.assertIn("Step 2 description.", second)
        self.assertNotIn("Step 3 description.", second)
        self.assertNotIn("Background 1.", scicode.prompt(self.p, 0, [], background=False))
        self.assertIn("import numpy as np", second)

    def test_extract_variants(self):
        text = "Background: foo\n```python\n# Background: doubling\nimport numpy as np\ndef double(x):\n    return 2 * x\n" \
               "print(double(2))\n```\nExample:\n```python\nprint(double(5))\n```"
        code, status = scicode.extract(text, "double")
        self.assertEqual(status, "fenced")
        self.assertIn("def double", code)
        self.assertNotIn("print", code)
        code, status = scicode.extract("```python\nclass Acc:\n    def __init__(self):\n        self.v = 0\n```", "Acc")
        self.assertEqual((status, "class Acc" in code), ("fenced", True))
        self.assertEqual(scicode.extract("```python\ndef other(x):\n    return 1\n```", "double")[1], "no_entry")
        self.assertEqual(scicode.extract("```python\ndef double(x):\n    return x\n", "double")[1], "unclosed")
        self.assertEqual(scicode.extract("no code here", "double")[1], "empty")

    def test_named_definition_keeps_only_the_step_function(self):
        code = "import numpy as np\n\ndef helper(x):\n    return x\n\n@dec\ndef double(x):\n    return helper(2 * x)\n\nX = 3"
        out = scicode.named_definition(code, "double")
        self.assertTrue(out.startswith("@dec\ndef double"))
        self.assertNotIn("helper(x)", out.split("\n")[2] if False else "def helper")
        self.assertNotIn("X = 3", out)
        self.assertEqual(scicode.named_definition("def broken(:", "broken"), "def broken(:")
        self.assertEqual(scicode.named_definition("def other(): pass", "double"), "def other(): pass")

    def test_script_assembly(self):
        chain = ["def double(x):\n    return 2 * x", "def addone(x):\n    return x + 1", ""]
        src = scicode.script(self.p, 1, chain, "def addone(x):\n    return x + 2")
        lines = src.splitlines()
        self.assertEqual(lines[0], "import numpy as np")
        self.assertLess(src.index("def double"), src.index("def addone"))
        self.assertNotIn("return x + 1", src)  # the previous code of THIS step is not used, the new one is
        self.assertIn("process_hdf5_to_tuple('7.2', 1)", src)
        self.assertIn("target = targets[0]\nassert addone(double(3)) == target", src)
        ref = scicode.reference_script(self.p, 1)
        self.assertIn("return x + 1", ref)
        self.assertIn("return 2 * x", ref)

    def test_chain_uses_the_models_own_earlier_code(self):
        self.assertEqual(scicode.chain_code(self.p, 0, "```python\ndef double(x):\n    return 3 * x\n```"),
                         "def double(x):\n    return 3 * x")
        self.assertEqual(scicode.chain_code(self.p, 0, ""), "")


class FakeHTTP:
    """A llama-server stand-in: answers from a script, records the prompts it saw."""

    def __init__(self, answers_by_header: dict):
        self.answers, self.prompts = answers_by_header, []

    def post(self, url, json=None):
        prompt = json["messages"][0]["content"]
        self.prompts.append(prompt)
        text = next(v for k, v in self.answers.items() if k in prompt.split("NEXT STEP")[-1])

        class R:
            status_code = 200

            def json(self_):
                return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 5, "prompt_tokens": 50}}
        return R()


class FakeOut:
    def __init__(self):
        self.rows, self.done = [], set()

    def write(self, row):
        self.rows.append(row)
        self.done.add((row["id"],))


class Generation(unittest.TestCase):
    def test_solve_problem_chains_and_resumes(self):
        p = synthetic_problem()
        replies = {"def double": "```python\ndef double(x):\n    return 2 * x\n```",
                   "def addone": "```python\ndef addone(x):\n    return x + 1\n```",
                   "class Acc": "```python\nclass Acc:\n    pass\n```"}

        class Srv:
            http = FakeHTTP(replies)
        a = type("A", (), {"no_background": False, "model": "m/x"})()
        out = FakeOut()
        n = run_aa.solve_problem(Srv, a, p, out, threading.Lock(), {})
        self.assertEqual((n, [r["id"] for r in out.rows]), (3, ["7.1", "7.2", "7.3"]))
        self.assertIn("return 2 * x", Srv.http.prompts[1])  # step 2 saw the model's own step 1
        self.assertNotIn("return x + 1", Srv.http.prompts[1])
        self.assertIn("return x + 1", Srv.http.prompts[2])
        # resume with the first two rows kept: only step 3 is asked, with the same chain
        class Srv2:
            http = FakeHTTP(replies)
        out2 = FakeOut()
        have = {r["id"]: r for r in out.rows[:2]}
        self.assertEqual(run_aa.solve_problem(Srv2, a, p, out2, threading.Lock(), have), 1)
        self.assertEqual(Srv2.http.prompts[0], Srv.http.prompts[2])

    def test_skipped_step_uses_official_code_and_is_not_generated(self):
        p = {"id": "62", "split": "test", "name": "p", "description": "D", "background": "", "io": "",
             "dependencies": "import numpy as np",
             "steps": [{"number": "62.1", "description": "s1", "background": "b1", "header": "class EnlargedBlock:",
                        "return_line": "r", "tests": ["assert True"], "reference": None},
                       {"number": "62.2", "description": "s2", "background": "b2", "header": "def g(x):", "return_line": "r",
                        "tests": ["assert True"], "reference": "def g(x):\n    return 1"}]}

        class Srv:
            http = FakeHTTP({"def g": "```python\ndef g(x):\n    return 2\n```"})
        out = FakeOut()
        a = type("A", (), {"no_background": False, "model": "m/x"})()
        self.assertEqual(run_aa.solve_problem(Srv, a, p, out, threading.Lock(), {}), 1)
        self.assertEqual([r["id"] for r in out.rows], ["62.2"])
        self.assertIn("class EnlargedBlock:", Srv.http.prompts[0])
        self.assertNotIn("class Block:", Srv.http.prompts[0])


class ExecJobs(unittest.TestCase):
    def test_oracle_defaults_to_dev_and_failing_harness_stops(self):
        for ok in (True, False):
            with self.subTest(ok=ok), tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(sys, "argv", ["exec_scicode.py", "--oracle"]), \
                    mock.patch.object(xs, "RESULTS", Path(d)), mock.patch.object(xs.signal, "signal"), \
                    mock.patch.object(scicode, "preflight", return_value=[]), \
                    mock.patch.object(scicode, "h5_path", return_value=Path(d) / "targets.h5"), \
                    mock.patch.object(scicode, "h5_identity", return_value={"sha256": "abc"}), \
                    mock.patch.object(xs.sandbox, "isolation", return_value={}), \
                    mock.patch.object(scicode, "problems", return_value=[synthetic_problem()]) as problems, \
                    mock.patch.object(scicode, "run_script", return_value={"ok": ok, "err": "AssertionError",
                                                                          "msg": "", "status": "ok", "ms": 1}), \
                    mock.patch.dict(sys.modules, {"numpy": type("N", (), {"__version__": "n"}),
                                                  "scipy": type("S", (), {"__version__": "s"})}), \
                    mock.patch("sys.stdout", io.StringIO()):
                if ok:
                    xs.main()
                else:
                    with self.assertRaises(SystemExit) as err:
                        xs.main()
                    self.assertEqual(err.exception.code, 1)
                problems.assert_called_once_with("dev")
                self.assertTrue(xs.exec_path("oracle", "", "dev").is_file())
                self.assertFalse(xs.exec_path("oracle", "", "test").exists())

    def test_oracle_test_rejected_before_preflight(self):
        with mock.patch.object(sys, "argv", ["exec_scicode.py", "--oracle", "--splits", "test"]), \
                mock.patch.object(scicode, "preflight") as preflight, mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as err:
                xs.main()
        self.assertEqual(err.exception.code, 2)
        preflight.assert_not_called()

    def test_jobs_follow_the_chain(self):
        p = synthetic_problem()
        rows = {"7.1": {"text": "```python\ndef double(x):\n    return 2 * x\n```"},
                "7.2": {"text": "no code"}, "7.3": {"text": "```python\nclass Acc:\n    pass\n```"}}
        jobs = xs.jobs_for_model([p], rows)
        self.assertEqual([j[0] for j in jobs], ["7.1", "7.2", "7.3"])
        self.assertIsNone(jobs[1][2])  # no code: fails without running
        self.assertIn("class Acc", jobs[2][2])
        self.assertNotIn("def addone", jobs[2][2])  # the failed step contributes nothing to the chain
        self.assertEqual(len(xs.jobs_for_oracle([p])), 3)


class RunScript(unittest.TestCase):
    """Real child processes through the sandbox; scripts need neither numpy nor the h5 file."""

    def test_pass_and_fail_modes(self):
        self.assertTrue(scicode.run_script("def f(x):\n    return 2 * x\nassert f(2) == 4\n", None)["ok"])
        bad = scicode.run_script("def f(x):\n    return 2 * x\nassert f(2) == 5\n", None)
        self.assertEqual((bad["ok"], bad["err"]), (False, "AssertionError"))
        self.assertFalse(scicode.run_script("raise ValueError('x')\n", None)["ok"])
        self.assertFalse(scicode.run_script("import sys\nsys.exit(0)\n", None)["ok"])  # an exit is not a pass
        self.assertFalse(scicode.run_script("import os\nos._exit(0)\n", None)["ok"])  # nor a hard exit

    def test_syntax_error_fails(self):
        self.assertFalse(scicode.run_script("def f(:\n", None)["ok"])


class Analysis(unittest.TestCase):
    def test_local_gpqa_override_selection_and_validation(self):
        valid, invalid = b"synthetic pinned CSV", b"invalid CSV"
        cases = ((None, None), (valid, None), (invalid, None), (None, valid),
                 (valid, invalid), (invalid, valid), (valid, valid))  # override, default
        for override_raw, default_raw in cases:
            with self.subTest(override=override_raw, default=default_raw), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                data = root / "data"
                data.mkdir()
                override, default = root / "override.csv", data / gpqa.FILE
                for path, raw in ((override, override_raw), (default, default_raw)):
                    if raw is not None:
                        path.write_bytes(raw)
                chosen = override if override_raw is not None else default if default_raw is not None else None
                chosen_raw = override_raw if override_raw is not None else default_raw
                with mock.patch.dict(os.environ, {"GPQA_DIAMOND_CSV": str(override)}), \
                        mock.patch.object(gpqa.data, "DATA", data), mock.patch.object(colab_jobs, "HERE", root), \
                        mock.patch.object(gpqa, "PINNED", {"size": len(valid), "git_blob_sha1": gpqa.git_blob_sha1(valid)}), \
                        mock.patch.object(gpqa.data, "hf_hub_download") as download, \
                        mock.patch.object(ae, "RESULTS", root), \
                        mock.patch.object(ae, "sci_section", return_value=[]), \
                        mock.patch.object(ae, "gpqa_section", return_value=[]) as section, \
                        mock.patch.object(sys, "argv", ["analyze_e12.py"]), mock.patch("sys.stdout", io.StringIO()):
                    self.assertEqual(gpqa.local_path(), chosen)
                    if chosen_raw == invalid:
                        for call in (ae.main, lambda: colab_jobs.aa_benches(), gpqa.read_csv_bytes):
                            with self.assertRaises(SystemExit):
                                call()
                        section.assert_not_called()
                    else:
                        ae.main()
                        self.assertEqual(section.call_count, int(chosen is not None))
                        self.assertEqual(colab_jobs.aa_benches(), ["gpqa", "scicode"] if chosen else ["scicode"])
                        if chosen is not None:
                            self.assertEqual(gpqa.read_csv_bytes(), valid)
                    download.assert_not_called()

    def test_default_analysis_without_gpqa_and_invalid_present_csv(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(gpqa.data, "DATA", Path(d)), \
                mock.patch.dict(os.environ, {"GPQA_DIAMOND_CSV": ""}), \
                mock.patch.object(ae, "RESULTS", Path(d)), mock.patch.object(ae, "sci_section", return_value=[]), \
                mock.patch.object(ae, "gpqa_section") as section, \
                mock.patch.object(sys, "argv", ["analyze_e12.py"]), mock.patch("sys.stdout", io.StringIO()):
            ae.main()
            section.assert_not_called()
            (Path(d) / gpqa.FILE).write_bytes(b"invalid")
            with self.assertRaises(SystemExit):
                ae.main()
            section.assert_not_called()

    def test_wilson_and_binomial(self):
        lo, hi = ae.wilson(50, 198)
        self.assertTrue(lo < 100 * 50 / 198 < hi)
        self.assertEqual(ae.wilson(0, 0), (0.0, 100.0))
        self.assertAlmostEqual(ae.binom_tail(0, 10, 0.25), 1.0)
        self.assertAlmostEqual(ae.binom_tail(10, 10, 0.25), 0.25 ** 10)
        self.assertAlmostEqual(ae.binom_tail(2, 3, 0.5), 0.5)
        self.assertLess(ae.binom_tail(70, 198, 0.25), 0.01)  # 35 % on 198 questions is clearly above chance

    def test_protocol_mismatch_is_refused(self):
        ae.check_same({"a": {"prompt": "v1"}, "b": {"prompt": "v1"}}, ("prompt",), "x")
        with self.assertRaises(SystemExit):
            ae.check_same({"a": {"prompt": "v1"}, "b": {"prompt": "v2"}}, ("prompt",), "x")

    def test_sci_vectors_exclusions_and_problem_level(self):
        p = synthetic_problem()
        q = {**synthetic_problem(), "id": "13", "steps": [dict(s, number=s["number"].replace("7.", "13.")) for s in synthetic_problem()["steps"]]}
        res = {"7.1": {"ok": True}, "7.2": {"ok": False}, "7.3": {"ok": True},
               "13.1": {"ok": True}, "13.2": {"ok": True}, "13.3": {"ok": True}}
        ids, ok, solved = ae.sci_vectors([p, q], res, excluded={"7.2"})
        self.assertNotIn("7.2", ids)
        self.assertNotIn("13.6", ids)
        self.assertEqual(solved, {"7": True, "13": True})  # the excluded failing step does not count against the model
        ids, ok, solved = ae.sci_vectors([p, q], res, excluded=set())
        self.assertEqual(solved, {"7": False, "13": True})


class EndToEnd(unittest.TestCase):
    """analyze_e12 on synthetic result files in a temporary directory (the real loaders are patched)."""

    def write(self, path: Path, manifest: dict, rows: list[dict]):
        if "_scicode_" in path.name:
            manifest = {**manifest, "skipped_code": scicode.SKIPPED_SHA256}
            if manifest.get("generation"):
                manifest["generation"] = {**manifest["generation"], "skipped_code": scicode.SKIPPED_SHA256}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        path.with_name(path.name + ".meta.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_report_from_synthetic_files(self):
        from unittest import mock
        items = gpqa.balanced_items(fake_rows(12))
        prob = synthetic_problem()
        fams = list(ae.FAMILIES.values())
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.object(ae, "RESULTS", d), mock.patch.object(run_aa, "RESULTS", d), \
                    mock.patch.object(xs, "RESULTS", d), mock.patch.object(gpqa, "diamond", lambda seed=0: items), \
                    mock.patch.object(scicode, "problems", lambda sp: [
                        {**prob, "split": sp, "steps": [dict(s, reference=None) if sp == "test" else s
                                                     for s in prob["steps"]]}]):
                proto = {k: 1 for k in ae.GPQA_PROTOCOL}
                for j, m in enumerate(fams + ae.REFS):
                    rows = [{"id": it["id"], "text": "The answer is (%s)." % ("ABCD"[it["answer"] if (n + j) % 3 else (it["answer"] + 1) % 4]),
                             "finish": "stop"} for n, it in enumerate(items)]
                    self.write(run_aa.path_for(m, "_t", "gpqa", "all"), {**proto, "n": len(items)}, rows)
                    for sp in ("dev", "test"):
                        gen_rows = [{"id": f"7.{k + 1}", "text": f"code {m} {k}"} for k in range(3)]
                        gman = {"prompt": scicode.PROMPT_VERSION, "n": 1, "n_steps": 3, "background": True, "model": m,
                                "gguf": "g", "weights": {"sha256": "w"}, "engine": "e", "backend": "b", "revision": "r"}
                        self.write(run_aa.path_for(m, "_t", "scicode", sp), gman, gen_rows)
                        man = {"n": 3, "h5": {"sha256": "abc"}, "official_commit": "c", "harness": xs.HARNESS_VERSION,
                               "timeout_s": 300, "oracle": False, "data": None,
                               "generation": {**gman, "rows_sha256": xs.gen_digest({r["id"]: r for r in gen_rows})}}
                        ok = [(j + k) % 2 == 0 for k in range(3)]
                        self.write(xs.exec_path(m, "_t", sp), man,
                                   [{"id": f"7.{k + 1}", "ok": ok[k]} for k in range(3)])
                for sp in ("dev",):
                    self.write(xs.exec_path("oracle", "_t", sp), {"n": 3, "h5": {"sha256": "abc"},
                                "harness": xs.HARNESS_VERSION, "oracle": True, "official_commit": "c",
                                "timeout_s": 300, "data": None},
                               [{"id": "7.1", "ok": True}, {"id": "7.2", "ok": True}, {"id": "7.3", "ok": False}])
                summary = {}
                text = "\n".join(ae.gpqa_section("_t", summary) + ae.sci_section("_t", summary))
        self.assertIn("essaim 7 familles, vote", text)
        self.assertIn("plafond", text)
        self.assertNotIn("meilleur pair de E4", text)  # no E4 summary: nothing is chosen on GPQA itself
        self.assertNotIn("vote pondéré (poids de E4)", text)
        self.assertEqual(summary["gpqa"]["n"], 12)
        self.assertEqual(summary["scicode"]["excluded"]["dev"], ["7.3"])
        self.assertEqual(summary["scicode"]["n_steps_test"], 3)
        self.assertEqual(summary["scicode"]["excluded"]["test"], [])
        for it in items:  # the report holds no question text
            self.assertNotIn(it["question"], text)

    def test_regenerated_text_or_other_protocol_is_refused(self):
        from unittest import mock
        prob = synthetic_problem()
        fams = list(ae.FAMILIES.values())
        for change, key in (("text", "rows_sha256"), ("protocol", "background")):
            with tempfile.TemporaryDirectory() as d, mock.patch.object(ae, "RESULTS", d), \
                    mock.patch.object(run_aa, "RESULTS", Path(d)), mock.patch.object(xs, "RESULTS", Path(d)), \
                    mock.patch.object(scicode, "problems", lambda sp: [
                        {**prob, "split": sp, "steps": [dict(s, reference=None) if sp == "test" else s
                                                     for s in prob["steps"]]}]):
                d = Path(d)
                for j, m in enumerate(fams):
                    for sp in ("dev", "test"):
                        rows = [{"id": f"7.{k + 1}", "text": f"code {k}"} for k in range(3)]
                        gman = {"prompt": scicode.PROMPT_VERSION, "n": 1, "n_steps": 3, "background": True, "model": m,
                                "gguf": "g", "weights": {"sha256": "w"}, "engine": "e", "backend": "b", "revision": "r"}
                        graded = {r["id"]: dict(r) for r in rows}
                        if j == 0 and sp == "test":
                            if change == "text":
                                rows[0]["text"] = "regenerated code"  # the grades below are for the old text
                            else:
                                gman["background"] = False
                        self.write(run_aa.path_for(m, "_t", "scicode", sp), gman, rows)
                        man = {"n": 3, "h5": {"sha256": "abc"}, "official_commit": "c", "harness": xs.HARNESS_VERSION,
                               "timeout_s": 300, "oracle": False, "data": None,
                               "generation": {**gman, "rows_sha256": xs.gen_digest(graded)}}
                        self.write(xs.exec_path(m, "_t", sp), man, [{"id": f"7.{k + 1}", "ok": True} for k in range(3)])
                for sp in ("dev",):
                    self.write(xs.exec_path("oracle", "_t", sp), {"n": 3, "h5": {"sha256": "abc"},
                                "harness": xs.HARNESS_VERSION, "oracle": True, "official_commit": "c",
                                "timeout_s": 300, "data": None},
                               [{"id": f"7.{k + 1}", "ok": True} for k in range(3)])
                with self.assertRaises(SystemExit):
                    ae.sci_section("_t", {})


class H5Path(unittest.TestCase):
    def test_relative_override_becomes_absolute(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.h5").write_bytes(b"x")
            old = os.getcwd()
            os.chdir(d)
            try:
                os.environ["SCICODE_H5"] = "t.h5"
                p = scicode.h5_path()
            finally:
                os.environ.pop("SCICODE_H5", None)
                os.chdir(old)
            self.assertTrue(p.is_absolute())
            self.assertEqual(p.name, "t.h5")


class Plan(unittest.TestCase):
    def test_reused_vm_does_not_keep_gpqa_from_previous_archive(self):
        # Load the pure unpacker without running the bootstrap's top-level VM operations.
        path = Path(colab_jobs.__file__).with_name("colab_bootstrap.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "unpack_code")
        for csv in (False, True):
            with self.subTest(csv=csv), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                work = root / "work"
                old = work / "phase0" / "data" / gpqa.FILE
                old.parent.mkdir(parents=True)
                old.write_bytes(b"old")
                source = root / "source"
                source.mkdir()
                (source / "colab_plan.txt").write_text("aa-1", encoding="utf-8")
                archive = root / "upload.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(source / "colab_plan.txt", arcname="phase0/colab_plan.txt")
                    if csv:
                        (source / gpqa.FILE).write_bytes(b"new")
                        tar.add(source / gpqa.FILE, arcname="phase0/data/" + gpqa.FILE)
                namespace = {"os": os, "tarfile": tarfile, "WORK": work.as_posix()}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                self.assertEqual(namespace["unpack_code"](archive), "aa-1")
                self.assertEqual(old.exists(), csv)
                if csv:
                    self.assertEqual(old.read_bytes(), b"new")

    @unittest.skipUnless(shutil.which("bash"), "bash is needed to exercise the upload archive")
    def test_upload_archive_with_optional_gpqa(self):
        script = Path(colab_jobs.__file__).parent / "colab" / "colab_phase0.sh"
        shell = '''
colab() {
    case "$1" in
      sessions) echo phase0 ;;
      upload) cp "$4" "$ARCHIVE" ;;
      *) return 0 ;;
    esac
}
export -f colab
bash "$SCRIPT" up aa-1 A100
'''
        for h5, csv in ((True, False), (True, True), (False, False)):
            with self.subTest(h5=h5, csv=csv), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                data = root / "phase0" / "data"
                data.mkdir(parents=True)
                if h5:
                    (data / scicode.H5_NAME).write_bytes(b"synthetic targets")
                if csv:
                    (data / gpqa.FILE).write_bytes(b"synthetic CSV")
                archive = root / "archive.tgz"
                env = {**os.environ, "DLLM_REPO": root.as_posix(), "SCRIPT": script.as_posix(),
                       "ARCHIVE": archive.as_posix()}
                r = subprocess.run([shutil.which("bash"), "-c", shell], env=env, capture_output=True, text=True)
                if not h5:
                    self.assertNotEqual(r.returncode, 0)
                    self.assertFalse(archive.exists())
                    continue
                self.assertEqual(r.returncode, 0, r.stderr)
                with tarfile.open(archive) as tar:
                    names = tar.getnames()
                self.assertIn("phase0/data/" + scicode.H5_NAME, names)
                self.assertEqual("phase0/data/" + gpqa.FILE in names, csv)

    def test_optional_gpqa_commands_and_preflight(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(colab_jobs, "HERE", Path(d)), \
                mock.patch.dict(os.environ, {"GPQA_DIAMOND_CSV": ""}), \
                mock.patch.object(gpqa, "diamond") as diamond, \
                mock.patch.object(scicode, "problems"), mock.patch.object(scicode, "h5_identity"), \
                mock.patch.object(scicode, "preflight", return_value=[]):
            cmd = colab_jobs.job_cmd("aa", "Qwen/Qwen3.5-4B", "all")
            self.assertEqual(cmd[cmd.index("--benches") + 1:cmd.index("--parallel")], ["scicode"])
            colab_jobs.prepare_aa()
            diamond.assert_not_called()
            path = Path(d) / "data" / gpqa.FILE
            path.parent.mkdir()
            path.write_bytes(b"invalid")
            for call in (colab_jobs.prepare_aa,
                         lambda: colab_jobs.job_cmd("aa", "Qwen/Qwen3.5-4B", "all")):
                with self.assertRaises(SystemExit):
                    call()
            with mock.patch.object(gpqa, "check_file") as check:
                cmd = colab_jobs.job_cmd("aa", "Qwen/Qwen3.5-4B", "all")
                self.assertEqual(cmd[cmd.index("--benches") + 1:cmd.index("--parallel")], ["gpqa", "scicode"])
                colab_jobs.prepare_aa()
                diamond.assert_called_once()
                check.assert_called_with(b"invalid")
        cmd = colab_jobs.job_cmd("aa-oracle", "e12", "all")
        self.assertEqual(cmd[cmd.index("--splits") + 1], "dev")

    def test_stages_and_commands(self):
        jobs = colab_jobs.PLANS["aa-1"]
        self.assertEqual(jobs[0][0], "aa-oracle")
        self.assertEqual(jobs[-1][0], "aa-exec")
        self.assertEqual([colab_jobs.stage_of(j) for j in (jobs[0], jobs[1], jobs[-1])], [0, 1, 2])
        self.assertEqual({j[1] for j in jobs if j[0] == "aa"}, set(colab_jobs.SOLO))
        for kind in ("aa-oracle", "aa-exec"):
            self.assertIn(kind, colab_jobs.NO_MODEL)
        cmd = colab_jobs.job_cmd("aa", "Qwen/Qwen3.8-27B", "all")
        self.assertEqual(cmd[1], "run_aa.py")
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "3")  # 8 slots of 3072 -> 3 of 8192: the same KV cache
        self.assertEqual(colab_jobs.job_cmd("aa", "Qwen/Qwen3.5-4B", "all")[-1], "6")
        self.assertIn("--oracle", colab_jobs.job_cmd("aa-oracle", "e12", "all"))
        self.assertNotIn("--oracle", colab_jobs.job_cmd("aa-exec", "e12", "all"))

    def test_kv_cache_reservation_unchanged(self):
        # run_aa uses CTX_PER_SLOT x parallel <= the E4 reservation (3072 x slots), so SOLO's memory figures hold
        for m, (_, _, _, slots) in colab_jobs.SOLO.items():
            par = int(colab_jobs.job_cmd("aa", m, "all")[-1])
            self.assertLessEqual(par * run_aa.CTX_PER_SLOT, slots * 3072, m)


if __name__ == "__main__":
    unittest.main()
