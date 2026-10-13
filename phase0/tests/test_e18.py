"""E18 (Lean-checked proofs): screening, file assembly, Lean message parsing, verdicts, feedback, prompts, Colab plan,
checker limits (with a fake `lean`) and the analysis on synthetic results. No Lean, no GPU, no network."""
import json
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "colab"))

import analyze_e18 as A  # noqa: E402
import colab_jobs as CJ  # noqa: E402
import e18_lean as E  # noqa: E402
import run_e18  # noqa: E402
from essaim import lean  # noqa: E402
from essaim.lean_runner import Checker  # noqa: E402
from essaim.results import ResultsFile  # noqa: E402

HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
PROB = {"id": "mathd_t1", "header": HEADER, "informal": "/-- Solve $2x=4$. -/",
        "formal_statement": "theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) :\n  x = 2 := by\n"}
PULL = re.compile(r"^(mc|gen|solo|sot|code|codeexec|aa|sciexec|e18)_.*_colab_.*\.jsonl$")  # colab_phase0.sh, pull


def completion(body: str, statement: str = "theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) : x = 2 := by", pre: str = "") -> str:
    return f"Plan: divide by two.\n```lean4\n{HEADER}{pre}{statement}\n{body}\n```\nDone."


def lean_out(*msgs) -> str:
    return "\n".join(json.dumps(m) for m in msgs)


def info(line, text):
    return {"severity": "information", "pos": {"line": line, "column": 0}, "endPos": None, "data": text,
            "fileName": "Check.lean"}


def error(line, col, text, end=None):
    return {"severity": "error", "pos": {"line": line, "column": col},
            "endPos": {"line": end[0], "column": end[1]} if end else None, "data": text, "fileName": "Check.lean"}


class Screening(unittest.TestCase):
    def test_build_uses_the_dataset_statement_and_the_model_body(self):
        b = lean.build(PROB, completion("  linarith"))
        self.assertEqual(b["status"], "ok")
        self.assertFalse(b["statement_rewritten"])  # same statement, other line breaks
        self.assertTrue(b["code"].startswith(HEADER))
        self.assertIn("theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) :\n  x = 2 := by\n  linarith", b["code"])
        self.assertEqual(b["code"].count("import Mathlib"), 1)  # the model's header lines are dropped
        full = lean.with_axioms(b["code"], b["name"])
        self.assertEqual(full.splitlines()[b["axioms_line"] - 1], "#print axioms mathd_t1")

    def test_a_changed_statement_is_never_compiled(self):
        b = lean.build(PROB, completion("  trivial", "theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) : True := by"))
        self.assertEqual(b["status"], "ok")
        self.assertTrue(b["statement_rewritten"])
        self.assertIn("x = 2 := by\n  trivial", b["code"])
        self.assertNotIn(": True", b["code"])

    def test_term_mode_and_auxiliary_lemmas(self):
        b = lean.build(PROB, completion("", "theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) : x = 2 :=\n  by linarith"))
        self.assertEqual(b["status"], "ok")
        self.assertIn("x = 2 :=\n  by linarith", b["code"])
        b = lean.build(PROB, completion("  exact aux x h₀", pre="/-- helper -/\nlemma aux (x : ℝ) (h : 2 * x = 4) : x = 2 := by\n  linarith\n\n"))
        self.assertEqual(b["status"], "ok")
        self.assertIn("lemma aux", b["code"])

    def test_indented_prelude_cannot_shadow_the_statement(self):
        problem = {"header": "import Mathlib\nopen Nat\n",
                   "formal_statement": "theorem target : succ 0 = 2 := by\n"}
        helper = "lemma aux : True := by\n  trivial\n"
        for pre in ("  def succ (n : Nat) : Nat := n + 2\n",
                    "\tabbrev succ (n : Nat) : Nat := n + 2\n",
                    helper + "  def succ (n : Nat) : Nat := n + 2\n",
                    helper + "    def\n      succ (n : Nat) : Nat := n + 2\n",
                    helper + "  /- a /- nested -/ comment -/ def succ (n : Nat) : Nat := n + 2\n"):
            with self.subTest(pre=pre):
                text = f"```lean4\n{problem['header']}{pre}{problem['formal_statement']}  rfl\n```"
                b = lean.build(problem, text)
                self.assertEqual(b["status"], "forbidden")
                self.assertNotIn("code", b)
                self.assertIn("theorem/lemma", b["reason"])

    def test_prelude_continuations_and_comments_are_preserved(self):
        preludes = (
            "  lemma aux (x : ℝ)\n    (h : 2 * x = 4) :\n    x = 2 := by\n    linarith\n",
            "lemma aux (x : ℝ) (h : 2 * x = 4) : x = 2 :=\n  by\n    linarith\n",
            "-- def succ\n/- abbrev /- def -/ -/\nlemma aux (x : ℝ) (h : 2 * x = 4) : x = 2 := by\n"
            "  have label : \"def succ\" = \"def succ\" := rfl\n  linarith\n",
            "lemma aux (x : ℝ) (h : 2 * x = 4) : x = 2 := by\n  linarith\n"
            "  theorem aux2 : True := by\n    trivial\n",
        )
        for pre in preludes:
            with self.subTest(pre=pre):
                b = lean.build(PROB, completion("  exact aux x h₀", pre=pre))
                self.assertEqual(b["status"], "ok")
                self.assertIn(pre.rstrip("\n"), b["code"])
                self.assertFalse(b["statement_rewritten"])

    def test_refusals(self):
        cases = {
            "  native_decide": "native_decide", "  sorry": "sorry", "  admit": "admit",
            "  linarith\n#eval IO.println \"x\"": "#eval",
            "  linarith\n#print axioms mathd_t1": "#print",
            "  run_tac do pure ()": "run_tac", "  exact Lean.Elab.foo": "Lean.",
            "  by_elab do pure ()": "by_elab",
        }
        for body, word in cases.items():
            b = lean.build(PROB, completion(body))
            self.assertEqual(b["status"], "forbidden", body)
            self.assertIn(word, b["reason"])
        # a top-level option line of the model is dropped (never compiled); inside the proof it is refused
        b = lean.build(PROB, completion("  linarith", pre="set_option debug.skipKernelTC true in\n"))
        self.assertNotIn("skipKernelTC", b.get("code", ""))
        b = lean.build(PROB, completion("  set_option debug.skipKernelTC true in\n  linarith"))
        self.assertEqual(b["status"], "forbidden")
        for pre, word in (("instance : Add ℝ := ⟨fun _ _ => 0⟩\n", "instance"),
                          ("axiom cheat : False\n", "axiom"), ("notation \"two\" => 3\n", "notation"),
                          ("@[simp] lemma l : True := trivial\n", "@["),
                          ("def f (x : ℝ) := x\n", "theorem/lemma"),
                          ("lemma mathd_t1' : True := trivial\ntheorem mathd_t1 : True := trivial\n", "redéfinit")):
            b = lean.build(PROB, completion("  linarith", pre=pre))
            self.assertEqual(b["status"], "forbidden", pre)
            self.assertIn(word, b["reason"])
        # allowed: resource options, forbidden words inside comments and strings, the header's own lines
        ok = lean.build(PROB, completion("  -- sorry, native_decide\n  /- axiom -/ linarith",
                                         pre="set_option maxHeartbeats 400000 in\n"))
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(lean.screen("  simp [show \"sorry\" = \"sorry\" from rfl]"), None)
        self.assertIsNone(lean.screen("  set_option maxRecDepth 1000 in\n  linarith"))

    def test_no_code_no_theorem_and_block_choice(self):
        self.assertEqual(lean.build(PROB, "I cannot prove it.")["status"], "no_code")
        self.assertEqual(lean.build(PROB, "```lean4\ntheorem other : True := trivial\n```")["status"], "no_theorem")
        text = completion("  linarith") + "\n```\nsome plain block\n```"
        self.assertIn("linarith", lean.last_code_block(text))  # a tagged Lean block wins over a later plain block

    def test_comments_and_assign(self):
        self.assertEqual(lean.strip_comments("a /- b /- c -/ d -/ e -- f\ng"), "a " + " " * 17 + " e" + " " * 5 + "\ng")
        s = "theorem t (h : f (x := 1)) : x = 2 := by"
        self.assertEqual(lean.assign_index(s), s.index(":= by"))
        self.assertEqual(lean.statement_prefix(PROB["formal_statement"]), "theorem mathd_t1 (x : ℝ) (h₀ : 2 * x = 4) :\n  x = 2 :=")
        self.assertEqual(lean.theorem_name("lemma foo.bar (x : ℕ) : x = x := by"), "foo.bar")


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.b = lean.build(PROB, completion("  linarith"))
        self.line = self.b["axioms_line"]

    def run_(self, *msgs, rc=0, **kw):
        return {"returncode": rc, "stdout": lean_out(*msgs), **kw}

    def test_verified_only_with_our_axioms_report(self):
        ok = self.run_(info(self.line, "'mathd_t1' depends on axioms: [propext, Classical.choice, Quot.sound]"))
        self.assertEqual(lean.verdict(ok, "mathd_t1", self.line)["status"], "verified")
        none = self.run_(info(self.line, "'mathd_t1' does not depend on any axioms"))
        self.assertTrue(lean.verdict(none, "mathd_t1", self.line)["verified"])
        spoof = self.run_(info(self.line - 3, "'mathd_t1' does not depend on any axioms"))
        self.assertEqual(lean.verdict(spoof, "mathd_t1", self.line)["status"], "no_axioms_report")
        other = self.run_(info(self.line, "'other' does not depend on any axioms"))
        self.assertEqual(lean.verdict(other, "mathd_t1", self.line)["status"], "no_axioms_report")

    def test_rejections(self):
        rep = info(self.line, "'mathd_t1' depends on axioms: [propext]")
        cases = [
            (self.run_(error(12, 2, "linarith failed"), rep, rc=1), "lean_error"),
            (self.run_({"severity": "warning", "pos": {"line": 11, "column": 8}, "data": "declaration uses 'sorry'"},
                       info(self.line, "'mathd_t1' depends on axioms: [sorryAx]")), "sorry"),
            (self.run_(info(self.line, "'mathd_t1' depends on axioms: [propext, cheat]")), "bad_axioms"),
            (self.run_(info(self.line, "'mathd_t1' depends on axioms: [propext, Lean.ofReduceBool]")), "bad_axioms"),
            (self.run_(rep, rc=137), "crash"),
            (self.run_(rep, timeout=True), "timeout"),
            (self.run_(rep, memory=True), "memory"),
            (self.run_(rep, overflow=True), "overflow"),
            ({"returncode": 0, "stdout": "garbage\n" + json.dumps(rep)}, "lean_error"),
        ]
        for run, want in cases:
            self.assertEqual(lean.verdict(run, "mathd_t1", self.line)["status"], want, want)

    def test_stored_record_gives_the_same_verdict(self):
        noise = [info(3, f"note {k}") for k in range(100)]
        run = self.run_(*noise, info(self.line, "'mathd_t1' depends on axioms: [propext]"), ms=10, peak_mib=5)
        rec = E.lean_record(run, self.line)
        self.assertLess(len(rec["messages"]), 70)
        self.assertEqual(lean.verdict(rec, "mathd_t1", self.line), lean.verdict(run, "mathd_t1", self.line))
        bad = self.run_(*noise, error(12, 2, "x" * 5000), rc=1)
        self.assertEqual(lean.verdict(E.lean_record(bad, self.line), "mathd_t1", self.line)["status"], "lean_error")

    def test_oracle_verdict(self):
        f = lean.oracle_file(PROB)
        self.assertTrue(f["code"].endswith(":= by\n  sorry\n"))
        good = {"returncode": 0, "stdout": lean_out(
            {"severity": "warning", "pos": {"line": 7, "column": 8}, "data": "declaration uses 'sorry'"},
            info(f["axioms_line"], "'mathd_t1' depends on axioms: [sorryAx]"))}
        self.assertTrue(lean.oracle_verdict(good, "mathd_t1", f["axioms_line"])["ok"])
        bad = {"returncode": 1, "stdout": lean_out(error(7, 0, "unknown identifier"))}
        self.assertFalse(lean.oracle_verdict(bad, "mathd_t1", f["axioms_line"])["ok"])

    def test_feedback_marks_the_error_span(self):
        code = self.b["code"]
        n = code.splitlines().index("  linarith") + 1
        fb = lean.feedback({"stdout": lean_out(error(n, 2, "linarith failed to find a contradiction", (n, 10)))}, code)
        self.assertIn("  <error>linarith</error>", fb)
        self.assertIn("Error Message: linarith failed", fb)
        self.assertIn("Corresponding Code:\n```lean4\n", fb)
        many = lean.feedback({"stdout": lean_out(*[error(n, 2, "e") for _ in range(10)])}, code)
        self.assertIn("[Omitted 2 more errors]", many)
        self.assertLessEqual(len(lean.feedback({"stdout": lean_out(error(n, 2, "z" * 9000))}, code)), 6000 + 10)


class Prompts(unittest.TestCase):
    def test_styles_and_revision(self):
        g = E.messages(E.PROVERS["Goedel-LM/Goedel-Prover-V2-8B"], PROB, None)
        self.assertEqual([m["role"] for m in g], ["user"])
        self.assertTrue(g[0]["content"].startswith("Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib"))
        self.assertIn(":= by\n  sorry```\n\nBefore producing", g[0]["content"])
        k = E.messages(E.PROVERS["AI-MO/Kimina-Prover-Distill-8B"], PROB, None)
        self.assertEqual(k[0]["content"], "You are an expert in mathematics and Lean 4.")
        self.assertIn("# Problem:Solve $2x=4$.\n# Formal statement:\n```lean4\n", k[1]["content"])
        prev = {"round": 0, "code": "CODE", "feedback": "FEEDBACK", "text": ""}
        r = E.messages(E.PROVERS["Pythagoras-LM/Pythagoras-Prover-4B"], PROB, prev)
        self.assertEqual([m["role"] for m in r], ["user", "assistant", "user"])
        self.assertEqual(r[1]["content"], "```lean4\nCODE\n```")
        self.assertIn("The proof (Round 0) is not correct.", r[2]["content"])
        self.assertIn("FEEDBACK", r[2]["content"])

    def test_candidate_falls_back_to_the_reasoning(self):
        self.assertEqual(E.candidate("no code", "theorem x"), "```lean4\ntheorem x\n```")
        self.assertEqual(E.candidate(completion("  linarith"), "theorem x"), completion("  linarith"))
        self.assertEqual(E.candidate("no code", None), "no code")


class Table(unittest.TestCase):
    def test_provers_pinned_permissive_and_fit_an_a100(self):
        for p in E.PROVERS.values():
            self.assertRegex(p.sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(p.revision, r"^[0-9a-f]{40}$")
            self.assertIn(p.license, ("apache-2.0", "mit"))
            self.assertTrue(p.file.endswith(".gguf"))
        self.assertEqual(len({p.local(Path("m")) for p in E.PROVERS.values()}), len(E.PROVERS))
        self.assertNotIn(E.REFERENCE, E.SWARM)
        self.assertEqual(len(set(E.SWARM)), 3)
        for k, m, s in E.plan() + E.plan(pilot=True):
            self.assertLessEqual(E.reserve_gib(k, m, s) + 2, 40)  # MemoryBudget margin, A100 40GB
        # weights + f16 KV cache of every slot + 2 GiB, as written in the doc
        self.assertEqual(E.PROVERS["Goedel-LM/Goedel-Prover-V2-8B"].reserve_gib(True), 34)
        self.assertEqual(E.PROVERS[E.REFERENCE].reserve_gib(False), 35)

    def test_problems_split(self):
        rows = [{"name": f"p{k:03d}", "split": "valid" if k < 244 else "test", "header": HEADER,
                 "formal_statement": f"theorem p{k:03d} : True := by\n", "informal_prefix": ""} for k in range(488)]
        raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
        import hashlib
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E, "DATA", Path(d)), \
                mock.patch.object(E, "DATA_SHA256", hashlib.sha256(raw).hexdigest()):
            (Path(d) / "e18_minif2f_deepseek.jsonl").write_bytes(raw)
            dev, test = E.problems("dev"), E.problems("test")
            self.assertEqual((len(dev), len(test)), (E.DEV_N, 244))
            self.assertTrue(all(p["source_split"] == "valid" for p in dev))
            self.assertEqual(dev, E.problems("dev"))  # fixed draw
            with mock.patch.object(E, "DATA_SHA256", "0" * 64):
                with self.assertRaises(SystemExit):
                    E.problems("test")


class Plan(unittest.TestCase):
    def test_plans_stages_and_commands(self):
        p = CJ.PLANS["e18-1"]
        self.assertEqual(p[0], ("e18-lean", "minif2f", "all"))
        self.assertEqual(sum(1 for j in p if j[0] == "e18-prove"), 2 * len(E.SWARM) + 1)
        self.assertLess(CJ.stage_of(p[0]), CJ.stage_of(p[1]))
        for k, m, s in p + CJ.PLANS["e18-pilot"]:
            self.assertIn(k, CJ.NO_MODEL)
            self.assertIn(k, CJ.E16_KINDS)
            cmd = CJ.job_cmd(k, m, s)
            self.assertEqual(CJ.e16_need(k, m, s), E.reserve_gib(k, m, s))
            if k == "e18-lean":
                self.assertEqual(cmd[1], "e18_setup.py")
                continue
            self.assertEqual(cmd[1], "run_e18.py")
            self.assertEqual(Path(cmd[cmd.index("--gguf") + 1]), E.PROVERS[m].local(CJ.MODELS))
            name = E.result_path(m, cmd[cmd.index("--suffix") + 1], s).name
            self.assertRegex(name, PULL)
            if m == E.REFERENCE:
                self.assertEqual(cmd[cmd.index("--chains") + 1], "0")
        test = {m: CJ.job_cmd(*j) for j in p for m in [j[1]] if j[2] == "test" and j[0] == "e18-prove"}
        for m in E.SWARM:
            c = test[m]
            self.assertEqual([c[c.index(f) + 1] for f in ("--samples", "--chains", "--revisions")], ["3", "1", "1"])
        # names written by the setup job are pulled too; the other plans are untouched
        for name in (E.oracle_path("_colab", "dev").name, "e18_setup_colab_runs.jsonl",
                     E.wall_path(E.SWARM[0], "_colab", "test").name):
            self.assertRegex(name, PULL)
        self.assertEqual(CJ.PLANS["solo-1"], [("solo", m, "all") for m in CJ.SOLO])
        self.assertEqual(CJ.job_cmd("solo", "Qwen/Qwen3.5-4B", "all")[1], "run_solo.py")

    def test_session_capacity_and_families(self):
        import session_json
        self.assertEqual(session_json.families("e18-1"), ["e18"])
        self.assertEqual(session_json.capacity("e18-1", "A100"), 37)
        with self.assertRaises(ValueError):
            session_json.capacity("e18-1", "L4")
        self.assertEqual(session_json.select_gpu("e18-pilot"), "A100")

    def test_pull_script_and_checkpoint_key(self):
        script = (HERE / "colab" / "colab_phase0.sh").read_text(encoding="utf-8")
        self.assertIn("results/e18_*_colab_*.jsonl", script)
        self.assertIn("glob.glob('$REMOTE/e18_*_colab_*.jsonl')", script)
        self.assertIn("|e18)_", script)
        import checkpoint
        a = {"id": "x", "sample": 0, "round": 0}
        self.assertNotEqual(checkpoint.key(a), checkpoint.key({**a, "round": 1}))
        self.assertEqual(checkpoint.key({"id": "x", "sample": 0}), checkpoint.key({"id": "x", "sample": 0}))


FAKE_LEAN = textwrap.dedent("""
    import json, sys, time
    src = open(sys.argv[-1], encoding="utf-8").read()
    if "SLEEP" in src:
        time.sleep(30)
    lines = src.splitlines()
    n = next(i for i, l in enumerate(lines, 1) if l.startswith("#print axioms"))
    name = lines[n - 1].split()[-1]
    print(json.dumps({"severity": "information", "pos": {"line": n, "column": 0}, "data": f"'{name}' depends on axioms: [propext]"}))
""")


class Runner(unittest.TestCase):
    def test_fake_lean_verdict_and_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake_lean.py"
            fake.write_text(FAKE_LEAN, encoding="utf-8")
            ch = Checker({"lean": sys.executable}, timeout_s=3, mem_mib=4096, workers=2,
                         command=[sys.executable, str(fake)])
            b = lean.build(PROB, completion("  linarith"))
            run = ch.check(lean.with_axioms(b["code"], b["name"]))
            self.assertEqual(lean.verdict(run, b["name"], b["axioms_line"])["status"], "verified", run)
            slow = ch.check(lean.with_axioms(b["code"] + "-- SLEEP\n", b["name"]))
            self.assertTrue(slow["timeout"])
            self.assertLess(slow["ms"], 15000)
            self.assertEqual(lean.verdict(slow, b["name"], b["axioms_line"])["status"], "timeout")


class Setup(unittest.TestCase):
    """What can be checked without Lean in the setup controls (Codex audit of eedd458)."""

    def test_control_files_are_valid_lean_layouts(self):
        import e18_setup
        for name, code, want in e18_setup.control_files():
            lines = [l for l in code.splitlines() if l.strip()]
            first_other = next(k for k, l in enumerate(lines) if not l.startswith("import "))
            self.assertFalse(any(l.startswith("import ") for l in lines[first_other:]), name)  # imports first only
            self.assertEqual(code.count("theorem e18_control"), 1, name)

    def test_completion_controls_reach_the_expected_stage(self):
        import e18_setup
        for name, text, want in e18_setup.control_completions():
            b = lean.build(e18_setup.CONTROL, text)
            if want == "forbidden":
                self.assertEqual(b["status"], "forbidden", name)
            else:
                self.assertEqual(b["status"], "ok", name)
        changed = lean.build(e18_setup.CONTROL, e18_setup.control_completions()[1][1])
        self.assertTrue(changed["statement_rewritten"])

    def test_lean_path_becomes_absolute(self):
        import e18_setup
        root = Path(tempfile.gettempdir()) / "mathlib4"
        absolute = str(Path(tempfile.gettempdir()).resolve() / "abs")
        out = e18_setup.absolute_path_list(f"./.lake/build/lib;{absolute};", root, sep=";").split(";")  # ";": Windows
        self.assertEqual(Path(out[0]), (root / ".lake/build/lib").resolve())
        self.assertEqual(out[1], absolute)
        self.assertEqual(len(out), 2)


class ProcessTree(unittest.TestCase):
    def test_children_of_every_thread_are_found(self):
        from essaim import lean_runner
        with tempfile.TemporaryDirectory() as proc:
            tree = {10: {"10": [11], "12": [13]}, 11: {"11": []}, 13: {"13": [14], "15": [11]}, 14: {"14": []}}
            for pid, tasks in tree.items():
                for tid, kids in tasks.items():
                    d = Path(proc) / str(pid) / "task" / tid
                    d.mkdir(parents=True)
                    (d / "children").write_text(" ".join(map(str, kids)))
            self.assertEqual(lean_runner.descendants(10, proc), [10, 11, 13, 14])  # 13: child of a secondary thread
            self.assertEqual(lean_runner.children(99, proc), set())


class Attempt(unittest.TestCase):
    def test_attempt_records_a_recheckable_row(self):
        b = lean.build(PROB, completion("  linarith"))

        class Srv:
            http = None

        class FakeChecker:
            def check(self, text):
                return {"returncode": 0, "timeout": False, "memory": False, "overflow": False, "ms": 1, "peak_mib": 0,
                        "stdout": lean_out(info(b["axioms_line"], "'mathd_t1' depends on axioms: [propext]"))}
        gen = {"text": completion("  linarith"), "reasoning_chars": 10, "reasoning_block": None, "finish": "stop",
               "n_tokens": 100, "prompt_tokens": 50, "ms": 5}
        spec = E.PROVERS[E.SWARM[0]]
        with mock.patch.object(run_e18, "chat", return_value=gen):
            row = run_e18.attempt(Srv(), FakeChecker(), spec, PROB, 0, 0, None, 8192)
        self.assertTrue(row["verified"])
        self.assertTrue(A.recheck(json.loads(json.dumps(row)), PROB))
        with mock.patch.object(run_e18, "chat", side_effect=run_e18.PromptTooLong("context")):
            row = run_e18.attempt(Srv(), FakeChecker(), spec, PROB, 0, 1, row, 8192)
        self.assertEqual(row["status"], "prompt_too_long")
        self.assertFalse(A.recheck(row, PROB))


def synth_row(model, it, sample, rnd, ok, toks=100):
    stmt = lean.statement_prefix(it["formal_statement"]) + " by"
    text = f"```lean4\n{stmt}\n  {'linarith' if ok else 'simp'}\n```"
    b = lean.build(it, text)
    msgs = [info(b["axioms_line"], f"'{it['id']}' depends on axioms: [propext]")] if ok else \
        [error(b["axioms_line"] - 3, 2, "simp failed")]
    rec = E.lean_record({"returncode": 0 if ok else 1, "stdout": lean_out(*msgs)}, b["axioms_line"])
    v = lean.verdict(rec, b["name"], b["axioms_line"])
    return {"id": it["id"], "sample": sample, "round": rnd, "model": model, "text": text, "reasoning_block": None,
            "n_tokens": toks, "build": "ok", "code": b["code"], "status": v["status"], "verified": v["verified"],
            "lean": rec, "axioms_line": b["axioms_line"]}


class Analysis(unittest.TestCase):
    def test_end_to_end_on_synthetic_results(self):
        items = {s: [{"id": f"{s}{k}", "header": HEADER, "informal": "",
                      "formal_statement": f"theorem {s}{k} (x : ℝ) (h : 2 * x = 4) : x = 2 := by\n"} for k in range(12)]
                 for s in E.SPLITS}
        # which (model, problem) succeed: the swarm members are complementary, the reference solves half
        good = {E.SWARM[0]: range(0, 4), E.SWARM[1]: range(3, 7), E.SWARM[2]: range(6, 9), E.REFERENCE: range(0, 6)}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E, "problems", lambda s: items[s]):
            res = Path(d)
            for s in E.SPLITS:
                o = ResultsFile(E.oracle_path("_t", s, res), {"kind": "oracle"}, key=("id",))
                for it in items[s]:
                    o.write({"id": it["id"], "ok": it["id"] != f"{s}11"})  # one statement excluded by the oracle
                o.release()
                usable = [it for it in items[s] if it["id"] != f"{s}11"]
                for m in E.SWARM + ([E.REFERENCE] if s == "test" else []):
                    ref = m == E.REFERENCE
                    S, C, R = (3, 0 if ref else 1, 0 if ref else 1) if s == "test" else (1, 0, 0)
                    man = {"data": 1, "bench": "minif2f", "prompt": "v", "max_tokens": 8192, "lean": 1, "checker": 1,
                           "screen": 1, "samples": S, "problems": "h", "n": len(usable), "chains": C, "revisions": R,
                           "ctx_per_slot": 9728 + (4608 if R else 0), "model": m, "weights": m}
                    f = ResultsFile(E.result_path(m, "_t", s, res), man, key=("id", "sample", "round"))
                    for k, it in enumerate(usable):
                        for smp in range(S):
                            ok0 = k in good[m] and smp == 0
                            f.write(synth_row(m, it, smp, 0, ok0))
                            if smp < C and not ok0:
                                f.write(synth_row(m, it, smp, 1, k == 10))  # revision rescues problem 10
                    f.release()
            summary, report = A.analyse("_t", res)
        test = summary["splits"]["test"]
        self.assertEqual(test["n"], 11)
        self.assertAlmostEqual(test["accuracy"]["swarm"], 100 * 9 / 11)
        self.assertAlmostEqual(test["accuracy"]["reference"], 100 * 6 / 11)
        self.assertAlmostEqual(test["accuracy"]["swarm_revise"], 100 * 10 / 11)
        self.assertEqual(summary["best_dev"], E.SWARM[0])  # 4 dev successes, tie broken by table order
        c = test["comparisons"]
        self.assertEqual(len(test["multiplicity"]["primary"]), 3)
        self.assertGreaterEqual(c["P2 essaim vs référence"]["p_holm"], c["P2 essaim vs référence"]["p_two_sided"])
        self.assertIn("non corrigé", report[-4] + "".join(report))
        self.assertEqual(test["complementarity"]["union"], 9)

    def test_tampered_rows_are_refused(self):
        it = dict(PROB)
        row = synth_row("m", it, 0, 0, False)
        row["verified"], row["status"] = True, "verified"
        with self.assertRaises(SystemExit):
            A.recheck(row, it)
        row = synth_row("m", it, 0, 0, True)
        row["code"] = row["code"].replace("linarith", "simp")
        with self.assertRaises(SystemExit):
            A.recheck(row, it)

    def test_helpers(self):
        a = [(False, 10), (True, 10)]
        b = [(True, 30)]
        self.assertEqual(A.round_robin([[1, 2], [3], [4, 5]], 4), [1, 3, 4, 2])
        self.assertEqual(A.token_matched([a], [b]), ([1.0], [0.0]))  # T = 20: b's 30-token success is too late
        self.assertEqual(A.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})


if __name__ == "__main__":
    unittest.main()
