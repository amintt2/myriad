"""E10 unit tests (no model, no GPU):  uv run python -m unittest discover -s tests"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_sot  # noqa: E402
import colab_jobs  # noqa: E402
from essaim import sot  # noqa: E402


class Skeleton(unittest.TestCase):
    def test_plain(self):
        r = sot.parse_skeleton("1. Intro\n2. Beaches of Oahu\n3. Local food\n4. Conclusion")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["points"], ["Intro", "Beaches of Oahu", "Local food", "Conclusion"])

    def test_markdown_and_noise(self):
        text = ("Skeleton:\n\n**1. Intro:** short\n2) *Beaches*\n   - sub bullet\n### 3. Food\n"
                "Some trailing remark.")
        r = sot.parse_skeleton(text)
        self.assertEqual(r["points"], ["Intro: short", "Beaches", "Food"])

    def test_second_list_ends_first(self):
        r = sot.parse_skeleton("1. A a\n2. B b\n3. C c\nDetails:\n1. x\n2. y\n3. z\n4. w")
        self.assertEqual(r["points"], ["A a", "B b", "C c"])

    def test_truncated_and_fallback(self):
        r = sot.parse_skeleton("\n".join(f"{i}. p{i}" for i in range(1, 11)))
        self.assertEqual((r["status"], len(r["points"])), ("truncated", sot.MAX_POINTS))
        self.assertEqual(sot.parse_skeleton("1. only\n2. two")["status"], "fallback")
        self.assertEqual(sot.parse_skeleton("Here is my answer in prose.")["points"], [])

    def test_bullets(self):
        r = sot.parse_skeleton("- a\n- b\n  - nested\n- c")
        self.assertEqual((r["status"], r["points"]), ("bullets", ["a", "b", "c"]))

    def test_thinking_ignored_and_long_points(self):
        r = sot.parse_skeleton("<think>1. no\n2. no\n3. no</think>1. " + "word " * 13 + "\n2. b\n3. c")
        self.assertEqual(r["n_long"], 1)
        self.assertEqual(r["points"][1:], ["b", "c"])


class Assembly(unittest.TestCase):
    def test_assign_round_robin(self):
        self.assertEqual(sot.assign(5, ["a", "b", "c"]), ["a", "b", "c", "a", "b"])

    def test_clean_expansion(self):
        self.assertEqual(sot.clean_expansion("**2. Beaches**\nSand.", 1, "Beaches"), "Sand.")
        self.assertEqual(sot.clean_expansion("2.\nSand.", 1, "Beaches"), "Sand.")
        self.assertEqual(sot.clean_expansion("Beaches are nice.\nMore.", 1, "Beaches"), "Beaches are nice.\nMore.")

    def test_assemble(self):
        self.assertEqual(sot.assemble(["A", "B"], ["x", "**2. B**\ny"]), "**1. A**\n\nx\n\n**2. B**\n\ny")


class Judge(unittest.TestCase):
    def test_letter_probs(self):
        import math
        p = sot.letter_probs([{"token": "A", "logprob": math.log(0.6)}, {"token": " C", "logprob": math.log(0.2)},
                              {"token": "**", "logprob": math.log(0.2)}])
        self.assertAlmostEqual(p["A"], 0.75)
        self.assertAlmostEqual(p["B"], 0.0)
        self.assertAlmostEqual(p["mass"], 0.8)
        self.assertIsNone(sot.letter_probs([]))

    def test_outcome(self):
        self.assertEqual(sot.sot_outcome("A", "A"), 1.0)
        self.assertEqual(sot.sot_outcome("A", "B"), 0.0)
        self.assertEqual(sot.sot_outcome("C", "B"), 0.5)

    def test_per_prompt_debiasing(self):
        rows = [{"id": "q1", "vs": "x", "order": "sot_first", "sot_is": "A", "verdict": "A", "probs": None},
                {"id": "q1", "vs": "x", "order": "sot_second", "sot_is": "B", "verdict": "A", "probs": None},
                {"id": "q2", "vs": "x", "order": "sot_first", "sot_is": "A", "verdict": "A", "probs": None},
                {"id": "q2", "vs": "x", "order": "sot_second", "sot_is": "B", "verdict": "B", "probs": None}]
        pp = analyze_sot.per_prompt(rows, "x", ["q1", "q2"])
        self.assertEqual(pp["q1"]["score"], 0.5)  # position-driven flip = tie
        self.assertFalse(pp["q1"]["consistent"])
        self.assertEqual(pp["q2"]["score"], 1.0)
        s = analyze_sot.summarize(pp, 200)
        self.assertEqual((s["win"], s["tie"], s["loss"]), (1, 1, 0))
        with self.assertRaises(SystemExit):
            analyze_sot.per_prompt(rows[:1], "x", ["q1"])

    def test_pick_best(self):
        q = {"a": {"score": 0.6, "soft": 0.5}, "b": {"score": 0.4, "soft": 0.5}, "c": {"score": 0.4, "soft": 0.3}}
        self.assertEqual(analyze_sot.pick_best(q), "c")


class SpeedModel(unittest.TestCase):
    def test_parallel_time(self):
        e = {"fallback": False, "outline_tokens": 50, "outline_prompt_tokens": 100,
             "expansions": [{"model": "a", "n_tokens": 100, "prompt_tokens": 0},
                            {"model": "b", "n_tokens": 200, "prompt_tokens": 0},
                            {"model": "a", "n_tokens": 100, "prompt_tokens": 0}]}
        speed = {"a": 50.0, "b": 100.0}.get
        # 2 RTT + outline 50/50 + max(a: 200/50, b: 200/100)
        self.assertAlmostEqual(analyze_sot.t_parallel(e, speed, "a", 0.1, None), 0.2 + 1 + 4)
        self.assertEqual(analyze_sot.parallel_tokens(e), 450)
        self.assertEqual(analyze_sot.parallel_tokens({**e, "fallback": True, "fallback_tokens": 300}), 300)
        fb = {"fallback": True, "outline_tokens": 50, "outline_prompt_tokens": 0, "fallback_tokens": 500,
              "fallback_prompt_tokens": 0, "expansions": []}
        self.assertAlmostEqual(analyze_sot.t_parallel(fb, speed, "a", 0.1, None), 0.2 + 1 + 10)
        self.assertAlmostEqual(analyze_sot.t_single(500, 100, 50.0, 0.1, None), 10.1)
        self.assertAlmostEqual(analyze_sot.t_single(500, 100, 50.0, 0.1, 1000.0), 10.2)


class Stages(unittest.TestCase):
    def test_sot_plan_order(self):
        jobs = colab_jobs.PLANS["sot-1"]
        stages = [colab_jobs.stage_of(j) for j in jobs]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual(set(stages), {0, 1, 2})
        judge = [j for j in jobs if j[0] == "sot-judge"]
        self.assertEqual(len(judge), 1)
        self.assertEqual(colab_jobs.stage_of(("solo", "x", "all")), 0)
        cmd = colab_jobs.job_cmd("sot-base", colab_jobs.SOT["outline"], "all")
        self.assertIn("outline", cmd)
        self.assertNotIn("outline", colab_jobs.job_cmd("sot-base", colab_jobs.SOT["peers"][1], "all"))

    def test_run_stages_waits_and_cancels(self):
        jobs = [("sot-base", "a", "all"), ("sot-base", "b", "all"), ("sot-expand", "a", "all"),
                ("sot-judge", "j", "all")]
        log, lock, states = [], threading.Lock(), {}

        def run(job, fail=()):
            with lock:
                log.append(("start", job))
            time.sleep(0.05 if job[1] == "a" else 0.01)
            with lock:
                log.append(("end", job))
            return job[0] not in fail

        with ThreadPoolExecutor(max_workers=3) as ex:
            res = colab_jobs.run_stages(jobs, run, ex, state=lambda j, v: states.__setitem__(j, v))
        self.assertEqual(res, [True] * 4)
        pos = {e: i for i, e in enumerate(log)}
        self.assertLess(pos[("end", jobs[0])], pos[("start", jobs[2])])  # expand after every baseline
        self.assertLess(pos[("end", jobs[1])], pos[("start", jobs[2])])
        self.assertLess(pos[("end", jobs[2])], pos[("start", jobs[3])])

        log.clear()
        with ThreadPoolExecutor(max_workers=3) as ex:
            res = colab_jobs.run_stages(jobs, lambda j: run(j, fail=("sot-expand",)), ex,
                                        state=lambda j, v: states.__setitem__(j, v))
        self.assertEqual(res.count(False), 2)
        self.assertNotIn(("start", jobs[3]), log)  # the judge never starts
        self.assertIn("annulé", states[jobs[3]])

        def boom(j):
            raise RuntimeError("x")
        with ThreadPoolExecutor(max_workers=3) as ex:
            res = colab_jobs.run_stages(jobs, boom, ex, state=lambda j, v: states.__setitem__(j, v))
        self.assertEqual(res, [False] * 4)


if __name__ == "__main__":
    unittest.main()
