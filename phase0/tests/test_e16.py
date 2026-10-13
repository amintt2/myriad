"""E16 (bigger quantized models): table, Colab plan, analysis helpers and consumer-speed estimate. No GPU, no network."""
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analyze_e16 as A  # noqa: E402
import colab_jobs as CJ  # noqa: E402
import e16_quant as Q  # noqa: E402
import exec_e16  # noqa: E402
import speed_quant  # noqa: E402

PULL = re.compile(r"^(mc|gen|solo|sot|code|codeexec|aa|sciexec)_.*_colab_.*\.jsonl$")  # colab_phase0.sh, pull
Q8_BYTES = {"Qwen/Qwen3.5-9B": 9527502048, "google/gemma-4-12B-it": 12669646976,
            "mistralai/Ministral-3-14B-Instruct-2512": 14359836224}  # Hugging Face API, 2026-10-13


class TableTests(unittest.TestCase):
    def test_every_file_is_pinned_and_within_budget(self):
        seen = set()
        for q in Q.QUANT:
            self.assertNotIn((q.model, q.tag), seen)
            seen.add((q.model, q.tag))
            self.assertTrue(q.fits(), q)
            self.assertRegex(q.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(q.file.endswith(".gguf"))
            self.assertRegex(q.revision, r"^[0-9a-f]{12}$")
            self.assertIn(q.model, CJ.SOLO)  # the Q8_0 reference and the pinned tokenizer revision exist for E4
            self.assertNotIn("_", Q.slug(q.tag))

    def test_suffix_lets_the_pull_find_the_files(self):
        q = Q.get("Qwen/Qwen3.5-9B", "IQ3_XXS")
        self.assertEqual(q.suffix, "_iq3xxs_colab")
        name = f"solo_{q.model.replace('/', '__')}{q.suffix}_gsm8k_dev.jsonl"
        self.assertRegex(name, PULL)
        self.assertRegex(f"code_{q.model.replace('/', '__')}{q.suffix}_mbppplus_test.jsonl", PULL)
        self.assertRegex(f"codeexec_{q.exec_tag}{q.suffix}_mbppplus_test.jsonl", PULL)
        self.assertRegex(f"solo_{Q.SPEED_STEM}_colab_bench_all.jsonl", PULL)
        # a model name in a result file never collides with another quantization of the same model
        self.assertEqual(len({q.suffix for q in Q.QUANT if q.model == "google/gemma-4-12B-it"}),
                         len([q for q in Q.QUANT if q.model == "google/gemma-4-12B-it"]))

    def test_overhead_is_the_one_of_the_q8_reservation(self):
        for model, size in Q8_BYTES.items():
            fixed = CJ.SOLO[model][2] - size / Q.GIB
            self.assertGreaterEqual(Q.OVERHEAD_GIB[model], fixed - 0.01, model)
            self.assertLess(Q.OVERHEAD_GIB[model] - fixed, 0.2, model)
        for q in Q.QUANT:
            self.assertLess(Q.reserve_gib(q), CJ.SOLO[q.model][2])  # less than the Q8_0 job
            self.assertGreater(Q.reserve_gib(q), q.gib)

    def test_speed_models_have_a_q8_entry_and_a_size(self):
        for m in Q.SPEED_Q8:
            self.assertIn(m, CJ.SOLO)
            self.assertIn(m, A.models.SIZES)

    def test_groups_by_tag(self):
        g = Q.groups()
        self.assertEqual([q.model for q in g["IQ3_XXS"]], Q.models())
        self.assertEqual(len(g["QAT_Q4_0"]), 1)
        with self.assertRaises(KeyError):
            Q.get("Qwen/Qwen3.5-9B", "Q2_K")
        self.assertEqual(Q.split_label(Q.label("a/b", "IQ3_M")), ("a/b", "IQ3_M"))


class PlanTests(unittest.TestCase):
    def test_plans(self):
        p = CJ.PLANS["e16-1"]
        self.assertEqual(len(p), len(set(p)))
        kinds = [j[0] for j in p]
        self.assertEqual(kinds.count("solo-q"), 12)
        self.assertEqual(kinds.count("code-q"), 6)
        self.assertEqual(kinds.count("code-exec"), 1)
        self.assertEqual(kinds.count("solo-speed"), 1)
        self.assertEqual({(m, t) for k, m, t in p if k == "code-q"}, Q.CODE_BOUNDS)
        rest = CJ.PLANS["e16-2"]
        self.assertEqual({(m, t) for k, m, t in rest if k == "code-q"} | Q.CODE_BOUNDS, {(q.model, q.tag) for q in Q.QUANT})
        self.assertEqual([j[0] for j in CJ.PLANS["e16-mc"]].count("code-q"), 0)
        self.assertNotIn("code-exec", [j[0] for j in CJ.PLANS["e16-mc"]])
        self.assertIn("solo-speed", [j[0] for j in CJ.PLANS["e16-mc"]])
        # e16-mc then e16-code is e16-1; e16-2 only adds code for the quantizations that are not bounds
        self.assertEqual(sorted(CJ.PLANS["e16-mc"] + CJ.PLANS["e16-code"]), sorted(CJ.PLANS["e16-1"]))
        self.assertEqual([j[0] for j in rest].count("solo-q"), 0)
        self.assertNotIn("solo-speed", [j[0] for j in rest])
        # the existing plans are untouched
        self.assertEqual(CJ.PLANS["solo-1"], [("solo", m, "all") for m in CJ.SOLO])

    def test_stages_order_execution_after_generation_and_speed_last(self):
        self.assertLess(CJ.STAGE["code-q"], CJ.STAGE["code-exec"])
        self.assertLess(CJ.STAGE["code-exec"], CJ.STAGE["solo-speed"])
        self.assertEqual(CJ.STAGE["solo-q"], 0)
        self.assertIn("solo-q", CJ.NO_MODEL)  # the main loop does not look for them in SOLO/GGUF
        self.assertIn("code-exec", CJ.NO_MODEL)

    def test_commands_and_reservations(self):
        for kind, model, tag in CJ.PLANS["e16-1"]:
            cmd = CJ.job_cmd(kind, model, tag)
            if kind in ("solo-q", "code-q"):
                q = Q.get(model, tag)
                self.assertEqual(cmd[1], "run_solo.py" if kind == "solo-q" else "run_code.py")
                self.assertEqual(cmd[cmd.index("--suffix") + 1], q.suffix)
                self.assertEqual(Path(cmd[cmd.index("--gguf") + 1]).name, q.file)
                self.assertEqual(cmd[cmd.index("--model") + 1], model)
                self.assertEqual(cmd[cmd.index("--parallel") + 1], "16")  # E4's 16 slots of 3072 tokens
                self.assertEqual(CJ.e16_need(kind, model, tag), Q.reserve_gib(q))
            elif kind == "code-exec":
                self.assertEqual(cmd[1], "exec_e16.py")
            else:
                self.assertEqual(cmd[1:], ["speed_quant.py"])
                self.assertGreaterEqual(CJ.e16_need(kind, model, tag), 14)
            self.assertIn(kind, CJ.E16_KINDS + ("code-exec",))
        # the other kinds still go through the original function
        self.assertEqual(CJ.job_cmd("solo", "Qwen/Qwen3.5-4B", "all")[1], "run_solo.py")

    def test_every_job_fits_the_gpu_and_two_run_together(self):
        usable = Q.DEFAULT_GPU_GIB - 2 - 0.5  # nvidia-smi free memory of a 40 GB card, minus MemoryBudget's margin
        need = [CJ.e16_need(k, m, t) for k, m, t in CJ.PLANS["e16-1"] if k in CJ.E16_KINDS]
        self.assertLessEqual(max(need), usable / 2)

    def test_checkpoint_families_of_the_pull(self):
        spec = importlib.util.spec_from_file_location("session_json_e16", HERE / "colab" / "session_json.py")
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:  # fcntl on Windows
            self.skipTest(str(e))
        self.assertEqual(mod.families("e16-1"), ["code", "codeexec", "solo"])
        self.assertEqual(mod.families("e16-mc"), ["solo"])

    def test_prepare_does_nothing_for_other_plans(self):
        with mock.patch.object(CJ, "sh", side_effect=AssertionError("no download")):
            CJ.prepare_e16([("solo", "Qwen/Qwen3.5-4B", "all")])

    def test_exec_launcher_runs_each_quantization_with_generations(self):
        have = {("Qwen/Qwen3.5-9B", "_iq3xxs_colab"), ("google/gemma-4-12B-it", "_iq3xxs_colab"),
                ("google/gemma-4-12B-it", "_qatq40_colab")}

        def fake(path):  # gen_path(model, suffix, bench, split)
            name = path.name
            return {} if any(f"{m.replace('/', '__')}{s}_" in name for m, s in have) else None
        with mock.patch.object(exec_e16, "read_manifest", fake):
            self.assertEqual([(q.model, q.suffix) for q in exec_e16.pending()], sorted(have, key=lambda x: [
                (q.model, q.suffix) for q in Q.QUANT].index(x)))
            self.assertEqual([(q.model, q.tag) for q in exec_e16.pending(["QAT_Q4_0"])], [("google/gemma-4-12B-it", "QAT_Q4_0")])

    def test_execution_files_never_change_their_model_list(self):
        # e16-2 runs after e16-1: a file set written for one (model, quantization) is never given another model
        tags = [q.exec_tag + q.suffix for q in Q.QUANT]
        self.assertEqual(len(set(tags)), len(tags))
        for q in Q.QUANT:
            self.assertNotIn("_", q.exec_tag)
            self.assertTrue(q.exec_tag.startswith(Q.EXEC_TAG))
            self.assertRegex(f"codeexec_{q.exec_tag}{q.suffix}_humanevalplus_dev.jsonl", PULL)
        self.assertEqual(Q.get("google/gemma-4-12B-it", "IQ3_XXS").exec_tag, "e16gemma412bit")


class SpeedTests(unittest.TestCase):
    RAW = json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 4100.5, "stddev_ts": 12.0, "gpu_info": "A100", "build_commit": "abc"},
                      {"n_prompt": 0, "n_gen": 128, "avg_ts": 101.25, "stddev_ts": 0.7, "gpu_info": "A100", "build_commit": "abc"}])

    def test_gpu_identity_is_recorded(self):
        ok = mock.Mock(returncode=0, stdout="NVIDIA A100-SXM4-40GB, 40960 MiB\n")
        with mock.patch.object(speed_quant.subprocess, "run", return_value=ok):
            self.assertEqual(speed_quant.gpu_name(), "NVIDIA A100-SXM4-40GB, 40960 MiB")
        with mock.patch.object(speed_quant.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
            with self.assertRaises(SystemExit):
                speed_quant.gpu_name()

    def test_speeds_from_two_gpus_are_refused(self):
        s = [synth("m/a@Q8_0", ["right"])]
        speed = {"m/a@Q8_0": {"gpu": "A100", "tg": {"tps": 1}, "pp": {"tps": 1}},
                 "m/b@Q8_0": {"gpu": "L4", "tg": {"tps": 1}, "pp": {"tps": 1}}}
        with self.assertRaises(SystemExit):
            A.build_mc(s, [], {}, speed)

    def test_parse(self):
        r = speed_quant.parse(self.RAW)
        self.assertEqual((r["tg"]["tps"], r["pp"]["tps"], r["gpu"]), (101.25, 4100.5, "A100"))
        with self.assertRaises(RuntimeError):
            speed_quant.parse("[]")

    def test_entries(self):
        e = speed_quant.entries()
        self.assertEqual(len(e), len(Q.QUANT) + len(Q.SPEED_Q8))
        self.assertEqual(len({lab for lab, _, _ in e}), len(e))

    def test_consumer_estimate_scales_with_the_file(self):
        calib = [(46.0, 4280404704), (93.0, 1834426016)]
        lo, mid, hi = Q.estimate_consumer_tps(5_000_000_000, calib)
        self.assertLess(lo, mid)
        self.assertLessEqual(mid, hi)
        lo2, mid2, hi2 = Q.estimate_consumer_tps(10_000_000_000, calib)
        self.assertAlmostEqual(mid2, mid / 2)
        self.assertAlmostEqual(Q.implied_bandwidth_gbs(46.0, 4280404704), 196.9, places=1)
        with self.assertRaises(ValueError):
            Q.estimate_consumer_tps(1, [])

    def test_calibration_file_covers_the_dense_models(self):
        speeds = json.loads((HERE / "speeds_consumer.json").read_text(encoding="utf-8"))
        for m, _ in Q.CALIBRATION_DENSE:
            self.assertIn(m, speeds)
        calib = A.consumer_calibration()
        self.assertEqual(len(calib), len(Q.CALIBRATION_DENSE))
        # the estimate must be of the right order for a Q8 file it was calibrated on
        lo, mid, hi = Q.estimate_consumer_tps(4280404704, calib)
        self.assertLess(lo, 46.0 + 1e-9)
        self.assertGreater(hi, 46.0 - 1e-9)

    def test_hours(self):
        self.assertAlmostEqual(Q.a100_hours(8.0, 2), 4.0)


def synth(label, answers_by_bench, gib=4.0, tag="Q8_0", model=None, golds=None, n=12):
    """A Sys whose answers are given as {bench: list of 'right'/'wrong'/None per question}, dev = test pattern."""
    s = A.Sys(label, model or label.partition("@")[0], tag, int(gib * Q.GIB))
    for b in A.BENCHES:
        pat = answers_by_bench[b] if isinstance(answers_by_bench, dict) else answers_by_bench
        for sp in A.SPLITS:
            rows = {}
            for i in range(n):
                v = pat[i % len(pat)]
                rows[f"q{i}"] = {"answer": None if v is None else ("A" if v == "right" else "B"), "gold": "A",
                                 "mean_logp": None, "finish": "stop", "n_tokens": 10, "ms": 1}
            s.rows[(b, sp)] = rows
            s.mans[(b, sp)] = {"data": {"x": 1}, "prompt": "solo-v1", "max_tokens": 1, "temperature": 0.0, "thinking": False,
                               "ctx_per_slot": 3072, "bench": b, "logprobs": False, "model": s.model, "revision": "r",
                               "gguf": "f", "weights": {"size": s.size, "sha256": "0" * 64}, "engine": "e", "backend": "b"}
    return s


class AnalysisHelpers(unittest.TestCase):
    def test_best_by_dev_prefers_the_smaller_file_on_a_tie(self):
        a = synth("m/a@Q8_0", ["right", "wrong"], gib=5.0)
        b = synth("m/b@Q8_0", ["right", "wrong"], gib=3.0)
        c = synth("m/c@Q8_0", ["right", "right", "wrong"], gib=6.0)
        self.assertIs(A.best_by_dev([a, b]), b)
        self.assertIs(A.best_by_dev([a, b, c]), c)
        self.assertEqual(A.in_budget([a, b, c], 5.0), [a, b])

    def test_rates(self):
        s = synth("m/a@Q8_0", ["right", None, "wrong", "wrong"])
        self.assertAlmostEqual(A.abstain_rate(s), 25.0)
        self.assertAlmostEqual(A.macro(s, "test"), 25.0)
        self.assertEqual(A.mean_tokens(s), 10)

    def test_swarm_votes_and_pools(self):
        # three peers with errors on different questions: the weighted vote fixes every single error
        n = 12
        def peer(label, wrong):
            return synth(label, ["wrong" if i in wrong else "right" for i in range(n)], n=n)
        peers = [peer("m/a@Q8_0", {0, 1}), peer("m/b@Q8_0", {2, 3}), peer("m/c@Q8_0", {4, 5})]
        v = A.swarm_vectors(peers)
        self.assertEqual(set(v), set(A.BENCHES))
        self.assertEqual(A.macro_of(v), 100.0)
        self.assertEqual(len(A.pooled(v)), n * len(A.BENCHES))
        sc = A.scaling(peers)
        self.assertEqual(sorted(sc), [1, 2, 3])
        self.assertEqual(sc[3]["n_subsets"], 1)
        self.assertAlmostEqual(sc[1]["mean_all_subsets"], 100 * 10 / 12, places=6)
        self.assertGreaterEqual(sc[3]["mean_all_subsets"], sc[1]["mean_all_subsets"])

    def test_versus_is_paired_and_signed(self):
        n = 40
        good = synth("m/g@Q8_0", ["right"], n=n)
        bad = synth("m/b@Q8_0", ["right", "wrong"], n=n)
        g = A.versus(A.alone_vectors(good), A.alone_vectors(bad))
        self.assertGreater(g["pooled"]["mean"], 0)
        self.assertEqual(g["pooled"]["n"], n * len(A.BENCHES))
        self.assertIn("gsm8k", g)

    def test_protocol_check(self):
        a, b = synth("m/a@Q8_0", ["right"]), synth("m/b@Q8_0", ["right"])
        A.check_protocol([a, b])
        b.mans[("gsm8k", "test")]["prompt"] = "other"
        with self.assertRaises(SystemExit):
            A.check_protocol([a, b])
        b2 = synth("m/b@Q8_0", ["right"])
        b2.mans[("arc", "dev")]["engine"] = "e2"  # dev and test of one system must come from the same engine
        with self.assertRaises(SystemExit):
            A.check_protocol([a, b2])
        c = synth("m/c@Q8_0", ["right"], n=11)
        with self.assertRaises(SystemExit):
            A.check_protocol([a, c])

    def test_pinned_file_is_checked(self):
        q = Q.get("Qwen/Qwen3.5-9B", "IQ3_XXS")
        s = synth(Q.label(q.model, q.tag), ["right"], tag=q.tag, model=q.model)
        s.size = q.size
        for k in s.mans:
            s.mans[k]["weights"] = {"size": q.size, "sha256": q.sha256}
        A.check_protocol([s])
        for key in list(s.mans):  # any benchmark and split, not only the first
            good = s.mans[key]["weights"]
            s.mans[key]["weights"] = {"size": q.size, "sha256": "1" * 64}
            with self.assertRaises(SystemExit, msg=key):
                A.check_protocol([s])
            s.mans[key]["weights"] = good

    def test_code_provenance_across_groups(self):
        class AC:  # the parts of analyze_code used by the check
            PROTOCOL = ("prompt", "samples")
            check_splits = staticmethod(lambda b, fit, test: None)
        q = Q.get("Qwen/Qwen3.5-9B", "IQ3_XXS")
        key = (q.suffix, q.exec_tag)
        groups = {("_colab", "e11"): {"m/small": "m/small@Q8_0"}, key: {q.model: Q.label(q.model, q.tag)}}

        def info(model, **kw):
            man = {"prompt": "code-v1", "samples": 4, "data": {"d": 1}, **kw}
            return (None, None, None, {"manifests": {model: man}})
        pinned = {"weights": {"size": q.size, "sha256": q.sha256}}

        def loaded(**kw):
            return {(k, b, sp): info(next(iter(g)), **({**pinned, **kw} if k == key else {}))
                    for k, g in groups.items() for b in ("humanevalplus", "mbppplus") for sp in A.SPLITS}
        A.check_code_provenance(AC, groups, loaded(), [q])
        with self.assertRaises(SystemExit):  # another sample count would give the model more candidates in the vote
            A.check_code_provenance(AC, groups, loaded(samples=2), [q])
        with self.assertRaises(SystemExit):  # not the pinned file
            A.check_code_provenance(AC, groups, loaded(weights={"size": q.size, "sha256": "2" * 64}), [q])

    def test_bits_per_weight(self):
        q = Q.get("Qwen/Qwen3.5-9B", "IQ3_XXS")
        s = synth(Q.label(q.model, q.tag), ["right"], tag=q.tag, model=q.model)
        s.size = q.size
        self.assertTrue(s.quantized)
        self.assertAlmostEqual(s.bpw, q.size * 8 / 9e9)
        self.assertTrue(3.5 < s.bpw < 4.5)


def fake_compare(a, b, delta):
    n = len(a)
    return {"n": n, "mean": 100 * (sum(a) - sum(b)) / n, "lo95": 0.0, "hi95": 0.0, "verdict": "stub", "delta": delta}


class EndToEnd(unittest.TestCase):
    """The whole report on E4/E11's own Q8_0 results, read as if three of the models had been quantized (their files
    are the same, so the cost of every quantization must be exactly zero)."""

    TAGS = ("IQ3_XXS", "Q3_K_M")
    BIG = ("Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "mistralai/Ministral-3-14B-Instruct-2512")

    def setUp(self):
        if not (A.RESULTS / "solo_Qwen__Qwen3.5-9B_colab_gsm8k_test.jsonl").exists():
            self.skipTest("résultats E4 absents")
        real = A.load_e4
        try:
            quants = []
            for m in self.BIG:
                w = A.load_system(m, "Q8_0", A.SMALL_SUFFIX).mans[("gsm8k", "dev")]["weights"]
                quants += [Q.Quant(m, t, "r", "f.gguf", w["size"], w["sha256"], "0" * 12) for t in self.TAGS]
        except (OSError, SystemExit, ValueError) as e:
            self.skipTest(str(e))
        patches = [mock.patch.object(Q, "QUANT", quants), mock.patch.object(Q, "BUDGET_GIB", 20.0),
                   mock.patch.object(Q, "TIERS_GIB", (12.0, 20.0)), mock.patch.object(A, "compare", fake_compare),
                   mock.patch.object(A, "load_e4", lambda m, suffix, b, sp: real(m, A.SMALL_SUFFIX, b, sp))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_mc_report(self):
        systems, L, summary = [], [], {"absent": []}
        for m in list(A.FAMILIES.values()) + A.EXTRA + A.REFS:
            systems.append(A.load_system(m, "Q8_0", A.SMALL_SUFFIX))
        for q in Q.QUANT:
            systems.append(A.load_system(q.model, q.tag, q.suffix))
        A.check_protocol(systems)
        A.build_mc(systems, L, summary, {})
        text = "\n".join(L)
        for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
            self.assertIn(h, text)
        self.assertEqual(summary["choices"]["best_small"], "Qwen/Qwen3.5-4B@Q8_0")
        for cost in summary["quantization_cost"].values():
            self.assertEqual(cost["mean"], 0.0)
        # the 7-family swarm reproduces E4's number (analyze_e4: 87.0 macro, weighted vote)
        self.assertAlmostEqual(summary["swarms"]["essaim de 7 familles Q8_0 (E4)"]["macro_test"], 87.0, places=1)
        self.assertEqual(sorted(summary["scaling"]["big"]), [1, 2, 3])
        json.dumps(summary, default=str)

    def test_code_report(self):
        import analyze_code as AC
        real_rm, real_load = A.read_manifest, AC.load
        strip = lambda p: p.with_name(re.sub(r"_(iq3xxs|q3km)_colab", "_colab", p.name))
        try:
            with mock.patch.object(A, "read_manifest", lambda p: real_rm(strip(p))):
                with mock.patch.object(AC, "load", lambda b, sp, suffix, tag, mm: real_load(b, sp, "_colab", "e11", mm)):
                    L, summary = [], {}
                    A.build_code(L, summary, {})
        except (OSError, SystemExit, ValueError) as e:
            self.skipTest(f"données E11 indisponibles : {e}")
        text = "\n".join(L)
        self.assertIn("meilleur gros seul", text)
        self.assertEqual(summary["code"]["systems"]["Qwen/Qwen3.5-9B@IQ3_XXS"], summary["code"]["systems"]["Qwen/Qwen3.5-9B@Q3_K_M"])
        self.assertEqual(len(summary["code"]["comparisons"]), 3)


if __name__ == "__main__":
    unittest.main()
