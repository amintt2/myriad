"""CPU-only protocol tests using invented problems, models, scores, and probabilities."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

ROOT = Path(__file__).resolve().parents[2]
# Discovery from phase0 must still import the package from the repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase0.percees_protocol import (
    PROMPTS, TEST_PROMPT, Calibration, Campaign, CandidateInputs, CandidateScore, Config, Distribution,
    Freeze, GraderGold, Model, Problem, SpeculationPlan, align_distribution, candidate_payload,
    choose_family_prompts, default_configs, digest, estimate_cost, infer, load_campaign, p07_budget,
    p11_training_plan, plan_jobs, pool_probabilities, select_functional, speculation_metrics,
)


def fixture() -> Campaign:
    # These identifiers are deliberately fictional: no model licence or speed claims.
    models = tuple(Model(f"fixture-{family}", family, "revision-fixture", "MIT", "fixture-only",
                         "tokenizer-fixture", "native-template-fixture", 1.0, 100.0, 20.0, 2.0)
                   for family in ("a", "b", "c"))
    problems = (Problem("fixture", "d1", "dev", "integer", "Return twice an integer.", "invented"),
                Problem("fixture", "d2", "dev", "integer", "Return three times an integer.", "invented"),
                Problem("fixture", "t1", "test", "integer", "Return four times an integer.", "invented"),
                Problem("external", "s1", "train", "integer", "Return seven times an integer.",
                        "invented external synthetic generator", True))
    return Campaign(problems, models, default_configs())


def calibration() -> Calibration:
    return Calibration("integer", ("fixture/d1", "fixture/d2"), 100, 100)


def freeze_for(campaign: Campaign) -> Freeze:
    selected = {}
    for config in campaign.configs:
        selected.setdefault(config.idea, config.id)
    return Freeze(campaign.hash, "external-preregistration-fixture", "2026-10-10T00:00:00Z",
                  ("fixture/d1", "fixture/d2"), tuple(sorted(selected.items())),
                  tuple((m.family, "contract_edges") for m in campaign.models), digest(asdict(calibration())))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.campaign = fixture()

    def test_manifest_duplicate_ids_and_cross_split_prompt_leak(self):
        duplicate_key = replace(self.campaign.problems[2], id="d1")
        duplicate_text = replace(self.campaign.problems[2], prompt=" RETURN   TWICE AN INTEGER. ")
        for duplicate in (duplicate_key, duplicate_text):
            with self.subTest(duplicate=duplicate):
                with self.assertRaises(ValueError):
                    replace(self.campaign, problems=self.campaign.problems[:2] + (duplicate,)).validate()

    def test_train_is_external_and_deduplicated_from_benchmarks(self):
        bad = replace(self.campaign.problems[-1], synthetic_external=False)
        with self.assertRaises(ValueError):
            replace(self.campaign, problems=self.campaign.problems[:-1] + (bad,)).validate()
        duplicate = replace(self.campaign.problems[-1], prompt=self.campaign.problems[0].prompt)
        with self.assertRaises(ValueError):
            replace(self.campaign, problems=self.campaign.problems[:-1] + (duplicate,)).validate()

    def test_loader_refuses_gold_solution_hidden_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            for field in ("solution", "hidden_tests", "gold", "candidate"):
                data = asdict(self.campaign)
                data["problems"][0][field] = "SENTINEL_SECRET"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(TypeError):
                    load_campaign(path)

    def test_test_freeze_and_mutation_guards(self):
        with self.assertRaises(ValueError):
            plan_jobs(self.campaign, "test")
        freeze = freeze_for(self.campaign)
        test_jobs = plan_jobs(self.campaign, "test", freeze)
        self.assertTrue(test_jobs)
        self.assertEqual({j["problem_id"] for j in test_jobs}, {"fixture/t1"})
        self.assertEqual(len({j["config"]["id"] for j in test_jobs}), len(freeze.chosen_configs))
        with self.assertRaises(ValueError):
            plan_jobs(replace(self.campaign, answer_tokens=513), "test", freeze)
        with self.assertRaises(ValueError):
            plan_jobs(self.campaign, "test", replace(freeze, dev_ids=("fixture/t1",)))
        with self.assertRaises(ValueError):
            plan_jobs(self.campaign, "test", replace(freeze, chosen_configs=freeze.chosen_configs[:-1]))

    def test_deterministic_dry_run_never_calls_network_or_models(self):
        # No backend is passed to the planner. Repeated serialization must be byte-identical.
        left = plan_jobs(self.campaign)
        self.assertEqual(left, plan_jobs(self.campaign))
        self.assertEqual(len({j["job_id"] for j in left}), len(left))
        self.assertEqual({j["split"] for j in left}, {"dev"})
        self.assertTrue(all(j["execution"] == "dry-run" for j in left))
        self.assertTrue(all(j["model"]["trust_remote_code"] is False for j in left))

    def test_planning_hash_is_computed_once_and_filtered_jobs_are_identical(self):
        snapshot = self.campaign.hash
        with patch.object(Campaign, "hash", new_callable=PropertyMock, return_value=snapshot) as counter:
            jobs = plan_jobs(self.campaign)
        self.assertEqual(counter.call_count, 1)
        job = jobs[0]
        identity = (job["problem_id"], job["config"]["id"], job["model"]["id"], job["seed"])
        filtered = plan_jobs(self.campaign, only=identity)
        self.assertEqual(filtered, [j for j in jobs if (j["problem_id"], j["config"]["id"],
                                                     j["model"]["id"], j["seed"]) == identity])

    def test_filtered_jobs_preserve_global_budget_validation(self):
        invalid = replace(self.campaign, p07_proxy_budget=1000)
        identity = ("fixture/d1", "P06-n1", "fixture-a", self.campaign.seeds[0])
        with self.assertRaises(ValueError):
            invalid.validate()
        for filters in ({}, {"only": identity}):
            with self.assertRaises(ValueError):
                plan_jobs(invalid, **filters)
        job = plan_jobs(self.campaign, only=identity)[0]
        with self.assertRaises(ValueError):
            infer(job, None, invalid, enabled=True)

    def test_test_generation_is_prompt_only_and_independent_of_candidate_parameters(self):
        jobs = [j for j in plan_jobs(self.campaign) if j["kind"] == "test_generation"]
        prompt = self.campaign.problems[0].prompt
        first = [j for j in jobs if j["problem_id"] == "fixture/d1"]
        self.assertTrue(first)
        self.assertTrue(all(j["messages"] == [{"role": "user", "content": TEST_PROMPT + prompt}] for j in first))
        self.assertEqual(len({json.dumps(j["messages"]) for j in first}), 1)
        self.assertTrue(all(j["candidate_independent"] for j in first))
        candidates = [j for j in plan_jobs(self.campaign) if j["kind"] == "candidate_plan"]
        self.assertTrue(all(TEST_PROMPT not in j["messages"][0]["content"] for j in candidates))

    def test_gold_never_enters_candidate_payload(self):
        inputs = CandidateInputs(("encoded-input",))
        self.assertEqual(candidate_payload(inputs), {"inputs": ["encoded-input"]})
        with self.assertRaises(ValueError):
            candidate_payload(GraderGold(("SENTINEL_GOLD",), ("SENTINEL_ASSERTION",)))

    def test_cpu_ceiling_is_bounded(self):
        for ceiling in (0, 31, math.inf):
            with self.assertRaises(ValueError):
                replace(self.campaign, candidate_cpu_seconds=ceiling).validate()

    def test_boolean_attestations_cannot_be_truthy_strings(self):
        with self.assertRaises(ValueError):
            replace(self.campaign.models[0], license_verified="false").validate()
        with self.assertRaises(ValueError):
            replace(self.campaign.problems[-1], synthetic_external="false").validate()
        with self.assertRaises(ValueError):
            p11_training_plan(self.campaign, e13_ready="false", e14_ready=True, p06_p07_gain_validated=True)
        with self.assertRaises(ValueError):
            Config("bad", "P07", width=3.0).validate()

    def test_family_vote_resists_candidate_multiplicity(self):
        scores = [CandidateScore(f"a{i}", "a", "wrong", 3, 3, "same-suite") for i in range(5)]
        scores += [CandidateScore("b", "b", "right", 3, 3, "same-suite"),
                   CandidateScore("c", "c", "right", 3, 3, "same-suite")]
        self.assertEqual(select_functional(scores, calibration(), self.campaign, "fixture/d1"), "right")

    def test_calibration_abstains_and_refuses_test_or_changed_thresholds(self):
        score = [CandidateScore("a", "a", "answer", 3, 3, "suite")]
        for cal in (None, replace(calibration(), total_proposals=10, valid_proposals=10),
                    replace(calibration(), valid_proposals=60)):
            self.assertIsNone(select_functional(score, cal, self.campaign, "fixture/d1"))
        with self.assertRaises(ValueError):
            select_functional(score, replace(calibration(), dev_ids=("fixture/t1",)), self.campaign, "fixture/d1")
        with self.assertRaises(ValueError):
            select_functional(score, replace(calibration(), minimum_lower_bound=0.1), self.campaign, "fixture/d1")
        other = replace(self.campaign.problems[2], domain="unvalidated")
        campaign = replace(self.campaign, problems=self.campaign.problems[:2] + (other,))
        self.assertIsNone(select_functional(score, calibration(), campaign, "fixture/t1", freeze_for(campaign)))

    def test_selector_requires_common_suite_and_frozen_calibration(self):
        score = CandidateScore("a", "a", "answer", 3, 3, "suite")
        for second in (replace(score, id="b", family="b", test_set_hash="different"),
                       replace(score, id="b", family="b", total=4)):
            with self.assertRaises(ValueError):
                select_functional([score, second], calibration(), self.campaign, "fixture/d1")
        with self.assertRaises(ValueError):
            select_functional([score], calibration(), self.campaign, "fixture/t1")
        with self.assertRaises(ValueError):
            select_functional([score], replace(calibration(), valid_proposals=99), self.campaign,
                              "fixture/t1", freeze_for(self.campaign))
        self.assertEqual(select_functional([score], calibration(), self.campaign, "fixture/t1",
                                          freeze_for(self.campaign)), "answer")

    def test_depth_width_sharing_accounts_for_generation_and_recipient_prefill(self):
        solo = Config("solo", "P07", width=3, reflection_tokens=1000)
        plain = p07_budget(solo, 100, 200, 5000)
        shared = p07_budget(replace(solo, share_tokens=128), 100, 200, 5000)
        self.assertEqual(shared["prefill_tokens"] - plain["prefill_tokens"], 2 * 128)
        self.assertEqual(shared["decode_tokens"] - plain["decode_tokens"], 128)
        self.assertEqual(shared["used_proxy"] + shared["unused_proxy"], 5000)
        self.assertTrue(shared["approximate"])
        with self.assertRaises(ValueError):
            p07_budget(replace(solo, share_tokens=128), 100, 200, plain["used_proxy"])
        with self.assertRaises(ValueError):
            p07_budget(Config("deep", "P07", reflection_tokens=16000, width=5), 512, 512, 10000)

    def test_cost_is_calculable_and_not_a_measurement(self):
        estimate = estimate_cost(self.campaign.models[0], 100, 40)
        self.assertEqual(estimate["estimated_seconds_serial"], 3)
        self.assertAlmostEqual(estimate["estimated_cost"], 3 * 2 / 3600)
        self.assertFalse(estimate["measured"])
        self.assertIn("training", estimate["excludes"])

    def test_prompt_selection_is_family_dev_only_with_matched_coverage(self):
        scores = [(m.family, variant, p.key, float(variant == "contract_edges"))
                  for m in self.campaign.models for variant in PROMPTS
                  for p in self.campaign.problems if p.split == "dev"]
        selected = choose_family_prompts(scores, self.campaign)
        self.assertEqual(selected, {"a": "contract_edges", "b": "contract_edges", "c": "contract_edges"})
        for bad in (scores[:-1], scores + [("a", "standard", "fixture/t1", 1.0)], scores + [scores[0]]):
            with self.assertRaises(ValueError):
                choose_family_prompts(bad, self.campaign)
        jobs = plan_jobs(self.campaign, "test", freeze_for(self.campaign))
        self.assertTrue(all(j["prompt_variant"] == "contract_edges" for j in jobs if j["config"]["idea"] == "P08"))

    def test_inference_is_explicit_and_preserves_native_template(self):
        class Backend:
            calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return "fixture-output"

        backend = Backend()
        job = next(j for j in plan_jobs(self.campaign) if j["kind"] == "test_generation")
        with self.assertRaises(ValueError):
            infer(job, backend, self.campaign)
        with self.assertRaises(ValueError):
            infer(job, backend, self.campaign, enabled=True)
        self.assertEqual(backend.calls, [])
        verified = replace(self.campaign, models=tuple(replace(m, license_verified=True) for m in self.campaign.models))
        job = next(j for j in plan_jobs(verified) if j["kind"] == "test_generation")
        for flag in ("false", "true", 1, None, False):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    infer(job, backend, verified, enabled=flag)
        self.assertEqual(backend.calls, [])
        self.assertEqual(infer(job, backend, verified, enabled=True), "fixture-output")
        self.assertEqual(backend.calls[0]["native_chat_template"], "native-template-fixture")
        unsupported = next(j for j in plan_jobs(self.campaign) if j["config"]["idea"] == "P07")
        with self.assertRaises(ValueError):
            infer(unsupported, backend, self.campaign, enabled=True)
        job["messages"][0]["content"] += " tampering"
        with self.assertRaises(ValueError):
            infer(job, backend, verified, enabled=True)
        test_job = next(j for j in plan_jobs(verified, "test", freeze_for(verified)) if j["kind"] == "test_generation")
        with self.assertRaises(ValueError):
            infer(test_job, backend, verified, enabled=True)

    def test_speculation_requires_all_tokenizers_or_string_law_verifier(self):
        exact = SpeculationPlan(("t", "t"), "t", 8, "identical-tokenizers", "target-ratio-residual")
        exact.validate()
        with self.assertRaises(ValueError):
            replace(exact, draft_tokenizers=("t", "other")).validate()
        retokenize = replace(exact, draft_tokenizers=("other",), alignment="retokenize-text")
        with self.assertRaises(ValueError):
            retokenize.validate()
        replace(retokenize, exact_verifier="target-string-law-residual").validate()
        with self.assertRaises(ValueError):
            replace(exact, transport="WAN-per-token").validate()
        metrics = speculation_metrics(8, 4, (0.01, 0.03))
        self.assertEqual(metrics["acceptance_rate"], 0.5)
        self.assertEqual(metrics["rtt_mean_seconds"], 0.02)
        self.assertIsNone(speculation_metrics(0, 0, ())["acceptance_rate"])
        with self.assertRaises(ValueError):
            speculation_metrics(4, 5, ())

    def test_pool_refuses_vocabularies_and_incomplete_mapping(self):
        a = Distribution("answers", ("A", "B"), (0.8, 0.2))
        b = Distribution("other", ("yes", "no"), (0.6, 0.4))
        with self.assertRaises(ValueError):
            pool_probabilities((a, b))
        for mapping in ({"yes": "A"}, {"yes": "A", "no": "A"}, {"yes": "A", "no": "C"}):
            with self.assertRaises(ValueError):
                align_distribution(b, "answers", ("A", "B"), mapping)
        aligned = align_distribution(b, "answers", ("A", "B"), {"yes": "B", "no": "A"})
        self.assertEqual(aligned.probabilities, (0.4, 0.6))
        pooled = pool_probabilities((a, aligned))
        self.assertAlmostEqual(pooled.probabilities[0], 0.6)

    def test_product_temperature_and_zero_support(self):
        a = Distribution("answer", ("A", "B"), (0.8, 0.2))
        b = Distribution("answer", ("A", "B"), (0.5, 0.5))
        geometric = pool_probabilities((a, b), operator="product")
        self.assertAlmostEqual(geometric.probabilities[0], 2 / 3)
        warmed = pool_probabilities((a,), temperature=2)
        self.assertAlmostEqual(warmed.probabilities[0], 2 / 3)
        left = replace(a, probabilities=(1.0, 0.0))
        right = replace(a, probabilities=(0.0, 1.0))
        with self.assertRaises(ValueError):
            pool_probabilities((left, right), operator="product")
        self.assertEqual(pool_probabilities((left, right), operator="product", weights=(1, 0)), left)
        for values in ((1.0, -1.0), (2.0, 1.0), (math.nan, 1.0)):
            with self.assertRaises(ValueError):
                pool_probabilities((replace(a, probabilities=values),))
        for temperature in (0, math.nan):
            with self.assertRaises(ValueError):
                pool_probabilities((a,), temperature=temperature)

    def test_product_keeps_small_nonzero_support_until_final_normalization(self):
        a = Distribution("answer", ("A", "B"), (1.0, 1e-200))
        b = Distribution("answer", ("A", "B"), (1e-200, 1.0))
        for temperature in (0.5, 1e-300):
            result = pool_probabilities((a, b), operator="product", temperature=temperature)
            self.assertEqual(result.probabilities, (0.5, 0.5))
        weighted = pool_probabilities((a, b), operator="product", temperature=0.5, weights=(1, 3))
        self.assertAlmostEqual(weighted.probabilities[0] / 1e-200, 1.0)

    def test_positive_weight_support_survives_weight_normalization_underflow(self):
        a = Distribution("answer", ("A", "B"), (1.0, 0.0))
        b = Distribution("answer", ("A", "B"), (0.0, 1.0))
        with self.assertRaises(ValueError):
            pool_probabilities((a, b), operator="product", weights=(1e-300, 1e300))
        self.assertEqual(pool_probabilities((a, b), operator="product", weights=(0, 1e300)), b)
        # A tiny positive weight still constrains support when the intersection is nonempty.
        mixed = replace(b, probabilities=(0.5, 0.5))
        self.assertEqual(pool_probabilities((a, mixed), operator="product", weights=(1e-300, 1e300)), a)

    def test_training_is_a_gated_future_plan_only(self):
        with self.assertRaises(ValueError):
            p11_training_plan(self.campaign, e13_ready=False, e14_ready=True, p06_p07_gain_validated=True)
        with self.assertRaises(ValueError):
            p11_training_plan(self.campaign, e13_ready=True, e14_ready=True, p06_p07_gain_validated=False)
        plan = p11_training_plan(self.campaign, e13_ready=True, e14_ready=True, p06_p07_gain_validated=True)
        self.assertFalse(plan["training_launched"])
        self.assertEqual(plan["train_ids"], ["external/s1"])
        self.assertEqual({j["config"]["idea"] for j in plan_jobs(self.campaign, "train")}, {"P11"})

    def test_cli_dry_run_and_test_guard_without_real_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = Path(tmp) / "manifest.json", Path(tmp) / "jobs.jsonl"
            manifest.write_text(json.dumps(asdict(self.campaign)), encoding="utf-8")
            command = [sys.executable, "-m", "phase0.percees_protocol", "--manifest", str(manifest),
                       "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["execution"], "dry-run")
            jobs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(jobs, plan_jobs(self.campaign))
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
            blocked = Path(tmp) / "blocked.jsonl"
            result = subprocess.run(command[:-1] + [str(blocked), "--split", "test"], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(blocked.exists())


if __name__ == "__main__":
    unittest.main()
