"""Offline VERIF-1 tests. Fake engines interpret declarative data, never candidate source."""
import hashlib
import io
import json
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from essaim.verif1_runtime import GGUFEngine, LinuxSandbox, SimulatedEngine, SimulatedSandbox
from verif1_prepare import public_contract, read_dev_prefix
from verif1_protocol import (AUTHOR_DECODER, AUTHORS, FAMILIES, MODELS, WEIGHTS, Ledger,
                             canonical, combine, confirmation_open,
                             cost, differential_scores, exclusive, file_sha, parse_judgment, parse_proposal,
                             relation, select, sha, shortlist, supported_scores, write_new)
from verif1_run import (inputs_for, judge_prompt, mutation_inputs, proposed, question_scores, run_pipeline,
                        validate_bundle)


def candidate(family="Qwen", sample=0, signature="a" * 16):
    model = next((m for m in MODELS if FAMILIES[m] == family), MODELS[0])
    source = "def f(x):\n    return x\n"
    return {"model": model, "family": family, "sample": sample, "code": source,
            "code_sha256": hashlib.sha256(source.encode()).hexdigest(), "prog": "public",
            "cost": {"n_tokens": 8, "prompt_tokens": 20, "ms": 10},
            "public": {"load": None, "visible": True, "visible_cases": [True], "visible_errors": [None],
                       "extra": [signature], "status": {"visible": "ok", "extra": "ok"}}}


def contract():
    value = {"id": "fixture/dev", "bench": "humanevalplus", "text": "Return x unchanged. f(x).",
             "entry": "f", "header": "", "public_inputs": [LinuxSandbox.input_value([2])], "public_only": True}
    return {**value, "sha256": sha(value)}


class DataEngine(SimulatedEngine):
    def __init__(self):
        self.calls = []
        self.decoders = []

    def generate(self, family, prompt, max_tokens, seed, decoder=None):
        self.calls.append((family, prompt, max_tokens, seed))
        self.decoders.append(decoder)
        response = super().generate(family, prompt, max_tokens, seed, decoder=decoder)
        if max_tokens == 512:
            response["text"] = canonical({"cases": [{"args": [2], "expected": 2}], "properties": [
                {"left": [2], "right": [2], "relation": "equal", "domain": "integer public example",
                 "uncertain": False}]})
        return response


class DataSandbox(SimulatedSandbox):
    def __init__(self):
        self.calls = []

    def run(self, candidate, contract, encoded_inputs):
        from essaim.sandbox import dec
        self.calls.append((candidate["family"], len(encoded_inputs)))
        # A data-only simulator: no parsing or execution of candidate code.
        return {"cells": [{"ok": True, "value": dec(s)[0], "error": None} for s in encoded_inputs],
                "seconds": 0, "status": "simulated", "ms": 0}


class VerifProtocolTests(unittest.TestCase):
    def test_judgment_rejects_invalid_scores(self):
        for score in ("9" * 310, "-" + "9" * 310, "NaN", "Infinity", "-Infinity", "1e310", "-1e310",
                      "true", "false", "-1", "2", "-0.01", "1.01", '"0.5"', "null"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_judgment('{"score":' + score + ',"reason":"synthetic"}')

    def test_judgment_preserves_valid_scores(self):
        for score in (0, 1, 0.0, 1.0, 0.5, -0.0):
            with self.subTest(score=score):
                judgment = {"score": score, "reason": "synthetic verdict"}
                result = parse_judgment(canonical(judgment))
                self.assertEqual(result, judgment)
                self.assertIs(type(result["score"]), type(score))

    def test_finite_schema_and_properties(self):
        parse_proposal('{"cases":[{"args":[1],"expected":1}],"properties":[]}')
        invalid = (
            '{"cases":[],"properties":[],"code":"print(1)"}',
            '{"cases":[],"cases":[],"properties":[]}',
            '{"cases":[{"args":[NaN],"expected":1}],"properties":[]}',
            '{"cases":[{"args":"eval(x)","expected":1}],"properties":[]}',
            '{"cases":[],"properties":[{"left":[],"right":[],"relation":"eval",'
            '"domain":"any","uncertain":false}]}',
        )
        for value in invalid:
            with self.assertRaises(ValueError):
                parse_proposal(value)
        self.assertTrue(relation([2, 1, 1], [1, 2, 1], "permutation"))
        self.assertFalse(relation([2, 1], [1, 1], "permutation"))
        self.assertFalse(relation(1, 1.0, "equal"))
        self.assertTrue(relation("ab", "cd", "length_equal"))
        self.assertFalse(relation(1, 2, "length_equal"))
        with self.assertRaises(ValueError):
            parse_judgment('{"score":true,"reason":"bad"}')

    def test_support_excludes_author_and_candidate_families(self):
        candidates = [candidate(f) for f in ("Qwen", "Gemma", "Granite", "SmolLM")]
        candidates += [candidate("Granite", 1)]
        proposals = [{"author": "Qwen", "kind": "cases", "item": {"expected": 2}, "indices": [0]}]
        matrix = [[{"ok": True, "value": 2}] for _ in candidates]
        scores, _ = supported_scores(candidates, proposals, matrix, 2, "cases")
        self.assertIsNone(scores[0])
        self.assertEqual(scores[1], 1)
        matrix[3] = [{"ok": False, "error": "Timeout"}]
        scores, _ = supported_scores(candidates, proposals, matrix, 2, "cases")
        self.assertIsNone(scores[1])  # Two Granite samples are still one family.
        self.assertEqual(supported_scores(candidates, proposals, matrix, 1, "cases")[0][1], 1)
        matrix[2] = matrix[4] = [{"ok": False, "error": "Timeout"}]
        self.assertIsNone(supported_scores(candidates, proposals, matrix, 1, "cases")[0][1])

    def test_properties_uncertain_and_differential_family_balance(self):
        candidates = [candidate(f) for f in ("Qwen", "Gemma", "Granite", "SmolLM")]
        item = {"relation": "equal", "uncertain": True}
        props = [{"author": "Qwen", "kind": "properties", "item": item, "indices": [0, 1]}]
        matrix = [[{"ok": True, "value": 2}, {"ok": True, "value": 2}] for _ in candidates]
        self.assertTrue(all(v is None for v in supported_scores(candidates, props, matrix, 1, "properties")[0]))
        item["uncertain"] = False
        self.assertEqual(supported_scores(candidates, props, matrix, 2, "properties")[0][1], 1)
        matrix[2][0]["value"] = 3
        self.assertAlmostEqual(differential_scores(candidates, matrix, [0])[0], 2 / 3)
        candidates.append(candidate("Granite", 1))
        matrix.append([{ "ok": True, "value": 2}])
        self.assertAlmostEqual(differential_scores(candidates, matrix, [0])[0], 2 / 3)
        matrix[2][0] = {"ok": False, "error": "Timeout"}
        self.assertAlmostEqual(differential_scores(candidates, matrix, [0])[0], 2 / 3)
        matrix[1][0] = matrix[3][0] = {"ok": False, "error": "Timeout"}
        self.assertEqual(differential_scores(candidates, matrix, [0])[0], 0)
        self.assertIsNone(differential_scores([candidates[0]], [matrix[0]], [0])[0])

    def test_shortlist_blind_judge_and_combinations(self):
        candidates = [candidate("Qwen"), candidate("Gemma"), candidate("Granite", signature="b" * 16),
                      candidate("SmolLM", signature="!"), candidate("Phi", signature="!")]
        self.assertEqual(shortlist(candidates), [0, 2, 3])
        for weights in WEIGHTS:
            self.assertIsNone(combine((None, None, None), weights))
        self.assertEqual(combine((0.3, None, 0.9), (1, 1, 1)), 0.6)
        self.assertEqual(select([None, None], [1, 0]), 1)
        self.assertEqual(select([0.5, 0.5], [1, 0]), 1)
        prompt = judge_prompt(contract(), candidates[0], [],
                              {"encoded": [], "public_indices": [], "mutation_indices": [],
                               "proposals": [], "mutation_domain": "uncertain"})
        for identity in ("Qwen", "Gemma", "Granite", "SmolLM", "model", "family", "hidden"):
            self.assertNotIn(identity, prompt)
        self.assertIn(candidates[0]["code"].replace("\n", "\\n"), prompt)

    def test_cost_and_mutations_are_deterministic(self):
        self.assertEqual(cost(64)["decode_cap"], 557056)
        self.assertEqual(cost(270)["decode_cap"], 2350080)
        small, large = mutation_inputs(contract(), 16), mutation_inputs(contract(), 64)
        self.assertEqual(small, large[:len(small)])
        self.assertEqual(large, mutation_inputs(contract(), 64))
        self.assertEqual(len(large), len(set(large)))
        self.assertNotIn(contract()["public_inputs"][0], large)

    def test_public_contract_strips_solution_and_label_fields(self):
        item = {"id": "HumanEval/dev", "entry_point": "f", "prompt": 'def f(x):\n    """Return x."""\n'
                '    return "SOLUTION"\n', "reference": "SECRET", "test": "HIDDEN", "label": True}
        public = public_contract("humanevalplus", item)
        text = canonical(public)
        self.assertNotIn("SOLUTION", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("HIDDEN", text)
        self.assertNotIn("label", text)
        item["reference"] = "OTHER"
        self.assertEqual(public_contract("humanevalplus", item), public)
        mbpp = {"id": "Mbpp/dev", "prompt": "Return x.", "test_list": ["assert f(2) == 2"],
                "test_imports": [], "reference": "SECRET", "test": "HIDDEN"}
        self.assertEqual(public_contract("mbppplus", mbpp)["entry"], "f")
        mbpp["test_list"] = ["assert f(2)", "assert not f(3)"]
        self.assertEqual(public_contract("mbppplus", mbpp)["entry"], "f")

    def test_prefix_reads_no_test_suffix(self):
        data = b'{"id":"dev","prompt":"public","reference":"secret"}\nDO NOT READ TEST'
        stream = io.BytesIO(data)
        # Keep BytesIO open so the exact consumed offset can be asserted.
        from contextlib import nullcontext
        path = Path("humanevalplus_all_s0_v2.jsonl")
        with patch.object(Path, "open", return_value=nullcontext(stream)):
            rows, _ = read_dev_prefix(path, 1)
        self.assertEqual(rows, [{"id": "dev", "prompt": "public"}])
        self.assertEqual(stream.tell(), data.index(b"\n") + 1)


class VerifLedgerTests(unittest.TestCase):
    def test_oversized_judgment_is_format_error_and_ledger_continues(self):
        engine = DataEngine()
        text = '{"score":' + "9" * 310 + ',"reason":"synthetic"}'
        raw = {"text": text, "truncated": False, "finish": "stop", "prompt_tokens": 1,
               "completion_tokens": 200}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            ledger = Ledger(path, "bind")
            with patch.object(engine, "generate", return_value=raw) as generate:
                result = proposed(ledger, "invalid", "Qwen", "public", 256, 20261010, engine, parse_judgment)
                generate.assert_called_once()
            self.assertEqual(result["status"], "format_error")
            self.assertEqual(result["error"], "ValueError")
            self.assertIsNone(result["parsed"])
            self.assertEqual(result["raw"], raw)
            valid = {**raw, "text": '{"score":0.5,"reason":"synthetic valid verdict"}'}
            with patch.object(engine, "generate", return_value=valid):
                next_result = proposed(ledger, "valid", "Qwen", "public", 256, 20261010, engine, parse_judgment)
            self.assertEqual(next_result["status"], "valid")
            self.assertEqual(next_result["parsed"]["score"], 0.5)
            resumed = Ledger(path, "bind")
            self.assertFalse(resumed.pending)
            self.assertTrue(all(event["status"] == "ok" for event in resumed.done.values()))
            with patch.object(engine, "generate", side_effect=AssertionError("no retry")):
                self.assertEqual(proposed(resumed, "invalid", "Qwen", "public", 256, 20261010,
                                          engine, parse_judgment), result)
                self.assertEqual(proposed(resumed, "valid", "Qwen", "public", 256, 20261010,
                                          engine, parse_judgment), next_result)

    def test_resume_no_retry_input_and_chain_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            journal = Ledger(path, "binding")
            self.assertEqual(journal.operation("a", {"x": 1}, lambda: {"ok": 2}), {"ok": 2})
            journal = Ledger(path, "binding")
            self.assertEqual(journal.operation("a", {"x": 1}, lambda: self.fail("recalled")), {"ok": 2})
            with self.assertRaises(ValueError):
                journal.operation("a", {"x": 2}, lambda: None)
            journal.append("reserve", "b", sha({}))
            with self.assertRaises(ValueError):
                Ledger(path, "binding")
            with self.assertRaises(ValueError):
                Ledger(path, "wrong")

    def test_confirmation_reserved_before_read_once_per_variant(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            freeze, source = root / "dev_freeze.json", root / "synthetic_confirmation.json"
            bindings = {k: "a" * 64 for k in ("code_sha256", "plan_sha256", "data_sha256")}
            write_new(freeze, {"split": "dev", "variant": "V03", "bindings": bindings,
                               "configuration": {"support": 2}})
            with self.assertRaises(ValueError):
                confirmation_open(root, "V03", freeze, "b" * 64, bindings, source)
            self.assertFalse((root / "VERIF-1-V03.opened.json").exists())
            # A failing read still consumes the one confirmation, before the open occurs.
            with self.assertRaises(FileNotFoundError):
                confirmation_open(root, "V03", freeze, file_sha(freeze), bindings, source)
            write_new(source, {"synthetic": True})
            with self.assertRaises(FileExistsError):
                confirmation_open(root, "V03", freeze, file_sha(freeze), bindings, source)
            self.assertTrue((root / "VERIF-1-V03.opened.json").exists())
            second = root / "dev_freeze_second.json"
            write_new(second, {"split": "dev", "variant": "V04", "bindings": bindings,
                               "configuration": {"mutations": 64}})
            with confirmation_open(root, "V04", second, file_sha(second), bindings, source) as stream:
                self.assertEqual(json.load(stream), {"synthetic": True})
            with self.assertRaises(FileExistsError):
                confirmation_open(root, "V04", second, file_sha(second), bindings, source)

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lock"
            with exclusive(path):
                with self.assertRaises(FileExistsError):
                    with exclusive(path):
                        self.fail("double lock")
            self.assertFalse(path.exists())

    def test_format_truncation_terminal_no_retry(self):
        engine = DataEngine()
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d) / "ledger", "bind")
            engine.generate = lambda *args, **kwargs: {"text": "{bad", "truncated": True, "finish": "length",
                                             "prompt_tokens": 1, "completion_tokens": 512}
            value = proposed(ledger, "x", "Qwen", "public", 512, 1, engine, parse_proposal)
            self.assertEqual(value["status"], "truncated")
            engine.generate = lambda *args, **kwargs: self.fail("no retry")
            self.assertEqual(proposed(ledger, "x", "Qwen", "public", 512, 1, engine, parse_proposal), value)


class VerifRuntimeTests(unittest.TestCase):
    def test_end_to_end_offline_matrix_judges_resume(self):
        fixed = [candidate(FAMILIES[m], s, ("abcdef0123456789"[s] * 16)) for m in MODELS for s in range(5)]
        for i, value in enumerate(fixed):
            value["public"]["extra"] = ["b" * 16 if i in (4, 24) else "c" * 16 if i in (15, 29) else "a" * 16]
        self.assertEqual(shortlist(fixed), [0, 4, 15])
        bundle = {"questions": [{"contract": contract(), "candidates": fixed}]}
        plan = {"config": {}}
        engine, sandbox = DataEngine(), DataSandbox()
        with tempfile.TemporaryDirectory() as d, patch.object(socket.socket, "connect", side_effect=AssertionError):
            scores, coverage, accounting = run_pipeline(bundle, plan, Path(d), engine, sandbox, simulation=True)
            self.assertEqual(len(engine.calls), 20)
            self.assertEqual(len(sandbox.calls), 35)
            self.assertEqual(coverage["authors_expected"], 14)
            self.assertLessEqual(coverage["judges_expected"], 6)
            self.assertEqual(accounting["candidate_original_cost"]["n_tokens"], 280)
            self.assertEqual(len(next(iter(scores.values()))["scores"]), 23)
            for family, prompt, budget, seed in engine.calls:
                if budget == 512:
                    self.assertNotIn("return x", prompt)
                    self.assertNotIn("Qwen", prompt)
                    self.assertNotIn("original_public_results", prompt)
                else:
                    self.assertNotIn(family, prompt)
            count = len(engine.calls)
            resumed = run_pipeline(bundle, plan, Path(d), engine, sandbox, simulation=True)
            self.assertEqual(resumed, (scores, coverage, accounting))
            self.assertEqual(len(engine.calls), count)
            self.assertEqual(len(sandbox.calls), 35)
            artifact = Path(d) / "verif1_pilot_costs.json"
            artifact.write_text('{}')
            with self.assertRaises(ValueError):
                run_pipeline(bundle, plan, Path(d), engine, sandbox, simulation=True)

    def test_concrete_gpu_adapter_usage_and_request(self):
        calls = []
        raw = {"choices": [{"message": {"content": '{"cases":[],"properties":[]}'}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 42, "completion_tokens": 12}}
        class FakeServer:
            version = "version: build 11505, commit ff5888f99"

            def __init__(self, gguf, parallel, ctx_per_slot):
                self.log_path = Path(gguf).with_suffix(".log")
                self.log_path.write_text("offloaded 29/29 layers to GPU")
                self._log = io.StringIO()
                self.http = self

            def post(self, endpoint, json):
                calls.append((endpoint, json))
                return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: raw)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.gguf"
            path.write_bytes(b"fixture")
            catalogue = {"Qwen": {"gguf": path.name, "weights": {"size": 7, "sha256": file_sha(path)}}}
            with patch.dict(sys.modules, {"run_solo": types.SimpleNamespace(ChatServer=FakeServer)}):
                engine = GGUFEngine(catalogue, Path(d))
                result = engine.generate("Qwen", "public only", 512, 20261010, decoder=AUTHOR_DECODER)
                self.assertEqual(result["prompt_tokens"], 42)
                self.assertEqual(result["completion_tokens"], 12)
                self.assertEqual(calls[0][0], "/v1/chat/completions")
                self.assertEqual(calls[0][1]["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(calls[0][1]["max_tokens"], 512)
                self.assertEqual(calls[0][1]["temperature"], .7)
                self.assertEqual(calls[0][1]["top_p"], .95)
                self.assertEqual(result["request_sha256"], sha(calls[0][1]))
                self.assertEqual(result["response_sha256"], sha(raw))
                raw["usage"] = {}
                malformed = engine.generate("Qwen", "public only", 512, 20261010)
                self.assertIsNone(malformed["completion_tokens"])
                self.assertEqual(malformed["response"], raw)
                engine.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "real candidates run only on Linux")
    def test_linux_sandbox_smoke(self):
        sandbox = LinuxSandbox()
        self.assertIn("landlock", sandbox.smoke()["isolation"])
        result = sandbox.run({"code": "def f(x):\n    return (x,)"}, {"entry": "f", "header": ""},
                             [sandbox.input_value([2])])
        self.assertTrue(result["cells"][0]["coverage_loss"])
        self.assertEqual(result["cells"][0]["error"], "non_plain_JSON_output")

    def test_historical_colab_without_landlock_is_recorded(self):
        from essaim import sandbox
        with patch("sys.platform", "linux"), patch.object(sandbox, "isolation", return_value={
                "landlock": "unavailable", "namespace": "unavailable", "historical": True}):
            runtime = LinuxSandbox()
            self.assertEqual(runtime.identity["landlock"], "unavailable")
            runtime.sandbox = types.SimpleNamespace(_candidate=lambda *a: (
                None, {0: {"digest": "opaque large output"}, 1: {"err": "OutputTimeout"}}, "ok", 0))
            result = runtime.run(candidate(), {"entry": "f", "header": ""}, ["one", "two"])
            self.assertTrue(all(c["coverage_loss"] for c in result["cells"]))


if __name__ == "__main__":
    unittest.main()
