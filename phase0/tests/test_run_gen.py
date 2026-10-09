"""Coordinator of experiment 2 (run_gen.py): malformed peer answers never enter a quorum."""
import asyncio
import unittest

import httpx

import run_gen

GOOD_PROPOSE = {"compute_ms": 12.0, "queue_ms": 0.0, "result": {"text": "The answer is 4.", "eos": True,
                                                                  "stop": "eos", "mean_logp": -0.3}}


class TestCheckResponse(unittest.TestCase):
    def test_propose(self):
        run_gen.check_response("/propose", {}, GOOD_PROPOSE)
        for bad in ({"compute_ms": 0, "result": {}},  # audit 2026-10-09 (phase0, 12): accepted, then KeyError
                    {"compute_ms": 0, "result": {"text": "x", "eos": True, "mean_logp": float("nan")}},
                    {"compute_ms": float("inf"), "result": GOOD_PROPOSE["result"]},
                    {"compute_ms": -1, "result": GOOD_PROPOSE["result"]},
                    {"compute_ms": 1, "result": {"text": 3, "eos": True, "mean_logp": 0.0}},
                    {"compute_ms": 1, "result": {"text": "x", "eos": "yes", "mean_logp": 0.0}},
                    {"result": GOOD_PROPOSE["result"]}, [], None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_gen.check_response("/propose", {}, bad)

    def test_score(self):
        body = {"candidates": ["one two", "three"]}
        good = {"compute_ms": 3, "result": [{"word_logp": [-1.0, -2.0]}, {"word_logp": [-0.5], "special_token": False}]}
        run_gen.check_response("/score", body, good)
        for res in ([{"word_logp": [-1.0]}, {"word_logp": [-0.5]}],  # one word missing
                    [{"word_logp": [-1.0, -2.0]}],  # one candidate missing
                    [{"word_logp": [-1.0, float("-inf")]}, {"word_logp": [-0.5]}],
                    [{"word_logp": [-1.0, -2.0]}, {"word_logp": [-0.5], "boundary_merged": "no"}],
                    {"word_logp": []}):
            with self.subTest(res=res), self.assertRaises(ValueError):
                run_gen.check_response("/score", body, {"compute_ms": 3, "result": res})


class TestQuorum(unittest.TestCase):
    def test_malformed_peer_is_an_error_not_an_answer(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 8101:  # HTTP 200 with an empty result
                return httpx.Response(200, json={"compute_ms": 0, "result": {}})
            return httpx.Response(200, json=GOOD_PROPOSE)

        async def go():
            peers = run_gen.Peers(["http://p:8101", "http://p:8102"], None)
            peers.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            got = await peers.first_k("/propose", {"user": "q", "n": 8}, 1)
            errors = dict(peers.last_errors)
            sol = await run_gen.solve_solo(peers, "q", 8)
            await peers.client.aclose()
            return got, errors, sol

        got, errors, sol = asyncio.run(go())
        self.assertEqual(list(got), [1])
        self.assertIn(0, errors)
        self.assertIsNone(sol[0]["text"])  # the malformed peer is recorded as failed, the run goes on
        self.assertEqual(sol[1]["answer"], "4")


if __name__ == "__main__":
    unittest.main()
