"""The simulated peers of the scalability benchmark: answer model and compute-time model."""
from __future__ import annotations

import math
import statistics

import pytest

from bench.sim import (AnswerModel, ComputeModel, MEASURED_FAMILIES, SimPeer, families, prior_entries, qid_of,
                       question)
from myriad.fusion import extract_answer


def test_accuracy_and_collision_rate():
    am = AnswerModel(collision=0.09, seed=1)
    p, n = 0.6, 40000
    right = sum(am.answer("a", p, q) == am.gold(q) for q in range(n))
    assert right / n == pytest.approx(p, abs=0.01)
    # Two wrong peers on the same question give the same wrong answer with probability c.
    same = both_wrong = 0
    for q in range(n):
        a, b = am.answer("a", p, q), am.answer("b", p, q)
        if a != am.gold(q) and b != am.gold(q):
            both_wrong += 1
            same += a == b
    assert both_wrong > 4000
    assert same / both_wrong == pytest.approx(0.09, abs=0.015)


@pytest.mark.parametrize("c", [0.0, 0.25, 1.0])
def test_collision_extremes(c):
    am = AnswerModel(collision=c, seed=2)
    pairs = [(am.answer("a", 0.0, q), am.answer("b", 0.0, q)) for q in range(4000)]
    rate = sum(a == b for a, b in pairs) / len(pairs)
    assert rate == pytest.approx(c, abs=0.03)
    assert all(a != am.gold(q) for q, (a, _) in enumerate(pairs))


def test_answers_are_deterministic_and_independent_per_node():
    am = AnswerModel(seed=3)
    assert [am.answer("x", 0.5, q) for q in range(50)] == [am.answer("x", 0.5, q) for q in range(50)]
    assert [am.answer("x", 0.5, q) for q in range(200)] != [am.answer("y", 0.5, q) for q in range(200)]
    assert [AnswerModel(seed=3).gold(q) for q in range(20)] == [am.gold(q) for q in range(20)]
    assert [AnswerModel(seed=4).gold(q) for q in range(20)] != [am.gold(q) for q in range(20)]
    with pytest.raises(ValueError):
        AnswerModel(collision=1.5)


def test_compute_model():
    cm = ComputeModel(median_s=4.0, sigma=0.36, seed=1)
    xs = [cm.seconds("n", q) for q in range(20000)]
    assert statistics.median(xs) == pytest.approx(4.0, rel=0.03)
    assert statistics.pstdev([math.log(x) for x in xs]) == pytest.approx(0.36, rel=0.05)
    fast = ComputeModel(median_s=4.0, sigma=0.36, seed=1, scale=0.05)
    assert fast.seconds("n", 5) == pytest.approx(0.05 * cm.seconds("n", 5))
    het = ComputeModel(node_sigma=0.5, seed=1)
    assert het.node_speed("a") != het.node_speed("b") and ComputeModel().node_speed("a") == 1.0


def test_sim_peer_reply_is_extracted_by_the_gateway():
    am, cm = AnswerModel(seed=5), ComputeModel(scale=0.01, seed=5)
    peer = SimPeer("n1", MEASURED_FAMILIES[0], am, cm)
    msgs = [{"role": "user", "content": question(123)}]
    assert qid_of(msgs) == 123
    assert extract_answer(peer.reply(msgs), "math") == am.answer("n1", MEASURED_FAMILIES[0].accuracy, 123)
    assert 0 < peer.delay(msgs) < 1.0
    peer.hung = True
    assert peer.delay(msgs) >= 3600


def test_families():
    f = families(7)
    assert [x.family for x in f[:4]] == ["smollm", "gemma", "qwen", "granite"] and len({x.family for x in f}) == 7
    assert all(x.measured for x in f[:4]) and not any(x.measured for x in f[4:])
    assert max(x.accuracy for x in f[4:]) < min(x.accuracy for x in f[:4])
    pri = prior_entries(f)
    assert set(pri) == {"synth5-2b", "synth6-2b", "synth7-2b"}
