import itertools
import math
import random

import pytest

from myriad.fusion import (W_MAX, W_MIN, count_options, detect_task_hint, extract_answer, medoid, reliability_weight,
                           stop_certificate, weighted_vote)


# ---------- extraction ----------
@pytest.mark.parametrize("text,expected", [
    ("Step 1... The answer is 42.", "42"),
    ("**The answer is 16.**", "16"),
    ("The answer is $1,234.", "1234"),
    ("So the total is 3.50 dollars. The answer is 3.5", "3.5"),
    ("The answer is -7.\nDone", "-7"),
    ("We get \\boxed{128}", "128"),
    ("Final answer: 12", "12"),
    ("La réponse est 9.", "9"),
    ("He has 3 apples, then 5, so 8 in total", "8"),  # last number as a fallback
    ("<think>The answer is 1.</think>The answer is 2.", "2"),  # thinking is ignored
    ("The answer is 10.00", "10"),
    ("no number here", None),
    ("", None),
])
def test_extract_math(text, expected):
    assert extract_answer(text, "math") == expected


@pytest.mark.parametrize("text,expected", [
    ("Reasoning... The answer is B.", "B"),
    ("The answer is (C)", "C"),
    ("Answer: D", "D"),
    ("\\boxed{A}", "A"),
    ("Let me think.\nB", "B"),
    ("(D)", "D"),
    ("The answer is a bit unclear", None),
    ("I cannot tell", None),
])
def test_extract_mc(text, expected):
    assert extract_answer(text, "mc") == expected


def test_extract_free_is_none():
    assert extract_answer("The answer is 42.", "free") is None
    assert extract_answer("The answer is 42.", None) is None


def test_huge_number_is_rejected_not_crashing():
    assert extract_answer("The answer is " + "9" * 500, "math") is None


def test_detect_task_hint():
    mc = [{"role": "user", "content": "Which?\nA) red\nB) blue\nC) green"}]
    math_q = [{"role": "user", "content": "Tom has 3 apples and buys 4 more. How many apples does he have?"}]
    free = [{"role": "user", "content": "Écris un poème sur la mer."}]
    assert detect_task_hint(mc) == "mc"
    assert count_options(mc) == 3
    assert detect_task_hint(math_q) == "math"
    assert detect_task_hint(free) == "free"


# ---------- weights ----------
def test_weights_are_clipped_k_class_log_odds():
    assert reliability_weight(0.75) == pytest.approx(math.log(3))  # c = 1: binary log-odds
    assert reliability_weight(0.4) == W_MIN == 0.0  # never negative
    assert reliability_weight(0.999999) == W_MAX
    assert reliability_weight(float("nan")) == 0.0
    # w = logit(p) - ln(c): c = 1/(K-1) for K options, 0.05 for open numeric answers
    assert reliability_weight(0.5, 1 / 3) == pytest.approx(math.log(3))
    assert reliability_weight(0.75, 0.05) == pytest.approx(math.log(3) + math.log(20))
    assert reliability_weight(0.865, 0.05) > reliability_weight(0.535, 0.05) > 0


def test_collision_probabilities_per_task():
    from myriad.priors import collision_prob
    assert collision_prob("math") == 0.05
    assert collision_prob("mc", 5) == pytest.approx(0.25)
    assert collision_prob("free") == 1.0


def test_open_answer_weights_let_agreement_beat_one_strong_peer():
    """With the phase-0 priors, binary weights hand the decision to SmolLM3 alone; K-class ones do not."""
    from myriad.priors import collision_prob, prior_accuracy
    models = ["Qwen/Qwen3-1.7B-GGUF", "ibm-granite/granite-3.3-2b-instruct-GGUF", "HuggingFaceTB/SmolLM3-3B-GGUF",
              "ggml-org/gemma-4-E2B-it-GGUF"]
    answers = ["7", "7", "9", "7"]
    binary = [reliability_weight(prior_accuracy(m)) for m in models]
    open_ = [reliability_weight(prior_accuracy(m), collision_prob("math")) for m in models]
    assert weighted_vote(answers, binary).answer == "9"
    assert weighted_vote(answers, open_).answer == "7"


# ---------- vote ----------
def test_weighted_vote_basic():
    v = weighted_vote(["1", "2", "2"], [3.0, 1.0, 1.0])
    assert v.answer == "1" and v.representative == 0
    v = weighted_vote(["1", "2", "2"], [1.5, 1.0, 1.0])
    assert v.answer == "2" and v.scores == {"1": 1.5, "2": 2.0}


def test_weighted_vote_abstentions_and_empty():
    assert weighted_vote([None, None], [1.0, 1.0]).answer is None
    assert weighted_vote([None, "3"], [5.0, 0.1]).answer == "3"


def test_tie_broken_by_logprob():
    v = weighted_vote(["a", "b"], [1.0, 1.0], [-0.5, -0.1])
    assert v.answer == "b" and v.tie_broken
    v = weighted_vote(["a", "b"], [1.0, 1.0], [-0.1, None])
    assert v.answer == "a"
    v = weighted_vote(["b", "a"], [1.0, 1.0], [None, None])  # deterministic fallback
    assert v.answer == "a"


def test_vote_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_vote(["a"], [-1.0])
    with pytest.raises(ValueError):
        weighted_vote(["a"], [1.0, 2.0])


# ---------- certificate ----------
def test_certificate_examples():
    assert stop_certificate({"42": 3.0}, 2.0) == "42"
    assert stop_certificate({"42": 3.0}, 3.0) is None  # the rest could tie
    assert stop_certificate({"42": 3.0, "7": 1.0}, 1.9) == "42"
    assert stop_certificate({"42": 3.0, "7": 1.0}, 2.0) is None
    assert stop_certificate({}, 0.0) is None
    assert stop_certificate({"a": 1.0, "b": 1.0}, 0.0) is None  # tie: not certain
    assert stop_certificate({"a": 1.0}, 0.0) == "a"


def _check_never_wrong(rng: random.Random, n: int, labels: list) -> int:
    answers = [rng.choice(labels) for _ in range(n)]
    weights = [rng.choice([0.05, 0.5, 1.0, 1.0, 2.0, rng.uniform(0.05, 10.0)]) for _ in range(n)]
    conf = [rng.choice([None, rng.uniform(-3, 0)]) for _ in range(n)]
    full = weighted_vote(answers, weights, conf).answer
    order = list(range(n))
    rng.shuffle(order)
    certified = 0
    for m in range(n + 1):
        seen, rest = order[:m], order[m:]
        partial: dict = {}
        for i in seen:
            if answers[i] is not None:
                partial[answers[i]] = partial.get(answers[i], 0.0) + weights[i]
        cert = stop_certificate(partial, sum(weights[i] for i in rest))
        if cert is not None:
            assert cert == full, (answers, weights, order, m)
            certified += 1
    return certified


def test_certificate_never_wrong_random():
    rng = random.Random(1234)
    hits = 0
    for _ in range(20000):
        hits += _check_never_wrong(rng, rng.randint(1, 7), ["a", "b", "c", None])
    assert hits > 1000  # the certificate does fire often


def test_certificate_never_wrong_exhaustive_small():
    """Every answer assignment of 4 peers over {a, b, None}, several weight vectors, every order."""
    weight_sets = [(1, 1, 1, 1), (3, 1, 1, 1), (2, 2, 1, 0.05), (4.7, 6.4, 5.6, 4.6), (1, 2, 3, 5)]
    for ws in weight_sets:
        for answers in itertools.product(["a", "b", None], repeat=4):
            full = weighted_vote(list(answers), list(ws)).answer
            for order in itertools.permutations(range(4)):
                for m in range(5):
                    partial: dict = {}
                    for i in order[:m]:
                        if answers[i] is not None:
                            partial[answers[i]] = partial.get(answers[i], 0.0) + ws[i]
                    cert = stop_certificate(partial, float(sum(ws[i] for i in order[m:])))
                    if cert is not None:
                        assert cert == full


def test_certificate_complete_when_all_answered():
    """With every answer in, the certificate holds exactly when the winner is unique."""
    rng = random.Random(7)
    for _ in range(2000):
        n = rng.randint(1, 6)
        answers = [rng.choice(["a", "b", "c"]) for _ in range(n)]
        weights = [rng.choice([1.0, 2.0, 0.5]) for _ in range(n)]
        v = weighted_vote(answers, weights)
        cert = stop_certificate(v.scores, 0.0)
        if v.tie_broken:
            assert cert is None
        else:
            assert cert == v.answer


# ---------- medoid ----------
def test_medoid_picks_central_text():
    texts = ["the cat sat on the mat", "a cat sat on the mat", "quantum chromodynamics of gluons", None]
    assert medoid(texts, [1, 1, 1, 1]) in (0, 1)
    assert medoid([None, "", "  "], [1, 1, 1]) is None
    assert medoid(["only one"], [1]) == 0


def test_medoid_prefers_the_reliable_of_two_similar_texts():
    texts = ["Paris is the capital of France.", "Paris is a capital."]
    assert medoid(texts, [2.0, 0.05]) == 0
    assert medoid(texts, [0.05, 2.0]) == 1


def test_medoid_weights_matter():
    texts = ["alpha beta", "alpha gamma", "delta gamma"]
    # text 1 shares a word with both others; it is the medoid with equal weights
    assert medoid(texts, [1, 1, 1]) == 1
