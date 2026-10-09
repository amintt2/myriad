"""Helpers of the end-to-end benchmark (E6) that need no GPU: node specs, question files, grader."""
from __future__ import annotations

import json

import pytest

from bench.bench_e2e import load_questions, parse_node, phase0_answers, summarize


def test_parse_node():
    assert parse_node("Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf") == ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf", None)
    assert parse_node("a/b:c.gguf=C:/m/c.gguf") == ("a/b", "c.gguf", "C:/m/c.gguf")
    with pytest.raises(SystemExit):
        parse_node("no-colon.gguf")


def test_question_files(tmp_path):
    rows = [{"id": "x1", "question": "2+2?", "answer": 4}, {"question": "3+3?", "answer": "6"}]
    (tmp_path / "q.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    for name in ("q.json", "q.jsonl"):
        items, src = load_questions(str(tmp_path / name), tmp_path)
        assert [i["id"] for i in items] == ["x1", "1"] and items[0]["answer"] == "4" and src["sha256"]
    with pytest.raises(SystemExit):  # the phase-0 cache is absent from this directory
        load_questions("gsm8k-test", tmp_path)


def test_phase0_grader_is_importable_next_to_the_app_package():
    g = phase0_answers()
    assert g.extract("gsm8k", "So 3 + 4 = 7.\n\n**The answer is 1,234.**", {}, ended=True) == "1234"
    assert g.norm_math("12.50") == g.norm_math("12.5")
    import myriad  # the app's package is untouched
    assert hasattr(myriad, "__version__")


def test_summarize():
    peers = [{"model": "m", "status": "ok", "correct": True, "compute_ms": 10.0, "latency_ms": 20.0, "error": None}]
    recs = [{"ok": True, "correct": True, "app_correct": True, "latency_s": 1.0, "peers_answered": 1,
             "early_stop": False, "peers": peers},
            {"ok": False, "correct": False, "latency_s": 9.0}]
    s = summarize(recs, 2.0)
    assert s["accuracy"] == 0.5 and s["ok"] == 1 and s["per_model"]["m"]["accuracy_when_answered"] == 1.0
    assert s["throughput_rps"] == 1.0
