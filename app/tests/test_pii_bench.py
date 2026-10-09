"""Local benchmark fixtures: partial span runs must survive analysis, calibration and replay."""
from __future__ import annotations

import gzip
import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
from bench.pii import analyze, registry, run_bench


@pytest.fixture
def guarantee_run(tmp_path, monkeypatch):
    docs = []
    for split in ("train", "cal", "test"):
        for label in (0, 1):
            docs.append({"id": f"{split}-{label}", "split": split, "label": label, "guarantee": True,
                         "source": "fixture", "lang": "fr", "text": "Camille" if label else "Calcul",
                         "spans": [{"start": 0, "end": 7, "type": "PERSON"}] if label else []})
    docs.append({"id": "omitted-gretel", "split": "test", "label": 1, "guarantee": False,
                 "source": "gretel", "lang": "fr", "text": "Camille",
                 "spans": [{"start": 0, "end": 7, "type": "PERSON"}]})
    runs, out = tmp_path / "runs", tmp_path / "results"
    runs.mkdir()
    for name in ("regex", "nym-small-edge"):
        with gzip.open(runs / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for d in docs:
                if name != "regex" and not d["guarantee"]:
                    continue
                tokens = [[0, len(d["text"]), 0.9 if d["label"] else 0.05, "PERSON"]] if name != "regex" else []
                f.write(json.dumps({"id": d["id"], "ms": 1.0, "tokens": tokens}) + "\n")
        (runs / f"{name}.meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(analyze, "RUNS", runs)
    monkeypatch.setattr(analyze, "OUT", out)
    monkeypatch.setattr(analyze, "load", lambda: docs)
    monkeypatch.setattr(analyze, "_DOCS", None)
    monkeypatch.setattr(analyze, "PROBES", [])
    return docs, out


def test_guarantee_span_subset_analysis_calibration_and_replay(guarantee_run, capsys):
    docs, out = guarantee_run
    analyze.main([])
    report = json.loads((out / "pii_report.json").read_text(encoding="utf-8"))
    model = report["detectors"]["nym-small-edge"]
    combined = report["detectors"]["regex+nym-small-edge"]
    assert model["n_cal_pos"] == 1 and model["n_test_pos"] == 1 and model["n_test_neg"] == 1
    assert "gretel" not in model["by_source"]
    assert combined["mask_policy"]["n_cal_pos"] == 1
    for replay in combined["scanner"].values():
        assert "gretel" not in replay and sum(replay["pos"].values()) == 1 and sum(replay["neg"].values()) == 1
    assert "nym-small-edge" in (out / "pii_report.md").read_text(encoding="utf-8")


def test_rejected_model_has_no_executable_catalogue():
    assert "rampart" not in registry.MODELS and "rampart" not in run_bench.ALL and "rampart" not in analyze.SPAN_MODELS
