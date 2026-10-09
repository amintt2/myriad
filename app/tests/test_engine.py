import subprocess
import sys

import pytest

from myriad.engine import EngineError, LlamaServerEngine, parse_chat_completion


def test_dead_llama_server_is_not_ready(tmp_path):
    eng = LlamaServerEngine(str(tmp_path / "m.gguf"), "x/y")
    eng.proc = subprocess.Popen([sys.executable, "-c", "pass"])
    eng.state = "prêt"
    eng.proc.wait()
    assert eng.state == "erreur"  # the process died after a successful start: no longer available


def test_parse_chat_completion():
    j = {"choices": [{"message": {"content": "The answer is 4."}, "finish_reason": "stop",
                      "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.5}, {"logprob": None}]}}],
         "usage": {"completion_tokens": 7}}
    g = parse_chat_completion(j, 12.0)
    assert g.text == "The answer is 4." and g.completion_tokens == 7 and g.mean_logprob == pytest.approx(-1.0)
    g = parse_chat_completion({"choices": [{"message": {"content": None}}]}, 1.0)
    assert g.text == "" and g.mean_logprob is None and g.completion_tokens == 0
    with pytest.raises(EngineError):
        parse_chat_completion({"choices": []}, 1.0)
