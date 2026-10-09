"""Optional: a real llama-server with a small GGUF (set MYRIAD_TEST_GGUF, or the former ESSAIM_TEST_GGUF,
and LLAMA_SERVER if the binary is not on the PATH). Run with: uv run pytest -m real"""
from __future__ import annotations

import os

import pytest

from myriad.engine import LlamaServerEngine

GGUF = os.environ.get("MYRIAD_TEST_GGUF") or os.environ.get("ESSAIM_TEST_GGUF")

pytestmark = [pytest.mark.real, pytest.mark.skipif(not GGUF, reason="MYRIAD_TEST_GGUF non défini")]


async def test_llama_server_generates(tmp_path):
    eng = LlamaServerEngine(GGUF, "test/model", ctx=2048, parallel=1, log_dir=tmp_path)
    try:
        await eng.start()
        assert eng.state == "prêt"
        gen = await eng.generate([{"role": "user", "content": "What is 2 + 3? Answer with one number."}],
                                 max_tokens=48, temperature=0.0, seed=1, timeout_s=300)
        assert gen.text.strip()
        assert gen.completion_tokens > 0
        assert gen.mean_logprob is None or gen.mean_logprob <= 0
    finally:
        await eng.close()
    assert eng.proc is None
