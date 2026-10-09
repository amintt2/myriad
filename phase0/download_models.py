"""Download the PC models from Hugging Face (official repos, safetensors only).

Xet's adaptive concurrency can collapse to a single connection on some home links;
plain HTTP with several parallel files is usually faster there:
    HF_HUB_DISABLE_XET=1 uv run python download_models.py
"""
import sys
import time

from huggingface_hub import snapshot_download

MODELS = sys.argv[1:] or ["ibm-granite/granite-3.3-2b-instruct", "HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]

for m in MODELS:
    t0 = time.time()
    p = snapshot_download(m, allow_patterns=PATTERNS, max_workers=8)
    print(f"{m} -> {p} ({time.time() - t0:.0f}s)", flush=True)
