"""Pinned candidates and datasets of the PII benchmark (bench only, never shipped).

Every model and dataset is pinned to a commit of its Hugging Face repository. Large (LFS) files carry
the SHA-256 published by the Hub; `fetch.py` checks it after download. Licences were read on the
model cards and repository metadata on 2026-10-09 (see docs/09_detection_pii.md for the table)."""
from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("MYRIAD_PII_BENCH_CACHE", Path.home() / ".cache" / "myriad-pii-bench"))

# kind: "tokcls" (BIO token classification), "gliner" (span classification), "laya" (typed decision
# head, document level), "embed" (sentence embedding + probe trained on the dev split, document level)
MODELS: dict[str, dict] = {
    "nym-small-edge": dict(
        kind="tokcls", repo="Wismut/nym-pii-multilingual-small", rev="4348999cd3c2e20c49615e9af7c6bbb45b64cd85",
        licence="mit", langs="22 (en fr de es it pt nl pl sv cs ro tr fi da el ru uk ja zh ko ar hi)",
        onnx="edge-int8/model_int8.onnx", tokenizer="edge-int8/tokenizer.json", config="edge-int8/config.json",
        lfs={"edge-int8/model_int8.onnx": "e9a3a8c8cd55b3bcf329de5a9307cfae5053ce93c00330c33facd022e60daa17",
             "edge-int8/tokenizer.json": "c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de"},
        max_len=512),
    "nym-small-int8": dict(
        kind="tokcls", repo="Wismut/nym-pii-multilingual-small", rev="4348999cd3c2e20c49615e9af7c6bbb45b64cd85",
        licence="mit", langs="22",
        onnx="int8/model_int8.onnx", tokenizer="int8/tokenizer.json", config="int8/config.json",
        lfs={"int8/model_int8.onnx": "139006aea2cbd8e709d322f056232570de54661f624143be4893aaa387190286",
             "int8/tokenizer.json": "c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de"},
        max_len=512),
    "bert-small-pii": dict(
        kind="tokcls", repo="onnx-community/bert-small-pii-detection-ONNX", rev="6cb4e77c2b2c7f81e731b88cffa9b7a6fc675a4c",
        licence="apache-2.0", langs="en",
        onnx="onnx/model_int8.onnx", tokenizer="tokenizer.json", config="config.json",
        lfs={"onnx/model_int8.onnx": "40e94266f077c088d3dda3e12fe7be8faa1cae862c3e3fe84b799439c509095a"},
        max_len=512),
    "ettin-32m": dict(
        kind="tokcls", repo="rulesentry-io/ettin-32m-nemotron-pii-onnx", rev="a7564cc972723bd22ccb3c7a248aadb456adb267",
        licence="mit", langs="en",
        onnx="model.onnx", tokenizer="tokenizer.json", config="config.json",
        lfs={"model.onnx": "b28686142e62f70dbd5ee9b325b8f8ad8a1daa6089581320026e26b67eb6d713"},
        max_len=512),
    "minilm-nemotron": dict(
        kind="tokcls", repo="Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection",
        rev="e43200abcd5e739eddd31f55929f986b2b912b6b", licence="mit", langs="en",
        onnx="onnx/model.onnx", tokenizer="tokenizer.json", config="config.json",
        lfs={"onnx/model.onnx": "875afd5fc38cc49487f57d0c6d74b9d4ff8557e582470c3398276c26d5fdffb2"},
        max_len=512),
    "gliner-pii-small": dict(
        kind="gliner", repo="knowledgator/gliner-pii-small-v1.0", rev="d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e",
        licence="apache-2.0", langs="en (multilingual tag)",
        onnx="onnx/model_quint8.onnx", tokenizer="tokenizer.json", config="gliner_config.json",
        lfs={"onnx/model_quint8.onnx": "891589426ee96f2748b16439f44fad8c3f97e198e002a6637e58dee989500216"},
        max_len=384),
    "bunker-laya": dict(
        kind="laya", repo="impacte/bunker-laya", rev="f18245a724515679780751ff9416aa590051f484",
        licence="apache-2.0 (weights; trained partly on CC-BY-NC data)", langs="en (ModernBERT-large)",
        onnx="onnx/model.onnx", extra=["onnx/model.onnx.data", "rl_agent_config.json", "encoder/config.json"],
        tokenizer="tokenizer/tokenizer.json", config="rl_agent_config.json",
        lfs={"onnx/model.onnx": "35d61ece7143958cf32c02e9902673615817a7bdbe5e5003f58c378443e3ac4d",
             "onnx/model.onnx.data": "f1bf96875c18428de4dcbbe3d62233ddbaba46698e70b1c2570ef75c91892a04"},
        max_len=1024),
    "e5-small-probe": dict(
        kind="embed", repo="intfloat/multilingual-e5-small", rev="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        licence="mit", langs="~100", prefix="query: ", pooling="mean",
        onnx="onnx/model_qint8_avx512_vnni.onnx", tokenizer="onnx/tokenizer.json", config="onnx/config.json",
        lfs={"onnx/model_qint8_avx512_vnni.onnx": "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
             "onnx/tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"},
        max_len=512),
    "mminilm-probe": dict(
        kind="embed", repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        rev="e8f8c211226b894fcb81acc59f3b34ba3efd5f42", licence="apache-2.0", langs="50+", prefix="", pooling="mean",
        onnx="onnx/model_quint8_avx2.onnx", tokenizer="tokenizer.json", config="config.json",
        lfs={"onnx/model_quint8_avx2.onnx": "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f",
             "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"},
        max_len=512),
    "gte-small-probe": dict(
        kind="embed", repo="thenlper/gte-small", rev="17e1f347d17fe144873b1201da91788898c639cd",
        licence="mit", langs="en", prefix="", pooling="mean",
        onnx="onnx/model_qint8_avx512_vnni.onnx", tokenizer="tokenizer.json", config="config.json",
        lfs={"onnx/model_qint8_avx512_vnni.onnx": "c9434b8d71617919a3ef61f1fafea4b15b4e02d782cc287623158713881e34cd"},
        max_len=512),
    "embeddinggemma2-probe": dict(
        kind="embed", repo="onnx-community/embeddinggemma-2-ONNX", rev="daa72c51243991dfcaf9f9137d2c573d8f7790c0",
        licence="apache-2.0", langs="100+", prefix="task: classification | query: ", pooling="model",
        onnx="onnx/model_q4.onnx", extra=["onnx/model_q4.onnx_data"], tokenizer="tokenizer.json", config="config.json",
        lfs={"onnx/model_q4.onnx": "f9eeba97acddf139b8ee2ddf04bc30dceafa88de93fadf74d7644e0d61a477a9",
             "onnx/model_q4.onnx_data": "c3975f2d1ab7a1878ae31a7d7a9b7804a827aff3800b60dfceafce21cac3df49",
             "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"},
        max_len=512),
}

DATASETS: dict[str, dict] = {
    # CC-BY-4.0, English, 100k synthetic documents with character spans (55 types)
    "nemotron": dict(repo="nvidia/Nemotron-PII", rev="b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
                     file="data/test-00000-of-00001.parquet", licence="cc-by-4.0"),
    # Apache-2.0, 7 languages (EN FR DE NL ES IT SV), synthetic financial documents with spans
    "gretel": dict(repo="gretelai/synthetic_pii_finance_multilingual", rev="7b844d16738527a04264f50214cb426a4cea0897",
                   file="data/test-00000-of-00001.parquet", licence="apache-2.0"),
    # CC-BY-4.0 (repo tag "other", licence section: CC-BY-4.0), 30 languages, template sentences.
    # Rampart was trained on train+validation of this corpus: its numbers here are optimistic.
    # The validation file is 1 GB and sorted by language: fetch.py reads evenly spaced byte ranges.
    "openpii": dict(repo="ai4privacy/pii-masking-openpii-1.5m", rev="a785eb528e28be2693c3718a27e066970de5dadb",
                    file="data/validation.jsonl", licence="cc-by-4.0", size=1065720636),
    # Apache-2.0, real first prompts of human users in 35 languages: presumed negatives (no labels).
    # A user prompt may still contain personal data: those found by inspection are listed in data.py.
    "oasst": dict(repo="OpenAssistant/oasst1", rev="fdf72ae0827c1cda404aff25b6603abec9e3399b",
                  file="data/train-00000-of-00001-b42a775f407cee45.parquet", licence="apache-2.0"),
    # MIT, 164 Python problems: code negatives without secrets
    "humaneval": dict(repo="openai/openai_humaneval", rev="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                      file="openai_humaneval/test-00000-of-00001.parquet", licence="mit"),
}

DETECTOR_PATH = lambda name: CACHE / MODELS[name]["repo"].replace("/", "__") / MODELS[name]["rev"][:12]  # noqa: E731
DATASET_PATH = lambda name: CACHE / "datasets" / DATASETS[name]["repo"].replace("/", "__") / DATASETS[name]["rev"][:12]  # noqa: E731
