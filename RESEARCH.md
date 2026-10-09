# The research behind Myriad

**Question.** Decentralised LLM inference usually splits one large model across volunteer machines, so
the wide-area network sits inside every decoding step and caps throughput at a few tokens per second.
We study the opposite design: every consumer PC runs a *whole* small open model (≤ 4 B parameters) of a
*different* family (Qwen, Gemma, Granite, SmolLM, Ministral, Phi…), a request costs **one network round
trip**, and the answers are fused locally. Can such a swarm match a single model several times larger,
and with which guarantees?

**Findings so far** (test halves, paired bootstrap 95 % intervals; the paper is in [`paper/`](paper/)):

- Voting pays, writing together does not. On GSM8K, four 1.7–3 B models of four families reach 89.0 %
  with one round trip (best member 86.5 %, Qwen3-4B alone 91.0 %, not significantly different); writing
  the answer together token block by token block needs 20–33 round trips for no gain.
- Wrong answers of different families *disperse*, so two agreeing peers almost always beat two
  disagreeing ones. This is why the Bayes-optimal K-class weight logit(p) − ln(c) works where binary
  log-odds over-trust the best model.
- An exact **stop certificate** returns the full vote's decision as soon as the missing peers can no
  longer change it: the requester waits for 2.5 of 4 peers on average, with identical decisions.
- A minority appeal judged by peers of the same size does not pay (E5); the gain identity B·r − C·h
  explains why.
- One tracker core sustains ≈ 74–96 requests/s from 64 to 4,096 simulated nodes (E7, RTT 100 ms).

## Experiments

| id | question | status | code → results |
| --- | --- | --- | --- |
| E1 | Calibrated fusion of 4 families on multiple choice (ARC-Challenge, MMLU-Pro), with and without thinking | done: ARC +3.7 points over the best member, [+1.2, +6.2] | `phase0/run_mc.py`, `run_think.py`, `analyze_mc.py` → `phase0/results/mc_*`, `think_*` |
| E2 | Generating together on GSM8K: vote vs. majority prefix vs. cross-scored blocks, with round trips | done | `phase0/essaim/peer.py`, `run_gen.py`, `analyze_gen.py` → `gen_*` |
| E3 | Reliability weights and stop certificate, replayed on E2 | done | `phase0/analyze_e3.py` → `e3_*` |
| E4 | 7 families against models 2–7× larger (GSM8K, MATH-500, ARC, MMLU-Pro) | in progress | `phase0/run_solo.py`, `analyze_e4.py` |
| E5 | Minority appeal on multiple choice | done, negative | `phase0/run_appeal.py`, `analyze_appeal.py` → `appeal_*` |
| E6 | The real app end to end with real models and an emulated WAN | harness ready; smoke test only | `app/bench/bench_e2e.py` → `app/bench/results/e6_*` |
| E7 | Scalability of the protocol and of a single tracker (up to 4,096 nodes, no GPU) | done | `app/bench/bench_scale.py` → `app/bench/results/e7_*` |
| E8 | (reserved, not defined yet) | — | — |
| E9 | Executable programs and constraints from different families, verified locally | planned | — |
| E10 | Parallel sections (Skeleton-of-Thought across heterogeneous peers), judged pairwise on MT-Bench | code ready, not run | `phase0/run_sot.py`, `judge_sot.py`, `analyze_sot.py` |

Rules: questions are split once into a development half (choices, weights, thresholds) and a test half
(reported numbers); every result file carries a manifest (model revision, GGUF hash, data identity,
prompt version) and refuses to be resumed with another configuration.

## Reproduce

All models are open-weight under Apache-2.0 or MIT, in safetensors or official GGUF, served in 8-bit by
llama.cpp (`llama-server` on the `PATH` or in `LLAMA_SERVER`). Details, in French:
[`phase0/README.md`](phase0/README.md) and [`app/README.md`](app/README.md).

```bash
# Phase-0 experiments (E1–E5, E10). Python 3.11–3.13, uv; PyTorch is only needed by the transformers path.
cd phase0 && uv sync
uv run python -m unittest discover -s tests                       # tests without a model
uv run python run_mc.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --suffix _gpu --split dev
# Analyses only read the stored JSONL answers in results/ (no model needed):
uv run python analyze_gen.py --split test --tag essaim4_colab --reference ref-Qwen3-4B_gpu   # E2
uv run python analyze_e3.py --tag essaim4_colab --reference ref-Qwen3-4B_gpu                 # E3
uv run python analyze_appeal.py --bench arc                                                  # E5

# App benchmarks (E6 needs a GPU; E7 runs on one PC without GPU).
cd ../app && uv sync
uv run pytest -q
uv run python -m bench.bench_scale run --plan smoke     # then --plan all / v11, and `report`

# Paper numbers and figures, regenerated from the stored results (no number is copied by hand).
cd ../paper && uv run --project ../phase0 python make_numbers.py && uv run --project ../phase0 python make_figures.py
```

The raw answers of every model (JSONL, one line per question and pass, with their `.meta.json`
manifests) are in `phase0/results/`, so all analyses and the paper's numbers can be recomputed without
running a model. Benchmarks: ARC-Challenge, MMLU-Pro, GSM8K, MT-Bench prompts, downloaded from Hugging
Face at pinned revisions by `phase0/essaim/data.py` (the research package of phase 0 kept its original
name, `essaim`; the app's package is `myriad`).
