"""One frozen language model, wrapped for mode B.

Every peer exposes the same three operations, whatever its family or tokenizer:
  - letter_probs: probability of each answer letter (multiple choice), one forward pass.
  - propose: greedy draft of up to n tokens after a shared assistant prefix, returned as TEXT.
  - score: log-probability of candidate texts, aggregated per WORD so that peers with
    different tokenizers can be compared on the same text.

The conversation is shared as plain text (system, user, assistant prefix); each peer
applies its own chat template. A small prefix cache avoids recomputing the shared prompt
on every call.
"""
from __future__ import annotations

import math
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .common import close_thought, letter_token_ids, render_chat, stop_point, think_spec, word_spans  # noqa: F401 (re-exported)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LM:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", threads: int | None = None,
                 revision: str | None = None):
        self.model_id, self.revision = model_id, revision
        self.device = pick_device(device)
        if threads:
            torch.set_num_threads(threads)
        if dtype == "auto":
            dtype = {"cpu": "float32", "mps": "bfloat16"}.get(self.device, "float16")
        self.dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.think = think_spec(str(self.tok.chat_template or ""))
        self._letter_ids: dict[tuple, list[list[int]]] = {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, revision=revision)
        except (ValueError, KeyError):
            # Multimodal checkpoints (Gemma 4, Qwen3.5) may only register under image-text-to-text.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype, revision=revision)
        self.model.to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos_ids = {i for i in eos + [self.tok.eos_token_id] if i is not None}
        self.special_ids = set(self.tok.all_special_ids)  # control tokens (see score)
        self.cache: DynamicCache | None = None
        self.cache_ids: list[int] = []
        self.stats = {"forward_tokens": 0, "forward_s": 0.0}

    # ---------- prompt ----------
    def render(self, user: str, system: str | None = None, assistant_prefix: str = "", think: bool = False) -> str:
        return render_chat(self.tok, user, system, think) + assistant_prefix

    def encode(self, text: str, offsets: bool = False):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
        return (enc["input_ids"], enc["offset_mapping"]) if offsets else enc["input_ids"]

    # ---------- forward with prefix cache ----------
    @torch.no_grad()
    def _forward(self, ids: list[int], max_start: int | None = None, keep: int = 1) -> tuple[torch.Tensor, int]:
        """Run the uncached suffix of `ids`. Returns (logits, pos0): logits[i] is the
        distribution after position pos0 + i, i.e. it predicts ids[pos0 + i + 1].
        Only the last `keep` positions are materialised (vocabularies reach 262k).
        `max_start` forces recomputation from that position."""
        common = 0
        for a, b in zip(self.cache_ids, ids):
            if a != b:
                break
            common += 1
        start = min(common, len(ids) - 1, max_start if max_start is not None else len(ids))
        cache = self.cache
        if cache is not None and start > 0:
            try:
                drop = len(self.cache_ids) - start
                if drop > 0:
                    cache.crop(-drop)  # negative = number of tokens to remove (positive values are deprecated)
            except Exception:
                cache, start = None, 0
        else:
            cache, start = None, 0
        if cache is None:
            cache = DynamicCache()
        t0 = time.perf_counter()
        inp = torch.tensor([ids[start:]], device=self.device)
        keep = max(1, min(keep, len(ids) - start))
        out = self.model(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=keep)
        self.stats["forward_s"] += time.perf_counter() - t0
        self.stats["forward_tokens"] += len(ids) - start
        self.cache, self.cache_ids = out.past_key_values, list(ids)
        return out.logits[0].float(), len(ids) - keep

    # ---------- operations ----------
    @torch.no_grad()
    def generate(self, user: str, n: int, temperature: float = 0.6, seed: int = 0, think: bool = True,
                 system: str | None = None) -> dict:
        """Sampled answer, optionally with the model's thinking mode."""
        torch.manual_seed(seed)
        ids = torch.tensor([self.encode(self.render(user, system, think=think))], device=self.device)
        t0 = time.perf_counter()
        out = self.model.generate(input_ids=ids, max_new_tokens=n, do_sample=temperature > 0, temperature=temperature,
                                  top_p=0.95, top_k=20)
        self.stats["forward_s"] += time.perf_counter() - t0
        new = out[0, ids.shape[1]:].tolist()
        return {"text": self.tok.decode(new, skip_special_tokens=True), "n_tokens": len(new),
                "truncated": len(new) >= n and new[-1] not in self.eos_ids}

    @torch.no_grad()
    def think_then_letters(self, user: str, letters: list[str], budget: int, seed: int,
                           temperature: float = 0.6) -> dict:
        """Thinking mode with a token budget (s1-style budget forcing), then the letter distribution."""
        torch.manual_seed(seed)
        open_tag, close_tag, _ = self.think
        prompt = self.render(user, think=True)
        ids = torch.tensor([self.encode(prompt)], device=self.device)
        out = self.model.generate(input_ids=ids, max_new_tokens=budget, do_sample=True, temperature=temperature,
                                  top_p=0.95, top_k=20, stop_strings=[close_tag], tokenizer=self.tok)
        new = out[0, ids.shape[1]:].tolist()
        ended_eos = bool(new) and new[-1] in self.eos_ids  # checked before the budget, as llama.cpp does
        thought = self.tok.decode(new, skip_special_tokens=False)
        for e in self.eos_ids:
            thought = thought.replace(self.tok.decode([e]), "")
        stop = "word" if close_tag in thought else ("eos" if ended_eos else ("limit" if len(new) >= budget else "eos"))
        forced = stop == "limit"
        thought = thought.split(close_tag)[0]
        closing = close_thought(thought, self.think, prompt.rstrip().endswith(open_tag))
        probs = self._letters_after(prompt + thought + closing + "Answer:", letters)
        return {"probs": probs, "n_tokens": len(new), "forced": forced, "stop": stop}

    def letter_probs(self, user: str, letters: list[str], system: str | None = None,
                     answer_prefix: str = "Answer:") -> list[float]:
        return self._letters_after(self.render(user, system, answer_prefix), letters)

    def _letters_after(self, prompt: str, letters: list[str]) -> list[float]:
        """Exact probability of each letter (all its single-token spellings), renormalised over the letters."""
        key = tuple(letters)
        if key not in self._letter_ids:
            self._letter_ids[key] = letter_token_ids(self.tok, letters)
        ids = self.encode(prompt)
        logits, _ = self._forward(ids)
        logp = torch.log_softmax(logits[-1], -1)
        scores = [torch.logsumexp(logp[cands], 0).item() for cands in self._letter_ids[key]]
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        z = sum(ex)
        return [e / z for e in ex]

    def propose(self, user: str, assistant_prefix: str, n: int, system: str | None = None,
                stop: list[str] | None = None, deadline_s: float | None = None) -> dict:
        """Greedy draft. `stop`: names from common.STOPS, checked on the whole draft after each token
        (aware of thinking blocks); the text is cut right after the stop point."""
        prompt = self.render(user, system, assistant_prefix)
        ids = self.encode(prompt)
        names = list(stop or [])
        t_end = time.monotonic() + deadline_s if deadline_s else None
        new, logps, ended, cut, why = [], [], False, None, "limit"
        for _ in range(n):
            logits, _ = self._forward(ids + new)
            lp = torch.log_softmax(logits[-1], -1)
            t = int(lp.argmax())
            if t in self.eos_ids:
                ended, why = True, "eos"
                break
            new.append(t)
            logps.append(float(lp[t]))
            if names:
                cut = stop_point(assistant_prefix, self.tok.decode(new, skip_special_tokens=False), names)
                if cut is not None:
                    why = "stop"
                    break
            if t_end and time.monotonic() > t_end:
                why = "deadline"
                break
        text = self.tok.decode(new, skip_special_tokens=False)
        for e in self.eos_ids:
            text = text.replace(self.tok.decode([e]), "")
        if cut is not None:
            text = text[:cut]
        return {"text": text, "n_tokens": len(new), "eos": ended, "stop": why,
                "mean_logp": (sum(logps) / len(logps)) if logps else 0.0}

    def score(self, user: str, assistant_prefix: str, candidates: list[str], system: str | None = None) -> list[dict]:
        """Per-word log-probabilities of each candidate continuation."""
        base = self.render(user, system, assistant_prefix)
        out = []
        for cand in candidates:
            full = base + cand
            ids, offs = self.encode(full, offsets=True)
            b = len(base)
            first = next((j for j, (s, e) in enumerate(offs) if e > b), len(ids))
            first = max(1, first)
            logits, pos0 = self._forward(ids, max_start=first - 1, keep=len(ids) - (first - 1))
            spans = word_spans(cand)
            words = [0.0] * len(spans)
            js = [j for j in range(max(first, pos0 + 1), len(ids))]
            if not js:
                out.append({"word_logp": words, "words": [cand[s:e] for s, e in spans]})
                continue
            rows = logits[[j - 1 - pos0 for j in js]]
            lps = (rows[torch.arange(len(js)), [ids[j] for j in js]] - torch.logsumexp(rows, -1)).tolist()
            for j, lp in zip(js, lps):
                s, e = offs[j]
                pos = max(e - 1, b) - b  # attribute the token to the word holding its last char
                k = next((i for i, (ws, we) in enumerate(spans) if ws <= pos < we), len(spans) - 1)
                if k >= 0:
                    words[k] += lp
            # When the first candidate token straddles the prefix/candidate boundary (prefix ending
            # mid-word), its log-prob is a different conditional than a continuation of the fixed prefix.
            out.append({"word_logp": words, "words": [cand[s:e] for s, e in spans],
                        "boundary_merged": first < len(ids) and offs[first][0] < b,
                        # same rule as the llama.cpp backend: our control tokens in a candidate's text
                        "special_token": any(t in self.special_ids for t in ids[first:])})
        return out

    def identity(self) -> dict:
        """What was actually loaded: used in manifests and compared on resume."""
        import transformers
        return {"model": self.model_id, "revision": self.revision, "backend": "pytorch", "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""), "engine": {"torch": torch.__version__,
                                                                          "transformers": transformers.__version__}}

    def info(self) -> dict:
        return {**self.identity(), "params_b": round(sum(p.numel() for p in self.model.parameters()) / 1e9, 2),
                **self.stats}
