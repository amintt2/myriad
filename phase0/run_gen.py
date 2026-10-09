"""Mode B on generation (GSM8K): peers of different families write one answer together.

Peers are HTTP servers (essaim.peer), on this PC or on another machine. Modes:
  solo        each peer answers alone, in one call (reference)
  vote        majority of the solo final answers (computed from the solo rows: one round trip in total)
  accord      1 round trip per round: every active peer drafts B tokens; the coordinator commits the
              longest word prefix shared by a quorum of the drafts, otherwise the most confident
              non-empty draft
  croise      2 round trips per round: drafts, then every peer scores every draft per word
              (tokenizer-independent); commit the best draft up to the first word the group finds unlikely

`--k K`: a round goes on as soon as K peers have answered. The others' answers are dropped for
that round; their requests still finish on their side (a real peer would cancel them), and a
peer is not sent new work while it is still busy with such a request.

    uv run python run_gen.py --peers http://127.0.0.1:8101 http://127.0.0.1:8102 http://192.168.1.20:8104 --tag essaim3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
import weakref
from collections import Counter
from pathlib import Path

import httpx

from essaim import data
from essaim.common import FIXED_DATE, GEN_PROTOCOL_VERSION, PROMPT_VERSION, final_answer_end, word_spans
from essaim.results import ResultsFile, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
USER_TMPL = ("{q}\n\nSolve it step by step, briefly. Finish with the sentence: "
             "\"The answer is N.\" where N is a number.")
MODES = ("solo", "vote", "accord", "croise")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_response(path: str, body: dict, j) -> None:
    """ValueError unless `j` is a complete, well-typed answer of a peer to `body` on `path`: only such answers
    enter a quorum (a malformed one, e.g. {"compute_ms": 0, "result": {}}, counts as a failed peer instead of
    crashing the coordinator later). Times are finite and non-negative; /propose gives a text, an end flag and a
    finite mean log-probability; /score gives, for every candidate, one finite log-probability per word."""
    if not isinstance(j, dict):
        raise ValueError("réponse qui n'est pas un objet")
    if not _finite(j.get("compute_ms")) or j["compute_ms"] < 0:
        raise ValueError("compute_ms absent ou invalide")
    if "queue_ms" in j and (not _finite(j["queue_ms"]) or j["queue_ms"] < 0):
        raise ValueError("queue_ms invalide")
    r = j.get("result")
    if path == "/propose":
        if not (isinstance(r, dict) and isinstance(r.get("text"), str) and isinstance(r.get("eos"), bool)
                and _finite(r.get("mean_logp"))):
            raise ValueError("résultat de /propose incomplet")
        if r.get("stop") is not None and not isinstance(r["stop"], str):
            raise ValueError("stop invalide")
    elif path == "/score":
        cands = body["candidates"]
        if not isinstance(r, list) or len(r) != len(cands):
            raise ValueError("résultat de /score : pas une note par candidat")
        for cand, s in zip(cands, r):
            wl = s.get("word_logp") if isinstance(s, dict) else None
            if not isinstance(wl, list) or len(wl) != len(word_spans(cand)) or not all(_finite(x) for x in wl):
                raise ValueError("résultat de /score : notes par mot invalides")
            if any(k in s and not isinstance(s[k], bool) for k in ("boundary_merged", "special_token")):
                raise ValueError("résultat de /score : indicateur invalide")
    else:
        raise ValueError(f"chemin inconnu {path}")


class Peers:
    """At most one request in flight per peer: a peer still busy with a dropped (late) request
    is not sent new work until it is done, so useless work never piles up behind its lock.
    Every call is timed and tagged with the question it belongs to."""

    def __init__(self, urls: list[str], token: str | None):
        self.urls = urls
        headers = {"X-Essaim-Token": token} if token else {}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(900.0), headers=headers)
        self.inflight: dict[int, asyncio.Task] = {}
        self.calls: list[dict] = []
        self.tag = None
        self.last_k_eff, self.last_errors = 0, {}
        self._finished_at: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # freed with their tasks

    def _on_done(self, t: asyncio.Task):
        self._finished_at[t] = time.perf_counter()
        if not t.cancelled():
            t.exception()  # retrieved, so a late failure is never reported as "never retrieved"

    async def call(self, i: int, path: str, body: dict, tag) -> tuple[int, dict]:
        """One request; its duration and outcome are recorded even when it fails."""
        t0 = time.perf_counter()
        try:
            r = await self.client.post(self.urls[i] + path, json=body)
            r.raise_for_status()
            j = r.json()
            check_response(path, body, j)  # a malformed answer is a failure, never part of the quorum
            compute, queue = float(j["compute_ms"]), float(j.get("queue_ms", 0.0))
        except Exception as e:  # HTTP, network or malformed answer: recorded, then raised
            self.calls.append({"tag": tag, "peer": i, "path": path, "status": f"{type(e).__name__}: {e}"[:200],
                               "total_ms": round((time.perf_counter() - t0) * 1000, 1)})
            raise
        total = (time.perf_counter() - t0) * 1000
        self.calls.append({"tag": tag, "peer": i, "path": path, "status": "ok", "total_ms": round(total, 1),
                           "compute_ms": compute, "queue_ms": queue, "net_ms": round(total - compute - queue, 1)})
        return i, j

    async def first_k(self, path: str, body: dict, k: int | None, only: list[int] | None = None) -> dict[int, dict]:
        """Ask the peers in `only` (default all), never more than one request in flight per peer, until
        k valid answers have arrived (k=None: all of them). A busy peer is asked as soon as it is free;
        a peer that fails is skipped for this call. Returns fewer than k answers only when no peer is
        left to ask; `self.last_k_eff` and `self.last_errors` record what happened."""
        targets = list(only if only is not None else range(len(self.urls)))
        want = len(targets) if k is None else min(k, len(targets))
        got: dict[int, dict] = {}
        errors: dict[int, str] = {}
        asked: dict[asyncio.Task, int] = {}
        todo = list(targets)
        while len(got) < want:
            for i in list(todo):  # send to every free peer not asked yet
                if i not in self.inflight or self.inflight[i].done():
                    t = asyncio.create_task(self.call(i, path, body, self.tag))
                    t.add_done_callback(self._on_done)  # completion time; late errors still retrieved
                    self.inflight[i] = t
                    asked[t] = i
                    todo.remove(i)
            mine = [t for t in asked if not t.done()]
            busy = [self.inflight[i] for i in todo]  # earlier, dropped requests still running on these peers
            if not mine and not busy:
                break
            done, _ = await asyncio.wait(mine + busy, return_when=asyncio.FIRST_COMPLETED)
            # Several tasks can finish together: take them in completion order and stop at k,
            # the extra ones are treated as late answers.
            for t in sorted((t for t in done if t in asked), key=lambda t: self._finished_at.get(t, 0.0)):
                if asked[t] in got or asked[t] in errors or len(got) >= want:
                    continue
                try:
                    i, j = t.result()
                    got[i] = j
                except Exception as e:  # one failing peer does not stop the round
                    errors[asked[t]] = f"{type(e).__name__}: {e}"
        self.last_k_eff, self.last_errors = len(got), errors
        return got

    async def drain(self):
        pending = [t for t in self.inflight.values() if not t.done()]
        if pending:
            await asyncio.wait(pending)

    async def infos(self) -> list[dict]:
        return [(await self.client.get(u + "/info")).json() for u in self.urls]


def norm_words(text: str) -> list[str]:
    return [w.strip() for w in re.findall(r"\S+\s*", text)]


def cut_words(text: str, k: int) -> str:
    ends = [m.end() for m in re.finditer(r"\S+", text)]
    return text[: ends[k - 1]] if 0 < k <= len(ends) else (text if k else "")


def quorum_prefix(drafts: list[str], q: int) -> tuple[int, str, str]:
    """Longest word prefix shared by at least q drafts: (number of words, prefix text, draft it was cut from)."""
    words = [norm_words(d) for d in drafts]
    best_k, best_text, best_src = 0, "", ""
    for k in range(1, max(map(len, words), default=0) + 1):
        groups = Counter(tuple(w[:k]) for w in words if len(w) >= k)
        if not groups:
            break
        pref, cnt = groups.most_common(1)[0]
        if cnt < q:
            break
        src = next(d for d, w in zip(drafts, words) if tuple(w[:k]) == pref)
        best_k, best_text, best_src = k, cut_words(src, k), src
    return best_k, best_text, best_src


async def solve_solo(peers: Peers, q: str, n: int) -> list[dict]:
    await peers.drain()  # every peer must answer: wait until none is busy
    got = await peers.first_k("/propose", {"user": USER_TMPL.format(q=q), "n": n, "stop": ["final_answer"]}, None)
    out = []
    for i in range(len(peers.urls)):
        if i not in got:  # this peer failed: no answer, the error is kept
            out.append({"text": None, "answer": None, "conf": None, "eos": False,
                        "error": peers.last_errors.get(i, "pas de réponse")})
            continue
        r = got[i]["result"]
        out.append({"text": r["text"], "answer": data.extract_number(r["text"], ended=r["eos"]),
                    "conf": r["mean_logp"], "eos": r["eos"], "stop": r.get("stop"), "compute_ms": got[i]["compute_ms"]})
    return out


def vote(per_peer: list[dict]) -> str | None:
    answers = [p["answer"] for p in per_peer if p["answer"] is not None]
    if not answers:
        return None
    counts = Counter(answers)
    top = max(counts.values())
    tied = {a for a, c in counts.items() if c == top}
    return max((p for p in per_peer if p["answer"] in tied), key=lambda p: p["conf"])["answer"]


async def solve_together(peers: Peers, q: str, mode: str, block: int, max_rounds: int, tau: float, k: int | None) -> dict:
    user = USER_TMPL.format(q=q)
    committed, rounds, trips, log = "", 0, 0, []
    active = list(range(len(peers.urls)))  # peers that have not finished their answer
    finished = False
    while rounds < max_rounds and active and not finished:
        rounds += 1
        got = await peers.first_k("/propose", {"user": user, "assistant_prefix": committed, "n": block}, k, active)
        trips += 1
        entry = {"round": rounds, "propose": {"answered": sorted(got), "k_eff": len(got), "errors": peers.last_errors or None}}
        log.append(entry)  # every phase is logged, even when it yields nothing
        for i, j in got.items():  # a peer that ends with nothing to add has finished
            if j["result"]["eos"] and not j["result"]["text"].strip() and i in active:
                active.remove(i)
        if not got:  # every asked peer failed or none is left: stop rather than spin
            entry["how"] = "aucune réponse"
            break
        drafts = {i: j["result"]["text"] for i, j in got.items() if j["result"]["text"].strip()}
        confs = {i: got[i]["result"]["mean_logp"] for i in drafts}
        if not drafts:
            entry["how"] = "aucun brouillon (pairs terminés)"
            continue  # only finished peers answered this round: ask the remaining active ones
        if mode == "accord":
            kq, text, src = quorum_prefix(list(drafts.values()), len(drafts) // 2 + 1)
            how = f"quorum {kq} mots"
            if kq == 0:
                leader = max(drafts, key=lambda i: confs[i])
                text, src, how = drafts[leader], drafts[leader], f"meneur {leader}"
        else:  # croise
            uniq = list(dict.fromkeys(drafts.values()))
            scores = await peers.first_k("/score", {"user": user, "assistant_prefix": committed, "candidates": uniq},
                                         k, list(got))
            trips += 1
            entry["score"] = {"answered": sorted(scores), "k_eff": len(scores), "errors": peers.last_errors or None}
            best, best_val, best_words = None, -1e9, None
            # Drafts holding a scorer's control token (e.g. "<|im_end|>" written as text by another family)
            # are not prose: never committed, not even by the fallback below.
            banned = {cand for ci, cand in enumerate(uniq if scores else [])
                      if any(s["result"][ci].get("special_token") for s in scores.values())}
            for ci, cand in enumerate(uniq if scores else []):
                if cand in banned:
                    continue
                if any(s["result"][ci].get("boundary_merged") for s in scores.values()):
                    continue  # scored as a different conditional (prefix ended mid-word): not comparable
                nw = len(next(iter(scores.values()))["result"][ci]["word_logp"])
                per_word = [sum(s["result"][ci]["word_logp"][w] for s in scores.values()) / len(scores) for w in range(nw)]
                val = sum(per_word) / max(1, nw)
                if val > best_val:
                    best, best_val, best_words = cand, val, per_word
            if best is None:  # every candidate straddled the boundary: fall back to the most confident draft
                allowed = [i for i in drafts if drafts[i] not in banned]
                if not allowed:  # nothing admissible, and the same peers would propose the same drafts again
                    entry["how"] = "aucun brouillon admissible (jetons de contrôle)"
                    break
                leader = max(allowed, key=lambda i: confs[i])
                text, src, how = drafts[leader], drafts[leader], f"meneur {leader} (frontière ou pas de note)"
            else:
                keep = next((i for i, v in enumerate(best_words) if v < tau and i > 0), len(best_words))
                text, src = (cut_words(best, keep) if keep < len(best_words) else best), best
                how = f"croisé {keep}/{len(best_words)} mots"
        entry.update({"how": how, "added": text})
        if not text.strip():
            break
        raw = committed + text
        # Finalised if the answer and its delimiter are in the committed text, or if the delimiter is the
        # very next character of the draft it was cut from (cutting at a word end drops that space).
        end = final_answer_end(committed + src)
        finished = end is not None and end <= len(raw) + 1
        committed = raw.rstrip(" ")
    return {"text": committed, "answer": data.extract_number(committed, ended=finished or not active), "rounds": rounds,
            "trips": trips, "log": log, "complete": finished or not active}  # explicit completion flag


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", nargs="+", required=True)
    ap.add_argument("--token", default=os.environ.get("ESSAIM_TOKEN"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", choices=data.SPLITS, default="dev")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=["solo", "vote", "accord", "croise"])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--k", type=int, default=None, help="go on after the first k answers (default: all peers)")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--tau", type=float, default=-2.5)
    ap.add_argument("--solo-tokens", type=int, default=320)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    peers = Peers(a.peers, a.token)
    infos = await peers.infos()
    names = [i["model"] for i in infos]
    print("peers:", names, flush=True)
    items = data.gsm8k(a.n, split=a.split)
    manifest = {"peers": [{k: i.get(k) for k in ("model", "revision", "backend", "gguf", "weights", "device", "dtype", "engine", "prompt")}
                          for i in infos],
                "data": data.dataset_identity("gsm8k"), "split": a.split, "n": a.n,
                "block": a.block, "k": a.k, "max_rounds": a.max_rounds, "tau": a.tau, "solo_tokens": a.solo_tokens,
                "prompt": PROMPT_VERSION + "+gsm8k", "protocol": GEN_PROTOCOL_VERSION, "template_date": str(FIXED_DATE)}
    path = RESULTS / f"gen_{a.tag}_{a.split}.jsonl"
    out = ResultsFile(path, manifest, key=("mode", "id"))
    solo_rows = {r["id"]: r for r in read_rows(path) if r["mode"] == "solo"}
    modes = [m for m in MODES if m in a.modes]  # vote always after the solo answers it is computed from
    if "vote" in modes and "solo" not in modes:
        modes.insert(0, "solo")
        print("vote demandé : les réponses solo sont calculées aussi", flush=True)
    for it in items:
        for mode in modes:
            if (mode, it["id"]) in out.done:
                continue
            t0 = time.perf_counter()
            peers.tag = (it["id"], mode)
            if mode == "solo":
                sol = await solve_solo(peers, it["question"], a.solo_tokens)
                rec = {"per_peer": [{"model": m, **s} for m, s in zip(names, sol)]}
                solo_rows[it["id"]] = {**rec, "wall_s": round(time.perf_counter() - t0, 2)}
                if peers.last_k_eff < len(names):
                    rec["missing_peers"] = peers.last_errors
            elif mode == "vote":
                solo = solo_rows[it["id"]]
                rec = {"answer": vote(solo["per_peer"]), "from": "solo",
                       "cost_from_solo_s": solo.get("wall_s")}  # the inference cost is that of the solo answers
            else:
                rec = await solve_together(peers, it["question"], mode, a.block, a.max_rounds, a.tau, a.k)
            decision_s = time.perf_counter() - t0
            # Late (dropped) requests finish here, outside the decision time, so that the next mode starts
            # with idle peers; their cost is still recorded with this mode.
            await peers.drain()
            rec.update({"mode": mode, "id": it["id"], "gold": it["answer"], "peers": names,
                        "wall_s": round(decision_s, 2), "drained_s": round(time.perf_counter() - t0, 2),
                        "calls": [c for c in peers.calls if c["tag"] == (it["id"], mode)]})
            peers.calls = [c for c in peers.calls if c["tag"] != (it["id"], mode)]
            out.write(rec)
            ok = [p["answer"] == it["answer"] for p in rec["per_peer"]] if mode == "solo" else rec.get("answer") == it["answer"]
            print(it["id"], mode, ok, rec.get("rounds", ""), f"{rec['wall_s']}s", flush=True)
    out.release()


if __name__ == "__main__":
    asyncio.run(main())
