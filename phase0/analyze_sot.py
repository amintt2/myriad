"""E10: quality and modelled speed of the parallel answer (skeleton + round-robin expansions).

Quality (judge_sot.py verdicts). For each prompt and single model X, the two display orders give the
parallel answer a score of 1 (win), 0.5 (tie) or 0 (loss) each; the prompt's score is their mean, so a
verdict that flips with the order counts as a tie (order-debiased). Over prompts: wins (> 0.5), ties,
losses (< 0.5), mean score (0.5 = parity) with a bootstrap CI over prompts (the two answers of a prompt
are judged together, so the resampling is paired), and the same from the judge's letter probabilities
("soft": P(parallel better) + P(tie) / 2, averaged over the orders).

Pre-declared comparisons: the outline model alone, and the "best single model" = the single model
against which the parallel answer has the LOWEST mean score on the selection split (dev), ties broken
by the soft score; the other splits (test, other) are then reported against that frozen choice.

Speed (a model, not a measurement of wall time). The token counts are measured (greedy generations of
run_sot.py); the speeds are measured single-stream decoding speeds of each model on a consumer GPU
(speeds_consumer.json: RX 6650 XT, tg128, Q8), unknown models take --default-speed; RTT is a parameter.
  single model X:   T = RTT + tokens_X / v_X
  parallel answer:  T = 2 RTT + outline_tokens / v_outline + max over peers (sum of its points' tokens / v_peer)
                    (a peer with several points writes them one after the other; a skeleton fallback
                     costs the outline, then the outline model's single answer)
Effective speed = answer tokens / T (outline + expansion tokens for the parallel answer), summed over prompts.
Prompt processing is ignored unless --pp-speed is given; the wall times recorded by run_sot.py were
measured under batched serving and are not used.

    uv run python analyze_sot.py --tag sot1 --outline-model Qwen/Qwen3.5-4B --peers <P> --baselines <P> \
        --judge Qwen/Qwen3.8-27B --suffix _colab
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from essaim import data, sot
from essaim.results import read_manifest, read_rows

HERE = Path(__file__).resolve().parent


def validate_split(a, items: list[dict]) -> dict[str, str]:
    """Refuse truncated, foreign or incomplete stage files before analysing any answers."""
    ids = [it["id"] for it in items]
    fingerprints = {}
    outlines = {}
    judge_path = sot.judge_path(a.tag, a.judge, a.suffix, a.split)
    judge_manifest = read_manifest(judge_path)
    if judge_manifest is None:
        raise SystemExit(f"{judge_path.name} absent ou sans manifeste")
    baselines = judge_manifest.get("baselines", [])
    if not baselines or len(baselines) != len(set(baselines)) or not set(a.baselines) <= set(baselines):
        raise SystemExit(f"{judge_path.name} : baselines demandées non jugées ou manifeste invalide")
    paths = [(sot.outline_path(a.outline_model, a.suffix, a.split), "outline", a.outline_model)]
    # Every answer the judge saw is loaded for its input fingerprint, even for a subset analysis.
    used_baselines = list(dict.fromkeys([*baselines, a.outline_model]))
    paths += [(sot.base_path(m, a.suffix, a.split), "baseline", m) for m in used_baselines]
    paths += [(sot.expand_path(a.tag, m, a.suffix, a.split), "expand", m) for m in a.peers]
    paths += [(judge_path, "judge", a.judge)]
    identities = {}
    for path, stage, model in paths:
        man = read_manifest(path)
        if man is None or not path.exists():
            raise SystemExit(f"{path.name} absent ou sans manifeste")
        expected = {"split": a.split, "n": len(ids), "data": data.dataset_identity("mtbench"),
                    "prompt": sot.PROMPT_VERSION, "temperature": 0.0, "thinking": False,
                    "judge" if stage == "judge" else "model": model}
        if stage == "judge":
            expected.update(judge_prompt=sot.JUDGE_VERSION)
        else:
            expected.update(stage=stage, seed=sot.SEED,
                            max_tokens={"baseline": sot.BASE_MAX_TOKENS, "outline": sot.OUTLINE_MAX_TOKENS,
                                        "expand": sot.EXPAND_MAX_TOKENS}[stage])
        if stage in ("expand", "judge"):
            expected.update(tag=a.tag, outline_model=a.outline_model, peers=a.peers)
        if any(man.get(k) != v for k, v in expected.items()):
            bad = [k for k, v in expected.items() if man.get(k) != v]
            raise SystemExit(f"{path.name} : manifeste incohérent ({', '.join(bad)})")
        identity = {k: man.get(k) for k in ("revision", "weights", "engine", "gguf")}
        if model in identities and identities[model] != identity:
            raise SystemExit(f"{path.name} : provenance du modèle incohérente")
        identities[model] = identity
        raw = path.read_bytes()
        try:
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        except (ValueError, UnicodeError) as exc:
            raise SystemExit(f"{path.name} : JSONL corrompu ou tronqué") from exc
        key = ("id", "point") if stage == "expand" else ("id", "vs", "order") if stage == "judge" else ("id",)
        keys = [tuple(r[k] for k in key) for r in rows]
        if stage == "expand":
            wanted = {(i, k) for i, o in outlines.items() for k, m in enumerate(sot.assign(len(o["points"]), a.peers))
                      if m == model}
        elif stage == "judge":
            wanted = {(i, m, order) for i in ids for m in baselines for order in ("sot_first", "sot_second")}
        else:
            wanted = {(i,) for i in ids}
        if len(keys) != len(set(keys)) or set(keys) != wanted:
            raise SystemExit(f"{path.name} : clés incomplètes, surnuméraires ou dupliquées "
                             f"({len(wanted - set(keys))} manquantes, {len(set(keys) - wanted)} inattendues)")
        for r in rows:
            if stage == "judge":
                if r["verdict"] not in ("A", "B", "C") or r["sot_is"] != ("A" if r["order"] == "sot_first" else "B"):
                    raise SystemExit(f"{path.name} : verdict ou ordre invalide")
                if (not r.get("probs") and not r.get("identical")) or (r.get("identical") and r["verdict"] != "C"):
                    raise SystemExit(f"{path.name} : probabilités absentes ou égalité identique incohérente")
            elif r["model"] != model or not isinstance(r["n_tokens"], int) or not 0 <= r["n_tokens"] <= man["max_tokens"]:
                raise SystemExit(f"{path.name} : modèle ou nombre de jetons invalide")
            if stage == "outline" and any(r.get(k) != v for k, v in sot.parse_skeleton(r["text"]).items()):
                raise SystemExit(f"{path.name} : squelette enregistré différent du texte")
            if stage == "expand" and (r["point_text"] != outlines[r["id"]]["points"][r["point"]]
                                      or r["n_points"] != len(outlines[r["id"]]["points"])):
                raise SystemExit(f"{path.name} : point développé incohérent")
        if stage == "outline":
            outlines = {r["id"]: r for r in rows}
        for f in (path, path.with_name(path.name + ".meta.json")):
            fingerprints[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return fingerprints


# ---------- quality ----------

def per_prompt(rows: list[dict], x: str, ids: list[str]) -> dict[str, dict]:
    """Order-debiased score of the parallel answer against x, per prompt (both orders required)."""
    by = defaultdict(dict)
    for r in rows:
        if r["vs"] == x:
            by[r["id"]][r["order"]] = r
    out = {}
    for i in ids:
        if set(by[i]) != {"sot_first", "sot_second"}:
            raise SystemExit(f"verdicts manquants pour {i} contre {x}")
        hard, soft, letters = [], [], []
        for r in by[i].values():
            hard.append(sot.sot_outcome(r["verdict"], r["sot_is"]))
            p = r.get("probs")
            soft.append(0.5 if r.get("identical") or not p else p[r["sot_is"]] + 0.5 * p["C"])
            letters.append(r["verdict"])
        out[i] = {"score": sum(hard) / 2, "soft": sum(soft) / 2, "consistent": hard[0] == hard[1],
                  "letters": letters}
    return out


def boot_ci(vals: list[float], boot: int, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(pp: dict[str, dict], boot: int) -> dict:
    s = [d["score"] for d in pp.values()]
    soft = [d["soft"] for d in pp.values()]
    w, l = sum(x > 0.5 for x in s), sum(x < 0.5 for x in s)
    letters = [L for d in pp.values() for L in d["letters"]]
    return {"n": len(s), "win": w, "tie": len(s) - w - l, "loss": l,
            "score": float(np.mean(s)), "score_ci": boot_ci(s, boot),
            "net": (w - l) / len(s), "net_ci": boot_ci([(x > 0.5) - (x < 0.5) for x in s], boot),
            "soft": float(np.mean(soft)), "soft_ci": boot_ci(soft, boot),
            "consistency": sum(d["consistent"] for d in pp.values()) / len(s),
            "share_A": letters.count("A") / len(letters), "share_C": letters.count("C") / len(letters)}


def pick_best(quality: dict[str, dict]) -> str:
    """Pre-declared: the strongest single model against the parallel answer (lowest score, then soft)."""
    return min(quality, key=lambda x: (quality[x]["score"], quality[x]["soft"], x))


# ---------- speed model ----------

def t_single(tokens: int, prompt_tokens: int | None, v: float, rtt: float, pp: float | None) -> float:
    return rtt + ((prompt_tokens or 0) / pp if pp else 0.0) + tokens / v


def t_parallel(e: dict, speed, outline_model: str, rtt: float, pp: float | None) -> float:
    def pre(n):
        return (n or 0) / pp if pp else 0.0
    vo = speed(outline_model)
    t = 2 * rtt + pre(e["outline_prompt_tokens"]) + e["outline_tokens"] / vo
    if e["fallback"]:
        return t + pre(e["fallback_prompt_tokens"]) + e["fallback_tokens"] / vo
    per_peer = defaultdict(float)
    for x in e["expansions"]:
        per_peer[x["model"]] += pre(x["prompt_tokens"]) + x["n_tokens"] / speed(x["model"])
    return t + max(per_peer.values())


def parallel_tokens(e: dict) -> int:
    """Tokens of the answer shown: skeleton + expansions, or only the single answer after a fallback."""
    return e["fallback_tokens"] if e["fallback"] else e["outline_tokens"] + sum(x["n_tokens"] for x in e["expansions"])


def speed_table(par: dict, singles: dict, xs: list[str], speed, outline_model: str, rtt: float, pp) -> dict:
    ids = list(par)
    tp = [t_parallel(par[i], speed, outline_model, rtt, pp) for i in ids]
    tok_p = sum(parallel_tokens(par[i]) for i in ids)
    out = {"parallel": {"mean_s": float(np.mean(tp)), "eff_tok_s": tok_p / sum(tp)}}
    for x in xs:
        tx = [t_single(singles[x][i]["n_tokens"], singles[x][i].get("prompt_tokens"), speed(x), rtt, pp) for i in ids]
        tok_x = sum(singles[x][i]["n_tokens"] for i in ids)
        out[x] = {"mean_s": float(np.mean(tx)), "eff_tok_s": tok_x / sum(tx),
                  "speedup_latency": float(np.mean(tx) / np.mean(tp)),
                  "speedup_median": float(statistics.median(a / b for a, b in zip(tx, tp))),
                  "speedup_eff": (tok_p / sum(tp)) / (tok_x / sum(tx))}
    return out


def structure(par: dict, singles: dict, outline_rows: dict) -> dict:
    nonfb = [e for e in par.values() if not e["fallback"]]
    exps = [x for e in nonfb for x in e["expansions"]]
    return {"n": len(par), "fallback": sum(e["fallback"] for e in par.values()),
            "status": dict(sorted(Counter(r["status"] for r in outline_rows.values()).items())),
            "long_points": sum(r.get("n_long", 0) for r in outline_rows.values()),
            "mean_points": float(np.mean([len(e["points"]) for e in nonfb])) if nonfb else 0.0,
            "mean_peers_used": float(np.mean([len({x["model"] for x in e["expansions"]}) for e in nonfb])) if nonfb else 0.0,
            "expansions_cut": sum(x["finish"] == "length" for x in exps), "expansions": len(exps),
            "mean_tokens_parallel": float(np.mean([parallel_tokens(e) for e in par.values()])),
            "mean_tokens_single": {x: float(np.mean([r["n_tokens"] for r in s.values()])) for x, s in singles.items()},
            "single_cut": {x: sum(r["finish"] == "length" for r in s.values()) for x, s in singles.items()}}


def fmt_ci(v, ci, pct=False):
    f = (lambda z: f"{100 * z:.1f}") if pct else (lambda z: f"{z:.3f}")
    return f"{f(v)} [{f(ci[0])} ; {f(ci[1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outline-model", required=True)
    ap.add_argument("--peers", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--splits", nargs="+", default=list(data.MT_SPLITS), choices=list(data.MT_SPLITS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--select-split", default="dev", help="split on which the best single model is chosen")
    ap.add_argument("--best", help="force the best single model instead of choosing it")
    ap.add_argument("--speeds", default=str(HERE / "speeds_consumer.json"))
    ap.add_argument("--default-speed", type=float, default=40.0, help="tok/s of models absent from --speeds")
    ap.add_argument("--speed", nargs="*", default=[], metavar="MODEL=TOKS", help="override speeds")
    ap.add_argument("--uniform-speed", type=float, default=50.0, help="variant: every model at this speed")
    ap.add_argument("--rtt", nargs="+", type=float, default=[50, 100, 150], help="round-trip times (ms)")
    ap.add_argument("--pp-speed", type=float, default=None, help="prompt processing tok/s (default: ignored)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json", help="write every number to this JSON file")
    a = ap.parse_args()

    table = {k: float(v) for k, v in json.loads(Path(a.speeds).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    for kv in a.speed:
        m, v = kv.rsplit("=", 1)
        table[m] = float(v)
    models = list(dict.fromkeys([a.outline_model, *a.peers, *a.baselines]))
    defaulted = [m for m in models if m not in table]
    measured = lambda m: table.get(m, a.default_speed)  # noqa: E731
    uniform = lambda m: a.uniform_speed  # noqa: E731
    print("vitesses (tok/s) : " + ", ".join(f"{m.split('/')[-1]} {measured(m):g}" for m in models)
          + (f" ; par défaut ({a.default_speed:g}) : {', '.join(m.split('/')[-1] for m in defaulted)}" if defaulted else ""))
    print("Vitesse MODÉLISÉE : nombres de jetons mesurés, débits mono-flux mesurés ou supposés, RTT en paramètre.\n")

    report = {"protocol": {"tag": a.tag, "outline_model": a.outline_model, "judge": a.judge,
                           "peers": a.peers, "baselines": a.baselines, "suffix": a.suffix,
                           "select_split": a.select_split, "forced_best": a.best, "n": a.n,
                           "boot": a.boot, "rtt_ms": a.rtt, "uniform_speed": a.uniform_speed,
                           "default_speed": a.default_speed, "pp_speed": a.pp_speed,
                           "budgets": {"baseline": sot.BASE_MAX_TOKENS, "outline": sot.OUTLINE_MAX_TOKENS,
                                       "expand": sot.EXPAND_MAX_TOKENS}},
              "complete": a.n is None and set(a.splits) == set(data.MT_SPLITS), "sources_sha256": {},
              "speeds": {m: measured(m) for m in models}, "defaulted": defaulted, "splits": {}}
    quality_by_split, loaded = {}, {}
    for split in a.splits:
        items = data.mt_bench(a.n, split=split)
        a.split = split
        report["sources_sha256"].update(validate_split(a, items))
        ids = [it["id"] for it in items]
        par = sot.load_sot(a.tag, a.outline_model, a.peers, a.suffix, split, ids)
        singles = {x: {r["id"]: r for r in sot.complete(sot.base_path(x, a.suffix, split), ids, what="baseline")}
                   for x in a.baselines}
        outline_rows = {r["id"]: r for r in sot.complete(sot.outline_path(a.outline_model, a.suffix, split), ids)}
        jp = sot.judge_path(a.tag, a.judge, a.suffix, split)
        man = read_manifest(jp)
        if man is None:
            raise SystemExit(f"{jp.name} absent : lancer judge_sot.py")
        if not set(a.baselines) <= set(man["baselines"]):
            raise SystemExit(f"{jp.name} : {sorted(set(a.baselines) - set(man['baselines']))} non jugés")
        judged = {x: singles[x] if x in singles else
                  {r["id"]: r for r in sot.complete(sot.base_path(x, a.suffix, split), ids, what="baseline")}
                  for x in man["baselines"]}  # every answer the judge saw, to check its fingerprint
        inputs = {i: {"sot": par[i]["text"], **{x: judged[x][i]["text"] for x in man["baselines"]}} for i in ids}
        if man.get("inputs_sha256") != sot.canonical_sha(inputs):
            raise SystemExit(f"{jp.name} : jugé sur d'autres réponses ou d'autres modèles que ceux analysés")
        jrows = read_rows(jp, ("id", "vs", "order"))
        quality_by_split[split] = {x: summarize(per_prompt(jrows, x, ids), a.boot) for x in a.baselines}
        loaded[split] = (par, singles, outline_rows)

    if a.best:
        best = a.best
    elif a.select_split in quality_by_split:
        best = pick_best(quality_by_split[a.select_split])
    else:
        raise SystemExit(f"--select-split {a.select_split} non analysé : ajoute-le à --splits ou donne --best")
    print(f"meilleur modèle seul (choisi sur {a.best and 'la ligne de commande' or a.select_split}) : {best}\n")
    report["best_single"] = best
    focus = [x for x in dict.fromkeys([a.outline_model, best]) if x in a.baselines]

    for split in a.splits:
        par, singles, outline_rows = loaded[split]
        q = quality_by_split[split]
        st = structure(par, singles, outline_rows)
        print(f"===== {split} ({st['n']} prompts) =====")
        print(f"squelettes : {st['status']}, repli {st['fallback']}, points > {sot.MAX_POINT_WORDS} mots : "
              f"{st['long_points']}, {st['mean_points']:.1f} points et {st['mean_peers_used']:.1f} pairs par réponse, "
              f"expansions coupées {st['expansions_cut']}/{st['expansions']}")
        print(f"jetons moyens : parallèle {st['mean_tokens_parallel']:.0f} ; "
              + ", ".join(f"{x.split('/')[-1]} {v:.0f} (coupées {st['single_cut'][x]})"
                          for x, v in st["mean_tokens_single"].items()))
        print("qualité (réponse parallèle contre le modèle seul ; score 0,5 = parité) :")
        for x in a.baselines:
            s = q[x]
            mark = " <- modèle du squelette" if x == a.outline_model else ""
            mark += " <- meilleur seul" if x == best else ""
            print(f"  {x.split('/')[-1]:<34} V/N/D {s['win']}/{s['tie']}/{s['loss']}  score {fmt_ci(s['score'], s['score_ci'])}"
                  f"  net {fmt_ci(s['net'], s['net_ci'], True)} %  souple {fmt_ci(s['soft'], s['soft_ci'])}"
                  f"  cohérence {100 * s['consistency']:.0f} %  A {100 * s['share_A']:.0f} %{mark}")
        sp = {}
        for label, fn in (("mono-flux + défauts", measured), (f"toutes à {a.uniform_speed:g} tok/s", uniform)):
            for rtt in a.rtt:
                t = speed_table(par, singles, a.baselines, fn, a.outline_model, rtt / 1000, a.pp_speed)
                sp[f"{label}|{rtt:g}"] = t
                print(f"vitesse ({label}, RTT {rtt:g} ms) : parallèle {t['parallel']['mean_s']:.1f} s, "
                      f"{t['parallel']['eff_tok_s']:.0f} tok/s effectifs")
                for x in focus:
                    r = t[x]
                    print(f"    contre {x.split('/')[-1]:<30} {r['mean_s']:.1f} s, {r['eff_tok_s']:.0f} tok/s ; "
                          f"accélération latence x{r['speedup_latency']:.2f} (médiane x{r['speedup_median']:.2f}), "
                          f"débit effectif x{r['speedup_eff']:.2f}")
        print()
        report["splits"][split] = {"structure": st, "quality": q, "speed": sp}
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
