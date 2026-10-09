"""E6: the real app end to end, with real models (llama-server), an optional emulated WAN delay, and a
question set replayed through the gateway's OpenAI-compatible endpoint.

One process: a tracker (uvicorn, optional WAN delay; or --tracker-url for an external one), K nodes
(one llama-server each, via myriad.engine.LlamaServerEngine), one client node with its gateway
(uvicorn on 127.0.0.1), and a client sending POST /v1/chat/completions with bounded concurrency.
Grading: phase0/essaim/answers.py (imported read-only), the grader of the phase-0 measurements.

    uv run python -m bench.bench_e2e \\
        --node Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf \\
        --node ibm-granite/granite-3.3-2b-instruct-GGUF:granite-3.3-2b-instruct-Q8_0.gguf \\
        --questions gsm8k-test --n 200 --rtt-ms 0,50,100,150 --early-stop both --parallel 4 --label vm

A node is REPO:FILE (downloaded from Hugging Face if absent) or REPO:FILE=LOCAL_PATH.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python bench/bench_e2e.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "bench"

from bench.common import RESULTS, pct, r

ROOT = Path(__file__).resolve().parents[2]  # repository root (app/bench -> app -> root)
GSM8K_PART = 300  # phase0/essaim/data.py: dev = shuffled items [0, 300), test = [300, 600)


# ------------------------------------------------------------ questions and grading
def phase0_answers(root: Path = ROOT):
    """phase0/essaim/answers.py, imported under its own package name `phase0_essaim` (phase0 is not an
    installed package, and its name `essaim` was also the app's before the rename); it only depends on
    the standard library."""
    if "phase0_essaim.answers" in sys.modules:
        return sys.modules["phase0_essaim.answers"]
    pkg_dir = root / "phase0" / "essaim"
    spec = importlib.util.spec_from_file_location("phase0_essaim", pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["phase0_essaim"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("phase0_essaim.answers")


def load_gsm8k(split: str, data_dir: Path) -> list[dict]:
    """The GSM8K partition of phase 0 (same seeded shuffle, same cache file, hash checked)."""
    f = data_dir / "gsm8k_all_s0_v2.jsonl"
    meta = data_dir / "gsm8k_all_s0_v2.meta.json"
    if not f.exists() or not meta.exists():
        raise SystemExit(
            f"{f} introuvable. Le construire une fois avec la phase 0 :\n"
            "  cd phase0 && uv run python -c \"from myriad import data; data.gsm8k(300, split='test')\"\n"
            "ou donner --phase0-data DOSSIER, ou --questions FICHIER.json")
    raw = f.read_bytes()
    m = json.loads(meta.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != m["sha256"]:
        raise SystemExit(f"{f.name} ne correspond pas à son empreinte ({meta.name})")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    lo = 0 if split == "dev" else GSM8K_PART
    return items[lo:lo + GSM8K_PART]


def load_questions(spec: str, data_dir: Path) -> tuple[list[dict], dict]:
    if spec in ("gsm8k-test", "gsm8k-dev"):
        split = spec.split("-")[1]
        return load_gsm8k(split, data_dir), {"source": f"phase0 GSM8K {split} (openai/gsm8k, seeded shuffle, seed 0)"}
    p = Path(spec)
    text = p.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    out = [{"id": str(x.get("id", i)), "question": x["question"], "answer": str(x["answer"])} for i, x in enumerate(rows)]
    return out, {"source": str(p), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


# ------------------------------------------------------------ swarm
def parse_node(spec: str) -> tuple[str, str, str | None]:
    """REPO:FILE[=PATH] -> (repo, file, path or None)."""
    path = None
    if "=" in spec:
        spec, path = spec.split("=", 1)
    if ":" not in spec:
        raise SystemExit(f"--node attend DÉPÔT:FICHIER[=CHEMIN], pas {spec!r}")
    repo, fname = spec.split(":", 1)
    return repo, fname, path


async def start_tracker(rtt_ms: float, sigma: float, db: Path):
    import uvicorn

    from myriad.engine import free_port
    from myriad.netem import wan_from_rtt
    from myriad.protocol import MAX_FRAME_BYTES
    from myriad.tracker import Tracker

    # Starter credit 1e9: the requester must not run out of credits during a long replay.
    tracker = Tracker(db_path=db, starter_credit=1e9, wan=wan_from_rtt(rtt_ms, sigma))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(tracker.app, host="127.0.0.1", port=port, log_level="warning",
                                           ws_max_size=MAX_FRAME_BYTES, timeout_graceful_shutdown=2))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            raise RuntimeError("le traqueur n'a pas démarré")
        await asyncio.sleep(0.05)
    return tracker, server, task, f"http://127.0.0.1:{port}"


async def connect_swarm(url: str, engines: list, specs: list, args):
    """Serving nodes (one per engine) and one client node with its gateway on a local port."""
    import uvicorn

    from myriad.crypto import Identity
    from myriad.engine import free_port
    from myriad.gateway import Gateway
    from myriad.node import NodeClient
    from myriad.priors import family_of, params_of

    nodes, tasks = [], []
    for eng, (repo, fname, path) in zip(engines, specs):
        n = NodeClient(Identity.generate(), url, engine=eng, model=repo, family=family_of(repo), gguf=fname,
                       params_b=params_of(repo), ctx=args.ctx, max_parallel=args.parallel, reconnect=False,
                       max_job_tokens=args.max_tokens)
        tasks.append(asyncio.create_task(n.run()))
        nodes.append(n)
    client = NodeClient(Identity.generate(), url, reconnect=False)
    tasks.append(asyncio.create_task(client.run()))
    await asyncio.wait_for(asyncio.gather(*(n.connected.wait() for n in [*nodes, client])), 60)
    gw = Gateway(client, default_k=args.k, timeout_s=args.timeout)
    gport = free_port()
    gserver = uvicorn.Server(uvicorn.Config(gw.app, host="127.0.0.1", port=gport, log_level="warning"))
    tasks.append(asyncio.create_task(gserver.serve()))
    while not gserver.started:
        await asyncio.sleep(0.05)
    return nodes, client, gw, gserver, tasks, f"http://127.0.0.1:{gport}/v1"


async def stop_server(server, task, timeout: float = 5.0) -> None:
    """Let a uvicorn server shut down by itself (no traceback), cancel it only if it hangs."""
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
        task.cancel()


async def replay(base: str, items: list[dict], args, early_stop: bool, grader) -> list[dict]:
    import httpx

    sem = asyncio.Semaphore(args.concurrency)
    out: list[dict] = []

    async def one(it: dict, http: httpx.AsyncClient, record: bool = True) -> None:
        body = {"model": "myriad", "messages": [{"role": "user", "content": it["question"]}],
                "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0,
                "myriad": {"k": args.k, "task_hint": "math", "timeout_s": args.timeout, "early_stop": early_stop}}
        async with sem:
            t_sub = time.time()
            t0 = time.perf_counter()
            try:
                resp = await http.post(f"{base}/chat/completions", json=body, timeout=args.timeout + 30)
                status, j = resp.status_code, resp.json()
            except Exception as e:  # noqa: BLE001 - recorded
                status, j = None, {"error": {"message": f"{type(e).__name__}: {e}"[:300]}}
            lat = time.perf_counter() - t0
        if not record:
            return
        gold = grader.norm_math(str(it["answer"]))
        rec = {"id": it["id"], "gold": gold, "t": t_sub, "latency_s": lat, "status": status}
        if status == 200:
            ch = j["choices"][0]
            text = ch["message"]["content"] or ""
            meta = j.get("myriad", {})
            graded = grader.extract("gsm8k", text, {}, ended=ch.get("finish_reason") == "stop")
            rec.update(ok=True, answer=graded, correct=graded == gold, app_answer=meta.get("answer"),
                       app_correct=grader.norm_math(meta["answer"]) == gold if meta.get("answer") else False,
                       finish_reason=ch.get("finish_reason"), early_stop=meta.get("early_stop"),
                       certificate=meta.get("certificate"), decision=meta.get("decision"),
                       peers_asked=meta.get("peers_asked"), peers_answered=meta.get("peers_answered"),
                       gateway_latency_ms=meta.get("latency_ms"),
                       peers=[{"model": p["model"], "status": p["status"], "answer": p["answer"],
                               "correct": (grader.norm_math(p["answer"]) == gold) if p["answer"] else False,
                               "latency_ms": p["latency_ms"], "compute_ms": p["compute_ms"],
                               "completion_tokens": p["completion_tokens"], "error": p["error"],
                               "weight": p["weight"]} for p in meta.get("peers", [])])
        else:
            rec.update(ok=False, correct=False, error=(j.get("error") or {}).get("message"))
        out.append(rec)

    async with httpx.AsyncClient() as http:
        for it in items[:args.warmup]:  # warm the llama-servers (neither recorded nor timed)
            await one(it, http, record=False)
        t0 = time.time()
        await asyncio.gather(*(one(it, http) for it in items))
        wall = time.time() - t0
    return sorted(out, key=lambda x: x["t"]), wall


def summarize(recs: list[dict], wall_s: float) -> dict:
    ok = [x for x in recs if x.get("ok")]
    lat = [x["latency_s"] for x in ok]
    per_model: dict[str, dict] = defaultdict(lambda: {"answered": 0, "correct": 0, "compute_ms": [], "latency_ms": []})
    for x in ok:
        for p in x["peers"]:
            m = per_model[p["model"]]
            if p["status"] == "ok":
                m["answered"] += 1
                m["correct"] += bool(p["correct"])
                if p["compute_ms"] is not None:
                    m["compute_ms"].append(p["compute_ms"])
                if p["latency_ms"] is not None:
                    m["latency_ms"].append(p["latency_ms"])
    n = len(recs)
    return {
        "requests": n, "ok": len(ok), "failed": n - len(ok),
        "accuracy": r(sum(x["correct"] for x in recs) / n if n else None, 4),
        "accuracy_app_extractor": r(sum(x.get("app_correct", False) for x in recs) / n if n else None, 4),
        "latency_s": {"p50": r(pct(lat, 50)), "p95": r(pct(lat, 95)), "p99": r(pct(lat, 99)),
                      "mean": r(statistics.mean(lat) if lat else None), "max": r(max(lat) if lat else None)},
        "peers_answered_mean": r(statistics.mean(x["peers_answered"] for x in ok) if ok else None),
        "early_stop_frac": r(statistics.mean(bool(x["early_stop"]) for x in ok) if ok else None),
        "peer_status": dict(Counter(p["status"] for x in ok for p in x["peers"])),
        "peer_errors": dict(Counter(p["error"] for x in ok for p in x["peers"] if p["error"])),
        "per_model": {k: {"answered": v["answered"], "accuracy_when_answered": r(v["correct"] / v["answered"], 4)
                          if v["answered"] else None, "compute_ms_p50": r(pct(v["compute_ms"], 50), 1),
                          "latency_ms_p50": r(pct(v["latency_ms"], 50), 1)} for k, v in per_model.items()},
        "wall_s": r(wall_s, 1), "throughput_rps": r(n / wall_s if wall_s else None),
    }


def tool_version(binary: str) -> str | None:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        lines = [x.strip() for x in (out.stdout + out.stderr).splitlines() if x.strip()]
        keep = [x for x in lines if x.startswith(("version", "built with"))] or lines[-2:]
        return " | ".join(keep)[:300]
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    smi = shutil.which("nvidia-smi")
    if smi is None and os.name == "nt":
        cand = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(cand) if cand.exists() else None
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------ main
async def main_async(args) -> Path:
    from myriad.engine import LlamaServerEngine, resolve_binary

    grader = phase0_answers()
    items, qsrc = load_questions(args.questions, Path(args.phase0_data))
    items = items[args.offset:args.offset + args.n]
    specs = [parse_node(s) for s in args.node]
    if not specs and not args.tracker_url:
        raise SystemExit("au moins un --node (ou --tracker-url d'un réseau existant)")
    binary = resolve_binary(args.llama_server) if specs else None
    workdir = Path(args.workdir or (Path(tempfile.gettempdir()) / "myriad_e6"))
    workdir.mkdir(parents=True, exist_ok=True)
    engines, models_info = [], []
    for repo, fname, path in specs:
        if path is None:
            from myriad.cli import download_model
            path = download_model(repo, fname)
        engines.append(LlamaServerEngine(path, repo, binary=binary, ctx=args.ctx, parallel=args.parallel,
                                         n_gpu_layers=args.n_gpu_layers, log_dir=workdir))
        models_info.append({"repo": repo, "file": fname, "size": Path(path).stat().st_size})
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.early_stop]
    rtts = [float(x) for x in str(args.rtt_ms).split(",") if x.strip()]
    runs = []
    try:
        print(f"Démarrage de {len(engines)} llama-server…", flush=True)
        await asyncio.gather(*(e.start() for e in engines))
        for rtt in rtts:
            if args.tracker_url:
                tracker = server = ttask = None
                url = args.tracker_url
            else:
                # A fresh ledger per run: reliability counters must not leak from earlier runs.
                db = Path(tempfile.mkdtemp(prefix="tracker_", dir=workdir)) / "tracker.sqlite"
                tracker, server, ttask, url = await start_tracker(rtt, args.wan_sigma, db)
            nodes, client, gw, gserver, tasks, base = await connect_swarm(url, engines, specs, args)
            try:
                for es in modes:
                    print(f"RTT {rtt:g} ms, certificat {'oui' if es else 'non'} : {len(items)} questions…", flush=True)
                    recs, wall = await replay(base, items, args, es, grader)
                    summ = summarize(recs, wall)
                    print("  " + json.dumps({k: summ[k] for k in ("ok", "accuracy", "latency_s", "peers_answered_mean",
                                                                  "early_stop_frac")}), flush=True)
                    runs.append({"rtt_ms": rtt, "wan_sigma": args.wan_sigma if rtt > 0 else None,
                                 "early_stop": es, "summary": summ, "records": recs})
            finally:
                await stop_server(gserver, tasks[-1])
                await gw.close()
                for n in [*nodes, client]:
                    await n.stop()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if server is not None:
                    await stop_server(server, ttask)
                    tracker.ledger.close()
    finally:
        for e in engines:
            await e.close()
    label = args.label or time.strftime("%Y%m%d_%H%M")
    out = RESULTS / f"e6_{label}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": "E6", "smoke_test": bool(args.smoke), "label": label, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine": {"platform": platform.platform(), "python": platform.python_version(), "gpu": gpu_name(),
                        "llama_server": tool_version(binary) if binary else None},
            "settings": {"nodes": models_info, "k": args.k, "parallel_slots": args.parallel, "ctx_per_slot": args.ctx,
                         "concurrency": args.concurrency, "max_tokens": args.max_tokens, "timeout_s": args.timeout,
                         "temperature": 0.0, "questions": {**qsrc, "offset": args.offset, "n": len(items)},
                         "grader": "phase0/essaim/answers.py extract('gsm8k', ended = finish_reason == 'stop')",
                         "tracker": args.tracker_url or "in-process, starter credit 1e9",
                         "wan_sigma": args.wan_sigma, "warmup_requests": args.warmup},
            "runs": runs}
    out.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = write_md(meta, out.with_suffix(".md"))
    print(f"Résultats : {out}\n           {md}")
    return out


def write_md(meta: dict, path: Path) -> Path:
    s = meta["settings"]
    L = [f"# E6 : l'app de bout en bout avec de vrais modèles ({meta['label']})", ""]
    if meta["smoke_test"]:
        L += ["> **Test de fumée** : quelques questions sur le PC du dépôt, pour vérifier le montage. Ces chiffres ne "
              "sont **pas** des résultats de l'article.", ""]
    L += [f"- Date : {meta['date']}. Machine : {meta['machine']['platform']}, GPU : {meta['machine']['gpu'] or '–'}.",
          f"- llama-server : {meta['machine']['llama_server'] or '–'}.",
          "- Nœuds : " + ", ".join(f"{m['repo']}:{m['file']}" for m in s["nodes"]) + ".",
          f"- k = {s['k']}, {s['parallel_slots']} emplacement(s) par llama-server, {s['concurrency']} requête(s) "
          f"simultanée(s), {s['max_tokens']} jetons au plus, température 0, délai {s['timeout_s']:g} s.",
          f"- Questions : {s['questions']['source']}, {s['questions']['n']} à partir de {s['questions']['offset']}. "
          f"Correcteur : {s['grader']}.",
          f"- Traqueur : {s['tracker']}. WAN émulé dans le relais : retard aller simple lognormal "
          f"(σ = {s['wan_sigma']}), « RTT » = aller-retour médian nominal client–traqueur.", "",
          "| RTT (ms) | certificat | requêtes | servies | exactitude | p50 (s) | p95 (s) | p99 (s) | pairs attendus | "
          "arrêt anticipé | req/s |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in meta["runs"]:
        x = run["summary"]
        lt = x["latency_s"]

        def g(v, nd=2):
            return "–" if v is None else f"{v:.{nd}f}"
        L.append(f"| {run['rtt_ms']:g} | {'oui' if run['early_stop'] else 'non'} | {x['requests']} | {x['ok']} | "
                 f"{g(100 * x['accuracy'] if x['accuracy'] is not None else None, 1)} % | {g(lt['p50'])} | "
                 f"{g(lt['p95'])} | {g(lt['p99'])} | {g(x['peers_answered_mean'])} | "
                 f"{g(100 * x['early_stop_frac'] if x['early_stop_frac'] is not None else None, 0)} % | "
                 f"{g(x['throughput_rps'], 3)} |")
    L += ["", "Par modèle (réponses reçues avant la décision) :", "",
          "| RTT (ms) | certificat | modèle | réponses | exactitude | calcul p50 (ms) | latence p50 (ms) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in meta["runs"]:
        for m, v in run["summary"]["per_model"].items():
            acc = v["accuracy_when_answered"]
            L.append(f"| {run['rtt_ms']:g} | {'oui' if run['early_stop'] else 'non'} | {m} | {v['answered']} | "
                     f"{'–' if acc is None else f'{100 * acc:.1f} %'} | {v['compute_ms_p50']} | {v['latency_ms_p50']} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", action="append", default=[], help="DÉPÔT:FICHIER.gguf[=CHEMIN_LOCAL] (répétable)")
    ap.add_argument("--llama-server", help="chemin de llama-server (sinon LLAMA_SERVER ou le PATH)")
    ap.add_argument("--tracker-url", help="traqueur existant (sinon un traqueur local est lancé pour chaque RTT)")
    ap.add_argument("--rtt-ms", default="0", help="liste de RTT émulés, ex. 0,50,100,150 (traqueur local seulement)")
    ap.add_argument("--wan-sigma", type=float, default=0.25)
    ap.add_argument("--questions", default="gsm8k-test", help="gsm8k-test, gsm8k-dev, ou un fichier JSON/JSONL "
                                                             "{id, question, answer}")
    ap.add_argument("--phase0-data", default=str(ROOT / "phase0" / "data"), help="cache des questions de la phase 0")
    ap.add_argument("--n", type=int, default=200, help="questions (200 = le jeu test de la phase 0)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--early-stop", choices=["on", "off", "both"], default="both")
    ap.add_argument("--parallel", type=int, default=1, help="emplacements par llama-server (= jobs simultanés par nœud)")
    ap.add_argument("--concurrency", type=int, default=None, help="requêtes simultanées (défaut : --parallel)")
    ap.add_argument("--ctx", type=int, default=4096, help="contexte par emplacement")
    ap.add_argument("--n-gpu-layers", type=int, default=999)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--warmup", type=int, default=1, help="requêtes d'échauffement non comptées")
    ap.add_argument("--label", help="suffixe des fichiers de résultats (results/e6_<label>.json/.md)")
    ap.add_argument("--workdir", help="journaux de llama-server et base du traqueur (défaut : dossier temporaire myriad_e6)")
    ap.add_argument("--smoke", action="store_true", help="marquer les résultats comme test de fumée")
    args = ap.parse_args(argv)
    if args.concurrency is None:
        args.concurrency = max(1, args.parallel)
    import logging
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
