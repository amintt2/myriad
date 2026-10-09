"""Run phase-0 jobs on a cloud GPU (Colab), driven from the PC with the Colab CLI.

The PC uploads this code, starts this launcher on the VM, then periodically downloads the
results files (partial ones included, as checkpoints) and commits them itself: there is no
git and no token on the VM. Jobs run in parallel (one llama-server each) as long as the
free GPU memory allows; each job is a normal run_mc.py / run_think.py call, so manifests,
locks and resumption work exactly as on the PC. A job that needs the results of others has a later
stage (STAGE, by job kind): it starts only once every job of the earlier stages has succeeded.

    LLAMA_SERVER=/content/llama/bin/llama-server python colab_jobs.py --plan colab-1
Exit code 0 only if every job succeeded.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"
STATUS = HERE / "results" / "colab_status.json"

# model id -> (GGUF repo, GGUF file, GPU memory to reserve in GiB: weights + context + margin)
GGUF = {
    "Qwen/Qwen3-1.7B": ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf", 4),
    "Qwen/Qwen3-4B": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", 7),
    "HuggingFaceTB/SmolLM3-3B": ("ggml-org/SmolLM3-3B-GGUF", "SmolLM3-Q8_0.gguf", 6),
    "ibm-granite/granite-3.3-2b-instruct": ("ibm-granite/granite-3.3-2b-instruct-GGUF", "granite-3.3-2b-instruct-Q8_0.gguf", 5),
    "google/gemma-4-E2B-it": ("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf", 8),
}
# E4 (run_solo.py): every model answers every question once; the swarm is computed offline.
# model id -> (GGUF repo, GGUF file, GPU memory to reserve in GiB incl. the KV cache of all slots, slots)
SOLO = {
    # small peers (<= 4B), the newest of each family (2026-10)
    "Qwen/Qwen3.5-4B": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf", 13, 16),
    "Qwen/Qwen3.5-2B": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q8_0.gguf", 10, 16),
    "google/gemma-4-E4B-it": ("ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q8_0.gguf", 15, 16),
    "google/gemma-4-E2B-it": ("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf", 12, 16),
    "ibm-granite/granite-4.2-3b": ("ibm-granite/granite-4.2-3b-GGUF", "granite-4.2-3b-Q8_0.gguf", 12, 16),
    "HuggingFaceTB/SmolLM3-3B": ("ggml-org/SmolLM3-3B-GGUF", "SmolLM3-Q8_0.gguf", 12, 16),
    "mistralai/Ministral-3-3B-Instruct-2512": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                                               "Ministral-3-3B-Instruct-2512-Q8_0.gguf", 12, 16),
    "microsoft/Phi-4-mini-instruct": ("bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                                      "microsoft_Phi-4-mini-instruct-Q8_0.gguf", 13, 16),
    "allenai/OLMo-2-0425-1B-Instruct": ("allenai/OLMo-2-0425-1B-Instruct-GGUF", "OLMo-2-0425-1B-Instruct-Q8_0.gguf", 8, 16),
    # references, 2x to 7x bigger
    "Qwen/Qwen3.5-9B": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q8_0.gguf", 18, 16),
    "google/gemma-4-12B-it": ("ggml-org/gemma-4-12B-it-GGUF", "gemma-4-12B-it-Q8_0.gguf", 23, 16),
    "mistralai/Ministral-3-14B-Instruct-2512": ("mistralai/Ministral-3-14B-Instruct-2512-GGUF",
                                                "Ministral-3-14B-Instruct-2512-Q8_0.gguf", 25, 16),
    "Qwen/Qwen3.8-27B": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q8_0.gguf", 37, 8),
}
SUFFIX = "_colab"  # cloud results: a model keeps this machine for dev and test
# Thinking jobs are long generations: each sends THINK_PARALLEL questions at once to its own
# server (continuous batching), which needs a bigger KV cache. Separate processes only share
# the GPU by time slicing, so few jobs run at once and each one batches.
THINK_PARALLEL = 16
THINK_EXTRA_GIB = 7
MAX_JOBS = 3

PLANS = {
    "colab-1": [("mc", m, s) for s in ("dev", "test") for m in ("HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3-4B")]
               + [("think", m, s) for s in ("dev", "test") for m in GGUF],
    "solo-1": [("solo", m, "all") for m in SOLO],
    "solo-27b": [("solo", "Qwen/Qwen3.8-27B", "all")],  # rerun alone (the GPU must be free)
    "think-1": [("think", m, s) for s in ("dev", "test") for m in GGUF],
    # Experiment 2: the 4 experts write together, then the reference answers alone (same questions).
    "gen-1": [("gen", "essaim4", "dev"), ("gen", "essaim4", "test"),
              ("genref", "Qwen/Qwen3-4B", "dev"), ("genref", "Qwen/Qwen3-4B", "test")],
    "smoke": [("mc-smoke", "Qwen/Qwen3-1.7B", "dev")],
}

# E10 (run_sot.py, judge_sot.py): parallel sections. One model writes the skeleton, point i is expanded by
# SOT["peers"][i mod P]; every peer also answers alone (baseline); then the judge compares, alone on the GPU.
SOT = {"tag": "sot1", "outline": "Qwen/Qwen3.5-4B",
       "peers": ["Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-3b",
                 "mistralai/Ministral-3-3B-Instruct-2512", "microsoft/Phi-4-mini-instruct", "HuggingFaceTB/SmolLM3-3B"],
       "judge": "Qwen/Qwen3.8-27B"}
# The judge reads long prompts (question + two answers, up to ~4k tokens): half the slots of its SOLO entry
# with twice the context each, i.e. the same KV cache, hence the same memory reservation.
SOT_JUDGE_PARALLEL, SOT_JUDGE_CTX = SOLO[SOT["judge"]][3] // 2, 6144
PLANS["sot-1"] = ([("sot-base", m, "all") for m in SOT["peers"]] + [("sot-expand", m, "all") for m in SOT["peers"]]
                  + [("sot-judge", SOT["judge"], "all")])
# E11 (run_code.py, exec_code.py): every SOLO model writes code (greedy + 4 samples per problem), then one CPU job
# runs every distinct program in the sandbox (Linux: rlimits, per-case timeouts, no network if unshare works).
CODE_TAG = "e11"
PLANS["code-1"] = [("code", m, "all") for m in SOLO] + [("code-exec", CODE_TAG, "all")]
# E12 (run_aa.py, exec_scicode.py): GPQA Diamond and SciCode, the benchmarks of Artificial Analysis. Stage 0 runs the
# dev reference code through the SciCode harness on the CPU (a broken harness stops the plan before any GPU
# time is spent), stage 1 every SOLO model answers GPQA and writes the SciCode sub-problems, stage 2 grades them.
# Files from the owner, sent by colab/colab_phase0.sh: data/scicode_test_data.h5 (required) and
# data/gpqa_diamond.csv (optional, gated); see essaim/gpqa.py, essaim/scicode.py.
AA_TAG = "e12"
AA_CTX_PER_SLOT = 8192  # run_aa.CTX_PER_SLOT: the same KV cache as E4's slots, in fewer and longer slots
PLANS["aa-1"] = [("aa-oracle", AA_TAG, "all")] + [("aa", m, "all") for m in SOLO] + [("aa-exec", AA_TAG, "all")]
# Jobs run stage by stage: a stage starts once every job of the earlier stages has succeeded.
STAGE = {"sot-base": 0, "sot-expand": 1, "sot-judge": 2, "code": 0, "code-exec": 1, "aa-oracle": 0, "aa": 1, "aa-exec": 2}
NO_MODEL = ("gen", "genref", "code-exec", "aa-oracle", "aa-exec")  # job kinds whose second field is not a model to download
SCICODE_PIP = "scicode @ git+https://github.com/scicode-bench/SciCode@e3158ea011d4235245a547460d3688d7ccbf9900"  # = scicode.OFFICIAL_COMMIT


# Experiment 2 peers, all on this VM (one llama-server each); the coordinator talks to them over HTTP.
GEN_PEERS = {"Qwen/Qwen3-1.7B": 8101, "ibm-granite/granite-3.3-2b-instruct": 8102,
             "HuggingFaceTB/SmolLM3-3B": 8103, "google/gemma-4-E2B-it": 8104}
GEN_REF = {"Qwen/Qwen3-4B": 8105}
GEN_N = {"dev": 100, "test": 200}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def gpu_free_gib() -> float:
    out = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).stdout.strip()
    if not out:
        raise SystemExit("nvidia-smi ne répond pas : pas de GPU")
    return float(out.splitlines()[0]) / 1024


class MemoryBudget:
    """Admit a job only when its reserved memory fits in what is free (measured once at start)."""

    def __init__(self, free_gib: float, margin: float = 2.0):
        self.total = self.left = free_gib - margin
        self.cv = threading.Condition()

    def take(self, gib: float):
        with self.cv:
            if gib > self.total:  # would wait forever
                raise RuntimeError(f"{gib} Gio demandés, {self.total:.0f} Gio utilisables sur ce GPU")
            while gib > self.left:
                if STOPPING.is_set():  # the launcher is stopping: never start, never block its exit
                    raise RuntimeError("arrêt du lanceur")
                self.cv.wait(timeout=1)
            self.left -= gib

    def give(self, gib: float):
        with self.cv:
            self.left += gib
            self.cv.notify_all()


def exclusive(path: Path):
    """Process-wide exclusive lock file (fails if another launcher holds it). The PID is written to a
    temporary file first and linked into place, so the lock never exists empty."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}")
    tmp.write_text(str(os.getpid()))
    try:
        os.link(tmp, path)
    except FileExistsError:
        tmp.unlink()
        pid = path.read_text().strip()
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        if alive:
            raise SystemExit(f"un autre lanceur tourne déjà (pid {pid})")
        path.unlink()  # stale lock from a dead launcher
        return exclusive(path)
    tmp.unlink()


def clear_dead_result_locks():
    """Result locks (pid:token) left by processes that died with a previous launcher on this VM."""
    for lock in (HERE / "results").glob("*.jsonl.lock"):
        pid = lock.read_text().split(":")[0].strip()
        if not (pid.isdigit() and Path(f"/proc/{pid}").exists()):
            print(f"verrou orphelin retiré : {lock.name}", flush=True)
            lock.unlink()


def kill_orphan_servers():
    """llama-server processes adopted by init (their job died with a previous launcher) still hold GPU
    memory and would make the memory budget wrong: stop them before measuring free memory."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if cmd.endswith("llama-server") and ppid == 1:
            print(f"serveur orphelin arrêté : pid {d.name}", flush=True)
            try:
                os.kill(int(d.name), signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(2)


def download(model: str):
    repo, fname = (SOLO[model][:2] if model in SOLO else GGUF[model][:2])
    if (MODELS / fname).exists():
        return
    r = sh([sys.executable, str(HERE / "fastdl.py"), repo, fname, str(MODELS), "16"])
    if r.returncode != 0:
        raise RuntimeError(f"téléchargement {fname} : {r.stdout[-400:]} {r.stderr[-400:]}")


def exec_workers(per_child_gib: int = 4) -> int:
    """Sandbox children in parallel: one per CPU, at most one per `per_child_gib` of available RAM (a hidden-test
    child may use up to 4 GiB, essaim/sandbox.MEM_MB)."""
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 2**20
    except (OSError, StopIteration, ValueError):
        avail = 16
    return max(1, min(os.cpu_count() or 1, int(avail // per_child_gib)))


def job_cmd(kind: str, model: str, split: str) -> list[str]:
    if kind == "code":
        return [sys.executable, "run_code.py", "--model", model, "--gguf", str(MODELS / SOLO[model][1]),
                "--suffix", SUFFIX, "--parallel", str(SOLO[model][3])]
    if kind == "aa":
        benches = aa_benches()
        return [sys.executable, "run_aa.py", "--model", model, "--gguf", str(MODELS / SOLO[model][1]),
                "--suffix", SUFFIX, "--benches", *benches,
                "--parallel", str(max(1, SOLO[model][3] * 3072 // AA_CTX_PER_SLOT))]
    if kind == "aa-oracle":
        return [sys.executable, "exec_scicode.py", "--oracle", "--splits", "dev", "--suffix", SUFFIX,
                "--workers", str(exec_workers())]
    if kind == "aa-exec":
        return [sys.executable, "exec_scicode.py", "--suffix", SUFFIX, "--models", *SOLO, "--workers", str(exec_workers())]
    if kind == "code-exec":
        return [sys.executable, "exec_code.py", "--suffix", SUFFIX, "--tag", model, "--models", *SOLO,
                "--workers", str(exec_workers())]
    if kind == "gen":
        return [sys.executable, "run_gen.py", "--peers", *(f"http://127.0.0.1:{p}" for p in GEN_PEERS.values()),
                "--split", split, "--n", str(GEN_N[split]), "--tag", model + SUFFIX]
    if kind == "genref":
        return [sys.executable, "run_gen.py", "--peers", f"http://127.0.0.1:{GEN_REF[model]}", "--modes", "solo",
                "--split", split, "--n", str(GEN_N[split]), "--tag", "ref-" + model.split("/")[-1] + SUFFIX]
    if kind == "solo":
        return [sys.executable, "run_solo.py", "--model", model, "--gguf", str(MODELS / SOLO[model][1]),
                "--suffix", SUFFIX, "--parallel", str(SOLO[model][3])]
    if kind in ("sot-base", "sot-expand"):
        if kind == "sot-base":  # the outline model also writes the skeletons, in the same process
            stages = (["outline"] if model == SOT["outline"] else []) + ["baseline"]
        else:
            stages = ["expand"]
        return [sys.executable, "run_sot.py", "--model", model, "--gguf", str(MODELS / SOLO[model][1]),
                "--stages", *stages, "--tag", SOT["tag"], "--outline-model", SOT["outline"], "--peers", *SOT["peers"],
                "--suffix", SUFFIX, "--parallel", str(SOLO[model][3])]
    if kind == "sot-judge":
        return [sys.executable, "judge_sot.py", "--judge", model, "--gguf", str(MODELS / SOLO[model][1]),
                "--tag", SOT["tag"], "--outline-model", SOT["outline"], "--peers", *SOT["peers"],
                "--baselines", *SOT["peers"], "--suffix", SUFFIX, "--parallel", str(SOT_JUDGE_PARALLEL),
                "--ctx-per-slot", str(SOT_JUDGE_CTX)]
    gguf = str(MODELS / GGUF[model][1])
    if kind == "mc":
        return [sys.executable, "run_mc.py", "--model", model, "--gguf", gguf, "--suffix", SUFFIX, "--split", split]
    if kind == "mc-smoke":
        return [sys.executable, "run_mc.py", "--model", model, "--gguf", gguf, "--suffix", "_colabsmoke",
                "--split", split, "--n", "40", "--runs", "2"]
    if kind == "think":
        return [sys.executable, "run_think.py", "--model", model, "--gguf", gguf, "--suffix", SUFFIX, "--split", split,
                "--parallel", str(THINK_PARALLEL)]
    raise ValueError(kind)


STATE: dict[str, str] = {}
RUNNING: set[subprocess.Popen] = set()  # job processes in progress, stopped if the launcher is stopped
STOPPING = threading.Event()
RUNNING_LOCK = threading.Lock()


def killgroup(p: subprocess.Popen, sig: int):
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass
STATE_LOCK = threading.Lock()


def set_state(job, value: str):
    with STATE_LOCK:
        STATE[" ".join(job)] = value
        tmp = STATUS.with_name(STATUS.name + ".tmp")
        tmp.write_text(json.dumps(STATE, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATUS)


def start_peers(peers: dict[str, int]) -> list[subprocess.Popen]:
    """Start one essaim.peer per model (localhost only) and wait until each answers /info with its model."""
    procs = []

    def stop():  # each peer leads its own process group, with its llama-server: stop the whole group
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    atexit.register(stop)
    token = os.environ.get("ESSAIM_TOKEN")  # peers inherit it and then require it, even on localhost
    headers = {"X-Essaim-Token": token} if token else {}
    for model, port in peers.items():
        log = open(HERE / "results" / f"colab_peer_{port}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, "-m", "essaim.peer", "--model", model, "--gguf",
                                       str(MODELS / GGUF[model][1]), "--port", str(port)],
                                      cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                                      start_new_session=True))
    for (model, port), p in zip(peers.items(), procs):
        deadline = time.time() + 600
        while True:
            if p.poll() is not None:
                raise SystemExit(f"le pair {model} s'est arrêté, voir results/colab_peer_{port}.log")
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/info", headers=headers)
                with urllib.request.urlopen(req, timeout=5) as r:
                    if json.load(r).get("model") == model:
                        break
            except OSError:
                pass
            if time.time() > deadline:
                raise SystemExit(f"le pair {model} ne répond pas après 10 min")
            time.sleep(3)
        print(f"pair prêt : {model} sur {port}", flush=True)
    return procs


def run_job(job, budget: MemoryBudget) -> bool:
    kind, model, split = job
    if kind in ("gen", "genref", "code-exec", "aa-oracle", "aa-exec"):  # peers' memory already taken by start_peers; the rest: CPU only
        need = 0
    elif kind in ("solo", "code", "aa") or kind.startswith("sot-"):
        need = SOLO[model][2]
    else:
        need = GGUF[model][2] + (THINK_EXTRA_GIB if kind == "think" else 0)
    set_state(job, "en attente de mémoire GPU")
    budget.take(need)
    try:
        if STOPPING.is_set():
            set_state(job, "annulé (arrêt du lanceur)")
            return False
        set_state(job, "en cours")
        t0 = time.time()
        log = HERE / "results" / f"colab_{kind}_{model.replace('/', '__')}_{split}.log"
        with open(log, "w", encoding="utf-8") as f:
            with RUNNING_LOCK:  # a job is either registered before the stop, or never started
                if STOPPING.is_set():
                    set_state(job, "annulé (arrêt du lanceur)")
                    return False
                # Own process group: stopping the job also stops its llama-server.
                p = subprocess.Popen(job_cmd(kind, model, split), cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                                     env=os.environ.copy(), start_new_session=True)
                RUNNING.add(p)
            try:
                code = p.wait()
            finally:
                RUNNING.discard(p)
        ok = code == 0
        set_state(job, f"{'fini' if ok else f'ÉCHEC code {code}'} en {(time.time() - t0) / 60:.1f} min")
        return ok
    finally:
        budget.give(need)


def stage_of(job) -> int:
    return STAGE.get(job[0], 0)


def run_stages(jobs: list[tuple], run, ex: ThreadPoolExecutor, state=None) -> list[bool]:
    """Run the jobs through `ex`, stage by stage (stage_of): the jobs of a stage run in parallel, and only
    once every job of the earlier stages has succeeded; after a failure, the later stages are cancelled.
    Returns one success flag per job (in completion order)."""
    state = state or set_state
    results: list[bool] = []
    for s in sorted({stage_of(j) for j in jobs}):
        group = [j for j in jobs if stage_of(j) == s]
        if not all(results):
            for j in group:
                state(j, "annulé (une étape précédente a échoué)")
            results += [False] * len(group)
            continue
        futs = {ex.submit(run, j): j for j in group}
        for fut in as_completed(futs):
            try:
                results.append(bool(fut.result()))
            except Exception as e:
                state(futs[fut], f"ÉCHEC {type(e).__name__}: {e}")
                results.append(False)
    return results


def aa_benches() -> list[str]:
    """Select E12 benchmarks from the files sent by the PC; never download gated data implicitly."""
    from essaim import gpqa
    path = gpqa.local_path(HERE / "data" / gpqa.FILE)
    if path is None:
        return ["scicode"]
    gpqa.check_file(path.read_bytes())
    return ["gpqa", "scicode"]


def prepare_aa():
    """E12 prerequisites, checked before any job starts: the optional gated GPQA file and the SciCode targets (sent
    by the PC), the SciCode data, and the official scicode package (installed without its dependencies)."""
    from essaim import gpqa, scicode
    if "gpqa" in aa_benches():
        gpqa.diamond()
    for split in ("dev", "test"):
        scicode.problems(split)
    scicode.h5_identity()
    if any("scicode" in m for m in scicode.preflight()):
        r = sh(["uv", "pip", "install", "--python", sys.executable, "--no-deps", SCICODE_PIP])
        if r.returncode != 0:
            raise SystemExit(f"installation du paquet scicode impossible : {r.stderr[-400:]}")
    missing = scicode.preflight()
    if missing:
        raise SystemExit("environnement SciCode incomplet : " + " ; ".join(missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), default="colab-1")
    a = ap.parse_args()
    if not os.environ.get("LLAMA_SERVER"):
        raise SystemExit("LLAMA_SERVER doit pointer vers llama-server (build CUDA)")
    devices = sh([os.environ["LLAMA_SERVER"], "--list-devices"]).stdout
    if "CUDA0" not in devices:  # never measure on the CPU while the manifests say GPU
        raise SystemExit(f"llama.cpp ne voit aucun GPU CUDA :\n{devices}")
    # SIGTERM (e.g. to the launcher's process group) must still run the atexit cleanups: peers live in
    # their own process groups and would otherwise survive the launcher.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    lock = HERE / "results" / "colab_launcher.lock"
    exclusive(lock)
    atexit.register(lambda: lock.unlink(missing_ok=True) if lock.exists() and lock.read_text().strip() == str(os.getpid()) else None)
    clear_dead_result_locks()  # safe: we hold the launcher lock, so no job of ours is running yet
    kill_orphan_servers()
    jobs = PLANS[a.plan]
    gen = any(j[0] in ("gen", "genref") for j in jobs)
    models = list(dict.fromkeys([j[1] for j in jobs if j[0] not in NO_MODEL]
                                + (list(GEN_PEERS) + list(GEN_REF) if gen else [])))
    # Everything shared is prepared before the parallel part: GGUF files, pinned revisions, data caches.
    for m in models:
        download(m)
    sys.path.insert(0, str(HERE))
    from essaim import data
    from essaim.common import resolve_revision
    for m in models:
        resolve_revision(m)
    for split in data.SPLITS:
        data.arc(split=split), data.mmlu_pro(split=split), data.gsm8k(split=split), data.math500(split=split)
    if any(j[0].startswith("sot-") for j in jobs):
        for split in data.MT_SPLITS:
            data.mt_bench(split=split)
    if any(j[0].startswith("code") for j in jobs):
        for split in data.SPLITS:
            for load in data.CODE_LOADERS.values():
                load(split=split)
    if any(j[0].startswith("aa") for j in jobs):
        prepare_aa()
    free = gpu_free_gib()
    budget = MemoryBudget(free)
    print(f"GPU : {free:.0f} Gio libres, {len(jobs)} jobs", flush=True)
    for j in jobs:
        set_state(j, "prévu")
    if gen:  # timed runs: one at a time, against peers that only serve them
        start_peers({**GEN_PEERS, **GEN_REF})
    ex = ThreadPoolExecutor(max_workers=1 if gen else MAX_JOBS)
    try:
        results = run_stages(jobs, lambda j: run_job(j, budget), ex)
    except BaseException:  # stopped (SIGTERM, Ctrl-C): no new job, stop the running ones, then exit
        with RUNNING_LOCK:
            STOPPING.set()
            running = list(RUNNING)
        ex.shutdown(wait=False, cancel_futures=True)
        for p in running:
            killgroup(p, signal.SIGTERM)
        for p in running:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            killgroup(p, signal.SIGKILL)  # whatever is left of the group (its llama-server)
        raise  # atexit then stops the peers
    ex.shutdown()
    print(json.dumps(STATE, indent=1, ensure_ascii=False), flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
