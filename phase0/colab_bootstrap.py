"""Executed on the Colab VM by `colab exec --timeout 1800 -f colab_bootstrap.py` (colab/colab_phase0.sh up).

1. Refuses to touch anything while a launcher is running (code must not change under a campaign).
2. Unpacks the uploaded code, restores local checkpoints of cloud results that the VM lacks.
3. Installs llama.cpp b11505 (official CUDA 12.8 build, same version as the PC) and uv, syncs Python.
4. Starts colab_jobs.py in the background and checks that it is alive.
Each step is recorded in results/colab_bootstrap_status.json. Running it again resumes.
"""
import glob
import json
import os
import shutil
import subprocess
import tarfile
import time

WORK = "/content/dllm"
LLAMA = "/content/llama"
BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11505"
RESULTS = f"{WORK}/phase0/results"
STATUS = f"{RESULTS}/colab_bootstrap_status.json"
LOCK = f"{RESULTS}/colab_launcher.lock"
CAMPAIGN_ID = None


def status(step, **kw):
    os.makedirs(RESULTS, exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"step": step, "time": time.strftime("%H:%M:%S"), "campaign_id": CAMPAIGN_ID, **kw}, f)
    os.replace(tmp, STATUS)
    print(step, kw or "", flush=True)


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def launcher_alive():
    if not os.path.exists(LOCK):
        return None
    pid = open(LOCK).read().strip()
    return pid if pid.isdigit() and os.path.exists(f"/proc/{pid}") else None


def unpack_code(archive):
    with tarfile.open(archive) as t:
        has_gpqa = "phase0/data/gpqa_diamond.csv" in t.getnames()
        t.extractall(WORK, filter="data")
    with open(f"{WORK}/phase0/colab_plan.txt") as f:
        plan = f.read().strip()
    if plan.startswith("aa-") and not has_gpqa:
        try:
            os.remove(f"{WORK}/phase0/data/gpqa_diamond.csv")  # a reused VM must match the new archive
        except FileNotFoundError:
            pass
    return plan


try:
    pid = launcher_alive()
    if pid:
        print("refusé : un lanceur tourne déjà", flush=True)
        raise SystemExit(73)

    with tarfile.open("/content/dllm.tgz") as archive:
        try:
            CAMPAIGN_ID = archive.extractfile("phase0/colab_campaign_id.txt").read().decode().strip()
        except KeyError:
            pass  # historical campaigns have no supervisor identity

    status("décompression du code")
    PLAN = unpack_code("/content/dllm.tgz")
    restored = []
    for src in glob.glob(f"{WORK}/checkpoints/*.jsonl"):
        if src.endswith(".timing.jsonl"):
            continue  # timing observations are restored alongside their scientific result file
        dst = f"{RESULTS}/{os.path.basename(src)}"
        if not os.path.exists(dst):  # the VM's own copy, if any, is newer: never overwrite it
            shutil.copy2(src, dst)
            shutil.copy2(src + ".meta.json", dst + ".meta.json")
            if os.path.exists(src + ".timing.jsonl"):
                shutil.copy2(src + ".timing.jsonl", dst + ".timing.jsonl")
            restored.append(os.path.basename(src))
    status("points de reprise restaurés", fichiers=restored)

    if not os.path.exists(f"{LLAMA}/.ok"):
        status("installation de llama.cpp b11505 CUDA")
        os.makedirs(LLAMA, exist_ok=True)
        for name in ("llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz"):
            run(["curl", "-sSfL", "-o", f"{LLAMA}/{name}", f"{BASE}/{name}"])
            run(["tar", "xzf", f"{LLAMA}/{name}", "-C", LLAMA])
            os.remove(f"{LLAMA}/{name}")
        open(f"{LLAMA}/.ok", "w").close()
    server = sorted(glob.glob(f"{LLAMA}/**/llama-server", recursive=True))[0]
    libdirs = sorted({os.path.dirname(server)} | {os.path.dirname(p) for p in glob.glob(f"{LLAMA}/**/libcudart*", recursive=True)})
    # The NVIDIA driver library (libcuda.so.1) lives in /usr/lib64-nvidia on Colab; without it llama.cpp
    # silently falls back to the CPU.
    driver = sorted({os.path.dirname(p) for p in glob.glob("/usr/lib64-nvidia/libcuda.so.1")})
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env["PATH"]
    env["LLAMA_SERVER"] = server
    env["LD_LIBRARY_PATH"] = ":".join(libdirs + driver) + ":" + env.get("LD_LIBRARY_PATH", "")
    devices = subprocess.run([server, "--list-devices"], env=env, capture_output=True, text=True).stdout
    if "CUDA0" not in devices:
        status("ÉCHEC : llama.cpp ne voit pas le GPU", devices=devices[-500:])
        raise SystemExit(1)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    v = subprocess.run([server, "--version"], env=env, capture_output=True, text=True)
    version = [l for l in (v.stdout + v.stderr).splitlines() if "version" in l]

    status("environnement Python (uv)")
    if shutil.which("uv", path=env["PATH"]) is None:
        run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, env=env)
    groups = ["--group", "scicode"] if PLAN.startswith("aa-") else []  # E12: h5py, scipy, sympy for the SciCode tests
    run(["uv", "sync", "-q", *groups], cwd=f"{WORK}/phase0", env=env)

    log = open(f"{RESULTS}/colab_launcher.log", "a")
    p = subprocess.Popen(["uv", "run", *groups, "python", "colab_jobs.py", "--plan", PLAN], cwd=f"{WORK}/phase0", env=env,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(20)
    if p.poll() is not None:
        status("ÉCHEC : le lanceur s'est arrêté", code=p.returncode, log=open(f"{RESULTS}/colab_launcher.log").read()[-1500:])
        raise SystemExit(1)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    status("lanceur en marche", plan=PLAN, pid=p.pid, llama=version, gpu=gpu)
except SystemExit:
    raise
except Exception as e:
    status("ÉCHEC", erreur=f"{type(e).__name__}: {e}"[:800])
    raise
