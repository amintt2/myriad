"""Read structured campaign state on the VM, only through colab_phase0.sh snapshot."""
import json
import re
from pathlib import Path


def safe_snapshot(value):
    """Expose state categories only, never remote log text or exception messages."""
    if not isinstance(value, dict):
        return {"invalid_snapshot": True}
    boot = value.get("bootstrap")
    if isinstance(boot, dict):
        boot = {k: boot[k] for k in ("step", "plan", "pid", "campaign_id") if k in boot}
        step = boot.get("step", "")
        if isinstance(step, str) and step.startswith(("ÉCHEC", "refusé")):
            boot["step"] = step.split()[0]
        elif step != "lanceur en marche":
            boot["step"] = "unknown bootstrap state"
    jobs = value.get("jobs")
    if isinstance(jobs, dict):
        clean = {}
        for key, state in jobs.items():
            if isinstance(state, str) and state.startswith("ÉCHEC"):
                match = re.match(r"ÉCHEC code [0-9]+", state)
                state = match.group() if match else "ÉCHEC"
            elif isinstance(state, str) and state.startswith("annulé"):
                state = "annulé"
            elif state not in ("prévu", "en attente de mémoire GPU", "en cours") and not (
                    isinstance(state, str) and re.fullmatch(r"fini en [0-9]+\.[0-9]+ min", state)):
                state = "unknown job state"
            clean[key] = state
        jobs = clean
    diag = value.get("diagnostics")
    # Locally constructed diagnostics have no raw messages; avoid trusting arbitrary extra CLI fields.
    diag = {"launcher_alive": diag.get("launcher_alive")} if isinstance(diag, dict) else None
    return {"schema": value.get("schema"), "bootstrap": boot, "jobs": jobs,
            "campaign_present": value.get("campaign_present"), "diagnostics": diag}


def snapshot(root=Path("/content/dllm")):
    results = root / "phase0" / "results"

    def read(name):
        path = results / name
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None

    def exists(name):
        try:
            (results / name).stat()
            return True
        except FileNotFoundError:
            return False

    boot = read("colab_bootstrap_status.json")
    pid = boot.get("pid") if isinstance(boot, dict) else None
    alive = False
    if type(pid) is int and pid > 0:
        try:
            alive = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X", "x")
        except FileNotFoundError:
            pass
    models = [{"name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
              for p in sorted((root / "models").glob("*")) if p.suffix in (".gguf", ".part")]
    errors = {}
    for p in sorted(results.glob("colab_*.log")):
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        errors[p.name] = re.findall(r"(?m)^((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*):", tail)
    return {"schema": 1, "bootstrap": boot, "jobs": read("colab_status.json"),
            "campaign_present": any(exists(n) for n in
                                    ("colab_launcher.lock", "colab_bootstrap_status.json", "colab_status.json")),
            "diagnostics": {"launcher_alive": alive, "models": models, "exceptions": errors}}


if __name__ == "__main__":
    print(json.dumps(safe_snapshot(snapshot()), ensure_ascii=False))
