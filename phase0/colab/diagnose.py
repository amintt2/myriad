"""Read-only Colab diagnostics, executed only by colab_phase0.sh diagnose.

Inspect the bootstrap PID and its descendants without reading cmdline or environ. A .part file is
preallocated by fastdl: its size alone is not download progress; compare mtime and process I/O over time.
Log contents may contain signed URLs or credentials: report metadata and exception types only.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def file_state(path):
    try:
        st = path.stat()
        return {"name": path.name, "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    except OSError:
        return {"name": path.name, "status": "absent ou inaccessible"}


def processes(proc):
    out = {}
    for path in proc.glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()  # comm may contain spaces or parentheses
            pid = int(path.parent.name)
            out[pid] = {"pid": pid, "ppid": int(fields[1]), "state": fields[0],
                        "alive": fields[0] not in ("Z", "X", "x"),
                        "cpu_ticks": int(fields[11]) + int(fields[12])}
            io = path.parent / "io"
            if io.exists():
                for line in io.read_text().splitlines():
                    key, value = line.split(":", 1)
                    if key in ("rchar", "wchar", "read_bytes", "write_bytes"):
                        out[pid][key] = int(value)
        except (OSError, ValueError, IndexError):  # a process can exit during the snapshot
            continue
    return out


def diagnose(root=Path("/content/dllm"), proc=Path("/proc")):
    results = root / "phase0" / "results"
    try:
        boot = json.loads((results / "colab_bootstrap_status.json").read_text())
        pid = boot.get("pid") if isinstance(boot, dict) else None
        if isinstance(pid, str) and re.fullmatch(r"[0-9]+", pid):
            pid = int(pid)  # a refused bootstrap records launcher_alive()'s PID as a string
        pid = pid if type(pid) is int and pid > 0 else None
    except (OSError, ValueError):
        pid = None
    ps = processes(proc)
    launcher = ps.get(pid, {"pid": pid, "alive": False, "status": "absent ou non enregistré"})
    descendants, seen = [], {pid}
    while True:
        children = [p for p in ps.values() if p["ppid"] in seen and p["pid"] not in seen]
        if not children:
            break
        descendants.extend(children)
        seen.update(p["pid"] for p in children)
    models = sorted([*root.joinpath("models").glob("*.gguf"), *root.joinpath("models").glob("*.part")])
    logs = sorted(results.glob("colab_*.log"), key=lambda p: file_state(p).get("mtime", ""), reverse=True)[:5]
    log_states = []
    for path in logs:
        info = file_state(path)
        try:
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 8192))
                tail = f.read(8192).decode("utf-8", errors="replace")
            info["exceptions"] = re.findall(r"(?m)^((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*):", tail)
        except OSError:
            pass
        log_states.append(info)
    mem = {}
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapFree"):
                mem[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(root)
    return {"time": datetime.now(timezone.utc).isoformat(), "launcher": launcher, "descendants": descendants,
            "models": [file_state(p) for p in models], "disk_free_bytes": disk.free, "memory_bytes": mem,
            "status_files": [file_state(results / name) for name in
                             ("colab_bootstrap_status.json", "colab_status.json")], "logs": log_states}


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=1, ensure_ascii=False))
