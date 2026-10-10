"""Strict CLI adapters and persistent allocation ownership; never infer absence from an empty read."""
import contextlib
import hashlib
import io
import json
import math
import os
import re
import signal
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path


def parse(text):
    lines = text.strip().splitlines()
    if lines == ["[colab] No active sessions found on server."]:
        return {"schema": 1, "phase0": None}
    found = None
    if not lines:
        raise ValueError("empty sessions response")
    for line in lines:
        m = re.fullmatch(r"\[([^\]]+)\] (\S+) \| Hardware: (\S+) \| Shape: (.+) \| Variant: (\S+)", line)
        if not m:
            raise ValueError("unrecognized sessions response")
        if m[1] == "phase0":
            if found is not None:
                raise ValueError("duplicate phase0 session")
            found = {"fingerprint": hashlib.sha256(m[2].encode()).hexdigest(), "hardware": m[3]}
        if m[1] == "?":
            raise ValueError("untracked allocation: manual review required")
    return {"schema": 1, "phase0": found}


# Lifecycle state belongs to this wrapper, never to the installed CLI's credential store.
HERE = Path(__file__).resolve().parent


def receipt_path():
    return Path(os.environ.get("DLLM_CAMPAIGN_RECEIPT") or
                str(Path.home() / ".local/state/myriad-colab/ownership.json"))


@contextlib.contextmanager
def lifecycle_lock(path=None):
    import fcntl
    path = path or receipt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get("DLLM_LIFECYCLE_LOCK_FD")
    lock_path = Path(str(path) + ".lock")
    with (os.fdopen(os.dup(int(inherited)), "a") if inherited else lock_path.open("a")) as lock:
        held, expected = os.fstat(lock.fileno()), lock_path.stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise PermissionError("inherited lifecycle lock differs from receipt lock")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PermissionError("another lifecycle operation holds the lock")
        yield lock


def identity(saved):
    return {k: saved.get(k) for k in ("allocation_id", "campaign_id", "phase0")}


def owned_receipt():
    saved = json.loads(receipt_path().read_text(encoding="utf-8"))
    if saved.get("status") != "owned" or not isinstance(saved.get("phase0"), dict):
        raise PermissionError("ownership unconfirmed or foreign campaign")
    pinned = os.environ.get("DLLM_OPERATION_IDENTITY")
    if pinned is not None and identity(saved) != json.loads(pinned):
        raise PermissionError("operation receipt replaced")
    campaign = os.environ.get("DLLM_CAMPAIGN_ID")
    if campaign and saved.get("campaign_id") != campaign:
        raise PermissionError("operation campaign differs from receipt")
    return saved


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def invoke(args, seconds=None):
    from safe_cli import run
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run(args, seconds=max(.01, seconds) if seconds is not None else None)
    sys.stderr.write(err.getvalue())
    if code:
        sys.stderr.write(out.getvalue())
        raise RuntimeError(f"colab {args[0]}: exit {code}")
    return out.getvalue()


def usage(text):
    m = re.fullmatch(r"Current balance: ([0-9]+\.[0-9]{2}) compute units\n"
                     r"Usage rate: ([0-9]+\.[0-9]{2})/hr\nActive assignments: ([0-9]+)", text.strip())
    if not m:
        raise ValueError("unreadable compute-unit balance")
    balance, rate = float(m[1]), float(m[2])
    if not all(math.isfinite(v) for v in (balance, rate)):
        raise ValueError("nonfinite compute-unit usage")
    return {"schema": 1, "balance_units": balance, "rate_units_hour": rate, "assignments": int(m[3])}


def usage_record(path, before, after, error=None, campaign_id=None, release_confirmed=False):
    record = {"schema": 1, "time": time.time(), "scope": "account", "campaign_id": campaign_id,
              "before": before, "after": after, "error": error, "release_confirmed": release_confirmed,
              "balance_decrease_units": round(before["balance_units"] - after["balance_units"], 2)
              if before is not None and after is not None else None,
              "campaign_consumption_units": None}
    journal = Path(str(path) + ".usage.jsonl")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(record), flush=True)


def capacity(plan, gpu):
    sys.path.insert(0, str(HERE.parent))
    from colab_jobs import PLANS, SOLO, GGUF, THINK_EXTRA_GIB, GEN_PEERS, GEN_REF
    needs = []
    for kind, model, _ in PLANS[plan]:
        if kind in ("gen", "genref"):
            needs.append(sum(GGUF[m][2] for m in {**GEN_PEERS, **GEN_REF}))
        elif kind in ("code-exec", "aa-oracle", "aa-exec"):
            continue
        elif kind in ("solo", "code", "aa") or kind.startswith("sot-"):
            needs.append(SOLO[model][2])
        else:
            needs.append(GGUF[model][2] + (THINK_EXTRA_GIB if kind == "think" else 0))
    required = max(needs, default=0) + 2
    limits = {"T4": 15, "L4": 22, "A100": 40, "H100": 80}
    if gpu not in limits or required > limits[gpu]:
        raise ValueError(f"{plan} requires {required} GiB; {gpu} capacity refused")
    return required


def select_gpu(plan):
    for gpu in ("L4", "A100", "H100"):
        try:
            capacity(plan, gpu)
            return gpu
        except ValueError:
            pass
    raise ValueError("no supported GPU fits the unchanged plan")


def families(plan):
    sys.path.insert(0, str(HERE.parent))
    from colab_jobs import PLANS
    names = {"code-exec": "codeexec", "aa-exec": "sciexec", "aa-oracle": "sciexec", "genref": "gen",
             "think": "mc"}
    return sorted({names.get(kind, kind.split("-", 1)[0]) for kind, _, _ in PLANS[plan]})


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


def archive(source, plan, bootstrap):
    path = receipt_path()
    saved = owned_receipt()
    target = Path(str(path) + ".archive.tgz")
    boot = Path(str(target) + ".colab_bootstrap.py")
    for source_file, destination in ((Path(source), target), (Path(bootstrap), boot)):
        temporary = Path(str(destination) + ".tmp")
        shutil.copyfile(source_file, temporary)
        with temporary.open("rb") as content:
            os.fsync(content.fileno())
        temporary.replace(destination)
    save(path, {**saved, "plan": plan, "archive_sha256": digest(target), "bootstrap_sha256": digest(boot)})
    print(target)


def resume(plan, gpu):
    path = receipt_path()
    saved = owned_receipt()
    if saved.get("status") != "owned" or saved.get("plan") != plan:
        raise PermissionError("resume requires the original owned plan")
    if saved["phase0"]["hardware"] != gpu or parse(invoke(["sessions"]))["phase0"] != saved["phase0"]:
        raise PermissionError("resume session absent or replaced")
    target = Path(str(path) + ".archive.tgz")
    boot = Path(str(target) + ".colab_bootstrap.py")
    if digest(target) != saved["archive_sha256"] or digest(boot) != saved["bootstrap_sha256"]:
        raise ValueError("persisted archive or bootstrap changed")
    print(target)


def allocate(plan, gpu):
    from budget import limits, preflight
    path = receipt_path()
    if path.exists():
        raise ValueError(f"persistent ownership exists: run down first ({path})")
    required = capacity(plan, gpu)
    budget = float(os.environ.get("DLLM_BUDGET_UNITS", "nan"))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("explicit positive DLLM_BUDGET_UNITS required")
    if parse(invoke(["sessions"]))["phase0"] is not None:
        raise PermissionError("existing session refused")
    before = usage(invoke(["usage"]))
    campaign_id = os.environ.get("DLLM_CAMPAIGN_ID")
    usage_record(path, before, None, campaign_id=campaign_id)
    policy = limits()
    preflight(before, policy)
    pending = {"schema": 1, "status": "allocation-unconfirmed", "phase0": None,
               "campaign_id": campaign_id, "allocation_id": uuid.uuid4().hex, "plan": plan,
               "usage_before": before, "budget_units": budget, "budget_policy": policy}
    save(path, pending)  # Survives a killed process, a successful new with lost output, or a failed receipt read.
    invoke(["new", "-s", "phase0", "--gpu", gpu])  # Never retry an ambiguous allocation.
    pending["allocation_returned"] = True
    save(path, pending)
    # Only a confirmed new permits these bounded identity reads; new itself is never replayed.
    for attempt in range(3):
        try:
            owner = parse(invoke(["sessions"]))["phase0"]
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(min(8, float(os.environ.get("DLLM_RETRY_SECONDS", "1")) * 2 ** attempt))
    if owner is None or owner["hardware"] != gpu:
        raise ValueError("allocation identity unconfirmed")
    save(path, {**pending, "status": "owned", "phase0": owner})
    after = usage(invoke(["usage"]))
    usage_record(path, before, after, campaign_id=campaign_id)
    preflight(after, policy, allocated=True)
    script = "import subprocess, json\n" + (
        "r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], "
        "capture_output=True, text=True, check=True, timeout=10)\n"
        "print('DLLM_CAPACITY ' + json.dumps({'free_mib': float(r.stdout.strip())}))\n")
    with tempfile.NamedTemporaryFile(mode="w", suffix="capacity.py", encoding="utf-8") as output:
        output.write(script)
        output.flush()
        raw = invoke(["exec", "-s", "phase0", "--timeout", "30", "-f", output.name]).strip()
    if not raw.startswith("DLLM_CAPACITY "):
        raise ValueError("missing capacity acknowledgement")
    measured = json.loads(raw[len("DLLM_CAPACITY "):])
    if (not isinstance(measured, dict) or set(measured) != {"free_mib"}
            or type(measured["free_mib"]) not in (int, float) or not math.isfinite(measured["free_mib"])
            or measured["free_mib"] < required * 1024):
        raise ValueError("measured GPU memory insufficient")
    print(f"GPU capacity verified: {required} GiB required", flush=True)


def down():
    path = receipt_path()
    if not path.exists():
        raise PermissionError("no persistent ownership: down refused")
    saved = json.loads(path.read_text(encoding="utf-8"))
    pinned = os.environ.get("DLLM_OPERATION_IDENTITY")
    campaign = os.environ.get("DLLM_CAMPAIGN_ID")
    if campaign is not None and saved.get("campaign_id") != campaign:
        raise PermissionError("cleanup campaign differs from expected campaign: no operation authorized")
    if os.environ.get("DLLM_SUPERVISED") == "1" and not pinned and not campaign:
        raise PermissionError("supervised cleanup requires an expected campaign or explicit pinned adoption")
    if saved.get("status") == "foreign-campaign" or (pinned and identity(saved) != json.loads(pinned)):
        raise PermissionError("foreign or replaced cleanup receipt: no operation authorized")
    owner = saved.get("phase0")
    seconds = float(os.environ.get("DLLM_CLEANUP_SECONDS", "1800"))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid cleanup deadline")
    deadline = time.monotonic() + seconds
    attempt = 0
    if saved.get("status") == "released/accounting-pending":
        if saved.get("release_confirmed") is not True:
            raise ValueError("accounting-only receipt lacks explicit release confirmation")
        return accounting(path, saved, deadline)
    while time.monotonic() < deadline:
        try:
            current = parse(invoke(["sessions"], min(120, deadline - time.monotonic())))["phase0"]
            if json.loads(path.read_text(encoding="utf-8")) != saved:
                raise PermissionError("cleanup receipt replaced: no operation authorized")
            if current is None:
                print("owned allocation released: explicit absence confirmed", flush=True)
                saved = {**saved, "status": "released/accounting-pending", "release_confirmed": True}
                save(path, saved)
                return accounting(path, saved, deadline)
            if saved.get("status") != "owned" or owner is None:
                raise PermissionError("allocation ambiguous: identity reconciliation required; no stop authorized")
            if current != owner:
                raise PermissionError("session replaced: no stop authorized; persistent receipt retained")
            try:
                invoke(["stop", "-s", "phase0"], min(120, deadline - time.monotonic()))
            except RuntimeError as exc:
                print(f"stop ambiguous: {exc}; checking absence again", flush=True)
        except (ValueError, RuntimeError) as exc:
            print(f"CLEANUP PENDING: {exc}; ownership retained at {path}", flush=True)
        attempt += 1
        time.sleep(min(float(os.environ.get("DLLM_RETRY_SECONDS", "1")) * 2 ** min(attempt - 1, 3),
                       max(0, deadline - time.monotonic())))
    raise TimeoutError(f"CLEANUP PENDING: VM may still be billed; run down again; ownership retained at {path}")


def accounting(path, saved, deadline):
    for attempt in range(3):
        try:
            if time.monotonic() >= deadline:
                raise RuntimeError("accounting deadline exceeded")
            after = usage(invoke(["usage"], min(120, deadline - time.monotonic())))
            if json.loads(path.read_text(encoding="utf-8")) != saved:
                raise PermissionError("accounting receipt replaced: no deletion authorized")
            usage_record(path, saved.get("usage_before"), after, campaign_id=saved.get("campaign_id"),
                         release_confirmed=True)
            path.unlink()
            for suffix in (".archive.tgz", ".archive.tgz.colab_bootstrap.py"):
                Path(str(path) + suffix).unlink(missing_ok=True)
            return
        except (ValueError, RuntimeError) as exc:
            usage_record(path, saved.get("usage_before"), None, str(exc), saved.get("campaign_id"), True)
            if attempt < 2:
                time.sleep(min(float(os.environ.get("DLLM_RETRY_SECONDS", "1")) * 2 ** attempt,
                               max(0, deadline - time.monotonic())))
    raise TimeoutError(f"ACCOUNTING PENDING: release confirmed; retry down for usage only; receipt retained at {path}")


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if len(sys.argv) == 1:
            print(json.dumps(parse(sys.stdin.read())))
        elif sys.argv[1] == "usage":
            print(json.dumps(usage(invoke(["usage"]))))
        elif sys.argv[1] == "select-gpu":
            print(select_gpu(sys.argv[2]))
        elif sys.argv[1] == "identity":
            print(json.dumps(identity(owned_receipt())))
        elif sys.argv[1] == "plan":
            print(owned_receipt()["plan"])
        elif sys.argv[1] == "families":
            print(" " + " ".join(families(sys.argv[2])) + " ")
        elif sys.argv[1] in ("allocate", "down", "archive", "resume"):
            with lifecycle_lock():
                if sys.argv[1] == "allocate":
                    allocate(*sys.argv[2:])
                elif sys.argv[1] == "down":
                    down()
                elif sys.argv[1] == "archive":
                    archive(*sys.argv[2:])
                else:
                    resume(*sys.argv[2:])
        elif sys.argv[1] == "refuse":
            path = receipt_path()
            saved = owned_receipt()
            save(path, {**saved, "status": "foreign-campaign"})
        elif sys.argv[1] == "guard":
            snapshot = json.load(sys.stdin)
            if (not isinstance(snapshot, dict) or type(snapshot.get("schema")) is not int
                    or snapshot["schema"] != 1 or type(snapshot.get("campaign_present")) is not bool):
                raise ValueError("invalid preliminary snapshot")
            if snapshot["campaign_present"]:
                path = receipt_path()
                saved = owned_receipt()
                boot = snapshot.get("bootstrap")
                if (saved.get("campaign_id") and isinstance(boot, dict)
                        and boot.get("campaign_id") == saved["campaign_id"]):
                    raise PermissionError("owned campaign already present: resume refused without mutation")
                save(path, {**saved, "status": "foreign-campaign"})
                raise PermissionError("preexisting campaign: no mutation authorized")
        else:
            raise ValueError("unknown lifecycle command")
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 73
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 124
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"lifecycle refused: {exc}; unresolved ownership retained at {receipt_path()}", file=sys.stderr)
        return 65
    return 0


if __name__ == "__main__":
    sys.exit(main())
