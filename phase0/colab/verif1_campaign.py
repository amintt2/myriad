"""Single-campaign official CLI orchestration. Dry-run is offline; cloud actions are explicit."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tarfile
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from verif1_protocol import Ledger, canonical, cost, exclusive, file_sha, sha, utc, write_new
from session_json import parse as parse_sessions
from read_timeout import run as supervise

# These deadlines cover local connection setup and every descendant, independently of remote --timeout.
CLI_SECONDS = {"sessions": 45, "new": 180, "stop": 90, "upload": 600, "download": 600}
TRANSFER_SECONDS = 1800
CONTROL = None


class ProcessFailure(RuntimeError):
    def __init__(self, stdout, stderr):
        super().__init__(stdout + stderr)
        self.stdout = stdout


class ProcessTimeout(TimeoutError):
    def __init__(self, seconds, stdout):
        super().__init__(f"local process deadline exceeded ({seconds}s); descendants stopped")
        self.stdout = stdout


def process(argv, seconds, script=None):
    if sys.platform != "linux":
        raise RuntimeError("cloud control requires Linux/WSL process-group supervision")
    env = dict(os.environ, PATH=str(Path.home() / ".local/bin") + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        source.write((script or "").encode("utf-8"))
        source.seek(0)
        code = supervise(argv, seconds=seconds, stdin=source, stdout=stdout, stderr=stderr, env=env,
                         monitor=CONTROL.monitor if CONTROL is not None else None,
                         deadline=min(CONTROL.deadline, CONTROL.operation_deadline) if CONTROL is not None else None)
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")
        if code == 124:
            raise ProcessTimeout(seconds, out)
        if code:
            raise ProcessFailure(out, err)  # safe_cli.py sanitizes every official CLI stream.
        return out


def cli(args, script=None):
    seconds = (int(args[args.index("--timeout") + 1]) + 60 if args[0] == "exec"
               else CLI_SECONDS[args[0]])
    return process([sys.executable, str(HERE / "safe_cli.py"), *args], seconds, script)


def sessions():
    raw = cli(["sessions"])
    parsed = parse_sessions(raw)
    empty = raw.strip() == "[colab] No active sessions found on server."
    return parsed, empty, raw.strip().splitlines()


def owned(receipt):
    record = json.loads(receipt.read_text(encoding="utf-8"))
    parsed, _, lines = sessions()
    if (len(lines) != 1 or parsed["phase0"] != record["session"]
            or record["session"] is None or record["campaign"] != "VERIF-1"):
        raise ValueError("session ownership differs; never mutate another campaign")
    return record


def billing(record):
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(record["utc"])).total_seconds()
    return {"receipt_to_now_hours": max(0, seconds) / 3600, "hourly_rate": record["hourly_rate"],
            "estimated_monetary": (max(0, seconds) / 3600 * record["hourly_rate"]
                                   if record["hourly_rate"] > 0 else None),
            "limit": "receipt created after allocation; excludes prior allocation seconds; price is supplied"}


@contextmanager
def cleanup_mutex(path):
    """Shared OS lock, released by the kernel on process death; never unlink the shared lock inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def launch(scope, archive_hash, bootstrap_hash):
    """Generate an uploadable launcher for the fixed bootstrap, never for candidate source."""
    publication = "\n".join([
        "def publish(value):",
        "    temporary=None",
        "    try:",
        "        with tempfile.NamedTemporaryFile(mode='w',dir=state.parent,delete=False,encoding='utf-8') as output:",
        "            temporary=output.name",
        "            json.dump(value,output)",
        "            output.flush(); os.fsync(output.fileno())",
        "        os.replace(temporary,state)",
        "        temporary=None",
        "        if os.name!='nt':",
        "            directory=os.open(state.parent,os.O_RDONLY)",
        "            try: os.fsync(directory)",
        "            finally: os.close(directory)",
        "    finally:",
        "        if temporary is not None: Path(temporary).unlink(missing_ok=True)",
    ])
    return "\n".join([
        "import hashlib,json,os,subprocess,sys,tempfile,time",
        "from pathlib import Path",
        "work=Path('/content/verif-1')",
        "bootstrap=Path('/content/verif1_bootstrap.py')",
        "archive=Path('/content/dllm.tgz')",
        "def digest(p):",
        "    h=hashlib.sha256()",
        "    with p.open('rb') as f:",
        "        for b in iter(lambda:f.read(1048576),b''): h.update(b)",
        "    return h.hexdigest()",
        f"assert digest(archive)=={archive_hash!r} and digest(bootstrap)=={bootstrap_hash!r}",
        # Bootstrap expects an empty directory initially, so state/logs live outside the campaign until extraction.
        "state=Path('/content/verif1_launch.json')",
        "if state.exists():",
        "    previous=json.loads(state.read_text())",
        "    assert previous.get('complete') is True, 'previous campaign still running or launch interrupted'",
        f"    assert previous['archive_sha256']=={archive_hash!r}",
        f"    assert previous['scope']=='pilot' and {scope!r}=='full', 'no hidden relaunch'",
        "    assert previous.get('exit_code')==0, 'pilot failed; no extension'",
        f"record={{'scope':{scope!r},'archive_sha256':{archive_hash!r},'complete':False,'started':time.time()}}",
        publication,
        "publish(record)",
        "argv=[sys.executable,str(bootstrap),'--archive',str(archive),'--scope',record['scope'],'--run']",
        f"publication={publication!r}",
        "child = ('import json,os,subprocess,tempfile,time; from pathlib import Path\\n'",
        "         \"state=Path('/content/verif1_launch.json')\\n\"+publication+'\\nrecord='+repr(record)+'\\n'",
        "         \"record.update(pid=os.getpid()); publish(record)\\n\"",
        "         'code=subprocess.call('+repr(argv)+'); '",
        "         \"record.update(complete=True,exit_code=code,ended=time.time()); publish(record)\")",
        "with open('/content/verif1_bootstrap.log','ab',buffering=0) as log:",
        "    p=subprocess.Popen([sys.executable,'-c',child],stdout=log,stderr=log,start_new_session=True)",
        "for _ in range(50):",
        "    current=json.loads(state.read_text())",
        "    if current.get('pid')==p.pid or current.get('complete') is True: break",
        "    time.sleep(.1)",
        "assert current.get('pid')==p.pid, 'writer did not acknowledge startup'",
        "print(json.dumps({'launched_pid':p.pid,'scope':record['scope'],'archive_sha256':record['archive_sha256']}))",
    ]) + "\n"


def validate_archive(path):
    # Require a prepared archive, including exactly its frozen sources and no hidden datasets.
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if any(not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts for m in members):
            raise ValueError("unsafe package")
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive paths")
        plan = json.load(archive.extractfile("verif1_dev_plan.json"))
        expected = set(plan["sources"]) | {"verif1_dev_bundle.json", "verif1_dev_plan.json"}
        if set(names) != expected:
            raise ValueError("package is not the public prepared projection")
        import hashlib
        for name, digest in plan["sources"].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError("package frozen source differs")
        if file_sha(HERE / "verif1_bootstrap.py") != plan["sources"]["phase0/colab/verif1_bootstrap.py"]:
            raise ValueError("local bootstrap differs from package")
    from verif1_prepare import source_hashes
    if plan["sources"] != source_hashes():
        raise ValueError("local operational sources differ from the prepared package")
    return plan


def status_script(record):
    """Read only the launch marker and a bounded bootstrap tail, never campaign results."""
    return "\n".join([
        "import json,os",
        "from pathlib import Path",
        "state=Path('/content/verif1_launch.json')",
        "launch=json.loads(state.read_text()) if state.exists() else None",
        "if launch is not None:",
        f"    assert launch['archive_sha256']=={record['archive_sha256']!r}, 'launch ownership differs'",
        "    assert launch['scope'] in ('pilot','full') and type(launch['complete']) is bool",
        "    if launch['complete']: assert type(launch.get('exit_code')) is int",
        "phase=('not_launched' if launch is None else 'active' if not launch['complete'] else",
        "       'complete' if launch['exit_code']==0 else 'failed')",
        "log=Path('/content/verif1_bootstrap.log')",
        "tail=[]",
        "if log.exists():",
        "    with log.open('rb') as f:",
        "        f.seek(0,2); f.seek(max(0,f.tell()-8192))",
        "        tail=f.read(8192).decode('utf-8','replace').splitlines()[-20:]",
        "alive=False",
        "if launch is not None and type(launch.get('pid')) is int:",
        "    try: os.kill(launch['pid'],0); alive=True",
        "    except ProcessLookupError: pass",
        "print(json.dumps({'status':phase,'launch':launch,'writer_alive':alive,'log_tail':tail,'tail_max_bytes':8192,",
        "                  'tail_max_lines':20,'limit':'active means no completion marker; not a heartbeat'}))",
    ]) + "\n"


def execute(args):
    if args.action == "supervised":
        from verif1_supervisor import run
        return run(args)
    if args.action == "up" and not args.dry_run:
        raise ValueError("use supervised for one bounded pilot; unsupervised allocation disabled")
    if args.action == "down" and not args.dry_run and args.receipt.exists():
        record = json.loads(args.receipt.read_text(encoding="utf-8"))
        if record.get("schema") == 1:
            from verif1_supervisor import cleanup_only
            return cleanup_only(args)
    if args.action == "up":
        plan = validate_archive(args.archive)
        preview = {"session": "phase0", "scope": args.scope, "archive_sha256": file_sha(args.archive),
                   "cost": cost(64 if args.scope == "pilot" else 270), "hourly_rate": args.hourly_rate,
                   "network": not args.dry_run, "refuse_any_existing_session": True}
        if args.dry_run:
            print(canonical(preview))
            return
    else:
        if args.dry_run:
            print(canonical({"action": args.action, "receipt": str(args.receipt), "network": False}))
            return
        if args.action == "status":
            record = owned(args.receipt)
            print(canonical(json.loads(cli(["exec", "-s", "phase0", "--timeout", "30"], status_script(record)))))
            return
        # Serialize cleanup separately from work, without stealing an orphaned generation lock.
        lock = (cleanup_mutex(args.receipt.with_suffix(".cleanup.lock")) if args.action in ("down", "pull", "extend")
                else exclusive(args.receipt.with_suffix(".lock")))
        with lock:
            record = owned(args.receipt)
            if args.action == "down":
                # Explicit cleanup remains possible after an interrupted upload. Never retry campaign work.
                path = args.receipt.with_name(args.receipt.stem + f".down.{time.time_ns()}.ledger.jsonl")
                journal = Ledger(path, sha(record))
                result = journal.operation("down", record, lambda: {"stopped": cli(["stop", "-s", "phase0"]),
                                                                   "utc": utc(), "billing": billing(record)})
                print(canonical(result))
            elif args.action == "pull":
                # Retrieval is independently reserved per destination, even when control history is interrupted.
                path = args.output.with_name(args.output.name + ".pull.ledger.jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                binding = {"receipt": record, "output": str(args.output.resolve())}
                journal = Ledger(path, sha(binding))
                journal.operation("pull", binding, lambda: recover(record, args.output))
            else:
                # Keep scientific work fail-closed while excluding concurrent recovery/stop.
                with exclusive(args.receipt.with_suffix(".lock")):
                    raise ValueError("full extension requires a separate authorization; pilot never extends")



def upload(archive):
    # Supervise the whole existing chunk transfer, including every safe_cli subprocess it starts.
    process([sys.executable, str(HERE / "transfer.py"), str(archive)], TRANSFER_SECONDS)
    cli(["upload", "-s", "phase0", str(HERE / "verif1_bootstrap.py"), "/content/verif1_bootstrap.py"])
    return {"archive_sha256": file_sha(archive), "bootstrap_sha256": file_sha(HERE / "verif1_bootstrap.py")}


def recover(record, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / "verif1_recovered.tgz"
    if target.exists() or (output / "verif-1").exists():
        raise ValueError("recovery target exists; preserve earlier snapshots and use a new directory")
    script = "\n".join([
        "import hashlib,json,tarfile",
        "from pathlib import Path",
        "state=Path('/content/verif1_launch.json')",
        "record=json.loads(state.read_text())",
        f"assert record['archive_sha256']=={record['archive_sha256']!r}",
        "assert record.get('complete') is True, 'pull requires stopped writer; wait for run completion'",
        "with tarfile.open('/content/verif1_recovered.tgz','w:gz') as t:",
        "    t.add('/content/verif-1',arcname='verif-1',filter=lambda m: m if "
        "not any(p in ('models','hf','.venv','uv-cache','llama','__pycache__') for p in Path(m.name).parts) "
        "else None)",
        "    t.add('/content/verif1_bootstrap.log',arcname='verif-1/bootstrap.log')",
        "    t.add(str(state),arcname='verif-1/launch.json')",
        "    for log in sorted(Path('/content/verif-1/runtime/models/server-logs').glob('*.log')):",
        "        t.add(str(log),arcname='verif-1/model-logs/'+log.name)",
        "h=hashlib.sha256()",
        "with open('/content/verif1_recovered.tgz','rb') as f:",
        "    for b in iter(lambda:f.read(1048576),b''):h.update(b)",
        "print(json.dumps({'sha256':h.hexdigest(),'exit_code':record.get('exit_code'),'scope':record['scope']}))",
    ])
    result = json.loads(cli(["exec", "-s", "phase0", "--timeout", "180"], script))
    cli(["download", "-s", "phase0", "/content/verif1_recovered.tgz", str(target)])
    if file_sha(target) != result["sha256"]:
        raise ValueError("recovery archive differs")
    with tarfile.open(target) as archive:
        for member in archive.getmembers():
            if (member.name.startswith("/") or ".." in Path(member.name).parts
                    or not (member.isfile() or member.isdir())):
                raise ValueError("unsafe recovered archive")
        archive.extractall(output, filter="data")
    result["billing"] = billing(record)
    write_new(output / "verif1_recovery_receipt.json", {**result, "session": record["session"], "utc": utc()})
    print(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("up", "supervised", "status", "extend", "pull", "down"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--hourly-rate", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-units", type=float, default=float("nan"))
    parser.add_argument("--hours", type=float, default=float("nan"))
    parser.add_argument("--min-balance-units", type=float, default=15)
    parser.add_argument("--max-rate-units-hour", type=float, default=5.3)
    parser.add_argument("--cleanup-seconds", type=float, default=1800)
    parser.add_argument("--read-seconds", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--up-seconds", type=float, default=2400)
    args = parser.parse_args()
    if (args.action in ("up", "supervised") and not args.archive) or (args.action == "pull" and not args.output):
        parser.error("up requires --archive, pull requires --output")
    if not 0 <= args.hourly_rate < float("inf"):
        parser.error("hourly rate must be finite and nonnegative")
    if args.action == "extend" and not args.dry_run:
        parser.error("full extension is outside the single-pilot authorization")
    if args.action == "down" and not args.dry_run:
        record = json.loads(args.receipt.read_text(encoding="utf-8"))
        if record.get("schema") != 1:
            parser.error("legacy receipt requires explicit identity reconciliation; no cloud action authorized")
    execute(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
