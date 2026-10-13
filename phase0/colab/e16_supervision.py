"""One bounded E16 plan using the audited allocation and cleanup lifecycle."""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import signal
import sys
import tarfile
import tempfile
import time
import uuid
import zlib
from pathlib import Path

from budget import UsageTimeout, number, preflight, validate_usage
from campaign import state
from e17_probe import Probe, cli_signals
from read_timeout import run
import session_json as lifecycle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def admission(usage, prior_reserve_units=0):
    validate_usage(usage)
    number(prior_reserve_units, "prior reserve", True)
    plan = "e16-1" if usage["balance_units"] >= 45 else "e16-mc"
    low, high = (5.5, 7.6) if plan == "e16-1" else (4, 5.6)
    upper_budget = math.ceil(100 * 5.3 * (high + (900 + 1800 + 161) / 3600)) / 100
    minimum = 15 + prior_reserve_units
    available = usage["balance_units"] - minimum
    budget = min(upper_budget, available)
    # Leave analysis/recovery time inside the operation window, and cleanup/margin outside it.
    seconds = min(high * 3600 + 900, budget / 5.3 * 3600 - 1800 - 161)
    # Avoid binary rounding making preflight cost infinitesimally larger than the cap.
    seconds = math.floor(seconds * 1000000) / 1000000
    science_seconds = seconds - 900
    policy = {"hours": seconds / 3600, "cleanup": 1800, "margin": 161, "minimum": minimum,
              "rate": 5.3, "budget": budget, "prior_reserve": prior_reserve_units}
    value = {"plan": plan, "gpu": "A100", "estimated_hours": [low, high],
             "estimated_units": [low * 5.3, high * 5.3], "seconds": seconds, "policy": policy,
             "available_science_seconds": science_seconds, "upper_budget_units": upper_budget,
             "incompletion_risk": science_seconds < high * 3600,
             "warning": "Ces estimations ne promettent pas de finir ; tous les jobs sont conservés. "
                        "Si la projection dépasse la fenêtre disponible, le risque d’incomplétude est explicite ; "
                        "l’expiration conserve les partiels et clôture la VM.",
             "balance_units": usage["balance_units"], "available_units": available,
             "prior_reserve_units": prior_reserve_units}
    try:
        preflight(usage, policy)
        value["admitted"] = science_seconds > 0
    except ValueError:
        value["admitted"] = False
    return value


def validate_paths(output, receipt):
    outputs = [Path(str(output) + suffix).resolve() for suffix in ("", ".sources.json", ".snapshots.jsonl")]
    protected = [Path(str(receipt) + suffix).resolve() for suffix in
                 ("", ".lock", ".usage.jsonl", ".archive.tgz", ".archive.tgz.colab_bootstrap.py")]
    paths = outputs + protected
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("output/ownership companions collide")
    if any(path.exists() for path in outputs):
        raise ValueError("use a new output path for each attempt")
    # Nonces are generated later; reserve every recovery directory in this report's namespace.
    parent = output.parent.resolve()
    for path in outputs + protected:
        if parent in path.parents and re.fullmatch(r"e16-[0-9a-f]{32}", path.relative_to(parent).parts[0]):
            raise ValueError("report/ownership path collides with recovery directory namespace")


def package(directory, campaign, plan):
    # Git-selected code only. Existing result bytes may be copied/hashed but never analysed on the PC.
    import ast
    import subprocess
    import e16_quant as quant
    tracked = subprocess.check_output(["git", "ls-files", "-z", "phase0"], cwd=REPO).decode().split("\0")
    selected = [r for r in tracked if r and not any(x in Path(r).parts for x in
                ("results", "data", ".venv", "__pycache__")) and
                (r.endswith((".py", ".sh")) or Path(r).name in ("pyproject.toml", "uv.lock", "speeds_consumer.json"))]
    # Include the new supervision modules before the parent commits/exports the prepared tree.
    selected = sorted(set(selected) | {"phase0/colab/e16_remote.py", "phase0/colab/e16_supervision.py"})
    results = REPO / "phase0/results"
    families = next(ast.literal_eval(node.value) for node in ast.parse((HERE.parent / "analyze_e4.py").read_text()).body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILIES"
                                                          for t in node.targets))
    names = {f"solo_{m.replace('/', '__')}_colab_{b}_{s}.jsonl" for m in quant.SPEED_Q8
             for b in ("gsm8k", "math500", "arc", "mmlupro") for s in ("dev", "test")}
    if plan == "e16-1":
        names |= {f"code_{m.replace('/', '__')}_colab_{b}_{s}.jsonl" for m in families.values()
                  for b in ("humanevalplus", "mbppplus") for s in ("dev", "test")}
        names |= {f"codeexec_e11_colab_{b}_{s}.jsonl" for b in ("humanevalplus", "mbppplus") for s in ("dev", "test")}
    names |= {n + ".meta.json" for n in tuple(names)}
    if plan == "e16-1":
        names |= {"e11_summary_colab.json", "e11_validation_colab.json"}
    refs = [results / name for name in sorted(names)]
    missing = [p.name for p in refs if not p.is_file()]
    if missing:
        raise ValueError("required existing analysis references absent: " + ", ".join(missing))
    provenance = {"schema": 1, "campaign_id": campaign, "plan": plan,
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
                  "sources": {r: lifecycle.digest(REPO / r) for r in selected},
                  "references": {p.name: lifecycle.digest(p) for p in refs}}
    archive = directory / "prepared.tgz"
    with tarfile.open(archive, "w:gz") as out:
        for rel in selected:
            path = REPO / rel
            if path.is_symlink() or not path.is_file():
                raise ValueError("source absent or symlink")
            out.add(path, arcname=rel, recursive=False)
        for path in refs:
            if path.is_symlink() or path.stat().st_size > MAX_FILE:
                raise ValueError("reference exceeds limits")
            out.add(path, arcname="phase0/results/" + path.name, recursive=False)
        for name, text in (("phase0/colab_plan.txt", plan), ("phase0/colab_campaign_id.txt", campaign),
                           ("phase0/results/e16_sources.json", json.dumps(provenance, sort_keys=True))):
            import io
            content = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            out.addfile(member, io.BytesIO(content))
    if archive.stat().st_size > MAX_TOTAL:
        raise ValueError("archive exceeds preparation bounds")
    return archive, provenance


class E16(Probe):
    gpu = "A100"
    plan = None

    def __init__(self, args):
        args.sandbox_backend = "e11-existing"
        super().__init__(args)
        self.science_deadline = self.deadline
        self.work = None
        self.provenance = None
        self.admission = None
        self.recovered = []
        self.recovery_attempts = []
        self.account_failed = False
        self.preflight_only = not args.execute

    def prepare(self):
        choice = admission(self.before, getattr(self.a, "prior_reserve_units", 0))
        self.admission = choice
        print(json.dumps(choice, sort_keys=True), flush=True)
        lifecycle.save(self.a.output, {"schema": 1, "preflight": choice, "allocated": False, "prepared": False})
        self.plan, self.policy = choice["plan"], choice["policy"]
        self.deadline = self.started + choice["seconds"]
        self.recovery_deadline = self.deadline
        self.science_deadline = self.started + choice["available_science_seconds"]
        if not choice["admitted"]:
            raise ValueError("no useful science window after account reserves; no allocation")
        self.work = tempfile.TemporaryDirectory(prefix="myriad-e16-")
        self.archive, self.provenance = package(Path(self.work.name), self.campaign, self.plan)
        self.remote_source = (HERE / "e16_remote.py").read_bytes().decode("utf-8")
        self.bootstrap_source = (HERE.parent / "colab_bootstrap.py").read_bytes().decode("utf-8")
        for rel, source in (("phase0/colab/e16_remote.py", self.remote_source),
                            ("phase0/colab_bootstrap.py", self.bootstrap_source)):
            if hashlib.sha256(source.encode()).hexdigest() != self.provenance["sources"][rel]:
                raise ValueError("runtime source changed during preparation")
        lifecycle.save(Path(str(self.a.output) + ".sources.json"), self.provenance)
        lifecycle.save(self.a.output, {"schema": 1, "preflight": choice, "allocated": False,
                                      "prepared": True, "archive_sha256": lifecycle.digest(self.archive)})

    def operation_seconds(self, args):
        if args[0] == "exec":
            return float(args[args.index("--timeout") + 1]) + 30
        return super().operation_seconds(args)

    def cli(self, args, monitor=False):
        if args[0] in ("exec", "upload", "download", "stop") and self.owner is not None:
            if self.reconcile() != self.owner:
                raise ValueError("owned session absent before operation")
        return super().cli(args, monitor)

    def usage(self, allocated):
        if not allocated and self.absent and self.pending is not None:
            self.check_receipt()
            saved = json.loads(self.a.receipt.read_text())
            try:
                lifecycle.save(self.a.receipt, {**saved, "status": "released/accounting-pending",
                                               "phase0": self.owner, "release_confirmed": True})
            except OSError:
                # Do not lose memory authority when an absence marker cannot be persisted.
                print("absence marker persistence pending", flush=True)
        value = super().usage(allocated)
        if allocated:
            lifecycle.usage_record(self.a.receipt, self.before, value, campaign_id=self.campaign)
        return value

    def account_window_deadline(self):
        # Science expiry bounds its caller; the admitted recovery deadline bounds the account window.
        return getattr(self, "recovery_deadline", self.deadline)

    def monitor(self, deadline=None, expire=None):
        if self.account_failed:
            raise ValueError("account/budget became unknown; only cleanup is allowed")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("operation window expired")
        try:
            return super().monitor(deadline, expire)
        except BaseException as exc:
            # Operation expiry may use recovery, but an exhausted account wait authorizes only cleanup.
            if (self.account_waiting or isinstance(exc, UsageTimeout) or time.monotonic() < self.deadline
                    or self.deadline > self.science_deadline):
                self.account_failed = True
            raise

    def process(self, command, seconds):
        self.reconcile()
        code = run(command, seconds=min(seconds, max(.01, self.deadline - time.monotonic())),
                   grace=self.a.grace, deadline=self.deadline, monitor=self.monitor, monitor_expiry=True,
                   preexec_fn=cli_signals, env={**os.environ, "DLLM_REQUIRE_OWNER": "1",
                   "DLLM_CAMPAIGN_RECEIPT": str(self.a.receipt), "DLLM_CAMPAIGN_ID": self.campaign,
                   "DLLM_OPERATION_IDENTITY": json.dumps(lifecycle.identity(json.loads(self.a.receipt.read_text())))},
                   pass_fds=(int(os.environ["DLLM_LIFECYCLE_LOCK_FD"]),))
        if code:
            raise RuntimeError("bounded operation failed: " + str(code))

    def remote(self, source, timeout=50):
        if self.reconcile() != self.owner:
            raise ValueError("owned session absent")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as script:
            script.write(source)
            script.flush()
            return self.cli(["exec", "-s", "phase0", "--timeout", str(timeout), "-f", script.name], True)

    def snapshot(self):
        value = json.loads(self.remote((HERE / "snapshot.py").read_text()))
        with Path(str(self.a.output) + ".snapshots.jsonl").open("a") as output:
            output.write(json.dumps({"campaign_id": self.campaign, "time": time.time(), "snapshot": value}) + "\n")
            output.flush()
            os.fsync(output.fileno())
        boot = value.get("bootstrap")
        if isinstance(boot, dict) and (boot.get("campaign_id") != self.campaign
                or ("plan" in boot and boot["plan"] != self.plan)):
            self.mark_foreign()
        return value

    def mark_foreign(self):
        self.foreign = True
        self.check_receipt()
        saved = json.loads(self.a.receipt.read_text())
        lifecycle.save(self.a.receipt, {**saved, "status": "foreign-campaign"})
        raise PermissionError("foreign bootstrap; no stop authorized")

    def recover(self, final):
        request = {"campaign_id": self.campaign, "allocation_id": self.pending["allocation_id"],
                   "plan": self.plan, "nonce": uuid.uuid4().hex}
        source = self.remote_source
        payload = source + ("\nimport contextlib\n"
            "with open('/content/dllm/phase0/results/e16_verification.log', 'a') as log, contextlib.redirect_stdout(log), "
            "contextlib.redirect_stderr(log):\n"
            "    try:\n"
            f"        result = seal({request!r}, {final!r})\n"
            "    except PermissionError:\n"
            f"        result = {{**{request!r}, 'foreign_campaign': True}}\n"
            "import base64, zlib\n"
            "print('E16_SEAL ' + base64.b64encode(zlib.compress(json.dumps(result).encode())).decode())\n")
        # The official exec interpreter has no project dependencies; run checks in the VM project environment.
        wrapper = ("import os, subprocess, tempfile\n"
                   "with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as script:\n"
                   f"    script.write({payload!r})\n"
                   "    script.flush()\n"
                   "    subprocess.run([os.path.expanduser('~/.local/bin/uv'), 'run', '--group', 'paper', "
                   "'python', script.name], cwd='/content/dllm/phase0', "
                   "env={**os.environ, 'PYTHONPATH': '/content/dllm/phase0', "
                   "'PATH': os.path.expanduser('~/.local/bin') + ':' + os.environ['PATH']}, "
                   "check=True, timeout=660)\n")
        raw = self.remote(wrapper, 690 if final else 90)
        lines = raw.strip().splitlines()
        if len(lines) != 1 or not lines[0].startswith("E16_SEAL "):
            raise ValueError("missing recovery acknowledgement")
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(base64.b64decode(lines[0][9:], validate=True), 256 * 1024 + 1)
        if len(decoded) > 256 * 1024 or not decoder.eof or decoder.unused_data:
            raise ValueError("recovery acknowledgement exceeds decoded limit")
        seal = json.loads(decoded)
        if not isinstance(seal, dict):
            raise ValueError("invalid recovery acknowledgement")
        if all(seal.get(k) == v for k, v in request.items()) and seal.get("foreign_campaign") is True:
            self.mark_foreign()
        if (type(seal.get("schema")) is not int or seal["schema"] != 1
                or any(seal.get(k) != v for k, v in request.items()) or seal.get("complete") is not final):
            raise ValueError("stale recovery identity")
        files = seal.get("files")
        if not isinstance(files, dict) or not files or len(files) > 512:
            raise ValueError("invalid recovery inventory")
        total = 0
        for name, spec in files.items():
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..", "seal.json", "verified.json")
                    or not isinstance(spec, dict)
                    or type(spec.get("size")) is not int or not 0 <= spec["size"] <= MAX_FILE
                    or not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256")))):
                raise ValueError("unsafe recovery entry")
            total += spec["size"]
        if total > MAX_TOTAL:
            raise ValueError("recovery exceeds total limit")
        dest = self.a.output.parent / ("e16-" + self.campaign) / ("final-" if final else "partial-")
        dest = Path(str(dest) + request["nonce"])
        dest.mkdir(parents=True)
        attempt = {"directory": str(dest), "final_requested": final, "verified": False}
        self.recovery_attempts.append(attempt)
        lifecycle.save(dest / "seal.json", seal)
        for name, spec in files.items():
            path = dest / name
            # Downloaded bytes bypass the CLI stdout line limit, including large JSON lines.
            self.cli(["download", "-s", "phase0", "/content/e16-recovery-" + request["nonce"] + "/" + name,
                      str(path)], True)
            if path.stat().st_size != spec["size"] or lifecycle.digest(path) != spec["sha256"]:
                raise ValueError("recovered bytes do not match sealed VM artifact")
        sources = json.loads((dest / "e16_sources.json").read_text())
        if sources != self.provenance:
            raise ValueError("recovered provenance differs from prepared sources")
        lifecycle.save(dest / "verified.json", {"schema": 1, "seal_sha256": lifecycle.digest(dest / "seal.json")})
        attempt["verified"] = True
        self.recovered.append(str(dest))

    def perform(self):
        final_complete = False
        self.deadline = min(self.deadline, self.science_deadline)
        try:
            needed = lifecycle.capacity(self.plan, self.gpu)
            capacity = self.remote("import subprocess, json\n"
                "r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], "
                "capture_output=True, text=True, check=True, timeout=10)\n"
                "print('E16_CAPACITY ' + json.dumps({'free_mib': float(r.stdout.strip())}))\n")
            if not capacity.startswith("E16_CAPACITY "):
                raise ValueError("missing measured capacity")
            free = json.loads(capacity[13:]).get("free_mib")
            if type(free) not in (float, int) or not math.isfinite(free) or free < needed * 1024:
                raise ValueError("measured A100 capacity insufficient")
            self.process([sys.executable, str(HERE / "transfer.py"), str(self.archive)], 2400)
            self.remote(self.bootstrap_source, 1800)
            next_pull = time.monotonic() + 600
            while True:
                self.monitor()
                snap = self.snapshot()
                if state(snap, self.plan) == "complete":
                    self.deadline = self.recovery_deadline
                    self.recover(True)
                    final_complete = True
                    return {"verdict": "compatible", "complete": True, "plan": self.plan,
                            "sources": self.provenance, "recovered": self.recovered}
                if time.monotonic() >= next_pull:
                    self.recover(False)
                    next_pull = time.monotonic() + 600
                end = min(self.deadline, time.monotonic() + self.a.poll_seconds)
                while time.monotonic() < end:
                    self.monitor()
                    time.sleep(min(self.a.usage_seconds, max(0, end - time.monotonic())))
        finally:
            if not final_complete and not self.foreign and not self.account_failed:
                try:
                    self.deadline = self.recovery_deadline
                    self.next_usage = 0
                    self.monitor()
                    self.recover(False)
                except (ValueError, RuntimeError, OSError, TimeoutError):
                    print("partial recovery incomplete; retained files remain distinct", flush=True)

    def execute(self):
        try:
            if self.a.cleanup_only:
                saved = json.loads(self.a.receipt.read_text())
                if (saved.get("plan") not in ("e16-1", "e16-mc") or saved.get("status") not in
                        ("owned", "released/accounting-pending") or not isinstance(saved.get("phase0"), dict)
                        or saved["phase0"].get("hardware") != self.gpu):
                    raise PermissionError("cleanup-only requires a durable owned E16 identity")
                self.pending, self.owner, self.before = saved, saved["phase0"], saved["usage_before"]
                self.campaign, self.plan = saved["campaign_id"], saved["plan"]
                self.new_confirmed = True
                self.absent = saved.get("status") == "released/accounting-pending" and saved.get("release_confirmed") is True
                self.cleanup()
                lifecycle.save(self.a.output, {"schema": 1, "campaign_id": self.campaign,
                                              "closed": self.closed, "absence_confirmed": self.absent,
                                              "usage": self.observations, "cleanup_only": True})
                return 0
            if self.preflight_only:
                if self.a.receipt.exists() or self.session() is not None:
                    raise PermissionError("existing lifecycle refused")
                self.before = self.wait_usage(False)
                self.prepare()
                return 0
            code = super().execute()
            report = json.loads(self.a.output.read_text())
            # E16 keeps the existing E11 isolation recorded in its scientific manifests; it is not an E17 probe.
            report.pop("sandbox_backend", None)
            report["artifacts"] = report.pop("probe", None)
            lifecycle.save(self.a.output, {**report, "verdict": "complete" if code == 0 else "incomplete", "plan": self.plan,
                                          "recovered": self.recovered, "recovery_attempts": self.recovery_attempts,
                                          "sources": self.provenance, "admission": self.admission, "foreign_campaign": self.foreign,
                                          "new_confirmed": self.new_confirmed, "owner": self.owner,
                                          "allocation_id": self.pending.get("allocation_id") if self.pending else None})
            return code
        finally:
            if self.work is not None:
                self.work.cleanup()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, default=lifecycle.receipt_path())
    ap.add_argument("--poll-seconds", type=float, default=60)
    ap.add_argument("--prior-reserve-units", type=float, default=0)
    args = ap.parse_args()
    if args.execute and args.cleanup_only:
        ap.error("execute and cleanup-only are mutually exclusive")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        ap.error("finite positive poll interval required")
    if not math.isfinite(args.prior_reserve_units) or args.prior_reserve_units < 0:
        ap.error("finite nonnegative prior reserve required")
    try:
        validate_paths(args.output, args.receipt)
    except ValueError as exc:
        ap.error(str(exc))
    args.seconds, args.budget_units, args.min_balance_units = 300, 1, 15
    args.cleanup_seconds, args.max_rate_units_hour = 1800, 5.3
    args.read_seconds, args.usage_seconds, args.grace, args.retry_seconds = 120, 30, 5, 1
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with lifecycle.lifecycle_lock(args.receipt) as lock:
        previous = os.environ.get("DLLM_LIFECYCLE_LOCK_FD")
        os.environ["DLLM_LIFECYCLE_LOCK_FD"] = str(lock.fileno())
        try:
            try:
                return E16(args).execute()
            except (PermissionError, ValueError, OSError, TimeoutError) as exc:
                print(f"E16 refused: {exc}; inspect retained receipt", file=sys.stderr)
                return 73 if isinstance(exc, PermissionError) else 65
        finally:
            if previous is None:
                os.environ.pop("DLLM_LIFECYCLE_LOCK_FD", None)
            else:
                os.environ["DLLM_LIFECYCLE_LOCK_FD"] = previous


if __name__ == "__main__":
    sys.exit(main())
