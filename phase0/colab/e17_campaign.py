"""One owned DEV campaign; reuse the audited Probe lifecycle, never allocate during plan preparation."""
import argparse
import base64
import json
import math
import signal
import sys
import tempfile
import time
import uuid
import zlib
from pathlib import Path

import e17_probe
import session_json as lifecycle
from budget import preflight
from e17_campaign_remote import decode_response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e17_campaign import (CHUNK, artifacts, canonical, digest, plan, source_bundle,
                          validate_inventory, validate_retrieved)

HERE = Path(__file__).resolve().parent


def prepared(args):
    selected = plan(args.plan, args.reference_revision)
    files = source_bundle()
    return selected, files


def require_backend(files):
    # This is an early dependency guard, not a safety attestation; VM detection and receipts remain mandatory.
    if b"--sandbox-backend" not in files["essaim/code_pilot.py"]:
        raise ValueError("backend dependency missing: integrate code_pilot --sandbox-backend before allocation")


class Campaign(e17_probe.Probe):
    def __init__(self, args, selected, files):
        super().__init__(args)
        self.selected, self.files = selected, files
        self.request = self.inventory = None
        self.recovered = self.remote_attempted = False
        self.account_unknown = False
        self.recovery_error = None
        self.work_deadline = self.deadline - args.recover_seconds

    def monitor(self, deadline=None):
        if self.account_unknown:
            raise ValueError("account/budget became unknown; only cleanup is allowed")
        try:
            return super().monitor(deadline)
        except (ValueError, RuntimeError, OSError, TimeoutError):
            self.account_unknown = True
            raise

    def reconcile(self):
        if self.foreign:
            raise PermissionError("foreign campaign: no stop authorized")
        self.check_receipt()
        current = self.session()
        if current is not None and self.owner is None:
            if not self.new_confirmed or current["hardware"] != self.selected["gpu"]:
                raise PermissionError("allocation ambiguous: no stop authorized")
            self.owner = current
        if current is not None and current != self.owner:
            raise PermissionError("foreign session: no stop authorized")
        return current

    def rpc(self, operation, payload=None):
        if self.reconcile() != self.owner or self.owner is None:
            raise PermissionError("owned VM absent before campaign RPC")
        self.monitor()
        source = (HERE / "e17_campaign_remote.py").read_text(encoding="utf-8")
        source += f"\nemit_response(rpc({self.request!r}, {operation!r}, {payload!r}))\n"
        with tempfile.TemporaryDirectory(prefix="e17-rpc-") as directory:
            script = Path(directory) / "rpc.py"
            script.write_text(source, encoding="utf-8")
            raw = self.cli(["exec", "-s", "phase0", "--timeout", "50", "-f", str(script)], True)
        value = decode_response(raw)
        if value.get("request") != self.request or value.get("operation") != operation:
            raise ValueError("foreign/stale campaign acknowledgement")
        if value.get("foreign_campaign") is True:
            self.foreign = True
            lifecycle.save(self.a.receipt, {**self.pending, "status": "foreign-campaign", "phase0": self.owner})
            raise PermissionError("preexisting E17 directory: no execution or stop authorized")
        return value

    def recover(self):
        if not self.remote_attempted:
            return
        self.deadline = min(self.deadline, time.monotonic() + self.a.recover_seconds)
        self.rpc("cancel")
        while True:
            value = self.rpc("status")
            if value.get("inventory") is not None:
                self.inventory = validate_inventory(value["inventory"], self.request)
                break
            time.sleep(min(self.a.poll_seconds, max(0, self.deadline - time.monotonic())))
        directory = self.a.output.parent / (self.a.output.stem + "-artifacts")
        directory.mkdir(exist_ok=False)
        lifecycle.save(directory / "retrieval.json", self.inventory)
        for name, info in self.inventory["files"].items():
            temporary = directory / (name + ".part")
            offset = 0
            with temporary.open("wb") as dest:
                while offset < info["bytes"]:
                    value = self.rpc("chunk", {"name": name, "offset": offset})
                    if value.get("name") != name or value.get("offset") != offset:
                        raise ValueError("artifact chunk identity mismatch")
                    data = base64.b64decode(value["data"], validate=True)
                    if not data or len(data) > min(CHUNK, info["bytes"] - offset):
                        raise ValueError("artifact chunk exceeds declared coverage")
                    dest.write(data)
                    offset += len(data)
            temporary.replace(directory / name)
            if digest((directory / name).read_bytes()) != info["sha256"]:
                raise ValueError("artifact hash mismatch")
        lifecycle.save(directory / "recovery.json", {"all_inventory_hashes_verified": True,
                                                     "complete_claim": self.inventory["complete"]})
        # Hash all recovered files even after VM failure; completeness is a separate verdict.
        self.recovered = validate_retrieved(directory, self.inventory, self.request)

    def execute(self):
        if self.a.output.exists() or (self.a.output.parent / (self.a.output.stem + "-artifacts")).exists():
            raise ValueError("campaign evidence must be fresh; no silent retries")
        error = None
        try:
            if self.a.receipt.exists() or self.session() is not None:
                raise PermissionError("existing ownership/session refused")
            self.before = self.usage(False)
            preflight(self.before, self.policy)
            self.pending = {"schema": 1, "status": "allocation-unconfirmed", "phase0": None,
                            "campaign_id": self.campaign, "allocation_id": uuid.uuid4().hex,
                            "plan": self.selected["name"], "usage_before": self.before, "budget_policy": self.policy}
            lifecycle.save(self.a.receipt, self.pending)
            with e17_probe.defer_signals():
                self.cli(["new", "-s", "phase0", "--gpu", self.selected["gpu"]])
                self.new_confirmed = True
            lifecycle.save(self.a.receipt, {**self.pending, "allocation_returned": True})
            if self.reconcile() is None:
                raise ValueError("new did not establish a session")
            lifecycle.save(self.a.receipt, {**self.pending, "allocation_returned": True,
                                          "status": "owned", "phase0": self.owner})
            self.monitor()
            raw = self.cli(["exec", "-s", "phase0", "--timeout", "30", "-f", str(HERE / "snapshot.py")], True)
            snapshot = json.loads(raw)
            if snapshot.get("campaign_present") is True:
                self.foreign = True
                lifecycle.save(self.a.receipt, {**self.pending, "status": "foreign-campaign", "phase0": self.owner})
                raise PermissionError("preexisting remote campaign refused")
            if snapshot.get("schema") != 1 or snapshot.get("campaign_present") is not False:
                raise ValueError("remote snapshot unknown")
            available = self.work_deadline - time.monotonic() - 60
            if available <= 0:
                raise TimeoutError("preparation exhausted VM work envelope")
            self.request = {"campaign_id": self.campaign, "allocation_id": self.pending["allocation_id"],
                            "phase0": self.owner, "nonce": uuid.uuid4().hex, "plan": self.selected,
                            "sources": {name: digest(data) for name, data in self.files.items()},
                            "work_seconds": available}
            lifecycle.save(self.a.output.parent / (self.a.output.stem + "-request.json"), self.request)
            payload = base64.b64encode(zlib.compress(canonical({k: base64.b64encode(v).decode()
                                                               for k, v in self.files.items()}))).decode()
            self.remote_attempted = True  # Even a lost start ACK requires identity-bound cancellation/recovery.
            if self.rpc("start", payload).get("started") is not True:
                raise ValueError("remote start unconfirmed")
            while time.monotonic() < self.work_deadline:
                value = self.rpc("status")
                if value.get("inventory") is not None:
                    self.inventory = validate_inventory(value["inventory"], self.request)
                    break
                time.sleep(min(self.a.poll_seconds, max(0, self.work_deadline - time.monotonic())))
            else:
                raise TimeoutError("VM work envelope exhausted")
        except BaseException as exc:
            error = type(exc).__name__
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            try:
                self.recover()
            except BaseException as exc:
                self.recovery_error = type(exc).__name__
                error = error or self.recovery_error
            if self.pending is not None and self.a.receipt.exists():
                try:
                    self.cleanup()
                except BaseException as exc:
                    error = error or type(exc).__name__
                    try:
                        self.observations.append(lifecycle.usage(self.cli(["usage"])))
                    except (ValueError, RuntimeError, OSError, TimeoutError):
                        self.observations.append({"final_usage_unknown": True})
        success = self.recovered and self.closed and error is None
        report = {"schema": 1, "campaign_id": self.campaign, "plan": self.selected,
                  "verdict": "dev-complete" if success else "infrastructure-incomplete", "error": error,
                  "recovery_error": self.recovery_error, "retrieved_complete": self.recovered,
                  "inventory": self.inventory, "absence_confirmed": self.absent, "closed": self.closed,
                  "usage": self.observations, "prior_e17_units": self.a.spent_units,
                  "reserved_units": self.a.budget_units, "scientific_gain_claimed": False}
        lifecycle.save(self.a.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0 if success else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plan", choices=("e17-dev-single", "e17-dev-compare"), default="e17-dev-single")
    ap.add_argument("--reference-revision")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--receipt", type=Path, default=lifecycle.receipt_path())
    ap.add_argument("--budget-units", type=float, required=True)
    ap.add_argument("--spent-units", type=float, required=True)
    for name, default in (("seconds", 5400), ("min-balance-units", 15), ("cleanup-seconds", 1800),
                          ("max-rate-units-hour", 5.3), ("read-seconds", 120), ("usage-seconds", 30),
                          ("grace", 5), ("retry-seconds", 1), ("recover-seconds", 1800), ("poll-seconds", 15)):
        ap.add_argument("--" + name, type=float, default=default)
    args = ap.parse_args()
    if (any(not math.isfinite(v) or v <= 0 for k, v in vars(args).items()
            if isinstance(v, float) and k != "spent_units") or not math.isfinite(args.spent_units)
            or args.spent_units < 0 or args.budget_units + args.spent_units > 20 or args.min_balance_units < 15
            or args.recover_seconds >= args.seconds or args.seconds > 11500):
        ap.error("finite positive limits, E17 total <=20, reserve >=15 and bounded recovery are required")
    protected = [args.receipt, Path(str(args.receipt) + ".lock"), Path(str(args.receipt) + ".usage.jsonl")]
    outputs = [args.output, args.output.parent / (args.output.stem + "-request.json")]
    if any(p.resolve() in [q.resolve() for q in protected] for p in outputs):
        ap.error("output must differ from lifecycle evidence")
    if any(p.exists() for p in outputs) or (args.output.parent / (args.output.stem + "-artifacts")).exists():
        ap.error("campaign evidence must be fresh")
    selected, files = prepared(args)
    if not args.execute:
        if args.output.exists():
            ap.error("plan evidence must be fresh")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical({"plan": selected, "sources": {n: digest(v) for n, v in files.items()},
                                         "allocated": False, "score": None,
                                         "backend_dependency_ready": b"--sandbox-backend" in files["essaim/code_pilot.py"]}))
        return 0
    require_backend(files)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with lifecycle.lifecycle_lock(args.receipt):
        return Campaign(args, selected, files).execute()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, PermissionError, OSError) as exc:
        print(f"campaign refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(73 if isinstance(exc, PermissionError) else 65)
