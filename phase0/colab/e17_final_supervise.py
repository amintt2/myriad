"""Explicit final Colab route: exact approved freeze and permanent consumption before the owned Campaign."""
import argparse
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e17_final_freeze as final
import session_json as lifecycle

spec = importlib.util.spec_from_file_location("e17_owned_final_campaign", HERE / "e17_campaign.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class FinalCampaign(shared.Campaign):
    inventory_validator = staticmethod(final.validate_inventory)
    retrieval_validator = staticmethod(final.validate_retrieved)
    complete_verdict = "test-complete"
    source_directory = final.ROOT

    def __init__(self, args, selected, files):
        super().__init__(args, selected, files)
        self.policy["prior_reserve"] = selected["budget"]["prior_reserve_until_reconciliation"]

    def rpc_source(self):
        return super().rpc_source() + "\nWORKER_SCRIPT = 'e17_final_remote.py'\n"

    def check_prelaunch(self):
        if (final.git_revision(self.source_directory) != self.selected["parent_decision"]["code_commit"]
                or final.source_hashes(final.final_sources(self.source_directory)) != self.selected["sources"]):
            raise ValueError("source revision changed after final validation; no allocation/start authorized")


def consumption_path():
    # Fixed shared reservation marker, independent of output/freeze/receipt options and code revisions.
    return Path.home() / ".local/state/myriad-colab/e17-final-test-consumed.json"


def controller_args(plan, output, receipt):
    policy = plan["budget"]
    return SimpleNamespace(output=output, receipt=receipt, spent_units=policy["charged_prior_units"],
                           budget_units=policy["policy"]["budget"], seconds=policy["seconds"],
                           min_balance_units=policy["policy"]["minimum"], cleanup_seconds=policy["cleanup_seconds"],
                           max_rate_units_hour=policy["policy"]["rate"], recover_seconds=policy["recover_seconds"],
                           **{k: policy[k] for k in ("read_seconds", "usage_seconds", "grace", "retry_seconds",
                                                    "poll_seconds")})


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--receipt", type=Path, default=lifecycle.receipt_path())
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--parent-validated-freeze")
    args = cli.parse_args(argv)
    plan, files = final.require_final_freeze(args.freeze)
    if final.git_revision() != plan["parent_decision"]["code_commit"]:
        raise ValueError("code revision differs from exact scientific freeze")
    if not args.execute:
        print(json.dumps({"state": "frozen", "allocated": False, "freeze_sha256": plan["freeze_sha256"],
                          "runtime_verified": False, "budget_guaranteed": False}))
        return 0
    if not sys.platform.startswith("linux"):
        raise ValueError("final execution must use the supervised WSL wrapper")
    if args.parent_validated_freeze != plan["freeze_sha256"]:
        raise ValueError("parent validation of this exact freeze is required before allocation")
    marker = consumption_path()
    protected = [args.receipt, Path(str(args.receipt) + ".lock"), Path(str(args.receipt) + ".usage.jsonl"), marker]
    outputs = [args.output, args.output.parent / (args.output.stem + "-request.json"),
               args.output.parent / (args.output.stem + "-artifacts")]
    if (any(p.exists() for p in outputs) or any(p.resolve() == q.resolve() for p in outputs for q in protected)
            or any(p.resolve().is_relative_to(args.freeze.resolve()) for p in outputs)):
        raise ValueError("fresh campaign evidence must be separate from freeze and lifecycle evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with lifecycle.lifecycle_lock(args.receipt):
        final.consume(marker, {"schema": 1, "freeze_sha256": plan["freeze_sha256"],
                               "selection_sha256": plan["selection_sha256"], "state": "started-no-replay"})
        return FinalCampaign(controller_args(plan, args.output, args.receipt), plan, final.vm_bundle(files)).execute()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, PermissionError, OSError) as error:
        print("final campaign refused: " + str(error), file=sys.stderr)
        sys.exit(73 if isinstance(error, PermissionError) else 65)
