"""Build a local upload archive with DEV-only inputs and frozen sources; never open TEST or use network."""
import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verif1_prepare import ROOT
from verif1_run import validate_bundle


def package(bundle, plan, archive):
    value, registration = json.loads(bundle.read_text(encoding="utf-8")), json.loads(plan.read_text(encoding="utf-8"))
    validate_bundle(value, registration)
    with tarfile.open(archive, "x:gz") as target:
        for name in registration["sources"]:
            target.add(ROOT / name, arcname=name, recursive=False)
        target.add(bundle, arcname="verif1_dev_bundle.json", recursive=False)
        target.add(plan, arcname="verif1_dev_plan.json", recursive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--archive", "--output", dest="archive", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.prepared:
        if args.bundle or args.plan:
            parser.error("--prepared cannot be mixed with --bundle/--plan")
        args.bundle = args.prepared / "verif1_dev_bundle.json"
        args.plan = args.prepared / "verif1_dev_plan.json"
    if not args.bundle or not args.plan:
        parser.error("provide --prepared or both --bundle and --plan")
    package(args.bundle, args.plan, args.archive)


if __name__ == "__main__":
    main()
