"""Fresh worker processes and public runtime metadata only; no infrastructure or scientific data."""
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import e17_final_freeze as final
import e17_final_vm as vm
import test_e17_final_runner as fixture

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "bigcode_pilot/runtime.json"

CHILD = '''import json,sys
from pathlib import Path
from unittest import mock
root=Path.cwd()
sys.path.insert(0,str(root/'phase0/colab'))
import e17_final_remote as entry
assert 'e17_campaign' not in sys.modules
original=entry.shared.WORKER_PREFLIGHT
def run(*args,**kwargs):
    raise AssertionError('infrastructure must not run')
if sys.argv[1]=='final':
    import essaim.ephemeral_vm as backend
    def stop():
        raise RuntimeError('synthetic preflight stop')
    with mock.patch.object(backend,'require_colab',side_effect=stop), mock.patch('read_timeout.run',side_effect=run):
        entry.worker(root)
    assert entry.shared.WORKER_PREFLIGHT is original
else:
    def stop(*args):
        raise RuntimeError('synthetic preflight stop')
    entry.shared.WORKER_PREFLIGHT=stop
    with mock.patch('read_timeout.run',side_effect=run):
        entry.shared.worker(root)
status=json.loads((root/'artifacts/vm-status.json').read_text())
inventory=json.loads((root/'inventory.json').read_text())
assert status['phase']=='final-preflight'
assert status['state']=='incomplete' and status['error']=='RuntimeError'
assert 'synthetic preflight stop' in status['diagnostic']['trace']
assert inventory['complete'] is False and set(inventory['files'])=={'vm-status.json'}
assert inventory['request']==status['request']
assert Path(sys.modules['e17_campaign'].__file__).resolve()==(root/'phase0/e17_campaign.py').resolve()
print(json.dumps({'closed':True,'infrastructure_called':False,'inventory':inventory}))
'''


class FreshWorkerTests(unittest.TestCase):
    def exercise(self, route):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = fixture.fixture_plan(root / "fixture")
            # Copy source files only. Each child starts without cached protocol modules.
            for directory in (ROOT, ROOT / "colab", ROOT / "essaim"):
                for path in directory.glob("*.py"):
                    target = root / "phase0" / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            (root / "artifacts").mkdir()
            request = {"work_seconds": 10, "plan": plan if route == "final" else {"modes": ["single"]}}
            (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, route], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["closed"])

    def test_final_fresh_worker_closes_before_infrastructure(self):
        self.exercise("final")

    def test_historical_fresh_worker_closes_before_infrastructure(self):
        self.exercise("historical")


def observed_packages(packages):
    """Evaluate the actual production metadata expression without running the VM controller."""
    tree = ast.parse((ROOT / "e17_final_vm.py").read_text(encoding="utf-8"))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "runtime_info" for t in n.targets))
    expression = next(value for key, value in zip(assignment.value.keys, assignment.value.values)
                      if isinstance(key, ast.Constant) and key.value == "packages")
    metadata = SimpleNamespace(distributions=lambda: [SimpleNamespace(metadata={"Name": n}, version=v)
                                                     for n, v in packages])
    return eval(compile(ast.Expression(expression), "production-package-expression", "eval"),
                {"importlib": SimpleNamespace(metadata=metadata),
                 "package_versions": getattr(final, "package_versions", None)})


class RuntimePackagesTests(unittest.TestCase):
    def test_real_public_pin_survives_production_and_complete_recovery(self):
        packages = json.loads(PIN.read_text(encoding="utf-8"))["packages"]
        self.assertIn("Pygments", packages)
        original = fixture.fixture_plan
        original_manifest = fixture.fixture_manifest
        def pinned_fixture(root):
            plan, files = original(root)
            pin = copy.deepcopy(plan["runtime_pin"])
            pin["packages"] = packages
            files["phase0/bigcode_pilot/runtime.json"] = final.canonical(pin)
            decision = copy.deepcopy(plan["parent_decision"])
            decision["sources"] = final.source_hashes(files)
            return final.frozen_plan(files, plan["budget"], decision), files
        def observed_manifest(plan, request):
            manifest = original_manifest(plan, request)
            manifest["runtime"]["packages"] = observed_packages([(n.lower(), v) for n, v in packages.items()])
            return manifest
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(fixture, "fixture_plan", pinned_fixture), \
                mock.patch.object(fixture, "fixture_manifest", observed_manifest):
            root, output = Path(temporary) / "source", Path(temporary) / "artifacts"
            plan, request, manifest, _ = fixture.RunnerTests().run_fixture(root, output)
            for name, value in {"manifest.json": manifest, "runtime.json": manifest["runtime"],
                "dataset-manifest.json": manifest["dataset"], "provenance.json": manifest["execution_provenance"],
                "preparation-started.json": {"freeze_sha256": plan["freeze_sha256"], "request": request},
                "smoke.json": {"kind": "prefixed-synthetic-public-unittest", "freeze_sha256": plan["freeze_sha256"],
                               "oracle_test_used": False, "positive": fixture.grade(), "negative": fixture.grade(False),
                               "aider_positive": fixture.grade(), "aider_negative": fixture.grade(False)},
                "vm-status.json": {"request": request, "state": "complete", "error": None}}.items():
                (output / name).write_bytes(final.canonical(value))
            (output / "controller.log").write_bytes(b"synthetic closed controller")
            (output / "server-events.jsonl").write_bytes(b'{"synthetic":true}\n')
            inventory = {"schema": 1, "request": request, "complete": True, "files": {
                name: {"bytes": len((output / name).read_bytes()), "sha256": final.digest((output / name).read_bytes())}
                for name in final.artifacts(plan)}}
            runtime = final.read_json((output / "runtime.json").read_bytes())
            runtime["packages"] = observed_packages([(n.lower(), v) for n, v in packages.items()])
            manifest = final.read_json((output / "manifest.json").read_bytes())
            manifest["runtime"] = runtime
            for name, value in (("runtime.json", runtime), ("manifest.json", manifest)):
                raw = final.canonical(value)
                (output / name).write_bytes(raw)
                inventory["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            final.validate_retrieved(output, inventory, request)
            for changed in ({**runtime["packages"], "extra": "1"},
                            {k: v for k, v in runtime["packages"].items() if k != "pygments"},
                            {**runtime["packages"], "pygments": "wrong"},
                            {**runtime["packages"], "Pygments": runtime["packages"]["pygments"]}):
                runtime["packages"] = changed
                manifest["runtime"] = runtime
                for name, value in (("runtime.json", runtime), ("manifest.json", manifest)):
                    raw = final.canonical(value)
                    (output / name).write_bytes(raw)
                    inventory["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                with self.assertRaisesRegex(ValueError, "runtime binary/package|ambiguous normalized"):
                    final.validate_retrieved(output, inventory, request)

    def test_normalized_identity_and_duplicate_observations_are_strict(self):
        self.assertEqual(observed_packages([("Pygments", "2.21.0"), ("Example_Pkg.Name", "1")]),
                         {"pygments": "2.21.0", "example-pkg-name": "1"})
        for records in ([("Pygments", "1"), ("pygments", "1")],
                        [("Example_Pkg", "1"), ("example-pkg", "2")],
                        [(None, "1")], [("bad name", "1")], [("valid", None)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                observed_packages(records)
