"""Synthetic E17 backend checks; no datasets, model requests or real Colab CLI."""
import ctypes
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from essaim import aider_python, code_pilot, ephemeral_vm, local_linux
from essaim.aider_agent import TEST
from essaim.agent import SUBMIT
from test_code_pilot import FakeClient, answer, fixture

READY = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
         "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}


class Selection(unittest.TestCase):
    def test_default_and_no_implicit_fallback(self):
        self.assertEqual(code_pilot.parser().parse_args([]).sandbox_backend, "strict")
        with patch.object(ephemeral_vm, "require_colab", side_effect=AssertionError("must not detect")):
            self.assertIs(ephemeral_vm.select_backend("strict"), local_linux.LocalLinuxEnv)
        with self.assertRaises(ValueError):
            ephemeral_vm.select_backend("auto")

    def test_environment_variable_does_not_establish_colab(self):
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "fake", "COLAB_GPU": "1"}), \
             patch.object(Path, "is_file", return_value=False), patch.object(Path, "is_dir", return_value=False), \
             self.assertRaises(local_linux.IsolationError):
            ephemeral_vm.select_backend("ephemeral-vm")

    def test_two_real_markers_and_linux_required(self):
        for system, kernel, markers, allowed in (
                ("linux", "6.6", {"a": True, "b": True}, True),
                ("linux", "6.6", {"a": True, "b": False}, False),
                ("linux", "6.6-microsoft-standard-WSL2", {"a": True, "b": True}, False),
                ("win32", "6.6", {"a": True, "b": True}, False)):
            with self.subTest(system=system, kernel=kernel), patch.object(sys, "platform", system), \
                 patch.object(ephemeral_vm.platform, "machine", return_value="x86_64"), \
                 patch.object(ephemeral_vm.platform, "release", return_value=kernel), \
                 patch.object(ephemeral_vm, "colab_markers", return_value=markers):
                if allowed:
                    self.assertIs(ephemeral_vm.select_backend("ephemeral-vm"), ephemeral_vm.EphemeralVMEnv)
                else:
                    with self.assertRaises(local_linux.IsolationError):
                        ephemeral_vm.select_backend("ephemeral-vm")

    def test_refusal_precedes_tasks_endpoint_and_client(self):
        args = code_pilot.parser().parse_args(["--sandbox-backend", "ephemeral-vm", "--single", "fake"])
        with patch.object(ephemeral_vm, "require_colab", side_effect=local_linux.IsolationError("not Colab")), \
             patch.object(code_pilot, "provenance") as data, patch.object(code_pilot, "validate_endpoint") as endpoint, \
             patch.object(code_pilot, "DeadlineClient") as client:
            with self.assertRaises(local_linux.IsolationError):
                code_pilot.execute(args)
            data.assert_not_called()
            endpoint.assert_not_called()
            client.assert_not_called()

    def test_shared_launcher_limits_seccomp_and_strict_default(self):
        names = ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_CORE")
        limits = SimpleNamespace(**dict(zip(names, range(len(names)))), setrlimit=Mock())
        isolate = Mock(return_value=None)
        cfg = json.dumps({"work": ".", "runtime": ".", "cpu": 4, "cmd": "synthetic"}) + "\n"
        with patch.object(local_linux, "resource", limits, create=True), patch.object(sys, "stdin", io.StringIO(cfg)), \
             patch.object(local_linux.os, "chdir"), patch.object(local_linux.os, "execve", create=True) as execute, \
             patch.object(local_linux, "seccomp") as seccomp, patch.object(sys, "stdout", io.StringIO()) as out:
            local_linux.launch(isolate, ephemeral_vm.VERSION)
            self.assertEqual(json.loads(out.getvalue()), READY)
        seccomp.assert_called_once()
        self.assertIn(((limits.RLIMIT_AS, (512 << 20, 512 << 20)), {}), limits.setrlimit.call_args_list)
        self.assertIn(((limits.RLIMIT_CPU, (4, 4)), {}), limits.setrlimit.call_args_list)
        self.assertEqual(execute.call_args.args[0], "/usr/bin/bash")
        self.assertIs(local_linux.launch.__defaults__[0], local_linux.landlock)

    def test_runtime_and_no_new_privs_fail_closed(self):
        with patch.object(Path, "resolve", return_value=Path("/tmp/arbitrary/python")), \
             self.assertRaises(local_linux.IsolationError):
            ephemeral_vm.without_landlock(Path("."), Path("."))
        libc = Mock()
        libc.prctl.return_value = -1
        with patch.object(Path, "resolve", return_value=Path("/usr/bin/python3")), \
             patch.object(ctypes, "CDLL", return_value=libc), self.assertRaises(local_linux.IsolationError):
            ephemeral_vm.without_landlock(Path("."), Path("."))

    def test_wrong_receipt_refused(self):
        env = object.__new__(ephemeral_vm.EphemeralVMEnv)
        for receipt in ({**READY, "seccomp": "off"}, {**READY, "landlock_abi": 7},
                        {**READY, "namespaces": ["user"]}):
            env.isolation = receipt
            with patch.object(local_linux.LocalLinuxEnv, "run", return_value=(0, "")), \
                 self.assertRaises(local_linux.IsolationError):
                env.run("true", 1)

    def test_commands_official_tests_final_verifier_and_trace_share_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "cache")
            args = SimpleNamespace(tasks_dir=root / "cache", runtime=root, max_tokens=4096, total_tokens=16384,
                                   seed=0, max_steps=3, timeout=10, cmd_timeout=1, verify_timeout=5,
                                   reference=None, sandbox_backend="ephemeral-vm")
            class Env:
                version = ephemeral_vm.VERSION
                def __init__(self, work, runtime):
                    self.isolation = READY
                def run(self, command, timeout_s):
                    return 0, "synthetic"
                def close(self):
                    pass
            grade = {"status": "graded", "passed": True, "tests": 1, "tests_expected": 1,
                     "tests_complete": True, "calls": [{"complete": True}], "errors": 0,
                     "failures": 0, "broken": [], "seconds": 0}
            client = FakeClient([answer(f"```bash\n{command}\n```", 1)
                                 for command in ("true", TEST, f"echo {SUBMIT}")])
            with patch.object(code_pilot, "select_backend", return_value=Env), \
                 patch.object(code_pilot, "verify", return_value=grade) as verifier:
                row = code_pilot.run_one(args, "affine-cipher", "single", ["fake"], client, root, "synthetic")
            self.assertTrue(row["passed"], row)
            self.assertEqual(row["isolation"], READY)
            self.assertEqual(verifier.call_count, 2)
            for call in verifier.call_args_list:
                self.assertIs(call.kwargs["environment"], Env)
            trace = [json.loads(line) for line in (root / "single__affine-cipher.jsonl").read_text().splitlines()]
            self.assertTrue(all(item["sandbox_backend"] == "ephemeral-vm" for item in trace))

    def test_campaign_manifest_contains_established_backend_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = code_pilot.parser().parse_args(["--sandbox-backend", "ephemeral-vm", "--single", "fake",
                "--runtime", sys.prefix, "--output", str(root), "--provenance", str(root / "fake.json")])
            class Env:
                version = ephemeral_vm.VERSION
                def __init__(self, work, runtime):
                    self.isolation = READY
                def run(self, command, timeout_s):
                    return 0, "MYRIAD-ISOLATION-OK"
                def close(self):
                    pass
            with patch.object(code_pilot, "select_backend", return_value=Env), \
                 patch.object(code_pilot, "check_tasks"), patch.object(sys, "platform", "linux"), \
                 patch.object(code_pilot, "instructions", return_value="synthetic"), \
                 patch.object(code_pilot, "declarations", return_value={}), \
                 patch.object(code_pilot, "validate_endpoint"), \
                 patch.object(code_pilot.importlib.metadata, "version", return_value="synthetic"), \
                 patch.object(code_pilot, "smoke", return_value={}) as smoke, \
                 patch.object(code_pilot, "run_one", return_value={"status": "error", "passed": None}), \
                 patch.object(code_pilot, "DeadlineClient"):
                self.assertEqual(code_pilot.execute(args), 2)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["sandbox_backend"], "ephemeral-vm")
            self.assertEqual(manifest["sandbox"], "ephemeral-vm-v1")
            self.assertEqual(manifest["isolation"], READY)
            self.assertIn("ephemeral_vm.py", manifest["code_hashes"])
            self.assertIn("sandbox.py", manifest["code_hashes"])
            self.assertIs(smoke.call_args.kwargs["environment"], Env)


@unittest.skipUnless(sys.platform == "linux", "synthetic Python worker subprocess requires Linux/WSL")
class Worker(unittest.TestCase):
    def test_ephemeral_runtime_with_explicit_synthetic_colab_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, runtime = root / "work", root / "runtime"
            work.mkdir()
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin/python").symlink_to("/usr/bin/python3")
            launcher = root / "synthetic_launcher.py"
            launcher.write_text(
                "import sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(code_pilot.ROOT)!r})\n"
                "from essaim import ephemeral_vm as vm, local_linux as ll\n"
                "vm.require_colab = lambda: {'synthetic': True}\n"
                "if sys.argv[1] == '--enter': ll.enter(int(sys.argv[2]), str(Path(__file__).resolve()))\n"
                "else: ll.launch(vm.without_landlock, vm.VERSION)\n")
            with patch.object(ephemeral_vm, "require_colab", return_value={"synthetic": True}):
                env = ephemeral_vm.EphemeralVMEnv(work, runtime)
            env.launcher = launcher
            try:
                script = ("import json,resource; print(json.dumps([resource.getrlimit(resource.RLIMIT_AS),"
                          "resource.getrlimit(resource.RLIMIT_CPU)]))")
                code, out = env.run("python -c " + shlex.quote(script), 5)
                self.assertEqual(code, 0, out)
                self.assertEqual(json.loads(out), [[512 << 20, 512 << 20], [5, 5]])
                self.assertEqual(env.isolation, READY)
                code, out = env.run("python -c 'import socket; socket.socket()'", 5)
                self.assertNotEqual(code, 0)
                self.assertIn("Operation not permitted", out)
                self.assertNotEqual(env.run("unshare --user --map-root-user true", 5)[0], 0)
                self.assertEqual(env.run("sleep 30", .4)[0], 124)
                invalid = root / "invalid"
                (invalid / "bin").mkdir(parents=True)
                (invalid / "bin/python").symlink_to("/bin/echo")
                env.runtime = invalid
                with self.assertRaises(local_linux.IsolationError):
                    env.run("touch forbidden", 5)
                self.assertFalse((work / "forbidden").exists())
            finally:
                env.close()

    def test_real_entrypoint_refuses_wsl_even_with_user_environment(self):
        if "microsoft" not in ephemeral_vm.platform.release().lower():
            self.skipTest("explicit WSL refusal test")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for action in (["--enter", str(os.getpid())], ["--launch"]):
                result = subprocess.run([sys.executable, "-I", str(ephemeral_vm.EphemeralVMEnv.launcher), *action],
                    input=json.dumps({"work": directory, "runtime": directory, "cpu": 1,
                                      "cmd": "touch forbidden"}) + "\n", cwd=work,
                    env={**os.environ, "COLAB_GPU": "1", "COLAB_RELEASE_TAG": "synthetic"},
                    capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires a detected Colab VM", result.stderr)
                self.assertFalse((work / "forbidden").exists())

    def test_reused_hook_refuses_external_files_processes_network_and_ctypes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            secret = root / "synthetic-private"
            secret.write_text("synthetic")
            for code in (f"open({str(secret)!r}).read()", "__import__('os').system('true')",
                         "__import__('socket').socket()", "__import__('ctypes').CDLL(None)"):
                script = work / "worker.py"
                script.write_text(ephemeral_vm.candidate_guard(work) + code)
                result = subprocess.run([sys.executable, "-I", str(script)], cwd=work,
                                        capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("PermissionError", result.stderr)
            script.write_text(ephemeral_vm.candidate_guard(work) + aider_python.CALL_WORKER)
            (work / "synthetic.py").write_text("def add(a,b): return a+b")
            result = subprocess.run([sys.executable, str(script)], cwd=work, input=json.dumps(
                {"module": "synthetic", "function": "add", "args": [2, 2], "kwargs": {}}),
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((work / "_return.json").read_text()), {"value": 4})


if __name__ == "__main__":
    unittest.main()
