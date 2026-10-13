"""Synthetic local transport, lifecycle and fail-closed DEV checks; no real server or scientific assets."""
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import e17_local as local


class LocalTests(unittest.TestCase):
    def test_stage_reads_only_fixed_dev_files_and_checks_hashes(self):
        from essaim import aider_python
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root / "cache", root / "output"
            output.mkdir()
            files = {"README.md": b"synthetic attribution", "LICENSE.python": b"synthetic licence"}
            for task in local.TASKS:
                files[f"python/exercises/practice/{task}/solution.py"] = b"synthetic skeleton"
            for name, data in files.items():
                path = cache / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            metadata = {"files": {n: {"size": len(b), "sha256": local.digest(b)} for n, b in files.items()},
                        "tasks": local.TASKS, "pilot": local.TASKS}
            metadata["files"]["python/exercises/practice/unopened/poison.py"] = {"size": 1, "sha256": "0" * 64}
            with mock.patch.object(aider_python, "provenance", return_value=metadata), \
                    mock.patch.object(aider_python, "MANIFEST", aider_python.MANIFEST):
                data, view = local.stage_dev(cache, output)
                self.assertEqual(set(view["files"]), set(files))
                self.assertFalse((data / "python/exercises/practice/unopened").exists())
                (cache / "README.md").write_bytes(b"mutated")
                second = root / "second"
                second.mkdir()
                with self.assertRaises(ValueError):
                    local.stage_dev(cache, second)

    def test_run_fixed_r00_modes_and_incomplete_never_complete(self):
        from essaim import code_pilot, local_linux
        for mode, status, closed, exit_error in (("single", "graded", True, False),
                ("vote", "graded", True, False), ("single", "error", True, False),
                ("single", "graded", False, False), ("single", "graded", True, True)):
            complete = status == "graded" and closed and not exit_error
            with self.subTest(mode=mode, status=status, closed=closed, exit_error=exit_error), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = SimpleNamespace(mode=mode, action="run", output=root / "fresh", runtime=Path(sys.prefix),
                    binary="synthetic", archive="synthetic", models_dir="synthetic", windows_python=Path("fake"),
                    tasks_dir=root)
                env = mock.Mock(isolation={"landlock_abi": 7, "seccomp": "on"})
                env.run.return_value = (0, "E17-STRICT-OK")
                client = mock.Mock(loading_seconds=1, request_seconds=2)
                client.receive.side_effect = [{"ready": {"archive": local.ASSET}, "session": "0" * 32, "id": 0},
                                              {"closed": closed, "session": "0" * 32, "id": 4}]
                if exit_error:
                    client.close.side_effect = RuntimeError("worker exited nonzero after closed")
                row = {"status": status, "transcript_sha256": "0" * 64, "passed": False}
                with mock.patch.object(local.sys, "platform", "linux"), \
                        mock.patch.object(local.platform, "release", return_value="synthetic-microsoft"), \
                        mock.patch.object(local, "win_path", return_value="synthetic"), \
                        mock.patch.object(local, "PipeClient", return_value=client), \
                        mock.patch.object(local, "stage_dev", return_value=(root, {"synthetic": True})), \
                        mock.patch.object(local, "validate_smoke"), \
                        mock.patch.object(importlib.metadata, "version", return_value="synthetic"), \
                        mock.patch.object(local_linux, "LocalLinuxEnv", return_value=env), \
                        mock.patch.object(code_pilot, "smoke", return_value={"synthetic": True}), \
                        mock.patch.object(code_pilot, "run_one", return_value=row) as run:
                    if not complete:
                        with self.assertRaises(RuntimeError):
                            local.execute(args)
                    else:
                        self.assertEqual(local.execute(args), 0)
                    registry = json.loads((args.output / "registry.json").read_text())
                    self.assertEqual(registry["complete"], complete)
                    self.assertFalse((args.output / "summary.json").exists())
                    self.assertEqual(run.call_count, 3 if status == "graded" else 1)
                    for call in run.call_args_list:
                        for name, value in local.LIMITS.items():
                            self.assertEqual(getattr(call.args[0], name), value)
                        self.assertEqual(call.args[2], mode)
                        self.assertEqual(call.args[3], [local.QWEN] if mode == "single" else
                                         [local.QWEN, local.GRANITE])
                    self.assertEqual(client.close.call_count, 2 if exit_error else 1)
                    with self.assertRaises(FileExistsError):
                        local.execute(args)

    def test_smoke_refuses_partial_green_and_missing_landlock(self):
        receipt = {"seccomp": "on", "landlock_abi": 7}
        value = {"sandbox_backend": "strict", "model_measurement": False, "tasks": []}
        for task, count in (("affine-cipher", 16), ("beer-song", 8), ("book-store", 20)):
            grade = {"status": "graded", "tests_expected": count, "tests": count, "tests_complete": True,
                     "calls": [{"sandbox_backend": "strict", "isolation": receipt}]}
            value["tasks"].append({"task": task, "reference": {**grade, "passed": True},
                                   "incorrect": {**grade, "passed": False}})
        local.validate_smoke(value)
        for mutate in (lambda v: v["tasks"][0]["reference"].update(tests=1),
                       lambda v: v["tasks"][0]["reference"]["calls"][0].update(isolation={}),
                       lambda v: v.update(sandbox_backend="ephemeral-vm")):
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                local.validate_smoke(changed)

    def test_windows_switch_stops_our_previous_process_and_reuses_bounded_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama-server.exe"
            binary.write_bytes(b"synthetic")
            response = SimpleNamespace(status_code=200, json=lambda: {"model": local.QWEN},
                                       raise_for_status=lambda: None)
            http = mock.MagicMock()
            http.__enter__.return_value.get.return_value = response
            factory = mock.Mock(return_value=SimpleNamespace(post=lambda *a, **k: response))
            process = mock.Mock(pid=123)
            process.poll.return_value = None
            process.returncode = 0
            with mock.patch.object(local.subprocess, "Popen", return_value=process) as launch, \
                    mock.patch.object(local.subprocess, "CREATE_NO_WINDOW", 0, create=True), \
                    mock.patch("httpx.Client", return_value=http):
                client = local.WindowsClient(binary, {m: local.MODELS[m] for m in (local.QWEN, local.GRANITE)},
                                             root, root, time.monotonic() + 30, factory)
                try:
                    client.post("chat/completions", {"model": local.QWEN}, 5)
                    response.json = lambda: {"model": local.GRANITE}
                    client.post("chat/completions", {"model": local.GRANITE}, 5)
                    self.assertEqual(process.terminate.call_count, 1)
                    self.assertEqual(launch.call_count, 2)
                    for call in launch.call_args_list:
                        args = call.args[0]
                        self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
                        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
                        self.assertEqual(args[args.index("-np") + 1], "1")
                    self.assertEqual(len(list(root.glob("server-*.log"))), 2)
                    process.poll.return_value = 1
                    with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                        client.post("chat/completions", {"model": local.QWEN}, 5)
                finally:
                    client.close()
                self.assertEqual(process.terminate.call_count, 3)
                self.assertTrue(client.events.closed)

    def test_archive_mismatch_refuses_before_binary_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "synthetic.zip"
            archive.write_bytes(b"bad")
            with mock.patch.object(local.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    local.inspect_assets({"binary": str(archive), "archive": str(archive)})
                run.assert_not_called()

    def test_inspection_checks_installed_dll_models_and_observed_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, binary = root / "synthetic.zip", root / "llama-server.exe"
            binary.write_bytes(b"synthetic exe")
            library = root / "ggml-vulkan.dll"
            library.write_bytes(b"synthetic dll")
            with zipfile.ZipFile(archive, "w") as packed:
                packed.write(binary, binary.name)
                packed.write(library, library.name)
            model = root / "synthetic.gguf"
            model.write_bytes(b"synthetic weights")
            asset = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": local.file_hash(archive)}
            info = {"file": model.name, "bytes": model.stat().st_size, "sha256": local.file_hash(model)}
            config = {"archive": str(archive), "binary": str(binary), "models": [local.QWEN],
                      "models_dir": str(root)}
            outputs = [SimpleNamespace(stdout="11505 ff5888f99", stderr=""),
                       SimpleNamespace(stdout="Vulkan0: AMD Radeon RX 6650 XT (8176 MiB, 7378 MiB free)", stderr="")]
            with mock.patch.object(local, "ASSET", asset), mock.patch.dict(local.MODELS, {local.QWEN: info}), \
                    mock.patch.object(local.subprocess, "run", side_effect=outputs) as run:
                observed = local.inspect_assets(config)
                self.assertEqual(observed["binary_sha256"], local.file_hash(binary))
                self.assertEqual(observed["server_files"][library.name], local.file_hash(library))
                self.assertEqual(run.call_count, 2)
                library.write_bytes(b"wrong dll")
                with self.assertRaises(ValueError):
                    local.inspect_assets(config)
                self.assertEqual(run.call_count, 2)

    @unittest.skipUnless(sys.platform.startswith("linux"), "selectable pipe tests run in WSL")
    def test_real_pipe_roundtrip_and_deadline_eof_cleanup(self):
        script = '''import json,sys,time
config=json.loads(sys.stdin.readline())
print(json.dumps({'ready':config,'session':config['session'],'id':0}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request.get('hang'): time.sleep(30)
    print(json.dumps({'outcome':'response','response':request['json'],'session':request['session'],
                      'id':request['id'],'loading_seconds':1,'request_seconds':2}),flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "fake.py"
            helper.write_text(script)
            with (root / "closed.log").open("wb") as log:
                client = local.PipeClient([sys.executable, str(helper)], {"synthetic": True, "execute": True}, log)
                self.assertTrue(client.receive(5)["ready"]["synthetic"])
                response = client.post("chat/completions", {"model": local.QWEN}, 5)
                self.assertEqual(response.json(), {"model": local.QWEN})
                self.assertEqual((client.loading_seconds, client.request_seconds), (1, 2))
                with self.assertRaises(ValueError):
                    client.post("chat/completions", {"model": "forbidden"}, 5)
                client.close()
                self.assertEqual(client.process.returncode, 0)
                hung = local.PipeClient([sys.executable, str(helper)], {"execute": True}, log)
                hung.receive(5)
                hung.send({"hang": True})
                with self.assertRaises(TimeoutError):
                    hung.receive(.05)
                # This fake intentionally lacks the Windows EOF watchdog; closure cannot claim success.
                with mock.patch.object(hung.process, "wait", side_effect=subprocess.TimeoutExpired("fake", 10)):
                    with self.assertRaisesRegex(RuntimeError, "exit unconfirmed"):
                        hung.close()
                hung.process.kill()
                hung.process.wait()

    @unittest.skipUnless(os.name == "nt", "Windows job ownership test")
    def test_real_windows_job_eof_kills_our_child_only(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        script = '''import sys,subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import e17_local as local
local.inspect_assets=lambda cfg: {'pid':subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']).pid}
local.WindowsClient=lambda *a,**k: None
local.worker()
'''
        outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        helper = subprocess.Popen([sys.executable, "-c", script, str(local.ROOT)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        handle = None
        try:
            helper.stdin.write(local.canonical({"models": [local.QWEN], "execute": True, "binary": "fake",
                "models_dir": "fake", "output": "fake", "session": "0" * 32}) + b"\n")
            helper.stdin.flush()
            ready = json.loads(helper.stdout.readline())
            handle = kernel.OpenProcess(0x100000, False, ready["ready"]["pid"])
            self.assertTrue(handle)
            helper.stdin.close()
            self.assertEqual(helper.wait(timeout=10), 143)
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            self.assertIsNone(outsider.poll())
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.wait()
            helper.stdout.close()
            helper.stderr.close()
            outsider.terminate()
            outsider.wait()
            if handle:
                kernel.CloseHandle(handle)

    @unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("E17_WINDOWS_TEST_PYTHON"),
                         "WSL interop fixture requires an existing Windows interpreter")
    def test_wsl_to_windows_stdio_eof_watchdog_without_model(self):
        script = '''import sys,subprocess
sys.path.insert(0,sys.argv[1])
import e17_local as local
local.inspect_assets=lambda cfg: {'synthetic':True}
local.WindowsClient=lambda *a,**k: None
local.worker()
'''
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "worker.log").open("wb") as log:
                command = [os.environ["E17_WINDOWS_TEST_PYTHON"], "-c", script, local.win_path(local.ROOT)]
                client = local.PipeClient(command, {"models": [local.QWEN], "execute": True,
                    "binary": "fake", "models_dir": "fake", "output": "fake"}, log)
                self.assertEqual(client.receive(10)["ready"], {"synthetic": True})
                client.close()
                self.assertEqual(client.process.returncode, 143)

    @unittest.skipUnless(os.name == "nt", "real Windows normal finalization")
    def test_windows_normal_preflight_smoke_and_run_close_without_buffered_reader(self):
        script = '''import sys
sys.path.insert(0,sys.argv[1])
import e17_local as local
local.inspect_assets=lambda cfg: {'synthetic':True}
class Client:
    def __init__(self,*a,**k): pass
    def close(self): pass
local.WindowsClient=Client
local.worker()
'''
        for execute in (False, False, True, True, True):
            with self.subTest(execute=execute):
                process = subprocess.Popen([sys.executable, "-c", script, str(local.ROOT)], stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    config = {"models": [local.QWEN], "execute": execute, "binary": "fake",
                              "models_dir": "fake", "output": "fake", "session": "0" * 32}
                    process.stdin.write(local.canonical(config) + b"\n")
                    process.stdin.flush()
                    self.assertTrue(json.loads(process.stdout.readline())["ready"]["synthetic"])
                    if execute:
                        process.stdin.write(local.canonical({"session": "0" * 32, "id": 1, "close": True}) + b"\n")
                        process.stdin.flush()
                        self.assertTrue(json.loads(process.stdout.readline())["closed"])
                    self.assertEqual(process.wait(timeout=10), 0)
                    self.assertNotIn(b"Fatal Python error", process.stderr.read())
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    process.stdin.close()
                    process.stdout.close()
                    process.stderr.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "real selectable pipes in WSL")
    def test_late_partial_response_is_drained_and_foreign_ids_or_global_deadline_refuse_next_request(self):
        script = '''import json,sys,time
config=json.loads(sys.stdin.readline())
print(json.dumps({'ready':True,'session':config['session'],'id':0}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    value=request['json']
    data=json.dumps({'session':config['session'],'id':request['id']+value.get('wrong_id',0),
        'outcome':value.get('outcome','response'),'response':value,
        'loading_seconds':request['id'],'request_seconds':request['id']})
    sys.stdout.write(data[:10]);sys.stdout.flush()
    time.sleep(value.get('delay',0))
    print(data[10:],flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "fake.py"
            helper.write_text(script)
            with (root / "worker.log").open("wb") as log:
                client = local.PipeClient([sys.executable, str(helper)], {"execute": True}, log)
                try:
                    client.receive(5)
                    with self.assertRaises(TimeoutError):
                        client.post("chat/completions", {"model": local.QWEN, "prompt": "old", "delay": .2}, .05)
                    self.assertTrue(client.buffer)
                    response = client.post("chat/completions", {"model": local.GRANITE, "prompt": "new"}, 2)
                    self.assertEqual(response.json()["prompt"], "new")
                    self.assertEqual(client.sequence, 2)
                    with self.assertRaises(ValueError):
                        client.post("chat/completions", {"model": local.QWEN, "wrong_id": 1}, 2)
                    with self.assertRaises(RuntimeError):
                        client.post("chat/completions", {"model": local.QWEN}, 2)
                    self.assertEqual(client.sequence, 3)
                finally:
                    client.close()
                bounded = local.PipeClient([sys.executable, str(helper)], {"execute": True}, log)
                try:
                    bounded.receive(5)
                    bounded.deadline = time.monotonic() + .05
                    with self.assertRaises(TimeoutError):
                        bounded.post("chat/completions", {"model": local.QWEN, "delay": .2}, 2)
                    started = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        bounded.post("chat/completions", {"model": local.GRANITE}, 2)
                    self.assertLess(time.monotonic() - started, .2)
                    self.assertEqual(bounded.sequence, 1)
                finally:
                    bounded.close()
                failed = local.PipeClient([sys.executable, str(helper)], {"execute": True}, log)
                try:
                    failed.receive(5)
                    with self.assertRaises(TimeoutError):
                        failed.post("chat/completions", {"model": local.QWEN, "delay": .2,
                                                        "outcome": "infrastructure"}, .05)
                    with self.assertRaisesRegex(RuntimeError, "infrastructure"):
                        failed.post("chat/completions", {"model": local.GRANITE}, 2)
                    self.assertEqual(failed.sequence, 1)
                finally:
                    failed.close()

    @unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("E17_WINDOWS_TEST_PYTHON"),
                         "real Windows HTTP timeout and next episode through WSL")
    def test_http_generation_timeout_keeps_worker_and_next_episode_has_its_own_response(self):
        from essaim import aider_agent
        from essaim.code_pilot import BudgetChat
        server = '''import http.server,json,time
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if value['messages'][-1]['content']=='first': time.sleep(.8)
        data=json.dumps({'model':value['model'],'choices':[{'finish_reason':'stop',
            'message':{'content':'```bash\\nprintf SECOND\\n```'}}],'usage':{'completion_tokens':2}}).encode()
        self.send_response(200);self.end_headers()
        try: self.wfile.write(data)
        except OSError: pass
server=http.server.HTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
'''
        script = '''import sys,subprocess,time
sys.path.insert(0,sys.argv[1])
import e17_local as local
import httpx
class Client:
    loading_seconds=request_seconds=0
    def __init__(self,*a,**k): self.process=None
    def post(self,path,json,timeout):
        if self.process is None:
            self.process=subprocess.Popen([sys.executable,'-c',sys.argv[2]],stdout=subprocess.PIPE)
            port=int(self.process.stdout.readline())
            self.client=local.loopback_client('http://127.0.0.1:'+str(port)+'/v1','')
        started=time.monotonic()
        limit=min(timeout,.05) if json['messages'][-1]['content']=='first' else timeout
        try: return self.client.post(path,json=json,timeout=limit)
        finally: self.request_seconds+=time.monotonic()-started
    def stop(self):
        if self.process:
            self.process.terminate();self.process.wait(timeout=5);self.process.stdout.close();self.process=None
    def close(self): self.stop()
local.inspect_assets=lambda cfg: {'synthetic':True}
local.WindowsClient=Client
local.worker()
'''
        env = SimpleNamespace(run=lambda *a: (0, "SECOND"))
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "worker.log").open("wb") as log:
                client = local.PipeClient([os.environ["E17_WINDOWS_TEST_PYTHON"], "-c", script,
                    local.win_path(local.ROOT), server], {"models": [local.QWEN], "execute": True,
                    "binary": "fake", "models_dir": "fake", "output": "fake"}, log)
                try:
                    client.receive(10)
                    process = client.process
                    first_events, second_events = [], []
                    for task, events, expected in (("first", first_events, "timeout"),
                                                   ("second", second_events, "max_steps")):
                        chat = BudgetChat(client, 4096, 49152, 0, time.monotonic() + 5, events.append, strict=True)
                        episode = aider_agent.run_episode(chat, env, task, [local.QWEN], max_steps=1,
                            timeout_s=5, snapshot=lambda: {"synthetic.py": "unchanged"}, test_sources=lambda *a: {})
                        self.assertEqual(episode["stopped"], expected)
                        client.synchronize()
                    self.assertIs(client.process, process)
                    self.assertEqual(first_events[0]["exception_type"], "GenerationDeadline")
                    self.assertNotIn("reply", first_events[0])
                    self.assertIn("SECOND", second_events[0]["reply"])
                    client.send({"close": True})
                    self.assertTrue(client.receive(5)["closed"])
                finally:
                    client.close()
                self.assertEqual(client.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
