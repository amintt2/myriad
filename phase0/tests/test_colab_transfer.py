"""Offline synthetic transfer smoke: real wrapper and processes, entirely fake official CLI."""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

COLAB = Path(__file__).resolve().parents[1] / "colab"


def load(name):
    spec = importlib.util.spec_from_file_location(name, COLAB / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load("reconstruct")


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


class Reconstruction(unittest.TestCase):
    def fixture(self, root):
        attempt = "a" * 32
        prefix = "dllm-transfer-" + attempt
        parts, whole = [], hashlib.sha256()
        block = b"synthetic archive " * 4096
        for i, size in enumerate((remote.CHUNK, 131072)):
            path = root / f"{prefix}.part-{i:06d}"
            hashed = hashlib.sha256()
            with path.open("wb") as output:
                remaining = size
                while remaining:
                    data = block[:min(remaining, len(block))]
                    output.write(data)
                    hashed.update(data)
                    whole.update(data)
                    remaining -= len(data)
            parts.append({"index": i, "name": path.name, "size": size, "sha256": hashed.hexdigest()})
        manifest = {"schema": 1, "attempt": attempt, "size": sum(p["size"] for p in parts),
                    "sha256": whole.hexdigest(), "parts": parts}
        config = {k: manifest[k] for k in ("attempt", "size", "sha256")}
        return manifest, config

    def publish(self, root, manifest, config):
        raw = json.dumps(manifest).encode()
        (root / f'dllm-transfer-{config["attempt"]}.json').write_bytes(raw)
        return {**config, "manifest_sha256": hashlib.sha256(raw).hexdigest()}

    def test_streaming_exact_hash_and_known_cleanup_only(self):
        with tempfile.TemporaryDirectory(dir=COLAB.parents[1]) as directory:
            root = Path(directory)
            manifest, config = self.fixture(root)
            config = self.publish(root, manifest, config)
            sentinel = root / "model-and-results-sentinel"
            sentinel.write_bytes(b"preserve")
            (root / "dllm.tgz").write_bytes(b"old archive")
            tracemalloc.start()
            try:
                result = remote.reconstruct(config, root)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 4 * 1024 * 1024)
            self.assertEqual(result, {"schema": 1, "status": "ok", **config})
            self.assertEqual(digest(root / "dllm.tgz"), config["sha256"])
            self.assertEqual((root / "dllm.tgz").stat().st_size, config["size"])
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertEqual(set(p.name for p in root.iterdir()), {"dllm.tgz", sentinel.name})

    def test_invalid_parts_never_replace_or_cleanup(self):
        for mode in ("missing", "corrupt", "truncated", "order", "index", "outside", "absolute", "size",
                     "whole", "manifest", "mixed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=COLAB.parents[1]) as directory:
                root = Path(directory)
                manifest, config = self.fixture(root)
                path = root / manifest["parts"][0]["name"]
                if mode == "missing": path.unlink()
                if mode == "corrupt":
                    with path.open("r+b") as output: output.write(b"bad")
                if mode == "truncated":
                    with path.open("r+b") as output: output.truncate(10)
                if mode == "order": manifest["parts"].reverse()
                if mode == "index": manifest["parts"][0]["index"] = -1
                if mode == "outside": manifest["parts"][0]["name"] = "../sentinel"
                if mode == "absolute": manifest["parts"][0]["name"] = "/content/sentinel"
                if mode == "size": manifest["parts"][0]["size"] -= 1
                if mode == "whole": manifest["sha256"] = config["sha256"] = "b" * 64
                if mode == "mixed": manifest["attempt"] = "b" * 32
                config = self.publish(root, manifest, config)
                if mode == "manifest": config["manifest_sha256"] = "0" * 64
                target = root / "dllm.tgz"
                target.write_bytes(b"existing archive")
                before = set(root.iterdir())
                with self.assertRaises((ValueError, FileNotFoundError)):
                    remote.reconstruct(config, root)
                self.assertEqual(target.read_bytes(), b"existing archive")
                self.assertEqual(set(root.iterdir()), before)

    def test_published_archive_resumes_partial_cleanup_with_fresh_ack(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, config = self.fixture(root)
                config = {**self.publish(root, manifest, config), "challenge": "first"}
                original_unlink = Path.unlink
                removed = []

                def interrupted(path, *args, **kwargs):
                    if ".part-" in path.name:
                        if removed:
                            raise InterruptedError("synthetic cleanup interruption")
                        removed.append(path.name)
                    return original_unlink(path, *args, **kwargs)

                with mock.patch.object(Path, "unlink", interrupted):
                    with self.assertRaises(InterruptedError): remote.reconstruct(config, root)
                self.assertTrue((root / f'dllm-transfer-{config["attempt"]}.json').exists())
                self.assertEqual(digest(root / "dllm.tgz"), config["sha256"])
                sentinel = root / ("dllm-transfer-" + "b" * 32 + ".part-000000")
                sentinel.write_bytes(b"other attempt")
                if corrupt:
                    (root / "dllm.tgz").write_bytes(b"corrupt published archive")
                    with self.assertRaises(FileNotFoundError): remote.reconstruct(config, root)
                else:
                    fresh = {**config, "challenge": "second"}
                    self.assertEqual(remote.reconstruct(fresh, root), {"schema": 1, "status": "ok", **fresh})
                    self.assertEqual(remote.reconstruct(fresh, root), {"schema": 1, "status": "ok", **fresh})
                    self.assertEqual(set(root.iterdir()), {root / "dllm.tgz", sentinel})
                self.assertEqual(sentinel.read_bytes(), b"other attempt")

    @unittest.skipUnless(sys.platform == "linux", "symlink test requires Linux")
    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory(dir=COLAB.parents[1]) as directory:
            root = Path(directory)
            manifest, config = self.fixture(root)
            path = root / manifest["parts"][0]["name"]
            path.rename(root / "sentinel")
            path.symlink_to(root / "sentinel")
            config = self.publish(root, manifest, config)
            with self.assertRaises(ValueError): remote.reconstruct(config, root)

    def test_redaction_unknown_signed_queries_and_bearer(self):
        safe = load("safe_cli")
        raw = ('HTTP 500 https://synthetic.invalid/path?colab-runtime-proxy-token=SYNTHETIC_ONE&x=2 '
               'access_token=SYNTHETIC_TWO token: "SYNTHETIC_THREE" api_key=SYNTHETIC_FOUR '
               'Authorization: Bearer SYNTHETIC_FIVE')
        text = safe.redact(raw)
        for name in ("ONE", "TWO", "THREE", "FOUR", "FIVE"):
            self.assertNotIn("SYNTHETIC_" + name, text)
        self.assertIn("HTTP 500", text)
        self.assertNotIn("SYNTHETIC_UNKNOWN", safe.redact('https://synthetic.invalid/?signature=SYNTHETIC_UNKNOWN'))
        out = io.StringIO()
        safe.emit(io.BytesIO(("x" * 65534 + "token=SYNTHETIC_SPLIT\nHTTP 500\n").encode()), out)
        self.assertNotIn("SYNTHETIC_SPLIT", out.getvalue())
        self.assertIn("omitted", out.getvalue())
        self.assertIn("HTTP 500", out.getvalue())

    def test_local_splitter_memory_stays_bounded(self):
        with mock.patch.dict(sys.modules, {"reconstruct": remote, "safe_cli": load("safe_cli")}):
            transfer = load("transfer")
        with tempfile.TemporaryDirectory(dir=COLAB.parents[1]) as directory:
            root = Path(directory)
            content = root / "content"
            content.mkdir()
            archive = root / "synthetic archive"
            with archive.open("wb") as output:
                for _ in range(25): output.write(b"s" * (1024 * 1024))
            sizes = []
            def fake_cli(args, output=None):
                if args[0] == "upload":
                    path = Path(args[3])
                    sizes.append(path.stat().st_size)
                    shutil.copyfile(path, content / Path(args[4]).name)
                else:
                    import ast
                    config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(" = ", 1)[1])
                    if args[-1].endswith('verify_part.py'):
                        valid = remote.matches(content / config['name'], config['size'], config['sha256'])
                        output.write(('DLLM_PART_ACK ' + json.dumps({**config, 'matches': valid}) + "\n").encode())
                        return
                    result = remote.reconstruct(config, content)
                    output.write((remote.ACK + json.dumps(result) + "\n").encode())
            tracemalloc.start()
            try:
                with mock.patch.object(transfer, "cli", side_effect=fake_cli), mock.patch("builtins.print"):
                    transfer.transfer(archive)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 4 * 1024 * 1024)
            self.assertTrue(all(size <= remote.CHUNK for size in sizes))
            self.assertEqual(len(sizes), 5)  # Four parts and the manifest.
            self.assertEqual(digest(archive), digest(content / "dllm.tgz"))


CLI = r'''import ast, hashlib, json, os, shutil, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
content = root / 'content'
content.mkdir(exist_ok=True)
args, mode = sys.argv[1:], os.environ['MODE']
with (root / 'calls').open('a') as output: output.write(args[0] + '\n')
if args[0] == 'sessions':
    print('[phase0] synthetic | Hardware: A100 | Shape: Standard | Variant: GPU'
          if (root / 'active').exists() else '[colab] No active sessions found on server.')
elif args[0] == 'usage':
    print('Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0')
elif args[0] == 'new': (root / 'active').touch()
elif args[0] == 'stop': (root / 'active').unlink()
elif args[0] == 'upload':
    assert Path(args[4]).parent == Path('/content')
    source = Path(args[3])
    assert source.stat().st_size <= 8 * 1024 * 1024
    print('HTTP 500 token=SYNTHETIC_STDOUT' if mode == 'upload-error' else 'upload OK')
    print('HTTP 500 https://synthetic.invalid/?colab-runtime-proxy-token=SYNTHETIC_STDERR', file=sys.stderr)
    if mode == 'upload-error': sys.exit(29)
    if mode == 'signal':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = os.fork()
        if child == 0: time.sleep(60); sys.exit(0)
        (root / 'pids.tmp').write_text(json.dumps([os.getpid(), child]))
        (root / 'pids.tmp').replace(root / 'pids')
        time.sleep(60)
    shutil.copyfile(source, content / Path(args[4]).name)
elif args[0] == 'download': sys.exit(31)
elif args[0] == 'exec':
    if args[-1].endswith('capacity.py'):
        print('DLLM_CAPACITY ' + json.dumps({'free_mib': 40960}))
    elif args[-1].endswith('verify_part.py'):
        code = Path(args[-1]).read_text()
        config = ast.literal_eval(code.splitlines()[0].split(' = ', 1)[1])
        path = content / config['name']
        valid = (path.exists() and path.stat().st_size == config['size']
                 and hashlib.sha256(path.read_bytes()).hexdigest() == config['sha256'])
        print('DLLM_PART_ACK ' + json.dumps({**config, 'matches': valid}))
    elif args[-1].endswith('snapshot.py'):
        print(json.dumps({'schema': 1, 'campaign_present': False}))
    elif args[-1].endswith('colab_bootstrap.py'):
        assert (root / 'verified').exists()
        (root / 'boot').touch()
    elif args[-1].endswith('reconstruct.py'):
        code = Path(args[-1]).read_text()
        config = ast.literal_eval(code.splitlines()[0].split(' = ', 1)[1])
        if mode == 'no-ack': print('SystemExit: 1'); sys.exit(0)
        namespace = {'__name__': 'synthetic_reconstruction'}
        exec(compile(code, 'synthetic_reconstruction', 'exec'), namespace)
        manifest_path = content / ('dllm-transfer-' + config['attempt'] + '.json')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            (root / 'manifest-proof').write_text(json.dumps(manifest))
        if mode == 'corrupt-remote':
            with (content / manifest['parts'][0]['name']).open('r+b') as part: part.write(b'bad')
        try:
            result = namespace['reconstruct'](config, content)
        except Exception:
            print('DLLM_TRANSFER_ERROR ValueError')
            sys.exit(0)  # Match the official CLI's remote exception / local success behavior.
        with (content / 'dllm.tgz').open('rb') as archive:
            actual = hashlib.file_digest(archive, 'sha256').hexdigest()
        assert actual == config['sha256']
        (root / 'verified').write_text(actual)
        if mode == 'bad-ack': result['sha256'] = 'b' * 64
        if mode == 'stale-ack': result['attempt'] = 'b' * 32
        if mode == 'error-and-ack': print('DLLM_TRANSFER_ERROR ValueError')
        print('DLLM_TRANSFER_ACK ' + json.dumps(result))
        if mode == 'multiple-ack': print('DLLM_TRANSFER_ACK ' + json.dumps(result))
    else:
        sys.stdin.read()
        print('LISTE-OK')
else: sys.exit(81)
'''


@unittest.skipUnless(sys.platform == "linux", "real wrapper smoke requires Linux/WSL")
class WrapperTransfer(unittest.TestCase):
    def setup_root(self, root, mode, large=False):
        shutil.copytree(COLAB, root / "phase0" / "colab")
        shutil.copyfile(COLAB.parent / "colab_jobs.py", root / "phase0" / "colab_jobs.py")
        shutil.copyfile(COLAB.parent / "colab_bootstrap.py", root / "phase0" / "colab_bootstrap.py")
        data = root / "phase0" / "data"
        data.mkdir()
        with (data / "scicode_test_data.h5").open("wb") as output:
            if large:
                block = os.urandom(1024 * 1024)
                for _ in range(17): output.write(block)
            else:
                output.write(b"synthetic data, never executed")
        bins = root / ".local" / "bin"
        bins.mkdir(parents=True)
        fake = bins / "colab"
        fake.write_text(f"#!{sys.executable}\n" + CLI)
        fake.chmod(0o755)
        (root / "content").mkdir()
        (root / "content" / "model-sentinel").write_bytes(b"preserve")
        (root / "content" / "dllm.tgz").write_bytes(b"previous archive")
        return {**os.environ, "HOME": str(root), "TEST_ROOT": str(root), "MODE": mode,
                "DLLM_REPO": str(root), "DLLM_CAMPAIGN_RECEIPT": str(root / "receipt"),
                "DLLM_CAMPAIGN_ID": "synthetic-campaign", "DLLM_BUDGET_UNITS": "20", "DLLM_RETRY_SECONDS": ".01"}

    def test_real_wrapper_multi_part_and_ack_failures(self):
        for mode, expected in (("success", 0), ("upload-error", 29), ("no-ack", 65), ("bad-ack", 65),
                               ("stale-ack", 65), ("corrupt-remote", 65), ("error-and-ack", 65),
                               ("multiple-ack", 65)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="transfer spaces ", dir=COLAB.parents[1]) as directory:
                root = Path(directory)
                env = self.setup_root(root, mode, large=mode == "success")
                wrapper = root / "phase0" / "colab" / "colab_phase0.sh"
                result = subprocess.run(["bash", str(wrapper), "up", "aa-1", "A100"], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertNotIn("SYNTHETIC_STD", result.stdout + result.stderr)
                self.assertIn("HTTP 500", result.stderr)
                self.assertEqual((root / "boot").exists(), mode == "success")
                self.assertEqual((root / "content" / "model-sentinel").read_bytes(), b"preserve")
                if mode in ("upload-error", "no-ack", "corrupt-remote"):
                    self.assertEqual((root / "content" / "dllm.tgz").read_bytes(), b"previous archive")
                if mode == "upload-error":
                    self.assertEqual((root / "calls").read_text().splitlines().count("upload"), 3)
                    self.assertIn("colab upload: exit 29", result.stderr)
                if mode == "success":
                    import tarfile
                    manifest = json.loads((root / "manifest-proof").read_text())
                    self.assertGreater(len(manifest["parts"]), 2)
                    self.assertEqual(digest(root / "content" / "dllm.tgz"), manifest["sha256"])
                    self.assertEqual(set(p.name for p in (root / "content").iterdir()),
                                     {"dllm.tgz", "model-sentinel"})
                    with tarfile.open(root / "content" / "dllm.tgz") as archive:
                        with archive.extractfile("phase0/data/scicode_test_data.h5") as source:
                            self.assertEqual(hashlib.file_digest(source, "sha256").hexdigest(),
                                             digest(root / "phase0" / "data" / "scicode_test_data.h5"))

    def test_supervised_upload_interruption_kills_group_and_releases_owned_session(self):
        with tempfile.TemporaryDirectory(prefix="supervised transfer spaces ", dir=COLAB.parents[1]) as directory:
            root = Path(directory)
            env = self.setup_root(root, "signal")
            p = subprocess.Popen(["bash", str(root / "phase0" / "colab" / "chain4.sh"), "--lock",
                                  str(root / "lock"), "--read-seconds", "3", "--up-seconds", "10",
                                  "--cleanup-seconds", "20", "--grace", ".1"], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            pids = []
            try:
                deadline = time.monotonic() + 10
                while not (root / "pids").exists() and p.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "pids").exists())
                pids = json.loads((root / "pids").read_text())
                for pid in pids:
                    status = Path(f"/proc/{pid}/status").read_text()
                    mask = int(next(l.split()[1] for l in status.splitlines() if l.startswith("SigIgn:")), 16)
                    self.assertTrue(mask & (1 << (signal.SIGTERM - 1)))
                calls = (root / "calls").read_text()
                wrapper = root / "phase0/colab/colab_phase0.sh"
                for operation in (("resume-up", "aa-1", "A100"), ("down",)):
                    competitor = subprocess.run(["bash", str(wrapper), *operation], env=env,
                                                capture_output=True, text=True, timeout=3)
                    self.assertEqual(competitor.returncode, 73, competitor.stdout + competitor.stderr)
                    self.assertIn("lifecycle operation holds the lock", competitor.stdout)
                    self.assertEqual((root / "calls").read_text(), calls)
                p.send_signal(signal.SIGTERM)
                output, _ = p.communicate(timeout=25)
                self.assertEqual(p.returncode, 143, output)
                self.assertFalse((root / "boot").exists())
                self.assertFalse((root / "active").exists())
                self.assertIn("owned allocation released", output)
                self.assertNotIn("SYNTHETIC_STD", output)
                for pid in pids:
                    path = Path(f"/proc/{pid}/stat")
                    if path.exists():
                        self.assertIn(path.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X", "x"))
            finally:
                if p.poll() is None:
                    p.kill()
                    p.communicate()
                for pid in pids:
                    try: os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass


if __name__ == "__main__":
    unittest.main()
