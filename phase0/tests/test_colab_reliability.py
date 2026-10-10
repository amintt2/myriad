"""Real local Linux processes and fake CLI only: network ambiguity, ownership and billing evidence."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.test_colab_transfer import CLI, COLAB, WrapperTransfer, load


NETWORK = r'''
counter = root / ('count-' + args[0])
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
def dns():
    print('Temporary failure in name resolution', file=sys.stdout if mode.endswith('-stdout') else sys.stderr)
    sys.exit(27)
if args[0] == 'exec' and args[-1].endswith('verify_part.py'):
    config = ast.literal_eval(Path(args[-1]).read_text().splitlines()[0].split(' = ', 1)[1])
    with (root / 'challenges').open('a') as out: out.write(json.dumps(config) + '\n')
    if mode == 'dns-before' and n == 1: dns()
    if mode == 'dns-after' and n == 2: dns()
    if mode == 'fail-second' and config['name'].endswith('part-000001'): dns()
if args[0] == 'upload':
    with (root / 'uploads').open('a') as out: out.write(Path(args[4]).name + '\n')
    if mode == 'last-upload-lost' and n <= 3:
        if n == 3: shutil.copyfile(args[3], content / Path(args[4]).name)
        dns()
    if mode in ('lost-upload', 'lost-upload-stdout') and n == 1:
        shutil.copyfile(args[3], content / Path(args[4]).name)
        dns()
'''

TRANSFER_CLI = CLI.replace("if args[0] == 'sessions':", NETWORK + "\nif args[0] == 'sessions':")
TRANSFER_CLI = TRANSFER_CLI.replace("    shutil.copyfile(source, content / Path(args[4]).name)",
                                  "    shutil.copyfile(source, content / Path(args[4]).name)\n"
                                  "    if mode == 'corrupt-once' and n == 1:\n"
                                  "        (content / Path(args[4]).name).write_bytes(b'corrupt')")
TRANSFER_CLI = TRANSFER_CLI.replace("        print('DLLM_TRANSFER_ACK ' + json.dumps(result))",
                                  "        if mode == 'lost-reconstruction' and not (root / 'lost').exists():\n"
                                  "            (root / 'lost').touch()\n"
                                  "            dns()\n"
                                  "        print('DLLM_TRANSFER_ACK ' + json.dumps(result))")
TRANSFER_CLI = TRANSFER_CLI.replace("            result = namespace['reconstruct'](config, content)", r'''
            if mode == 'interrupted-cleanup' and not (root / 'interrupted').exists():
                original_unlink = Path.unlink
                def interrupted(path, *args, **kwargs):
                    if path.name.startswith('dllm-transfer-'):
                        if (root / 'interrupted').exists(): dns()
                        original_unlink(path, *args, **kwargs)
                        (root / 'interrupted').touch()
                        return
                    return original_unlink(path, *args, **kwargs)
                Path.unlink = interrupted
            result = namespace['reconstruct'](config, content)
''')

LIFECYCLE_CLI = r'''import hashlib, json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['TEST_ROOT'])
args, mode = sys.argv[1:], os.environ['MODE']
operation = args[0]
counter = root / ('count-' + operation)
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
with (root / 'calls').open('a') as out: out.write(operation + '\n')
def dns():
    print('Temporary failure in name resolution', file=sys.stderr)
    sys.exit(27)
active = root / 'active'
if operation == 'sessions':
    if mode == 'empty-sessions': sys.exit(0)
    if mode == 'cleanup-dns' and n <= 3 or mode == 'durable-dns': dns()
    if mode == 'stop-read-lost' and n == 2: dns()
    if mode == 'receipt-lost' and n >= 2: dns()
    if mode == 'receipt-transient' and n == 2: dns()
    endpoint = 'foreign' if mode == 'replaced' else 'synthetic'
    print(f'[phase0] {endpoint} | Hardware: A100 | Shape: Standard | Variant: GPU'
          if active.exists() else '[colab] No active sessions found on server.')
elif operation == 'new':
    active.touch()
    if mode == 'new-lost': dns()
elif operation == 'stop':
    if mode == 'stop-dns' and n <= 3: dns()
    active.unlink()
    if mode == 'stop-ack-lost': dns()
elif operation == 'usage':
    if mode == 'empty-usage' or mode == 'after-empty' and not active.exists(): sys.exit(0)
    balance = '1.00' if mode == 'low-balance' else '97.50' if mode == 'after' else '100.00'
    print(f'Current balance: {balance} compute units\nUsage rate: 0.00/hr\nActive assignments: 0')
elif operation == 'exec':
    print('DLLM_CAPACITY ' + json.dumps({'free_mib': 20000 if mode == 'low-memory' else 40960}))
else: sys.exit(81)
'''


@unittest.skipUnless(sys.platform == "linux", "real processes require Linux/WSL")
class Reliability(unittest.TestCase):
    def fixture(self, root, source, mode):
        bins = root / "bin"
        bins.mkdir()
        fake = bins / "colab"
        fake.write_text(f"#!{sys.executable}\n" + source)
        fake.chmod(0o755)
        return {**os.environ, "HOME": str(root), "PATH": str(bins) + ":" + os.environ["PATH"], "MODE": mode,
                "TEST_ROOT": str(root), "DLLM_CAMPAIGN_RECEIPT": str(root / "receipt"),
                "DLLM_BUDGET_UNITS": "20", "DLLM_RETRY_SECONDS": ".01", "DLLM_CLEANUP_SECONDS": "1.5"}

    def call(self, env, *args):
        return subprocess.run([sys.executable, str(COLAB / "session_json.py"), *args], env=env,
                              capture_output=True, text=True, timeout=8)

    def owned(self, root):
        (root / "active").touch()
        value = {"schema": 1, "status": "owned", "phase0": {
            "fingerprint": hashlib.sha256(b"synthetic").hexdigest(), "hardware": "A100"},
                 "usage_before": {"schema": 1, "balance_units": 100, "rate_units_hour": 0, "assignments": 0}}
        (root / "receipt").write_text(json.dumps(value))

    def test_chunk_dns_ambiguous_upload_corruption_and_fresh_reconstruction(self):
        for mode in ("dns-before", "dns-after", "lost-upload", "lost-upload-stdout", "corrupt-once",
                     "lost-reconstruction", "last-upload-lost", "interrupted-cleanup"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                env = self.fixture(root, TRANSFER_CLI, mode)
                archive = root / "archive"
                archive.write_bytes(b"synthetic transfer" * 10000)
                result = subprocess.run([sys.executable, str(COLAB / "transfer.py"), str(archive)], env=env,
                                        capture_output=True, text=True, timeout=12)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((root / "content/dllm.tgz").read_bytes(), archive.read_bytes())
                parts = [n for n in (root / "uploads").read_text().splitlines() if ".part-" in n]
                self.assertEqual(len(parts), 3 if mode == "last-upload-lost" else 2 if mode == "corrupt-once" else 1)
                if mode == "last-upload-lost":
                    probes = [json.loads(line) for line in (root / "challenges").read_text().splitlines()]
                    probes = [p for p in probes if ".part-" in p["name"]]
                    self.assertEqual(len(probes), 4)
                    self.assertEqual(len({p["challenge"] for p in probes}), 4)
                if mode == "interrupted-cleanup":
                    self.assertEqual(set(p.name for p in (root / "content").iterdir()), {"dllm.tgz"})

    def test_process_restart_does_not_resend_verified_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, TRANSFER_CLI, "fail-second")
            archive = root / "archive"
            archive.write_bytes(b"s" * (8 * 1024 * 1024 + 1000))
            cmd = [sys.executable, str(COLAB / "transfer.py"), str(archive)]
            first = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=12)
            self.assertEqual(first.returncode, 27, first.stdout + first.stderr)
            second = subprocess.run(cmd, env={**env, "MODE": "success"}, capture_output=True, text=True, timeout=12)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            uploads = (root / "uploads").read_text().splitlines()
            self.assertEqual(sum(n.endswith("part-000000") for n in uploads), 1)
            self.assertEqual(sum(n.endswith("part-000001") for n in uploads), 1)

    def test_manual_wrapper_resumes_the_exact_archive_without_new_allocation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
            env = helper.setup_root(root, "fail-second", large=True)
            (root / ".local/bin/colab").write_text(f"#!{sys.executable}\n" + TRANSFER_CLI)
            wrapper = root / "phase0/colab/colab_phase0.sh"
            first = subprocess.run(["bash", str(wrapper), "up", "aa-1", "A100"], env=env,
                                   capture_output=True, text=True, timeout=20)
            self.assertEqual(first.returncode, 27, first.stdout + first.stderr)
            saved = json.loads((root / "receipt").read_text())
            self.assertEqual(saved["status"], "owned")
            self.assertFalse((root / "boot").exists())
            second = subprocess.run(["bash", str(wrapper), "resume-up", "aa-1", "A100"],
                                    env={**env, "MODE": "success"}, capture_output=True, text=True, timeout=20)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertTrue((root / "boot").exists())
            uploads = (root / "uploads").read_text().splitlines()
            self.assertEqual(sum(n.endswith("part-000000") for n in uploads), 1)
            self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)
            archive = root / "receipt.archive.tgz"
            archive.write_bytes(b"corrupted local cache")
            refused = subprocess.run(["bash", str(wrapper), "resume-up", "aa-1", "A100"],
                                     env={**env, "MODE": "success"}, capture_output=True, text=True, timeout=5)
            self.assertEqual(refused.returncode, 65, refused.stdout + refused.stderr)

    def wrapper_fixture(self, root):
        helper = WrapperTransfer("test_real_wrapper_multi_part_and_ack_failures")
        env = helper.setup_root(root, "success")
        return env, root / "phase0/colab/colab_phase0.sh", root / ".local/bin/colab"

    def test_full_up_resume_and_cleanup_share_one_nonblocking_lock(self):
        gate = r'''
        if os.environ.get('GATE') == '1':
            (root / 'ready').touch()
            while not (root / 'release').exists(): time.sleep(.01)
'''
        source = CLI.replace("        (root / 'boot').touch()", gate + "        (root / 'boot').touch()")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env, wrapper, fake = self.wrapper_fixture(root)
            fake.write_text(f"#!{sys.executable}\n" + source)
            for operation in ("up", "resume-up"):
                with self.subTest(operation=operation):
                    (root / "ready").unlink(missing_ok=True)
                    (root / "release").unlink(missing_ok=True)
                    cmd = ["bash", str(wrapper), operation, "aa-1", "A100"]
                    p = subprocess.Popen(cmd, env={**env, "GATE": "1"}, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True)
                    try:
                        deadline = time.monotonic() + 10
                        while not (root / "ready").exists() and p.poll() is None and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue((root / "ready").exists())
                        calls = (root / "calls").read_text()
                        for competing in (("resume-up", "aa-1", "A100"), ("up", "aa-1", "A100"), ("down",)):
                            blocked = subprocess.run(["bash", str(wrapper), *competing], env=env,
                                                     capture_output=True, text=True, timeout=3)
                            self.assertEqual(blocked.returncode, 73, blocked.stdout + blocked.stderr)
                            self.assertIn("lifecycle operation holds the lock", blocked.stdout)
                            self.assertEqual((root / "calls").read_text(), calls)
                        (root / "release").touch()
                        out, _ = p.communicate(timeout=5)
                        self.assertEqual(p.returncode, 0, out)
                    finally:
                        (root / "release").touch()
                        if p.poll() is None: p.kill()
                        p.communicate()
            cleaned = subprocess.run(["bash", str(wrapper), "down"], env=env,
                                     capture_output=True, text=True, timeout=5)
            self.assertEqual(cleaned.returncode, 0, cleaned.stdout + cleaned.stderr)
            self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)

    def test_resume_pins_initial_receipt_even_when_same_session_receipt_is_rewritten(self):
        rewrite = r'''
        if os.environ.get('REWRITE') == '1':
            receipt = Path(os.environ['DLLM_CAMPAIGN_RECEIPT'])
            saved = json.loads(receipt.read_text())
            receipt.write_text(json.dumps({**saved, 'allocation_id': 'new-allocation'}))
'''
        source = CLI.replace("    elif args[-1].endswith('snapshot.py'):",
                             "    elif args[-1].endswith('snapshot.py'):\n" + rewrite)
        source = source.replace("[phase0] synthetic |", "[phase0] {('foreign' if (root / 'replaced').exists() "
                                "else 'synthetic')} |")
        source = source.replace("print('[phase0]", "print(f'[phase0]")
        source = source.replace("        if os.environ.get('REWRITE') == '1':",
                                "        if os.environ.get('REWRITE') == 'foreign': (root / 'replaced').touch()\n"
                                "        if os.environ.get('REWRITE') == '1':")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env, wrapper, fake = self.wrapper_fixture(root)
            fake.write_text(f"#!{sys.executable}\n" + source)
            first = subprocess.run(["bash", str(wrapper), "up", "aa-1", "A100"], env=env,
                                   capture_output=True, text=True, timeout=12)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            (root / "boot").unlink()
            original = (root / "receipt").read_bytes()
            for replacement in ("1", "foreign"):
                with self.subTest(replacement=replacement):
                    (root / "receipt").write_bytes(original)
                    before = (root / "calls").read_text().splitlines()
                    resumed = subprocess.run(["bash", str(wrapper), "resume-up", "aa-1", "A100"],
                                             env={**env, "REWRITE": replacement}, capture_output=True, text=True, timeout=8)
                    self.assertEqual(resumed.returncode, 73, resumed.stdout + resumed.stderr)
                    self.assertFalse((root / "boot").exists())
                    calls = (root / "calls").read_text().splitlines()[len(before):]
                    self.assertEqual(calls.count("exec"), 1)  # Snapshot only, no transfer or bootstrap.
                    self.assertNotIn("upload", calls)
                    self.assertNotIn("stop", calls)

    def test_foreign_receipt_blocks_pull_before_any_cli_or_local_metadata_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env, wrapper, _ = self.wrapper_fixture(root)
            self.owned(root)
            results = root / "phase0/results"
            results.mkdir()
            metadata = results / "colab_bootstrap_status.json"
            metadata.write_bytes(b'"original local metadata"')
            saved = json.loads((root / "receipt").read_text())
            for status, campaign in (("foreign-campaign", "synthetic-campaign"), ("owned", "foreign")):
                with self.subTest(status=status, campaign=campaign):
                    (root / "receipt").write_text(json.dumps({**saved, "status": status, "plan": "aa-1",
                                                             "campaign_id": campaign}))
                    result = subprocess.run(["bash", str(wrapper), "pull"], env=env,
                                            capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
                    self.assertFalse((root / "calls").exists())
                    self.assertEqual(metadata.read_bytes(), b'"original local metadata"')

    def test_code_plan_receipt_preserves_checkpoint_upload_and_pull_without_e12_metadata(self):
        import tarfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env, wrapper, fake = self.wrapper_fixture(root)
            results = root / "phase0/results"
            results.mkdir()
            names = ("code_synthetic_colab_mbpp_all.jsonl", "codeexec_synthetic_colab_mbpp_all.jsonl")
            for name in (*names, "aa_synthetic_colab_scicode_dev.jsonl"):
                (results / name).write_text('{"id": "synthetic"}\n')
                (results / (name + ".meta.json")).write_text('{"synthetic": true}')
            first = subprocess.run(["bash", str(wrapper), "up", "code-1", "A100"], env=env,
                                   capture_output=True, text=True, timeout=12)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            with tarfile.open(root / "receipt.archive.tgz") as archive:
                checkpoints = {Path(n).name for n in archive.getnames() if n.startswith("checkpoints/")}
            self.assertTrue(all(n in checkpoints and n + ".meta.json" in checkpoints for n in names))
            self.assertNotIn("aa_synthetic_colab_scicode_dev.jsonl", checkpoints)
            source = CLI.replace("elif args[0] == 'download': sys.exit(31)", r'''
elif args[0] == 'download':
    name = Path(args[3]).name
    assert name.startswith(('code_', 'codeexec_')), 'unexpected E12 provenance request'
    Path(args[4]).write_text((root / 'phase0/results' / name).read_text())
''').replace("        print('LISTE-OK')", "        print('LISTE-OK')\n"
            + "        print('code_synthetic_colab_mbpp_all.jsonl')\n"
            + "        print('codeexec_synthetic_colab_mbpp_all.jsonl')\n"
            + "        print('aa_synthetic_colab_scicode_dev.jsonl')")
            fake.write_text(f"#!{sys.executable}\n" + source)
            pulled = subprocess.run(["bash", str(wrapper), "pull", "final"], env=env,
                                    capture_output=True, text=True, timeout=8)
            self.assertEqual(pulled.returncode, 0, pulled.stdout + pulled.stderr)
            self.assertTrue(all((results / n).exists() for n in names))
            self.assertFalse((results / "e12_campaign_sources.json").exists())

    def test_foreign_downloaded_campaign_metadata_never_replaces_local_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env, wrapper, fake = self.wrapper_fixture(root)
            self.owned(root)
            saved = json.loads((root / "receipt").read_text())
            (root / "receipt").write_text(json.dumps({**saved, "plan": "aa-1", "campaign_id": "synthetic-campaign"}))
            results = root / "phase0/results"
            results.mkdir()
            metadata = results / "colab_bootstrap_status.json"
            metadata.write_bytes(b'"original local metadata"')
            source = CLI.replace("elif args[0] == 'download': sys.exit(31)", r'''
elif args[0] == 'download':
    Path(args[4]).write_text(json.dumps({'campaign_id': 'foreign', 'schema': 1}))
''')
            fake.write_text(f"#!{sys.executable}\n" + source)
            pulled = subprocess.run(["bash", str(wrapper), "pull"], env=env,
                                    capture_output=True, text=True, timeout=8)
            self.assertNotEqual(pulled.returncode, 0, pulled.stdout + pulled.stderr)
            self.assertIn("foreign recovered campaign", pulled.stderr)
            self.assertEqual(metadata.read_bytes(), b'"original local metadata"')
            self.assertFalse((results / "e12_campaign_sources.json").exists())

    def test_cleanup_retries_and_confirms_stop_with_lost_reads(self):
        for mode in ("cleanup-dns", "stop-dns", "stop-read-lost", "stop-ack-lost", "after"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                env = self.fixture(root, LIFECYCLE_CLI, mode)
                self.owned(root)
                result = self.call(env, "down")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((root / "active").exists())
                self.assertFalse((root / "receipt").exists())
                self.assertIn("explicit absence confirmed", result.stdout)
                evidence = json.loads((root / "receipt.usage.jsonl").read_text())
                self.assertTrue(evidence["release_confirmed"])
                self.assertIsNone(evidence["campaign_consumption_units"])
                self.assertEqual(evidence["balance_decrease_units"], 2.5 if mode == "after" else 0)
                if mode == "stop-dns": self.assertEqual((root / "count-stop").read_text(), "4")

    def test_confirmed_new_retries_identity_without_reallocating(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, LIFECYCLE_CLI, "receipt-transient")
            allocated = self.call(env, "allocate", "aa-1", "A100")
            self.assertEqual(allocated.returncode, 0, allocated.stdout + allocated.stderr)
            self.assertEqual(json.loads((root / "receipt").read_text())["status"], "owned")
            cleaned = self.call(env, "down")
            self.assertEqual(cleaned.returncode, 0, cleaned.stdout + cleaned.stderr)
            self.assertEqual((root / "calls").read_text().splitlines().count("new"), 1)

    def test_release_with_unreadable_usage_resumes_accounting_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, LIFECYCLE_CLI, "after-empty")
            self.owned(root)
            result = self.call(env, "down")
            self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
            self.assertFalse((root / "active").exists())
            saved = json.loads((root / "receipt").read_text())
            self.assertEqual(saved["status"], "released/accounting-pending")
            self.assertTrue(saved["release_confirmed"])
            self.assertNotIn("VM may still be billed", result.stderr)
            records = [json.loads(line) for line in (root / "receipt.usage.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["after"] is None and r["balance_decrease_units"] is None for r in records))
            blocked = self.call(env, "allocate", "aa-1", "A100")
            self.assertEqual(blocked.returncode, 65)
            before = (root / "calls").read_text().splitlines()
            result = subprocess.run([sys.executable, str(COLAB / "campaign.py"), "--cleanup-only",
                                     "--receipt", str(root / "receipt"), "--lock", str(root / "supervisor.lock"),
                                     "--cleanup-seconds", "2"], env={**env, "MODE": "after"},
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "calls").read_text().splitlines()[len(before):], ["usage"])
            self.assertFalse((root / "receipt").exists())
            final = json.loads((root / "receipt.usage.jsonl").read_text().splitlines()[-1])
            self.assertEqual(final["balance_decrease_units"], 2.5)

    def test_ambiguous_allocation_and_replaced_session_never_authorize_foreign_stop(self):
        for mode in ("new-lost", "receipt-lost", "replaced"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                env = self.fixture(root, LIFECYCLE_CLI, mode)
                if mode == "replaced":
                    self.owned(root)
                else:
                    result = self.call(env, "allocate", "aa-1", "A100")
                    self.assertEqual(result.returncode, 65, result.stdout + result.stderr)
                    saved = json.loads((root / "receipt").read_text())
                    self.assertEqual(saved["status"], "allocation-unconfirmed")
                    self.assertEqual(saved.get("allocation_returned", False), mode == "receipt-lost")
                    if mode == "receipt-lost": self.assertEqual((root / "count-sessions").read_text(), "4")
                blocked = self.call(env, "down")
                self.assertNotEqual(blocked.returncode, 0)
                self.assertTrue((root / "receipt").exists())
                self.assertNotIn("stop", (root / "calls").read_text().splitlines())
                again = self.call(env, "allocate", "aa-1", "A100")
                self.assertEqual(again.returncode, 65)
                self.assertLessEqual((root / "calls").read_text().splitlines().count("new"), 1)

    def test_durable_dns_retains_ownership_and_cleanup_resumes_without_allocation(self):
        for mode in ("durable-dns", "empty-sessions"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                self.pending_cleanup(Path(d), mode)

    def pending_cleanup(self, root, mode):
        env = self.fixture(root, LIFECYCLE_CLI, mode)
        self.owned(root)
        first = self.call({**env, "DLLM_CLEANUP_SECONDS": ".3"}, "down")
        self.assertEqual(first.returncode, 124, first.stdout + first.stderr)
        self.assertTrue((root / "receipt").exists())
        self.assertNotIn("explicit absence confirmed", first.stdout)
        self.assertIn("VM may still be billed", first.stderr)
        second = self.call({**env, "MODE": "success"}, "down")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertNotIn("new", (root / "calls").read_text().splitlines())

    def test_replaced_session_refuses_transfer_before_remote_code_or_upload(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, LIFECYCLE_CLI, "replaced")
            self.owned(root)
            archive = root / "archive"
            archive.write_bytes(b"synthetic archive")
            result = subprocess.run([sys.executable, str(COLAB / "transfer.py"), str(archive)],
                                    env={**env, "DLLM_REQUIRE_OWNER": "1"}, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertNotIn("exec", calls)
            self.assertNotIn("upload", calls)
            self.assertNotIn("stop", calls)
            self.assertTrue((root / "receipt").exists())

    def test_owned_campaign_refuses_resume_but_retains_cleanup_authority(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, LIFECYCLE_CLI, "success")
            self.owned(root)
            saved = json.loads((root / "receipt").read_text())
            saved["campaign_id"] = "original-campaign"
            (root / "receipt").write_text(json.dumps(saved))
            snapshot = {"schema": 1, "campaign_present": True, "bootstrap": {"campaign_id": "original-campaign"}}
            guarded = subprocess.run([sys.executable, str(COLAB / "session_json.py"), "guard"], env=env,
                                     input=json.dumps(snapshot), capture_output=True, text=True, timeout=5)
            self.assertEqual(guarded.returncode, 73, guarded.stdout + guarded.stderr)
            self.assertEqual(json.loads((root / "receipt").read_text())["status"], "owned")
            cleaned = self.call(env, "down")
            self.assertEqual(cleaned.returncode, 0, cleaned.stdout + cleaned.stderr)

    def test_balance_capacity_and_empty_responses_fail_closed(self):
        for mode, gpu in (("empty-usage", "A100"), ("low-balance", "A100"), ("success", "L4"),
                          ("low-memory", "A100")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                env = self.fixture(root, LIFECYCLE_CLI, mode)
                result = self.call(env, "allocate", "aa-1", gpu)
                self.assertEqual(result.returncode, 65, result.stdout + result.stderr)
                calls = (root / "calls").read_text().splitlines() if (root / "calls").exists() else []
                self.assertEqual(calls.count("new"), int(mode == "low-memory"))
                if mode == "low-memory": self.assertTrue((root / "receipt").exists())

    def test_every_cli_operation_has_a_local_deadline_and_kills_resistant_child(self):
        source = r'''import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0: time.sleep(30); sys.exit(0)
Path(os.environ['TEST_ROOT'], 'pids').write_text(json.dumps([os.getpid(), child]))
time.sleep(30)
'''
        for operation in ("sessions", "usage", "new", "upload", "download", "stop", "exec"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                env = self.fixture(root, source, "block")
                env.update(DLLM_CLI_SECONDS=".3", DLLM_CLI_GRACE=".1")
                result = subprocess.run([sys.executable, str(COLAB / "safe_cli.py"), operation], env=env,
                                        capture_output=True, text=True, timeout=4)
                self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
                for pid in json.loads((root / "pids").read_text()):
                    path = Path(f"/proc/{pid}/stat")
                    if path.exists():
                        self.assertIn(path.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X", "x"))

    def test_signal_to_transfer_parent_alone_kills_cli_and_child(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.fixture(root, TRANSFER_CLI, "signal")
            archive = root / "archive"
            archive.write_bytes(b"synthetic archive")
            p = subprocess.Popen([sys.executable, str(COLAB / "transfer.py"), str(archive)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            pids = []
            try:
                deadline = time.monotonic() + 5
                while not (root / "pids").exists() and p.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / "pids").exists())
                pids = json.loads((root / "pids").read_text())
                p.send_signal(signal.SIGTERM)  # Only this PID; do not rely on an outer group signal.
                output, _ = p.communicate(timeout=5)
                self.assertEqual(p.returncode, 143, output)
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


class UsageParser(unittest.TestCase):
    def test_strict_usage_parser_and_capacity_profiles(self):
        adapter = load("session_json")
        self.assertEqual(adapter.usage("Current balance: 65.00 compute units\nUsage rate: 0.00/hr\n"
                                       "Active assignments: 0")["balance_units"], 65)
        for raw in ("", "0", "Current balance: nan compute units", "connection error",
                    "Current balance: " + "9" * 400 + ".00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"):
            with self.assertRaises(ValueError): adapter.usage(raw)
        self.assertEqual(adapter.capacity("aa-1", "A100"), 39)
        with self.assertRaises(ValueError): adapter.capacity("aa-1", "L4")
        self.assertEqual(adapter.capacity("smoke", "L4"), 6)
        self.assertEqual(adapter.select_gpu("smoke"), "L4")
        self.assertEqual(adapter.select_gpu("aa-1"), "A100")
