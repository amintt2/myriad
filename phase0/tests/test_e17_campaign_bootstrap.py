"""Hermetic UV archives and unavailable system packaging; no network, dataset or model."""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import test_e17_campaign as fixtures
import e17_campaign_remote as remote
import read_timeout


def archive(members=None, data=b"#!/bin/sh\nprintf 'uv 0.11.21 (synthetic)\\n'\n"):
    out = io.BytesIO()
    members = members if members is not None else [(remote.UV_MEMBER, tarfile.REGTYPE)]
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../../outside"
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            tar.addfile(member, io.BytesIO(data) if member.size else None)
    return out.getvalue()


class UVBootstrap(unittest.TestCase):
    def prepare(self, root, data, deadline=None, pins=None, version_output=b"uv 0.11.21 (synthetic)\n", version_code=0):
        (root / "artifacts").mkdir(exist_ok=True)
        pins = pins or {**remote.UV, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        def version(command, **kwargs):
            self.assertEqual(command, [str(root / "installer/bin/uv"), "--version"])
            self.assertNotIn("start_new_session", kwargs)
            self.assertLessEqual(kwargs["timeout"], 30)
            kwargs["stdout"].write(version_output)
            return SimpleNamespace(returncode=version_code)
        with mock.patch.object(remote, "UV", pins), \
             mock.patch.object(remote.urllib.request, "urlopen", return_value=io.BytesIO(data)) as network, \
             mock.patch.object(remote.subprocess, "run", side_effect=version):
            remote.bootstrap_uv(root, deadline or time.monotonic() + 60)
        network.assert_called_once()
        return json.loads((root / "artifacts/runtime.json").read_text())["uv_bootstrap"]

    def test_valid_archive_without_system_packaging_and_preserved_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            request = {"plan": fixtures.campaign.plan("e17-dev-single"), "work_seconds": 3600}
            remote.save(root / "request.json", request)
            commands, deadlines = [], []
            def run(command, **kwargs):
                self.assertNotIn("venv", command)
                self.assertNotIn("pip", command)
                self.assertNotIn("ensurepip", command)
                commands.append(command)
                deadlines.append(kwargs["deadline"])
                if "--bootstrap-uv" in command:
                    self.prepare(root, archive(), float(command[-1]))
                    return 0
                if "sync" in command:
                    self.assertEqual(command[1:6], ["sync", "--frozen", "--python", "3.12.15", "--project"])
                    self.assertEqual(kwargs["seconds"], 600)
                    self.assertTrue((root / "installer/bin/uv").is_file())
                    return 0
                # The controller is deliberately incomplete; no synthetic safety attestation is invented.
                self.assertEqual(command[-2], str(root))
                self.assertEqual(float(command[-1]), deadlines[0])
                return 17
            with mock.patch.object(read_timeout, "run", side_effect=run), mock.patch.object(remote.signal, "signal"):
                remote.worker(root)
            self.assertEqual(len(commands), 3)
            self.assertEqual(deadlines, [deadlines[0]] * 3)
            state = json.loads((root / "artifacts/vm-status.json").read_text())
            self.assertEqual((state["phase"], state["returncode"], state["state"]), ("controller", 17, "incomplete"))
            self.assertEqual(json.loads((root / "request.json").read_text()), request)

    def test_pinned_metadata_matches_parent(self):
        self.assertEqual(remote.UV["version"], "0.11.21")
        self.assertEqual(remote.UV["bytes"], 24437894)
        self.assertEqual(remote.UV["sha256"], "8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")

    def test_hash_and_short_or_oversized_download_refused(self):
        data = archive()
        for change in ({"sha256": "0" * 64}, {"bytes": len(data) - 1}, {"bytes": len(data) + 1}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pins = {**remote.UV, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **change}
                with self.assertRaises(ValueError):
                    self.prepare(root, data, pins=pins)
                self.assertFalse((root / "installer").exists())
                self.assertFalse((root / "uv.tar.gz.part").exists())

    def test_unsafe_paths_links_duplicates_and_missing_binary_refused(self):
        bad = [("../outside", tarfile.REGTYPE), ("/outside", tarfile.REGTYPE),
               ("uv-x86_64-unknown-linux-gnu/../outside", tarfile.REGTYPE),
               ("uv-x86_64-unknown-linux-gnu\\outside", tarfile.REGTYPE),
               ("uv-x86_64-unknown-linux-gnu/link", tarfile.SYMTYPE),
               ("uv-x86_64-unknown-linux-gnu/link", tarfile.LNKTYPE)]
        cases = [[(remote.UV_MEMBER, tarfile.REGTYPE), member] for member in bad]
        cases += [[(remote.UV_MEMBER, tarfile.REGTYPE)] * 2, [(remote.UV_MEMBER, tarfile.DIRTYPE)], []]
        for members in cases:
            with self.subTest(members=members), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(ValueError):
                    self.prepare(root, archive(members))
                self.assertFalse((root / "installer").exists())

    def test_existing_target_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "installer/bin").mkdir(parents=True)
            target = root / "installer/bin/uv"
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                self.prepare(root, archive())
            self.assertEqual(target.read_bytes(), b"existing")

    def test_wrong_version_nonzero_and_unbounded_output_refused(self):
        for output, code in ((b"uv 0.11.20\n", 0), (b"uv 0.11.21\n", 1), (b"x" * 257, 0)):
            with self.subTest(code=code, output_bytes=len(output)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.prepare(root, archive(), version_output=output, version_code=code)
                info = json.loads((root / "artifacts/runtime.json").read_text())["uv_bootstrap"]
                self.assertEqual(info["phase"], "uv-version")
                self.assertFalse(info["complete"])
                self.assertLessEqual(len(info["diagnostic"]["trace"]), 4096)

    def test_preparation_timeout_leaves_partial_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            remote.save(root / "request.json", {"plan": fixtures.campaign.plan("e17-dev-single"), "work_seconds": 600})
            with mock.patch.object(remote.signal, "signal"), \
                 mock.patch.object(read_timeout, "run", return_value=124) as run:
                remote.worker(root)
            run.assert_called_once()
            inventory = json.loads((root / "inventory.json").read_text())
            self.assertFalse(inventory["complete"])
            self.assertEqual(set(inventory["files"]), {"vm-status.json", "runtime.json", "controller.log"})
            for name, info in inventory["files"].items():
                data = (root / "artifacts" / name).read_bytes()
                self.assertEqual(info, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            state = json.loads((root / "artifacts/vm-status.json").read_text())
            self.assertEqual((state["phase"], state["returncode"]), ("uv-bootstrap", 124))
            self.assertIn("returned 124", state["diagnostic"]["trace"])

    def test_download_deadline_records_phase_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(remote.time, "monotonic", side_effect=[1, 2, 3, 20]), \
                 self.assertRaises(TimeoutError):
                self.prepare(root, archive(), deadline=10)
            info = json.loads((root / "artifacts/runtime.json").read_text())["uv_bootstrap"]
            self.assertEqual(info["phase"], "uv-download")
            self.assertIn("UV download deadline", info["diagnostic"]["trace"])

    @unittest.skipIf(os.name == "nt", "POSIX executable and process-group watchdog")
    def test_real_synthetic_uv_executable_version(self):
        data = archive(data=b"#!/bin/sh\nprintf 'uv 0.11.21 (synthetic)\\n'\n"
                            b'ps -o pgid= -p $$ > "$(dirname "$0")/pgid"\n')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            pins = {**remote.UV, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            with mock.patch.object(remote, "UV", pins), \
                 mock.patch.object(remote.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                remote.bootstrap_uv(root, time.monotonic() + 60)
            info = json.loads((root / "artifacts/runtime.json").read_text())["uv_bootstrap"]
            self.assertTrue(info["complete"])
            self.assertEqual(info["version_observed"], "uv 0.11.21 (synthetic)")
            self.assertEqual(int((root / "installer/bin/pgid").read_text()), os.getpgrp())


if __name__ == "__main__":
    unittest.main()
