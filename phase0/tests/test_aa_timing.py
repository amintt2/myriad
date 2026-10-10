"""Synthetic timing recovery, including interrupted atomic publication."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import aa_timing as t
import run_aa


class Timing(unittest.TestCase):
    def test_exact_final_object_and_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            out = SimpleNamespace(path=Path(d) / "out")
            path = Path(str(out.path) + ".timing.jsonl")
            path.write_bytes(b'{"kind":"call","wall_s":1}')
            run_aa.timing(out, threading.Lock(), {"kind": "call", "wall_s": 1})
            self.assertEqual(len(t.parse(path.read_bytes())), 2)
            # Appends do not rescan an ever-growing trace.
            with mock.patch.object(t, "parse", side_effect=AssertionError("rescan")):
                run_aa.timing(out, threading.Lock(), {"kind": "call", "wall_s": 2})

    def test_corruption_refuses_before_inference_and_preserves_evidence(self):
        for raw in (b'{"kind":"call","wall_s":1}\n{"kind":',
                    b'{"wall_s":1}\nbroken\n{"wall_s":2}\n'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as d:
                out = SimpleNamespace(path=Path(d) / "out")
                path = Path(str(out.path) + ".timing.jsonl")
                path.write_bytes(raw)
                chat = mock.Mock()
                with self.assertRaisesRegex(ValueError, "unknown; resume refused"):
                    run_aa.observed_chat(chat, out, threading.Lock())
                chat.assert_not_called()
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(next(path.parent.glob("*.rejected-*")).read_bytes(), raw)

    def test_atomic_publication_never_regresses_or_accepts_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, dest, meta = (root / n for n in ("source", "out.timing.jsonl", "meta"))
            meta.write_text('{"measurements":"e12-wall-v1"}')
            old = b'{"wall_s":1}\n{"wall_s":2}\n'
            dest.write_bytes(old)
            for raw in (b'{"wall_s":1}\n', old + b'{"kind":', b'{"wall_s":9}\n'):
                source.write_bytes(raw)
                with self.assertRaises(ValueError): t.publish(source, dest, meta)
                self.assertEqual(dest.read_bytes(), old)
            source.write_bytes(old + b'{"wall_s":2}')
            with mock.patch.object(t.os, "replace", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): t.publish(source, dest, meta)
            self.assertEqual(dest.read_bytes(), old)
            t.publish(source, dest, meta)
            self.assertEqual(len(t.parse(dest.read_bytes())), 3)

    def test_protocol_binding_and_initialization_before_call(self):
        with tempfile.TemporaryDirectory() as d:
            out = SimpleNamespace(path=Path(d) / "aa_synthetic.jsonl")
            meta = {"measurements": "e12-wall-v2", "prompt": "synthetic"}
            Path(str(out.path) + ".meta.json").write_text(json.dumps(meta))
            t.initialize(out)
            self.assertEqual(t.parse(out._timing_path.read_bytes(), t.header(out.path, meta)),
                             [t.header(out.path, meta)])
            with self.assertRaises(ValueError):
                t.parse(out._timing_path.read_bytes(), t.header(out.path, {**meta, "prompt": "changed"}))

    def test_snapshot_filters_messages_without_changing_originals(self):
        from tests.test_campaign import load
        original = {"schema": 1, "bootstrap": {"step": "ÉCHEC https://signed?secret", "log": "secret",
                                               "erreur": "secret", "plan": "aa-1"},
                    "jobs": {"job": "ÉCHEC RuntimeError: https://signed?secret"},
                    "diagnostics": {"launcher_alive": False, "message": "secret"}}
        safe = load("snapshot").safe_snapshot(original)
        self.assertNotIn("secret", json.dumps(safe))
        self.assertIn("secret", json.dumps(original))

    def test_missing_observation_refuses_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = SimpleNamespace(path=Path(d) / "aa_synthetic.jsonl")
            meta = {"measurements": "e12-wall-v2"}
            Path(str(out.path) + ".meta.json").write_text(json.dumps(meta))
            t.initialize(out)
            out.path.write_text('{"id":"lost"}\n')
            del out._timing_path
            with self.assertRaisesRegex(ValueError, "unknown; resume refused"):
                t.initialize(out)
