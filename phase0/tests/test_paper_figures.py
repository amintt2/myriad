"""paper/make_figures.py: independent figures, no crash on an empty E4 summary, no stale figure left behind.
Needs matplotlib (uv run --group paper); skipped otherwise."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


@unittest.skipUnless(HAVE_MPL, "matplotlib absent (uv run --group paper)")
class TestFigures(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("make_figures", ROOT / "paper" / "make_figures.py")
        self.mf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mf)
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "res").mkdir()
        (d / "paper").mkdir()
        self.mf.RES, self.mf.HERE = d / "res", d / "paper"
        (d / "paper" / "main.tex").write_text("\\includegraphics[width=1cm]{fig_protocol.pdf}\n", encoding="utf-8")
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_e4_summary_does_not_block_the_other_figures(self):
        # Audit 2026-10-09 (paper, 6): "benches": {} made plt.subplots(1, 0) raise before fig_protocol.
        (self.d / "res" / "e4_summary_colab.json").write_text(json.dumps({"benches": {}, "families": {}}))
        g = {"cost": {m: {"trips_mean": t} for m, t in (("vote", 1), ("croise", 40), ("accord", 20))},
             "accuracy": {"vote": 89, "croise": 89.5, "accord": 87, "m": 86.5, "référence : Qwen/Qwen3-4B": 91},
             "best_peer": "m"}
        (self.d / "res" / "gen_summary_essaim4_colab_test.json").write_text(json.dumps(g), encoding="utf-8")
        (self.d / "paper" / "fig_scaling.pdf").write_bytes(b"old")
        self.assertEqual(self.mf.main(), 0)
        self.assertTrue((self.d / "paper" / "fig_protocol.pdf").exists())
        self.assertFalse((self.d / "paper" / "fig_scaling.pdf").exists())  # the stale figure is removed

    def test_missing_source_of_an_included_figure_fails(self):
        # Audit 2026-10-09 (paper, 7): the old fig_protocol.pdf stayed and was compiled into the paper.
        (self.d / "paper" / "fig_protocol.pdf").write_bytes(b"old")
        self.assertEqual(self.mf.main(), 1)
        self.assertFalse((self.d / "paper" / "fig_protocol.pdf").exists())


if __name__ == "__main__":
    unittest.main()
