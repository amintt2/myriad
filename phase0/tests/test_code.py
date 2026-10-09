"""E11 unit tests (no model, no GPU):  uv run python -m unittest tests.test_code

The sandbox tests start real child processes (a few seconds in all)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_code as ac  # noqa: E402
import colab_jobs  # noqa: E402
from essaim import code, sandbox  # noqa: E402


class Extraction(unittest.TestCase):
    def test_fenced_block_with_entry(self):
        text = ("Here is the solution:\n```python\nimport math\n\ndef area(r):\n    return math.pi * r * r\n\n"
                "print(area(2))\nassert area(1) > 3\nif __name__ == '__main__':\n    print(area(3))\n```\n"
                "Usage:\n```python\nprint(area(5))\n```")
        src, status = code.extract_code(text, "area")
        self.assertEqual(status, "fenced")
        self.assertIn("import math", src)
        self.assertIn("def area(r):", src)
        for junk in ("print(", "assert", "__main__"):
            self.assertNotIn(junk, src)

    def test_last_block_defining_entry(self):
        text = "```python\ndef f(x):\n    return x\n```\nBetter:\n```py\ndef f(x):\n    return x + 1\n```"
        self.assertIn("x + 1", code.extract_code(text, "f")[0])

    def test_unclosed_unfenced_and_missing(self):
        src, st = code.extract_code("```python\ndef f(x):\n    return x * 2\n", "f")
        self.assertEqual(st, "unclosed")
        self.assertIn("x * 2", src)
        src, st = code.extract_code("Sure.\ndef f(x):\n    return 3\nThis returns three.", "f")
        self.assertEqual((st, src.strip()), ("unfenced", "def f(x):\n    return 3"))
        self.assertEqual(code.extract_code("```python\ndef g(x):\n    return 1\n```", "f")[1], "no_entry")
        self.assertEqual(code.extract_code("I cannot do that.", "f"), ("", "empty"))

    def test_sanitize_keeps_helpers_and_constants(self):
        src = ("from typing import List\nMOD = 10 ** 9 + 7\n@staticmethod\ndef helper(a):\n    return a\n"
               "def f(xs: List[int]) -> int:\n    return helper(sum(xs)) % MOD\nresult = f([1, 2])\nprint(result)\n")
        out = code.sanitize(src, "f")
        self.assertIn("MOD = 10 ** 9 + 7", out)
        self.assertIn("@staticmethod\ndef helper", out)
        self.assertNotIn("result =", out)
        self.assertNotIn("print", out)

    def test_sanitize_keeps_needed_initialisation(self):
        # Audit 2026-10-09 (phase0, 11): `solver = Solver()` was removed as "example usage" -> NameError.
        src = ("class Solver:\n    def solve(self, x):\n        return 2 * x\n\nsolver = Solver()\n"
               "def build():\n    return [1]\n\nTABLE = build()\ndef f(x):\n    return solver.solve(x) + TABLE[0]\n\n"
               "demo = f(3)\nprint(demo)\n")
        out = code.sanitize(src, "f")
        self.assertIn("solver = Solver()", out)
        self.assertIn("TABLE = build()", out)
        self.assertNotIn("demo", out)
        ns = {}
        exec(out, ns)
        self.assertEqual(ns["f"](3), 7)
        # a LOCAL variable of the function that shares the example's name does not keep the example (Codex)
        out = code.sanitize("def f(x):\n    result = 10 / x\n    return result\n\nresult = f(0)\n", "f")
        self.assertNotIn("f(0)", out)
        # a class body reads the global before assigning the same name (Codex, 2nd pass)
        out = code.sanitize("def make():\n    return 3\nvalue = make()\nclass C:\n    value = value\n"
                            "def f():\n    return C.value\nr = f()\n", "f")
        ns = {}
        exec(out, ns)
        self.assertEqual(ns["f"](), 3)
        self.assertNotIn("r = f()", out)
        # Class bodies are read conservatively (Codex, 3rd and 4th passes: names already bound by the class,
        # conditional bindings, comprehensions, method annotations): a kept call is guarded, so an example kept
        # by a name collision never breaks the load, and a needed initialisation is never dropped.
        for src in ("class C:\n    value = 10\n    doubled = value * 2\ndef f(x):\n    return C.doubled / x\n"
                    "value = f(0)\n",
                    "def make():\n    return 3\nvalue = make()\nclass C:\n    if False:\n        value = 10\n"
                    "    doubled = value * 2\ndef f(x):\n    return C.doubled / x\n",
                    "class C:\n    values = [value * 2 for value in range(3)]\ndef f(x):\n    return C.values[1] / x\n"
                    "value = f(0)\n",
                    "def make():\n    return int\nT = make()\nclass C:\n    def m(self, x: T):\n        return x\n"
                    "    T = str\ndef f(x):\n    return C().m(x)\n"):
            with self.subTest(src=src):
                out = code.sanitize(src, "f")
                ns = {}
                exec(out, ns)
                self.assertIsNotNone(ns["f"](1))
                self.assertNotIn("f(0)", out)  # names the class binds itself are not global reads
        # Codex, 5th pass: an example with a side effect is not kept for a name the class binds itself, and a
        # guarded initialisation keeps its multi-line string literal unchanged.
        for extra in ("", "    def method(self, value):\n        del value\n"):  # 7th pass: a method's own del
            out = code.sanitize("class C:\n    value = []\n" + extra + "    alias = value\ndef f(x):\n"
                                "    C.value.append(x)\n    return len(C.value)\nvalue = f(0)\n", "f")
            ns = {}
            exec(out, ns)
            self.assertEqual(ns["f"](1), 1)
        for src in ("def make():\n    return 3\nvalue = make()\nclass C:\n    value = 1\n    if True:\n        del value\n"
                    "    result = value\ndef f(x):\n    return C.result\n",  # Codex, 6th pass: nested unbinding
                    "def make():\n    return 3\nvalue = make()\nclass C:\n    value = 1\n    try:\n        1 / 0\n"
                    "    except ZeroDivisionError as value:\n        pass\n    result = value\ndef f(x):\n    return C.result\n"):
            ns = {}
            exec(code.sanitize(src, "f"), ns)
            self.assertEqual(ns["f"](1), 3)
        out = code.sanitize('def make(s):\n    return s\nTABLE = make("""a\nb""")\ndef f(x):\n    return TABLE\n', "f")
        ns = {}
        exec(out, ns)
        self.assertEqual(ns["f"](1), "a\nb")

    def test_normalized_text_ignores_comments_and_docstrings(self):
        a = 'def f(x):\n    """Doc."""\n    # comment\n    return x+1\n'
        b = "def f(x):\n    return x + 1"
        self.assertEqual(code.normalized_text(a), code.normalized_text(b))
        self.assertNotEqual(code.normalized_text(b), code.normalized_text("def f(x):\n    return 1 + x"))


def he_item(doc: str, sig: str = "def f(x):") -> dict:
    return {"id": "T/0", "entry_point": "f", "prompt": f'{sig}\n    """\n{doc}\n    """\n'}


class VisibleTests(unittest.TestCase):
    def test_formats(self):
        doc = ("    >>> f(1)\n    2\n    >>> f(3) == 4\n    f(5) ➞ 6\n    f(7) => true\n    f('a') -> 'b' # note\n"
               "    For x = 9 the output should be 10.\n    >>> round(f(1), 2)\n    2")
        ex = code.humaneval_examples(he_item(doc))
        self.assertEqual(ex, [((1,), 2), ((3,), 4), ((5,), 6), ((7,), True), (("a",), "b"), ((9,), 10)])

    def test_input_output_block_and_named_args(self):
        doc = "    Example:\n        Input: arr = [1, 2], k = 1\n        Output: [2]\n"
        ex = code.humaneval_examples(he_item(doc, "def f(arr, k):"))
        self.assertEqual(ex, [(([1, 2], 1), [2])])
        tests = code.visible_tests("humanevalplus", he_item(doc, "def f(arr, k):"))
        self.assertEqual(tests, ["assert _close(f(*([1, 2], 1)), [2])"])

    def test_mbpp_inputs_and_entry(self):
        item = {"id": "Mbpp/1", "test_list": ["assert set(g((1, 2), [3])) == set((1,))",
                                              "assert math.isclose(g((4,), []), 1.0, rel_tol=0.1)", "assert g(x, 1) == 2"],
                "test": "inputs = [[1]]\nresults = [1]\nfor i, (inp, exp) in enumerate(zip(inputs, results)):\n"
                        "    assertion(g(*inp), exp, 0)\n"}
        self.assertEqual(code.entry_point("mbppplus", item), "g")
        self.assertEqual(code.visible_inputs("mbppplus", item), [((1, 2), [3]), ((4,), [])])

    def test_extra_inputs_deterministic(self):
        item = {"id": "Mbpp/2", "test_list": ["assert g([3, 1, 2], 'ab') == 1"],
                "test": "for i, inp in enumerate(inputs):\n    assertion(g(*inp), ref_func(*inp), 0)\n"}
        a = code.extra_inputs("mbppplus", item, 12)
        self.assertEqual(a, code.extra_inputs("mbppplus", item, 12))
        self.assertEqual(a[0], ([3, 1, 2], "ab"))
        self.assertEqual(len(a), 13)
        self.assertEqual(len({repr(x) for x in a}), 13)
        for x in a:
            self.assertEqual(len(x), 2)
            self.assertIsInstance(x[0], list)
            self.assertIsInstance(x[1], str)

    def test_typed_inputs_without_examples(self):
        item = he_item("    no examples here", "def f(s: str, n: int):")
        a = code.extra_inputs("humanevalplus", item, 5)
        self.assertEqual(len(a), 5)
        self.assertTrue(all(isinstance(x[0], str) and isinstance(x[1], int) for x in a))

    def test_hidden_spec(self):
        test = ("import numpy as np\ndef assertion(out, exp, atol):\n    assert out == exp\n\n"
                "def check(candidate):\n    inputs = [[1], [2]]\n    results = [2, 3]\n"
                "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n        assertion(candidate(*inp), exp, 0)\n")
        spec = code.hidden_spec("humanevalplus", {"id": "T/1", "test": test})
        self.assertEqual(spec["setup"], ["inputs = [[1], [2]]", "results = [2, 3]"])
        self.assertEqual((spec["target"], spec["iter"]), ("i, (inp, exp)", "enumerate(zip(inputs, results))"))
        self.assertIn("def assertion", spec["prelude"])
        self.assertNotIn("check", spec["prelude"])

    def test_non_asserting_helper_repaired(self):
        test = ("def assertion(out, exp, atol):\n    if isinstance(out, bool):\n        exact_match = out == exp\n"
                "    else:\n        exact_match = exp == (out is not None)\n\ninputs = [[1]]\nresults = [True]\n"
                "for i, (inp, exp) in enumerate(zip(inputs, results)):\n    assertion(g(*inp), exp, 0)\n")
        spec = code.hidden_spec("mbppplus", {"id": "Mbpp/9", "test": test})
        self.assertTrue(spec["prelude"].rstrip().endswith("assert exact_match"))
        r = sandbox.run_hidden("def g(x):\n    return False", "g", "", spec)
        self.assertFalse(r["pass"])


class Sandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = dict(sandbox.LIMITS)
        sandbox.LIMITS.update({"visible": (1.0, 4.0), "extra": (1.0, 4.0), "hidden": (1.0, 6.0)})

    @classmethod
    def tearDownClass(cls):
        sandbox.LIMITS.clear()
        sandbox.LIMITS.update(cls.saved)

    def vis(self, src, tests=("assert f(1) == 2",)):
        return sandbox.run_visible(src, "f", "", list(tests))

    def test_pass_fail_and_noise(self):
        r = self.vis("def f(x):\n    print('noise' * 1000)\n    return x + 1\n", ["assert f(1) == 2", "assert f(2) == 4"])
        self.assertEqual((r["status"], r["load"]), ("ok", None))
        self.assertEqual([c["ok"] for c in r["cases"]], [True, False])

    def test_infinite_loop(self):
        r = self.vis("def f(x):\n    while True:\n        pass\n", ["assert f(1) == 2", "assert f(2) == 3"])
        self.assertEqual([c["err"] for c in r["cases"]], ["Timeout", "Timeout"])
        self.assertEqual(r["status"], "ok" if sandbox.POSIX else "timeout")  # per-case alarm on Linux only

    def test_huge_allocation(self):
        r = self.vis("def f(x):\n    return len(bytearray(10 ** 10))\n")  # 10 GB at once: over the limit
        self.assertEqual(r["cases"][0]["err"], "MemoryError")
        r = self.vis("def f(x):\n    a = []\n    while True:\n        a.append(bytearray(10 ** 8))\n")
        self.assertFalse(r["cases"][0]["ok"])  # growing: stopped by the memory limit or by the clock
        self.assertIn(r["cases"][0]["err"], ("MemoryError", "Timeout", "Crash"))

    def test_no_write_outside_temp(self):
        with tempfile.TemporaryDirectory() as d:
            victim = Path(d) / "victim.txt"
            victim.write_text("keep me")
            for body in (f"os.remove({str(victim)!r})", f"shutil.rmtree({d!r})", f"open({str(victim)!r}, 'w').write('x')",
                         f"os.rename({str(victim)!r}, 'stolen')"):
                r = self.vis(f"import os, shutil\ndef f(x):\n    {body}\n    return 2\n")
                self.assertEqual(r["cases"][0]["err"], "PermissionError", body)
            self.assertEqual(victim.read_text(), "keep me")

    def test_write_inside_temp_allowed(self):
        src = ("import os\ndef f(x):\n    with open('t.txt', 'w') as h:\n        h.write('ok')\n"
               "    os.mkdir('d')\n    os.remove('t.txt')\n    return 2\n")
        self.assertTrue(self.vis(src)["cases"][0]["ok"])

    def test_no_network_no_process(self):
        for body in ("socket.create_connection(('127.0.0.1', 9), timeout=1)", "socket.socket()",
                     "subprocess.run(['echo', 'hi'])", "os.system('echo hi')"):
            r = self.vis(f"import os, socket, subprocess\ndef f(x):\n    {body}\n    return 2\n")
            self.assertEqual(r["cases"][0]["err"], "PermissionError", body)

    def test_exit_during_load(self):
        self.assertEqual(self.vis("import sys\nsys.exit(3)\ndef f(x):\n    return 2\n")["load"], "SystemExit")
        self.assertEqual(self.vis("import os\nos._exit(0)\n")["status"], "crash")
        self.assertEqual(self.vis("def g(x):\n    return 2\n")["load"], "NoEntry")

    def test_signatures(self):
        inputs = [(3,), (0.1,), ("ab",), ([3, 1],)]
        a = sandbox.run_extra("def f(x):\n    return {'k': x, 'z': [x, x]} if not isinstance(x, float) else x + 0.2",
                              "f", "", inputs)
        b = sandbox.run_extra("def f(x):\n    if isinstance(x, float):\n        return 0.3\n"
                              "    d = {}\n    d['z'] = [x, x]\n    d['k'] = x\n    return d", "f", "", inputs)
        self.assertEqual(a["sigs"], b["sigs"])
        c = sandbox.run_extra("def f(x):\n    return {x} if isinstance(x, int) else 1 / 0", "f", "", inputs)
        self.assertEqual(c["sigs"][1:], ["!", "!", "!"])
        self.assertNotEqual(c["sigs"][0], a["sigs"][0])
        r = sandbox.run_extra("import random\ndef f(x):\n    return random.random()", "f", "", inputs)
        self.assertEqual(r["sigs"], sandbox.run_extra("import random\ndef f(x):\n    return random.random()",
                                                      "f", "", inputs)["sigs"])

    def test_hidden(self):
        spec = {"prelude": "def assertion(out, exp, atol):\n    assert out == exp",
                "setup": ["inputs = [[1], [2], [3]]", "results = [2, 3, 4]"], "target": "i, (inp, exp)",
                "iter": "enumerate(zip(inputs, results))", "body": "assertion(candidate(*inp), exp, 0)"}
        ok = sandbox.run_hidden("def f(x):\n    return x + 1", "f", "", spec)
        self.assertEqual((ok["pass"], ok["n_pass"], ok["n"]), (True, 3, 3))
        bad = sandbox.run_hidden("def f(x):\n    return x + 1 if x < 3 else 0", "f", "", spec)
        self.assertEqual((bad["pass"], bad["n_pass"], bad["err"]), (False, 2, "AssertionError"))
        mut = sandbox.run_hidden("def f(x):\n    return x + 1", "f", "def f(x):\n    'stub'", spec)
        self.assertTrue(mut["pass"])
        stub = sandbox.run_hidden("def g(x):\n    return x + 1", "f", "def f(x):\n    'stub'", spec)
        self.assertEqual((stub["pass"], stub["load"]), (False, "NoEntry"))

    SPEC = {"prelude": "def assertion(out, exp, atol):\n    assert out == exp", "setup": ["inputs = [[1], [2]]",
            "results = [2, 3]"], "target": "i, (inp, exp)", "iter": "enumerate(zip(inputs, results))",
            "body": "assertion(candidate(*inp), exp, 0)"}

    def test_expected_values_never_reach_the_candidate(self):
        # Audit 2026-10-09 (phase0, 2): `__import__('__main__')._tns['exp']` passed every hidden test.
        for body in ("return __import__('__main__')._tns['exp']",
                     "m = __import__('__main__')\n    return [v for k, v in vars(m).items() if 'exp' in k or 'result' in k][0]",
                     "import gc\n    return [o for o in gc.get_objects() if isinstance(o, list) and o == [2, 3]][0][x - 1]",
                     "__import__('__main__')._emit(kind='graded', hidden={'ok': True, 'n_pass': 2, 'n': 2, 'err': None})\n"
                     "    return 0"):
            r = sandbox.run_hidden(f"def f(x):\n    {body}", "f", "", self.SPEC)
            self.assertFalse(r["pass"], body)
        self.assertTrue(sandbox.run_hidden("def f(x):\n    return x + 1", "f", "", self.SPEC)["pass"])

    def test_no_read_of_the_data(self):
        secret = Path(sandbox.PHASE0) / "data" / "_sandbox_secret_test.txt"
        secret.parent.mkdir(exist_ok=True)
        secret.write_text("42")
        try:
            r = self.vis(f"def f(x):\n    return int(open({str(secret)!r}).read()) - 40\n")
            self.assertEqual(r["cases"][0]["err"], "PermissionError")
        finally:
            secret.unlink()
        with tempfile.TemporaryDirectory() as d:  # anywhere outside the Python installation
            (Path(d) / "x.txt").write_text("2")
            r = self.vis(f"def f(x):\n    return int(open({str(Path(d) / 'x.txt')!r}).read())\n")
            self.assertEqual(r["cases"][0]["err"], "PermissionError")
        self.assertTrue(self.vis("import json, math, collections\ndef f(x):\n    return x + 1\n")["cases"][0]["ok"])

    def test_dir_fd_refused(self):
        # Audit 2026-10-09 (phase0, 3): os.remove("victim", dir_fd=fd) was checked relative to the temp dir.
        with tempfile.TemporaryDirectory() as d:
            victim = Path(d) / "victim.txt"
            victim.write_text("keep me")
            for body in (f"fd = os.open({d!r}, os.O_RDONLY)\n    os.remove('victim.txt', dir_fd=fd)",
                         "fd = os.open(sys.prefix, os.O_RDONLY)\n    os.remove('x', dir_fd=fd)",
                         "fd = os.open('.', os.O_RDONLY)\n    os.remove('../x', dir_fd=fd)",
                         f"fd = os.open({d!r}, os.O_RDONLY)\n    os.open('victim.txt', os.O_WRONLY | os.O_TRUNC, dir_fd=fd)"):
                r = self.vis(f"import os, sys\ndef f(x):\n    {body}\n    return 2\n")
                self.assertIn(r["cases"][0]["err"], ("PermissionError",) if sandbox.POSIX else
                              ("PermissionError", "NotImplementedError"), body)
            self.assertEqual(victim.read_text(), "keep me")

    def test_large_output_by_digest(self):
        spec = {"prelude": "def assertion(out, exp, atol):\n    assert out == exp\ndef ref_func(n):\n"
                           "    return [(i, str(i)) for i in range(n)]", "setup": ["inputs = [[3], [400]]"],
                "target": "i, inp", "iter": "enumerate(inputs)", "body": "assertion(candidate(*inp), ref_func(*inp), 0)"}
        saved = sandbox.NODE_BUDGET
        sandbox.NODE_BUDGET = 100  # the second output (1201 nodes) travels as a digest
        try:
            ok = sandbox.run_hidden("def f(n):\n    return [(i, str(i)) for i in range(n)]", "f", "", spec)
            self.assertEqual((ok["pass"], ok["n_pass"]), (True, 2))
            for wrong in ("[(i, str(i)) for i in range(n - 1)] + [(0, '0')]", "[[i, str(i)] for i in range(n)]"):
                bad = sandbox.run_hidden(f"def f(n):\n    if n < 10:\n        return [(i, str(i)) for i in range(n)]\n"
                                         f"    return {wrong}", "f", "", spec)
                self.assertEqual((bad["pass"], bad["n_pass"]), (False, 1), wrong)
        finally:
            sandbox.NODE_BUDGET = saved

    def test_visible_replay(self):
        tests = ["assert set(f((3, 4), (4, 5))) == set((4,))", "assert f((1,), (1,)) == [1]",
                 "assert math.isclose(len(f((1, 2), (2,))), 1.0)", "assert f(*[(1,), (2,)]) == []", "assert True"]
        r = sandbox.run_visible("def f(a, b):\n    return sorted(set(a) & set(b))", "f", "import math", tests)
        self.assertEqual([c["ok"] for c in r["cases"]], [True, True, True, True, False])
        self.assertEqual(r["cases"][4]["err"], "NotReplayable")  # no call of the entry point
        r = sandbox.run_visible("def f(x):\n    return [x, 0.1 + 0.2]", "f", "",
                                ["assert _close(f(*(1,)), [1, 0.3])", "assert _close(f(*(2,)), [1, 0.3])"])
        self.assertEqual([c["ok"] for c in r["cases"]], [True, False])

    def test_numpy_and_no_native_calls(self):
        # Codex review: on Windows, `import numpy` loads kernel32 through ctypes, which the hook refused.
        r = self.vis("import numpy as np\ndef f(x):\n    return int(np.sum(np.arange(x + 2)))\n", ["assert f(1) == 3"])
        self.assertTrue(r["cases"][0]["ok"], r)
        r = self.vis("import ctypes\ndef f(x):\n    ctypes.CDLL(None if __import__('os').name == 'posix' else 'msvcrt')\n"
                     "    return 2\n")
        self.assertEqual(r["cases"][0]["err"], "PermissionError")

    def test_isolation_survives_json(self):
        # The execution manifest is compared with the stored one on resume: tuples (LIMITS) came back as lists
        # and every resume of exec_code.py was refused.
        iso = sandbox.isolation()
        self.assertEqual(json.loads(json.dumps(iso)), iso)

    def test_children_killed_on_interruption(self):
        seen = []

        def boom(p, data, timeout, kill):
            seen.append(p)
            raise KeyboardInterrupt

        saved = sandbox._exchange
        sandbox._exchange = boom
        try:
            with self.assertRaises(KeyboardInterrupt):
                sandbox._spawn(sandbox.CHILD, {"code": "import time\ntime.sleep(60)", "entry": "f", "header": "",
                                               "phase": "visible", "inputs": []}, "visible", sandboxed=True)
        finally:
            sandbox._exchange = saved
        self.assertIsNotNone(seen[0].poll())  # killed and reaped
        self.assertEqual(sandbox._LIVE, {})

    def test_shutdown_kills_running_children(self):
        import threading
        out = []
        t = threading.Thread(target=lambda: out.append(self.vis("import time\ntime.sleep(3)\ndef f(x):\n    return 2\n")))
        t0 = __import__("time").perf_counter()
        t.start()
        __import__("time").sleep(1.0)
        sandbox.shutdown()
        t.join(30)
        sandbox._STOP.clear()
        self.assertFalse(t.is_alive())
        self.assertLess(__import__("time").perf_counter() - t0, 3.0)  # killed, not waited for
        self.assertEqual(sandbox._LIVE, {})

    @unittest.skipUnless(sandbox.LINUX, "Landlock: Linux only")
    def test_landlock_enforced_by_the_kernel(self):
        if not sandbox.isolation()["landlock"].startswith("on"):
            self.skipTest("Landlock not available on this kernel")
        # os.listdir is not refused by the audit hook: only the kernel rules stop it outside the read roots
        data = str(Path(sandbox.PHASE0) / "data")
        r = self.vis(f"import os\ndef f(x):\n    return len(os.listdir({data!r})) * 0 + 2\n")
        self.assertEqual(r["cases"][0]["err"], "PermissionError")
        r = self.vis("import os\ndef f(x):\n    os.listdir('.')\n    return len(os.listdir(os.path.dirname(os.__file__))) * 0 + 2\n")
        self.assertTrue(r["cases"][0]["ok"])

    @unittest.skipUnless(sandbox.LINUX, "PR_SET_PDEATHSIG: Linux only")
    def test_child_dies_with_its_parent(self):
        import signal
        import subprocess
        import time
        # The candidate disarms its alarm and sleeps for a minute; its parent is then killed with SIGKILL (no
        # cleanup can run): the child must die at once, not when its own limits expire.
        cand = ("import signal, time\nsignal.setitimer(signal.ITIMER_REAL, 0)\n"
                "signal.signal(signal.SIGALRM, signal.SIG_IGN)\ntime.sleep(60)\n")
        prog = ("import sys, time\nsys.path.insert(0, %r)\nfrom essaim import sandbox\n"
                "def ex(p, d, t, k):\n    p.stdin.write(d)\n    p.stdin.close()\n    print(p.pid, flush=True)\n"
                "    time.sleep(60)\nsandbox._exchange = ex\n"
                "sandbox._spawn(sandbox.CHILD, {'code': %r, 'entry': 'f', 'header': '', 'phase': 'hidden', "
                "'inputs': []}, 'hidden', sandboxed=True)\n") % (str(sandbox.PHASE0), cand)
        mid = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE, text=True)
        pid = int(mid.stdout.readline())
        time.sleep(1.0)  # the child is now sleeping in the candidate's code
        os.kill(pid, 0)
        mid.send_signal(signal.SIGKILL)
        mid.wait()
        mid.stdout.close()
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("the sandbox child outlived its parent")


def C(fam, sample=0, prog=None, text=None, qid="q"):
    prog = prog or f"{fam}{sample}"
    return ac.Cand(f"m-{fam}", fam, sample, prog, text if text is not None else prog, qid)


class Signatures(unittest.TestCase):
    def test_unambiguous(self):
        # Audit 2026-10-09 (phase0, 7): (1,) and ["tuple", 1] had the same signature, and big ints lost their
        # sign and high bits.
        vals = [(1,), ["tuple", 1], [1], 2 ** 260, -2 ** 260, 2 ** 260 + 2 ** 258, "nan", float("nan"), {1: 2},
                ["dict", [1, 2]], {1}, ["set", 1], frozenset({1}), b"a", "a", 1, 1.0, True, None, "None"]
        sigs = [sandbox.signature(v) for v in vals]
        self.assertEqual(len(set(sigs)), len(vals))
        self.assertEqual(sandbox.signature({"a": 1, "b": 2}), sandbox.signature({"b": 2, "a": 1}))
        self.assertEqual(sandbox.signature(0.1 + 0.2), sandbox.signature(0.3))  # 9 significant digits
        self.assertEqual(sandbox.signature(-0.0), sandbox.signature(0.0))

    def test_codec_round_trip(self):
        for v in [None, True, 3, -2 ** 300, 0.1, float("inf"), -0.0, 1 + 2j, "é\ud800", b"\x00", bytearray(b"x"),
                  [1, (2, [3])], {(1, "a"): {2: [None]}}, {"b": 1, "a": 2, 3: 0}, {1, 2}, frozenset({"x"})]:
            with self.subTest(v=repr(v)):
                self.assertEqual(repr(sandbox.dec(sandbox.enc_text(v))), repr(v))
        for bad in ('["i", "12"]', '["x"]', '["l", 3]', '["d", [[["l", []], ["N"]]]]', '"plain"', '["f", "zz"]',
                    '["i", "0x' + "f" * 2000000 + '"]', "[" * 300 + '"N"' + "]" * 300):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                sandbox.dec(bad)

    def test_child_uses_the_same_signatures(self):
        r = sandbox.run_extra("def f(x):\n    return (x,) if x else ['tuple', x]", "f", "", [(1,), (0,)])
        self.assertEqual(r["sigs"], [sandbox.signature((1,)), sandbox.signature(["tuple", 0])])


def Fe(vis=True, sig=("a",), correct=False, load=True, vis_n=None):
    return ac.Feat(load, vis, vis_n if vis_n is not None else int(vis), sig, correct)


class Selection(unittest.TestCase):
    order = {"A": 0, "B": -1, "C": -2}

    def test_text_vote(self):
        cs = [C("A", text="x"), C("B", text="y"), C("C", text="y")]
        self.assertEqual(ac.text_vote(cs, {"A": 3, "B": 1, "C": 1}, self.order).fam, "B")
        cs = [C("A", text="x"), C("B", text="y")]  # tie: heavier family
        self.assertEqual(ac.text_vote(cs, {"A": 1, "B": 2}, self.order).fam, "B")

    def test_visible_weight(self):
        a, b, c = C("A"), C("B"), C("C", 1)
        F = {a: Fe(vis=False, vis_n=1), b: Fe(vis=True), c: Fe(vis=True)}
        self.assertEqual(ac.visible_weight([a, b, c], F, {"A": 5, "B": 1, "C": 2}, self.order), c)
        F = {a: Fe(vis=False, vis_n=2), b: Fe(vis=False, vis_n=1), c: Fe(vis=False, vis_n=0)}
        self.assertEqual(ac.visible_weight([a, b, c], F, {"A": 0, "B": 9, "C": 9}, self.order), a)

    def test_functional_scores(self):
        a0, a1, a2, b, c = C("A"), C("A", 1), C("A", 2), C("B"), C("C")
        F = {a0: Fe(sig=("x",)), a1: Fe(sig=("x",)), a2: Fe(sig=("x",)), b: Fe(sig=("y",)), c: Fe(sig=("y",))}
        w = {"A": 1.0, "B": 1.0, "C": 1.0}
        self.assertEqual(ac.functional([a0, a1, a2, b, c], F, w, self.order, "count")[0], a0)
        sel, info = ac.functional([a0, a1, a2, b, c], F, w, self.order, "families")
        self.assertEqual((sel.fam, info["top_fams"], info["n_clusters"]), ("B", 2, 2))
        self.assertEqual(ac.functional([a0, a1, a2, b, c], F, {"A": 3, "B": 1, "C": 1}, self.order, "wfamilies")[0], a0)

    def test_functional_ignores_visible_failures_and_cascade(self):
        a, b, c = C("A"), C("B"), C("C")
        F = {a: Fe(vis=False, sig=("x",)), b: Fe(sig=("y",)), c: Fe(vis=False, sig=("y",))}
        sel, info = ac.functional([a, b, c], F, {"A": 1, "B": 1, "C": 1}, self.order, "families")
        self.assertEqual((sel, info["any_pass"], info["top_fams"]), (b, True, 1))
        ref = C("R")
        self.assertEqual(ac.cascade([a, b, c], F, {}, self.order, "families", ref, 1), (b, False))
        self.assertEqual(ac.cascade([a, b, c], F, {}, self.order, "families", ref, 2), (ref, True))
        F = {a: Fe(vis=False), b: Fe(vis=False), c: Fe(vis=False)}
        self.assertEqual(ac.cascade([a, b, c], F, {}, self.order, "families", ref, 1), (ref, True))

    def test_no_extra_inputs_is_no_evidence(self):
        a, b, c = C("A"), C("B"), C("C")
        F = {a: Fe(sig=()), b: Fe(sig=()), c: Fe(sig=())}
        sel, info = ac.functional([a, b, c], F, {"A": 1, "B": 3, "C": 1}, self.order, "families")
        self.assertEqual((sel, info["top_fams"]), (b, 0))
        self.assertEqual(ac.cascade([a, b, c], F, {}, self.order, "families", C("R"), 1)[1], True)
        r = ac.pair_rates([[(a, F[a]), (b, F[b])]])
        self.assertIsNone(r["c"])
        self.assertEqual(r["vis_fa"], 1.0)  # visible-test acceptance does not depend on extra inputs

    def test_check_splits(self):
        man = {"model": "m", "gguf": "x.gguf", "prompt": "code-v1", "n": 3}
        ex = {"tests": "t", "extra_n": 16}
        ac.check_splits("b", {"manifests": {"m": man}, "exec": ex}, {"manifests": {"m": {**man, "n": 4}}, "exec": ex})
        with self.assertRaises(SystemExit):
            ac.check_splits("b", {"manifests": {"m": man}, "exec": ex},
                            {"manifests": {"m": {**man, "gguf": "y.gguf"}}, "exec": ex})
        with self.assertRaises(SystemExit):
            ac.check_splits("b", {"manifests": {"m": man}, "exec": ex},
                            {"manifests": {"m": man}, "exec": {**ex, "extra_n": 8}})

    def test_failures_are_not_agreement(self):
        # Audit 2026-10-09 (phase0, 8): three programs failing everywhere formed a cluster of three families.
        a, b, c = C("A"), C("B"), C("C")
        F = {a: Fe(sig=("!", "!")), b: Fe(sig=("!", "!")), c: Fe(sig=("!", "!"))}
        sel, info = ac.functional([a, b, c], F, {"A": 1, "B": 3, "C": 1}, self.order, "families")
        self.assertEqual((sel, info["top_fams"], info["n_clusters"]), (b, 0, 0))
        self.assertEqual(ac.cascade([a, b, c], F, {}, self.order, "families", C("R"), 3)[1], True)
        r = ac.pair_rates([[(a, F[a]), (b, F[b]), (c, F[c])]])
        self.assertIsNone(r["c"])
        # one valid output in common is agreement; failing on the same input is allowed
        F = {a: Fe(sig=("!", "x")), b: Fe(sig=("!", "x")), c: Fe(sig=("!", "!"))}
        sel, info = ac.functional([a, b, c], F, {"A": 1, "B": 1, "C": 9}, self.order, "families")
        self.assertEqual((info["top_fams"], info["n_clusters"]), (2, 1))
        self.assertIn(sel, (a, b))

    def test_features_keyed_by_problem(self):
        # Audit 2026-10-09 (phase0, 1): the same program answering two problems shared one feature entry.
        ident = {"dataset": "t", "sha256": "0"}
        items = [{"id": "H/1", "entry_point": "f"}, {"id": "H/2", "entry_point": "f"}]
        text = "```python\ndef f(x):\n    return 1\n```"
        src, _ = code.extract_code(text, "f")
        prog = code.prog_id(src)
        with tempfile.TemporaryDirectory() as d:
            saved = (code.RESULTS, ac.data.CODE_LOADERS["humanevalplus"], ac.data.dataset_identity)
            code.RESULTS = Path(d)
            ac.data.CODE_LOADERS["humanevalplus"] = lambda n, split: items
            ac.data.dataset_identity = lambda b: ident
            try:
                gp = code.gen_path("m", "_t", "humanevalplus", "test")
                gp.write_text("".join(json.dumps({"id": i["id"], "sample": 0, "text": text}) + "\n" for i in items))
                gp.with_name(gp.name + ".meta.json").write_text(json.dumps({"n": 2, "samples": 0, "data": ident}))
                ep = code.exec_path("e11", "_t", "humanevalplus", "test")
                rows = [{"id": i["id"], "prog": "reference", "load": None, "visible": [], "extra": [],
                         "hidden": {"pass": True}} for i in items]
                rows += [{"id": "H/1", "prog": prog, "load": None, "visible": [], "extra": ["a"], "hidden": {"pass": True}},
                         {"id": "H/2", "prog": prog, "load": None, "visible": [], "extra": ["b"], "hidden": {"pass": False}}]
                ep.write_text("".join(json.dumps(r) + "\n" for r in rows))
                ep.with_name(ep.name + ".meta.json").write_text(json.dumps(
                    {"data": ident, "tests": code.TESTS_VERSION, "extract": code.EXTRACT_VERSION}))
                ids, cands, feats, _ = ac.load("humanevalplus", "test", "_t", "e11", {"m": "A"})
            finally:
                code.RESULTS, ac.data.CODE_LOADERS["humanevalplus"], ac.data.dataset_identity = saved
        self.assertEqual(len(feats), 2)
        self.assertEqual([feats[cands[q][0]].correct for q in ids], [True, False])
        self.assertEqual([feats[cands[q][0]].sig for q in ids], [("a",), ("b",)])

    def test_pair_rates(self):
        a, b, c, d = C("A"), C("B"), C("C"), C("A", 1)
        q = [(a, Fe(sig=("x",))), (b, Fe(sig=("x",))), (c, Fe(sig=("z",))), (d, Fe(sig=("x",)))]
        r = ac.pair_rates([q])
        self.assertAlmostEqual(r["c"], 2 / 5)  # pairs AB, BD agree; AC, BC, CD do not; AD is one family
        q2 = [(a, Fe(sig=("x",), correct=True)), (b, Fe(sig=("x",), correct=True)), (c, Fe(sig=("y",), correct=True))]
        self.assertAlmostEqual(ac.pair_rates([q2])["a"], 1 / 3)

    def test_weights(self):
        w = ac.weights({"A": 0.9, "B": 0.5, "C": 0.1}, 0.2)
        self.assertGreater(w["A"], w["B"])
        self.assertGreater(w["B"], w["C"])
        self.assertEqual(w["C"], 0.0)


class ColabPlan(unittest.TestCase):
    def test_code_plan(self):
        jobs = colab_jobs.PLANS["code-1"]
        self.assertEqual({j[1] for j in jobs if j[0] == "code"}, set(colab_jobs.SOLO))
        self.assertEqual([colab_jobs.stage_of(j) for j in jobs if j[0] == "code-exec"], [1])
        self.assertTrue(all(colab_jobs.stage_of(j) == 0 for j in jobs if j[0] == "code"))
        cmd = colab_jobs.job_cmd("code-exec", colab_jobs.CODE_TAG, "all")
        self.assertIn("exec_code.py", cmd)
        self.assertEqual(cmd[cmd.index("--suffix") + 1], colab_jobs.SUFFIX)
        self.assertIn("code-exec", colab_jobs.NO_MODEL)


if __name__ == "__main__":
    unittest.main()
