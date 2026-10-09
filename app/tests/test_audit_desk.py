"""Regression tests of the desktop audit (2026-10-09): cancellation during the restart, shutdown of the
native window, concurrent installs, extraction and cancellation, half-extracted engines, disk space
with partial downloads, llama-server slots, macOS minimum, release workflow."""
from __future__ import annotations

import asyncio
import re
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

from myriad import llamacpp
from myriad.config import Config
from myriad.hardware import Hardware
from myriad.runtime import NodeRuntime
from myriad.ui import make_ui_app
from myriad.wizard import SetupError, SetupWizard

from .conftest import wait_until
from .test_desktop import MODEL_BYTES, FakeRuntime, fake_hw, tiny_release  # noqa: F401 - fixture

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ 1. cancel during the restart
class _Closable:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _SlowNode:
    def __init__(self):
        self.stopped = False

    async def stop(self):
        await asyncio.sleep(0.3)
        self.stopped = True


async def test_runtime_stop_finishes_even_when_cancelled():
    node, gw, eng = _SlowNode(), _Closable(), _Closable()
    rt = NodeRuntime.attached(node, gw, Config(), None)
    rt.engine = eng
    t = asyncio.create_task(rt.stop())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert node.stopped and gw.closed and eng.closed  # llama-server terminated all the same
    assert rt.node is None and rt.engine is None and rt.state == "arrêté"


class _SlowRestartRuntime(FakeRuntime):
    def __init__(self, home):
        super().__init__(home)
        self.in_restart = asyncio.Event()
        self.restart_done = False

    async def restart(self):
        self.in_restart.set()
        await asyncio.sleep(0.4)
        self.restart_done = True
        self.restarts += 1


async def test_wizard_cancel_during_restart_waits_for_it(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    rt = _SlowRestartRuntime(home)
    wiz = SetupWizard(home, rt, detect=fake_hw, http=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(404))))
    await wiz.start({"model": None, "tracker_url": "http://127.0.0.1:8500"})
    await asyncio.wait_for(rt.in_restart.wait(), 5)
    await wiz.cancel()
    assert rt.restart_done  # the restart was not cut in the middle
    assert wiz.job.state == "terminé"  # too late to cancel: the new configuration is in place


# ------------------------------------------------------------------ 2. native window: quit waits for the stop
def test_second_quit_waits_for_the_first_one():
    from myriad.desktop import Quitter

    calls = []

    class Svc:
        def stop(self):
            calls.append("start")
            time.sleep(0.5)
            calls.append("end")

    q = Quitter(Svc())
    threading.Thread(target=q, daemon=True).start()  # e.g. on_closing
    time.sleep(0.1)
    t0 = time.monotonic()
    q()  # after webview.start() returned
    assert calls == ["start", "end"] and time.monotonic() - t0 >= 0.3
    assert q.quitting.is_set() and q.done.is_set()
    q()  # idempotent
    assert calls == ["start", "end"]


# ------------------------------------------------------------------ 3. concurrent installs
async def test_concurrent_installs_are_exclusive(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    def slow_hw():
        time.sleep(0.2)
        return fake_hw()

    rt = FakeRuntime(home)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as http:
        wiz = SetupWizard(home, rt, detect=slow_hw, http=http)
        body = {"model": None, "tracker_url": "http://127.0.0.1:8500"}
        res = await asyncio.gather(wiz.start(body), wiz.start(body), return_exceptions=True)
        assert sum(isinstance(r, SetupError) for r in res) == 1, res
        await wiz.job.task
        assert rt.restarts == 1


# ------------------------------------------------------------------ 4. cancel during the extraction
async def test_cancel_waits_for_the_extraction_worker(tmp_path, tiny_release, monkeypatch):
    transport, _ = tiny_release
    home = tmp_path / "home"
    home.mkdir()
    state = {"running": False, "started": threading.Event()}

    def slow_extract(archive, dest, cancel=None):
        state["running"] = True
        state["started"].set()
        try:
            for _ in range(40):  # ~2 s, unless told to stop
                if cancel is not None and cancel.is_set():
                    return
                time.sleep(0.05)
        finally:
            state["running"] = False

    monkeypatch.setattr(llamacpp, "extract", slow_extract)
    async with httpx.AsyncClient(transport=transport) as http:
        wiz = SetupWizard(home, FakeRuntime(home), detect=fake_hw, http=http)
        await wiz.start({"model": "tiny", "tracker_url": "http://127.0.0.1:8500"})
        await asyncio.to_thread(state["started"].wait, 5)
        t0 = time.monotonic()
        await wiz.cancel()
        assert not state["running"]  # no worker left writing into the .tmp directory
        assert time.monotonic() - t0 < 1.5  # and it was told to stop early
        assert wiz.job.state == "annulé"


# ------------------------------------------------------------------ 5. half-extracted engines
def test_temporary_engine_directory_is_not_an_install(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    tmp = tmp_path / "llama.cpp" / f"{llamacpp.BUILD}-cpu.tmp"
    tmp.mkdir(parents=True)
    (tmp / llamacpp.EXE).write_bytes(b"x")
    assert llamacpp.find_existing(tmp_path) is None
    assert llamacpp.find_existing(tmp_path, str(tmp / llamacpp.EXE)) is None  # saved by an older version
    done =tmp_path / "llama.cpp" / f"{llamacpp.BUILD}-cpu"
    done.mkdir()
    (done / llamacpp.EXE).write_bytes(b"x")
    assert llamacpp.find_existing(tmp_path) == str(done / llamacpp.EXE)


# ------------------------------------------------------------------ 6. disk space and partial downloads
async def test_disk_check_counts_partial_downloads(tmp_path, tiny_release, monkeypatch):
    transport, _ = tiny_release
    home = tmp_path / "home"
    home.mkdir()
    from myriad import catalog
    from myriad import wizard as wmod
    part = wmod.model_path(home, catalog.get("tiny"), "Q4_K_M")
    part = part.with_name(part.name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(MODEL_BYTES[:5000])  # 4 bytes left to download
    monkeypatch.setattr(wmod.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=(1 << 30) + 1000))
    async with httpx.AsyncClient(transport=transport) as http:
        wiz = SetupWizard(home, FakeRuntime(home), detect=fake_hw, http=http)
        await wiz.start({"model": "tiny", "tracker_url": "http://127.0.0.1:8500"})
        await wiz.job.task
        assert wiz.job.state == "terminé", wiz.job.error


# ------------------------------------------------------------------ 7. jobs in parallel vs llama-server slots
class _LimitNode:
    def __init__(self):
        self.max_parallel, self.accepting, self.active_hours = 1, True, None

    async def set_limits(self, max_parallel=None, accepting=None, active_hours=...):
        if max_parallel is not None:
            self.max_parallel = max_parallel
        if accepting is not None:
            self.accepting = accepting
        if active_hours is not ...:
            self.active_hours = active_hours

    def status(self):
        return {"max_parallel": self.max_parallel}


async def test_parallel_jobs_never_exceed_engine_slots(tmp_path):
    node = _LimitNode()
    rt = NodeRuntime.attached(node, None, Config(), None)
    rt.engine = types.SimpleNamespace(parallel=1)  # llama-server started with -np 1
    app = make_ui_app(token="tok", runtime=rt)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        r = await c.post("/api/limits", json={"max_parallel": 8}, headers={"X-Myriad-Token": "tok"})
        assert r.status_code == 200
    assert node.max_parallel == 1  # cannot restart this engine: capped to its slots


class _RestartableRuntime(FakeRuntime):
    def __init__(self, home):
        super().__init__(home)
        self.node = _LimitNode()
        self.engine = types.SimpleNamespace(parallel=1)
        self.config = Config()

    async def restart(self):
        self.restarts += 1
        cfg = Config.load(self.home)
        self.node = _LimitNode()
        self.node.max_parallel = cfg.max_parallel
        self.engine = types.SimpleNamespace(parallel=cfg.max_parallel)


async def test_more_parallel_jobs_restart_the_engine(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    Config().save(home)
    rt = _RestartableRuntime(home)
    app = make_ui_app(token="tok", runtime=rt)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        r = await c.post("/api/limits", json={"max_parallel": 4}, headers={"X-Myriad-Token": "tok"})
        assert r.status_code == 200 and r.json()["restarted"] is True
        assert rt.restarts == 1 and rt.engine.parallel == 4 and rt.node.max_parallel == 4
        assert Config.load(home).max_parallel == 4
        r = await c.post("/api/limits", json={"max_parallel": 2}, headers={"X-Myriad-Token": "tok"})
        assert r.json()["restarted"] is False and rt.restarts == 1  # fewer jobs than slots: no restart
        assert rt.node.max_parallel == 2 and Config.load(home).max_parallel == 2


# ------------------------------------------------------------------ 8. macOS minimum version
def test_macos_minimum_matches_the_engine(tmp_path, tiny_release, monkeypatch):
    spec = (ROOT / "app" / "packaging" / "myriad.spec").read_text(encoding="utf-8")
    assert re.search(r'"LSMinimumSystemVersion":\s*"13\.3"', spec)
    assert llamacpp.unsupported_reason("macos", (12, 7)) and llamacpp.unsupported_reason("macos", (13, 3)) is None
    assert llamacpp.unsupported_reason("windows", None) is None


async def test_wizard_refuses_engine_on_old_macos(tmp_path, tiny_release, monkeypatch):
    transport, _ = tiny_release
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setitem(llamacpp.ASSETS, ("macos", "x64", "cpu"), llamacpp.ASSETS[("testos", "x64", "cpu")])
    monkeypatch.setattr(llamacpp, "macos_version", lambda: (12, 7))
    mac = lambda: Hardware("macos", "x64", "Intel", 8, 16.0, [])  # noqa: E731
    async with httpx.AsyncClient(transport=transport) as http:
        wiz = SetupWizard(home, FakeRuntime(home), detect=mac, http=http)
        await wiz.start({"model": "tiny", "tracker_url": "http://127.0.0.1:8500"})
        await wiz.job.task
        assert wiz.job.state == "erreur" and "13.3" in wiz.job.error


# ------------------------------------------------------------------ 9, 10. release workflow
def _run_blocks(text: str) -> list[str]:
    """The shell scripts of a workflow (`run: |` blocks and one-line `run:`)."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = re.match(r"^(\s*)(?:- )?run:\s*(\|)?\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        if m.group(2):
            ind = len(m.group(1))
            block = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > ind):
                block.append(lines[i])
                i += 1
            out.append("\n".join(block))
        else:
            out.append(m.group(3))
            i += 1
    return out


def test_release_workflow_tags_the_built_commit_and_keeps_secrets_out_of_scripts():
    wf = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    scripts = _run_blocks(wf)
    assert scripts
    for s in scripts:
        assert "secrets." not in s, s  # secrets reach the scripts through the environment only
        assert "inputs." not in s, s
    create = next(s for s in scripts if "gh release create" in s)
    assert '--target "$GITHUB_SHA"' in create
    assert "$GITHUB_SHA" in create.split("gh release create")[0]  # an existing tag must name the built commit
