"""Desktop app back end: hardware parsing, model catalogue and recommendation, resumable verified
downloads, llama.cpp release table and archive safety, setup wizard API, live chat stream, network
statistics, single-instance lock."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from myriad import catalog, hardware, llamacpp
from myriad.config import Config, key_path
from myriad.desktop import InstanceLock
from myriad.downloader import DownloadError, Progress, download
from myriad.hardware import Gpu, Hardware
from myriad.runtime import is_configured
from myriad.ui import make_ui_app, my_tokens_per_s
from myriad.wizard import SetupWizard

from .conftest import wait_until

# ------------------------------------------------------------------ hardware

NVIDIA_SMI = "NVIDIA GeForce RTX 3060, 12288, 560.94\nNVIDIA GeForce GTX 1660 SUPER, 6144, 560.94\n"
VULKAN_SUMMARY = """==========
VULKANINFO
==========

Devices:
========
GPU0:
	apiVersion         = 1.3.280
	driverVersion      = 2.0.302
	vendorID           = 0x1002
	deviceID           = 0x73ef
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = AMD Radeon RX 6650 XT
GPU1:
	apiVersion         = 1.3.274
	vendorID           = 0x10005
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
"""


def test_parse_nvidia_smi():
    g = hardware.parse_nvidia_smi(NVIDIA_SMI + "garbage\n, \n")
    assert [x.name for x in g] == ["NVIDIA GeForce RTX 3060", "NVIDIA GeForce GTX 1660 SUPER"]
    assert g[0].vram_gb == 12.0 and g[0].vendor == "nvidia" and g[0].driver == "560.94"


def test_parse_vulkan_meminfo_cpuinfo_apple():
    g = hardware.parse_vulkaninfo_summary(VULKAN_SUMMARY)
    assert len(g) == 1 and g[0].name == "AMD Radeon RX 6650 XT" and g[0].vendor == "amd"
    assert hardware.parse_meminfo("MemTotal:       16303428 kB\nMemFree: 1 kB\n") == pytest.approx(15.55, abs=0.01)
    assert hardware.parse_cpuinfo("processor : 0\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\n") \
        == "AMD Ryzen 7 5800X 8-Core Processor"
    assert hardware.apple_chip("Apple M2 Pro") == "m2 pro" and hardware.apple_chip("Apple M1") == "m1"
    assert hardware.apple_chip("Intel(R) Core(TM) i7-9750H") is None
    assert hardware.vendor_of("Intel(R) UHD Graphics 630") == "intel"


def test_gpu_choice_budget_and_speed():
    rtx = Gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0)
    igpu = Gpu("AMD Radeon(TM) Graphics", "amd", 0.5)
    hw = Hardware("windows", "x64", "cpu", 16, 64.0, [igpu, rtx])
    assert hw.best_gpu is rtx and hardware.accelerator(hw) == "cuda"
    assert hardware.model_budget_gb(hw) == pytest.approx(22.8)
    assert hardware.bandwidth_gbs(hw) == 1008
    assert hardware.estimate_tps(hw, 5.0) == pytest.approx(0.6 * 1008 / 5.0)
    cpu_only = Hardware("linux", "x64", "cpu", 8, 16.0, [igpu])  # an iGPU with shared memory is not used
    assert cpu_only.best_gpu is None and hardware.accelerator(cpu_only) == "cpu"
    assert hardware.model_budget_gb(cpu_only) == pytest.approx(6.4)
    mac = Hardware("macos", "arm64", "Apple M2 Pro", 12, 16.0, [Gpu("Apple M2 Pro", "apple", 16.0, unified=True)])
    assert hardware.accelerator(mac) == "metal" and hardware.bandwidth_gbs(mac) == 200
    assert hardware.model_budget_gb(mac) == pytest.approx(9.6)
    d = mac.to_dict()
    assert d["accelerator"] == "metal" and d["best_gpu"]["unified"] is True
    merged = hardware.merge_gpus([Gpu("AMD Radeon RX 6650 XT", "amd", None)], [Gpu("AMD Radeon RX 6650 XT", "amd", 7.98)])
    assert len(merged) == 1 and merged[0].vram_gb == 7.98


# ------------------------------------------------------------------ catalogue

def test_catalog_is_permissive_and_pinned():
    ids = set()
    for m in catalog.CATALOG:
        assert m.licence in ("apache-2.0", "mit") and m.id not in ids
        ids.add(m.id)
        assert len(m.revision) == 40 and set(catalog.QUANTS) <= set(m.files)
        for q, f in m.files.items():
            assert len(f.sha256) == 64 and f.size > 100e6 and f.file.endswith(".gguf")
            assert m.url(q) == f"https://huggingface.co/{m.repo}/resolve/{m.revision}/{f.file}"
            assert m.spec(q) == f"{m.repo}:{f.file}"
    assert len({m.family for m in catalog.CATALOG}) >= 5


def test_recommendation_follows_hardware_and_network():
    weak = Hardware("windows", "x64", "cpu", 4, 8.0, [])
    rec = catalog.recommend(weak)
    assert rec["quant"] == "Q4_K_M"
    assert catalog.get(rec["model"]).file_for("Q4_K_M").size < 1.5e9  # only a small model fits 8 GB of RAM
    big = Hardware("linux", "x64", "cpu", 16, 64.0, [Gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0)])
    r = catalog.recommend(big)
    assert catalog.get(r["model"]).params_b >= 3.8 and r["quant"] == "Q8_0"
    # A family missing from the network gets a bonus over one that is already everywhere.
    mid = Hardware("windows", "x64", "cpu", 12, 32.0, [Gpu("AMD Radeon RX 6650 XT", "amd", 8.0)])
    crowded = {f: 50 for f in {m.family for m in catalog.CATALOG}}
    crowded.pop("granite")
    assert catalog.get(catalog.recommend(mid, crowded)["model"]).family == "granite"
    ev = catalog.evaluate(weak, {"qwen": 3})
    e4b = next(m for m in ev if m["id"] == "gemma-4-e4b")
    assert e4b["quants"]["Q8_0"]["fits"] is False and next(m for m in ev if m["family"] == "qwen")["peers"] == 3
    with pytest.raises(KeyError):
        catalog.get("nope")


# ------------------------------------------------------------------ downloads

DATA = bytes(range(256)) * 4000  # ~1 MB
SHA = hashlib.sha256(DATA).hexdigest()


def range_server(data: bytes, honour_range: bool = True, log: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        if log is not None:
            log.append(rng)
        if rng and honour_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                return httpx.Response(416)
            return httpx.Response(206, content=data[start:],
                                  headers={"content-range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
        return httpx.Response(200, content=data)
    return httpx.MockTransport(handler)


async def test_download_fresh_and_skip_existing(tmp_path):
    dest = tmp_path / "m" / "model.gguf"
    log: list = []
    async with httpx.AsyncClient(transport=range_server(DATA, log=log)) as c:
        prog = Progress("model")
        await download("https://x/model.gguf", dest, sha256=SHA, size=len(DATA), client=c, progress=prog)
        assert dest.read_bytes() == DATA and prog.state == "terminé" and prog.done == len(DATA)
        assert not dest.with_name("model.gguf.part").exists() and log == [None]
        await download("https://x/model.gguf", dest, sha256=SHA, size=len(DATA), client=c)
        assert log == [None]  # already there and verified: no request


async def test_download_resumes_with_range(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(DATA[:300_000])
    log: list = []
    async with httpx.AsyncClient(transport=range_server(DATA, log=log)) as c:
        prog = Progress("model")
        await download("https://x/m", dest, sha256=SHA, size=len(DATA), client=c, progress=prog)
    assert log == ["bytes=300000-"] and prog.resumed_from == 300_000 and dest.read_bytes() == DATA


async def test_download_restarts_when_range_ignored(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(b"x" * 1000)  # wrong bytes, the server ignores Range
    async with httpx.AsyncClient(transport=range_server(DATA, honour_range=False)) as c:
        await download("https://x/m", dest, sha256=SHA, size=len(DATA), client=c)
    assert dest.read_bytes() == DATA


async def test_download_complete_part_and_bad_digest(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(DATA)  # complete part: only the check remains
    log: list = []
    async with httpx.AsyncClient(transport=range_server(DATA, log=log)) as c:
        await download("https://x/m", dest, sha256=SHA, size=len(DATA), client=c)
        assert log == [] and dest.read_bytes() == DATA
        bad = tmp_path / "bad.gguf"
        with pytest.raises(DownloadError, match="SHA-256"):
            await download("https://x/m", bad, sha256="0" * 64, size=len(DATA), client=c)
        assert not bad.exists() and not bad.with_name("bad.gguf.part").exists()
        with pytest.raises(DownloadError, match="taille"):
            await download("https://x/m", tmp_path / "short.gguf", size=len(DATA) + 5, client=c)


async def test_download_cancel_keeps_part(tmp_path):
    dest = tmp_path / "model.gguf"
    cancel = asyncio.Event()
    cancel.set()
    async with httpx.AsyncClient(transport=range_server(DATA)) as c:
        with pytest.raises(asyncio.CancelledError):
            await download("https://x/m", dest, sha256=SHA, size=len(DATA), client=c, cancel=cancel)
    assert not dest.exists()


async def test_download_http_error(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as c:
        with pytest.raises(DownloadError, match="404"):
            await download("https://x/m", tmp_path / "f", client=c)


# ------------------------------------------------------------------ llama.cpp

def test_llamacpp_table_and_backend_choice():
    for (os_name, arch, backend), assets in llamacpp.ASSETS.items():
        assert assets and all(len(a.sha256) == 64 and a.size > 1e6 for a in assets)
        assert all(a.url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/b11505/") for a in assets)
        assert llamacpp.BUILD in assets[0].name
    assert llamacpp.default_backend("macos", "arm64", "metal") == "metal"
    assert llamacpp.default_backend("windows", "x64", "cuda") == "vulkan"
    assert llamacpp.default_backend("windows", "x64", "cpu") == "cpu"
    assert llamacpp.default_backend("linux", "x64", "vulkan") == "vulkan"
    assert llamacpp.default_backend("macos", "x64", "cpu") == "cpu"
    assert len(llamacpp.assets_for("windows", "x64", "cuda")) == 2  # with the CUDA runtime
    with pytest.raises(KeyError):
        llamacpp.assets_for("windows", "x64", "metal")


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_extract_and_find(tmp_path, monkeypatch):
    arc = tmp_path / "llama.zip"
    arc.write_bytes(_zip({f"build/bin/{llamacpp.EXE}": b"bin", "build/bin/ggml.dll": b"x"}))
    dest = tmp_path / "out"
    llamacpp.extract(arc, dest)
    found = llamacpp.find_in(dest)
    assert found is not None and found.read_bytes() == b"bin"
    evil = tmp_path / "evil.zip"
    evil.write_bytes(_zip({"../../escape.txt": b"x"}))
    with pytest.raises(ValueError):
        llamacpp.extract(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()
    home = tmp_path / "home"
    (home / "llama.cpp" / f"{llamacpp.BUILD}-vulkan").mkdir(parents=True)
    (home / "llama.cpp" / f"{llamacpp.BUILD}-vulkan" / llamacpp.EXE).write_bytes(b"x")
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    assert llamacpp.find_existing(home).endswith(llamacpp.EXE)
    monkeypatch.setenv("LLAMA_SERVER", str(found))
    assert llamacpp.find_existing(tmp_path / "empty") == str(found)


# ------------------------------------------------------------------ wizard API

class FakeRuntime:
    def __init__(self, home: Path):
        self.home, self.restarts = home, 0
        self.config = Config()
        self.node = self.gateway = None
        self.error = None

    def configured(self) -> bool:
        return is_configured(self.home)

    def status(self) -> dict:
        return {"state": "arrêté", "error": None, "gateway_error": None, "configured": self.configured()}

    async def restart(self) -> None:
        self.restarts += 1


MODEL_BYTES = b"GGUF" + bytes(5000)


@pytest.fixture
def tiny_release(monkeypatch):
    """A tiny model and a tiny llama.cpp archive served by a mock transport."""
    model = catalog.CatalogModel(
        id="tiny", name="Tiny", family="tinyfam", params_b=0.1, licence="apache-2.0", repo="org/Tiny-GGUF",
        revision="a" * 40, files={"Q4_K_M": catalog.ModelFile("tiny-Q4_K_M.gguf", len(MODEL_BYTES),
                                                               hashlib.sha256(MODEL_BYTES).hexdigest())},
        blurb_fr="", blurb_en="")
    monkeypatch.setitem(catalog.BY_ID, "tiny", model)
    archive = _zip({f"bin/{llamacpp.EXE}": b"#!llama"})
    asset = llamacpp.Asset(f"llama-{llamacpp.BUILD}-test.zip", len(archive), hashlib.sha256(archive).hexdigest())
    monkeypatch.setitem(llamacpp.ASSETS, ("testos", "x64", "cpu"), (asset,))
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/v1/peers"):
            return httpx.Response(200, json={"peers": [{"family": "qwen", "model": "x"}]})
        if request.url.path.endswith(".gguf"):
            return httpx.Response(200, content=MODEL_BYTES)
        if request.url.path.endswith(".zip"):
            return httpx.Response(200, content=archive)
        return httpx.Response(404)
    return httpx.MockTransport(handler), seen


def fake_hw() -> Hardware:
    return Hardware("testos", "x64", "Test CPU", 8, 16.0, [])


async def test_wizard_api_install_flow(tmp_path, tiny_release):
    transport, seen = tiny_release
    home = tmp_path / "home"
    home.mkdir()
    rt = FakeRuntime(home)
    async with httpx.AsyncClient(transport=transport) as http:
        wiz = SetupWizard(home, rt, detect=fake_hw, http=http)
        app = make_ui_app(config=Config(), home=home, token="tok", runtime=rt, wizard=wiz)
        hdr = {"X-Myriad-Token": "tok"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
            st = (await c.get("/api/status")).json()
            assert st["node"] is None and st["runtime"]["configured"] is False and st["app"] == "Myriad"
            s = (await c.get("/api/setup")).json()
            assert s["hardware"]["cpu"] == "Test CPU" and s["network_families"] == {"qwen": 1}
            assert s["recommendation"]["model"] in {m["id"] for m in s["catalog"]}
            assert s["engine"]["default_backend"] == "cpu" and s["engine"]["existing"] is None
            body = {"model": "tiny", "quant": "Q4_K_M", "max_parallel": 2, "accepting": True, "active_hours": "8-23",
                    "tracker_url": "https://tracker.example.org"}
            assert (await c.post("/api/setup/install", json=body)).status_code == 403  # no token
            bad = await c.post("/api/setup/install", json={**body, "tracker_url": "ftp://x"}, headers=hdr)
            assert bad.status_code == 400 and "traqueur" in bad.json()["error"]
            bad = await c.post("/api/setup/install", json={**body, "max_parallel": 99}, headers=hdr)
            assert bad.status_code == 400
            bad = await c.post("/api/setup/install", json={**body, "model": "nope"}, headers=hdr)
            assert bad.status_code == 400
            r = await c.post("/api/setup/install", json=body, headers=hdr)
            assert r.status_code == 200, r.text

            async def done():
                p = (await c.get("/api/setup/progress")).json()["job"]
                return p if p["state"] != "en cours" else None
            job = None
            for _ in range(200):
                job = await done()
                if job:
                    break
                await asyncio.sleep(0.02)
            assert job and job["state"] == "terminé", job
            assert [s["id"] for s in job["steps"]] == [f"engine:llama-{llamacpp.BUILD}-test.zip", "model"]
            assert all(s["state"] == "terminé" for s in job["steps"])
    cfg = Config.load(home)
    assert cfg.model == "org/Tiny-GGUF:tiny-Q4_K_M.gguf" and Path(cfg.gguf_path).read_bytes() == MODEL_BYTES
    assert cfg.tracker_url == "https://tracker.example.org" and cfg.max_parallel == 2 and cfg.active_hours == "8-23"
    assert Path(cfg.llama_server).read_bytes() == b"#!llama" and cfg.family == "tinyfam"
    assert cfg.extra["catalog_id"] == "tiny" and cfg.extra["backend"] == "cpu"
    assert key_path(home).exists() and is_configured(home) and rt.restarts == 1
    assert any("/resolve/" + "a" * 40 + "/tiny-Q4_K_M.gguf" in u for u in seen)
    assert not (home / "downloads" / f"llama-{llamacpp.BUILD}-test.zip").exists()  # archive removed


async def test_wizard_client_only_and_disk_check(tmp_path, tiny_release, monkeypatch):
    transport, _ = tiny_release
    home = tmp_path / "home"
    home.mkdir()
    rt = FakeRuntime(home)
    async with httpx.AsyncClient(transport=transport) as http:
        wiz = SetupWizard(home, rt, detect=fake_hw, http=http)
        await wiz.start({"model": None, "tracker_url": "http://127.0.0.1:8500"})
        await wiz.job.task
        assert wiz.job.state == "terminé" and wiz.job.steps == []
        cfg = Config.load(home)
        assert cfg.model is None and is_configured(home) and rt.restarts == 1
        from myriad import wizard as wmod
        monkeypatch.setattr(wmod.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
        await wiz.start({"model": "tiny"})
        await wiz.job.task
        assert wiz.job.state == "erreur" and "disque" in wiz.job.error


# ------------------------------------------------------------------ live chat stream, stats, prefs

async def test_chat_stream_and_network_stats(swarm, tmp_path):
    gw, _ = await swarm.add_gateway()
    node = swarm.nodes["client"]
    app = make_ui_app(node, gw, Config(tracker_url=swarm.url), tmp_path, "tok")
    hdr = {"X-Myriad-Token": "tok"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401",
                                 timeout=30) as c:
        events = []
        async with c.stream("POST", "/api/chat/stream", headers=hdr,
                            json={"message": "Tom has 40 apples and buys 2 more. How many apples?"}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        kinds = [e["type"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "final"
        asked = [e for e in events if e["type"] == "asked"]
        assert len(asked) == 4
        # essaim/1.1: the tracker picks the peers and names them in Assigned frames (essaim/1: in the job).
        named = [e for e in events if e["type"] == "assigned"] or asked
        assert {e["family"] for e in named} == {"qwen", "smollm", "gemma", "granite"}
        answered = [e for e in events if e["type"] == "answered"]
        assert answered and all(e["answer"] == "42" for e in answered)
        assert any(e["type"] == "failed" for e in events)  # granite fails
        final = events[-1]["body"]["myriad"]
        assert final["answer"] == "42" and final["early_stop"] is True
        assert not node._observers  # the tap is removed at the end of the stream
        await wait_until(lambda: swarm.tracker.ledger.recent(50) and any(
            e["kind"] != "starter" for e in swarm.tracker.ledger.recent(50)))
        net = (await c.get("/api/network")).json()
        st = net["stats"]
        assert st["nodes_online"] == 5 and st["families"] == {"qwen": 1, "smollm": 1, "gemma": 1, "granite": 1}
        assert net["me"] == node.node_id and st["account"]["node_id"] == node.node_id
        assert st["tokens_per_s"] > 0 and st["jobs_per_min"] > 0
        await wait_until(lambda: True)
        # Stats straight from the tracker: the requester spent what the servers earned.
        from myriad.crypto import account_params
        async with httpx.AsyncClient(base_url=swarm.url) as t:
            s = (await t.get("/v1/stats", params={"node_id": node.node_id, **account_params(node.identity)})).json()
            assert s["account"]["spent"] > 0 and s["account"]["earned"] == 0
            qn = swarm.nodes["qwen"]
            s2 = (await t.get("/v1/stats", params={"node_id": qn.node_id, **account_params(qn.identity)})).json()
            assert s2["account"]["earned"] > 0 and s2["account"]["jobs_served"] >= 1
            # essaim/1.3: an account's figures are not public (they would link a debit to a served job)
            assert "account" not in (await t.get("/v1/stats", params={"node_id": qn.node_id})).json()
            assert (await t.get(f"/v1/balance/{qn.node_id}")).status_code == 403
            assert "balances" not in (await t.get("/v1/balances")).json()
        r = await c.post("/api/prefs", json={"lang": "en"}, headers=hdr)
        assert r.status_code == 200
        assert (await c.post("/api/prefs", json={"lang": "xx"}, headers=hdr)).status_code == 400
        assert (await c.get("/static/fonts/inter-latin-wght-normal.woff2")).headers["content-type"] == "font/woff2"
        assert (await c.get("/static/../ui.py")).status_code == 404
        assert "font-src 'self'" in (await c.get("/")).headers["content-security-policy"]


def test_my_tokens_per_s():
    now = 1000.0
    recent = [{"ts": 990.0, "tokens": 120}, {"ts": 900.0, "tokens": 999}, {"ts": 995.0}]
    assert my_tokens_per_s(recent, now) == 2.0
    # A long job (started 3 min ago) counts when it completes, not when it started.
    assert my_tokens_per_s([{"ts": 820.0, "done_ts": 998.0, "tokens": 300}], now) == 5.0


def test_client_only_node_needs_no_model(tmp_path):
    from myriad.crypto import Identity
    Identity.load_or_create(key_path(tmp_path))
    Config(model="org/X-GGUF:x.gguf", gguf_path=None).save(tmp_path)
    assert not is_configured(tmp_path) and is_configured(tmp_path, serve=False)


def test_tray_without_menu_is_not_used(monkeypatch, tmp_path):
    import types

    from myriad import desktop

    class NoMenuIcon:
        HAS_MENU = False

        def __init__(self, *a, **k):
            pass
    fake = types.SimpleNamespace(Icon=NoMenuIcon, Menu=lambda *a: None, MenuItem=lambda *a, **k: None)
    fake.Menu.SEPARATOR = None
    monkeypatch.setitem(__import__("sys").modules, "pystray", fake)
    monkeypatch.setattr(desktop, "tray_image", lambda: None)
    assert desktop.make_tray(None, lambda: None, lambda: None, tmp_path) is None


# ------------------------------------------------------------------ single instance, icon

def test_instance_lock(tmp_path):
    a, b = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_icon_geometry():
    from myriad.icon import points, svg
    pts = points()
    assert len(pts) == 89 and all(0 < x < 1 and 0 < y < 1 for x, y, _, _ in pts)
    s = svg(256)
    assert s.startswith("<svg") and s.count("<circle") == 89


async def test_change_tracker(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    rt = FakeRuntime(home)
    app = make_ui_app(config=Config(), home=home, token="tok", runtime=rt)
    hdr = {"X-Myriad-Token": "tok"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        r = await c.post("/api/tracker", json={"url": "https://other.example.org"}, headers=hdr)
        assert r.status_code == 409  # nothing configured yet: the wizard sets the tracker
        Config().save(home)
        assert Config.load(home).tracker_url == "https://myriad.french-web.com"  # the public default
        r = await c.post("/api/tracker", json={"url": "file:///etc/passwd"}, headers=hdr)
        assert r.status_code == 400
        r = await c.post("/api/tracker", json={"url": "https://other.example.org/"}, headers=hdr)
        assert r.status_code == 200 and r.json()["tracker_url"] == "https://other.example.org"
    assert Config.load(home).tracker_url == "https://other.example.org" and rt.restarts == 1
