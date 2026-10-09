"""Local interface (token, limits, chat) and command line (init, status)."""
from __future__ import annotations

import json

import httpx
import pytest

from myriad.cli import main, parse_model
from myriad.config import Config, within_hours
from myriad.ui import make_ui_app

from .conftest import MODELS, wait_until


async def test_ui_token_limits_and_chat(swarm, tmp_path):
    gw, _ = await swarm.add_gateway()
    node = swarm.nodes["qwen"]
    cfg = Config(tracker_url=swarm.url)
    app = make_ui_app(node, gw, cfg, tmp_path, "secret-token")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        page = await c.get("/")
        assert page.status_code == 200 and 'content="secret-token"' in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert (await c.get("/static/app.js")).status_code == 200
        assert (await c.get("/static/ui.py")).status_code == 404
        st = (await c.get("/api/status")).json()
        assert st["node"]["node_id"] == node.node_id and st["node"]["state"] == "connecté"
        # Without the token, nothing changes.
        r = await c.post("/api/limits", json={"max_parallel": 3})
        assert r.status_code == 403 and node.max_parallel == 2
        hdr = {"X-Myriad-Token": "secret-token"}
        r = await c.post("/api/limits", json={"max_parallel": 3, "accepting": False, "active_hours": "8-23"},
                         headers=hdr)
        assert r.status_code == 200, r.text
        assert node.max_parallel == 3 and not node.accepting and node.active_hours == "8-23"
        await wait_until(lambda: not swarm.tracker.conns[node.node_id].info.accepting)
        saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert saved["max_parallel"] == 3 and saved["accepting"] is False
        r = await c.post("/api/limits", json={"active_hours": "99-3"}, headers=hdr)
        assert r.status_code == 400
        r = await c.post("/api/limits", json={"accepting": True, "active_hours": None}, headers=hdr)
        assert r.status_code == 200 and node.accepting and node.active_hours is None
        peers = (await c.get("/api/peers")).json()
        assert {p["model"] for p in peers["peers"]} == set(MODELS.values())
        r = await c.post("/api/chat", json={"message": "Tom has 40 apples and buys 2 more. How many apples?"},
                         headers=hdr)
        assert r.status_code == 200 and r.json()["myriad"]["answer"] == "42"
        r = await c.post("/api/chat", json={"message": "hi"}, headers={**hdr, "Host": "attacker.example"})
        assert r.status_code == 403


def test_active_hours():
    assert within_hours(None, 3)
    assert within_hours("8-23", 8) and not within_hours("8-23", 23) and not within_hours("8-23", 2)
    assert within_hours("22-6", 23) and within_hours("22-6", 5) and not within_hours("22-6", 12)


def test_parse_model():
    assert parse_model("Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf") == ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf")
    for bad in ("Qwen3-1.7B", "Qwen/Qwen3:model.bin", "a/b:../x.gguf"):
        with pytest.raises(SystemExit):
            parse_model(bad)


def test_cli_init_and_status(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MYRIAD_HOME", raising=False)
    monkeypatch.delenv("ESSAIM_HOME", raising=False)
    home = tmp_path / "home"
    rc = main(["--home", str(home), "init", "--model", "Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf",
               "--tracker", "http://127.0.0.1:9", "--no-download", "--max-parallel", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Identifiant" in out
    cfg = Config.load(home)
    assert cfg.model == "Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf" and cfg.family == "qwen"
    assert cfg.params_b == 1.7 and cfg.max_parallel == 2 and cfg.gguf_path is None
    assert (home / "node_key.pem").exists()
    # A second init keeps the same identity.
    first_key = (home / "node_key.pem").read_bytes()
    main(["--home", str(home), "init", "--no-download"])
    assert (home / "node_key.pem").read_bytes() == first_key
    # Status without a running node: identity and configuration, no crash.
    cfg.gateway_port = 9  # nothing listens there
    cfg.save(home)
    assert main(["--home", str(home), "status"]) == 0
    assert "ne tourne pas" in capsys.readouterr().out


def test_cli_chat_request_works_with_old_and_new_gateways(tmp_path, capsys, monkeypatch):
    """`myriad chat` asks the swarm without naming a model, with its options under `myriad` and
    `essaim`: gateways from before the rename (model id and field `essaim` only) understand it too."""
    from myriad.gateway import chat_args

    sent = []

    class Resp:
        status_code = 200

        def __init__(self, meta_key):
            self.meta_key = meta_key

        def json(self):
            return {"choices": [{"message": {"content": "The answer is 42."}}],
                    self.meta_key: {"decision": "vote", "answer": "42", "peers_answered": 3, "peers_asked": 4,
                                    "latency_ms": 10, "peers": []}}

    for meta_key in ("myriad", "essaim"):  # a new gateway, then an old one (metadata under `essaim` only)
        monkeypatch.setattr(httpx, "post", lambda url, json, timeout, k=meta_key: sent.append(json) or Resp(k))
        assert main(["--home", str(tmp_path), "chat", "6 x 7 ?", "--k", "3", "--hint", "math"]) == 0
        assert "réponse 42, 3/4 pairs" in capsys.readouterr().out
    body = sent[0]
    assert "model" not in body and body["myriad"] == body["essaim"] == {"k": 3, "task_hint": "math"}
    args = chat_args(body)  # the new gateway
    assert args["model"] is None and args["k"] == 3 and args["task_hint"] == "math"
    ext = body.get("essaim") or {}  # what an old gateway reads (essaim/gateway.py before the rename)
    assert body.get("model") is None and ext["k"] == 3 and ext["task_hint"] == "math"
    main(["--home", str(tmp_path), "chat", "q", "--model", "Qwen/Qwen3-1.7B-GGUF"])
    assert sent[-1]["model"] == "Qwen/Qwen3-1.7B-GGUF"
