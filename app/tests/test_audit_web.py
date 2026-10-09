"""Regression tests of the interface scripts (myriad/web/*.js) run under Node with a minimal fake DOM
(tests/js/harness.cjs): network map with hostile family names, chat stream cut before its end,
answers rejected by the gateway, the wizard's active hours. Skipped when Node is not installed."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "harness.cjs"
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js absent")

APP_SCRIPTS = '["i18n.js", "netviz.js", "agents.js", "app.js"]'
STATUS = {"app": "Myriad", "version": "0", "runtime": {"state": "en marche", "configured": True}, "node": None,
          "balance": None, "history": [], "gateway": None, "default_k": 4, "setup_job": None, "update": None}


def run_js(tmp_path: Path, body: str) -> dict:
    """Run a scenario (async JS body with `H` = the harness module) and return what it prints as JSON."""
    script = tmp_path / "scenario.cjs"
    script.write_text(
        f"const H = require({json.dumps(str(HARNESS))});\n"
        f"const STATUS = {json.dumps(STATUS)};\n"
        "(async () => {\n" + body + "\n})().then((r) => { process.stdout.write(JSON.stringify(r)); process.exit(0); },"
        " (e) => { process.stdout.write(JSON.stringify({error: String(e && e.stack || e)})); process.exit(0); });\n",
        encoding="utf-8")
    p = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert "error" not in out, out["error"]
    return out


def test_netviz_survives_inherited_property_names_as_families(tmp_path):
    out = run_js(tmp_path, """
      const env = H.load(["netviz.js"]);
      const NV = env.window.NetViz;
      const names = ["constructor", "toString", "__proto__", "hasOwnProperty", "valueOf"];
      const colors = names.map((f) => NV.familyColor(f));
      const viz = NV.create(env.document.createElement("canvas"), {});
      viz.setPeers(names.map((f, i) => ({ node_id: "n" + i, family: f, model: "m", accepting: true, busy: 1,
                                            max_parallel: 2 })), "me");
      viz.pulse("me", "n0");
      let drawn = 0, err = null;
      for (let i = 0; i < 3; i++) { try { env.flushFrames(); drawn++; } catch (e) { err = String(e); break; } }
      return { colors, drawn, err };
    """)
    assert all(isinstance(c, str) and c.startswith("hsl(") for c in out["colors"]), out
    assert out["err"] is None and out["drawn"] == 3, out


def _chat_scenario(events: list[dict]) -> str:
    return f"""
      let stream = null;
      const env = H.load({APP_SCRIPTS}, {{ fetch: async (url) => {{
        if (url === "/api/status") return H.json(STATUS);
        if (url === "/api/chat/stream") return H.sseResponse({json.dumps(events)});
        return H.json({{}}, 404);
      }} }});
      await env.tick(10);
      const $ = env.$;
      $("question").value = "What is 6 x 7?";
      await Promise.all($("chat").fire("submit", {{ preventDefault() {{}} }}));
      await env.tick(30);
      const turn = $("thread").children[$("thread").children.length - 1];
      const els = turn.all();
      const err = els.find((e) => e.className === "error");
      const text = els.find((e) => (e.className || "").startsWith("a-text"));
      const peers = els.filter((e) => (e.className || "").split(" ").includes("peer"));
      return {{ errHidden: err.hidden, errText: err.textContent, text: text.textContent, waiting: text.className.includes("wait"),
                sendDisabled: $("send").disabled, peers: peers.map((p) => p.textContent) }};
    """


def test_chat_stream_closed_without_final_event_is_reported(tmp_path):
    events = [{"type": "start", "task_hint": "math", "k": 2},
              {"type": "asked", "job_id": "j1", "node_id": "a" * 32, "model": "org/M-GGUF", "family": "qwen", "ms": 1}]
    out = run_js(tmp_path, _chat_scenario(events))
    assert out["errHidden"] is False and out["errText"], out  # the interruption is shown
    assert out["waiting"] is False and "réfléchit" not in out["text"], out
    assert out["sendDisabled"] is False
    assert all("réfléchit" not in p for p in out["peers"]), out  # no peer is left "thinking"


def test_chat_final_drops_answers_the_gateway_rejected(tmp_path):
    node = "b" * 32
    events = [
        {"type": "start", "task_hint": "math", "k": 1},
        {"type": "asked", "job_id": "j1", "node_id": node, "model": "org/M-GGUF", "family": "qwen", "ms": 1},
        {"type": "answered", "job_id": "j1", "node_id": node, "model": "org/M-GGUF", "answer": "666",
         "text": "FORGED TEXT", "tokens": 3, "ms": 5},
        {"type": "final", "ms": 9, "body": {"choices": [{"message": {"content": "no answer"}}], "myriad": {
            "answer": None, "decision": "none", "peers_answered": 0, "peers_asked": 1, "latency_ms": 9,
            "task_hint": "math", "certificate": False, "early_stop": False,
            "peers": [{"node_id": node, "model": "org/M-GGUF", "family": "qwen", "weight": 0.5, "chosen": False,
                       "status": "erreur", "answer": None, "error": "bad_signature", "latency_ms": 5,
                       "completion_tokens": 0}]}}},
    ]
    out = run_js(tmp_path, _chat_scenario(events))
    assert out["peers"] and all("666" not in p and "FORGED" not in p for p in out["peers"]), out


def test_wizard_reflects_active_hours_turned_off(tmp_path):
    out = run_js(tmp_path, """
      let hours = "8-23";
      const setup = () => ({ defaults: { tracker_url: "https://t.example", max_parallel: 1, accepting: true,
                                         active_hours: hours, model: "m", quant: "Q4_K_M" },
                             recommendation: { model: "m", quant: "Q4_K_M" }, catalog: [], hardware: null,
                             engine: { existing: null, backends: [], default_backend: "cpu", build: "b" },
                             network_families: {}, job: null, configured: true });
      const env = H.load(["i18n.js", "netviz.js", "agents.js", "app.js"], { fetch: async (url) => {
        if (url === "/api/status") return H.json(STATUS);
        if (url.startsWith("/api/setup")) return H.json(setup());
        return H.json({}, 404);
      } });
      const $ = env.$;
      await env.tick(10);
      $("change-model").fire("click");
      await env.tick(20);
      const first = { checked: $("wz-sched").checked, hidden: $("wz-hours").hidden };
      $("wz-close").fire("click");
      hours = null;  // turned off from the dashboard meanwhile
      $("change-model").fire("click");
      await env.tick(20);
      return { first, second: { checked: $("wz-sched").checked, hidden: $("wz-hours").hidden } };
    """)
    assert out["first"] == {"checked": True, "hidden": False}, out
    assert out["second"] == {"checked": False, "hidden": True}, out
