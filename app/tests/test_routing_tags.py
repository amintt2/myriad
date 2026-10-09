"""essaim/1.2 skill tags: tag indexes in the tracker, routing names on the gateway (myriad,
myriad:<family>, myriad:<tag>, fallback), /v1/models with live counts, backward compatibility."""
from __future__ import annotations

import functools
import json

import pytest

import myriad.tracker as tracker_mod
from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.gateway import GatewayError
from myriad.node import NodeClient
from myriad.protocol import (Assigned, Hello, Job, JobFrame, NodeInfo, PeerCard, Route, dump_frame, parse_frame)
from myriad.routing import RouteError, live_routes, model_tags, node_tags, normalize_tags, parse_model
from myriad.tracker import Tracker

from .conftest import MODELS, start_swarm, wait_until
from .test_scale_v11 import MATH, fake_conn, sent_frames, signed_job

TEXT = {"role": "user", "content": "Write a haiku about the sea."}


def tagged(tags):
    return functools.partial(NodeClient, tags=tags)


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0.0, seed=5)
    yield t
    t.ledger.close()


def tag_conn(t, fam, tags, **kw):
    c = fake_conn(t, None, fam, **kw)  # registered without a model first: no slot
    c.info = c.info.model_copy(update={"model": MODELS[fam], "tags": tags})
    t._reindex(c)
    return c


# ============================================================ names and tags
def test_parse_model_names():
    assert parse_model("myriad").kind == "swarm" and parse_model("essaim").kind == "swarm"
    assert parse_model(None).kind == "swarm"
    s = parse_model("myriad:gemma")
    assert (s.kind, s.value) == ("family", "gemma")
    s = parse_model("myriad:Python")
    assert (s.kind, s.value) == ("tag", "python")
    assert parse_model("myriad:tag=gemma").kind == "tag" and parse_model("myriad:family=newfam").kind == "family"
    assert parse_model("myriad:olmo", families={"olmo"}).kind == "family"
    assert parse_model("Qwen/Qwen3-1.7B-GGUF").kind == "model"
    with pytest.raises(RouteError):
        parse_model("myriad:not a tag!")


def test_tags_normalized_and_from_model():
    assert normalize_tags([" Python", "python", "type script", "bad tag!", 3, "c++"]) == ["python", "type-script", "c++"]
    assert normalize_tags([f"t{i}" for i in range(40)]) == [f"t{i}" for i in range(16)]
    assert "code" in model_tags("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
    assert "orchestrator" in model_tags("unsloth/Qwen3.5-4B-GGUF") and model_tags("Qwen/Qwen3-1.7B-GGUF") == []
    assert node_tags(None, ["python"]) is None  # a client-only node advertises nothing
    assert node_tags("Qwen/Qwen3-1.7B-GGUF", ["Python"]) == ["python"]


def test_live_routes_counts():
    peers = [{"node_id": "a", "model": MODELS["qwen"], "family": "qwen", "accepting": True, "busy": 0, "max_parallel": 1,
              "tags": ["python"]},
             {"node_id": "b", "model": MODELS["qwen"], "family": "qwen", "accepting": True, "busy": 1, "max_parallel": 1,
              "tags": ["python", "review"]},
             {"node_id": "c", "model": MODELS["gemma"], "family": "gemma", "accepting": False, "tags": None}]
    r = live_routes(peers)
    assert r["myriad"] == {"kind": "swarm", "value": None, "peers": 3, "available": 1}
    assert r["myriad:python"]["peers"] == 2 and r["myriad:python"]["available"] == 1
    assert r["myriad:gemma"]["kind"] == "family" and r["myriad:gemma"]["available"] == 0
    assert r["myriad:review"]["peers"] == 1 and r[MODELS["qwen"]]["kind"] == "model"


# ============================================================ tracker indexes
def test_tracker_tag_index_follows_state(tracker):
    t = tracker
    py_q = tag_conn(t, "qwen", ["python"])
    py_s = tag_conn(t, "smollm", ["python", "review"])
    tag_conn(t, "gemma", None)
    assert [c.family for c in t.select(3, tag="python")] == ["smollm", "qwen"]  # most reliable first
    assert [c.family for c in t.select(1, tag="review")] == ["smollm"]
    assert t.select(1, tag="rust") == []
    assert len(t.select(4)) == 3  # untagged selection unchanged
    py_s.busy = 1
    t._reindex(py_s)
    assert [c.family for c in t.select(3, tag="python")] == ["qwen"] and t.select(1, tag="review") == []
    t._suspend(py_q, "test")
    assert t.select(1, tag="python") == [] and "python" not in t._tag_pools
    t._readmit(py_q)
    assert t.select(1, tag="python") == [py_q]
    t._drop(py_q)
    assert t.select(1, tag="python") == []
    assert t.select(1, tag="python", model=MODELS["qwen"]) == []


def test_tracker_routes_by_tag_with_fallback_and_family(tracker):
    t = tracker
    py = tag_conn(t, "qwen", ["python"], max_parallel=8)
    other = tag_conn(t, "smollm", None, max_parallel=4)
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    for tag, want, match in (("python", py, True), ("rust", other, False)):
        job = signed_job(req.identity)
        t._submit(req, JobFrame(job=job, route=Route(group=job.job_id, tag=tag)))
        a = next(f for f in sent_frames(req) if isinstance(f, Assigned))
        assert a.peer.node_id == want.node_id and a.tag_match is match and a.peer.tags is not None
    job = signed_job(req.identity)
    t._submit(req, JobFrame(job=job, route=Route(group=job.job_id, family="qwen")))
    a = next(f for f in sent_frames(req) if isinstance(f, Assigned))
    assert a.peer.node_id == py.node_id and a.tag_match is None and '"tag_match"' not in dump_frame(a)
    job = signed_job(req.identity)
    t._submit(req, JobFrame(job=job, route=Route(group=job.job_id, family="gemma")))  # strict: no fallback
    err = sent_frames(req)
    assert err and err[-1].t == "job_error" and err[-1].error == "no_peer"


# ============================================================ wire compatibility
def test_new_fields_left_out_when_unset():
    ident = Identity.generate()
    info = NodeInfo(node_id=ident.node_id, pubkey=ident.pubkey, model=MODELS["qwen"])
    h = Hello.make(ident, info, "ab" * 32)
    assert '"tags"' not in dump_frame(h) and parse_frame(dump_frame(h)).valid()
    old_style = json.loads(dump_frame(h))  # what an essaim/1.1 node sends: no tags key at all
    assert Hello.model_validate(old_style).valid()
    h2 = Hello.make(ident, info.model_copy(update={"tags": ["python"]}), "cd" * 32)
    assert '"tags":["python"]' in dump_frame(h2) and parse_frame(dump_frame(h2)).valid()
    job = Job(job_id="1" * 32, requester_id=ident.node_id, messages=[MATH]).signed_by(ident)
    plain = dump_frame(JobFrame(job=job, route=Route(group="2" * 32)))
    assert '"tag"' not in plain and '"family"' not in plain
    assert '"tag":"python"' in dump_frame(JobFrame(job=job, route=Route(group="2" * 32, tag="python")))
    card = PeerCard(node_id=ident.node_id, pubkey=ident.pubkey, model="m", reliability=0.5)
    assert '"tags"' not in dump_frame(Assigned(job_id="1" * 32, peer=card))


async def test_node_retries_without_tags_on_an_older_tracker(tmp_path, monkeypatch):
    s = await start_swarm(tmp_path)
    try:
        orig = s.tracker._check_hello

        def older(hello, nonce):  # an essaim/1.1 tracker: a Hello with an unknown field does not parse
            if hello is not None and hello.info.tags is not None:
                return "hello_expected"
            return orig(hello, nonce)

        monkeypatch.setattr(s.tracker, "_check_hello", older)
        n = await s.add_node("qwen", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=tagged(["python"]))
        assert n._tags_refused and n.info().tags is None
        assert s.tracker.conns[n.node_id].info.tags is None
    finally:
        await s.close()


# ============================================================ gateway, end to end
@pytest.fixture
async def tswarm(tmp_path):
    s = await start_swarm(tmp_path)
    await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=5), MODELS["qwen"],
                     cls=tagged(["python", "review"]))
    await s.add_node("smollm", FakeEngine(MODELS["smollm"], "The answer is 42.", tokens=5), MODELS["smollm"],
                     cls=tagged(["typescript"]))
    await s.add_node("gemma", FakeEngine(MODELS["gemma"], "The answer is 42.", tokens=5), MODELS["gemma"])
    try:
        yield s
    finally:
        await s.close()


@pytest.mark.parametrize("routing", ["tracker", "directory"])
async def test_gateway_routes_by_tag_family_and_falls_back(tswarm, routing):
    gw, client = await tswarm.add_gateway(routing=routing)
    for _ in range(3):
        ans = await gw.ask([TEXT], tag="python", k=1)
        assert [p["family"] for p in ans.meta["peers"]] == ["qwen"]
        assert ans.meta["route"] == {"tag": "python", "family": None, "fallback": False}
        assert ans.candidates[0]["tag_match"] is True and ans.candidates[0]["text"] == "The answer is 42."
    ans = await gw.ask([TEXT], tag="rust", k=1)  # nobody has it: any peer, flagged
    assert ans.meta["route"]["fallback"] is True and ans.meta["peers"][0]["tag_match"] is False
    ans = await gw.ask([TEXT], tag="python", k=2)  # one tagged peer, the second one is a fallback
    assert sorted(str(p["tag_match"]) for p in ans.meta["peers"]) == ["False", "True"]
    assert ans.meta["route"]["fallback"] is True
    for _ in range(3):
        ans = await gw.ask([TEXT], family="gemma")
        assert ans.meta["peers"][0]["family"] == "gemma" and ans.meta["decision"] == "single"
    with pytest.raises(GatewayError) as e:
        await gw.ask([TEXT], family="granite")
    assert e.value.status == 503
    ans = await gw.ask([TEXT], k=3, avoid=[tswarm.nodes["qwen"].node_id])
    assert "qwen" not in {p["family"] for p in ans.meta["peers"]}


async def test_openai_names_and_models_listing(tswarm):
    gw, client = await tswarm.add_gateway()
    r = await client.post("/v1/chat/completions", json={"model": "myriad:python", "messages": [TEXT]})
    j = r.json()
    assert r.status_code == 200 and j["model"] == "myriad:python"
    assert j["myriad"]["route"]["kind"] == "tag" and j["myriad"]["route"]["fallback"] is False
    assert j["myriad"]["peers"][0]["family"] == "qwen"
    r = await client.post("/v1/chat/completions", json={"model": "myriad:gemma", "messages": [TEXT]})
    assert r.json()["myriad"]["route"]["kind"] == "family" and r.json()["myriad"]["peers"][0]["family"] == "gemma"
    r = await client.post("/v1/chat/completions", json={"model": "myriad:golang", "messages": [TEXT]})
    assert r.status_code == 200 and r.json()["myriad"]["route"]["fallback"] is True
    r = await client.post("/v1/chat/completions", json={"model": "myriad", "messages": [MATH], "myriad": {"k": 3}})
    assert r.json()["myriad"]["route"]["kind"] == "swarm" and r.json()["myriad"]["peers_asked"] == 3
    r = await client.post("/v1/chat/completions", json={"model": "myriad:bad!name", "messages": [TEXT]})
    assert r.status_code == 400
    data = (await client.get("/v1/models")).json()["data"]
    ids = [m["id"] for m in data]
    assert ids[:2] == ["myriad", "essaim"] and MODELS["qwen"] in ids
    by = {m["id"]: m["myriad"] for m in data}
    assert by["myriad"]["peers"] == 3
    assert by["myriad:python"] == {"kind": "tag", "value": "python", "peers": 1, "available": 1}
    assert by["myriad:qwen"]["kind"] == "family" and "myriad:typescript" in by and "myriad:review" in by


async def test_gateway_uses_directory_for_tags_with_an_older_tracker(tswarm, monkeypatch):
    monkeypatch.setattr(tracker_mod, "FEATURES", ("route", "ping", "select"))
    from myriad.security import Security
    strict, _ = await tswarm.add_gateway("strict")
    with pytest.raises(GatewayError) as e:  # an essaim/1.1 tracker cannot relay encrypted jobs: nothing sent
        await strict.ask([TEXT], tag="typescript")
    assert e.value.code == "e2e_unavailable"
    gw, client = await tswarm.add_gateway(security=Security({"require_e2e": False}))  # plaintext allowed
    ans = await gw.ask([TEXT], tag="typescript")
    assert ans.meta["routing"] == "directory" and ans.meta["peers"][0]["family"] == "smollm"
    ans = await gw.ask([MATH], k=2)
    assert ans.meta["routing"] == "tracker"
    await wait_until(lambda: True)


# ============================================================ audit fixes
def test_unknown_tags_are_not_cached_and_family_prefers_the_tag(tracker):
    t = tracker
    for i in range(50):
        assert t.select(1, tag=f"nobody-{i}") == []
    assert t._tag_order == {}
    plain = tag_conn(t, "qwen", None, max_parallel=4)
    py = tag_conn(t, "qwen", ["python"], max_parallel=8)
    py.busy = 1  # partly busy: the idle untagged peer of the family would win without the preference
    t._reindex(py)
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    for _ in range(5):
        job = signed_job(req.identity)
        t._submit(req, JobFrame(job=job, route=Route(group=job.job_id, family="qwen", tag="python")))
        a = next(f for f in sent_frames(req) if isinstance(f, Assigned))
        assert a.peer.node_id == py.node_id != plain.node_id and a.tag_match is True


def test_tag_spelled_like_a_family_gets_an_explicit_name():
    peers = [{"node_id": "a", "model": MODELS["qwen"], "family": "qwen", "accepting": True, "tags": ["gemma"]}]
    r = live_routes(peers)
    assert "myriad:gemma" not in r and r["myriad:tag=gemma"]["kind"] == "tag"
    assert parse_model("myriad:tag=gemma").kind == "tag"


@pytest.mark.parametrize("routing", ["tracker", "directory"])
async def test_replacement_keeps_the_callers_avoid_list(tmp_path, routing):
    from .test_scale_v11 import RefusingNode
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("r", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=RefusingNode)
        x = await s.add_node("x", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        gw, _ = await s.add_gateway(routing=routing)
        with pytest.raises(GatewayError):
            await gw.ask([TEXT], family="qwen", avoid=[x.node_id])
        assert x.engine.calls == 0  # the refusal was not replaced by the avoided node
    finally:
        await s.close()
