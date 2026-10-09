"""End to end, without GPU or network: real tracker, 4 nodes with scripted engines, the gateway."""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from websockets.asyncio.client import connect

from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.node import NodeClient, ws_url
from myriad.priors import credit_factor, params_of
from myriad.protocol import (Challenge, ErrorFrame, Hello, Job, JobError, JobResult, NodeInfo, Receipt, ResultFrame,
                             dump_frame, parse_frame)

from .conftest import MODELS, start_swarm, wait_until

MATH = {"role": "user", "content": "Tom has 40 apples and buys 2 more. How many apples does he have?"}


def ledger_kinds(tracker, kind):
    return [e for e in tracker.ledger.recent(500) if e["kind"] == kind]


async def test_end_to_end_vote_early_stop_and_credits(swarm):
    gw, client = await swarm.add_gateway()
    requester = swarm.nodes["client"]
    t0 = time.perf_counter()
    r = await client.post("/v1/chat/completions", json={"model": "myriad", "messages": [MATH]})
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    meta = body["myriad"]
    assert body["essaim"] == meta  # deprecated alias, same content
    assert body["object"] == "chat.completion"
    assert "The answer is 42" in body["choices"][0]["message"]["content"]
    assert meta["decision"] == "vote" and meta["answer"] == "42" and meta["task_hint"] == "math"
    # The certificate fires before the slow (and wrong) gemma node answers.
    assert meta["early_stop"] and meta["certificate"]
    assert elapsed < 3.0, elapsed
    assert meta["peers_asked"] == 4 and meta["peers_answered"] == 2
    status = {p["model"]: p["status"] for p in meta["peers"]}
    assert status[MODELS["qwen"]] == status[MODELS["smollm"]] == "ok" and status[MODELS["gemma"]] == "annulé"
    # The failing node either failed before the decision or was cancelled with the straggler.
    assert status[MODELS["granite"]] in ("erreur", "annulé")
    # The straggler is cancelled on its node.
    await wait_until(lambda: swarm.engines["gemma"].cancelled == 1, 5)
    # Receipts settle the credits: requester pays, the two answering nodes earn, the others nothing.
    tr = swarm.tracker
    await wait_until(lambda: len(ledger_kinds(tr, "receipt")) == 2, 5)
    bal = tr.ledger.balances()
    f_qwen = 10 * credit_factor(params_of(MODELS["qwen"]))
    f_smol = 10 * credit_factor(params_of(MODELS["smollm"]))
    assert bal[swarm.nodes["qwen"].node_id] == pytest.approx(1000 + f_qwen)
    assert bal[swarm.nodes["smollm"].node_id] == pytest.approx(1000 + f_smol)
    assert bal[swarm.nodes["gemma"].node_id] == pytest.approx(1000)
    assert bal[swarm.nodes["granite"].node_id] == pytest.approx(1000)
    assert bal[requester.node_id] == pytest.approx(1000 - f_qwen - f_smol)
    # Agreement with the certified decision feeds the reliability of each model.
    stats = tr.ledger.model_stats()
    assert stats[MODELS["qwen"]] == (1, 0) and stats[MODELS["smollm"]] == (1, 0)
    # Signed evidence is kept for every settlement.
    ev = tr.ledger.evidence(ledger_kinds(tr, "receipt")[0]["job_id"])
    assert ev["receipt"]["signature"] and ev["result"]["signature"]


async def test_free_text_medoid_and_streaming(swarm):
    swarm.engines["gemma"].delay_s = 0.0
    gw, client = await swarm.add_gateway()
    q = {"role": "user", "content": "What is the capital of France?"}
    r = await client.post("/v1/chat/completions", json={"messages": [q], "stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    meta = chunks[-1]["myriad"]
    assert chunks[-1]["essaim"] == meta  # deprecated alias, same content
    assert meta["decision"] == "medoid" and meta["task_hint"] == "free" and not meta["early_stop"]
    assert meta["peers_answered"] == 3  # granite fails, the others all answered
    failed = [p for p in meta["peers"] if p["status"] == "erreur"]
    assert [p["model"] for p in failed] == [MODELS["granite"]] and "panne simulée" in failed[0]["error"]
    assert "Paris" in text


async def test_paused_node_is_not_asked(swarm):
    gw, client = await swarm.add_gateway()
    smol = swarm.nodes["smollm"]
    await smol.set_limits(accepting=False)
    await wait_until(lambda: not swarm.tracker.conns[smol.node_id].info.accepting)
    r = await client.post("/v1/chat/completions", json={"messages": [MATH], "myriad": {"k": 4}})
    meta = r.json()["myriad"]
    assert MODELS["smollm"] not in {p["model"] for p in meta["peers"]}
    await smol.set_limits(accepting=True)
    await wait_until(lambda: swarm.tracker.conns[smol.node_id].info.accepting)


async def test_negative_balance_blocks_requests(swarm):
    gw, client = await swarm.add_gateway()
    swarm.tracker.ledger.db.execute("UPDATE accounts SET balance=-1 WHERE node_id=?", (swarm.nodes["client"].node_id,))
    r = await client.post("/v1/chat/completions", json={"messages": [MATH]})
    assert r.status_code == 402
    assert r.json()["error"]["type"] == "insufficient_credits"
    # A failed request leaves no job registration behind.
    await wait_until(lambda: not swarm.nodes["client"].waiters)


async def test_invalid_hello_signature_is_rejected(swarm):
    evil, other = Identity.generate(), Identity.generate()
    async with connect(ws_url(swarm.url)) as ws:
        ch = parse_frame(await ws.recv())
        assert isinstance(ch, Challenge)
        info = NodeInfo(node_id=evil.node_id, pubkey=evil.pubkey, model="x/y", accepting=True)
        forged = Hello.make(other, info, ch.nonce)  # signed with a key that is not the node's
        await ws.send(dump_frame(forged))
        err = parse_frame(await ws.recv())
        assert isinstance(err, ErrorFrame) and err.error == "bad_signature"
    assert evil.node_id not in swarm.tracker.conns


async def test_invalid_job_and_receipt_signatures_are_rejected(swarm):
    gw, client = await swarm.add_gateway()
    req = swarm.nodes["client"]
    target = swarm.nodes["qwen"]
    calls = swarm.engines["qwen"].calls
    # A job signed by someone else than its requester.
    forged = Job(job_id="ab" * 16, requester_id=req.node_id, messages=[MATH]).signed_by(Identity.generate())
    q: asyncio.Queue = asyncio.Queue()
    req.register([forged.job_id], q)
    await req.submit(target.node_id, forged)
    ev = await asyncio.wait_for(q.get(), 5)
    assert isinstance(ev, JobError) and ev.error == "bad_signature"
    assert swarm.engines["qwen"].calls == calls  # never reached the node
    # A valid job, then a receipt signed by the wrong key: no settlement.
    job = Job(job_id="cd" * 16, requester_id=req.node_id, messages=[MATH]).signed_by(req.identity)
    req.register([job.job_id], q)
    await req.submit(target.node_id, job)
    ev = await asyncio.wait_for(q.get(), 5)
    assert isinstance(ev, ResultFrame) and ev.result.verify(target.identity.pubkey)
    rc = Receipt(job_id=job.job_id, requester_id=req.node_id, node_id=target.node_id,
                 completion_tokens=ev.result.completion_tokens, model=ev.result.model).signed_by(Identity.generate())
    from myriad.protocol import ReceiptFrame
    await req.send(ReceiptFrame(receipt=rc))
    await wait_until(lambda: req.last_error and "bad_receipt: bad_signature" in req.last_error)
    assert not ledger_kinds(swarm.tracker, "receipt")
    # The right receipt is accepted, once.
    await req.send_receipt(rc.model_copy(update={"signature": ""}))
    await wait_until(lambda: len(ledger_kinds(swarm.tracker, "receipt")) == 1)
    await req.send_receipt(rc.model_copy(update={"signature": ""}))
    await wait_until(lambda: "already_settled" in (req.last_error or ""))
    assert len(ledger_kinds(swarm.tracker, "receipt")) == 1


class ForgingNode(NodeClient):
    """A node that signs its results with a key that is not its own."""

    async def _execute(self, job):
        res = JobResult(job_id=job.job_id, node_id=self.node_id, model=self.model, text="The answer is 1.",
                        completion_tokens=2000).signed_by(Identity.generate())
        await self.send(ResultFrame(result=res))
        self.running.pop(job.job_id, None)


async def test_forged_result_is_rejected(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("forger", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=ForgingNode)
        gw, client = await s.add_gateway()
        r = await client.post("/v1/chat/completions", json={"messages": [MATH]})
        assert r.status_code == 502
        assert "bad_result_signature" in r.json()["error"]["message"]
        assert s.tracker.ledger.balances()[s.nodes["forger"].node_id] == pytest.approx(1000)
    finally:
        await s.close()


async def test_spot_check_duplicates_greedy_jobs(tmp_path):
    s = await start_swarm(tmp_path, spot_rate=1.0, seed=1)
    try:
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        b = await s.add_node("b", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        gw, client = await s.add_gateway()
        r = await client.post("/v1/chat/completions",
                              json={"model": MODELS["qwen"], "messages": [MATH], "temperature": 0})
        assert r.status_code == 200 and r.json()["myriad"]["decision"] == "single"
        await wait_until(lambda: len(ledger_kinds(s.tracker, "spot")) == 1, 5)
        st_a, st_b = s.tracker.ledger.node_stats(a.node_id), s.tracker.ledger.node_stats(b.node_id)
        assert st_a["spot_agree"] == 1 and st_b["spot_agree"] == 1
        assert s.tracker.ledger.model_spot_stats()[MODELS["qwen"]] == (1, 0)
        assert s.tracker.ledger.node_spot(a.node_id, MODELS["qwen"]) == (1, 0)
        assert s.tracker.reliability()[MODELS["qwen"]]["spot_checks"] == 1
        # Free text is never spot-checked (no extracted answer to compare).
        await client.post("/v1/chat/completions", json={"model": MODELS["qwen"], "temperature": 0,
                                                        "messages": [{"role": "user", "content": "Say hello."}]})
        await asyncio.sleep(0.3)
        assert len(ledger_kinds(s.tracker, "spot")) == 1
        peers = (await gw.http.get("/v1/peers")).json()["peers"]
        assert all(p["reputation"] > 0.5 for p in peers)
    finally:
        await s.close()


async def test_gateway_refuses_non_local_and_cross_site(swarm):
    gw, client = await swarm.add_gateway()
    body = {"messages": [MATH]}
    r = await client.post("/v1/chat/completions", json=body, headers={"Host": "evil.example"})
    assert r.status_code == 403
    r = await client.post("/v1/chat/completions", json=body, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = await client.post("/v1/chat/completions", content=json.dumps(body), headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    r = await client.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert ids[:2] == ["myriad", "essaim"] and MODELS["qwen"] in ids  # essaim: deprecated alias
    for path in ("/v1/myriad/status", "/v1/essaim/status"):  # the second one: deprecated alias
        r = await client.get(path)
        assert r.status_code == 200 and "node" in r.json() and "balance" in r.json()


async def test_node_disconnect_fails_its_jobs(swarm):
    gw, client = await swarm.add_gateway()
    swarm.engines["gemma"].fail = False
    task = asyncio.create_task(client.post("/v1/chat/completions",
                                           json={"model": MODELS["gemma"], "messages": [MATH]}))
    await wait_until(lambda: swarm.engines["gemma"].calls == 1)
    await swarm.nodes["gemma"].stop()
    r = await task
    assert r.status_code == 502
    assert "peer_disconnected" in r.json()["error"]["message"]


def test_reputation_penalises_only_well_above_baseline():
    from myriad.tracker import reputation, spot_baseline
    alpha = spot_baseline(90, 10)  # honest runs disagree ~10 % of the time
    assert reputation(0, 0, alpha) == 1.0
    assert reputation(18, 2, alpha) == 1.0  # at the baseline: no penalty
    assert reputation(15, 5, alpha) == 1.0  # a bit above, within noise
    assert reputation(2, 18, alpha) < 0.3  # a cheater who disagrees almost always is excluded
    assert reputation(0, 1, alpha) == 1.0  # one disagreement proves nothing


async def test_cancelled_request_cancels_jobs_and_cleans_up(swarm):
    gw, client = await swarm.add_gateway()
    task = asyncio.create_task(client.post("/v1/chat/completions",
                                           json={"model": MODELS["gemma"], "messages": [MATH]}))
    await wait_until(lambda: swarm.engines["gemma"].calls == 1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await wait_until(lambda: swarm.engines["gemma"].cancelled == 1, 5)  # the job is cancelled on the node
    await wait_until(lambda: not swarm.nodes["client"].waiters, 10)  # and the registration dropped
