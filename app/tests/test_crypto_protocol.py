import json

import pytest
from pydantic import ValidationError

from myriad.crypto import Identity, canonical_json, node_id_from_pubkey, pubkey_matches, verify
from myriad.protocol import (MAX_FRAME_BYTES, Hello, Job, JobFrame, JobResult, NodeInfo, Receipt, ResultFrame,
                             dump_frame, parse_frame)


def make_job(ident: Identity, **kw) -> Job:
    base = dict(job_id="ab" * 16, requester_id=ident.node_id, messages=[{"role": "user", "content": "2+2?"}],
                max_tokens=64, temperature=0, seed=1, deadline_ms=5000, task_hint="math")
    base.update(kw)
    return Job(**base).signed_by(ident)


# ---------- crypto ----------
def test_canonical_json_is_sorted_compact_utf8():
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode("utf-8")
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sign_verify_and_tamper():
    a, b = Identity.generate(), Identity.generate()
    sig = a.sign("result", {"x": 1})
    assert verify(a.pubkey, "result", {"x": 1}, sig)
    assert not verify(a.pubkey, "result", {"x": 2}, sig)  # payload changed
    assert not verify(a.pubkey, "receipt", {"x": 1}, sig)  # other kind: no cross-replay
    assert not verify(b.pubkey, "result", {"x": 1}, sig)  # other key
    assert not verify("zz", "result", {"x": 1}, sig)  # garbage key
    assert not verify(a.pubkey, "result", {"x": 1}, "00")  # garbage signature


def test_node_id_is_hash_of_pubkey(tmp_path):
    a = Identity.generate()
    assert a.node_id == node_id_from_pubkey(a.pubkey) and len(a.node_id) == 32
    assert pubkey_matches(a.node_id, a.pubkey)
    assert not pubkey_matches(a.node_id, Identity.generate().pubkey)
    p = tmp_path / "k.pem"
    a.save(p)
    assert Identity.load(p).node_id == a.node_id
    with pytest.raises(FileExistsError):  # never overwrites a key
        Identity.generate().save(p)
    assert Identity.load_or_create(p).node_id == a.node_id


# ---------- protocol ----------
def test_job_signature_survives_the_wire():
    a = Identity.generate()
    job = make_job(a)
    frame = parse_frame(dump_frame(JobFrame(job=job, target="cd" * 16)))
    assert isinstance(frame, JobFrame)
    assert frame.job.verify(a.pubkey)
    assert frame.job.temperature == 0.0


def test_tampered_job_fails():
    a = Identity.generate()
    job = make_job(a)
    evil = job.model_copy(update={"max_tokens": 2000})
    assert not evil.verify(a.pubkey)
    assert not make_job(a).model_copy(update={"signature": ""}).verify(a.pubkey)


def test_result_and_receipt_signatures():
    a = Identity.generate()
    res = JobResult(job_id="ab" * 16, node_id=a.node_id, model="m", text="The answer is 4.", completion_tokens=5,
                    mean_logprob=-0.25, compute_ms=12.5).signed_by(a)
    back = parse_frame(dump_frame(ResultFrame(result=res))).result
    assert back.verify(a.pubkey)
    rc = Receipt(job_id="ab" * 16, requester_id=a.node_id, node_id="cd" * 16, completion_tokens=5, model="m",
                 agreed=True).signed_by(a)
    assert rc.verify(a.pubkey)
    # a receipt signature is not a valid result signature
    assert not res.model_copy(update={"signature": rc.signature}).verify(a.pubkey)


def test_hello_signature():
    a = Identity.generate()
    info = NodeInfo(node_id=a.node_id, pubkey=a.pubkey, model="Qwen/Qwen3-1.7B-GGUF", accepting=True)
    h = parse_frame(dump_frame(Hello.make(a, info, "11" * 32)))
    assert h.valid()
    h2 = h.model_copy(update={"nonce": "22" * 32})
    assert not h2.valid()


def test_limits_are_enforced():
    a = Identity.generate()
    with pytest.raises(ValidationError):
        make_job(a, max_tokens=4096)
    with pytest.raises(ValidationError):
        make_job(a, messages=[{"role": "user", "content": "x" * 40_000}])
    with pytest.raises(ValidationError):
        make_job(a, messages=[{"role": "tool", "content": "x"}])
    with pytest.raises(ValidationError):
        make_job(a, deadline_ms=10**9)
    with pytest.raises(ValidationError):
        make_job(a, job_id="../etc/passwd")
    with pytest.raises(ValidationError):
        parse_frame(json.dumps({"t": "job", "job": {}, "extra": 1}))
    with pytest.raises(ValueError):
        parse_frame("x" * (MAX_FRAME_BYTES + 1))
    with pytest.raises(ValidationError):
        parse_frame(json.dumps({"t": "unknown"}))
    with pytest.raises(ValidationError):
        JobResult(job_id="ab" * 16, node_id=a.node_id, model="m", text="t", mean_logprob=float("inf"))
