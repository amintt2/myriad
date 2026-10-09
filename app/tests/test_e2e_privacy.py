"""Unit tests (no network) of the end-to-end encryption (e2e.py), the privacy guard (privacy.py) and the
local protection logic (security.py)."""
from __future__ import annotations

import json
import sys
import types

import pytest

from myriad import privacy
from myriad.crypto import Identity
from myriad.e2e import (BUCKETS, CLOCK_SKEW_S, E2EError, Keyring, ReplayCache, b64d, b64e, kx_problem, make_cert,
                        open_job, open_result, pad, seal_job, seal_result, unpad)
from myriad.protocol import Job, SealedJobFrame, SealedResultFrame, dump_frame, parse_frame
from myriad.security import (Quarantine, Security, ServeGuard, invite_code, parse_invite, swarm_group, swarm_key,
                             swarm_proof)

MSG = [{"role": "user", "content": "Tom has 40 apples and buys 2 more. How many apples does he have?"}]


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def job(requester: Identity, **kw) -> Job:
    base = dict(job_id="ab" * 16, requester_id=requester.node_id, messages=MSG, max_tokens=64, temperature=0.0,
                seed=7, deadline_ms=30_000, task_hint="math")
    base.update(kw)
    return Job(**base)


@pytest.fixture
def pair():
    """A requester, a peer with its keyring, and a sealed job for it."""
    req, peer = Identity.generate(), Identity.generate()
    ring = Keyring(peer)
    sj, session = seal_job(job(req), peer.node_id, ring.cert())
    return req, peer, ring, sj, session


# ---------------------------------------------------------------- encryption
def test_round_trip_job_and_result(pair):
    req, peer, ring, sj, session = pair
    wire = parse_frame(dump_frame(SealedJobFrame(job=sj, requester_pubkey=session.pseudonym.pubkey))).job
    got, psession = open_job(wire, session.pseudonym.pubkey, ring, ReplayCache(), peer.node_id)
    assert [m.model_dump() for m in got.messages] == MSG and got.seed == 7 and got.task_hint == "math"
    assert got.requester_id == session.pseudonym.node_id != req.node_id  # the peer sees a pseudonym only
    assert "apples" not in sj.model_dump_json()  # nothing of the content in clear
    sr = seal_result(peer, psession, "m", "The answer is 42.", "stop", 9, -0.2, 12.5)
    sr = parse_frame(dump_frame(SealedResultFrame(result=sr))).result
    assert "answer is" not in json.dumps(sr.chunks)
    res, attest = open_result(sr, session, peer.pubkey)
    assert res.text == "The answer is 42." and res.completion_tokens == 9 and res.mean_logprob == -0.2
    assert attest  # the peer's signature over the plaintext digest


def test_padding_hides_lengths():
    assert len(pad(b"x")) == len(pad(b"x" * 900)) == 1024
    assert len(pad(b"x" * 1100)) == 2048 and len(pad(b"x", min_bucket=4096)) == 4096
    assert unpad(pad(b"hello")) == b"hello"
    with pytest.raises(E2EError):
        unpad(pad(b"hello")[:-1])  # not a bucket size
    bad = bytearray(pad(b"hello"))
    bad[-1] = 1
    with pytest.raises(E2EError):
        unpad(bytes(bad))  # non-zero padding
    with pytest.raises(E2EError):
        pad(b"x" * BUCKETS[-1])


def test_tampering_is_rejected(pair):
    req, peer, ring, sj, session = pair
    pk = session.pseudonym.pubkey
    ct = bytearray(b64d(sj.ct))
    ct[5] ^= 1
    forged = sj.model_copy(update={"ct": b64e(bytes(ct))}).signed_by(session.pseudonym)  # re-signed: AEAD fails
    with pytest.raises(E2EError) as e:
        open_job(forged, pk, ring, ReplayCache(), peer.node_id)
    assert e.value.code == "decrypt_failed"
    for change in ({"max_tokens": 2048}, {"deadline_ms": 600_000}, {"target": "cd" * 16}):
        with pytest.raises(E2EError) as e:  # a header changed by the relay: the pseudonym's signature fails
            open_job(sj.model_copy(update=change), pk, ring, ReplayCache(), peer.node_id)
        assert e.value.code == "bad_job_signature"
    other = Identity.generate()  # re-signed by someone else with the header changed: the AEAD fails
    resigned = sj.model_copy(update={"requester_id": other.node_id, "max_tokens": 2048}).signed_by(other)
    with pytest.raises(E2EError) as e:
        open_job(resigned, other.pubkey, ring, ReplayCache(), peer.node_id)
    assert e.value.code == "decrypt_failed"
    with pytest.raises(E2EError) as e:
        open_job(sj, pk, ring, ReplayCache(), Identity.generate().node_id)
    assert e.value.code == "wrong_target"


def test_replay_and_expiry_are_rejected(pair):
    req, peer, ring, sj, session = pair
    pk = session.pseudonym.pubkey
    cache = ReplayCache()
    open_job(sj, pk, ring, cache, peer.node_id)
    with pytest.raises(E2EError) as e:
        open_job(sj, pk, ring, cache, peer.node_id)
    assert e.value.code == "replay"
    with pytest.raises(E2EError) as e:  # an old envelope replayed after its expiry
        open_job(sj, pk, ring, ReplayCache(), peer.node_id, now=sj.expires_at + CLOCK_SKEW_S + 1)
    assert e.value.code == "expired"
    full = ReplayCache(max_items=1)  # fail-closed when full
    assert full.add("a" * 32, 10**10, 0) and not full.add("b" * 32, 10**10, 0)


def test_answer_chunks_cannot_be_truncated_reordered_or_spliced(pair):
    req, peer, ring, sj, session = pair
    _, psession = open_job(sj, session.pseudonym.pubkey, ring, ReplayCache(), peer.node_id)
    long_text = "€" * 60_000  # 180 kB of UTF-8: 4 chunks of 64 KiB
    sr = seal_result(peer, psession, "m", long_text, "stop", 5, None, 1.0)
    assert len(sr.chunks) >= 3
    assert open_result(sr, session, peer.pubkey)[0].text == long_text
    cases = {"truncated": sr.chunks[:-1], "reordered": [sr.chunks[1], sr.chunks[0]] + sr.chunks[2:],
             "duplicated": sr.chunks[:1] + sr.chunks}
    for name, chunks in cases.items():
        bad = sr.model_copy(update={"chunks": chunks}).signed_by(peer)  # even re-signed by the peer itself
        with pytest.raises(E2EError):
            open_result(bad, session, peer.pubkey)
    # an answer sealed for ANOTHER job (same peer) does not open under this job's session
    sj2, session2 = seal_job(job(req, job_id="cd" * 16), peer.node_id, ring.cert())
    _, p2 = open_job(sj2, session2.pseudonym.pubkey, ring, ReplayCache(), peer.node_id)
    other = seal_result(peer, p2, "m", "The answer is 7.", "stop", 5, None, 1.0)
    with pytest.raises(E2EError):
        open_result(other.model_copy(update={"job_id": sj.job_id}).signed_by(peer), session, peer.pubkey)
    with pytest.raises(E2EError):  # the envelope signed by someone else (the tracker)
        open_result(sr.model_copy(update={"signature": ""}).signed_by(Identity.generate()), session, peer.pubkey)


def test_key_substitution_by_the_tracker_is_rejected():
    peer, tracker = Identity.generate(), Identity.generate()
    clock = Clock()
    good = Keyring(peer, clock=clock).cert()
    assert kx_problem(peer.node_id, peer.pubkey, good, clock()) is None
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    own = make_cert(tracker, X25519PrivateKey.generate(), int(clock()))  # the tracker's key, its signature
    assert kx_problem(peer.node_id, peer.pubkey, own, clock()) == "kx_node_mismatch"
    relabelled = own.model_copy(update={"node_id": peer.node_id})
    assert kx_problem(peer.node_id, peer.pubkey, relabelled, clock()) == "bad_kx_signature"
    # a different pubkey offered for the same node id does not hash to it
    assert kx_problem(peer.node_id, tracker.pubkey, relabelled, clock()) == "pubkey_mismatch"
    swapped = good.model_copy(update={"kx": own.kx})  # the peer's signature, the tracker's key
    assert kx_problem(peer.node_id, peer.pubkey, swapped, clock()) == "bad_kx_signature"
    assert kx_problem(peer.node_id, peer.pubkey, None) == "no_kx"


def test_expired_key_is_rejected_and_rotation_keeps_the_previous_key():
    peer, req = Identity.generate(), Identity.generate()
    clock = Clock()
    ring = Keyring(peer, clock=clock)
    old = ring.cert()
    sj, session = seal_job(job(req), peer.node_id, old, now=clock())
    clock.t += 24 * 3600 + 1
    assert ring.maybe_rotate() and ring.cert().kx != old.kx
    # a job sealed for the previous key (in flight during the rotation) still opens
    open_job(sj.model_copy(), session.pseudonym.pubkey, ring, ReplayCache(), peer.node_id,
             now=sj.expires_at - 1)
    clock.t = old.expires + 1
    assert kx_problem(peer.node_id, peer.pubkey, old, clock()) == "kx_expired"
    clock.t = old.expires + CLOCK_SKEW_S + 1
    ring.maybe_rotate()
    assert ring.private(old.kx) is None  # dropped once expired
    sj2, s2 = seal_job(job(req, job_id="ef" * 16), peer.node_id, old, now=clock())
    with pytest.raises(E2EError) as e:
        open_job(sj2, s2.pseudonym.pubkey, ring, ReplayCache(), peer.node_id, now=clock())
    assert e.value.code == "unknown_kx"


# ---------------------------------------------------------------- privacy guard
SECRETS = {
    "api_key": ["sk-proj-abcdefghijklmnopqrstuvwx1234", "ghp_" + "A" * 36, "AKIAABCDEFGHIJKLMNOP",
                "hf_" + "b" * 34, "password = Sup3rS3cret!"],
    "private_key": ["-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----"],
    "card": ["4111 1111 1111 1111", "5500-0000-0000-0004"],
    "iban": ["FR76 3000 6000 0112 3456 7890 189", "DE89370400440532013000"],
    "email": ["jean.dupont@example.org"],
    "phone": ["06 12 34 56 78", "+33 6 12 34 56 78", "+1 415 555 0132"],
    "path": [r"C:\Users\jdupont\Documents\a.txt", "/home/alice/projects"],
    "name": ["Je m'appelle Marie Curie", "Dear John Smith"],
}
CLEAN = ["Le 2024-01-15 à 14:30, version 3.12.1, 42 pommes.", "Order 1234 5678 9012 3456 shipped",
         "FR76 3000 6000 0112 3456 7890 100", "Call me maybe", "The Eiffel Tower is in Paris.",
         "function f(x) { return x * 2; }", "ID 12345678901234 and 3.14159"]


@pytest.mark.parametrize("kind", sorted(SECRETS))
def test_detectors_find_each_type(kind):
    for text in SECRETS[kind]:
        spans = privacy.detect(f"Bonjour, voici : {text} merci")
        assert any(s.type == kind for s in spans), (kind, text, spans)


def test_detectors_false_positives():
    for text in CLEAN:
        spans = [s for s in privacy.detect(text) if s.type in ("card", "iban", "api_key", "private_key", "email", "phone")]
        assert spans == [], (text, spans)
    assert privacy.luhn_ok("4111111111111111") and not privacy.luhn_ok("4111111111111112")
    assert privacy.iban_ok("DE89370400440532013000") and not privacy.iban_ok("DE89370400440532013001")


def test_pseudonymisation_round_trip_and_stable_placeholders():
    text = ("Mon email est jean.dupont@example.org, ma clé sk-proj-abcdefghijklmnopqrstuvwx1234 ; "
            "réponds à jean.dupont@example.org. EMAIL_1 est déjà pris.")
    st = privacy.PrivacySettings.from_dict({"modes": {"email": "mask"}})
    a = privacy.analyze([{"role": "user", "content": text}], st)
    out = a.messages[0]["content"]
    assert "jean.dupont" not in out and "sk-proj" not in out
    assert out.count("EMAIL_2") == 2 and "CLE_1" in out and "EMAIL_1 est déjà pris" in out  # stable, no reuse
    assert a.decision == "mask" and a.summary()["masked"] == {"email": 2, "api_key": 1}
    answer = "J'écris à EMAIL_2 avec CLE_1 ; EMAIL_1 reste tel quel."
    back = privacy.restore(answer, a.mapping)
    assert "jean.dupont@example.org" in back and "sk-proj-abcdefghijklmnopqrstuvwx1234" in back and "EMAIL_1" in back
    assert "jean" not in json.dumps(a.summary())  # the summary never echoes a value


def test_modes_confirmation_and_local_rule():
    st = privacy.PrivacySettings()
    a = privacy.analyze([{"role": "user", "content": "écris à bob@example.org"}], st)
    assert a.decision == "ask" and a.confirm_types == ["email"]  # first time unmasked: confirmation
    assert privacy.analyze([{"role": "user", "content": "écris à bob@example.org"}], st, confirm=True).decision == "send"
    st.confirmed = ["email"]
    assert privacy.analyze([{"role": "user", "content": "écris à bob@example.org"}], st).decision == "send"
    local = privacy.PrivacySettings.from_dict({"modes": {"iban": "local"}})
    assert privacy.analyze([{"role": "user", "content": "IBAN DE89370400440532013000"}], local).decision == "local"
    terms = privacy.PrivacySettings.from_dict({"terms": ["Projet Hermès"]})
    a = privacy.analyze([{"role": "user", "content": "Le projet hermès avance"}], terms)
    assert "hermès" not in a.messages[0]["content"].lower() and "TERME_1" in a.messages[0]["content"]


def test_plugged_detector_union_conservative_and_fail_closed(monkeypatch):
    mod = types.ModuleType("myriad.privacy_model")
    mod.scan = lambda text: {"p_sensitive": 0.9, "decision": "mask",
                             "spans": [{"start": text.index("Zorg"), "end": text.index("Zorg") + 4, "type": "PERSON",
                                        "score": 0.9}]}
    monkeypatch.setitem(sys.modules, "myriad.privacy_model", mod)
    a = privacy.analyze([{"role": "user", "content": "Ask Zorg about sk-proj-abcdefghijklmnopqrstuvwx1234"}],
                        privacy.PrivacySettings.from_dict({"modes": {"name": "mask"}}))
    assert "Zorg" not in a.messages[0]["content"] and "PERSONNE_1" in a.messages[0]["content"]
    assert "CLE_1" in a.messages[0]["content"]  # the union of both detectors
    mod.scan = lambda text: {"decision": "local", "spans": []}
    assert privacy.analyze([{"role": "user", "content": "hello"}]).decision == "local"  # most conservative wins

    def broken(text):
        raise RuntimeError("model crashed")
    mod.scan = broken
    a = privacy.analyze([{"role": "user", "content": "hello"}])
    assert a.decision == "ask" and "other" in a.confirm_types  # fail-closed
    monkeypatch.delitem(sys.modules, "myriad.privacy_model")
    assert privacy.analyze([{"role": "user", "content": "hello"}]).decision == "send"  # absent: no-op


def test_a_lenient_overlapping_detection_never_hides_a_stricter_one(monkeypatch):
    key = "sk-" + "A" * 24
    text = f"token {key}@example.org"  # the e-mail detection swallows the key
    a = privacy.analyze([{"role": "user", "content": text}], privacy.PrivacySettings.from_dict({"modes": {"email": "off"}}))
    assert key not in a.messages[0]["content"] and a.decision == "mask"
    st = privacy.PrivacySettings.from_dict({"modes": {"custom": "local"}, "terms": [key]})
    assert privacy.analyze([{"role": "user", "content": f"use {key}"}], st).decision == "local"
    # a partial mask never leaves the rest of a detection exposed (re-audit)
    st = privacy.PrivacySettings.from_dict({"terms": ["example"]})
    a = privacy.analyze([{"role": "user", "content": "write to alice@example.com"}], st)
    assert "alice" not in a.messages[0]["content"] and privacy.restore(a.messages[0]["content"], a.mapping) == \
        "write to alice@example.com"
    # merged unmasked detections keep each type's confirmation (3rd audit pass)
    st = privacy.PrivacySettings.from_dict({"modes": {"custom": "warn"}, "terms": ["com +33"],
                                            "confirmed": ["email", "custom"]})
    a = privacy.analyze([{"role": "user", "content": "alice@example.com +33 6 12 34 56 78"}], st)
    assert a.decision == "ask" and "phone" in a.confirm_types


def test_plugged_detector_failure_on_any_message_needs_a_confirmation(monkeypatch):
    mod = types.ModuleType("myriad.privacy_model")

    def scan(text):
        if "second" in text:
            raise RuntimeError("detector crashed")
        i = text.index("Zorg")
        return {"decision": "mask", "spans": [{"start": i, "end": i + 4, "type": "person", "score": 1}]}
    mod.scan = scan
    monkeypatch.setitem(sys.modules, "myriad.privacy_model", mod)
    msgs = [{"role": "user", "content": "first: ask Zorg"}, {"role": "user", "content": "second message"}]
    a = privacy.analyze(msgs, privacy.PrivacySettings.from_dict({"modes": {"name": "mask"}}))
    assert a.decision == "ask" and "other" in a.confirm_types


# ---------------------------------------------------------------- local protection
def test_invite_codes_and_private_swarm_proofs():
    a = Identity.generate()
    code = invite_code(a.node_id)
    assert code.startswith("myr1-") and parse_invite(code) == a.node_id and parse_invite(a.node_id) == a.node_id
    mid = len(code) // 2  # a character carrying id bits (the last one also carries a padding bit)
    assert parse_invite(code[:mid] + ("a" if code[mid] != "a" else "b") + code[mid + 1:]) is None  # checksum
    k = swarm_key("a long shared secret, 2026")
    assert len(k) == 64 and k != swarm_key("another secret, 2026")
    sec = Security({"swarm_key": k})
    proof = sec.swarm_proofs(a.node_id)
    assert proof[0]["g"] == swarm_group(k) and sec.swarm_member(a.node_id, proof)
    b = Identity.generate()
    assert not sec.swarm_member(b.node_id, proof)  # a copied proof does not hold for another node id
    assert not Security({"swarm_key": swarm_key("guess, guess, guess")}).swarm_member(a.node_id, proof)
    assert swarm_proof(k, a.node_id) == proof[0]["p"]
    with pytest.raises(ValueError):
        Security().set_swarm_secret("short")


def test_quarantine_decays_and_expires():
    clock = Clock()
    q = Quarantine(clock)
    for _ in range(3):
        assert not q.note("n1", "disagree")
    assert q.note("n1", "disagree") and q.active("n1")  # 4 disagreements, none agreeing
    clock.t += 30 * 60 + 1
    assert not q.active("n1")  # expires by itself
    for _ in range(10):
        q.note("n2", "agree")
    for _ in range(5):
        q.note("n2", "disagree")
    assert not q.active("n2")  # an honest peer that is sometimes wrong
    q.note("n3", "severe")
    clock.t += 7 * 24 * 3600  # decay: an old incident is forgotten
    assert not q.note("n3", "severe")
    q.note("n3", "severe")
    assert q.active("n3") and q.lift("n3") and not q.active("n3")


def test_check_peer_reasons():
    a, b, c = Identity.generate(), Identity.generate(), Identity.generate()
    sec = Security({"blocked": [{"node_id": a.node_id, "reason": "x", "ts": 0}],
                    "blocked_families": [{"family": "gemma", "reason": "", "ts": 0}], "min_reliability": 0.5})
    p = lambda i, model="Qwen/Qwen3-1.7B-GGUF", **kw: {"node_id": i.node_id, "pubkey": i.pubkey, "model": model, **kw}
    assert sec.check_peer(p(a)) == "blocked"
    assert sec.check_peer(p(b, "ggml-org/gemma-4-E2B-it-GGUF", family="qwen")) == "family_blocked"  # canonical family
    assert sec.check_peer(p(b), reliability=0.4) == "low_reliability"
    assert sec.check_peer(p(b), e2e_ok=False) == "no_e2e"
    assert sec.check_peer(p(b, reputation=0.2)) == "quarantined"  # failed canaries
    assert sec.check_peer(p(c), reliability=0.9, e2e_ok=True) is None
    sec.settings.trusted_only = True
    sec.trust(invite_code(c.node_id), "mon PC")
    assert sec.check_peer(p(c), 0.9, True) is None and sec.check_peer(p(Identity.generate()), 0.9, True) == "not_trusted"
    f = sec.route_fields()
    assert f["only"] == [c.node_id] and a.node_id in f["deny"] and f["deny_families"] == ["gemma"] and f["e2e"]


def test_serve_guard_limits():
    clock = Clock(0.0)
    blocked = Identity.generate().node_id
    g = ServeGuard(Security({"rate_per_min": 3, "max_concurrent": 1, "max_prompt_tokens": 10,
                             "blocked": [{"node_id": blocked, "reason": "", "ts": 0}]}), clock=clock)
    assert g.check(blocked, 4) == "blocked"
    assert g.check("r" * 32, 41) == "prompt_too_long"
    assert g.check("r" * 32, 4) is None
    g.started("r" * 32)
    assert g.check("r" * 32, 4) == "busy_requester"
    g.finished("r" * 32)
    assert g.check("r" * 32, 4) is None and g.check("r" * 32, 4) is None
    assert g.check("r" * 32, 4) == "rate_limited"
    clock.t += 61
    assert g.check("r" * 32, 4) is None
    assert g.serve_policy() == {"deny": [blocked], "rate_per_min": 3, "max_concurrent": 1}


def test_history_cap():
    from myriad.gateway import trim_history
    msgs = [{"role": "system", "content": "s"}] + [{"role": r, "content": f"{r}{i}"} for i in range(5)
                                                    for r in ("user", "assistant")] + [{"role": "user", "content": "q"}]
    kept, dropped = trim_history(msgs, 2)
    assert [m["content"] for m in kept] == ["s", "user4", "assistant4", "q"] and dropped == 8
    assert trim_history(msgs, 0) == (msgs, 0)
