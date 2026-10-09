"""End-to-end encryption between a requester and the peer that runs its job (essaim/1.3, feature "e2e").

Keys
- Each serving node has an X25519 key-agreement key, certified by its ed25519 identity (KxCert, kind
  "kx", valid KX_LIFETIME_S, replaced every KX_ROTATE_S; the previous key is kept until its certificate
  expires, for the jobs in flight). A requester accepts a certificate only if node_id is the hash of
  the node's ed25519 pubkey, the signature checks out with that pubkey and it is not expired: the
  tracker, which relays everything, cannot substitute a key of its own for a given node id.
- Per job, the requester draws an ephemeral X25519 key and a one-job ed25519 pseudonym (the peer never
  sees the requester's node id; the tracker charges the authenticated connection that sent the job).

Derivation: HKDF-SHA256(salt "myriad-e2e-v1", ikm = X25519(eph, kx), info = signing_bytes("e2e-ctx",
{job_id, requester_id (pseudonym), peer_id, eph, kx})) -> 64 bytes: one ChaCha20-Poly1305 key per
direction (requester -> peer, peer -> requester), so the two directions never share a (key, nonce).

Job: one AEAD message, nonce 0, associated data = the cleartext header (SealedJob.header(): job id,
pseudonym, target, token limit, deadline, expiry, both public keys). The header is also signed by the
pseudonym. Replay: the peer refuses an expired header (`expires_at`, with CLOCK_SKEW_S) and a job id it
has already opened before that expiry (ReplayCache, fail-closed when full).

Answer: padded, then cut in chunks of at most CHUNK bytes; chunk i uses nonce i and associated data
{job header digest, i, final}; the opener refuses a chunk out of order, and an answer whose last chunk
is not authenticated as final (no truncation, no reordering, no splicing across jobs). The peer signs
the envelope with its identity (the tracker settles on it) and, inside the encryption, a digest of the
plaintext answer (kind "rdigest"), which the requester can show to a third party to prove what the
peer answered.

Length hiding: plaintexts are padded to a power of two (1 KiB .. 512 KiB): the tracker learns the
bucket, not the length. What stays visible to the tracker: who asks (the paying account), which peer
answers, when, the bucket sizes, the token limit and count, timings. See docs/08_securite.md.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import heapq
import json
import struct
import time
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import ValidationError

from .crypto import Identity, canonical_json, pubkey_matches, signing_bytes, verify
from .protocol import (MAX_CHUNKS, MAX_DEADLINE_MS, MAX_TEXT_CHARS, Job, JobResult, KxCert, SealedJob,
                       SealedResult)

KX_LIFETIME_S = 26 * 3600  # validity of a key certificate
KX_ROTATE_S = 24 * 3600  # a new key after this; the previous one is kept until its certificate expires
KX_MAX_LIFETIME_S = 48 * 3600  # a requester refuses longer-lived certificates
CLOCK_SKEW_S = 120
BUCKETS = tuple(1 << i for i in range(10, 20))  # padded plaintext sizes: 1 KiB .. 512 KiB
CHUNK = 64 * 1024
SALT = b"myriad-e2e-v1"
INNER_JOB = ("messages", "temperature", "seed", "task_hint", "thinking")


class E2EError(ValueError):
    """`code` is a short, content-free reason, safe to send over the network."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ---------- encoding, padding ----------
def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def b64d(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise E2EError("bad_encoding") from e


def bucket_of(n: int, min_bucket: int = 0) -> int:
    need = max(n, min_bucket)
    for b in BUCKETS:
        if b >= need:
            return b
    raise E2EError("too_large")


def pad(data: bytes, min_bucket: int = 0) -> bytes:
    """4-byte big-endian length, the data, zeros up to the bucket size."""
    size = bucket_of(len(data) + 4, min_bucket)
    return struct.pack(">I", len(data)) + data + bytes(size - len(data) - 4)


def unpad(buf: bytes) -> bytes:
    if len(buf) not in BUCKETS:
        raise E2EError("bad_padding")
    (n,) = struct.unpack(">I", buf[:4])
    if n > len(buf) - 4 or any(buf[4 + n:]):
        raise E2EError("bad_padding")
    return buf[4:4 + n]


def sealed_size(sj: SealedJob) -> int:
    """Padded plaintext size of a sealed job (its bucket), from the ciphertext length."""
    return max(0, len(b64d(sj.ct)) - 16)


# ---------- key certificates ----------
def _raw_pub(key) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def make_cert(identity: Identity, priv: X25519PrivateKey, created: int, lifetime: int = KX_LIFETIME_S,
              model: str | None = None) -> KxCert:
    cert = KxCert(node_id=identity.node_id, kx=_raw_pub(priv).hex(), created=created, expires=created + lifetime,
                  model=model)
    return cert.model_copy(update={"signature": identity.sign("kx", cert.payload())})


def kx_problem(node_id: str, pubkey: str, cert, now: float | None = None, model: str | None = None) -> str | None:
    """Why a peer's key certificate must not be used (None: it can). `cert`: a KxCert or its dict. `model`:
    the model the peer is said to serve (a certificate naming another one is refused)."""
    if cert is None:
        return "no_kx"
    if isinstance(cert, dict):
        try:
            cert = KxCert.model_validate(cert)
        except ValidationError:
            return "bad_kx"
    now = time.time() if now is None else now
    if cert.node_id != node_id:
        return "kx_node_mismatch"
    if not pubkey_matches(node_id, pubkey):
        return "pubkey_mismatch"
    if not cert.verify(pubkey):
        return "bad_kx_signature"
    if cert.expires - cert.created > KX_MAX_LIFETIME_S or cert.expires < cert.created:
        return "kx_lifetime"
    if cert.created > now + CLOCK_SKEW_S:
        return "kx_not_yet_valid"
    if cert.expires < now:
        return "kx_expired"
    if model is not None and cert.model is not None and cert.model != model:
        return "kx_model_mismatch"
    return None


class Keyring:
    """A node's key-agreement keys: the current one and those whose certificate has not expired yet."""

    def __init__(self, identity: Identity, lifetime: int = KX_LIFETIME_S, rotate: int = KX_ROTATE_S, clock=time.time,
                 model: str | None = None):
        self.model = model  # bound into each certificate
        self.identity, self.lifetime, self.rotate_s, self.clock = identity, lifetime, rotate, clock
        self._keys: list[tuple[KxCert, X25519PrivateKey]] = []
        self.rotations = 0
        self._new()

    def _new(self) -> None:
        priv = X25519PrivateKey.generate()
        self._keys.insert(0, (make_cert(self.identity, priv, int(self.clock()), self.lifetime, self.model), priv))
        self.rotations += 1

    def maybe_rotate(self) -> bool:
        """Draw a new key when the current one is KX_ROTATE_S old; True if it did."""
        now = self.clock()
        self._keys = [(c, k) for c, k in self._keys if c.expires + CLOCK_SKEW_S >= now] or self._keys[:1]
        if now - self._keys[0][0].created >= self.rotate_s:
            self._new()
            return True
        return False

    def cert(self) -> KxCert:
        self.maybe_rotate()
        return self._keys[0][0]

    def private(self, kx_hex: str) -> X25519PrivateKey | None:
        now = self.clock()
        for c, k in self._keys:
            if c.kx == kx_hex and c.expires + CLOCK_SKEW_S >= now:
                return k
        return None


class ReplayCache:
    """Job ids already opened, kept until their header expires. Fail-closed when full."""

    def __init__(self, max_items: int = 200_000):
        self.max_items = max_items
        self._seen: dict[str, float] = {}
        self._heap: list[tuple[float, str]] = []

    def add(self, job_id: str, expires_at: float, now: float) -> bool:
        while self._heap and self._heap[0][0] < now:
            _, j = heapq.heappop(self._heap)
            if self._seen.get(j, now + 1) < now:
                del self._seen[j]
        if job_id in self._seen or len(self._seen) >= self.max_items:
            return False
        keep = expires_at + CLOCK_SKEW_S
        self._seen[job_id] = keep
        heapq.heappush(self._heap, (keep, job_id))
        return True


# ---------- sessions ----------
@dataclass
class Session:
    """What both ends of one job share after the key agreement (never sent)."""
    job_id: str
    requester_id: str  # the pseudonym
    peer_id: str
    k_req: bytes
    k_resp: bytes
    header_digest: bytes
    pseudonym: Identity | None = None  # requester side only
    eph_secret: bytes | None = None  # requester side only: revealed to the tracker only in a dispute


def _derive(shared: bytes, job_id: str, requester_id: str, peer_id: str, eph: str, kx: str) -> tuple[bytes, bytes]:
    if not any(shared):  # low-order point: no shared secret
        raise E2EError("bad_key")
    info = signing_bytes("e2e-ctx", {"job_id": job_id, "requester_id": requester_id, "peer_id": peer_id,
                                     "eph": eph, "kx": kx})
    okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=SALT, info=info).derive(shared)
    return okm[:32], okm[32:]


def _nonce(i: int) -> bytes:
    return b"\x00\x00\x00\x00" + struct.pack(">Q", i)


def _header_digest(sj: SealedJob) -> bytes:
    return hashlib.sha256(canonical_json(sj.header())).digest()


def expiry(deadline_ms: int, now: float | None = None) -> int:
    now = time.time() if now is None else now
    return int(now + deadline_ms / 1000) + 1


def seal_job(job: Job, peer_id: str, cert: KxCert, pseudonym: Identity | None = None, now: float | None = None,
             min_bucket: int = 0) -> tuple[SealedJob, Session]:
    """Encrypt `job` for the peer `peer_id` whose (already checked) certificate is `cert`. The sealed job
    is signed by `pseudonym` (a fresh one-job identity by default), never by the requester's identity."""
    pseudonym = pseudonym or Identity.generate()
    eph = X25519PrivateKey.generate()
    eph_hex = _raw_pub(eph).hex()
    try:
        shared = eph.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(cert.kx)))
    except ValueError as e:
        raise E2EError("bad_key") from e
    k_req, k_resp = _derive(shared, job.job_id, pseudonym.node_id, peer_id, eph_hex, cert.kx)
    inner = job.model_dump(mode="json", include=set(INNER_JOB))
    base = SealedJob(job_id=job.job_id, requester_id=pseudonym.node_id, target=peer_id, max_tokens=job.max_tokens,
                     deadline_ms=job.deadline_ms, expires_at=expiry(job.deadline_ms, now), kx=cert.kx, eph=eph_hex,
                     ct="AAAA")
    aad = signing_bytes("e2e-job", base.header())
    ct = ChaCha20Poly1305(k_req).encrypt(_nonce(0), pad(canonical_json(inner), min_bucket), aad)
    sj = base.model_copy(update={"ct": b64e(ct)}).signed_by(pseudonym)
    secret = eph.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                               serialization.NoEncryption())
    return sj, Session(job.job_id, pseudonym.node_id, peer_id, k_req, k_resp, _header_digest(sj), pseudonym, secret)


def dispute_session(sj: SealedJob, eph_secret_hex: str) -> Session:
    """The tracker's view of a disputed job: the requester reveals its ephemeral secret for THIS job (and
    so this job's content), the tracker checks it matches the envelope's `eph` and derives the job's keys
    itself, then tries to open the peer's answer. A wrong secret cannot be passed off as a bad answer."""
    try:
        e = X25519PrivateKey.from_private_bytes(bytes.fromhex(eph_secret_hex))
    except ValueError as exc:
        raise E2EError("bad_dispute") from exc
    if _raw_pub(e).hex() != sj.eph:
        raise E2EError("bad_dispute")
    try:
        shared = e.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(sj.kx)))
    except ValueError as exc:
        raise E2EError("bad_key") from exc
    k_req, k_resp = _derive(shared, sj.job_id, sj.requester_id, sj.target, sj.eph, sj.kx)
    return Session(sj.job_id, sj.requester_id, sj.target, k_req, k_resp, _header_digest(sj))


def open_job(sj: SealedJob, requester_pubkey: str, keyring: Keyring, replay: ReplayCache, peer_id: str,
             now: float | None = None) -> tuple[Job, Session]:
    """Peer side: check, decrypt and rebuild the job (requester_id is the pseudonym). Raises E2EError."""
    now = time.time() if now is None else now
    if not pubkey_matches(sj.requester_id, requester_pubkey) or not sj.verify(requester_pubkey):
        raise E2EError("bad_job_signature")
    if sj.target != peer_id:
        raise E2EError("wrong_target")
    if sj.expires_at + CLOCK_SKEW_S < now or sj.expires_at > now + MAX_DEADLINE_MS / 1000 + CLOCK_SKEW_S + 1:
        raise E2EError("expired")
    priv = keyring.private(sj.kx)
    if priv is None:
        raise E2EError("unknown_kx")
    try:
        shared = priv.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(sj.eph)))
    except ValueError as e:
        raise E2EError("bad_key") from e
    k_req, k_resp = _derive(shared, sj.job_id, sj.requester_id, peer_id, sj.eph, sj.kx)
    try:
        plain = ChaCha20Poly1305(k_req).decrypt(_nonce(0), b64d(sj.ct), signing_bytes("e2e-job", sj.header()))
    except InvalidTag as e:
        raise E2EError("decrypt_failed") from e
    try:
        inner = json.loads(unpad(plain))
        if not isinstance(inner, dict) or set(inner) - set(INNER_JOB):
            raise E2EError("bad_payload")
        job = Job(job_id=sj.job_id, requester_id=sj.requester_id, max_tokens=sj.max_tokens,
                  deadline_ms=sj.deadline_ms, **inner)
    except (ValueError, TypeError, ValidationError) as e:
        raise E2EError(getattr(e, "code", "bad_payload")) from e
    if not replay.add(sj.job_id, sj.expires_at, now):
        raise E2EError("replay")
    return job, Session(sj.job_id, sj.requester_id, peer_id, k_req, k_resp, _header_digest(sj))


# ---------- answers ----------
class ChunkSealer:
    """Answer chunks for one job: counter nonces, the last one flagged final in the associated data."""

    def __init__(self, session: Session):
        self.s, self.i, self.done = session, 0, False
        self.aead = ChaCha20Poly1305(session.k_resp)

    def seal(self, data: bytes, final: bool) -> bytes:
        if self.done:
            raise E2EError("stream_closed")
        aad = signing_bytes("e2e-res", {"job": self.s.header_digest.hex(), "i": self.i, "final": final})
        ct = self.aead.encrypt(_nonce(self.i), data, aad)
        self.i += 1
        self.done = final
        return ct


class ChunkOpener:
    """Requester side: chunks must come in order; `finish` fails unless the final chunk was opened."""

    def __init__(self, session: Session):
        self.s, self.i, self.done = session, 0, False
        self.aead = ChaCha20Poly1305(session.k_resp)
        self.parts: list[bytes] = []

    def open(self, ct: bytes, final: bool) -> bytes:
        if self.done:
            raise E2EError("after_final")
        aad = signing_bytes("e2e-res", {"job": self.s.header_digest.hex(), "i": self.i, "final": final})
        try:
            data = self.aead.decrypt(_nonce(self.i), ct, aad)
        except InvalidTag as e:
            raise E2EError("decrypt_failed") from e
        self.i += 1
        self.done = final
        self.parts.append(data)
        return data

    def finish(self) -> bytes:
        if not self.done:
            raise E2EError("truncated")
        return b"".join(self.parts)


def attest_payload(job_id: str, node_id: str, model: str, completion_tokens: int, text: str, requester_id: str) -> dict:
    return {"job_id": job_id, "node_id": node_id, "model": model, "completion_tokens": completion_tokens,
            "requester_id": requester_id, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def seal_result(identity: Identity, session: Session, model: str, text: str, finish_reason: str | None,
                completion_tokens: int, mean_logprob: float | None, compute_ms: float) -> SealedResult:
    text = text[:MAX_TEXT_CHARS]
    attest = identity.sign("rdigest", attest_payload(session.job_id, identity.node_id, model, completion_tokens, text,
                                                     session.requester_id))
    inner = {"text": text, "finish_reason": finish_reason, "mean_logprob": mean_logprob,
             "compute_ms": round(float(compute_ms), 1), "completion_tokens": completion_tokens, "attest": attest}
    padded = pad(canonical_json(inner))
    parts = [padded[i:i + CHUNK] for i in range(0, len(padded), CHUNK)]
    if len(parts) > MAX_CHUNKS:
        raise E2EError("too_large")
    sealer = ChunkSealer(session)
    chunks = [b64e(sealer.seal(p, i == len(parts) - 1)) for i, p in enumerate(parts)]
    return SealedResult(job_id=session.job_id, node_id=identity.node_id, model=model,
                        completion_tokens=completion_tokens, chunks=chunks).signed_by(identity)


def open_result(sr: SealedResult, session: Session, peer_pubkey: str) -> tuple[JobResult, str]:
    """Requester side: check the envelope signature, decrypt, check the inner digest signature.
    Returns the plaintext result (unsigned JobResult) and the peer's "rdigest" signature."""
    if (sr.job_id != session.job_id or sr.node_id != session.peer_id or not pubkey_matches(sr.node_id, peer_pubkey)
            or not sr.verify(peer_pubkey)):
        raise E2EError("bad_signature")
    opener = ChunkOpener(session)
    for i, c in enumerate(sr.chunks):
        opener.open(b64d(c), i == len(sr.chunks) - 1)
    try:
        inner = json.loads(unpad(opener.finish()))
        if not isinstance(inner, dict):
            raise E2EError("bad_payload")
        text, attest = inner.get("text"), inner.get("attest")
        if not isinstance(text, str) or not isinstance(attest, str) or inner.get("completion_tokens") != sr.completion_tokens:
            raise E2EError("bad_payload")
        res = JobResult(job_id=sr.job_id, node_id=sr.node_id, model=sr.model, text=text,
                        finish_reason=inner.get("finish_reason"), completion_tokens=sr.completion_tokens,
                        mean_logprob=inner.get("mean_logprob"), compute_ms=inner.get("compute_ms") or 0.0)
    except (ValueError, TypeError, ValidationError) as e:
        raise E2EError(getattr(e, "code", "bad_payload")) from e
    if not verify(peer_pubkey, "rdigest", attest_payload(sr.job_id, sr.node_id, sr.model, sr.completion_tokens, text,
                                                         session.requester_id), attest):
        raise E2EError("bad_attest")
    return res, attest
