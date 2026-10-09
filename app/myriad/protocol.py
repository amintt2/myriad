"""Wire protocol essaim/1: pydantic models for every message, with size limits on every field.

All frames are JSON text frames over one WebSocket per node (node -> tracker, outbound, so a node
behind a NAT box needs no open port). Job, JobResult and Receipt are signed by their author
(see crypto.py); the WebSocket session itself is authenticated by a signed challenge.

essaim/1.1 adds, without changing any essaim/1 frame or signature:
- JobFrame.route (requester -> tracker, no target): the tracker picks the peer itself and answers
  with an Assigned frame naming it, before relaying anything else about that job;
- Ping (tracker -> node) and Pong (node -> tracker): application-level liveness check, the node
  answering after probing its engine; a Pong also carries the node's limits (like Status).
A new frame is only sent to a peer known to support it: a gateway routes jobs through the tracker only
when the tracker announces the "route" feature (GET /v1/health), and the tracker keeps pinging only
the nodes that answered a first ping. JobFrame.route is left out of the JSON when unset, so an
essaim/1 tracker still parses every frame of a new node.

Skill tags (feature "tags", additive, every new field optional and left out of the JSON when unset):
- NodeInfo.tags: skills a node advertises (base model capabilities, loaded adapters), e.g. "python";
- Route.tag / Route.family: route a job to a node with that tag (any node if none has it: the
  Assigned frame then says tag_match=false) or of that model family (strict);
- PeerCard.tags and Assigned.tag_match: only set for a job routed by tag.
A node sends tags only while the tracker accepts them (an older tracker refuses the Hello: the node
reconnects without them), and a gateway routes by tag or family only when the tracker announces "tags".

App updates (feature "update", additive): a node asks for it in the query string of its WebSocket URL
(`/v1/ws?features=update`), which an older tracker simply ignores. Only for such a node, the tracker
sets Welcome.latest_version and sends an UpdateAvailable frame when a newer release is published. Both
are hints: the node validates the version strictly and builds every download URL itself (updater.py).

essaim/1.3 (features "e2e", "policy", "report"; every new field optional and left out of the JSON when
unset, every new frame sent only to a peer known to support it):
- NodeInfo.kx: the node's X25519 key, certified by its ed25519 identity (KxCert, rotated every day);
  KxFrame announces a rotated key. NodeInfo.swarms: private-swarm membership proofs (SwarmProof).
- SealedJob / SealedResult: a job and its result encrypted end to end between the requester and the
  computing peer (e2e.py). The tracker only sees the cleartext header it needs to relay and settle.
  The requester signs a SealedJob with a one-job pseudonym key, so the peer does not learn who asks.
- Reserve (requester -> tracker): pick a peer for a routed job and hold its slot; the Assigned frame
  then carries the peer's KxCert, and the requester sends the SealedJob only after checking the peer.
- Route.deny / deny_families / only / swarm / e2e / min_rel / soft_avoid: the requester's peer policy
  (blocklist, trusted nodes, private swarm, encryption required, minimum reliability, rotation).
- ServePolicy (node -> tracker): requesters this node refuses and its per-requester limits, enforced by
  the tracker (the node itself only sees pseudonyms).
"""
from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from . import PROTOCOL
from .crypto import Identity, verify

MAX_FRAME_BYTES = 1 << 20  # 1 MiB per WebSocket frame
MAX_MESSAGES = 64
MAX_CONTENT_CHARS = 32_000
MAX_PROMPT_CHARS = 128_000  # sum over the messages of a job
MAX_TEXT_CHARS = 64_000  # one generated answer
MAX_TOKENS = 2048
MAX_DEADLINE_MS = 600_000
TASK_HINTS = ("math", "mc", "free")

Hex32 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
PubKey = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Sig = Annotated[str, Field(pattern=r"^([0-9a-f]{128})?$")]
JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
ShortStr = Annotated[str, Field(max_length=200)]
TaskHint = Literal["math", "mc", "free"]
TAG_PATTERN = r"^[a-z0-9][a-z0-9_.+-]{0,31}$"
MAX_TAGS = 16
Tag = Annotated[str, Field(pattern=TAG_PATTERN)]
Tags = Annotated[list[Tag], Field(max_length=MAX_TAGS)]
# essaim/1.3: end-to-end encryption (e2e.py). Sizes: a job's padded plaintext is at most 512 KiB, an
# answer is sealed in chunks of at most 64 KiB (base64 on the wire, inside the 1 MiB frame limit).
MAX_SEALED_JOB_B64 = 700_000
MAX_CHUNK_B64 = 87_500
MAX_CHUNKS = 8
MAX_POLICY_IDS = 1024
B64Job = Annotated[str, Field(pattern=r"^[A-Za-z0-9+/]*={0,2}$", min_length=4, max_length=MAX_SEALED_JOB_B64)]
B64Chunk = Annotated[str, Field(pattern=r"^[A-Za-z0-9+/]*={0,2}$", min_length=4, max_length=MAX_CHUNK_B64)]
UnixTime = Annotated[int, Field(ge=0, le=2**40)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="null")


class ChatMessage(_Model):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(max_length=MAX_CONTENT_CHARS)]


class KxCert(_Model):
    """essaim/1.3: a node's X25519 key-agreement key, signed by its ed25519 identity (kind "kx"). A
    requester checks the signature with the node's pubkey and that node_id is the hash of that pubkey,
    so a tracker cannot substitute its own key for a node."""
    node_id: Hex32
    kx: PubKey
    created: UnixTime
    expires: UnixTime
    model: ShortStr | None = None  # the model this key serves: the tracker cannot relabel the peer's model
    signature: Sig = ""

    def payload(self) -> dict:
        d = {"node_id": self.node_id, "kx": self.kx, "created": self.created, "expires": self.expires}
        if self.model is not None:
            d["model"] = self.model
        return d

    def verify(self, pubkey_hex: str) -> bool:
        return bool(self.signature) and verify(pubkey_hex, "kx", self.payload(), self.signature)


class SwarmProof(_Model):
    """essaim/1.3: membership of a private swarm. g = HMAC(K, "group") identifies the swarm (the tracker
    filters on it); p = HMAC(K, "member" + node_id) proves that this node knows the swarm key K. Only
    members can check p, and it cannot be copied to another node id. K never leaves the members' machines."""
    g: Hex32
    p: Hex32


class NodeInfo(_Model):
    node_id: Hex32
    pubkey: PubKey
    model: ShortStr | None = None  # Hugging Face repo id; None for a client-only node
    family: ShortStr | None = None
    gguf: ShortStr | None = None
    params_b: Annotated[float, Field(ge=0, le=2000, allow_inf_nan=False)] | None = None
    ctx: Annotated[int, Field(ge=0, le=1_048_576)] = 0
    max_parallel: Annotated[int, Field(ge=0, le=64)] = 1
    version: Literal["essaim/1"] = PROTOCOL
    accepting: bool = False
    tags: Tags | None = None  # skill tags (feature "tags"); None: left out of the wire format
    kx: KxCert | None = None  # essaim/1.3 (feature "e2e")
    swarms: Annotated[list[SwarmProof], Field(max_length=8)] | None = None  # essaim/1.3 private swarms

    OPTIONAL: ClassVar[tuple] = ("tags", "kx", "swarms")

    def wire(self) -> dict:
        """The signed form: optional fields left out when unset, exactly as an older node signs it."""
        d = self.model_dump(mode="json")
        for f in self.OPTIONAL:
            if getattr(self, f) is None:
                d.pop(f, None)
        return d

    def unset(self) -> set:
        return {f for f in self.OPTIONAL if getattr(self, f) is None}


class _Signed(_Model):
    signature: Sig = ""
    KIND: ClassVar[str] = ""

    def payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"signature"})

    def signed_by(self, ident: Identity):
        return self.model_copy(update={"signature": ident.sign(self.KIND, self.payload())})

    def verify(self, pubkey_hex: str) -> bool:
        return bool(self.signature) and verify(pubkey_hex, self.KIND, self.payload(), self.signature)


class Job(_Signed):
    KIND: ClassVar[str] = "job"
    job_id: JobId
    requester_id: Hex32
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=MAX_MESSAGES)]
    max_tokens: Annotated[int, Field(ge=1, le=MAX_TOKENS)] = 512
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    deadline_ms: Annotated[int, Field(ge=100, le=MAX_DEADLINE_MS)] = 120_000
    task_hint: TaskHint | None = None
    thinking: bool = False

    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


class JobResult(_Signed):
    KIND: ClassVar[str] = "result"
    job_id: JobId
    node_id: Hex32
    model: ShortStr
    text: Annotated[str, Field(max_length=MAX_TEXT_CHARS)]
    finish_reason: ShortStr | None = None
    completion_tokens: Annotated[int, Field(ge=0, le=MAX_TOKENS)] = 0
    mean_logprob: Annotated[float, Field(le=0, allow_inf_nan=False)] | None = None
    compute_ms: Annotated[float, Field(ge=0, le=10 * MAX_DEADLINE_MS, allow_inf_nan=False)] = 0.0


class Receipt(_Signed):
    """Signed by the requester: 'I received this result'. The tracker settles credits on it.
    `agreed`: did this answer agree with the fused decision (None: not judged)."""
    KIND: ClassVar[str] = "receipt"
    job_id: JobId
    requester_id: Hex32
    node_id: Hex32
    completion_tokens: Annotated[int, Field(ge=0, le=MAX_TOKENS)]
    model: ShortStr
    agreed: bool | None = None


class SealedJob(_Signed):
    """essaim/1.3: a job encrypted for one peer (e2e.py). Cleartext header: what the tracker needs to
    relay and settle (job id, target, token limit, deadline) and what the peer needs to decrypt (its
    key `kx`, the requester's ephemeral key `eph`); `ct` holds the padded, encrypted job. Signed by the
    requester's one-job pseudonym key (requester_id is that key's hash, not the requester's node id);
    the header is also the AEAD's associated data."""
    KIND: ClassVar[str] = "sjob"
    job_id: JobId
    requester_id: Hex32
    target: Hex32
    max_tokens: Annotated[int, Field(ge=1, le=MAX_TOKENS)] = 512
    deadline_ms: Annotated[int, Field(ge=100, le=MAX_DEADLINE_MS)] = 120_000
    expires_at: UnixTime
    kx: PubKey
    eph: PubKey
    ct: B64Job

    def header(self) -> dict:
        return self.model_dump(mode="json", exclude={"signature", "ct"})


class SealedResult(_Signed):
    """essaim/1.3: a result encrypted for the requester. Cleartext: what the tracker settles on (model,
    completion tokens); `chunks`: the padded answer, one AEAD message per chunk (counter nonces, the
    last chunk authenticated as final). Signed by the peer's identity for the tracker's accounting;
    inside the encryption the peer also signs a digest of the plaintext answer (kind "rdigest")."""
    KIND: ClassVar[str] = "sresult"
    job_id: JobId
    node_id: Hex32
    model: ShortStr
    completion_tokens: Annotated[int, Field(ge=0, le=MAX_TOKENS)] = 0
    chunks: Annotated[list[B64Chunk], Field(min_length=1, max_length=MAX_CHUNKS)]


class NodeReport(_Signed):
    """essaim/1.3: a signed report about a node, sent to the tracker (POST /v1/report). `job_id`: a job
    the reporter paid that node for (required for "disagreement")."""
    KIND: ClassVar[str] = "report"
    reporter_id: Hex32
    node_id: Hex32
    reason: Literal["disagreement", "bad_output", "abuse", "spam", "other"]
    job_id: JobId | None = None
    ts: UnixTime
    note: Annotated[str, Field(max_length=200)] = ""


# ---------- frames ----------
class Challenge(_Model):
    t: Literal["challenge"] = "challenge"
    nonce: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tracker_id: Hex32
    tracker_pubkey: PubKey


class Hello(_Model):
    """Answer to the challenge: the node signs (nonce, info) with kind 'hello'."""
    t: Literal["hello"] = "hello"
    info: NodeInfo
    nonce: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    signature: Sig

    @staticmethod
    def make(ident: Identity, info: NodeInfo, nonce: str) -> "Hello":
        sig = ident.sign("hello", {"info": info.wire(), "nonce": nonce})
        return Hello(info=info, nonce=nonce, signature=sig)

    def valid(self) -> bool:
        return verify(self.info.pubkey, "hello", {"info": self.info.wire(), "nonce": self.nonce},
                      self.signature)


VersionHint = Annotated[str, Field(max_length=64)]  # validated strictly by the receiver (release.py)


class Welcome(_Model):
    t: Literal["welcome"] = "welcome"
    node_id: Hex32
    balance: float
    latest_version: VersionHint | None = None  # feature "update": only for a node that asked for it


class UpdateAvailable(_Model):
    """Feature "update", tracker -> node: a new app release was published (a hint, see updater.py)."""
    t: Literal["update"] = "update"
    version: VersionHint


class Status(_Model):
    """Node -> tracker: current limits (sent on change and as a heartbeat)."""
    t: Literal["status"] = "status"
    accepting: bool
    max_parallel: Annotated[int, Field(ge=0, le=64)]


class Route(_Model):
    """essaim/1.1: the tracker chooses the target of a job sent without one.

    `group` ties the jobs of one gateway request together: the tracker gives them peers of distinct
    model families. `replaces`: this job replaces a job of the same group that was refused or failed
    (the tracker then tries an unused family first, else the replaced job's family)."""
    group: Hex32
    model: ShortStr | None = None  # only nodes serving this model
    exclude: Annotated[list[Hex32], Field(max_length=64)] = []  # never these nodes
    replaces: JobId | None = None
    tag: Tag | None = None  # feature "tags": a node advertising this tag, else any node (tag_match false)
    family: ShortStr | None = None  # feature "tags": only nodes of this model family
    # essaim/1.3 (feature "policy"): the requester's peer policy, applied by the tracker's selection
    # (the requester checks the assigned peer again before sending anything).
    deny: Annotated[list[Hex32], Field(max_length=MAX_POLICY_IDS)] = []  # never these nodes (blocklist)
    deny_families: Annotated[list[ShortStr], Field(max_length=32)] = []  # never these model families
    only: Annotated[list[Hex32], Field(max_length=MAX_POLICY_IDS)] | None = None  # only these (trusted)
    swarm: Hex32 | None = None  # only members of this private swarm (SwarmProof.g)
    e2e: bool | None = None  # only nodes with a valid KxCert
    min_rel: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None  # minimum model reliability
    soft_avoid: Annotated[list[Hex32], Field(max_length=64)] = []  # avoided when possible (peer rotation)

    NEW: ClassVar[tuple] = ("tag", "family", "deny", "deny_families", "only", "swarm", "e2e", "min_rel",
                            "soft_avoid")

    def unset(self) -> set:
        return {f for f in self.NEW if getattr(self, f) is None or getattr(self, f) == []}


class JobFrame(_Model):
    """Requester -> tracker (target set, or route set in essaim/1.1), then tracker -> target node
    (requester_pubkey set)."""
    t: Literal["job"] = "job"
    job: Job
    target: Hex32 | None = None
    requester_pubkey: PubKey | None = None
    route: Route | None = None


class PeerCard(_Model):
    """What a requester needs to know about a peer chosen for it: identity (to verify its signed
    result), model and family, and the model's reliability published by the tracker (its vote weight)."""
    node_id: Hex32
    pubkey: PubKey
    model: ShortStr
    family: ShortStr | None = None
    gguf: ShortStr | None = None
    params_b: Annotated[float, Field(ge=0, le=2000, allow_inf_nan=False)] | None = None
    reliability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tags: Tags | None = None  # set only for a job routed by tag
    # essaim/1.3, set only for a requester that reserved the job (feature "e2e"):
    kx: KxCert | None = None
    swarms: Annotated[list[SwarmProof], Field(max_length=8)] | None = None
    reputation: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None

    OPTIONAL: ClassVar[tuple] = ("tags", "kx", "swarms", "reputation")

    def unset(self) -> set:
        return {f for f in self.OPTIONAL if getattr(self, f) is None}


class Assigned(_Model):
    """essaim/1.1, tracker -> requester: the peer chosen for a routed job. Sent on the requester's
    connection before the job is relayed, so it always arrives before that job's result or error."""
    t: Literal["assigned"] = "assigned"
    job_id: JobId
    peer: PeerCard
    tag_match: bool | None = None  # job routed by tag: does the peer advertise it (False: fallback)


class Ping(_Model):
    """essaim/1.1, tracker -> node: are you (and your engine) alive? Answer with Pong(seq)."""
    t: Literal["ping"] = "ping"
    seq: Annotated[int, Field(ge=0, le=2**53)]


class Pong(_Model):
    """essaim/1.1, node -> tracker: answer to Ping(seq), sent after a short engine health probe. It
    also carries the node's current limits, so it replaces the Status heartbeat."""
    t: Literal["pong"] = "pong"
    seq: Annotated[int, Field(ge=0, le=2**53)]
    accepting: bool
    max_parallel: Annotated[int, Field(ge=0, le=64)]
    engine_ok: bool = True
    running: Annotated[int, Field(ge=0, le=1024)] = 0


class ResultFrame(_Model):
    t: Literal["result"] = "result"
    result: JobResult


class JobError(_Model):
    t: Literal["job_error"] = "job_error"
    job_id: JobId
    node_id: Hex32 | None = None
    error: Annotated[str, Field(max_length=500)]


class Cancel(_Model):
    t: Literal["cancel"] = "cancel"
    job_id: JobId


class ReceiptFrame(_Model):
    t: Literal["receipt"] = "receipt"
    receipt: Receipt


class ErrorFrame(_Model):
    t: Literal["error"] = "error"
    error: Annotated[str, Field(max_length=500)]


class Reserve(_Model):
    """essaim/1.3, requester -> tracker: choose a peer for this routed job and hold its slot. The tracker
    answers with an Assigned frame (carrying the peer's KxCert), then waits RESERVE_TTL_S for the
    SealedJob (or, for a peer without encryption, a JobFrame naming it)."""
    t: Literal["reserve"] = "reserve"
    job_id: JobId
    route: Route
    max_tokens: Annotated[int, Field(ge=1, le=MAX_TOKENS)] = 512
    deadline_ms: Annotated[int, Field(ge=100, le=MAX_DEADLINE_MS)] = 120_000


class SealedJobFrame(_Model):
    """essaim/1.3: requester -> tracker -> job.target. `requester_pubkey`: the pseudonym key that signed
    the SealedJob (the tracker charges the authenticated connection that sent it)."""
    t: Literal["sjob"] = "sjob"
    job: SealedJob
    requester_pubkey: PubKey


class SealedResultFrame(_Model):
    t: Literal["sresult"] = "sresult"
    result: SealedResult


class KxFrame(_Model):
    """essaim/1.3, node -> tracker: the node's new key-agreement key (daily rotation)."""
    t: Literal["kx"] = "kx"
    kx: KxCert


class Dispute(_Model):
    """essaim/1.3, requester -> tracker: this encrypted answer does not open. The requester reveals the
    job's ephemeral secret (so the tracker can read THIS job) for the tracker to check it: an answer that
    really does not open is not paid and counts against the peer; a false dispute is paid as usual."""
    t: Literal["dispute"] = "dispute"
    job_id: JobId
    eph_secret: PubKey  # 32 bytes, hex


class ServePolicy(_Model):
    """essaim/1.3, node -> tracker: requesters this node refuses (blocklist) and its limits per
    requester. A peer only sees one-job pseudonyms, so the tracker, which knows the paying account,
    enforces them."""
    t: Literal["spolicy"] = "spolicy"
    deny: Annotated[list[Hex32], Field(max_length=MAX_POLICY_IDS)] = []
    rate_per_min: Annotated[int, Field(ge=0, le=10_000)] = 0  # 0: no limit
    max_concurrent: Annotated[int, Field(ge=0, le=64)] = 0  # 0: no limit


Frame = Annotated[Union[Challenge, Hello, Welcome, Status, JobFrame, ResultFrame, JobError, Cancel, ReceiptFrame,
                        ErrorFrame, Assigned, Ping, Pong, UpdateAvailable, Reserve, SealedJobFrame,
                        SealedResultFrame, KxFrame, ServePolicy, Dispute], Field(discriminator="t")]
FRAME = TypeAdapter(Frame)


def parse_frame(raw: str | bytes):
    """Parse one frame; raises ValueError on oversize or invalid input."""
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("frame too large")
    return FRAME.validate_json(raw)


def dump_frame(frame: _Model) -> str:
    """JSON text of a frame. Optional fields added after essaim/1 are left out when unset: an older
    peer forbids unknown fields."""
    exclude: dict = {}
    if isinstance(frame, (JobFrame, Reserve)):
        if frame.route is None:
            exclude["route"] = True
        else:
            sub = frame.route.unset()
            if sub:
                exclude["route"] = sub
    elif isinstance(frame, Hello):
        sub = frame.info.unset()
        if sub:
            exclude["info"] = sub
    elif isinstance(frame, Welcome) and frame.latest_version is None:
        exclude["latest_version"] = True
    elif isinstance(frame, Assigned):
        sub = frame.peer.unset()
        if sub:
            exclude["peer"] = sub
        if frame.tag_match is None:
            exclude["tag_match"] = True
    return frame.model_dump_json(exclude=exclude) if exclude else frame.model_dump_json()
