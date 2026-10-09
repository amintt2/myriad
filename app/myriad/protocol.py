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


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="null")


class ChatMessage(_Model):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(max_length=MAX_CONTENT_CHARS)]


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

    def wire(self) -> dict:
        """The signed form: without `tags` when unset, exactly as a node without tags signs it."""
        d = self.model_dump(mode="json")
        if self.tags is None:
            d.pop("tags", None)
        return d


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


class Welcome(_Model):
    t: Literal["welcome"] = "welcome"
    node_id: Hex32
    balance: float


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


Frame = Annotated[Union[Challenge, Hello, Welcome, Status, JobFrame, ResultFrame, JobError, Cancel, ReceiptFrame,
                        ErrorFrame, Assigned, Ping, Pong], Field(discriminator="t")]
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
    if isinstance(frame, JobFrame):
        if frame.route is None:
            exclude["route"] = True
        else:
            sub = {f for f in ("tag", "family") if getattr(frame.route, f) is None}
            if sub:
                exclude["route"] = sub
    elif isinstance(frame, Hello) and frame.info.tags is None:
        exclude["info"] = {"tags"}
    elif isinstance(frame, Assigned):
        if frame.peer.tags is None:
            exclude["peer"] = {"tags"}
        if frame.tag_match is None:
            exclude["tag_match"] = True
    return frame.model_dump_json(exclude=exclude) if exclude else frame.model_dump_json()
