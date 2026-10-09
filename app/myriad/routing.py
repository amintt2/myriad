"""Model routing names of the OpenAI-compatible gateway, and node skill tags.

Names a client can put in `model`:
- `myriad` (or `essaim`): the fused swarm, k peers of distinct families (unchanged behaviour);
- `myriad:<family>`: one peer of that model family (qwen, gemma, granite, smollm...), strict;
- `myriad:<tag>`: a peer advertising that skill tag (python, typescript, orchestrator, review...),
  any peer when no node has the tag (flagged `fallback` in the response metadata);
- an exact model id of the network (Hugging Face repo id): one peer serving it.

A name `myriad:<x>` is a family when `x` is a known family (from the priors or the live directory),
otherwise a tag. `myriad:family=<x>` and `myriad:tag=<x>` force the reading.

Tags come from the configuration (`tags` in config.json) plus capabilities read from the base model
name (a "coder" model gets `code`...); adapters loaded later will add theirs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .priors import _FAMILY_KEYS, family_of, load, params_of
from .protocol import MAX_TAGS, TAG_PATTERN

SWARM_NAMES = ("myriad", "essaim")
PREFIXES = ("myriad:", "essaim:")
_TAG_RE = re.compile(TAG_PATTERN)
MIN_PARAMS_ORCHESTRATOR = 3.5  # billions of parameters: a base model this size advertises orchestrator/review


class RouteError(ValueError):
    pass


@dataclass(frozen=True)
class RouteSpec:
    kind: str  # swarm, family, tag, model
    value: str | None = None
    name: str = "myriad"  # as the client wrote it

    def public(self, fallback: bool = False) -> dict:
        return {"name": self.name, "kind": self.kind, "value": self.value, "fallback": fallback}


def normalize_tag(raw) -> str | None:
    """'  Python ' -> 'python'; None for anything that is not a valid tag."""
    if not isinstance(raw, str):
        return None
    t = raw.strip().lower().replace(" ", "-")
    return t if _TAG_RE.match(t) else None


def normalize_tags(raw) -> list[str]:
    """Valid, lower-case, de-duplicated tags (order kept), at most MAX_TAGS; invalid ones dropped."""
    out: list[str] = []
    for x in raw or []:
        t = normalize_tag(x)
        if t and t not in out:
            out.append(t)
    return out[:MAX_TAGS]


def model_tags(model_id: str | None) -> list[str]:
    """Capabilities read from the base model name: what any node serving it can advertise."""
    if not model_id:
        return []
    name = model_id.lower()
    tags: list[str] = []
    if "coder" in name or "code" in name.rsplit("/", 1)[-1]:
        tags += ["code", "python", "typescript"]
    if "math" in name:
        tags.append("math")
    p = params_of(model_id)
    if p is not None and p >= MIN_PARAMS_ORCHESTRATOR:
        tags += ["orchestrator", "review"]
    return normalize_tags(tags)


def node_tags(model_id: str | None, configured=None) -> list[str] | None:
    """Tags a serving node advertises: its configuration first, then its base model's capabilities.
    None (no tags field at all) for a client-only node or a node with nothing to advertise."""
    if not model_id:
        return None
    tags = normalize_tags(list(configured or []) + model_tags(model_id))
    return tags or None


def known_families(peers: list[dict] | None = None) -> set[str]:
    fams = {fam for _, fam in _FAMILY_KEYS}
    fams |= {m.get("family") for m in load().get("models", {}).values() if m.get("family")}
    for p in peers or []:
        f = p.get("family") or (family_of(p.get("model")) if p.get("model") else None)
        if f:
            fams.add(str(f).lower())
    return fams


def parse_model(name, families: set[str] | None = None) -> RouteSpec:
    """OpenAI `model` field -> RouteSpec. Raises RouteError for a malformed `myriad:` name."""
    if name is None or name == "" or (isinstance(name, str) and name.strip().lower() in SWARM_NAMES):
        return RouteSpec("swarm", None, str(name or "myriad"))
    if not isinstance(name, str) or len(name) > 200:
        raise RouteError("nom de modèle invalide")
    low = name.strip().lower()
    for pre in PREFIXES:
        if low.startswith(pre):
            rest = low[len(pre):]
            forced = None
            if rest.startswith(("family=", "tag=")):
                forced, _, rest = rest.partition("=")
            if forced == "family" or (forced is None and rest in (families or known_families())):
                if not rest or len(rest) > 64:
                    raise RouteError(f"famille invalide : {rest!r}")
                return RouteSpec("family", rest, name)
            tag = normalize_tag(rest)
            if tag is None:
                raise RouteError(f"compétence invalide : {rest!r} (lettres minuscules, chiffres, - _ . +)")
            return RouteSpec("tag", tag, name)
    return RouteSpec("model", name, name)


def available(p: dict) -> bool:
    """Same eligibility rule as the selection (directory view)."""
    return bool(p.get("model") and p.get("accepting") and not p.get("suspended") and p.get("reputation", 1.0) >= 0.3
                and p.get("busy", 0) < max(1, p.get("max_parallel", 1)))


def live_routes(peers: list[dict]) -> dict[str, dict]:
    """Routing names with live counts, from the directory: {name: {kind, value, peers, available}}."""
    out: dict[str, dict] = {"myriad": {"kind": "swarm", "value": None, "peers": 0, "available": 0}}
    families = known_families(peers)  # the same reading as parse_model: a family name is never a tag

    def add(name: str, kind: str, value: str | None, p: dict) -> None:
        r = out.setdefault(name, {"kind": kind, "value": value, "peers": 0, "available": 0})
        r["peers"] += 1
        r["available"] += int(available(p))

    for p in peers:
        if not p.get("model"):
            continue
        add("myriad", "swarm", None, p)
        fam = str(p.get("family") or family_of(p["model"])).lower()
        add(f"myriad:{fam}", "family", fam, p)
        for t in normalize_tags(p.get("tags") or []):
            add(f"myriad:tag={t}" if t in families else f"myriad:{t}", "tag", t, p)
        add(p["model"], "model", p["model"], p)
    return out
