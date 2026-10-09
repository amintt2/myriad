"""Ed25519 identities and signatures over canonical JSON.

Canonical JSON: keys sorted, UTF-8, no whitespace, no NaN/Infinity. Every signature is
domain-separated by the protocol version and a message kind, so a signed receipt can never be
replayed as a signed result (or a login answer).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import PROTOCOL

NODE_ID_HEX = 32  # 128-bit identifier: first 32 hex digits of sha256(raw public key)


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def signing_bytes(kind: str, payload) -> bytes:
    return f"{PROTOCOL}\x00{kind}\x00".encode("utf-8") + canonical_json(payload)


def node_id_from_pubkey(pubkey_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pubkey_hex)).hexdigest()[:NODE_ID_HEX]


def pubkey_matches(node_id: str, pubkey_hex: str) -> bool:
    try:
        return node_id_from_pubkey(pubkey_hex) == node_id
    except ValueError:
        return False


def verify(pubkey_hex: str, kind: str, payload, signature_hex: str) -> bool:
    """True only for a valid signature by this key; malformed input is simply invalid."""
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        key.verify(bytes.fromhex(signature_hex), signing_bytes(kind, payload))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


ACCOUNT_AUTH_S = 300  # a signed account query is valid this long (clock skew included)


def account_params(identity: "Identity", now: float | None = None) -> dict:
    """Query parameters proving that the caller owns `identity` (balances and per-account statistics are
    not public: a peer could otherwise match a debit to the job it served)."""
    import time
    ts = int(time.time() if now is None else now)
    return {"ts": ts, "sig": identity.sign("account", {"node_id": identity.node_id, "ts": ts})}


def account_authorized(node_id: str, pubkey: str | None, ts, sig: str | None, now: float | None = None) -> bool:
    import time
    now = time.time() if now is None else now
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return False
    return (pubkey is not None and isinstance(sig, str) and abs(now - ts) <= ACCOUNT_AUTH_S
            and verify(pubkey, "account", {"node_id": node_id, "ts": ts}, sig))


class Identity:
    def __init__(self, key: Ed25519PrivateKey):
        self._key = key
        raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.pubkey = raw.hex()
        self.node_id = node_id_from_pubkey(self.pubkey)

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_pem(cls, data: bytes) -> "Identity":
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("la clé n'est pas une clé ed25519")
        return cls(key)

    def to_pem(self) -> bytes:
        return self._key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption())

    @classmethod
    def load(cls, path: Path) -> "Identity":
        return cls.from_pem(Path(path).read_bytes())

    def save(self, path: Path) -> None:
        """Written with owner-only permissions where the OS supports it; never overwrites a key."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(self.to_pem())

    @classmethod
    def load_or_create(cls, path: Path) -> "Identity":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        ident = cls.generate()
        ident.save(path)
        return ident

    def sign(self, kind: str, payload) -> str:
        return self._key.sign(signing_bytes(kind, payload)).hex()
