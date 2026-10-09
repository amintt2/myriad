"""Read-only network statistics of a tracker (GET /v1/stats), for the dashboards of the nodes.

Kept apart from tracker.py on purpose: it only reads. Peer figures come from the tracker's cached
/v1/peers snapshot (re-aggregated only when the snapshot changes), traffic figures from one indexed
SQL aggregate over the last minute of the ledger; the response is cached for CACHE_S, and the
per-account part (`?node_id=`) uses indexed sums. Nothing here walks every connection per call."""
from __future__ import annotations

import json
import time
from collections import Counter

from fastapi import FastAPI

from .priors import MILLI

WINDOW_S = 60.0
CACHE_S = 2.0
MAX_ACCOUNTS_CACHED = 4096


def aggregate_peers(peers: list[dict]) -> dict:
    """Peer figures from a /v1/peers listing (serving nodes only)."""
    accepting = [p for p in peers if p.get("accepting") and not p.get("suspended")]
    families = Counter((p.get("family") or p.get("model")) for p in peers)
    models = Counter(p.get("model") for p in peers)
    return {
        "nodes_serving": len(peers),
        "nodes_accepting": len(accepting),
        "families": dict(families.most_common()),
        "models": dict(models.most_common()),
        "busy_slots": sum(int(p.get("busy") or 0) for p in peers),
        "capacity_slots": sum(int(p.get("max_parallel") or 0) for p in accepting),
    }


def traffic(db, window_s: float = WINDOW_S, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    r = db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS t, COUNT(DISTINCT requester_id) AS q "
                   "FROM ledger WHERE ts >= ? AND kind != 'starter'", (now - window_s,)).fetchone()
    return {"window_s": window_s, "tokens_per_s": round(r["t"] / window_s, 2),
            "jobs_per_min": round(r["n"] * 60.0 / window_s, 2),
            "requesters_per_min": round(r["q"] * 60.0 / window_s, 2)}


def account_stats(tracker, node_id: str) -> dict:
    """Credits earned by serving others, and spent asking them (serving yourself moves nothing)."""
    db = tracker.ledger.db
    earned = db.execute("SELECT COALESCE(SUM(amount), 0) AS s, COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS t "
                        "FROM ledger WHERE node_id=? AND kind != 'starter' "
                        "AND (requester_id IS NULL OR requester_id != node_id)", (node_id,)).fetchone()
    spent = db.execute("SELECT COALESCE(SUM(amount), 0) AS s, COUNT(*) AS n FROM ledger "
                       "WHERE requester_id=? AND node_id != requester_id", (node_id,)).fetchone()
    return {"node_id": node_id, "earned": earned["s"] / MILLI, "jobs_served": earned["n"],
            "tokens_served": earned["t"], "spent": spent["s"] / MILLI, "jobs_bought": spent["n"],
            "balance": tracker.ledger.balance(node_id)}


def install(app: FastAPI, tracker) -> None:
    """Add GET /v1/stats to the tracker app."""
    db = tracker.ledger.db
    for sql in ("CREATE INDEX IF NOT EXISTS ledger_ts_idx ON ledger(ts)",
                "CREATE INDEX IF NOT EXISTS ledger_node_idx ON ledger(node_id)",
                "CREATE INDEX IF NOT EXISTS ledger_requester_idx ON ledger(requester_id)"):
        db.execute(sql)
    state: dict = {"net": None, "peers": None, "accounts": {}}

    async def peer_figures() -> dict:
        snap = getattr(tracker, "peers_snapshot", None)
        if snap is None:  # a tracker without the snapshot (older version)
            return aggregate_peers([{**c.info.model_dump(mode="json"), "busy": getattr(c, "busy", 0)}
                                    for c in tracker.conns.values() if c.info.model])
        body = await snap()
        if state["peers"] is None or state["peers"][0] is not body:  # re-aggregate only a new snapshot
            state["peers"] = (body, aggregate_peers(json.loads(body)["peers"]))
        return state["peers"][1]

    @app.get("/v1/stats")
    async def stats(node_id: str | None = None):
        now = time.monotonic()
        hit = state["net"]
        if hit is None or now - hit[0] > CACHE_S:
            figures = {"nodes_online": len(tracker.conns), **(await peer_figures()), **traffic(db), "ts": time.time()}
            hit = state["net"] = (now, figures)
        out = dict(hit[1])
        if node_id:
            node_id = node_id[:128]
            accounts = state["accounts"]
            acct = accounts.get(node_id)
            if acct is None or now - acct[0] > CACHE_S:
                if len(accounts) >= MAX_ACCOUNTS_CACHED:
                    accounts.clear()
                acct = accounts[node_id] = (now, account_stats(tracker, node_id))
            out["account"] = acct[1]
        return out
