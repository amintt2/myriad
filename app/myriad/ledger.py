"""SQLite store of the tracker: accounts and credits, settlement log with signed evidence,
reliability statistics per model, spot-check statistics per node, tracker key.

Amounts are integers in thousandths of a credit (priors.MILLI). Every settlement is unique per job
id, so a receipt can never be paid twice."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .priors import MILLI

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
  node_id TEXT PRIMARY KEY, pubkey TEXT NOT NULL, balance INTEGER NOT NULL,
  created_at REAL NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, job_id TEXT UNIQUE,
  requester_id TEXT, node_id TEXT NOT NULL, model TEXT, tokens INTEGER NOT NULL DEFAULT 0,
  amount INTEGER NOT NULL, kind TEXT NOT NULL, receipt TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS model_stats (model TEXT PRIMARY KEY, agree INTEGER NOT NULL DEFAULT 0,
  disagree INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS node_stats (node_id TEXT PRIMARY KEY, spot_agree INTEGER NOT NULL DEFAULT 0,
  spot_disagree INTEGER NOT NULL DEFAULT 0, jobs_ok INTEGER NOT NULL DEFAULT 0,
  jobs_failed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS model_spot (model TEXT PRIMARY KEY, agree INTEGER NOT NULL DEFAULT 0,
  disagree INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS node_spot (node_id TEXT NOT NULL, model TEXT NOT NULL,
  agree INTEGER NOT NULL DEFAULT 0, disagree INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (node_id, model));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Ledger:
    def __init__(self, path: str | Path, starter_credit: float = 1000.0):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.starter = int(round(starter_credit * MILLI))
        self._lock = threading.Lock()

    def close(self):
        self.db.close()

    # ---------- meta ----------
    def get_meta(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ---------- accounts ----------
    def ensure_account(self, node_id: str, pubkey: str) -> bool:
        """Create the account with the starter credit; True if it is new."""
        now = time.time()
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO accounts(node_id, pubkey, balance, created_at, last_seen) VALUES (?,?,?,?,?)",
                    (node_id, pubkey, self.starter, now, now))
                created = cur.rowcount == 1
                if created:
                    self.db.execute("INSERT INTO ledger(ts, node_id, amount, kind) VALUES (?,?,?,?)",
                                    (now, node_id, self.starter, "starter"))
                else:
                    self.db.execute("UPDATE accounts SET last_seen=? WHERE node_id=?", (now, node_id))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return created

    def balance_milli(self, node_id: str) -> int | None:
        r = self.db.execute("SELECT balance FROM accounts WHERE node_id=?", (node_id,)).fetchone()
        return int(r["balance"]) if r else None

    def balance(self, node_id: str) -> float | None:
        b = self.balance_milli(node_id)
        return None if b is None else b / MILLI

    def balances(self) -> dict[str, float]:
        return {r["node_id"]: r["balance"] / MILLI for r in self.db.execute("SELECT node_id, balance FROM accounts")}

    def has_job(self, job_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM ledger WHERE job_id=?", (job_id,)).fetchone() is not None

    # ---------- settlement ----------
    def settle(self, job_id: str, requester_id: str | None, node_id: str, model: str, tokens: int, amount: int,
               kind: str, receipt: dict | None = None, result: dict | None = None) -> bool:
        """Credit `node_id` and debit `requester_id` (None: minted by the network, e.g. spot checks).
        A job id is settled at most once; returns False if it already was."""
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO ledger(ts, job_id, requester_id, node_id, model, tokens, amount, kind, receipt, result)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (time.time(), job_id, requester_id, node_id, model, tokens, amount, kind,
                     json.dumps(receipt) if receipt else None, json.dumps(result) if result else None))
                if cur.rowcount != 1:
                    self.db.execute("ROLLBACK")
                    return False
                if requester_id != node_id:  # serving yourself moves no credit
                    self.db.execute("UPDATE accounts SET balance = balance + ? WHERE node_id=?", (amount, node_id))
                    if requester_id is not None:
                        self.db.execute("UPDATE accounts SET balance = balance - ? WHERE node_id=?",
                                        (amount, requester_id))
                self.db.execute("INSERT OR IGNORE INTO node_stats(node_id) VALUES (?)", (node_id,))
                self.db.execute("UPDATE node_stats SET jobs_ok = jobs_ok + 1 WHERE node_id=?", (node_id,))
                self.db.execute("COMMIT")
                return True
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT ts, job_id, requester_id, node_id, model, tokens, amount, kind FROM ledger"
                               " ORDER BY id DESC LIMIT ?", (limit,))
        return [{**dict(r), "amount": r["amount"] / MILLI} for r in rows]

    def evidence(self, job_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM ledger WHERE job_id=?", (job_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["amount"] = d["amount"] / MILLI
        for k in ("receipt", "result"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    # ---------- statistics ----------
    def record_agreement(self, model: str, agreed: bool) -> None:
        col = "agree" if agreed else "disagree"
        self.db.execute("INSERT OR IGNORE INTO model_stats(model) VALUES (?)", (model,))
        self.db.execute(f"UPDATE model_stats SET {col} = {col} + 1 WHERE model=?", (model,))

    def model_stats(self) -> dict[str, tuple[int, int]]:
        return {r["model"]: (r["agree"], r["disagree"]) for r in self.db.execute("SELECT * FROM model_stats")}

    def record_spot(self, node_id: str, model: str, agreed: bool) -> None:
        """One spot-check comparison involving `node_id` while it served `model` (lifetime totals are
        kept in node_stats; reputation uses the per-model counts)."""
        col = "spot_agree" if agreed else "spot_disagree"
        self.db.execute("INSERT OR IGNORE INTO node_stats(node_id) VALUES (?)", (node_id,))
        self.db.execute(f"UPDATE node_stats SET {col} = {col} + 1 WHERE node_id=?", (node_id,))
        col = "agree" if agreed else "disagree"
        self.db.execute("INSERT OR IGNORE INTO node_spot(node_id, model) VALUES (?, ?)", (node_id, model))
        self.db.execute(f"UPDATE node_spot SET {col} = {col} + 1 WHERE node_id=? AND model=?", (node_id, model))

    def node_spot(self, node_id: str, model: str) -> tuple[int, int]:
        r = self.db.execute("SELECT agree, disagree FROM node_spot WHERE node_id=? AND model=?",
                            (node_id, model)).fetchone()
        return (r["agree"], r["disagree"]) if r else (0, 0)

    def record_model_spot(self, model: str, agreed: bool) -> None:
        """One spot-check comparison between two nodes of `model` (the baseline disagreement rate)."""
        col = "agree" if agreed else "disagree"
        self.db.execute("INSERT OR IGNORE INTO model_spot(model) VALUES (?)", (model,))
        self.db.execute(f"UPDATE model_spot SET {col} = {col} + 1 WHERE model=?", (model,))

    def model_spot_stats(self) -> dict[str, tuple[int, int]]:
        return {r["model"]: (r["agree"], r["disagree"]) for r in self.db.execute("SELECT * FROM model_spot")}

    def record_failure(self, node_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO node_stats(node_id) VALUES (?)", (node_id,))
        self.db.execute("UPDATE node_stats SET jobs_failed = jobs_failed + 1 WHERE node_id=?", (node_id,))

    def node_stats(self, node_id: str) -> dict:
        r = self.db.execute("SELECT * FROM node_stats WHERE node_id=?", (node_id,)).fetchone()
        return dict(r) if r else {"node_id": node_id, "spot_agree": 0, "spot_disagree": 0, "jobs_ok": 0, "jobs_failed": 0}
