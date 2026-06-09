"""Optional SQLite persistence for all loopkit state.

All gems work purely in-memory by default.  SQLiteStore adds durable
persistence so state survives process restarts.  It snapshots the full
state of a BeliefEngine, CUSUMBank, CircuitBreakerRegistry, and
DegradationLadder into a single SQLite database.

Usage::

    store = SQLiteStore("loopkit.db")
    store.save_beliefs(engine)
    engine = store.load_beliefs()
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from loopkit.beliefs import BetaBelief, BeliefEngine
from loopkit.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, State
from loopkit.cusum import CUSUM, CUSUMBank


class SQLiteStore:
    """Persist loopkit state to SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS beliefs (
                entity_type TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                context     TEXT NOT NULL DEFAULT 'global',
                alpha       REAL NOT NULL DEFAULT 1.0,
                beta        REAL NOT NULL DEFAULT 1.0,
                total_obs   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (entity_type, entity_id, context)
            );

            CREATE TABLE IF NOT EXISTS cusum_state (
                metric_name       TEXT PRIMARY KEY,
                baseline          REAL NOT NULL,
                allowance_k       REAL NOT NULL DEFAULT 0.05,
                threshold_h       REAL NOT NULL DEFAULT 4.0,
                cusum_upper       REAL NOT NULL DEFAULT 0.0,
                cusum_lower       REAL NOT NULL DEFAULT 0.0,
                observation_count INTEGER NOT NULL DEFAULT 0,
                alert_count       INTEGER NOT NULL DEFAULT 0,
                calibrating       INTEGER NOT NULL DEFAULT 1,
                cal_sum           REAL NOT NULL DEFAULT 0.0,
                cal_n             INTEGER NOT NULL DEFAULT 0,
                cal_target        INTEGER NOT NULL DEFAULT 30
            );

            CREATE TABLE IF NOT EXISTS circuit_breakers (
                service_id  TEXT PRIMARY KEY,
                state       TEXT NOT NULL DEFAULT 'CLOSED',
                failures    INTEGER NOT NULL DEFAULT 0,
                alpha       REAL NOT NULL DEFAULT 1.0,
                beta        REAL NOT NULL DEFAULT 1.0,
                cooldown_s  REAL NOT NULL DEFAULT 60.0,
                threshold   INTEGER NOT NULL DEFAULT 3,
                last_fail   REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS skill_graph (
                parent_skill TEXT NOT NULL,
                child_skill  TEXT NOT NULL,
                PRIMARY KEY (parent_skill, child_skill)
            );

            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                source    TEXT NOT NULL,
                event     TEXT NOT NULL,
                data      TEXT
            );
        """)

    # ── Beliefs ─────────────────────────────────────────────────────

    def save_beliefs(self, engine: BeliefEngine) -> None:
        c = self.conn
        c.execute("DELETE FROM beliefs")
        for (etype, eid, ctx), belief in engine._beliefs.items():
            c.execute(
                "INSERT INTO beliefs (entity_type, entity_id, context, alpha, beta, total_obs) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (etype, eid, ctx, belief.alpha, belief.beta, belief.total_obs),
            )
        c.commit()

    def load_beliefs(self, decay_factor: float = 0.99) -> BeliefEngine:
        engine = BeliefEngine(decay_factor=decay_factor)
        for row in self.conn.execute("SELECT * FROM beliefs"):
            key = (row["entity_type"], row["entity_id"], row["context"])
            engine._beliefs[key] = BetaBelief(
                alpha=row["alpha"],
                beta=row["beta"],
                total_obs=row["total_obs"],
            )
        return engine

    # ── CUSUM ───────────────────────────────────────────────────────

    def save_cusum(self, bank: CUSUMBank) -> None:
        c = self.conn
        c.execute("DELETE FROM cusum_state")
        for name, det in bank._detectors.items():
            c.execute(
                "INSERT INTO cusum_state (metric_name, baseline, allowance_k, threshold_h, "
                "cusum_upper, cusum_lower, observation_count, alert_count, "
                "calibrating, cal_sum, cal_n, cal_target) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name, det.baseline, det.allowance_k, det.threshold_h,
                    det.upper, det.lower, det.observation_count, det.alert_count,
                    int(det._calibrating), det._cal_sum, det._cal_n, det._cal_target,
                ),
            )
        c.commit()

    def load_cusum(self) -> CUSUMBank:
        bank = CUSUMBank()
        for row in self.conn.execute("SELECT * FROM cusum_state"):
            det = CUSUM(
                baseline=row["baseline"],
                allowance_k=row["allowance_k"],
                threshold_h=row["threshold_h"],
                _calibrating=bool(row["calibrating"]),
                _cal_sum=row["cal_sum"],
                _cal_n=row["cal_n"],
                _cal_target=row["cal_target"],
            )
            det.upper = row["cusum_upper"]
            det.lower = row["cusum_lower"]
            det.observation_count = row["observation_count"]
            det.alert_count = row["alert_count"]
            bank._detectors[row["metric_name"]] = det
        return bank

    # ── Circuit Breakers ────────────────────────────────────────────

    def save_circuit_breakers(self, registry: CircuitBreakerRegistry) -> None:
        c = self.conn
        c.execute("DELETE FROM circuit_breakers")
        for name, cb in registry._breakers.items():
            c.execute(
                "INSERT INTO circuit_breakers (service_id, state, failures, alpha, beta, "
                "cooldown_s, threshold, last_fail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name, cb.state.value, cb.failures, cb.alpha, cb.beta_param,
                    cb._cooldown, cb.threshold, cb._last_fail_time,
                ),
            )
        c.commit()

    def load_circuit_breakers(
        self,
        default_threshold: int = 3,
        default_base_cooldown: float = 60.0,
    ) -> CircuitBreakerRegistry:
        registry = CircuitBreakerRegistry(default_threshold, default_base_cooldown)
        for row in self.conn.execute("SELECT * FROM circuit_breakers"):
            cb = CircuitBreaker(
                threshold=row["threshold"],
                base_cooldown=default_base_cooldown,
            )
            cb.state = State(row["state"])
            cb.failures = row["failures"]
            cb.alpha = row["alpha"]
            cb.beta_param = row["beta"]
            cb._cooldown = row["cooldown_s"]
            cb._last_fail_time = row["last_fail"]
            registry._breakers[row["service_id"]] = cb
        return registry

    # ── Skill Graph ─────────────────────────────────────────────────

    def save_skill_graph(self, dag: "SkillDAG") -> None:
        c = self.conn
        c.execute("DELETE FROM skill_graph")
        for parent, children in dag._edges.items():
            for child in children:
                c.execute(
                    "INSERT OR IGNORE INTO skill_graph (parent_skill, child_skill) VALUES (?, ?)",
                    (parent, child),
                )
        c.commit()

    def load_skill_graph(self, beliefs: Optional[BeliefEngine] = None) -> "SkillDAG":
        from loopkit.skill_dag import SkillDAG

        dag = SkillDAG(beliefs=beliefs)
        for row in self.conn.execute("SELECT * FROM skill_graph"):
            dag.add_edge(row["parent_skill"], row["child_skill"])
        return dag

    # ── Events ──────────────────────────────────────────────────────

    def log_event(self, source: str, event: str, data: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO events (source, event, data) VALUES (?, ?, ?)",
            (source, event, json.dumps(data) if data else None),
        )
        self.conn.commit()

    def recent_events(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "source": r["source"],
                "event": r["event"],
                "data": json.loads(r["data"]) if r["data"] else None,
            }
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
