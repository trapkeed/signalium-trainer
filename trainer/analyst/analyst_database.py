"""Phase 23 — Analyst Database.

SQLite persistence for SAQRA.
6 tables: chat_sessions, research_queries, generated_reports,
          recommendation_history, prompt_templates, llm_usage.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_models import (
    ChatSession,
    Recommendation,
    ResearchQuery,
    ResearchReport,
)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id   TEXT PRIMARY KEY,
    researcher   TEXT,
    role         TEXT,
    message_count INTEGER DEFAULT 0,
    active       INTEGER DEFAULT 1,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS research_queries (
    query_id     TEXT PRIMARY KEY,
    session_id   TEXT,
    text         TEXT NOT NULL,
    query_type   TEXT,
    role         TEXT,
    timestamp    REAL NOT NULL,
    response_preview TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS generated_reports (
    report_id    TEXT PRIMARY KEY,
    report_type  TEXT NOT NULL,
    title        TEXT,
    period       TEXT,
    word_count   INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    grounded     INTEGER DEFAULT 1,
    generated_at REAL NOT NULL,
    content      TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS recommendation_history (
    rec_id       TEXT PRIMARY KEY,
    rec_type     TEXT NOT NULL,
    title        TEXT,
    description  TEXT,
    priority     TEXT DEFAULT 'MEDIUM',
    implemented  INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    timestamp    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    template_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    query_type   TEXT,
    template     TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    usage_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    model_name    TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    response_tokens INTEGER DEFAULT 0,
    latency_ms    REAL DEFAULT 0,
    grounded      INTEGER DEFAULT 1,
    safe          INTEGER DEFAULT 1,
    query_type    TEXT DEFAULT '',
    timestamp     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_queries_session   ON research_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_type      ON research_queries(query_type);
CREATE INDEX IF NOT EXISTS idx_reports_type      ON generated_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_recs_priority     ON recommendation_history(priority);
CREATE INDEX IF NOT EXISTS idx_usage_provider    ON llm_usage(provider);
"""


class AnalystDatabase:
    """SQLite persistence for the SAQRA analyst system."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._persistent_conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLES)

    @contextlib.contextmanager
    def _conn(self):
        if self._db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _commit(self, conn) -> None:
        if self._db_path == ":memory:":
            conn.commit()

    # ── Chat sessions ─────────────────────────────────────────────────────────

    def save_session(self, session: ChatSession) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO chat_sessions
                   (session_id, researcher, role, message_count, active, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    session.session_id,
                    session.researcher,
                    session.role.value,
                    session.message_count,
                    int(session.active),
                    session.created_at,
                    session.updated_at,
                ),
            )
            self._commit(conn)

    def count_sessions(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]

    def active_session_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT session_id FROM chat_sessions WHERE active=1").fetchall()
        return [r["session_id"] for r in rows]

    # ── Research queries ──────────────────────────────────────────────────────

    def save_query(self, query: ResearchQuery, response_preview: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO research_queries
                   (query_id, session_id, text, query_type, role, timestamp, response_preview)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    query.query_id,
                    query.session_id,
                    query.text,
                    query.query_type.value,
                    query.role.value,
                    query.timestamp,
                    response_preview[:300],
                ),
            )
            self._commit(conn)

    def count_queries(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM research_queries").fetchone()[0]

    def recent_queries(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM research_queries ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Reports ───────────────────────────────────────────────────────────────

    def save_report(self, report: ResearchReport) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO generated_reports
                   (report_id, report_type, title, period, word_count,
                    citation_count, grounded, generated_at, content)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    report.report_id,
                    report.report_type.value,
                    report.title,
                    report.period,
                    report.word_count,
                    len(report.citations),
                    int(report.grounded),
                    report.generated_at,
                    report.content[:2000],
                ),
            )
            self._commit(conn)

    def count_reports(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM generated_reports").fetchone()[0]

    def reports_by_type(self, report_type: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM generated_reports WHERE report_type=? ORDER BY generated_at DESC",
                (report_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Recommendations ───────────────────────────────────────────────────────

    def save_recommendation(self, rec: Recommendation) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO recommendation_history
                   (rec_id, rec_type, title, description, priority,
                    implemented, citation_count, timestamp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    rec.rec_id,
                    rec.rec_type.value,
                    rec.title,
                    rec.description,
                    rec.priority,
                    int(rec.implemented),
                    len(rec.citations),
                    rec.timestamp,
                ),
            )
            self._commit(conn)

    def count_recommendations(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()[0]

    def pending_recommendations(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM recommendation_history WHERE implemented=0 ORDER BY priority, timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Prompt templates ──────────────────────────────────────────────────────

    def save_template(self, name: str, query_type: str, template: str) -> None:
        with self._conn() as conn:
            now = time.time()
            conn.execute(
                """INSERT OR REPLACE INTO prompt_templates
                   (name, query_type, template, created_at, updated_at)
                   VALUES (?,?,?,?,?)""",
                (name, query_type, template, now, now),
            )
            self._commit(conn)

    def count_templates(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM prompt_templates").fetchone()[0]

    # ── LLM usage ─────────────────────────────────────────────────────────────

    def record_usage(
        self,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        response_tokens: int,
        latency_ms: float,
        grounded: bool,
        safe: bool,
        query_type: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO llm_usage
                   (provider, model_name, prompt_tokens, response_tokens,
                    latency_ms, grounded, safe, query_type, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    provider,
                    model_name,
                    prompt_tokens,
                    response_tokens,
                    latency_ms,
                    int(grounded),
                    int(safe),
                    query_type,
                    time.time(),
                ),
            )
            self._commit(conn)

    def total_tokens(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT SUM(prompt_tokens + response_tokens) FROM llm_usage"
            ).fetchone()
        return row[0] or 0

    def usage_by_provider(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT provider, COUNT(*) as cnt FROM llm_usage GROUP BY provider"
            ).fetchall()
        return {r["provider"]: r["cnt"] for r in rows}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "sessions": self.count_sessions(),
            "queries": self.count_queries(),
            "reports": self.count_reports(),
            "recommendations": self.count_recommendations(),
            "templates": self.count_templates(),
            "total_tokens": self.total_tokens(),
        }
