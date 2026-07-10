-- Migration: 0001_qde_brain_tables
-- Purpose: Create tables backing the QDE Brain's metacognition and
--          self-awareness engines (per-symbol brain state, reflections,
--          and experience log), wired into QDEBrain.evaluate_self().
--
-- This mirrors the schema in trainer/qde/brain/persistence.py. The
-- application already creates these tables automatically on startup
-- (CREATE TABLE IF NOT EXISTS), so running this migration is optional
-- and safe to run multiple times (fully idempotent).
--
-- Target DB: SQLite (default path: data/qde_brain.db, override via
-- the QDE brain database's db_path argument / relevant env var).
--
-- Usage:
--   sqlite3 data/qde_brain.db < migrations/0001_qde_brain_tables.sql

CREATE TABLE IF NOT EXISTS brain_state (
    instrument      TEXT PRIMARY KEY,
    tick_count      INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS brain_reflections (
    reflection_id   TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_brain_reflections_instrument ON brain_reflections(instrument);

CREATE TABLE IF NOT EXISTS brain_experiences (
    experience_id   TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    kind            TEXT NOT NULL,
    created_at      REAL NOT NULL,
    data_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_brain_experiences_instrument ON brain_experiences(instrument);
