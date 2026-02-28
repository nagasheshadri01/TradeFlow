"""
tests/test_database.py
───────────────────────
Sanity-checks for modules/database.py.

Each test gets a fresh, isolated SQLite database via a tmpdir-based fixture.
We monkeypatch get_connection to point at the temp DB so no real data is touched.

v2: adds deduplication tests for headline_hash / headline_seen().
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest

import modules.database as db_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Give every test its own temporary SQLite file.

    We monkeypatch db_module.get_connection with a closure over the temp path,
    then bootstrap the schema.  After the test the tmpdir is cleaned up by pytest.
    """
    db_file = tmp_path / "test_sentiment.db"

    @contextmanager
    def _tmp_conn() -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            str(db_file),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_module, "get_connection", _tmp_conn)

    # Bootstrap schema in the temp DB
    with _tmp_conn() as conn:
        conn.executescript(db_module.SCHEMA_SQL)


# ── Basic insert / fetch ──────────────────────────────────────────────────────

class TestInsertAndFetch:
    def test_insert_and_fetch_signal(self):
        db_module.insert_signal("AAPL", "Great earnings!", 0.8, "POSITIVE", "vader")
        rows = db_module.fetch_signals("AAPL")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["label"] == "POSITIVE"

    def test_insert_and_fetch_decision(self):
        db_module.insert_decision("MSFT", "BUY", "Strong uptrend", 0.65)
        rows = db_module.fetch_decisions("MSFT")
        assert len(rows) == 1
        assert rows[0]["action"] == "BUY"

    def test_fetch_empty(self):
        assert db_module.fetch_signals("TSLA") == []
        assert db_module.fetch_decisions("TSLA") == []

    def test_insert_signal_returns_true_on_new(self):
        """insert_signal should return True when a new row is written."""
        result = db_module.insert_signal("GOOG", "Record ad revenue", 0.7, "POSITIVE", "vader")
        assert result is True

    def test_signal_has_headline_hash(self):
        """Stored rows must carry a 64-char SHA-256 headline_hash."""
        db_module.insert_signal("AAPL", "Strong iPhone sales", 0.6, "POSITIVE", "vader")
        rows = db_module.fetch_signals("AAPL")
        assert rows[0]["headline_hash"] != ""
        assert len(rows[0]["headline_hash"]) == 64


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_duplicate_headline_ignored(self):
        """Same headline + same ticker → only one row stored."""
        db_module.insert_signal("AAPL", "Best quarter ever", 0.9, "POSITIVE", "vader")
        db_module.insert_signal("AAPL", "Best quarter ever", 0.9, "POSITIVE", "vader")
        assert len(db_module.fetch_signals("AAPL")) == 1

    def test_duplicate_returns_false(self):
        """insert_signal returns False when the duplicate INSERT OR IGNORE fires."""
        db_module.insert_signal("AAPL", "Duplicate headline", 0.5, "NEUTRAL", "vader")
        second = db_module.insert_signal("AAPL", "Duplicate headline", 0.5, "NEUTRAL", "vader")
        assert second is False

    def test_same_headline_different_ticker_allowed(self):
        """Same text, different ticker → two independent rows, not a duplicate."""
        db_module.insert_signal("AAPL", "Tech stocks rally", 0.7, "POSITIVE", "vader")
        db_module.insert_signal("MSFT", "Tech stocks rally", 0.7, "POSITIVE", "vader")
        assert len(db_module.fetch_signals("AAPL")) == 1
        assert len(db_module.fetch_signals("MSFT")) == 1

    def test_case_insensitive_dedup_via_hash(self):
        """_make_hash lowercases so casing variants resolve to the same hash."""
        db_module.insert_signal("AAPL", "Apple Hits All-Time High", 0.8, "POSITIVE", "vader")
        # different casing — should still be considered 'seen'
        assert db_module.headline_seen("AAPL", "apple hits all-time high") is True

    def test_headline_seen_false_for_unknown(self):
        result = db_module.headline_seen("AAPL", "This headline was never stored")
        assert result is False

    def test_headline_seen_true_after_insert(self):
        headline = "Massive buyback announced"
        db_module.insert_signal("MSFT", headline, 0.75, "POSITIVE", "vader")
        assert db_module.headline_seen("MSFT", headline) is True
