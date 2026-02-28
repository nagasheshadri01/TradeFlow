"""
modules/database.py
────────────────────
Thin SQLite helper built on the standard-library sqlite3 module.
Provides table creation and basic CRUD helpers used by agents and the dashboard.

v2 — Deduplication:
  - headline_hash column on sentiment_signals (SHA-256 of ticker+headline text)
  - UNIQUE constraint on (ticker, headline_hash) prevents re-processing the same
    headline across multiple watcher runs.
  - headline_seen() helper lets callers check before scoring.
  - insert_signal() now uses INSERT OR IGNORE and returns a bool.

v3 — Paper Trade Simulator:
  - positions table  : one row per open paper position (UNIQUE ticker).
  - trade_history table: immutable ledger of all closed trades with realised P&L.
  - upsert_position() / delete_position() / get_position() helpers.
  - insert_trade_history() / fetch_trade_history() helpers.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield an open SQLite connection; commits on success, rolls back on error.

    Thread-safety notes
    -------------------
    check_same_thread=False
        Required because Streamlit runs each user session in a separate thread
        while sharing the same Python process.  SQLite itself is thread-safe
        when each thread opens its own connection (which we do — this factory
        creates a fresh connection per call and closes it on exit).

    timeout=30
        If the DB file is locked (e.g. Watcher writing while dashboard reads),
        SQLite will retry for up to 30 seconds before raising OperationalError.
        This prevents spurious crashes during concurrent access.

    WAL journal mode
        Write-Ahead Logging allows concurrent readers while a write is in
        progress.  Set once per connection; SQLite persists the mode on disk
        so subsequent connections inherit it automatically.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,   # safe: one connection per call, closed on exit
        timeout=30,                # wait up to 30 s on a locked DB
    )
    conn.row_factory = sqlite3.Row
    # WAL mode: readers don't block writers and writers don't block readers
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    headline       TEXT    NOT NULL,
    headline_hash  TEXT    NOT NULL,
    score          REAL    NOT NULL,
    label          TEXT    NOT NULL,
    engine         TEXT    NOT NULL,
    source         TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, headline_hash)          -- deduplication constraint
);

CREATE TABLE IF NOT EXISTS trade_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    action      TEXT    NOT NULL,   -- BUY | SELL | HOLD
    reasoning   TEXT,
    avg_score   REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper Trade Simulator: one row per open position (UNIQUE ticker)
CREATE TABLE IF NOT EXISTS positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL UNIQUE,
    quantity         REAL    NOT NULL,
    entry_price      REAL    NOT NULL,
    total_cost       REAL    NOT NULL,      -- quantity * entry_price
    entry_timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Immutable ledger of closed paper trades with realised P&L
CREATE TABLE IF NOT EXISTS trade_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL,
    action           TEXT    NOT NULL,   -- BUY | SELL
    quantity         REAL    NOT NULL,
    entry_price      REAL    NOT NULL,
    exit_price       REAL,               -- NULL for BUY records
    total_cost       REAL    NOT NULL,
    realised_pnl     REAL,               -- NULL for BUY records
    realised_pnl_pct REAL,               -- NULL for BUY records
    confidence_score REAL,               -- risk engine confidence at trade time
    csi_score        REAL,               -- adjusted CSI at trade time
    trade_timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ─── Migrations ───────────────────────────────────────────────────────────────
# Idempotent ALTER TABLE statements to patch existing DBs.

def _migrate_db(conn: sqlite3.Connection) -> None:
    """Apply all pending schema migrations (safe to run on every startup)."""
    # v2: headline_hash on sentiment_signals
    cols_sig = {row[1] for row in conn.execute("PRAGMA table_info(sentiment_signals)")}
    if "headline_hash" not in cols_sig:
        try:
            conn.execute(
                "ALTER TABLE sentiment_signals ADD COLUMN headline_hash TEXT NOT NULL DEFAULT ''"
            )
            logger.info("Migration v2: added headline_hash to sentiment_signals.")
        except sqlite3.OperationalError as exc:
            logger.warning("Migration v2 skipped: %s", exc)

    # v3: total_cost on positions (if table was created without it)
    try:
        cols_pos = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
        if "total_cost" not in cols_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN total_cost REAL NOT NULL DEFAULT 0")
            logger.info("Migration v3: added total_cost to positions.")
    except sqlite3.OperationalError:
        pass   # table doesn't exist yet — SCHEMA_SQL will create it


def init_db() -> None:
    """Create all tables and run any pending migrations."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate_db(conn)
    logger.info("Database initialised at %s", DB_PATH)


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _make_hash(ticker: str, headline: str) -> str:
    """
    Deterministic SHA-256 hash of (ticker + headline text).

    Lower-casing and stripping normalise minor whitespace differences
    so near-identical headlines from different sources are collapsed.
    """
    key = f"{ticker.upper()}::{headline.strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def headline_seen(ticker: str, headline: str) -> bool:
    """
    Return True if this headline has already been processed for *ticker*.

    Used by the watcher and the agent to skip already-scored headlines
    without hitting the NLP engine.
    """
    h = _make_hash(ticker, headline)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM sentiment_signals WHERE ticker=? AND headline_hash=? LIMIT 1",
            (ticker, h),
        ).fetchone()
    return row is not None


# ── Write helpers ─────────────────────────────────────────────────────────────

def insert_signal(
    ticker: str,
    headline: str,
    score: float,
    label: str,
    engine: str,
    source: str | None = None,
) -> bool:
    """
    Insert a scored headline into the database.

    Uses INSERT OR IGNORE so duplicate (ticker, headline_hash) pairs are
    silently skipped — no exception, no duplicate rows.

    Returns
    -------
    bool
        True  – row was inserted (new headline)
        False – row was ignored  (duplicate)
    """
    h = _make_hash(ticker, headline)
    sql = """
        INSERT OR IGNORE INTO sentiment_signals
            (ticker, headline, headline_hash, score, label, engine, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (ticker, headline, h, score, label, engine, source))
        inserted = cursor.rowcount > 0
    if not inserted:
        logger.debug("[%s] Duplicate headline skipped: %.60s…", ticker, headline)
    return inserted


def insert_decision(
    ticker: str,
    action: str,
    reasoning: str = "",
    avg_score: float = 0.0,
) -> None:
    sql = """
        INSERT INTO trade_decisions (ticker, action, reasoning, avg_score)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (ticker, action, reasoning, avg_score))


# ── Signal / Decision read helpers ───────────────────────────────────────────

def fetch_signals(ticker: str, limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sentiment_signals WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return rows


def fetch_decisions(ticker: str, limit: int = 20) -> list[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trade_decisions WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return rows


def fetch_latest_scores(tickers: list[str]) -> dict[str, float]:
    """
    Return the most recent avg_score from trade_decisions for each ticker.
    """
    result: dict[str, float] = {}
    with get_connection() as conn:
        for ticker in tickers:
            row = conn.execute(
                "SELECT avg_score FROM trade_decisions WHERE ticker=? ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if row is not None:
                result[ticker] = float(row["avg_score"])
    return result


# ── Position helpers (Paper Trade Simulator) ──────────────────────────────────

def upsert_position(
    ticker:      str,
    quantity:    float,
    entry_price: float,
    total_cost:  float,
) -> None:
    """
    Insert or replace a paper position for *ticker*.
    Uses UPSERT so a re-entered position overwrites the old one.
    """
    sql = """
        INSERT INTO positions (ticker, quantity, entry_price, total_cost)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            quantity        = excluded.quantity,
            entry_price     = excluded.entry_price,
            total_cost      = excluded.total_cost,
            entry_timestamp = CURRENT_TIMESTAMP
    """
    with get_connection() as conn:
        conn.execute(sql, (ticker, quantity, entry_price, total_cost))
    logger.info(
        "Position upserted: %s  qty=%.4f  entry=$%.4f  cost=$%.2f",
        ticker, quantity, entry_price, total_cost,
    )


def delete_position(ticker: str) -> None:
    """Remove the open position for *ticker*. No-op if none exists."""
    with get_connection() as conn:
        conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
    logger.info("Position deleted: %s", ticker)


def get_position(ticker: str) -> sqlite3.Row | None:
    """Return the open position row for *ticker*, or None if not held."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM positions WHERE ticker = ? LIMIT 1", (ticker,)
        ).fetchone()


def get_all_positions() -> list[sqlite3.Row]:
    """Return all open positions ordered by entry date (newest first)."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM positions ORDER BY entry_timestamp DESC"
        ).fetchall()


# ── Trade history helpers ─────────────────────────────────────────────────────

def insert_trade_history(
    ticker:           str,
    action:           str,          # "BUY" | "SELL"
    quantity:         float,
    entry_price:      float,
    total_cost:       float,
    exit_price:       float | None  = None,
    realised_pnl:     float | None  = None,
    realised_pnl_pct: float | None  = None,
    confidence_score: float | None  = None,
    csi_score:        float | None  = None,
) -> None:
    """
    Append a trade record to the immutable trade_history ledger.
    Call on both BUY (exit_price=None) and SELL (exit_price set).
    """
    sql = """
        INSERT INTO trade_history
            (ticker, action, quantity, entry_price, exit_price,
             total_cost, realised_pnl, realised_pnl_pct,
             confidence_score, csi_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            ticker, action, quantity, entry_price, exit_price,
            total_cost, realised_pnl, realised_pnl_pct,
            confidence_score, csi_score,
        ))
    logger.info(
        "Trade history: %s %s  qty=%.4f  pnl=%s",
        action, ticker, quantity,
        f"${realised_pnl:+.2f}" if realised_pnl is not None else "N/A",
    )


def fetch_trade_history(ticker: str | None = None, limit: int = 100) -> list[sqlite3.Row]:
    """
    Return trade history rows.
    If *ticker* is given, filter to that symbol; otherwise return all.
    """
    with get_connection() as conn:
        if ticker:
            return conn.execute(
                "SELECT * FROM trade_history WHERE ticker=? ORDER BY trade_timestamp DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM trade_history ORDER BY trade_timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_unrealized_pnl() -> list[dict]:
    """
    Calculate unrealised P&L for all open positions using live market prices.

    Fetches each ticker's current price via yfinance and computes:
        unrealized_pnl     = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price / entry_price - 1) * 100

    Returns
    -------
    list[dict]  One dict per open position with keys:
        ticker, quantity, entry_price, total_cost, current_price,
        market_value, unrealized_pnl, unrealized_pnl_pct
        Positions whose price cannot be fetched are skipped.
    """
    from modules.data_fetcher import fetch_current_price  # local import avoids circular dep

    positions = get_all_positions()
    results: list[dict] = []

    for row in positions:
        ticker      = row["ticker"]
        quantity    = float(row["quantity"])
        entry_price = float(row["entry_price"])
        total_cost  = float(row["total_cost"])

        current = fetch_current_price(ticker)
        if current is None:
            logger.debug("No live price for %s — skipping unrealised P&L.", ticker)
            continue

        unreal_pnl     = (current - entry_price) * quantity
        unreal_pnl_pct = ((current - entry_price) / entry_price * 100) if entry_price else 0.0
        market_value   = current * quantity

        def _r2(v: float) -> float:
            return int(v * 100) / 100

        results.append({
            "ticker":             ticker,
            "quantity":           quantity,
            "entry_price":        _r2(entry_price),
            "total_cost":         _r2(total_cost),
            "current_price":      _r2(current),
            "market_value":       _r2(market_value),
            "unrealized_pnl":     _r2(unreal_pnl),
            "unrealized_pnl_pct": _r2(unreal_pnl_pct),
        })

    return results


def get_total_realized_pnl() -> float:
    """
    Sum all realised_pnl values from SELL records in trade_history.

    Returns 0.0 if no SELL trades have been recorded yet.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(realised_pnl), 0.0) as total FROM trade_history WHERE action='SELL'"
        ).fetchone()
    return float(row["total"]) if row else 0.0

