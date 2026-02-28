"""
modules/portfolio.py
─────────────────────
Paper-trading portfolio layer.

Manages open/close positions in a SQLite `portfolio` table and provides
a P&L calculator that compares entry prices against live market prices.

All DB access goes through the shared `get_connection()` context manager
from modules.database to keep transaction handling consistent.
"""

from __future__ import annotations

import logging
import sqlite3

from modules.database import get_connection

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_PORTFOLIO_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL UNIQUE,
    quantity         REAL    NOT NULL,
    entry_price      REAL    NOT NULL,
    entry_timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure_table() -> None:
    """Create the portfolio table if it doesn't already exist."""
    with get_connection() as conn:
        conn.executescript(_PORTFOLIO_SCHEMA)


# ── Position management ───────────────────────────────────────────────────────

def open_position(ticker: str, quantity: float, entry_price: float) -> None:
    """
    Open a new paper-trading position, or update it if one already exists
    for the same ticker (upsert semantics).

    Parameters
    ----------
    ticker      : str   e.g. "AAPL"
    quantity    : float number of shares (e.g. 10)
    entry_price : float price per share at entry
    """
    _ensure_table()
    sql = """
        INSERT INTO portfolio (ticker, quantity, entry_price)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            quantity        = excluded.quantity,
            entry_price     = excluded.entry_price,
            entry_timestamp = CURRENT_TIMESTAMP
    """
    with get_connection() as conn:
        conn.execute(sql, (ticker, quantity, entry_price))
    logger.info("Position opened: %s  qty=%.0f  entry=$%.2f", ticker, quantity, entry_price)


def close_position(ticker: str) -> None:
    """
    Close (delete) a paper-trading position for *ticker*.
    No-op if no position exists.
    """
    _ensure_table()
    with get_connection() as conn:
        conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
    logger.info("Position closed: %s", ticker)


def get_portfolio() -> list[sqlite3.Row]:
    """Return all open positions ordered by entry date (newest first)."""
    _ensure_table()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolio ORDER BY entry_timestamp DESC"
        ).fetchall()
    return rows


# ── P&L calculation ───────────────────────────────────────────────────────────

def calculate_pnl(current_prices: dict[str, float]) -> list[dict]:
    """
    Calculate unrealised P&L for every open position.

    Parameters
    ----------
    current_prices : dict[str, float]
        Mapping of {ticker: current_market_price} from yfinance.
        Tickers not in this dict are skipped.

    Returns
    -------
    list[dict]
        One dict per position with keys:
        ticker, quantity, entry_price, current_price,
        portfolio_value, pnl, pnl_pct
    """
    rows = get_portfolio()
    results: list[dict] = []

    for row in rows:
        ticker = row["ticker"]
        quantity = float(row["quantity"])
        entry_price = float(row["entry_price"])
        current = current_prices.get(ticker)

        if current is None:
            logger.debug("No current price for %s — skipping P&L row.", ticker)
            continue

        pnl = (current - entry_price) * quantity
        pnl_pct = ((current - entry_price) / entry_price * 100) if entry_price else 0.0
        portfolio_value = current * quantity

        def _r2(v: float) -> float:
            return int(v * 100) / 100

        results.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "entry_price": _r2(entry_price),
                "current_price": _r2(current),
                "portfolio_value": _r2(portfolio_value),
                "pnl": _r2(pnl),
                "pnl_pct": _r2(pnl_pct),
            }
        )

    return results
