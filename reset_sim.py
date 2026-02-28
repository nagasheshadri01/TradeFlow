"""
reset_sim.py
────────────
Paper Trading Season Reset

Clears all simulated positions and trade history so you can start a
fresh "Paper Trading Season" without touching the sentiment signals
or past decisions.

What it deletes
───────────────
    positions     – all open paper positions
    trade_history – the full BUY/SELL audit trail

What it KEEPS
─────────────
    sentiment_signals  – scored headlines (analysis history)
    trade_decisions    – BUY/HOLD/SELL decisions log
    All other tables

Usage
─────
    python reset_sim.py              # prompts for confirmation
    python reset_sim.py --yes        # skip confirmation (for scripting)
    python reset_sim.py --dry-run    # show what would be deleted without deleting
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reset_sim")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count(conn, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return -1  # table may not exist yet


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset paper trading tables (positions + trade_history)."
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show row counts that WOULD be deleted, but do nothing.",
    )
    args = parser.parse_args()

    from modules.database import get_connection

    # ── Preview ──
    with get_connection() as conn:
        tables = {
            "positions":     _count(conn, "positions")     if _table_exists(conn, "positions")     else 0,
            "trade_history": _count(conn, "trade_history") if _table_exists(conn, "trade_history") else 0,
        }

    log.info("=" * 58)
    log.info("  Paper Trading Season Reset")
    log.info("=" * 58)
    log.info("  The following tables will be TRUNCATED:")
    for table, n in tables.items():
        log.info("    %-20s  %d rows", table, n)
    log.info("")
    log.info("  These tables will NOT be touched:")
    log.info("    %-20s  (sentiment analysis history)", "sentiment_signals")
    log.info("    %-20s  (BUY/HOLD/SELL decisions log)", "trade_decisions")
    log.info("=" * 58)

    if args.dry_run:
        log.info("  DRY RUN \u2014 no changes made.")
        sys.exit(0)

    # ── Confirmation ──
    if not args.yes:
        print()
        answer = input("  Type 'yes' to confirm reset, anything else to cancel: ").strip().lower()
        if answer != "yes":
            log.info("  Reset cancelled.")
            sys.exit(0)

    # ── Execute ──
    log.info("")
    log.info("  Resetting...")
    with get_connection() as conn:
        if _table_exists(conn, "positions"):
            conn.execute("DELETE FROM positions")
            log.info("  [OK] positions cleared   (%d rows removed)", tables["positions"])
        else:
            log.info("  [--] positions table does not exist yet \u2014 nothing to clear.")

        if _table_exists(conn, "trade_history"):
            conn.execute("DELETE FROM trade_history")
            log.info("  [OK] trade_history cleared (%d rows removed)", tables["trade_history"])
        else:
            log.info("  [--] trade_history table does not exist yet \u2014 nothing to clear.")

    log.info("")
    log.info("  Reset complete. Ready for a new Paper Trading Season.")
    log.info("  Run the Watcher or click 'Analyze Ticker' in the dashboard to begin.")
    log.info("=" * 58)


if __name__ == "__main__":
    main()
