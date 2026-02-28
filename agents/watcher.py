"""
agents/watcher.py
──────────────────
Background scheduler that runs a full sentiment analysis cycle over all
DEFAULT_TICKERS once per hour and auto-manages paper trading positions.

Run with (from project root):
    python agents/watcher.py

Stop with Ctrl+C.  Logs are written to logs/watcher.log.
A PID file is maintained at logs/watcher.pid so the dashboard can show
live system-status without any extra dependencies.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# ── Bootstrap project root ────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import DEFAULT_TICKERS, LOG_LEVEL
from modules.data_fetcher import fetch_headlines, fetch_current_price
from modules.portfolio import open_position, close_position, get_portfolio
from agents.sentiment_agent import run_agent

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = root_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = LOG_DIR / "watcher.pid"

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.FileHandler(LOG_DIR / "watcher.log", encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)

logger = logging.getLogger("watcher")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False  # don't bubble up to root logger


# ── Core cycle ────────────────────────────────────────────────────────────────

def run_once(engine: str = "vader") -> None:
    """
    Execute one full analysis cycle across all DEFAULT_TICKERS.

    For each ticker:
      1. Fetch latest news headlines.
      2. Run the LangGraph sentiment agent.
      3. If BUY and no open position  → open 10-share paper position.
      4. If SELL and position exists  → close paper position.
      5. A 1-second delay between tickers to avoid yfinance rate limits.
    """
    logger.info("═══ Watcher cycle started  (engine=%s, tickers=%s) ═══", engine, DEFAULT_TICKERS)
    portfolio_map = {row["ticker"]: row for row in get_portfolio()}

    for ticker in DEFAULT_TICKERS:
        try:
            logger.info("[%s] ── Fetching headlines…", ticker)
            headlines = fetch_headlines(ticker)

            if not headlines:
                logger.warning("[%s] No headlines found — skipping.", ticker)
                continue

            result = run_agent(ticker, headlines, engine=engine)
            decision = result["decision"]
            avg_score = result["avg_score"]
            logger.info("[%s] Decision=%-4s  avg_score=%+.4f", ticker, decision, avg_score)

            current_price = fetch_current_price(ticker)

            if decision == "BUY" and ticker not in portfolio_map:
                if current_price:
                    open_position(ticker, quantity=10, entry_price=current_price)
                    logger.info("[%s] ✅  Opened 10 shares @ $%.2f", ticker, current_price)
                else:
                    logger.warning("[%s] BUY signal but could not fetch price — skipped.", ticker)

            elif decision == "SELL" and ticker in portfolio_map:
                close_position(ticker)
                logger.info("[%s] ❌  Closed position (SELL signal)", ticker)

            else:
                logger.info("[%s] HOLD — no position change.", ticker)

        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] ‼  Unhandled error: %s", ticker, exc, exc_info=True)

        finally:
            time.sleep(1)  # Rate-limit: 1 s between tickers

    # Refresh portfolio snapshot for next iteration
    logger.info("═══ Cycle complete ═══\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def _write_pid() -> None:
    PID_FILE.write_text(str(os.getpid()))
    logger.debug("PID %d written to %s", os.getpid(), PID_FILE)


def _remove_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> None:
    _write_pid()
    logger.info("Watcher service started (PID=%d).  Tickers: %s", os.getpid(), DEFAULT_TICKERS)
    logger.info("Interval: 3600 s (1 h).  Logs: %s", LOG_DIR / "watcher.log")

    try:
        while True:
            run_once()
            logger.info("Sleeping for 3600 s until next cycle…")
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user (KeyboardInterrupt).")
    finally:
        _remove_pid()


if __name__ == "__main__":
    main()
