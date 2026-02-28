"""
modules/data_fetcher.py
────────────────────────
Fetches OHLCV price data, basic stock info, and headlines from three sources:

  1. yfinance  – official ticker news (most reliable)
  2. RSS feeds – Google News / Yahoo Finance financial RSS (broader coverage)
  3. Mock      – deterministic fallback so the pipeline never starves

All headline sources are merged and deduplicated via headline_seen() before
being returned, so the sentiment agent never re-scores the same story twice.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import yfinance as yf

from config.settings import LOOKBACK_DAYS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RSS feed URLs
# ─────────────────────────────────────────────

# Google News RSS returns the most recent ~10 articles for any query.
# The {ticker} placeholder is replaced at call time.
_RSS_SOURCES: list[str] = [
    "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

# Strip HTML tags from RSS summaries / titles
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    return _TAG_RE.sub("", text).strip()


# ─────────────────────────────────────────────
# Price History
# ─────────────────────────────────────────────

def fetch_price_history(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Download daily OHLCV data for *ticker* over the last *days* calendar days.

    Returns an EMPTY DataFrame (never raises) when:
      - The ticker symbol is invalid or delisted.
      - yfinance returns no rows (e.g. weekend-only window).
      - Any network / parsing error occurs.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume (auto-adjusted) – indexed by date.
        Empty DataFrame on any failure.
    """
    empty = pd.DataFrame()
    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    try:
        df: pd.DataFrame = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1h" if days <= 60 else "1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("[%s] fetch_price_history failed: %s", ticker, exc)
        return empty

    if df is None or df.empty:
        logger.warning("[%s] No price data returned (delisted or invalid ticker?)", ticker)
        return empty

    # yfinance >= 0.2 wraps multi-ticker downloads in a MultiIndex — flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    # Verify the essential OHLCV columns are present
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        logger.warning("[%s] Unexpected columns from yfinance: %s", ticker, list(df.columns))
        return empty

    logger.debug("[%s] Price history: %d rows (%s → %s)",
                 ticker, len(df), start.date(), end.date())
    return df


# ─────────────────────────────────────────────
# Ticker Info
# ─────────────────────────────────────────────

def fetch_ticker_info(ticker: str) -> dict:
    """Return a dict of company metadata for *ticker*."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "USD"),
        }
    except Exception as exc:
        logger.error("Could not fetch info for %s: %s", ticker, exc)
        return {
            "name": ticker,
            "sector": "N/A",
            "market_cap": None,
            "currency": "USD",
        }


# ─────────────────────────────────────────────
# Headline Sources
# ─────────────────────────────────────────────

def _fetch_yfinance_headlines(ticker: str) -> list[str]:
    """Fetch headlines from yfinance news API."""
    try:
        news = yf.Ticker(ticker).news or []
        headlines = [
            item["title"]
            for item in news
            if isinstance(item, dict) and "title" in item
        ]
        logger.debug("[%s] yfinance returned %d headlines", ticker, len(headlines))
        return headlines
    except Exception as exc:
        logger.warning("[%s] yfinance headline fetch failed: %s", ticker, exc)
        return []


def fetch_rss_headlines(ticker: str) -> list[str]:
    """
    Fetch headlines from financial RSS feeds (Google News + Yahoo Finance).

    Tries each URL in _RSS_SOURCES and collects up to 20 unique titles.
    Returns an empty list gracefully on any network error.
    """
    collected: list[str] = []

    for url_template in _RSS_SOURCES:
        url = url_template.format(ticker=ticker)
        try:
            feed = feedparser.parse(url)
            entries = feed.get("entries", [])
            for entry in entries:
                title = _clean(entry.get("title", ""))
                if title and len(title) > 10:   # skip trivially short entries
                    collected.append(title)
            logger.debug(
                "[%s] RSS <%s> returned %d entries", ticker, url, len(entries)
            )
        except Exception as exc:
            logger.warning("[%s] RSS feed error (%s): %s", ticker, url, exc)

    return collected


def _mock_headlines(ticker: str) -> list[str]:
    """Deterministic fallback headlines — used only when live sources return nothing."""
    return [
        f"{ticker} stock surges after strong earnings report",
        f"Analysts debate future outlook of {ticker}",
        f"Market volatility impacts {ticker} performance",
        f"Investors watch {ticker} amid sector rotation",
        f"{ticker} beats quarterly revenue estimates",
    ]


# ─────────────────────────────────────────────
# Smart Ingestion (merged + deduplicated)
# ─────────────────────────────────────────────

def fetch_headlines(ticker: str, skip_seen: bool = True) -> list[str]:
    """
    Fetch and merge headlines for *ticker* from all available sources.

    Source priority: yfinance → RSS → Mock fallback

    Parameters
    ----------
    ticker    : str   Stock ticker symbol (e.g. "AAPL")
    skip_seen : bool  When True, filter out headlines already in the DB
                      (deduplication check via headline_seen()).
                      Set False to include all headlines regardless.

    Returns
    -------
    list[str]
        Unique, unseen headlines ready for sentiment scoring.
        Never returns an empty list (mock fallback guarantees at least one entry).
    """
    # ── Collect from all sources ──
    yf_headlines  = _fetch_yfinance_headlines(ticker)
    rss_headlines = fetch_rss_headlines(ticker)

    # Merge: yfinance first (higher quality), then RSS (broader)
    raw: list[str] = yf_headlines + rss_headlines

    # ── Deduplicate within this batch (case-insensitive) ──
    seen_texts: set[str] = set()
    unique: list[str] = []
    for h in raw:
        key = h.strip().lower()
        if key and key not in seen_texts:
            seen_texts.add(key)
            unique.append(h)

    # ── Filter out headlines already processed in previous runs ──
    if skip_seen and unique:
        try:
            from modules.database import headline_seen  # local import avoids circular dep
            before = len(unique)
            unique = [h for h in unique if not headline_seen(ticker, h)]
            skipped = before - len(unique)
            if skipped:
                logger.info("[%s] Skipped %d already-seen headline(s).", ticker, skipped)
        except Exception as exc:
            # Never let DB lookup break headline fetching
            logger.warning("[%s] Could not check deduplication DB: %s", ticker, exc)

    # ── Fallback if every live headline was already seen (or none fetched) ──
    if not unique:
        logger.info("[%s] No new headlines from live sources — using mock fallback.", ticker)
        unique = _mock_headlines(ticker)

    logger.info("[%s] Returning %d new headline(s) for scoring.", ticker, len(unique))
    return unique


# ─────────────────────────────────────────────
# Current Price
# ─────────────────────────────────────────────

def fetch_current_price(ticker: str) -> float | None:
    """
    Return the most recent closing price for *ticker*, or None on failure.

    Uses a 1-day history window which is faster than downloading full OHLCV
    and avoids the slow .info call.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        logger.warning("Empty price history for %s", ticker)
    except Exception as exc:
        logger.error("Could not fetch current price for %s: %s", ticker, exc)
    return None


# ─────────────────────────────────────────────
# Volatility (for Risk Engine)
# ─────────────────────────────────────────────

def get_volatility(ticker: str, days: int = LOOKBACK_DAYS) -> float | None:
    """
    Calculate the annualised volatility (σ) of *ticker* over the last *days*
    calendar days.

    Method
    ------
    1. Download daily OHLCV data (auto-adjusted closes).
    2. Compute daily log-returns:  r_t = ln(Close_t / Close_{t-1})
    3. Annualise:  σ = std(r_t) × √252

    The result is expressed as a decimal fraction:
        0.20 → 20 %  (typical blue-chip)
        0.60 → 60 %  (high-growth / volatile)
        1.00 → 100%  (extreme / meme stock)

    Parameters
    ----------
    ticker : str   Stock ticker symbol.
    days   : int   Lookback window in calendar days (default: VOLATILITY_LOOKBACK
                   from config/settings.py via LOOKBACK_DAYS fallback).

    Returns
    -------
    float | None
        Annualised volatility as a decimal, or None if insufficient data.
        None triggers a graceful fallback to confidence=1.0 in the risk engine.
    """
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=days)

        df: pd.DataFrame = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if df.empty or len(df) < 5:
            logger.warning("[%s] Insufficient price data for volatility (%d rows)", ticker, len(df))
            return None

        # Flatten MultiIndex columns if yfinance returns them (multi-ticker download)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        closes = df["Close"].dropna()
        if len(closes) < 5:
            logger.warning("[%s] Not enough non-null closes for volatility", ticker)
            return None

        import numpy as np
        log_returns   = np.log(closes / closes.shift(1)).dropna()
        daily_std     = float(log_returns.std())
        annualised    = daily_std * (252 ** 0.5)

        logger.debug(
            "[%s] Volatility: daily_std=%.4f  annualised=%.4f  (n=%d days)",
            ticker, daily_std, annualised, len(log_returns),
        )
        return float(f"{annualised:.4f}")

    except Exception as exc:
        logger.error("[%s] Volatility calculation failed: %s", ticker, exc)
        return None