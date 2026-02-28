"""
config/settings.py
──────────────────
Central settings loader for the fully-local Sentiment Trader.
No external LLM or API keys required.
All modules should import constants from here instead of reading env vars directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Resolve the project root (one level above config/) ────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root; override=False keeps real env vars intact.
load_dotenv(ROOT_DIR / ".env", override=False)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require(key: str) -> str:
    """Return the value of a required environment variable or raise."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env file."
        )
    return value


# ── Application ───────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: Path = ROOT_DIR / os.getenv("DB_PATH", "data/sentiment_trader.db")

# ── Model (fully local) ───────────────────────────────────────────────────────
# HuggingFace model ID — used only during setup_local.py download.
HF_MODEL_NAME: str = os.getenv(
    "HF_MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
# Absolute path to the locally-cached model weights written by setup_local.py.
# Falls back to HF_MODEL_NAME so the pipeline uses the HF cache if the local
# directory hasn't been created yet.
_local_model_rel = os.getenv("LOCAL_MODEL_DIR", "models/distilbert-sst2")
LOCAL_MODEL_DIR: Path = ROOT_DIR / _local_model_rel
MODEL_PATH: str = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_NAME

# ── Trading ───────────────────────────────────────────────────────────────────
DEFAULT_TICKERS: list[str] = [
    t.strip()
    for t in os.getenv("DEFAULT_TICKERS", "AAPL,MSFT,GOOGL").split(",")
    if t.strip()
]
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "30"))

# ── Hybrid sentiment engine ───────────────────────────────────────────────────
# VADER compound scores in (-THRESHOLD, +THRESHOLD) are considered ambiguous.
# analyze_with_filter() short-circuits as NEUTRAL for these — skipping the
# expensive DistilBERT transformer call entirely.
# Set via .env:  HYBRID_VADER_THRESHOLD=0.15
HYBRID_VADER_THRESHOLD: float = float(os.getenv("HYBRID_VADER_THRESHOLD", "0.1"))

# Label thresholds applied to the final raw score (after VADER or DistilBERT).
# These are intentionally separate from HYBRID_VADER_THRESHOLD so you can tune
# them independently (e.g. tighter gating but same label boundaries).
SENTIMENT_POSITIVE_THRESHOLD: float = float(os.getenv("SENTIMENT_POSITIVE_THRESHOLD", "0.05"))
SENTIMENT_NEGATIVE_THRESHOLD: float = float(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", "-0.05"))

# ── CSI — Composite Sentiment Index ──────────────────────────────────────────
# Exponential decay applied to each signal score based on headline age:
#   S_weighted = Σ ( score_i · exp(-λ · Δt_hours_i) )
#
# DECAY_LAMBDA (λ): controls how fast old news loses influence.
#   λ = 0.5  → a 1-hour-old headline retains  exp(-0.5)  ≈ 60 % weight
#   λ = 0.5  → a 4-hour-old headline retains  exp(-2.0)  ≈ 14 % weight
#   λ = 0.5  → a 24-hour-old headline retains exp(-12.0) ≈  0 % weight
#   Increase λ to make the agent more reactive to fresh news.
#   Decrease λ to smooth over a longer time window.
# Set via .env:  DECAY_LAMBDA=0.3
DECAY_LAMBDA: float = float(os.getenv("DECAY_LAMBDA", "0.5"))

# Decision thresholds applied to the final CSI score (weighted sum, NOT mean).
# These intentionally differ from HYBRID_VADER_THRESHOLD — CSI is a sum, not
# a per-signal average, so its scale grows with headline count.
# Set via .env:  CSI_BUY_THRESHOLD=0.2
CSI_BUY_THRESHOLD:  float = float(os.getenv("CSI_BUY_THRESHOLD",  "0.15"))
CSI_SELL_THRESHOLD: float = float(os.getenv("CSI_SELL_THRESHOLD", "-0.15"))

# ── Risk Engine — Volatility-Adjusted Confidence ──────────────────────────────
# Annualised volatility (σ) of daily log-returns is compared against regime
# thresholds to produce a Confidence Score in [0.0, 1.0]:
#
#   σ < VOLATILITY_LOW_THRESHOLD   → Confidence = 1.0  (green light)
#   σ < VOLATILITY_HIGH_THRESHOLD  → Confidence linearly interpolated
#   σ < MAX_VOLATILITY_THRESHOLD   → Confidence = 0.5  (yellow — halve signal)
#   σ ≥ MAX_VOLATILITY_THRESHOLD   → Confidence = 0.0  (red — force HOLD)
#
# Annualised σ = daily_std × √252  (252 trading days)
#
# Typical reference values:
#   S&P 500 long-run average   ≈ 15–20 %
#   High-growth single stocks  ≈ 40–60 %
#   Crypto / meme stocks       ≈ 80–150%
#
# Set via .env:  VOLATILITY_LOOKBACK=30  MAX_VOLATILITY_THRESHOLD=1.0
VOLATILITY_LOOKBACK: int = int(os.getenv("VOLATILITY_LOOKBACK", "30"))

# Regime thresholds (as decimals: 0.20 = 20 %)
VOLATILITY_LOW_THRESHOLD:     float = float(os.getenv("VOLATILITY_LOW_THRESHOLD",     "0.20"))
VOLATILITY_HIGH_THRESHOLD:    float = float(os.getenv("VOLATILITY_HIGH_THRESHOLD",    "0.60"))
MAX_VOLATILITY_THRESHOLD:     float = float(os.getenv("MAX_VOLATILITY_THRESHOLD",     "1.00"))

# Minimum confidence when volatility is high-but-not-extreme (yellow zone)
VOLATILITY_MIN_CONFIDENCE:    float = float(os.getenv("VOLATILITY_MIN_CONFIDENCE",    "0.50"))

# ── Paper Trade Simulator ─────────────────────────────────────────────────────
# Fixed USD amount allocated to each paper BUY order.
# Shares purchased = DEFAULT_TRADE_SIZE_USD / current_price
# Set via .env:  DEFAULT_TRADE_SIZE_USD=5000
DEFAULT_TRADE_SIZE_USD: float = float(os.getenv("DEFAULT_TRADE_SIZE_USD", "10000.0"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.debug("Settings loaded (APP_ENV=%s)", APP_ENV)
