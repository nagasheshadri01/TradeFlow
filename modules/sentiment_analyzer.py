"""
modules/sentiment_analyzer.py
──────────────────────────────
Three-tier sentiment scoring — fully local, no cloud API required.

  Tier 1 ── VADER (always runs)
      Fast rule-based scorer, <1 ms per headline.
      Used as both a standalone engine AND as the gatekeeper for DistilBERT.

  Tier 2 ── DistilBERT (runs only when VADER is ambiguous)
      Local transformer (distilbert-base-uncased-finetuned-sst-2-english).
      Loaded once into memory via @lru_cache; never reloaded between calls.
      CPU-only inference: ~50–200 ms per headline depending on hardware.

  Tier 3 ── Hybrid gating  ← NEW
      analyze_with_filter(text) runs VADER first.
      If |VADER compound| < HYBRID_VADER_THRESHOLD (default 0.1), the text is
      considered ambiguously neutral and we skip DistilBERT entirely
      (engine tag: "vader_only_neutral").
      If |VADER compound| ≥ threshold, DistilBERT is invoked to confirm or
      override (engine tag: "hybrid_bert").
      This cuts transformer calls by ~40–60 % on typical financial newsfeeds.

All scorers return a float in [-1.0, +1.0]:
  -1.0 = very negative  |  0.0 = neutral  |  +1.0 = very positive

Run `python setup_local.py` once to cache the model and VADER lexicon
before going offline.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config.settings import (
    HYBRID_VADER_THRESHOLD,
    MODEL_PATH,
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── Bootstrap VADER lexicon (once per process) ────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    logger.info("Downloading VADER lexicon …")
    nltk.download("vader_lexicon", quiet=True)


# ── Engine constants (returned in the "engine" field) ────────────────────────

ENGINE_VADER          = "vader"
ENGINE_DISTILBERT     = "distilbert"
ENGINE_HYBRID_BERT    = "hybrid_bert"         # VADER triggered → DistilBERT confirmed
ENGINE_VADER_NEUTRAL  = "vader_only_neutral"  # VADER below threshold → skipped BERT


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — VADER
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    """Singleton VADER analyser — loaded once, reused forever."""
    return SentimentIntensityAnalyzer()


def vader_score(text: str) -> float:
    """Return VADER compound score for *text* (range: -1 → +1)."""
    return _vader().polarity_scores(text)["compound"]


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — DistilBERT
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _distilbert_pipeline():
    """
    Singleton DistilBERT pipeline — loaded lazily on first use.

    @lru_cache(maxsize=1) guarantees the model weights are loaded exactly once
    per process and kept in RAM for all subsequent calls — no reload overhead.
    Returns None if the model cannot be loaded (graceful degradation).
    """
    try:
        from transformers import pipeline  # noqa: PLC0415

        logger.info("Loading DistilBERT model from: %s", MODEL_PATH)
        pipe = pipeline(
            "sentiment-analysis",
            model=MODEL_PATH,
            truncation=True,
            max_length=512,
        )
        logger.info("DistilBERT loaded and ready.")
        return pipe
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load DistilBERT: %s", exc)
        return None


def distilbert_score(text: str) -> float:
    """
    Return a DistilBERT sentiment score (range: -1 → +1).
    Falls back to VADER automatically if the model is unavailable.
    """
    pipe = _distilbert_pipeline()
    if pipe is None:
        logger.warning("DistilBERT unavailable — falling back to VADER.")
        return vader_score(text)

    result = pipe(text)[0]
    label: str  = result["label"].upper()  # "POSITIVE" or "NEGATIVE"
    score: float = result["score"]          # confidence in [0, 1]
    return score if label == "POSITIVE" else -score


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Hybrid gating  (VADER gatekeeper → selective DistilBERT)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_with_filter(
    text: str,
    threshold: float = HYBRID_VADER_THRESHOLD,
) -> dict:
    """
    Performance-optimised hybrid scorer.

    Algorithm
    ---------
    1. Score with VADER (always fast, <1 ms).
    2. If |vader_compound| < threshold → headline is ambiguously neutral.
       Short-circuit: return VADER score, engine="vader_only_neutral".
       DistilBERT is NOT invoked — saves ~50-200 ms per filtered headline.
    3. If |vader_compound| ≥ threshold → headline has a detectable signal.
       Invoke DistilBERT to confirm or refine.
       Return DistilBERT score, engine="hybrid_bert".
       Falls back to the raw VADER score if DistilBERT is unavailable.

    Parameters
    ----------
    text      : str    Headline or short text to score.
    threshold : float  |VADER compound| below this → skip DistilBERT.
                       Default: HYBRID_VADER_THRESHOLD from config/settings.py.

    Returns
    -------
    dict with keys:
        score  : float   Sentiment score in [-1.0, +1.0]
        label  : str     "POSITIVE" | "NEUTRAL" | "NEGATIVE"
        engine : str     Which engine produced the final score
        vader_score : float   Always present — the raw VADER pre-check value
    """
    v_score = vader_score(text)

    if abs(v_score) < threshold:
        # ── Gate: ambiguous → skip transformer ──
        raw    = v_score
        engine = ENGINE_VADER_NEUTRAL
        logger.debug(
            "Hybrid SKIP  |vader|=%.3f < %.3f  → neutral  | %.60s",
            abs(v_score), threshold, text,
        )
    else:
        # ── Gate: opinionated → run DistilBERT ──
        raw    = distilbert_score(text)
        engine = ENGINE_HYBRID_BERT
        logger.debug(
            "Hybrid BERT  |vader|=%.3f ≥ %.3f  → bert=%.3f | %.60s",
            abs(v_score), threshold, raw, text,
        )

    label = (
        "POSITIVE" if raw > SENTIMENT_POSITIVE_THRESHOLD
        else "NEGATIVE" if raw < SENTIMENT_NEGATIVE_THRESHOLD
        else "NEUTRAL"
    )

    return {
        "score":       round(raw, 4),
        "label":       label,
        "engine":      engine,
        "vader_score": round(v_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public unified API  (used by sentiment_agent.py)
# ══════════════════════════════════════════════════════════════════════════════

def analyze(text: str, engine: str = "vader") -> dict:
    """
    Score *text* with the chosen engine.

    Parameters
    ----------
    engine : str
        "vader"      – VADER only (fast, default)
        "distilbert" – DistilBERT only (accurate, slower)
        "hybrid"     – VADER gatekeeper → DistilBERT where needed (balanced)

    Returns
    -------
    dict  {score, label, engine, [vader_score]}
    """
    engine = engine.lower()

    if engine == "hybrid":
        return analyze_with_filter(text)

    if engine == "distilbert":
        raw = distilbert_score(text)
        eng = ENGINE_DISTILBERT
    else:
        raw = vader_score(text)
        eng = ENGINE_VADER

    label = (
        "POSITIVE" if raw > SENTIMENT_POSITIVE_THRESHOLD
        else "NEGATIVE" if raw < SENTIMENT_NEGATIVE_THRESHOLD
        else "NEUTRAL"
    )
    return {"score": round(raw, 4), "label": label, "engine": eng}
