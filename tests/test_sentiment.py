"""
tests/test_sentiment.py
────────────────────────
Basic sanity checks for the sentiment_analyzer module.
Run with:  pytest tests/
"""

import pytest

from modules.sentiment_analyzer import analyze, vader_score


class TestVaderScore:
    def test_positive_text(self):
        score = vader_score("The stock soared to an all-time high – outstanding performance!")
        assert score > 0, "Expected a positive VADER score"

    def test_negative_text(self):
        score = vader_score("The company collapsed and filed for bankruptcy.")
        assert score < 0, "Expected a negative VADER score"

    def test_neutral_text(self):
        score = vader_score("The market opened today.")
        assert -0.5 < score < 0.5, "Expected a near-neutral VADER score"

    def test_returns_float(self):
        assert isinstance(vader_score("test"), float)


class TestAnalyze:
    def test_analyze_returns_dict(self):
        result = analyze("Great earnings report!", engine="vader")
        assert isinstance(result, dict)
        assert "score" in result
        assert "label" in result
        assert "engine" in result

    def test_label_is_valid(self):
        result = analyze("Terrible losses reported.", engine="vader")
        assert result["label"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")

    def test_engine_fallback(self):
        # DistilBERT may not be available in CI; must not raise
        result = analyze("Stock surged after earnings.", engine="distilbert")
        assert "score" in result
