"""
agents/sentiment_agent.py
──────────────────────────
LangGraph agent pipeline:

  score_headlines → aggregate_csi → risk_assessment → decide → simulator_execution

Step 1 — score_headlines
    Score each headline with the chosen NLP engine. Stamp UTC timestamp.
    Persist to DB (INSERT OR IGNORE deduplication).

Step 2 — aggregate_csi
    Composite Sentiment Index: exponentially time-decayed weighted sum.
    CSI = Sigma( score_i * exp(-lambda * Dt_hours_i) )
    Fresh breaking news dominates; stale coverage fades automatically.

Step 3 — risk_assessment
    Cross-reference CSI with the stock's 30-day annualised volatility.
    Derive a Confidence Score [0.0, 1.0] and produce adjusted_csi.

Step 4 — decide
    BUY / HOLD / SELL based on adjusted_csi vs CSI thresholds.

Step 5 — simulator_execution  <- NEW
    Virtually execute the trade decision against the paper portfolio:
    BUY  -> fetch price, size position ($DEFAULT_TRADE_SIZE_USD / price),
             upsert into positions table, log BUY to trade_history.
    SELL -> read open position, compute realised P&L, delete from positions,
             log SELL + P&L to trade_history.
    HOLD -> no-op.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph

from config.settings import (
    CSI_BUY_THRESHOLD,
    CSI_SELL_THRESHOLD,
    DECAY_LAMBDA,
    DEFAULT_TRADE_SIZE_USD,
    MAX_VOLATILITY_THRESHOLD,
    VOLATILITY_HIGH_THRESHOLD,
    VOLATILITY_LOOKBACK,
    VOLATILITY_LOW_THRESHOLD,
    VOLATILITY_MIN_CONFIDENCE,
)
from modules.database import (
    delete_position,
    get_position,
    insert_decision,
    insert_signal,
    insert_trade_history,
    upsert_position,
)
from modules.sentiment_analyzer import analyze

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    ticker:           str
    headlines:        list[str]
    engine:           str           # "vader" | "distilbert" | "hybrid"
    signals:          list[dict]    # {headline, score, label, engine, timestamp}
    csi_score:        float         # raw Composite Sentiment Index
    avg_score:        float         # alias for adjusted_csi (dashboard compat)
    volatility:       float         # annualised sigma (decimal: 0.20 = 20 %)
    confidence_score: float         # risk engine output [0.0, 1.0]
    adjusted_csi:     float         # csi_score * confidence_score
    decision:         str           # "BUY" | "HOLD" | "SELL"
    sim_action:       str           # what the simulator actually did
    sim_price:        float         # price at which the sim trade executed
    sim_pnl:          float | None  # realised P&L on SELL (None for BUY/HOLD)


# ══════════════════════════════════════════════════════════════════════════════
# CSI calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_csi(
    signals: list[dict],
    decay_lambda: float = DECAY_LAMBDA,
    now: datetime | None = None,
) -> float:
    """
    Compute the Composite Sentiment Index (CSI) using exponential time decay.

        CSI = Σ ( score_i · exp(-λ · Δt_hours_i) )

    Parameters
    ----------
    signals      : list[dict]  Each dict needs "score" and optionally "timestamp".
    decay_lambda : float       Decay constant λ.
    now          : datetime    Reference time (injectable for testing).

    Returns
    -------
    float  CSI rounded to 4 decimal places.  0.0 for empty list.
    """
    if not signals:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    weighted_sum = 0.0
    total_weight = 0.0

    for sig in signals:
        score: float = float(sig.get("score", 0.0))
        ts_raw = sig.get("timestamp")

        if ts_raw is None:
            delta_hours = 0.0
        elif isinstance(ts_raw, datetime):
            ts = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
            delta_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        else:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                if not ts.tzinfo:
                    ts = ts.replace(tzinfo=timezone.utc)
                delta_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                logger.warning("Unparseable timestamp %r — treating as age=0", ts_raw)
                delta_hours = 0.0

        weight        = math.exp(-decay_lambda * delta_hours)
        weighted_sum += score * weight
        total_weight += weight
        logger.debug(
            "CSI  score=%+.4f  Δt=%.2fh  w=%.4f  contrib=%+.4f",
            score, delta_hours, weight, score * weight,
        )

    logger.debug("CSI=%.4f  Σw=%.4f", weighted_sum, total_weight)
    return round(weighted_sum, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Risk Engine
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(volatility: float | None) -> float:
    """
    Map annualised volatility σ → Confidence Score ∈ [0.0, 1.0].

    Regime table (defaults, all tunable via .env):
    ┌──────────────────────────────┬──────────────────────────────┬────────────┐
    │  Volatility                  │  Confidence                  │  Signal    │
    ├──────────────────────────────┼──────────────────────────────┼────────────┤
    │  σ < 20 %  (LOW)             │  1.0                         │  🟢 Full   │
    │  20 % ≤ σ < 60 % (MEDIUM)   │  linear 1.0 → 0.5            │  🟡 Scaled │
    │  60 % ≤ σ < 100% (HIGH)     │  0.5  (VOLATILITY_MIN_CONF)  │  🟡 Halved │
    │  σ ≥ 100% (EXTREME)         │  0.0  (force HOLD)           │  🔴 Veto   │
    └──────────────────────────────┴──────────────────────────────┴────────────┘

    Parameters
    ----------
    volatility : float | None
        Annualised σ as a decimal (0.20 = 20 %).
        None → data unavailable → assume low risk (confidence = 1.0).

    Returns
    -------
    float  Confidence score in [0.0, 1.0].
    """
    if volatility is None:
        logger.warning("Volatility unavailable — defaulting to confidence=1.0")
        return 1.0

    vol = volatility

    # [RED] Extreme volatility — veto regardless of sentiment
    if vol >= MAX_VOLATILITY_THRESHOLD:
        conf = 0.0

    # [YELLOW] High (but not extreme) — clamp to minimum confidence
    elif vol >= VOLATILITY_HIGH_THRESHOLD:
        conf = VOLATILITY_MIN_CONFIDENCE

    # [YELLOW] Medium — linear interpolation between 1.0 and VOLATILITY_MIN_CONFIDENCE
    elif vol >= VOLATILITY_LOW_THRESHOLD:
        span = VOLATILITY_HIGH_THRESHOLD - VOLATILITY_LOW_THRESHOLD
        frac = (vol - VOLATILITY_LOW_THRESHOLD) / span if span > 0 else 1.0
        conf = 1.0 - frac * (1.0 - VOLATILITY_MIN_CONFIDENCE)

    # [GREEN] Low — full confidence
    else:
        conf = 1.0

    return round(conf, 4)


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph Node functions
# ══════════════════════════════════════════════════════════════════════════════

def score_headlines(state: AgentState) -> AgentState:
    """Score every headline, stamp UTC timestamp, persist to DB."""
    signals: list[dict] = []

    for headline in state["headlines"]:
        scored_at = datetime.now(timezone.utc).isoformat()
        result    = analyze(headline, engine=state.get("engine", "vader"))
        result["headline"]  = headline
        result["timestamp"] = scored_at
        signals.append(result)

        insert_signal(
            ticker=state["ticker"],
            headline=headline,
            score=result["score"],
            label=result["label"],
            engine=result["engine"],
        )

    logger.debug("Scored %d headlines for %s", len(signals), state["ticker"])
    return {**state, "signals": signals}


def aggregate_csi(state: AgentState) -> AgentState:
    """Compute CSI with exponential time-decay weighting."""
    csi = calculate_csi(state["signals"])
    logger.info(
        "CSI for %s: %.4f  (λ=%.2f, n=%d)",
        state["ticker"], csi, DECAY_LAMBDA, len(state["signals"]),
    )
    return {**state, "csi_score": csi, "avg_score": csi}


def risk_assessment(state: AgentState) -> AgentState:
    """
    Fetch live volatility for the ticker and compute a Confidence Score.

    The confidence score is multiplied with the raw CSI to produce an
    adjusted_csi that the decide node uses for the final BUY/HOLD/SELL.

    Side effects: fetches ~30 days of yfinance OHLCV (cached by OS/network).
    Falls back gracefully to confidence=1.0 if data is unavailable.
    """
    from modules.data_fetcher import get_volatility  # local import: avoids circular

    vol  = get_volatility(state["ticker"], days=VOLATILITY_LOOKBACK)
    conf = compute_confidence(vol)
    raw  = state["csi_score"]
    adj  = round(raw * conf, 4)

    vol_pct  = f"{vol * 100:.1f}%" if vol is not None else "N/A"
    regime   = (
        "[RED]    Extreme (HOLD override)" if conf == 0.0 else
        "[YELLOW] High (signal halved)"    if conf <= VOLATILITY_MIN_CONFIDENCE else
        "[YELLOW] Medium (partial scale)"  if conf < 1.0 else
        "[GREEN]  Low (full confidence)"
    )

    logger.info(
        "Risk [%s]: σ=%s  regime=%s  confidence=%.2f  "
        "CSI=%+.4f → adjusted=%+.4f",
        state["ticker"], vol_pct, regime, conf, raw, adj,
    )
    return {
        **state,
        "volatility":       vol if vol is not None else 0.0,
        "confidence_score": conf,
        "adjusted_csi":     adj,
    }


def decide(state: AgentState) -> AgentState:
    """Translate risk-adjusted CSI into BUY / HOLD / SELL."""
    adj  = state["adjusted_csi"]
    raw  = state["csi_score"]
    conf = state["confidence_score"]
    vol  = state["volatility"]
    n    = len(state["signals"])
    vol_pct = f"{vol * 100:.1f}%"

    # Confidence = 0 → hard HOLD regardless of sentiment
    if conf == 0.0:
        action = "HOLD"
        reasoning = (
            f"HOLD forced by Risk Engine: extreme volatility σ={vol_pct} "
            f"(≥ {MAX_VOLATILITY_THRESHOLD * 100:.0f}%) exceeds safe threshold. "
            f"Raw CSI was {raw:+.4f} but was vetoed."
        )
    elif adj >= CSI_BUY_THRESHOLD:
        action = "BUY"
        reasoning = (
            f"BUY: adjusted_CSI={adj:+.4f} ≥ {CSI_BUY_THRESHOLD}. "
            f"Raw CSI={raw:+.4f}, σ={vol_pct}, confidence={conf:.2f} "
            f"({n} signals, λ={DECAY_LAMBDA})."
        )
    elif adj <= CSI_SELL_THRESHOLD:
        action = "SELL"
        reasoning = (
            f"SELL: adjusted_CSI={adj:+.4f} ≤ {CSI_SELL_THRESHOLD}. "
            f"Raw CSI={raw:+.4f}, σ={vol_pct}, confidence={conf:.2f} "
            f"({n} signals, λ={DECAY_LAMBDA})."
        )
    else:
        action = "HOLD"
        reasoning = (
            f"HOLD: adjusted_CSI={adj:+.4f} within "
            f"[{CSI_SELL_THRESHOLD}, {CSI_BUY_THRESHOLD}]. "
            f"Raw CSI={raw:+.4f}, σ={vol_pct}, confidence={conf:.2f} "
            f"({n} signals, λ={DECAY_LAMBDA})."
        )

    insert_decision(
        ticker=state["ticker"],
        action=action,
        reasoning=reasoning,
        avg_score=adj,          # store adjusted CSI as the authoritative score
    )
    logger.info("Decision for %s: %s  adj_csi=%.4f", state["ticker"], action, adj)
    return {**state, "decision": action, "avg_score": adj}


def simulator_execution(state: AgentState) -> AgentState:
    """
    Virtually execute the agent's decision against the paper portfolio.

    BUY
      - Fetch current market price.
      - Skip if position already held (no averaging-in).
      - Compute quantity = DEFAULT_TRADE_SIZE_USD / price.
      - Upsert into positions table, append BUY to trade_history.

    SELL
      - Fetch current market price.
      - Skip if no open position exists.
      - Calculate realised P&L: (exit - entry) * quantity.
      - Delete position, append SELL + P&L to trade_history.

    HOLD
      - No-op.
    """
    from modules.data_fetcher import fetch_current_price  # local import avoids circular dep

    decision = state["decision"]
    ticker   = state["ticker"]
    conf     = state["confidence_score"]
    adj_csi  = state["adjusted_csi"]

    sim_action: str        = "NONE"
    sim_price:  float      = 0.0
    sim_pnl:    float | None = None

    if decision == "BUY":
        existing = get_position(ticker)
        if existing is not None:
            logger.info(
                "[SIM] %s already in portfolio (qty=%.4f) -- skipping duplicate BUY.",
                ticker, float(existing["quantity"]),
            )
            sim_action = "SKIPPED_ALREADY_HELD"
        else:
            price = fetch_current_price(ticker)
            if price is None or price <= 0:
                logger.warning("[SIM] Could not fetch price for %s -- BUY skipped.", ticker)
                sim_action = "SKIPPED_NO_PRICE"
            else:
                quantity   = DEFAULT_TRADE_SIZE_USD / price
                total_cost = quantity * price

                upsert_position(
                    ticker=ticker,
                    quantity=quantity,
                    entry_price=price,
                    total_cost=total_cost,
                )
                insert_trade_history(
                    ticker=ticker,
                    action="BUY",
                    quantity=quantity,
                    entry_price=price,
                    total_cost=total_cost,
                    confidence_score=conf,
                    csi_score=adj_csi,
                )
                sim_action = "BUY_EXECUTED"
                sim_price  = price
                logger.info(
                    "[SIM] BUY  %s  qty=%.4f @ $%.4f  cost=$%.2f",
                    ticker, quantity, price, total_cost,
                )

    elif decision == "SELL":
        existing = get_position(ticker)
        if existing is None:
            logger.info("[SIM] %s not in portfolio -- nothing to SELL.", ticker)
            sim_action = "SKIPPED_NOT_HELD"
        else:
            exit_price = fetch_current_price(ticker)
            if exit_price is None or exit_price <= 0:
                logger.warning("[SIM] Could not fetch exit price for %s -- SELL skipped.", ticker)
                sim_action = "SKIPPED_NO_PRICE"
            else:
                quantity    = float(existing["quantity"])
                entry_price = float(existing["entry_price"])
                total_cost  = float(existing["total_cost"])

                realised_pnl     = (exit_price - entry_price) * quantity
                realised_pnl_pct = (
                    ((exit_price - entry_price) / entry_price * 100)
                    if entry_price else 0.0
                )

                delete_position(ticker)
                insert_trade_history(
                    ticker=ticker,
                    action="SELL",
                    quantity=quantity,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    total_cost=total_cost,
                    realised_pnl=realised_pnl,
                    realised_pnl_pct=realised_pnl_pct,
                    confidence_score=conf,
                    csi_score=adj_csi,
                )
                sim_action = "SELL_EXECUTED"
                sim_price  = exit_price
                sim_pnl    = realised_pnl
                logger.info(
                    "[SIM] SELL %s  qty=%.4f @ $%.4f  P&L=$%+.2f (%.1f%%)",
                    ticker, quantity, exit_price, realised_pnl, realised_pnl_pct,
                )
    else:
        sim_action = "HOLD_NO_OP"
        logger.debug("[SIM] HOLD %s -- no position change.", ticker)

    return {**state, "sim_action": sim_action, "sim_price": sim_price, "sim_pnl": sim_pnl}


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("score_headlines",    score_headlines)
    graph.add_node("aggregate_csi",      aggregate_csi)
    graph.add_node("risk_assessment",    risk_assessment)
    graph.add_node("decide",             decide)
    graph.add_node("simulator_execution", simulator_execution)  # <- NEW

    graph.set_entry_point("score_headlines")
    graph.add_edge("score_headlines",     "aggregate_csi")
    graph.add_edge("aggregate_csi",       "risk_assessment")
    graph.add_edge("risk_assessment",     "decide")
    graph.add_edge("decide",              "simulator_execution")  # <- NEW
    graph.add_edge("simulator_execution", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_app = None   # compiled graph — built once per process


def run_agent(
    ticker:    str,
    headlines: list[str],
    engine:    str = "vader",
) -> AgentState:
    """
    Run the full sentiment + risk pipeline for *ticker*.

    Parameters
    ----------
    ticker    : str         e.g. "AAPL"
    headlines : list[str]   Pre-deduplicated headlines.
    engine    : str         "vader" | "distilbert" | "hybrid"

    Returns
    -------
    AgentState with keys:
        signals          : scored headline dicts (with timestamps)
        csi_score        : raw Composite Sentiment Index
        volatility       : annualised σ (decimal)
        confidence_score : Risk Engine output [0.0, 1.0]
        adjusted_csi     : csi_score × confidence_score
        avg_score        : alias of adjusted_csi (dashboard compat)
        decision         : "BUY" | "HOLD" | "SELL"
    """
    global _app
    if _app is None:
        _app = build_graph()

    initial_state: AgentState = {
        "ticker":           ticker,
        "headlines":        headlines,
        "engine":           engine,
        "signals":          [],
        "csi_score":        0.0,
        "avg_score":        0.0,
        "volatility":       0.0,
        "confidence_score": 1.0,
        "adjusted_csi":     0.0,
        "decision":         "HOLD",
        "sim_action":       "NONE",
        "sim_price":        0.0,
        "sim_pnl":          None,
    }
    return _app.invoke(initial_state)
