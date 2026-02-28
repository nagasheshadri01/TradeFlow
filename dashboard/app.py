"""
dashboard/app.py
─────────────────
Sentiment Trader — Command Center Dashboard

Sections:
  1.  Command Center (KPI row): Realized P&L, Portfolio Value, System Status
  2.  Risk Meter: Annualized Volatility + Confidence Score from last agent run
  3.  Sentiment Heatmap: All tracked tickers
  4.  Ticker Deep-Dive: Price chart, live info, agent analysis
  5.  Open Positions: Real-time unrealised P&L table
  6.  Audit Trail: Full trade history, newest first
  7.  Sidebar records: Recent signals + past decisions

Run with (from project root):
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.sentiment_agent import run_agent
from config.settings import DEFAULT_TICKERS
from modules.data_fetcher import (
    fetch_current_price,
    fetch_headlines,
    fetch_price_history,
    fetch_ticker_info,
    get_volatility,
)
from modules.database import (
    fetch_decisions,
    fetch_latest_scores,
    fetch_signals,
    fetch_trade_history,
    get_all_positions,
    get_total_realized_pnl,
    get_unrealized_pnl,
    init_db,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Trader | Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    transition: transform .15s, box-shadow .15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Section headers */
h2, h3 { letter-spacing: -0.02em; }

/* Risk badge */
.risk-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 1rem;
    margin: 4px 8px 4px 0;
}
.risk-green  { background: rgba(34,197,94,0.15);  border: 1px solid #22c55e; color: #22c55e; }
.risk-yellow { background: rgba(234,179,8,0.15);  border: 1px solid #eab308; color: #eab308; }
.risk-orange { background: rgba(249,115,22,0.15); border: 1px solid #f97316; color: #f97316; }
.risk-red    { background: rgba(239,68,68,0.15);  border: 1px solid #ef4444; color: #ef4444; }

/* Divider */
.section-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# Initialise DB on first load
init_db()

# ── Persistent session state for risk metrics from last agent run ─────────────
if "last_volatility"   not in st.session_state: st.session_state.last_volatility   = None
if "last_confidence"   not in st.session_state: st.session_state.last_confidence   = None
if "last_csi"          not in st.session_state: st.session_state.last_csi          = None
if "last_adj_csi"      not in st.session_state: st.session_state.last_adj_csi      = None
if "last_run_ticker"   not in st.session_state: st.session_state.last_run_ticker   = None
if "last_run_time"     not in st.session_state: st.session_state.last_run_time     = None


# ── Clean terminology mappings ────────────────────────────────────────────────
# Used everywhere in the dashboard to replace internal code strings with
# professional financial language that's clear to non-technical viewers.
UI_MAPPINGS = {
    # ── Analysis engines ──
    "hybrid_bert":        "Neural Analysis",
    "hybrid":             "Neural Analysis",
    "vader_only_neutral": "Standard Filter",
    "vader":              "Performance Engine",
    "distilbert":         "Accuracy Engine",
    # ── Trade signals ──
    "BUY":                "Bullish Entry",
    "SELL":               "Bearish Exit",
    "HOLD":               "Neutral/Wait",
    "NONE":               "None",
    # ── Data sources ──
    "Mock":               "Historical Scenario Data",
}



# ── Helpers ───────────────────────────────────────────────────────────────────
_PID_FILE = root_dir / "logs" / "watcher.pid"

def _is_watcher_running() -> bool:
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=3,
        )
        return str(pid) in r.stdout
    except Exception:
        return False

def _pnl_color(val: float) -> str:
    if val > 0:   return "#22c55e"
    if val < 0:   return "#ef4444"
    return "#94a3b8"

def _confidence_badge(conf: float | None, vol: float | None) -> str:
    if conf is None:
        return '<span class="risk-badge risk-yellow">No data yet</span>'
    vol_pct = f"{vol * 100:.1f}%" if vol is not None else "N/A"
    if conf == 0.0:
        return (f'<span class="risk-badge risk-red">EXTREME RISK — vol={vol_pct} '
                f'| Confidence=0% | Signal VETOED</span>')
    if conf <= 0.5:
        return (f'<span class="risk-badge risk-orange">HIGH RISK — vol={vol_pct} '
                f'| Confidence={conf*100:.0f}% | Signal Halved</span>')
    if conf < 1.0:
        return (f'<span class="risk-badge risk-yellow">MEDIUM RISK — vol={vol_pct} '
                f'| Confidence={conf*100:.0f}%</span>')
    return (f'<span class="risk-badge risk-green">LOW RISK — vol={vol_pct} '
            f'| Confidence=100% | Full Signal</span>')


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Sentiment Trader")
    st.caption("Command Center · MVP")
    st.markdown("---")

    # ── System status ──
    is_running = _is_watcher_running()
    dot   = "🟢" if is_running else "🔴"
    label = "Running" if is_running else "Stopped"
    st.markdown(f"**Watcher** &nbsp; {dot} {label}")
    if not is_running:
        st.caption("Start: `python agents/watcher.py`")

    st.markdown("---")

    ticker = st.text_input(
        "Ticker Symbol",
        value=DEFAULT_TICKERS[0] if DEFAULT_TICKERS else "AAPL",
        placeholder="e.g. AAPL",
    ).strip().upper()

    mode = st.radio(
        "Analysis Mode",
        ["⚡ Performance (VADER)", "🧠 Hybrid (VADER+BERT)", "🎯 Accuracy (DistilBERT)"],
        index=0,
        help=(
            "**Performance** — VADER only, instant.  \n\n"
            "**Hybrid** — VADER gatekeeper; DistilBERT only on strong signals (~40% faster than full BERT).  \n\n"
            "**Accuracy** — DistilBERT every headline."
        ),
    )
    engine = "vader" if "VADER" in mode and "Hybrid" not in mode else (
        "hybrid" if "Hybrid" in mode else "distilbert"
    )
    st.caption(f"Engine: `{engine}`")

    st.markdown("---")
    run_btn = st.button("▶ Analyze Ticker", use_container_width=True, type="primary")
    refresh_btn = st.button("🔄 Refresh Dashboard", use_container_width=True)

    st.markdown("---")

    # ── Sidebar: signals + decisions for current ticker ──
    if ticker:
        with st.expander("📡 Recent Signals", expanded=False):
            sig_rows = fetch_signals(ticker, limit=15)
            if sig_rows:
                df_sig = pd.DataFrame([dict(r) for r in sig_rows])
                # Map engine and source to UI-friendly terms
                if "engine" in df_sig.columns:
                    df_sig["engine"] = df_sig["engine"].apply(lambda x: UI_MAPPINGS.get(x, x))
                if "source" in df_sig.columns:
                    df_sig["source"] = df_sig["source"].apply(lambda x: UI_MAPPINGS.get(x, x))
                
                # Render relevant columns
                display_cols = [c for c in ["headline", "score", "label", "engine", "source", "created_at"] if c in df_sig.columns]
                st.dataframe(df_sig[display_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No signals for this ticker yet.")

        with st.expander("📋 Past Decisions", expanded=False):
            dec_rows = fetch_decisions(ticker, limit=10)
            if dec_rows:
                df_dec = pd.DataFrame([dict(r) for r in dec_rows])
                if "action" in df_dec.columns:
                    df_dec["action"] = df_dec["action"].apply(lambda x: UI_MAPPINGS.get(x, x))
                
                display_cols = [c for c in ["action", "avg_score", "reasoning", "created_at"] if c in df_dec.columns]
                st.dataframe(df_dec[display_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No decisions for this ticker yet.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 Sentiment Trader · Command Center")
st.caption("Automated sentiment-driven paper trading — risk-adjusted, time-weighted signals")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · COMMAND CENTER  (Realized P&L · Portfolio Value · System Status)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Command Center")

realized_pnl   = get_total_realized_pnl()
unrealized_data = get_unrealized_pnl()

total_market_value  = sum(p["market_value"]   for p in unrealized_data)
total_unrealized_pnl = sum(p["unrealized_pnl"] for p in unrealized_data)
total_cost_basis    = sum(p["total_cost"]      for p in unrealized_data)
total_pnl_pct = (
    ((total_market_value - total_cost_basis) / total_cost_basis * 100)
    if total_cost_basis > 0 else 0.0
)
combined_pnl = realized_pnl + total_unrealized_pnl

last_run_str = (
    st.session_state.last_run_time.strftime("%H:%M:%S UTC")
    if st.session_state.last_run_time else "Never"
)

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="💰 Total Realized P&L",
        value=f"${realized_pnl:+,.2f}",
        delta=f"${combined_pnl:+,.2f} incl. unrealised" if unrealized_data else None,
        delta_color="normal",
    )

with kpi2:
    n_pos          = len(unrealized_data)
    winners        = sum(1 for p in unrealized_data if p["unrealized_pnl"] > 0)
    losers         = sum(1 for p in unrealized_data if p["unrealized_pnl"] < 0)
    st.metric(
        label="📊 Portfolio Value (Live)",
        value=f"${total_market_value:,.2f}" if n_pos else "$0.00",
        delta=f"{n_pos} positions  {winners}W / {losers}L  ({total_pnl_pct:+.1f}%)" if n_pos else "No open positions",
        delta_color="normal" if total_pnl_pct >= 0 else "inverse",
    )

with kpi3:
    watcher_status = f"{'Running' if is_running else 'Stopped'}"
    last_ticker_label = (
        f"Last: {st.session_state.last_run_ticker}  @{last_run_str}"
        if st.session_state.last_run_ticker else "No analysis run yet"
    )
    st.metric(
        label="⚙️ System Status",
        value=f"Watcher {watcher_status}",
        delta=last_ticker_label,
        delta_color="off",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · RISK METER  (Volatility + Confidence from last agent run)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(
    "⚠️ Risk Meter",
    help="Volatility and Confidence metrics are calculated by the Risk Engine to determine if market conditions are stable enough for the Sentiment Signal."
)

risk_col, csi_col = st.columns([2, 1])

with risk_col:
    badge_html = _confidence_badge(
        st.session_state.last_confidence,
        st.session_state.last_volatility,
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    if st.session_state.last_volatility is not None:
        vol_pct = st.session_state.last_volatility * 100
        conf    = st.session_state.last_confidence
        st.caption(
            f"Ticker: **{st.session_state.last_run_ticker}** · "
            f"Annualised σ = **{vol_pct:.1f}%** · "
            f"Confidence = **{conf*100:.0f}%** · "
            f"Engine: `{engine}`"
        )
    else:
        st.caption("Run **▶ Analyze Ticker** to populate risk metrics.")

with csi_col:
    if st.session_state.last_csi is not None:
        raw  = st.session_state.last_csi
        adj  = st.session_state.last_adj_csi
        conf = st.session_state.last_confidence
        st.metric("Raw CSI", f"{raw:+.4f}",
                  help="Combined Sentiment Index — the time-decay sum of all recent headline scores.")
        st.metric("Adjusted CSI", f"{adj:+.4f}",
                  delta=f"×{conf:.2f} confidence" if conf is not None else None,
                  delta_color="off",
                  help="The final trade signal after the Risk Engine multiplies Raw CSI by Confidence.")
    else:
        st.metric("CSI", "—")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · SENTIMENT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🌡️ Sentiment Heatmap — Tracked Tickers")

latest_scores = fetch_latest_scores(DEFAULT_TICKERS)

if latest_scores:
    sorted_items = sorted(latest_scores.items(), key=lambda kv: kv[1])
    bar_tickers  = [t for t, _ in sorted_items]
    bar_scores   = [s for _, s in sorted_items]
    bar_colors   = [
        "#ef4444" if s <= -0.15 else
        "#f97316" if s <   0.0  else
        "#eab308" if s <   0.15 else
        "#22c55e"
        for s in bar_scores
    ]

    fig_heat = go.Figure(go.Bar(
        x=bar_scores, y=bar_tickers,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:+.4f}" for s in bar_scores],
        textposition="outside",
        cliponaxis=False,
    ))
    fig_heat.update_layout(
        height=max(160, len(bar_tickers) * 54 + 60),
        xaxis={
            "title": "Adjusted CSI (latest)", "range": [-1.3, 1.3],
            "zeroline": True, "zerolinecolor": "#555", "zerolinewidth": 2,
            "gridcolor": "rgba(255,255,255,0.05)",
        },
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
        margin={"l": 10, "r": 90, "t": 10, "b": 40},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"size": 13},
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No sentiment scores yet. Analyze a ticker to populate the heatmap.", icon="ℹ️")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · TICKER DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to continue.")
    st.stop()

info     = fetch_ticker_info(ticker)
live_px  = fetch_current_price(ticker)
mc       = info.get("market_cap")

st.subheader(f"🔍 {info.get('name', ticker)}  ({ticker})")

mc_col, sec_col, cur_col, price_col = st.columns(4)
mc_col.metric("Market Cap",  f"${mc / 1e9:.1f}B" if mc else "N/A")
sec_col.metric("Sector",     info.get("sector", "N/A"))
cur_col.metric("Currency",   info.get("currency", "USD"))
price_col.metric("Live Price", f"${live_px:,.2f}" if live_px else "N/A")

st.markdown(f"**Real-Time Market Momentum: {info.get('name', ticker)} ({ticker})**")
st.caption(f"Last live update: {datetime.now().strftime('%H:%M:%S')}  ·  Hourly OHLCV  ·  Last 30 trading days")
with st.spinner("Fetching price data…"):
    price_df = fetch_price_history(ticker)

if not price_df.empty:
    # Flatten MultiIndex columns from yfinance if present
    if hasattr(price_df.columns, "get_level_values"):
        try:
            price_df.columns = price_df.columns.get_level_values(0)
        except Exception:
            pass

    company_name = info.get("name", ticker)
    fig_c = go.Figure(data=[go.Candlestick(
        x=price_df.index.astype(str),   # str labels → categorical axis → zero gaps
        open=price_df["Open"],  high=price_df["High"],
        low=price_df["Low"],    close=price_df["Close"],
        name=ticker,
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="rgba(34,197,94,0.15)",
        decreasing_fillcolor="rgba(239,68,68,0.15)",
    )])

    # ── The "category" type is the true gap-free fix:
    # Each x-value is a discrete string label — Plotly has zero calendar
    # concept and therefore zero physical space for weekend / overnight gaps.
    fig_c.update_xaxes(
        type="category",            # kills ALL gaps (weekends + after-hours)
        nticks=10,                  # keeps labels readable, not crowded
        rangeslider_visible=False,  # hide the bulky bottom slider
        showgrid=False,             # no vertical grid lines on dark bg
        zeroline=False,
        tickangle=-30,
        tickfont={"color": "#94a3b8", "size": 11},
    )
    fig_c.update_layout(
        title={
            "text": f"Real-Time Market Momentum: {company_name} ({ticker})",
            "font": {"size": 15, "color": "#e2e8f0"},
            "x": 0, "xanchor": "left",
        },
        height=420,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        # Fully transparent — blends with Streamlit dark theme
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        yaxis={
            "gridcolor": "rgba(255,255,255,0.06)",
            "tickprefix": "$",
            "tickformat": ",.2f",
            "tickfont": {"color": "#94a3b8", "size": 11},
            "zeroline": False,
        },
    )
    st.plotly_chart(fig_c, use_container_width=True)
else:
    st.warning("No price data available for this ticker.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT RUN  (triggered by "▶ Analyze Ticker")
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner(f"Fetching headlines for **{ticker}**…"):
        headlines = fetch_headlines(ticker)

    if not headlines:
        st.warning(f"No headlines found for **{ticker}**.")
    else:
        with st.spinner(f"Running sentiment pipeline ({engine})…"):
            result = run_agent(ticker, headlines, engine=engine)

        # ── Persist risk metrics to session state for Risk Meter ──
        st.session_state.last_volatility  = result.get("volatility")
        st.session_state.last_confidence  = result.get("confidence_score")
        st.session_state.last_csi         = result.get("csi_score")
        st.session_state.last_adj_csi     = result.get("adjusted_csi")
        st.session_state.last_run_ticker  = ticker
        st.session_state.last_run_time    = datetime.now(timezone.utc)

        action   = result["decision"]
        pretty_action = UI_MAPPINGS.get(action, action)
        adj_csi  = result.get("adjusted_csi", result.get("avg_score", 0.0))
        sim_act  = result.get("sim_action", "NONE")
        sim_px   = result.get("sim_price", 0.0)
        sim_pnl  = result.get("sim_pnl")

        # ── Decision callout ──
        colour   = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(action, "gray")
        conf_lbl = (
            f"  ·  Confidence = {result.get('confidence_score', 1.0) * 100:.0f}%"
            if result.get("confidence_score") is not None else ""
        )
        vol_lbl  = (
            f"  ·  σ = {result.get('volatility', 0) * 100:.1f}%"
            if result.get("volatility") else ""
        )
        st.subheader("Analysis Results")
        st.markdown(
            f"### Decision: :{colour}[**{pretty_action}**]"
            f"  ·  Adjusted CSI = `{adj_csi:+.4f}`{conf_lbl}{vol_lbl}"
        )

        # ── Simulator feedback ──
        if sim_act == "BUY_EXECUTED":
            qty = result.get("sim_price", 0)
            total_cost = 10000  # DEFAULT_TRADE_SIZE_USD
            st.success(
                f"**[SIM] BUY executed** — "
                f"${total_cost:,.0f} worth of {ticker} @ ${sim_px:,.2f}",
                icon="✅",
            )
        elif sim_act == "SELL_EXECUTED":
            pnl_str = f"  ·  P&L: **${sim_pnl:+,.2f}**" if sim_pnl is not None else ""
            pnl_color = "green" if (sim_pnl or 0) >= 0 else "red"
            st.success(
                f"**[SIM] SELL executed** — {ticker} closed @ ${sim_px:,.2f}{pnl_str}",
                icon="✅",
            )
        elif sim_act in ("SKIPPED_ALREADY_HELD", "SKIPPED_NOT_HELD"):
            st.info(
                f"[SIM] {pretty_action} signal — no trade executed "
                f"({'already held' if 'HELD' in sim_act else 'not in portfolio'})",
                icon="ℹ️",
            )
        elif "SKIPPED_NO_PRICE" in (sim_act or ""):
            st.warning("[SIM] Could not fetch live price — trade skipped.", icon="⚠️")

        # ── Headline scores table ──
        if result.get("signals"):
            st.markdown("#### Headline Scores")
            sig_df = pd.DataFrame(result["signals"])
            if "engine" in sig_df.columns:
                sig_df["engine"] = sig_df["engine"].apply(lambda x: UI_MAPPINGS.get(x, x))
            if "source" in sig_df.columns:
                sig_df["source"] = sig_df["source"].apply(lambda x: UI_MAPPINGS.get(x, x))
            
            display_cols = [c for c in ["headline", "score", "label", "engine", "source"] if c in sig_df.columns]
            st.dataframe(sig_df[display_cols], use_container_width=True, hide_index=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · OPEN POSITIONS  (real-time unrealised P&L)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💼 Open Positions")

pos_data = get_unrealized_pnl()

if pos_data:
    df_pos = pd.DataFrame(pos_data).rename(columns={
        "ticker":             "Ticker",
        "quantity":           "Shares",
        "entry_price":        "Entry $",
        "total_cost":         "Cost $",
        "current_price":      "Current $",
        "market_value":       "Market Val $",
        "unrealized_pnl":     "Unreal. P&L $",
        "unrealized_pnl_pct": "Unreal. P&L %",
    })

    # Colour-code the P&L column
    def _style_pnl(val):
        color = "#22c55e" if val > 0 else "#ef4444" if val < 0 else "#94a3b8"
        return f"color: {color}; font-weight: 600;"

    styled = df_pos.style.applymap(_style_pnl, subset=["Unreal. P&L $", "Unreal. P&L %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary bar
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Total Cost Basis",  f"${total_cost_basis:,.2f}")
    sum_col2.metric("Total Market Value", f"${total_market_value:,.2f}")
    pnl_sign = "+" if total_unrealized_pnl >= 0 else ""
    sum_col3.metric(
        "Total Unrealised P&L",
        f"${pnl_sign}{total_unrealized_pnl:,.2f}",
        delta=f"{total_pnl_pct:+.2f}%",
        delta_color="normal" if total_pnl_pct >= 0 else "inverse",
    )
else:
    st.info(
        "No open positions. Run **▶ Analyze Ticker** — a BUY decision will "
        "open a paper position automatically.",
        icon="ℹ️",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · AUDIT TRAIL  (complete trade_history, newest first)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Audit Trail — Full Trade History")

history_rows = fetch_trade_history(limit=200)

if history_rows:
    df_hist = pd.DataFrame([dict(r) for r in history_rows])

    # Choose and rename columns that are always present
    base_cols  = ["trade_timestamp", "ticker", "action", "quantity",
                   "entry_price", "exit_price", "total_cost",
                   "realised_pnl", "realised_pnl_pct",
                   "confidence_score", "csi_score"]
    avail_cols = [c for c in base_cols if c in df_hist.columns]
    df_hist    = df_hist[avail_cols].rename(columns={
        "trade_timestamp":  "Timestamp",
        "ticker":           "Ticker",
        "action":           "Action",
        "quantity":         "Shares",
        "entry_price":      "Entry $",
        "exit_price":       "Exit $",
        "total_cost":       "Cost $",
        "realised_pnl":     "Realised P&L $",
        "realised_pnl_pct": "P&L %",
        "confidence_score": "Confidence",
        "csi_score":        "Adj. CSI",
    })

    def _style_action(val):
        if val in ("BUY", UI_MAPPINGS["BUY"]):
            return "color: #22c55e; font-weight:700"
        if val in ("SELL", UI_MAPPINGS["SELL"]):
            return "color: #ef4444; font-weight:700"
        return ""

    def _style_pnl_hist(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "color: #64748b"
        return f"color: {'#22c55e' if val > 0 else '#ef4444' if val < 0 else '#94a3b8'}; font-weight:600"

    # Map the Action column to UI_MAPPINGS before rendering
    if "Action" in df_hist.columns:
        df_hist["Action"] = df_hist["Action"].apply(lambda x: UI_MAPPINGS.get(x, x))

    styled_hist = df_hist.style
    if "Action" in df_hist.columns:
        styled_hist = styled_hist.applymap(_style_action, subset=["Action"])
    pnl_cols = [c for c in ["Realised P&L $", "P&L %"] if c in df_hist.columns]
    if pnl_cols:
        styled_hist = styled_hist.applymap(_style_pnl_hist, subset=pnl_cols)

    st.dataframe(styled_hist, use_container_width=True, hide_index=True)

    # Realised P&L summary line
    total_r = sum(
        float(dict(r).get("realised_pnl", 0))
        for r in history_rows
        if r["action"] == "SELL"
    )
    trades_closed = sum(1 for r in history_rows if r["action"] == "SELL")
    sign = "+" if total_r >= 0 else ""
    st.caption(
        f"**{len(history_rows)} trade records**  ·  "
        f"**{trades_closed} closed trades**  ·  "
        f"Total Realised P&L: **${sign}{total_r:,.2f}**"
    )
else:
    st.info(
        "No trade history yet. The Audit Trail populates automatically "
        "as the simulator executes BUY/SELL orders.",
        icon="ℹ️",
    )
