# 📈 Stock Market Sentiment Agentic Trader — MVP

A minimal, end-to-end agentic trading assistant that scores news headlines with
VADER / DistilBERT sentiment, stores signals in SQLite, produces BUY / HOLD / SELL
decisions via a LangGraph agent, and displays everything in a Streamlit dashboard.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Dashboard | Streamlit + Plotly |
| Agentic Orchestration | LangGraph |
| Fast Sentiment | NLTK VADER |
| Deep Sentiment | HuggingFace DistilBERT (local) |
| Market Data | yfinance |
| Storage | SQLite (stdlib `sqlite3`) |
| Data Manipulation | pandas + numpy |
| Config | python-dotenv |

---

## Folder Structure

```
sentiment_trader/
├── agents/
│   ├── __init__.py
│   └── sentiment_agent.py   # LangGraph graph: score → aggregate → decide
├── config/
│   ├── __init__.py
│   └── settings.py          # dotenv-backed settings
├── dashboard/
│   ├── __init__.py
│   └── app.py               # Streamlit entry-point
├── data/                    # SQLite DB lives here (git-ignored)
├── modules/
│   ├── __init__.py
│   ├── data_fetcher.py      # yfinance wrapper
│   ├── database.py          # SQLite CRUD helpers
│   └── sentiment_analyzer.py# VADER + DistilBERT dual engine
├── tests/
│   ├── __init__.py
│   ├── test_database.py
│   └── test_sentiment.py
├── .env                     # secrets & config (not committed)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone & enter the project

```bash
git clone <https://github.com/nagasheshadri01/TradeFlow.git>
cd sentiment_trader
```

### 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip (important — avoids wheel build failures)

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** remove the `--index-url` suffix from the `torch` line in
> `requirements.txt` and reinstall to get CUDA support.

### 5. Configure environment variables

Edit `.env` in the project root:

```dotenv
HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
LOCAL_MODEL_DIR=models/distilbert-sst2
DEFAULT_TICKERS=AAPL,MSFT,GOOGL,TSLA,AMZN
LOOKBACK_DAYS=30
DB_PATH=data/sentiment_trader.db
```

No external API keys are required.

### 6. Run the one-time local setup *(requires internet, do once)*

```bash
python setup_local.py
```

This will:
- Download **DistilBERT** weights → `models/distilbert-sst2/`
- Download the **NLTK VADER** lexicon
- Smoke-test both assets

After this step **no internet connection is needed**.

Optional flags:
```bash
python setup_local.py --skip-model   # VADER only
python setup_local.py --skip-vader   # model only
python setup_local.py --model-dir /path/to/custom/dir
```

---

## Running the Dashboard

```bash
# From the sentiment_trader/ directory
streamlit run dashboard/app.py
```

The app will open at **http://localhost:8501**.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Usage

1. Select a **ticker** from the sidebar.
2. Paste **news headlines** (one per line) into the text area.
3. Choose a sentiment **engine** (VADER is faster; DistilBERT is more accurate).
4. Click **▶ Run Agent**.
5. The agent scores each headline, aggregates the scores, and displays a
   **BUY / HOLD / SELL** decision with the underlying signal table.

---

## Roadmap (post-MVP)

- [ ] Automated headline ingestion via NewsAPI
- [ ] Real brokerage integration (Alpaca / Interactive Brokers)
- [ ] Portfolio tracking and P&L simulation
- [ ] Multi-agent debate pattern (bull vs bear)
- [ ] Scheduled background runs (cron / Celery)
- [ ] Docker containerisation

---

## License

MIT
