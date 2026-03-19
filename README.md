# LZ-Quant — Dual Market Sentiment Trading Engine

![LZ-Quant Logo](logos/bgLZQ.png)

A real-time quantitative trading system that combines AI sentiment analysis with automated paper trading across both crypto and stock markets. Runs entirely on consumer hardware with zero API costs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-19-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-orange.svg)
![ONNX](https://img.shields.io/badge/ONNX-Inference-yellow.svg)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Data Sources](#data-sources)
- [Trading Engine](#trading-engine)
- [Dashboard](#dashboard)
- [CSV Logging](#csv-logging)
- [Contributing](#contributing)

---

## Features

### Sentiment Analysis
- **DistilBERT fine-tuned with LoRA** — 296K trainable params out of 66M total
- **ONNX export** for fast inference (~5ms GPU, ~55ms CPU)
- **3-class classification**: Bearish, Neutral, Bullish
- **Real text ingestion** from Reddit and RSS feeds (no API keys needed)

### Markets
- **Crypto**: Binance WebSocket (BTCUSDT, ETHUSDT, SOLUSDT)
- **Stocks**: Alpaca WebSocket (AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META, AMZN, SPY, QQQ)

### Trading
- **Divergence Detection** — Rolling Z-score analysis of sentiment vs price
- **Paper Trading** with full P&L tracking
- **Automated exits**: Stop-loss (1.5%), Take-profit (2.5%), Trailing stop (1.0%)
- **Max drawdown circuit breaker** (30%)
- **Pause/Resume** trading without losing state

### Dashboard
- **Real-time signal feed** with sentiment bars
- **Dual market panels** (crypto + stocks)
- **Equity curve** with reference line
- **Sentiment history** over time
- **Latency distribution** chart
- **Open positions** and **trade log**
- **Divergence alerts**
- **Pause/Resume controls**
- **Demo mode** when server is offline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LZ-Quant Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────────┐     ┌──────────────┐                        │
│    │   Reddit     │     │     RSS      │                        │
│    │  (10 subs)   │     │  (6 feeds)   │                        │
│    └──────┬───────┘     └──────┬───────┘                        │
│           │                    │                                │
│           └────────┬───────────┘                                │
│                    ▼                                            │
│          ┌────────────────┐                                     │
│          │  TextRouter    │                                     │
│          │ (Deduplication)│                                     │
│          └────────┬───────┘                                     │
│                   ▼                                             │
│    ┌──────────────┴──────────────┐                              │
│    │                             │                              │
│    ▼                             ▼                              │
│ ┌────────────┐           ┌────────────┐                         │
│ │  Binance   │           │   Alpaca   │                         │
│ │ WebSocket  │           │ WebSocket  │                         │
│ └─────┬──────┘           └─────┬──────┘                         │
│       │                         │                               │
│       └──────────┬──────────────┘                               │
│                  ▼                                              │
│         ┌───────────────┐                                       │
│         │   ONNX Model  │  ←──── SentimentEngine                │
│         │ (DistilBERT)  │                                       │
│         └───────┬───────┘                                       │
│                 ▼                                               │
│         ┌───────────────┐                                       │
│         │  Divergence   │                                       │
│         │   Detector    │                                       │
│         └───────┬───────┘                                       │
│                 ▼                                               │
│         ┌───────────────┐                                       │
│         │   Paper       │                                       │
│         │   Trading     │                                       │
│         └───────┬───────┘                                       │
│                 ▼                                               │
│    ┌────────────┼────────────┐                                  │
│    ▼            ▼            ▼                                  │
│ ┌──────┐   ┌─────────┐   ┌──────┐                               │
│ │ CSV  │   │Dashboard│   │Trade │                               │
│ │ Log  │   │(HTML/JS)│   │ Log  │                               │
│ └──────┘   └─────────┘   └──────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/AtharvVatsal/lz-quant.git
cd lz-quant
pip install -r requirements.txt
```

### 2. Download the Trained Model

The ONNX model (255MB) is too large for GitHub. Download it from releases:

```bash
# Download from GitHub Releases or HuggingFace
# Option A: From GitHub (if released)
# Option B: Direct download
curl -L -o output/finbert-lora/sentiment_model.onnx https://huggingface.co/YOUR_USERNAME/lz-quant/resolve/main/sentiment_model.onnx
```

> **Note**: Without the model, the system runs in **simulated mode** with random sentiment (for demo purposes only).

### 2. Start the Trading Engine

```bash
# Basic usage
python dual.py

# With custom capital
python dual.py --capacity 500000

# Full options
python dual.py --capacity 100000 --risk 25 --max-pos 6 --port 8765
```

### 3. Open Dashboard

- **HTML Dashboard**: http://localhost:8765
- **React Dashboard**: 
  ```bash
  cd lz-dashboard
  npm start
  ```

### 4. Configure Alpaca (Optional for Stocks)

Create `.env` file:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

Without Alpaca keys, stock trading is disabled but crypto continues.

---

## Installation Options

| Command | Description |
|---------|-------------|
| `pip install -e .` | Core runtime (server, WebSocket, API) |
| `pip install -e ".[inference]"` | + ONNX inference (real sentiment model) |
| `pip install -e ".[training]"` | + Model training dependencies |
| `pip install -e ".[visualization]"` | + Training visualization |
| `pip install -e ".[all]"` | All dependencies |
| `pip install -e ".[gpu]"` | GPU support for inference/training |

---

## Project Structure

```
LZ Quant/
├── dual.py                    # Main dual-market engine (recommended)
├── integration.py              # Crypto-only integration
├── server.py                  # Simple crypto-only server
├── dataIngestion.py           # Reddit & RSS text ingestion
├── divergenceTrading.py       # Divergence detection + paper trading
├── inference.py               # ONNX inference engine
├── pipeline.py                # Binance WebSocket pipeline
├── train.py                   # Model training script
├── trainViz.py                # Training visualization
├── prepData.py                # Dataset preparation
├── testWS.py                  # WebSocket connectivity test
├── setup.py                   # Package installation
├── requirements.txt           # Python dependencies (alternative to setup.py)
├── output/
│   └── finbert-lora/         # Trained model artifacts
│       ├── sentiment_model.onnx
│       ├── tokenizer/
│       └── training_meta.json
├── lz-dashboard/             # React dashboard
│   ├── package.json
│   └── src/
│       ├── LZQuant.jsx       # Main dashboard component
│       ├── App.js
│       └── index.js
└── docs/
    └── index.html            # Marketing page
```

---

## Configuration

### Trading Parameters (dual.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--capacity` | 10000 | Starting capital (USD) |
| `--risk` | 25.0 | Risk per trade (%) |
| `--max-pos` | 6 | Maximum open positions |
| `--port` | 8765 | Server port |

### Trading Config (dual.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `risk_per_trade_pct` | 25.0% | Equity risked per trade |
| `max_position_pct` | 50.0% | Max position size |
| `stop_loss_pct` | 1.5% | Stop-loss threshold |
| `take_profit_pct` | 2.5% | Take-profit threshold |
| `trailing_stop_pct` | 1.0% | Trailing stop distance |
| `max_drawdown_pct` | 30.0% | Circuit breaker threshold |
| `min_confidence` | 0.45 | Minimum sentiment confidence |
| `require_divergence` | False | Require divergence for trades |
| `divergence_boost` | 2.0 | Position size multiplier on divergence |

### Data Refresh Intervals

| Source | Interval | Description |
|--------|----------|-------------|
| Reddit | 15 seconds | 10 subreddits |
| RSS | 45 seconds | 6 news feeds |
| Binance | Real-time | WebSocket streams |
| Alpaca | Real-time | WebSocket streams |
| Inference | 2 seconds | Per-symbol throttled |

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML Dashboard |
| GET | `/api/status` | System status |
| GET | `/api/ingestion` | Text ingestion stats |
| GET | `/api/trades` | Closed trades (last 50) |
| POST | `/api/trading/set-capital` | Set capital and reset |
| POST | `/api/trading/reset` | Reset all trades |
| POST | `/api/trading/pause` | Pause trading |
| POST | `/api/trading/resume` | Resume trading |
| GET | `/api/trading/status` | Get pause status |

### WebSocket Messages

**Client → Server**: No messages required (receive-only)

**Server → Client**:

| Type | Description |
|------|-------------|
| `signal` | Inference result with sentiment, action, scores, latency |
| `trade_execution` | Trade opened |
| `trade_exit` | Trade closed |
| `divergence` | Divergence detected |
| `price` | Individual trade tick |
| `ticker` | 24h market stats |
| `portfolio` | Equity, P&L, positions |
| `stats` | System metrics |
| `ingestion` | Text source info |
| `trading_status` | Pause/resume state |

---

## Data Sources

### Reddit (10 Subreddits)

| Subreddit | Market | Interval |
|-----------|--------|----------|
| r/CryptoCurrency | Crypto | 15s |
| r/Bitcoin | Crypto | 15s |
| r/ethereum | Crypto | 15s |
| r/solana | Crypto | 15s |
| r/CryptoMarkets | Crypto | 15s |
| r/wallstreetbets | Stocks | 15s |
| r/stocks | Stocks | 15s |
| r/investing | Stocks | 15s |
| r/options | Stocks | 15s |
| r/StockMarket | Stocks | 15s |

### RSS Feeds (6 Sources)

| Feed | Market | Interval |
|------|--------|----------|
| CoinDesk | Crypto | 45s |
| CoinTelegraph | Crypto | 45s |
| Decrypt | Crypto | 45s |
| MarketWatch | Stocks | 45s |
| Yahoo Finance | Stocks | 45s |
| CNBC | Stocks | 45s |

### Symbol Detection

**Crypto**: BTC, Bitcoin, ETH, ethereum, SOL, solana (+ keywords: crypto, defi, blockchain)

**Stocks**: AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META, AMZN, SPY, QQQ (+ keywords: stock, earnings, Fed, market)

---

## Trading Engine

### Signal Generation

```python
bull = scores.get("BULLISH", 0)
bear = scores.get("BEARISH", 0)
action = "BUY" if bull >= 0.45 else ("SELL" if bear >= 0.45 else "HOLD")
```

### Divergence Detection

- **BEARISH DIVERGENCE**: Sentiment Z-score > threshold AND Price Z-score < -threshold
- **BULLISH DIVERGENCE**: Sentiment Z-score < -threshold AND Price Z-score > threshold
- Severity = sigmoid(geometric_mean(Z_scores))

Divergence can override signals:
- `BEARISH_DIVERGENCE` + severity > 0.3 → Forces SELL
- `BULLISH_DIVERGENCE` + severity > 0.3 → Forces BUY

### Position Sizing

```
Base size = equity × risk_per_trade_pct%
If divergence signal: size × divergence_boost (2.0x)
```

### Exit Rules

1. **Stop-loss**: P&L < -1.5%
2. **Take-profit**: P&L > +2.5%
3. **Trailing stop**: Price drops 1.0% from high-water mark
4. **Signal reversal**: Opposite signal for same symbol

---

## Dashboard

### Setup Screen
- Enter starting capital ($100 - $10,000,000)
- Preset buttons: $1K, $5K, $10K, $50K
- Risk per trade and max drawdown display
- Start Trading button

### Main Dashboard
- **Header**: Live/Demo indicator, divergence alerts, pause button
- **Stats Grid**: Signals, latency, throughput, equity, P&L, win rate, trades, drawdown
- **Chart**: Toggle between Equity, Sentiment, Latency views
- **Market Panels**: Crypto (Binance) and Stocks (NASDAQ) side-by-side
- **Positions**: Open positions with entry price and notional
- **Trade Log**: Trade history with P&L

### Signal Card Display
- Action (BUY/SELL/HOLD) with color coding
- Symbol and confidence percentage
- Sentiment bar (Bullish/Neutral/Bearish)
- Divergence badge if applicable
- Trade action badge if executed

---

## CSV Logging

Trades are logged to `trades_log.csv` in the project root.

### Columns

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp of event |
| `symbol` | Trading symbol |
| `market` | "crypto" or "stocks" |
| `side` | LONG/SHORT |
| `action` | opened_long, closed, etc. |
| `entry_price` | Entry price |
| `exit_price` | Exit price |
| `quantity` | Position size |
| `entry_time` | Entry timestamp |
| `exit_time` | Exit timestamp |
| `entry_signal` | Signal at entry |
| `exit_reason` | take_profit, stop_loss, etc. |
| `pnl` | Absolute P&L (USD) |
| `pnl_pct` | Percentage return |
| `holding_duration_s` | Seconds in position |
| `had_divergence` | Boolean |
| `source` | binance, alpaca, reddit, rss |
| `sentiment` | BULLISH/BEARISH/NEUTRAL |
| `confidence` | Model confidence |
| `divergence_type` | NONE, BULLISH_DIVERGENCE, BEARISH_DIVERGENCE |
| `divergence_severity` | 0.0 - 1.0 |

### Behavior
- File is **overwritten** on each session start
- Both open and closed trades are logged
- Source indicates which data feed triggered the trade

---

## Model Training

### Quick Training

```bash
python train.py
```

### Training Data

- Financial PhraseBank (~9,000 sentences)
- Twitter Financial News Sentiment (~8,000 tweets)
- Optional: CryptoBERT for crypto-specific sentiment

### Output

Trained model saved to `./output/finbert-lora/`:
- `sentiment_model.onnx` — ONNX model for inference
- `adapter_config.json` — LoRA configuration
- `tokenizer/` — Tokenizer files
- `training_meta.json` — Training history

### Visualization

```bash
python trainViz.py
```

Generates charts in `./output/finbert-lora/charts/`:
- Training overview
- Confidence analysis
- Training config

---

## Requirements

### Python

Install using `setup.py` (recommended):

```bash
pip install -e ".[all]"      # All dependencies
pip install -e .             # Core only
```

Or install from `requirements.txt`:

```
# Core (always needed)
websockets>=12.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
numpy>=1.26.0

# Inference (optional)
onnx>=1.16.0
onnxruntime>=1.16.0
transformers>=4.40.0

# Training (optional)
torch>=2.0.0
datasets>=2.19.0
peft>=0.11.0
accelerate>=0.30.0
scikit-learn>=1.4.0

# Visualization (optional)
matplotlib>=3.8.0
seaborn>=0.12.0
```

### Node.js (for React Dashboard)

```bash
cd lz-dashboard
npm install
```

---

## Troubleshooting

### Server won't start
- Check if port 8765 is in use: `netstat -an | findstr 8765`
- Try a different port: `python dual.py --port 8766`

### No Alpaca data
- Create `.env` file with API keys
- Check Alpaca account status at alpaca.markets

### Model not loading
- Run `python train.py` to train the model
- Or use simulated inference (automatic fallback)

### WebSocket connection issues
- Check internet connection
- Binance/Alpaca may have rate limits
- System auto-reconnects with exponential backoff

---

## License

MIT License — Paper trading only. Not financial advice.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Logos & Branding

All branding assets are located in the `logos/` folder.

| Asset | Description |
|-------|-------------|
| `bgLZQ.png` | Main logo with gradient background — used in README.md and ARCHITECTURE.md headers |
| `favicon.png` | Site favicon — used in React dashboard and dual.py server browser tabs |
| `lzQ-Black-Trans.png` | Black logo with transparent background |
| `lzQ.png` | Simplest logo variant |

---

Built with ❤️ for quantitative trading enthusiasts
