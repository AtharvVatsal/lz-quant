# LZ-Quant Architecture

![LZ-Quant Logo](logos/bgLZQ.png)

Detailed technical documentation of the LZ-Quant dual-market sentiment trading system.

---

## Overview

LZ-Quant is a real-time trading system that combines:
- **AI Sentiment Analysis** (DistilBERT + LoRA via ONNX)
- **Multi-Market Data** (Binance crypto + Alpaca stocks)
- **Divergence Detection** (Z-score based)
- **Paper Trading Engine** (full P&L tracking)
- **Real-Time Dashboard** (WebSocket + React)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LZ-QUANT SYSTEM                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │   Reddit (10 subs)  │──┐
                              │   r/CryptoCurrency  │  │
                              │   r/Bitcoin         │  │
                              │   r/ethereum        │  │
                              │   r/solana          │  │
                              │   r/CryptoMarkets   │  │
                              │   r/wallstreetbets  │  │
                              │   r/stocks          │  │
                              │   r/investing       │  │
                              │   r/options         │  │
                              │   r/StockMarket     │  │
                              └─────────┬───────────┘  │
                                        │              │
                              ┌─────────┴──────────┐   │
                              │  dataIngestion.py  │   │
                              │  ───────────────── │   │
                              │  • RedditStream    │   │
                              │  • RSSStream       │   │
                              │  • SymbolDetector  │   │
                              │  • TextRouter      │   │
                              └─────────┬──────────┘   │
                                        │              │
                              ┌─────────┴──────────┐   │
                              │    TextItem Queue  │◄──┘
                              │  (AsyncQueue)      │
                              └─────────┬──────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              dual.py                                      │
│  ───────────────────────────────────────────────────────────────────────  │
│                                                                           │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐    │
│   │    Binance   │          │    Alpaca    │          │   Reddit/    │    │
│   │  WebSocket   │          │  WebSocket   │          │     RSS      │    │
│   │  (Crypto)    │          │  (Stocks)    │          │  Consumer    │    │
│   └───────┬──────┘          └───────┬──────┘          └───────┬──────┘    │
│           │                         │                         │           │
│           │    price ticks          │    price ticks          │ text      │
│           └────────────┬────────────┴─────────────────────────┘           │
│                        ▼                                                  │
│           ┌─────────────────────────────────────────┐                     │
│           │         MarketCommentary                │                     │
│           │  (Generates text from price data)       │                     │
│           └────────────────────┬────────────────────┘                     │
│                                │                                          │
│                                ▼                                          │
│           ┌─────────────────────────────────────────┐                     │
│           │         SentimentEngine                 │                     │
│           │  ─────────────────────────────────────  │                     │
│           │  • ONNX DistilBERT (if model exists)    │                     │
│           │  • Mock fallback (simulated inference)  │                     │
│           │  • Returns: (label, scores, ms)         │                     │
│           └────────────────────┬────────────────────┘                     │
│                                │                                          │
│               ┌────────────────┴────────────────┐                         │
│               ▼                                 ▼                         │
│    ┌─────────────────────┐           ┌──────────────────┐                 │
│    │  DivergenceDetector │           │  PaperTrading    │                 │
│    │  ────────────────   │           │  Engine          │                 │
│    │  • Rolling Z-score  │           │  ─────────────── │                 │
│    │  • Sentiment wind   │           │  • Positions     │                 │
│    │  • Price window     │           │  • P&L tracking  │                 │
│    │  • Cooldown         │           │  • Stop-loss     │                 │
│    └────────┬────────────┘           │  • Take-profit   │                 │
│             │                        │  • Trailing stop │                 │
│             └────────────┬───────────┴──────────────────┘                 │
│                          │                                                │
│                          ▼                                                │
│           ┌─────────────────────────────────────────┐                     │
│           │           DashManager                   │                     │
│           │  ─────────────────────────────────────  │                     │
│           │  • WebSocket connection management      │                     │
│           │  • Broadcast to all clients             │                     │
│           │  • Message queue                        │                     │
│           └────────────────────┬────────────────────┘                     │
│                                │                                          │
└────────────────────────────────┼──────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
   ┌────────────┐       ┌────────────┐        ┌────────────┐
   │  HTML      │       │  React     │        │   CSV      │
   │  Dashboard │       │  Dashboard │        │  Logger    │
   │  (FastAPI) │       │  (JSX)     │        └────────────┘
   └────────────┘       └────────────┘        trades_log.csv
   localhost:8765       localhost:3000        
```

---

## Component Details

### 1. Data Ingestion (`dataIngestion.py`)

#### RedditStream
```
Polls: 10 subreddits every 15 seconds
├── CryptoCurrency (crypto)
├── Bitcoin (crypto)
├── ethereum (crypto)
├── solana (crypto)
├── CryptoMarkets (crypto)
├── wallstreetbets (stocks)
├── stocks (stocks)
├── investing (stocks)
├── options (stocks)
└── StockMarket (stocks)
```

**Flow:**
1. Fetch `/new.json` and `/hot.json` from Reddit API
2. Deduplicate using `deque(maxlen=5000)`
3. Clean text (remove URLs, markdown, etc.)
4. Detect symbols via `SymbolDetector`
5. Push `TextItem` to `TextRouter` queue

#### RSSStream
```
Polls: 6 RSS feeds every 45 seconds
├── CoinDesk (crypto)
├── CoinTelegraph (crypto)
├── Decrypt (crypto)
├── MarketWatch (stocks)
├── Yahoo Finance (stocks)
└── CNBC (stocks)
```

**Flow:**
1. Fetch XML from RSS URL
2. Parse items (supports both RSS 2.0 and Atom)
3. Deduplicate using MD5 hash
4. Strip HTML tags
5. Detect symbols and push to queue

#### TextRouter
```
Central hub for all text data
├── Combines Reddit + RSS streams
├── Async queue (max 500 items)
├── Deduplication (MD5 hash)
├── get_next() returns TextItem or None
└── Stats tracking
```

---

### 2. Sentiment Engine (`dual.py` - `SentimentEngine`)

#### ONNX Mode (when model exists)
```
Input: Raw text string
Output: (label, scores_dict, inference_ms)
        label: "BULLISH" | "BEARISH" | "NEUTRAL"
        scores: {"BULLISH": 0.75, "NEUTRAL": 0.15, "BEARISH": 0.10}
```

**Pipeline:**
1. Tokenize text (max 128 tokens)
2. Run ONNX inference
3. Softmax probabilities
4. Return highest confidence label

#### Mock Mode (fallback)
```
Keyword-based sentiment simulation
Bullish keywords: surge, rising, rally, gain, beat, record, high
Bearish keywords: plunging, falling, sell, drop, low, miss, loss, crash
```

---

### 3. Market Commentary (`dual.py` - `MarketCommentary`)

Generates synthetic text from price data for inference:

```python
def generate_commentary(symbol, price, change_pct):
    # Tracks price trends
    # Generates description like:
    # "Bitcoin trading at $67400.00, rising 0.45%. Market showing buy pressure. Buy signal active."
```

Used by Binance and Alpaca pipelines to generate inference signals from price movements.

---

### 4. Divergence Detector (`divergenceTrading.py`)

#### Concept
Detects when sentiment diverges from price action:
- **Bullish Divergence**: Price falling but sentiment rising → BUY signal
- **Bearish Divergence**: Price rising but sentiment falling → SELL signal

#### Implementation
```python
DivergenceConfig:
    sentiment_window: 25      # trades to average
    price_window: 25         # trades for momentum
    zscore_window: 60        # history for Z-score
    sentiment_zscore_threshold: 0.8
    price_zscore_threshold: 0.8
    cooldown_ticks: 15
```

#### Algorithm
```
1. Rolling average: sentiment_history[-sentiment_window:]
2. Z-score: (current - mean) / std
3. If |sentiment_z| > threshold AND |price_z| > threshold:
     - Same direction → No divergence
     - Opposite direction → DIVERGENCE
4. Severity = sigmoid(geometric_mean(Z_scores))
```

---

### 5. Paper Trading Engine (`divergenceTrading.py`)

#### Position Sizing
```
Position size = equity × risk_per_trade_pct
Example: $10,000 × 25% = $2,500 per trade

With divergence boost (2.0x):
Position size = $2,500 × 2.0 = $5,000
```

#### Entry Rules
1. Signal confidence >= 0.45
2. Not already in position for symbol
3. Below max open positions (6)
4. Not halted (max drawdown not breached)

#### Exit Rules
```
1. Stop-loss: P&L < -1.5%
2. Take-profit: P&L > +2.5%
3. Trailing stop: price < high_water × 0.99
4. Signal reversal: opposite signal for symbol
5. Manual: user reset
```

#### Circuit Breaker
```
If current_equity < peak_equity × (1 - max_drawdown_pct):
    Close ALL positions
    Set is_halted = True
    Require manual reset
```

---

### 6. WebSocket Broadcasting (`dual.py`)

#### DashManager
```python
class DashManager:
    connections: list[WebSocket]
    
    async def connect(ws):     # Accept and track
    def disconnect(ws):        # Remove on close
    async def broadcast(msg):  # Send to all
```

#### Message Types (Server → Client)

| Type | Trigger | Contents |
|------|---------|----------|
| `signal` | Every inference | sentiment, action, scores, latency |
| `trade_execution` | New position | symbol, action, details |
| `trade_exit` | Position closed | symbol, pnl, exit_reason |
| `divergence` | Divergence detected | symbol, type, severity |
| `price` | Every trade tick | symbol, price, side |
| `ticker` | 24h stats | symbol, price, change_pct |
| `portfolio` | Periodic | equity, positions, metrics |
| `stats` | Periodic | total_signals, throughput |
| `ingestion` | Text received | source, text_preview |
| `trading_status` | Pause/Resume | is_paused |

---

### 7. Trade CSV Logger (`dual.py`)

```python
class TradeCSVLogger:
    filename = "trades_log.csv"
    
    def log_open_trade(trade_data):  # Position opened
    def log_trade(trade_data):       # Position closed
```

**Behavior:**
- Creates fresh file on each startup
- Appends open trades (entry info only)
- Updates with closed trades (full P&L)
- Overwrites on restart

---

## Data Flow

### Signal Flow
```
1. Reddit/RSS/Binance/Alpaca → TextRouter/Price
                                 │
                                 ▼
2. TextRouter.get_next() OR Price tick
                                 │
                                 ▼
3. Generate commentary (if price) OR use text
                                 │
                                 ▼
4. SentimentEngine.predict(text)
   → (label, scores, ms)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
5. DivergenceDetector.update()      Text → inference
   → DivergenceSignal              (already done)
                    │                         │
                    └────────────┬────────────┘
                                 ▼
6. PaperTradingEngine.process_signal()
   → Opens/closes position
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
7. DashManager.broadcast()        CSVLogger.log_*
   → WebSocket clients           → trades_log.csv
```

### WebSocket Flow
```
Browser ──connect──▶ /ws/dashboard
                        │
                        ▼
                 DashManager.connect()
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         dual.py    server.py   integration.py
            │           │           │
            └───────────┴───────────┘
                        │
                        ▼
                 All connected clients
                 receive broadcasts
```

---

## State Management

### dual.py State
```python
state = {
    "total_signals": int,      # Running count
    "total_messages": int,     # WebSocket messages
    "prices": dict,            # symbol → price
    "start_time": float,       # Unix timestamp
    "router": TextRouter,       # Text ingestion
}
```

### Divergence State (per symbol)
```python
DivergenceDetector._state[symbol] = {
    "sentiment_history": deque,  # Recent sentiment scores
    "price_history": deque,      # Recent prices
    "signal_history": deque,     # Recent signals
    "z_sentiment": float,       # Current Z-score
    "z_price": float,           # Current Z-score
    "last_divergence": str,     # Last divergence type
    "ticks_since_signal": int,  # Cooldown counter
}
```

### Trading State
```python
PaperTradingEngine:
    cash: float              # Available cash
    positions: dict          # symbol → Position
    closed_trades: list      # ClosedTrade objects
    returns_history: list    # Per-trade returns
    equity_curve: list       # Time series
    peak_equity: float      # High water mark
    is_halted: bool         # Circuit breaker
```

---

## API Endpoints

### GET /
Returns embedded HTML dashboard

### GET /api/status
```json
{
  "status": "running",
  "model": "real" | "simulated",
  "crypto": 3,
  "stocks": 10,
  "capital": 10000,
  "equity": 10500,
  "ingestion": {...}
}
```

### POST /api/trading/pause
Pauses trading (preserves state)

### POST /api/trading/resume
Resumes trading from current state

### GET /api/trades
```json
{
  "trades": [...],  // Last 50 closed trades
  "total": 125
}
```

---

## Configuration Hierarchy

```
1. Environment Variables (.env)
   └── ALPACA_API_KEY, ALPACA_SECRET_KEY

2. CLI Arguments (dual.py)
   └── --capacity, --risk, --max-pos, --port

3. Hardcoded Defaults (dual.py)
   └── TRADE_CFG, DIV_CFG, CRYPTO_SYMBOLS, STOCK_SYMBOLS

4. Data Ingestion Config (dataIngestion.py)
   └── REDDIT_SOURCES, RSS_SOURCES, CRYPTO_SYMBOLS, STOCK_SYMBOLS
```

---

## Error Handling

### WebSocket Reconnection
```
Exponential backoff:
1. Attempt 1: immediate
2. Attempt 2: 1 second
3. Attempt 3: 2 seconds
4. Attempt 4: 4 seconds
... max 30 seconds
```

### API Error Responses
```python
if amount < 100:
    return {"error": "Minimum capital is $100"}
if amount > 10_000_000:
    return {"error": "Maximum capital is $10,000,000"}
```

### Model Fallback
```
If ONNX model not found OR inference fails:
    → Use mock sentiment (keyword-based)
    → Log warning to console
```

---

## Performance Considerations

| Component | Performance |
|-----------|-------------|
| ONNX Inference | ~55ms CPU, ~5ms GPU |
| Mock Inference | <1ms |
| Reddit Poll | ~500ms (network) |
| RSS Poll | ~800ms (network) |
| Binance WS | <5ms (local network) |
| Alpaca WS | <10ms (network) |

### Throttling
```
Reddit/RSS: No throttling (queue handles burst)
Binance: 1 inference per symbol per 2 seconds
Alpaca: 1 inference per symbol per 2 seconds
```

---

## Security Notes

- **Paper trading only** — no real money
- **No credentials stored** — API keys in .env
- **Input sanitization** — regex for symbol detection
- **Rate limiting** — built into WebSocket protocols
- **CORS enabled** — for local development

---

## Extension Points

### Adding New Symbols
```python
# dual.py
CRYPTO_SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "newcoin"]  # Add here

# dataIngestion.py - CRYPTO_SYMBOLS dict
"NEWCOIN": [r"\bNEW\b", r"\bNewCoin\b"]  # Add patterns
```

### Adding New Reddit Subreddits
```python
# dataIngestion.py
REDDIT_SOURCES = [
    ...,
    ("newsubreddit", "stocks", 30),  # Add here
]
```

### Adding New RSS Feeds
```python
# dataIngestion.py
RSS_SOURCES = [
    ...,
    ("https://example.com/rss", "stocks", 60),  # Add here
]
```

### Custom Signal Thresholds
```python
# dual.py - run_inference_and_broadcast
action = "BUY" if bull >= 0.45 else ("SELL" if bear >= 0.45 else "HOLD")
# Change 0.45 to desired threshold
```

---

## File Dependencies

```
dual.py
├── divergenceTrading.py
│   └── (uses: dataclasses, deque)
├── dataIngestion.py
│   └── (uses: asyncio.Queue, re, hashlib)
└── fastapi, websockets, uvicorn
```

```
train.py
└── transformers, peft, datasets, onnx
```

```
lz-dashboard/
├── package.json
│   └── recharts, react-scripts
└── src/LZQuant.jsx
    └── (uses: useState, useEffect, useCallback)
```
