"""
INTEGRATION — Wiring Divergence + Trading into the Existing Pipeline

This file patches inference engine and dashboard server
to use the new divergence detection and paper trading systems.

There are two ways to use this:

  OPTION A: Run this file standalone (recommended for testing)
            python integration.py
            → Starts the full pipeline: Binance → Inference → Divergence → Trading → Dashboard

  OPTION B: Copy the relevant sections into inference.py and server.py
            (see the clearly marked integration points below)

"""

import asyncio
import json
import signal
import sys
import time
import os
from datetime import datetime, timezone
from collections import deque

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("Missing: pip install websockets")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    sys.exit("Missing: pip install fastapi uvicorn[standard]")

from divergenceTrading import (
    DivergenceDetector,
    DivergenceConfig,
    PaperTradingEngine,
    TradingConfig,
    DivergenceType,
)

# Optional ONNX model
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    from transformers import DistilBertTokenizerFast
    ONNX_AVAILABLE = True
except ImportError:
    pass

# CONFIG
SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
SERVER_PORT = 8765

DIVERGENCE_CONFIG = DivergenceConfig(
    sentiment_window=50,
    price_window=50,
    zscore_window=100,
    sentiment_zscore_threshold=1.2,
    price_zscore_threshold=1.2,
    cooldown_ticks=30,
)

TRADING_CONFIG = TradingConfig(
    starting_capital=10_000.0,
    risk_per_trade_pct=2.0,
    stop_loss_pct=1.5,
    take_profit_pct=3.0,
    trailing_stop_pct=1.0,
    max_drawdown_pct=15.0,
    min_confidence=0.65,
    require_divergence=False,
    divergence_boost=1.5,
)

# SENTIMENT ENGINE (real or simulated)
class SentimentEngine:
    def __init__(self, model_path="./output/finbert-lora/sentiment_model.onnx",
                 tokenizer_path="./output/finbert-lora"):
        self.session = None
        self.tokenizer = None
        self.is_mock = True

        if ONNX_AVAILABLE and os.path.exists(model_path):
            try:
                providers = ["CPUExecutionProvider"]
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(model_path, opts, providers=providers)
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
                self.is_mock = False
                self.predict("warmup")
                print("[MODEL] ONNX model loaded")
            except Exception as e:
                print(f"[MODEL] ONNX failed: {e} — using simulation")
        else:
            print("[MODEL] Using simulated inference (no ONNX model found)")

    def predict(self, text):
        t0 = time.perf_counter_ns()
        if not self.is_mock:
            enc = self.tokenizer(text, padding="max_length", truncation=True,
                                 max_length=128, return_tensors="np")
            logits = self.session.run(["logits"], {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })[0]
            e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            p = (e / e.sum(axis=-1, keepdims=True)).squeeze()
            scores = {"BEARISH": float(p[0]), "NEUTRAL": float(p[1]), "BULLISH": float(p[2])}
        else:
            scores = self._simulate(text)
        ms = (time.perf_counter_ns() - t0) / 1e6
        label = max(scores, key=scores.get)
        return label, scores, ms

    def _simulate(self, text):
        t = text.lower()
        bull = ["surge", "rising", "buying", "support", "gain", "high"]
        bear = ["plunging", "falling", "selling", "pressure", "drop", "low"]
        bh = sum(1 for k in bull if k in t)
        brh = sum(1 for k in bear if k in t)
        if bh > brh:     base = [0.10, 0.15, 0.75]
        elif brh > bh:   base = [0.75, 0.15, 0.10]
        else:             base = [0.20, 0.60, 0.20]
        noise = np.random.dirichlet([10, 10, 10])
        m = np.array(base) * 0.7 + noise * 0.3
        m /= m.sum()
        return {"BEARISH": float(m[0]), "NEUTRAL": float(m[1]), "BULLISH": float(m[2])}


# TEXT CONSTRUCTOR
class TextConstructor:
    """
    Converts raw market data into neutral, factual sentences.
    
    IMPORTANT: The language must be balanced — not biased toward bullish or bearish.
    The MODEL should decide sentiment based on price movement magnitude and context,
    not from loaded phrases like "aggressive buying" in every sentence.
    
    The old constructor used "aggressive buying" for ~50% of trades (whenever
    is_buyer_maker=False) regardless of actual market direction, which made the
    model predict 100% bullish on everything.
    """
    def __init__(self):
        self._prices = {}
        self._depths = {}

    def from_trade(self, data):
        sym = data.get("s", "???")
        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
        is_sell = data.get("m", False)
        usd_value = price * qty

        if sym not in self._prices:
            self._prices[sym] = deque(maxlen=50)
        self._prices[sym].append(price)
        h = self._prices[sym]
        chg = ((price - h[0]) / h[0] * 100) if len(h) > 1 and h[0] > 0 else 0

        # Neutral direction language — only use strong words for big moves
        if chg > 1.0:
            direction = "surging"
        elif chg > 0.3:
            direction = "rising"
        elif chg > 0.05:
            direction = "edging higher"
        elif chg < -1.0:
            direction = "plunging"
        elif chg < -0.3:
            direction = "falling"
        elif chg < -0.05:
            direction = "edging lower"
        else:
            direction = "trading flat"

        # Neutral trade side description
        side = "sell-side trade" if is_sell else "buy-side trade"

        # Size context (only mention if notable)
        size = ""
        if usd_value > 100_000:
            size = " in a large block"
        elif usd_value > 50_000:
            size = " with notable size"

        # Order book context (only if strongly imbalanced)
        book = ""
        if sym in self._depths:
            r = self._depths[sym].get("ratio", 1)
            if r > 1.5:
                book = ", order book bid-heavy"
            elif r < 0.65:
                book = ", order book ask-heavy"

        sign = "up" if chg >= 0 else "down"
        return f"{sym} {direction} at ${price:,.2f}, {sign} {abs(chg):.2f}% on {side}{size}{book}"

    def from_ticker(self, data):
        sym = data.get("s", "???")
        chg = float(data.get("P", 0))
        price = float(data.get("c", 0))
        vol = float(data.get("v", 0))

        if chg > 3:
            trend = "rallying strongly"
        elif chg > 1:
            trend = "trending higher"
        elif chg < -3:
            trend = "selling off sharply"
        elif chg < -1:
            trend = "trending lower"
        else:
            trend = "trading flat"

        return f"{sym} {trend}, 24h change {chg:+.2f}% at ${price:,.2f} on volume {vol:,.0f}"

    def from_depth(self, data, symbol):
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        bd = sum(float(b[1]) for b in bids) if bids else 0
        ad = sum(float(a[1]) for a in asks) if asks else 1
        ratio = bd / ad if ad > 0 else 1
        sym = symbol.upper()
        self._depths[sym] = {"ratio": ratio}

        if ratio > 1.5:
            p = "bids significantly outweigh asks"
        elif ratio > 1.15:
            p = "slightly more bids than asks"
        elif ratio < 0.65:
            p = "asks significantly outweigh bids"
        elif ratio < 0.85:
            p = "slightly more asks than bids"
        else:
            p = "balanced between bids and asks"

        return f"{sym} order book {p}, ratio {ratio:.2f}"

# DASHBOARD MANAGER
class DashboardManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg):
        if not self.connections:
            return
        payload = json.dumps(msg, default=str)
        dead = []
        for c in self.connections:
            try:
                await c.send_text(payload)
            except Exception:
                dead.append(c)
        for c in dead:
            self.connections.remove(c)

# FULL PIPELINE — Binance → Inference → Divergence → Trading → Dashboard
async def full_pipeline(engine, detector, trader, text_ctor, manager, state):
    """
    The complete enhanced pipeline with divergence detection and paper trading.

    ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐    ┌───────────┐
    │ Binance  │──▶│ Sentiment │───▶│ Divergence │──▶│  Paper   │───▶│ Dashboard │
    │ WebSocket│    │ Inference │    │ Detector   │    │  Trader  │    │ Broadcast │
    └──────────┘    └───────────┘    └────────────┘    └──────────┘    └───────────┘
    """
    trade_streams = "/".join(f"{s}@trade" for s in SYMBOLS)
    ticker_streams = "/".join(f"{s}@miniTicker" for s in SYMBOLS)
    depth_streams = "/".join(f"{s}@depth10@1000ms" for s in SYMBOLS)
    url = f"wss://stream.binance.com:9443/stream?streams={trade_streams}/{ticker_streams}/{depth_streams}"

    loop = asyncio.get_running_loop()
    ticker_count = 0
    depth_count = 0

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[STREAM] Connected to Binance")

                async for raw_msg in ws:
                    t0 = time.perf_counter_ns()
                    state["total_messages"] += 1

                    data = json.loads(raw_msg)
                    stream_id = data.get("stream", "")
                    payload = data.get("data", data)
                    symbol = payload.get("s", stream_id.split("@")[0].upper())

                    text = None
                    msg_type = None
                    price = 0.0

                    if "@trade" in stream_id:
                        msg_type = "trade"
                        text = text_ctor.from_trade(payload)
                        price = float(payload.get("p", 0))
                        state["prices"][symbol] = price

                        # Broadcast raw price tick
                        await manager.broadcast({
                            "type": "price",
                            "symbol": symbol,
                            "price": price,
                            "time": datetime.now(timezone.utc).isoformat(),
                            "side": "sell" if payload.get("m") else "buy",
                        })

                    elif "@miniTicker" in stream_id:
                        msg_type = "ticker"
                        ticker_count += 1
                        text = text_ctor.from_ticker(payload)
                        price = float(payload.get("c", 0))
                        await manager.broadcast({
                            "type": "ticker",
                            "symbol": symbol,
                            "price": price,
                            "change_pct": float(payload.get("P", 0)),
                            "high": float(payload.get("h", 0)),
                            "low": float(payload.get("l", 0)),
                            "volume": float(payload.get("v", 0)),
                        })
                        if ticker_count % 5 != 0:
                            text = None

                    elif "@depth" in stream_id:
                        msg_type = "depth"
                        depth_count += 1
                        stream_sym = stream_id.split("@")[0]
                        text = text_ctor.from_depth(payload, stream_sym)
                        if depth_count % 10 != 0:
                            text = None

                    if not text:
                        continue

                    # STEP 1: Sentiment inference
                    label, scores, inference_ms = await loop.run_in_executor(
                        None, engine.predict, text
                    )

                    # STEP 2: Divergence detection
                    current_price = state["prices"].get(symbol, price)
                    div_signal = detector.update(symbol, scores, current_price)

                    # STEP 3: Generate action
                    bull = scores.get("BULLISH", 0)
                    bear = scores.get("BEARISH", 0)

                    if bull >= 0.65:
                        action = "BUY"
                    elif bear >= 0.65:
                        action = "SELL"
                    else:
                        action = "HOLD"

                    # Divergence can override the action
                    if div_signal.divergence_type == DivergenceType.BEARISH_DIVERGENCE.value:
                        if div_signal.severity > 0.6:
                            action = "SELL"
                    elif div_signal.divergence_type == DivergenceType.BULLISH_DIVERGENCE.value:
                        if div_signal.severity > 0.6:
                            action = "BUY"

                    # STEP 4: Paper trading
                    trade_result = trader.process_signal(
                        symbol=symbol,
                        action=action,
                        price=current_price,
                        confidence=max(scores.values()),
                        sentiment=label,
                        divergence=div_signal,
                    )

                    # Also check stop-loss / take-profit on price updates
                    auto_exits = trader.update_prices(state["prices"])

                    t1 = time.perf_counter_ns()
                    latency_ms = (t1 - t0) / 1e6
                    state["total_signals"] += 1

                    # STEP 5: Broadcast everything to dashboard

                    # Signal with divergence + trading metadata
                    await manager.broadcast({
                        "type": "signal",
                        "id": state["total_signals"],
                        "symbol": symbol,
                        "action": action,
                        "sentiment": label,
                        "confidence": round(max(scores.values()), 4),
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                        "text": text,
                        "latency_ms": round(latency_ms, 2),
                        "inference_ms": round(inference_ms, 2),
                        "time": datetime.now(timezone.utc).isoformat(),
                        "source": msg_type,
                        "is_mock": engine.is_mock,
                        "divergence": div_signal.to_dict(),
                        "trade_action": trade_result["action_taken"],
                        "trade_reason": trade_result.get("reason", ""),
                    })

                    # Divergence alert (only when divergence fires)
                    if div_signal.divergence_type != DivergenceType.NONE.value:
                        await manager.broadcast({
                            "type": "divergence",
                            "symbol": symbol,
                            "divergence": div_signal.to_dict(),
                            "time": datetime.now(timezone.utc).isoformat(),
                        })

                    # Trade execution events
                    if trade_result["action_taken"] not in ("no_action", "halted"):
                        await manager.broadcast({
                            "type": "trade_execution",
                            "symbol": symbol,
                            "action": trade_result["action_taken"],
                            "details": trade_result.get("trade") or trade_result.get("position"),
                            "time": datetime.now(timezone.utc).isoformat(),
                        })

                    # Auto-exit events (stop-loss, take-profit)
                    for exit_trade in auto_exits:
                        await manager.broadcast({
                            "type": "trade_exit",
                            "trade": exit_trade,
                            "time": datetime.now(timezone.utc).isoformat(),
                        })

        except Exception as e:
            print(f"[STREAM] Error: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)


async def periodic_stats(trader, manager, state, interval=5):
    """Broadcast portfolio + pipeline stats every N seconds."""
    while True:
        await asyncio.sleep(interval)

        # Get latest price for metrics
        any_price = 0
        if state["prices"]:
            any_price = list(state["prices"].values())[0]

        metrics = trader.get_metrics(any_price)
        positions = trader.get_open_positions()

        await manager.broadcast({
            "type": "stats",
            "total_signals": state["total_signals"],
            "total_messages": state["total_messages"],
            "uptime_seconds": round(time.time() - state["start_time"], 1),
            "msg_per_sec": round(state["total_messages"] / max(time.time() - state["start_time"], 1), 1),
        })

        await manager.broadcast({
            "type": "portfolio",
            "metrics": metrics.to_dict(),
            "positions": positions,
            "equity_curve": trader.equity_curve[-100:],
        })

# FASTAPI APP
app = FastAPI(title="LZ-Quant v2 — Divergence + Trading")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

manager = DashboardManager()
state = {
    "total_signals": 0,
    "total_messages": 0,
    "prices": {},
    "start_time": time.time(),
}

engine = None
detector = None
trader = None

# SELF-CONTAINED HTML DASHBOARD (Divergence + Trading)
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LZ-Quant v2 — Divergence + Trading</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #04080f; --surface: #0a1118; --border: #162236; --border-b: #1e3350;
    --text: #94a8c4; --dim: #4a5e78; --bright: #e2ecf8;
    --bull: #00e676; --bear: #ff1744; --neut: #ffab00; --accent: #448aff; --div: #e040fb;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', sans-serif; padding: 14px; }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .header { display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid var(--border); margin-bottom:10px; }
  .title { font-size:17px; font-weight:800; color:var(--bright); }
  .title span { color:var(--accent); }
  .status { display:flex; align-items:center; gap:5px; font-size:10px; color:var(--dim); }
  .dot { width:7px; height:7px; border-radius:50%; }
  .dot.on { background:var(--bull); box-shadow:0 0 6px var(--bull); }
  .dot.off { background:var(--bear); box-shadow:0 0 6px var(--bear); }
  .stats-grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px; margin-bottom:10px; }
  .stat-card { background:var(--surface); border:1px solid var(--border); border-radius:5px; padding:8px; }
  .stat-label { font-size:8px; text-transform:uppercase; letter-spacing:1px; color:var(--dim); margin-bottom:1px; }
  .stat-value { font-size:15px; font-weight:700; color:var(--bright); font-family:'JetBrains Mono',monospace; }
  .stat-value .unit { font-size:9px; color:var(--dim); }
  .price-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:10px; }
  .price-card { background:var(--surface); border:1px solid var(--border); border-radius:5px; padding:8px; }
  .price-sym { font-weight:700; font-size:11px; color:var(--bright); }
  .price-val { font-size:19px; font-weight:800; font-family:'JetBrains Mono',monospace; color:var(--bright); margin:3px 0; }
  .price-chg { font-size:10px; font-weight:700; font-family:'JetBrains Mono',monospace; }
  .main-grid { display:grid; grid-template-columns:1fr 340px; gap:10px; }
  .panel { background:var(--surface); border:1px solid var(--border); border-radius:5px; padding:10px; }
  .panel-header { font-size:8px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:var(--dim); margin-bottom:6px; display:flex; justify-content:space-between; }
  .signal-feed { max-height:60vh; overflow-y:auto; display:flex; flex-direction:column; gap:4px; }
  .signal-card { background:var(--bg); border:1px solid var(--border); border-radius:4px; padding:6px 8px; font-size:10px; border-left:3px solid var(--dim); animation:fi 0.3s ease; }
  .signal-card.buy { border-left-color:var(--bull); }
  .signal-card.sell { border-left-color:var(--bear); }
  .signal-card.hold { border-left-color:var(--neut); }
  .signal-card.divergence { border-color:var(--div)40; }
  .signal-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:3px; }
  .signal-action { font-weight:800; font-family:'JetBrains Mono',monospace; font-size:11px; }
  .signal-action.buy { color:var(--bull); }
  .signal-action.sell { color:var(--bear); }
  .signal-action.hold { color:var(--neut); }
  .signal-text { color:var(--dim); font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:3px; }
  .signal-bar { display:flex; height:2px; border-radius:1px; overflow:hidden; }
  .signal-bar .bull { background:var(--bull); }
  .signal-bar .neut { background:var(--neut); }
  .signal-bar .bear { background:var(--bear); }
  .badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:9px; font-weight:700; font-family:'JetBrains Mono',monospace; }
  .div-alert { background:var(--div)15; border:1px solid var(--div)30; border-radius:5px; padding:8px; margin-bottom:6px; }
  .div-alert-header { font-size:9px; font-weight:700; color:var(--div); letter-spacing:1px; text-transform:uppercase; margin-bottom:4px; }
  .position-card { background:var(--bg); border:1px solid var(--border); border-radius:4px; padding:5px 8px; font-size:10px; display:flex; justify-content:space-between; align-items:center; margin-bottom:3px; }
  .footer { margin-top:10px; padding-top:6px; border-top:1px solid var(--border); font-size:8px; color:var(--dim); display:flex; justify-content:space-between; }
  @keyframes fi { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:translateY(0); } }
  ::-webkit-scrollbar { width:3px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:10px">
    <div class="title"><span>⚡</span> LZ-QUANT <span style="color:var(--dim);font-size:11px;font-weight:400">v2</span></div>
    <div class="status"><div class="dot off" id="statusDot"></div><span id="statusText">CONNECTING</span></div>
  </div>
  <div class="mono" style="font-size:10px;color:var(--dim)" id="clock"></div>
</div>

<div class="stats-grid">
  <div class="stat-card"><div class="stat-label">Signals</div><div class="stat-value" id="sSignals">0</div></div>
  <div class="stat-card"><div class="stat-label">Latency</div><div class="stat-value" id="sLatency">—<span class="unit">ms</span></div></div>
  <div class="stat-card"><div class="stat-label">Throughput</div><div class="stat-value" id="sThroughput">—<span class="unit">/s</span></div></div>
  <div class="stat-card"><div class="stat-label">Equity</div><div class="stat-value" id="sEquity">$10,000</div></div>
  <div class="stat-card"><div class="stat-label">P&L</div><div class="stat-value" id="sPnl">$0</div></div>
  <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value" id="sWinRate">—</div></div>
  <div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value" id="sTrades">0</div></div>
  <div class="stat-card"><div class="stat-label">Drawdown</div><div class="stat-value" id="sDrawdown">0%</div></div>
</div>

<div class="price-grid" id="priceGrid"></div>

<div class="main-grid">
  <div style="display:flex;flex-direction:column;gap:8px">
    <div class="panel" style="flex:1">
      <div class="panel-header"><span>SIGNAL FEED</span><span class="mono" id="sigCount">0</span></div>
      <div style="display:flex;height:4px;border-radius:2px;overflow:hidden;margin-bottom:6px">
        <div id="sentBull" style="background:var(--bull);width:33%;transition:width 0.3s"></div>
        <div id="sentNeut" style="background:var(--neut);width:34%;transition:width 0.3s"></div>
        <div id="sentBear" style="background:var(--bear);width:33%;transition:width 0.3s"></div>
      </div>
      <div class="signal-feed" id="signalFeed"></div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:8px">
    <div id="divAlerts"></div>
    <div class="panel">
      <div class="panel-header"><span>OPEN POSITIONS</span><span class="mono" id="posCount">0</span></div>
      <div id="positions"><div style="font-size:10px;color:var(--dim);text-align:center;padding:12px">No open positions</div></div>
    </div>
    <div class="panel" style="flex:1">
      <div class="panel-header"><span>TRADE LOG</span><span class="mono" id="tradeLogCount">0</span></div>
      <div class="signal-feed" id="tradeLog"></div>
    </div>
  </div>
</div>

<div class="footer">
  <span>LZ-QUANT — DIVERGENCE DETECTION + PAPER TRADING</span>
  <span id="modelInfo">Model: Loading...</span>
</div>

<script>
const WS_URL = `ws://${window.location.host}/ws/dashboard`;
const SYMC = {BTCUSDT:'#f7931a', ETHUSDT:'#627eea', SOLUSDT:'#00ffa3'};
let totalSig=0, sent={BULLISH:0,BEARISH:0,NEUTRAL:0}, lats=[], divs=[];

function $(id){return document.getElementById(id)}
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString('en-US',{hour12:false})},1000);

// Price cards
(function(){
  const g=$('priceGrid'); g.innerHTML='';
  for(const [s,c] of Object.entries(SYMC))
    g.innerHTML+=`<div class="price-card"><div style="display:flex;align-items:center;gap:5px"><div style="width:7px;height:7px;border-radius:50%;background:${c}"></div><span class="price-sym">${s.replace('USDT','')}</span><span class="price-chg" id="chg_${s}" style="margin-left:auto">—</span></div><div class="price-val" id="p_${s}">$—</div></div>`;
})();

function addSignal(sig){
  totalSig++;
  const a=sig.action||'HOLD', cls=a.toLowerCase(), ico={BUY:'▲',SELL:'▼',HOLD:'●'}[a]||'●';
  const conf=((sig.confidence||0)*100).toFixed(0), lat=(sig.latency_ms||0).toFixed(1);
  const sc=sig.scores||{}, b=(sc.BULLISH||0)*100, n=(sc.NEUTRAL||0)*100, r=(sc.BEARISH||0)*100;
  const hasDiv=sig.divergence&&sig.divergence.divergence_type&&sig.divergence.divergence_type!=='NONE';
  const hasTrade=sig.trade_action&&sig.trade_action!=='no_action';

  sent[sig.sentiment]=(sent[sig.sentiment]||0)+1;
  if(sig.latency_ms)lats.push(sig.latency_ms);
  if(lats.length>100)lats.shift();

  const t=sent.BULLISH+sent.BEARISH+sent.NEUTRAL;
  $('sentBull').style.width=(sent.BULLISH/t*100)+'%';
  $('sentNeut').style.width=(sent.NEUTRAL/t*100)+'%';
  $('sentBear').style.width=(sent.BEARISH/t*100)+'%';
  $('sigCount').textContent=totalSig;
  $('sSignals').textContent=totalSig;
  if(lats.length)$('sLatency').innerHTML=(lats.reduce((a,b)=>a+b,0)/lats.length).toFixed(1)+'<span class="unit">ms</span>';

  const sc_=SYMC[sig.symbol]||'#94a8c4';
  const acCol={buy:'var(--bull)',sell:'var(--bear)',hold:'var(--neut)'}[cls]||'var(--dim)';
  const d=document.createElement('div');
  d.className='signal-card '+cls+(hasDiv?' divergence':'');
  d.innerHTML=`<div class="signal-top"><div style="display:flex;align-items:center;gap:5px"><span class="signal-action ${cls}">${ico} ${a}</span><span style="color:${sc_};font-weight:700;font-size:10px">${(sig.symbol||'').replace('USDT','')}</span><span class="badge" style="background:${acCol}22;color:${acCol}">${conf}%</span>${hasDiv?'<span class="badge" style="background:var(--div)22;color:var(--div)">DIV</span>':''}${hasTrade?'<span class="badge" style="background:var(--accent)22;color:var(--accent)">'+sig.trade_action.replace('opened_','').toUpperCase()+'</span>':''}</div><span style="color:var(--dim);font-size:8px;font-family:JetBrains Mono,monospace">${lat}ms</span></div><div class="signal-text">${sig.text||''}</div><div class="signal-bar"><div class="bull" style="width:${b}%"></div><div class="neut" style="width:${n}%"></div><div class="bear" style="width:${r}%"></div></div>`;
  const f=$('signalFeed');
  f.insertBefore(d,f.firstChild);
  while(f.children.length>60)f.removeChild(f.lastChild);
}

function handleDivergence(msg){
  divs.push(msg);
  if(divs.length>5)divs.shift();
  const el=$('divAlerts');
  el.innerHTML='<div class="div-alert"><div class="div-alert-header">⚡ DIVERGENCE ALERTS ('+divs.length+')</div>'+divs.slice(-3).map(d=>{
    const dt=d.divergence||{};
    const isBear=dt.divergence_type&&dt.divergence_type.includes('BEARISH');
    return `<div style="font-size:10px;margin-bottom:3px"><span style="color:${isBear?'var(--bear)':'var(--bull)'};font-weight:700">${d.symbol} ${(dt.divergence_type||'').replace(/_/g,' ')}</span> <span style="color:var(--dim)">sev: ${(dt.severity||0).toFixed(2)}</span></div>`;
  }).join('')+'</div>';
}

function handleTradeExec(msg){
  const el=document.createElement('div');
  el.className='signal-card';
  const isOpen=(msg.action||'').includes('opened');
  el.style.borderLeftColor=isOpen?'var(--accent)':'var(--neut)';
  const det=msg.details||{};
  el.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center"><span style="color:var(--accent);font-weight:700;font-family:JetBrains Mono,monospace;font-size:10px">${(msg.action||'').toUpperCase()}</span><span style="color:${SYMC[msg.symbol]||'var(--bright)'};font-weight:700;font-size:10px">${(msg.symbol||'').replace('USDT','')}</span></div>`;
  const f=$('tradeLog');
  f.insertBefore(el,f.firstChild);
  while(f.children.length>30)f.removeChild(f.lastChild);
  $('tradeLogCount').textContent=f.children.length;
}

function handlePortfolio(msg){
  const m=msg.metrics||{};
  const pnl=m.total_pnl||0;
  $('sEquity').textContent='$'+(m.equity||10000).toLocaleString();
  $('sEquity').style.color=pnl>=0?'var(--bull)':'var(--bear)';
  $('sPnl').textContent=(pnl>=0?'+$':'−$')+Math.abs(pnl).toLocaleString();
  $('sPnl').style.color=pnl>=0?'var(--bull)':'var(--bear)';
  $('sWinRate').textContent=((m.win_rate||0)*100).toFixed(0)+'%';
  $('sWinRate').style.color=(m.win_rate||0)>=0.5?'var(--bull)':'var(--bear)';
  $('sTrades').textContent=m.total_trades||0;
  $('sDrawdown').textContent=(m.current_drawdown_pct||0).toFixed(1)+'%';
  $('sDrawdown').style.color=(m.current_drawdown_pct||0)>5?'var(--bear)':'var(--bull)';

  const pos=msg.positions||[];
  $('posCount').textContent=pos.length;
  if(!pos.length){$('positions').innerHTML='<div style="font-size:10px;color:var(--dim);text-align:center;padding:8px">No open positions</div>';return;}
  $('positions').innerHTML=pos.map(p=>`<div class="position-card"><div style="display:flex;align-items:center;gap:5px"><span style="color:${SYMC[p.symbol]||'var(--bright)'};font-weight:700">${(p.symbol||'').replace('USDT','')}</span><span class="badge" style="background:${p.side==='LONG'?'var(--bull)':'var(--bear)'}22;color:${p.side==='LONG'?'var(--bull)':'var(--bear)'}">${p.side}</span>${p.divergence?'<span class="badge" style="background:var(--div)22;color:var(--div)">DIV</span>':''}</div><span class="mono" style="font-size:9px;color:var(--dim)">@ $${(p.entry_price||0).toLocaleString()}</span></div>`).join('');
}

function updateTicker(d){
  const s=d.symbol;if(!s||!SYMC[s])return;
  const el=$('p_'+s),ch=$('chg_'+s);
  if(el)el.textContent='$'+parseFloat(d.price||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  if(ch){const c=parseFloat(d.change_pct||0);ch.textContent=(c>=0?'▲ +':'▼ ')+Math.abs(c).toFixed(2)+'%';ch.style.color=c>=0?'var(--bull)':'var(--bear)';}
}

function updateStats(d){
  if(d.total_messages)$('sSignals').textContent=totalSig;
  if(d.msg_per_sec)$('sThroughput').innerHTML=d.msg_per_sec.toFixed(1)+'<span class="unit">/s</span>';
}

function connect(){
  const ws=new WebSocket(WS_URL);
  ws.onopen=()=>{$('statusDot').className='dot on';$('statusText').textContent='LIVE';};
  ws.onclose=()=>{$('statusDot').className='dot off';$('statusText').textContent='RECONNECTING';setTimeout(connect,3000);};
  ws.onerror=()=>ws.close();
  ws.onmessage=(e)=>{try{
    const m=JSON.parse(e.data);
    if(m.type==='snapshot')(m.signals||[]).forEach(addSignal);
    if(m.type==='signal'){addSignal(m);$('modelInfo').textContent='Model: '+(m.is_mock===false?'ONNX DistilBERT-LoRA':'Simulated');}
    if(m.type==='ticker')updateTicker(m);
    if(m.type==='stats')updateStats(m);
    if(m.type==='divergence')handleDivergence(m);
    if(m.type==='trade_execution')handleTradeExec(m);
    if(m.type==='trade_exit')handleTradeExec({...m.trade,action:'EXIT: '+(m.trade?.exit_reason||''),symbol:m.trade?.symbol});
    if(m.type==='portfolio')handlePortfolio(m);
  }catch(err){}};
}
connect();
</script>
</body>
</html>"""


@app.on_event("startup")
async def startup():
    global engine, detector, trader
    engine = SentimentEngine()
    detector = DivergenceDetector(DIVERGENCE_CONFIG)
    trader = PaperTradingEngine(TRADING_CONFIG)
    text_ctor = TextConstructor()

    asyncio.create_task(full_pipeline(engine, detector, trader, text_ctor, manager, state))
    asyncio.create_task(periodic_stats(trader, manager, state))

    print(f"\n[SERVER] Full pipeline ready at http://localhost:{SERVER_PORT}")
    print(f"[SERVER] Features: Sentiment + Divergence + Paper Trading")
    print(f"[SERVER] Starting capital: ${TRADING_CONFIG.starting_capital:,.2f}\n")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    any_price = list(state["prices"].values())[0] if state["prices"] else 0
    return {
        "status": "running",
        "model": "real" if (engine and not engine.is_mock) else "simulated",
        "features": ["sentiment", "divergence_detection", "paper_trading"],
        "portfolio": trader.get_metrics(any_price).to_dict() if trader else {},
        "open_positions": trader.get_open_positions() if trader else [],
    }


@app.get("/api/trades")
async def api_trades():
    """Get full trade journal."""
    if not trader:
        return {"trades": []}
    return {
        "trades": [t.to_dict() for t in trader.closed_trades[-50:]],
        "total": len(trader.closed_trades),
    }


@app.get("/api/divergences")
async def api_divergences():
    """Get current divergence state for all symbols."""
    if not detector:
        return {}
    return detector.get_all_states()


@app.post("/api/trading/reset")
async def reset_trading():
    """Reset paper trading to initial state."""
    if trader:
        trader.reset()
    return {"status": "reset", "capital": TRADING_CONFIG.starting_capital}


@app.post("/api/trading/export")
async def export_journal():
    """Export trade journal to JSON file."""
    if trader:
        path = trader.export_journal()
        return {"status": "exported", "path": path}
    return {"status": "no_trader"}


@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            # Handle dashboard commands
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "reset_trading" and trader:
                    trader.reset()
                    await ws.send_text(json.dumps({"type": "trading_reset", "capital": TRADING_CONFIG.starting_capital}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)

# ENTRY POINT
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  LZ-QUANT v2 — Divergence Detection + Paper Trading")
    print(f"{'='*70}")
    print(f"  Symbols         : {', '.join(s.upper() for s in SYMBOLS)}")
    print(f"  Starting Capital: ${TRADING_CONFIG.starting_capital:,.2f}")
    print(f"  Risk Per Trade  : {TRADING_CONFIG.risk_per_trade_pct}%")
    print(f"  Stop Loss       : {TRADING_CONFIG.stop_loss_pct}%")
    print(f"  Take Profit     : {TRADING_CONFIG.take_profit_pct}%")
    print(f"  Max Drawdown    : {TRADING_CONFIG.max_drawdown_pct}%")
    print(f"  Server          : http://localhost:{SERVER_PORT}")
    print(f"{'='*70}\n")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")