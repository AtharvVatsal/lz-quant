"""
REAL-TIME DASHBOARD — FastAPI Backend Server
Target Hardware : Intel i5-13th Gen H | 32GB DDR5 | NVIDIA RTX 3050 (4GB VRAM)
Framework       : FastAPI + uvicorn + WebSockets
Purpose         : Bridge between inference engine and React dashboard

Architecture:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Binance WebSocket → Inference Engine (runs inside server)             │
  └────────────────┬───────────────────────────────────────────────────────┘
                   │ signals, prices, latency metrics
                   ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  FastAPI Server (this file)                                            │
  │   ├─ GET  /              → serves React dashboard (index.html)         │
  │   ├─ GET  /api/status    → health check + pipeline stats               │
  │   ├─ WS   /ws/dashboard  → real-time stream to all connected UIs       │
  │   └─ CORS enabled        → React dev server can connect on :3000       │
  └────────────────┬───────────────────────────────────────────────────────┘
                   │ WebSocket push
                   ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  React Dashboard (browser)                                             │
  │   Live trade feed │ Sentiment gauges │ Price chart │ Signal log        │
  └────────────────────────────────────────────────────────────────────────┘

Install:
  pip install fastapi uvicorn[standard] websockets aiohttp

Run:
  python server.py
  → Server starts at http://localhost:8765
  → Dashboard WebSocket at ws://localhost:8765/ws/dashboard
"""

import asyncio
import json
import time
import signal
import sys
import os
import random
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    sys.exit("Missing: pip install fastapi uvicorn[standard]")

try:
    import websockets as ws_client
except ImportError:
    sys.exit("Missing: pip install websockets")

try:
    import numpy as np
except ImportError:
    sys.exit("Missing: pip install numpy")

# Optional: import ONNX model
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    from transformers import DistilBertTokenizerFast
    ONNX_AVAILABLE = True
except ImportError:
    pass

# 1. CONFIGURATION
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765

    # Binance
    symbols: list = field(default_factory=lambda: ["btcusdt", "ethusdt", "solusdt"])

    # Model (set to None to use simulated inference)
    onnx_model_path: Optional[str] = "./output/finbert-lora/sentiment_model.onnx"
    tokenizer_path: Optional[str] = "./output/finbert-lora"
    max_seq_length: int = 128

    # Signal thresholds
    bullish_threshold: float = 0.65
    bearish_threshold: float = 0.65

    # History limits
    max_signals: int = 200       # Keep last N signals for new dashboard connections
    max_price_points: int = 300  # Price history points per symbol


CONFIG = ServerConfig()

# 2. DASHBOARD CONNECTION MANAGER
class DashboardManager:
    """Manages all connected dashboard WebSocket clients."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        print(f"[DASHBOARD] Client connected ({len(self.active_connections)} total)")

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
        print(f"[DASHBOARD] Client disconnected ({len(self.active_connections)} total)")

    async def broadcast(self, message: dict):
        """Send a message to all connected dashboards."""
        if not self.active_connections:
            return
        payload = json.dumps(message)
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.active_connections.remove(conn)

# 3. MARKET STATE — Centralized state for the dashboard
class MarketState:
    """Tracks all live market data and signals for the dashboard."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.signals: deque = deque(maxlen=config.max_signals)
        self.price_history: dict[str, deque] = {
            s.upper(): deque(maxlen=config.max_price_points)
            for s in config.symbols
        }
        self.latest_prices: dict[str, float] = {}
        self.latest_tickers: dict[str, dict] = {}
        self.latest_depth: dict[str, dict] = {}
        self.latencies: deque = deque(maxlen=500)
        self.total_signals: int = 0
        self.total_messages: int = 0
        self.start_time: float = time.time()
        self.sentiment_counts: dict = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}

    def get_snapshot(self) -> dict:
        """Full state snapshot for newly connected dashboards."""
        return {
            "type": "snapshot",
            "signals": list(self.signals),
            "price_history": {
                sym: list(pts) for sym, pts in self.price_history.items()
            },
            "latest_prices": self.latest_prices,
            "latest_tickers": self.latest_tickers,
            "stats": self._stats(),
        }

    def _stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "total_signals": self.total_signals,
            "total_messages": self.total_messages,
            "uptime_seconds": round(uptime, 1),
            "msg_per_sec": round(self.total_messages / max(uptime, 1), 1),
            "avg_latency_ms": round(float(np.mean(self.latencies)), 2) if self.latencies else 0,
            "p99_latency_ms": round(float(np.percentile(self.latencies, 99)), 2) if self.latencies else 0,
            "sentiment_counts": self.sentiment_counts,
        }

# 4. SENTIMENT MODEL WRAPPER (Real or Simulated)
class SentimentEngine:
    """
    Wraps 1st ONNX model. Falls back to simulation if model not found.
    Simulation produces realistic-looking scores based on keyword heuristics
    so the dashboard looks convincing even without a trained model.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.session = None
        self.tokenizer = None
        self.is_mock = True

        if ONNX_AVAILABLE and config.onnx_model_path and os.path.exists(config.onnx_model_path):
            try:
                self._load_real_model(config)
                self.is_mock = False
                print("[MODEL] Real ONNX model loaded")
            except Exception as e:
                print(f"[MODEL] Failed to load ONNX model: {e}")
                print("[MODEL] Falling back to simulated inference")
        else:
            print("[MODEL] No ONNX model found — using simulated inference")
            print("[MODEL] Train and export ONNX to enable real predictions")

    def _load_real_model(self, config):
        # Use CPU for inference in the async server context.
        # CUDA + asyncio run_in_executor has threading issues on Windows.
        # CPU at ~55ms is fast enough for sentiment-driven trading.
        providers = ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(config.onnx_model_path, sess_opts, providers=providers)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.tokenizer_path)
        # Warmup
        self.predict("warmup sentence")

    def predict(self, text: str) -> tuple:
        """Returns (label, scores_dict, inference_ms)."""
        t0 = time.perf_counter_ns()

        if self.is_mock:
            scores = self._simulate_scores(text)
        else:
            encoded = self.tokenizer(text, padding="max_length", truncation=True,
                                     max_length=self.config.max_seq_length, return_tensors="np")
            logits = self.session.run(["logits"], {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            })[0]
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = (exp_l / exp_l.sum(axis=-1, keepdims=True)).squeeze()
            scores = {"BEARISH": float(probs[0]), "NEUTRAL": float(probs[1]), "BULLISH": float(probs[2])}

        t1 = time.perf_counter_ns()
        inference_ms = (t1 - t0) / 1_000_000

        label = max(scores, key=scores.get)
        return label, scores, inference_ms

    def _simulate_scores(self, text: str) -> dict:
        """Keyword-driven heuristic simulation for demo mode."""
        text_lower = text.lower()
        bull_kw = ["surge", "surging", "rising", "buying", "support", "high", "gain", "broke", "record"]
        bear_kw = ["plunging", "falling", "selling", "pressure", "drop", "low", "crash", "dump", "weak"]

        bull_hits = sum(1 for kw in bull_kw if kw in text_lower)
        bear_hits = sum(1 for kw in bear_kw if kw in text_lower)

        # Base distribution with noise
        if bull_hits > bear_hits:
            base = [0.10, 0.15, 0.75]
        elif bear_hits > bull_hits:
            base = [0.75, 0.15, 0.10]
        else:
            base = [0.20, 0.60, 0.20]

        # Add realistic noise
        noise = np.random.dirichlet([10, 10, 10])
        mixed = np.array(base) * 0.7 + noise * 0.3
        mixed = mixed / mixed.sum()

        return {"BEARISH": float(mixed[0]), "NEUTRAL": float(mixed[1]), "BULLISH": float(mixed[2])}

# 5. TEXT CONSTRUCTOR
class TextConstructor:
    def __init__(self):
        self._prices: dict[str, deque] = {}
        self._depths: dict[str, dict] = {}

    def from_trade(self, data: dict) -> str:
        symbol = data.get("s", "???")
        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
        is_sell = data.get("m", False)
        usd_value = price * qty

        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=50)
        self._prices[symbol].append(price)

        h = self._prices[symbol]
        chg = ((price - h[0]) / h[0] * 100) if len(h) > 1 and h[0] > 0 else 0.0

        if chg > 1.0: direction = "surging"
        elif chg > 0.3: direction = "rising"
        elif chg > 0.05: direction = "edging higher"
        elif chg < -1.0: direction = "plunging"
        elif chg < -0.3: direction = "falling"
        elif chg < -0.05: direction = "edging lower"
        else: direction = "trading flat"

        side = "sell-side trade" if is_sell else "buy-side trade"
        size = ""
        if usd_value > 100_000: size = " in a large block"
        elif usd_value > 50_000: size = " with notable size"

        book = ""
        if symbol in self._depths:
            r = self._depths[symbol].get("ratio", 1)
            if r > 1.5: book = ", order book bid-heavy"
            elif r < 0.65: book = ", order book ask-heavy"

        sign = "up" if chg >= 0 else "down"
        return f"{symbol} {direction} at ${price:,.2f}, {sign} {abs(chg):.2f}% on {side}{size}{book}"

    def from_ticker(self, data: dict) -> str:
        sym = data.get("s", "???")
        chg = float(data.get("P", 0))
        price = float(data.get("c", 0))
        vol = float(data.get("v", 0))
        if chg > 3: trend = "rallying strongly"
        elif chg > 1: trend = "trending higher"
        elif chg < -3: trend = "selling off sharply"
        elif chg < -1: trend = "trending lower"
        else: trend = "trading flat"
        return f"{sym} {trend}, 24h change {chg:+.2f}% at ${price:,.2f} on volume {vol:,.0f}"

    def from_depth(self, data: dict, symbol: str) -> str:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        bd = sum(float(b[1]) for b in bids) if bids else 0
        ad = sum(float(a[1]) for a in asks) if asks else 1
        ratio = bd / ad if ad > 0 else 1.0
        sym = symbol.upper()
        self._depths[sym] = {"ratio": ratio}

        if ratio > 1.5: p = "bids significantly outweigh asks"
        elif ratio > 1.15: p = "slightly more bids than asks"
        elif ratio < 0.65: p = "asks significantly outweigh bids"
        elif ratio < 0.85: p = "slightly more asks than bids"
        else: p = "balanced between bids and asks"
        return f"{sym} order book {p}, ratio {ratio:.2f}"


# 6. BINANCE STREAM → INFERENCE → BROADCAST PIPELINE
async def binance_pipeline(
    config: ServerConfig,
    state: MarketState,
    engine: SentimentEngine,
    manager: DashboardManager,
):
    """
    Connects to Binance, runs inference on every message,
    and broadcasts results to all connected dashboards.
    """
    text_ctor = TextConstructor()

    trade_streams = "/".join(f"{s}@trade" for s in config.symbols)
    ticker_streams = "/".join(f"{s}@miniTicker" for s in config.symbols)
    depth_streams = "/".join(f"{s}@depth10@1000ms" for s in config.symbols)
    all_streams = f"{trade_streams}/{ticker_streams}/{depth_streams}"
    url = f"wss://stream.binance.com:9443/stream?streams={all_streams}"

    ticker_count = 0
    depth_count = 0
    loop = asyncio.get_running_loop()

    while True:
        try:
            async with ws_client.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[STREAM] Connected to Binance ({len(config.symbols)} symbols)")

                async for raw_msg in ws:
                    t0 = time.perf_counter_ns()
                    state.total_messages += 1

                    data = json.loads(raw_msg)
                    stream_id = data.get("stream", "")
                    payload = data.get("data", data)
                    symbol = payload.get("s", stream_id.split("@")[0].upper())

                    # Determine stream type and build text
                    text = None
                    msg_type = None

                    if "@trade" in stream_id:
                        msg_type = "trade"
                        text = text_ctor.from_trade(payload)
                        price = float(payload.get("p", 0))
                        state.latest_prices[symbol] = price
                        state.price_history.setdefault(
                            symbol, deque(maxlen=config.max_price_points)
                        ).append({
                            "time": datetime.now(timezone.utc).isoformat(),
                            "price": price,
                        })
                        # Broadcast price tick (always)
                        await manager.broadcast({
                            "type": "price",
                            "symbol": symbol,
                            "price": price,
                            "time": datetime.now(timezone.utc).isoformat(),
                            "side": "sell" if payload.get("m") else "buy",
                            "qty": float(payload.get("q", 0)),
                        })

                    elif "@miniTicker" in stream_id:
                        msg_type = "ticker"
                        ticker_count += 1
                        text = text_ctor.from_ticker(payload)
                        state.latest_tickers[symbol] = {
                            "price": float(payload.get("c", 0)),
                            "change_pct": float(payload.get("P", 0)),
                            "high": float(payload.get("h", 0)),
                            "low": float(payload.get("l", 0)),
                            "volume": float(payload.get("v", 0)),
                        }
                        await manager.broadcast({
                            "type": "ticker",
                            "symbol": symbol,
                            **state.latest_tickers[symbol],
                        })
                        if ticker_count % 5 != 0:
                            text = None  # Throttle inference on tickers

                    elif "@depth" in stream_id:
                        msg_type = "depth"
                        depth_count += 1
                        stream_sym = stream_id.split("@")[0]
                        text = text_ctor.from_depth(payload, stream_sym)
                        if depth_count % 10 != 0:
                            text = None

                    # Run inference if we have text
                    if text:
                        label, scores, inference_ms = await loop.run_in_executor(
                            None, engine.predict, text
                        )

                        t1 = time.perf_counter_ns()
                        latency_ms = (t1 - t0) / 1_000_000
                        state.latencies.append(latency_ms)

                        bull = scores.get("BULLISH", 0)
                        bear = scores.get("BEARISH", 0)
                        if bull >= config.bullish_threshold:
                            action = "BUY"
                        elif bear >= config.bearish_threshold:
                            action = "SELL"
                        else:
                            action = "HOLD"

                        state.total_signals += 1
                        state.sentiment_counts[label] = state.sentiment_counts.get(label, 0) + 1

                        signal_data = {
                            "type": "signal",
                            "id": state.total_signals,
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
                        }

                        state.signals.append(signal_data)
                        await manager.broadcast(signal_data)

        except Exception as e:
            print(f"[STREAM] Connection error: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)


async def stats_broadcaster(state: MarketState, manager: DashboardManager):
    """Push pipeline health stats to dashboards every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        await manager.broadcast({
            "type": "stats",
            **state._stats(),
        })

# 7. FASTAPI APP
app = FastAPI(title="LZ-Quant Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialized on startup)
manager = DashboardManager()
state: Optional[MarketState] = None
engine: Optional[SentimentEngine] = None

# SELF-CONTAINED HTML DASHBOARD
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LZ-Quant Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #04080f; --surface: #0a1118; --border: #162236; --border-b: #1e3350;
    --text: #94a8c4; --dim: #4a5e78; --bright: #e2ecf8;
    --bull: #00e676; --bear: #ff1744; --neut: #ffab00; --accent: #448aff;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', sans-serif; padding: 16px; }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 10px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
  .title { font-size: 18px; font-weight: 800; color: var(--bright); }
  .title span { color: var(--accent); }
  .status { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--dim); }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot.on { background: var(--bull); box-shadow: 0 0 8px var(--bull); }
  .dot.off { background: var(--bear); box-shadow: 0 0 8px var(--bear); }
  .stats-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 12px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .stat-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: var(--dim); margin-bottom: 2px; }
  .stat-value { font-size: 18px; font-weight: 700; color: var(--bright); font-family: 'JetBrains Mono', monospace; }
  .stat-value .unit { font-size: 10px; color: var(--dim); }
  .main-grid { display: grid; grid-template-columns: 1fr 360px; gap: 12px; }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
  .panel-header { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--dim); margin-bottom: 8px; display: flex; justify-content: space-between; }
  .signal-feed { max-height: 70vh; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
  .signal-card { background: var(--bg); border: 1px solid var(--border); border-radius: 5px; padding: 8px 10px; font-size: 11px; border-left: 3px solid var(--dim); animation: fadeIn 0.3s ease; }
  .signal-card.buy { border-left-color: var(--bull); }
  .signal-card.sell { border-left-color: var(--bear); }
  .signal-card.hold { border-left-color: var(--neut); }
  .signal-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .signal-action { font-weight: 800; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  .signal-action.buy { color: var(--bull); }
  .signal-action.sell { color: var(--bear); }
  .signal-action.hold { color: var(--neut); }
  .signal-text { color: var(--dim); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }
  .signal-bar { display: flex; height: 3px; border-radius: 2px; overflow: hidden; }
  .signal-bar .bull { background: var(--bull); transition: width 0.3s; }
  .signal-bar .neut { background: var(--neut); transition: width 0.3s; }
  .signal-bar .bear { background: var(--bear); transition: width 0.3s; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 10px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .latency { color: var(--dim); font-size: 9px; font-family: 'JetBrains Mono', monospace; }
  .price-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
  .price-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; cursor: pointer; transition: all 0.2s; }
  .price-card:hover { border-color: var(--border-b); }
  .price-sym { font-weight: 700; font-size: 12px; color: var(--bright); }
  .price-val { font-size: 22px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--bright); margin: 4px 0; }
  .price-chg { font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .sent-bar-wrap { display: flex; height: 5px; border-radius: 3px; overflow: hidden; width: 100%; margin-bottom: 8px; }
  .footer { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 9px; color: var(--dim); display: flex; justify-content: space-between; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; transform: translateY(0); } }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="title"><span>⚡</span> LZ-QUANT</div>
    <div class="status"><div class="dot off" id="statusDot"></div><span id="statusText">CONNECTING</span></div>
  </div>
  <div class="mono" style="font-size:11px;color:var(--dim)" id="clock"></div>
</div>

<div class="stats-grid">
  <div class="stat-card"><div class="stat-label">Signals</div><div class="stat-value" id="statSignals">0</div></div>
  <div class="stat-card"><div class="stat-label">Messages</div><div class="stat-value" id="statMessages">0</div></div>
  <div class="stat-card"><div class="stat-label">Avg Latency</div><div class="stat-value" id="statLatency">—<span class="unit">ms</span></div></div>
  <div class="stat-card"><div class="stat-label">Throughput</div><div class="stat-value" id="statThroughput">—<span class="unit">/s</span></div></div>
  <div class="stat-card"><div class="stat-label">Bullish</div><div class="stat-value" style="color:var(--bull)" id="statBull">0%</div></div>
  <div class="stat-card"><div class="stat-label">Bearish</div><div class="stat-value" style="color:var(--bear)" id="statBear">0%</div></div>
</div>

<div class="price-grid" id="priceGrid"></div>

<div class="main-grid">
  <div class="panel">
    <div class="panel-header"><span>RECENT SIGNALS</span><span class="mono" id="signalCount">0</span></div>
    <div style="display:flex;height:5px;border-radius:3px;overflow:hidden;margin-bottom:10px">
      <div id="sentBull" style="background:var(--bull);width:33%;transition:width 0.3s"></div>
      <div id="sentNeut" style="background:var(--neut);width:34%;transition:width 0.3s"></div>
      <div id="sentBear" style="background:var(--bear);width:33%;transition:width 0.3s"></div>
    </div>
    <div class="signal-feed" id="signalFeed"></div>
  </div>
  <div class="panel">
    <div class="panel-header"><span>LIVE TRADE FEED</span><span class="mono" id="tradeCount">0</span></div>
    <div class="signal-feed" id="tradeFeed" style="font-family:'JetBrains Mono',monospace;font-size:10px"></div>
  </div>
</div>

<div class="footer">
  <span>LZ-QUANT —REAL-TIME DASHBOARD</span>
  <span id="modelInfo">Model: Loading...</span>
</div>

<script>
const WS_URL = `ws://${window.location.host}/ws/dashboard`;
const SYMBOLS = {BTCUSDT:'#f7931a', ETHUSDT:'#627eea', SOLUSDT:'#00ffa3'};
let totalSignals = 0, sentCounts = {BULLISH:0, BEARISH:0, NEUTRAL:0};
let latencies = [], tradeNum = 0;
let tickers = {};

function $(id) { return document.getElementById(id); }

// Clock
setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('en-US',{hour12:false}); }, 1000);

// Price cards
function initPriceCards() {
  const grid = $('priceGrid');
  grid.innerHTML = '';
  for (const [sym, color] of Object.entries(SYMBOLS)) {
    grid.innerHTML += `<div class="price-card" id="pc_${sym}">
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:8px;height:8px;border-radius:50%;background:${color}"></div>
        <span class="price-sym">${sym.replace('USDT','')}</span>
        <span class="price-chg" id="chg_${sym}" style="margin-left:auto">—</span>
      </div>
      <div class="price-val" id="price_${sym}">$—</div>
    </div>`;
  }
}
initPriceCards();

function addSignal(sig) {
  totalSignals++;
  const action = sig.action || 'HOLD';
  const cls = action.toLowerCase();
  const icons = {BUY:'▲', SELL:'▼', HOLD:'●'};
  const conf = ((sig.confidence||0)*100).toFixed(0);
  const lat = (sig.latency_ms||0).toFixed(1);
  const scores = sig.scores || {};
  const bull = ((scores.BULLISH||0)*100).toFixed(0);
  const neut = ((scores.NEUTRAL||0)*100).toFixed(0);
  const bear = ((scores.BEARISH||0)*100).toFixed(0);

  sentCounts[sig.sentiment] = (sentCounts[sig.sentiment]||0) + 1;
  if (sig.latency_ms) latencies.push(sig.latency_ms);
  if (latencies.length > 100) latencies.shift();

  const total = sentCounts.BULLISH + sentCounts.BEARISH + sentCounts.NEUTRAL;
  $('sentBull').style.width = (sentCounts.BULLISH/total*100)+'%';
  $('sentNeut').style.width = (sentCounts.NEUTRAL/total*100)+'%';
  $('sentBear').style.width = (sentCounts.BEARISH/total*100)+'%';
  $('statBull').textContent = (sentCounts.BULLISH/total*100).toFixed(0)+'%';
  $('statBear').textContent = (sentCounts.BEARISH/total*100).toFixed(0)+'%';
  $('signalCount').textContent = totalSignals;
  $('statSignals').textContent = totalSignals;

  if (latencies.length > 0) {
    const avg = latencies.reduce((a,b)=>a+b,0)/latencies.length;
    $('statLatency').innerHTML = avg.toFixed(1)+'<span class="unit">ms</span>';
  }

  const symColor = SYMBOLS[sig.symbol] || '#94a8c4';
  const card = document.createElement('div');
  card.className = 'signal-card ' + cls;
  card.innerHTML = `
    <div class="signal-top">
      <div style="display:flex;align-items:center;gap:6px">
        <span class="signal-action ${cls}">${icons[action]||'●'} ${action}</span>
        <span style="color:${symColor};font-weight:700">${(sig.symbol||'').replace('USDT','')}</span>
        <span class="badge" style="background:${cls==='buy'?'var(--bull)':cls==='sell'?'var(--bear)':'var(--neut)'}22;color:${cls==='buy'?'var(--bull)':cls==='sell'?'var(--bear)':'var(--neut)'}">${conf}%</span>
      </div>
      <span class="latency">${lat}ms</span>
    </div>
    <div class="signal-text">${sig.text||''}</div>
    <div class="signal-bar">
      <div class="bull" style="width:${bull}%"></div>
      <div class="neut" style="width:${neut}%"></div>
      <div class="bear" style="width:${bear}%"></div>
    </div>`;
  const feed = $('signalFeed');
  feed.insertBefore(card, feed.firstChild);
  while (feed.children.length > 80) feed.removeChild(feed.lastChild);
}

function addTrade(data) {
  tradeNum++;
  const sym = data.symbol || '???';
  const price = parseFloat(data.price||0);
  const side = data.side || 'buy';
  const color = side === 'sell' ? 'var(--bear)' : 'var(--bull)';
  const symColor = SYMBOLS[sym] || '#94a8c4';

  const row = document.createElement('div');
  row.style.cssText = 'padding:3px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;animation:fadeIn 0.2s ease';
  row.innerHTML = `<span style="color:${symColor}">${sym.replace('USDT','')}</span> <span style="color:${color}">${side.toUpperCase()}</span> <span>$${price.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</span>`;
  const feed = $('tradeFeed');
  feed.insertBefore(row, feed.firstChild);
  while (feed.children.length > 100) feed.removeChild(feed.lastChild);
  $('tradeCount').textContent = tradeNum;
}

function updateTicker(data) {
  const sym = data.symbol;
  if (!sym || !SYMBOLS[sym]) return;
  tickers[sym] = data;
  const el = $('price_'+sym);
  const chgEl = $('chg_'+sym);
  if (el) el.textContent = '$' + parseFloat(data.price||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  if (chgEl) {
    const chg = parseFloat(data.change_pct||0);
    chgEl.textContent = (chg>=0?'▲ +':'▼ ') + Math.abs(chg).toFixed(2) + '%';
    chgEl.style.color = chg >= 0 ? 'var(--bull)' : 'var(--bear)';
  }
}

function updateStats(data) {
  if (data.total_messages) $('statMessages').textContent = data.total_messages.toLocaleString();
  if (data.msg_per_sec) $('statThroughput').innerHTML = data.msg_per_sec.toFixed(1)+'<span class="unit">/s</span>';
}

// WebSocket
function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    $('statusDot').className = 'dot on';
    $('statusText').textContent = 'LIVE';
    $('modelInfo').textContent = 'Model: Connecting...';
  };
  ws.onclose = () => {
    $('statusDot').className = 'dot off';
    $('statusText').textContent = 'RECONNECTING';
    setTimeout(connect, 3000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'snapshot') {
        (msg.signals||[]).forEach(addSignal);
        $('modelInfo').textContent = 'Model: ' + (msg.signals?.[0]?.is_mock === false ? 'ONNX DistilBERT-LoRA' : 'Simulated');
      }
      if (msg.type === 'signal') {
        addSignal(msg);
        $('modelInfo').textContent = 'Model: ' + (msg.is_mock === false ? 'ONNX DistilBERT-LoRA' : 'Simulated');
      }
      if (msg.type === 'price') addTrade(msg);
      if (msg.type === 'ticker') updateTicker(msg);
      if (msg.type === 'stats') updateStats(msg);
    } catch(err) {}
  };
}
connect();
</script>
</body>
</html>"""


@app.on_event("startup")
async def startup():
    global state, engine
    state = MarketState(CONFIG)
    engine = SentimentEngine(CONFIG)

    # Launch the Binance → inference → broadcast pipeline
    asyncio.create_task(binance_pipeline(CONFIG, state, engine, manager))
    asyncio.create_task(stats_broadcaster(state, manager))
    print(f"\n[SERVER] Dashboard ready at http://localhost:{CONFIG.port}")
    print(f"[SERVER] WebSocket at ws://localhost:{CONFIG.port}/ws/dashboard\n")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "status": "running",
        "model": "real" if (engine and not engine.is_mock) else "simulated",
        "stats": state._stats() if state else {},
    })


@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send full state snapshot on connect
        if state:
            await ws.send_text(json.dumps(state.get_snapshot()))
        # Keep alive — listen for pings or config changes from frontend
        while True:
            data = await ws.receive_text()
            # Future: handle dashboard commands (change symbols, thresholds, etc.)
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)

# 8. ENTRY POINT
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  LZ-QUANT (Latency-Zero): Dashboard Server")
    print(f"{'='*70}")
    print(f"  Symbols : {', '.join(s.upper() for s in CONFIG.symbols)}")
    print(f"  Server  : http://localhost:{CONFIG.port}")
    print(f"  Model   : {'Checking...'}")
    print(f"{'='*70}\n")

    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )