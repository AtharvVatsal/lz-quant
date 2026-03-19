"""
LZ-QUANT — DUAL MARKET ENGINE (Crypto + Stocks)
Architecture:
PRICES:    Binance WebSocket (Crypto) + Alpaca WebSocket (NASDAQ Stocks)
SENTIMENT: Reddit (10 subreddits) + RSS (6 news feeds) → ONNX DistilBERT
TRADING:   Divergence detection + Paper trading engine
Setup:
  1. Create .env file:  ALPACA_API_KEY=xxx  ALPACA_SECRET_KEY=xxx
  2. pip install python-dotenv
  3. python dual.py --capital 10000
  4. Open http://localhost:8765

Data flow:
  Reddit/RSS text → ONNX inference → BUY/SELL/HOLD signal
  Binance/Alpaca  → price tracking → stop-loss/take-profit checks
  Both combined   → paper trading → dashboard
"""

import asyncio, csv, json, signal, sys, time, os
from datetime import datetime, timezone
from collections import deque
from typing import Optional
import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    sys.exit("pip install fastapi uvicorn[standard]")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    from transformers import DistilBertTokenizerFast
    ONNX_AVAILABLE = True
except ImportError:
    pass

from divergenceTrading import (
    DivergenceDetector, DivergenceConfig,
    PaperTradingEngine, TradingConfig, DivergenceType,
)
from dataIngestion import TextRouter

# CONFIG
SERVER_PORT = 8765
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

CRYPTO_SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN", "SPY", "QQQ"]

DIV_CFG = DivergenceConfig(sentiment_window=25, price_window=25, zscore_window=60,
    sentiment_zscore_threshold=0.8, price_zscore_threshold=0.8, cooldown_ticks=15)
TRADE_CFG = TradingConfig(
    starting_capital=10_000,
    risk_per_trade_pct=25.0,       # 25% of equity per trade ($2,500 initially)
    max_position_pct=50.0,         # Up to 50% equity per position (simulates 2x leverage)
    stop_loss_pct=1.5,             # Cut losers at 1.5%
    take_profit_pct=2.5,           # Lock gains at 2.5%
    trailing_stop_pct=1.0,         # Trail 1% below high water
    max_drawdown_pct=30.0,         # Circuit breaker at 30%
    min_confidence=0.45,           # Trade on any directional lean
    max_open_positions=6,          # Up to 6 concurrent positions
    require_divergence=False,
    divergence_boost=2.0,          # Double size on divergence
)

# TRADE CSV LOGGER
class TradeCSVLogger:
    def __init__(self, filename="trades_log.csv"):
        self.filename = filename
        self._created = False
        self._write_header()

    def _write_header(self):
        if self._created:
            return
        headers = ["timestamp", "symbol", "market", "side", "action", "entry_price", 
                   "exit_price", "quantity", "entry_time", "exit_time", "entry_signal",
                   "exit_reason", "pnl", "pnl_pct", "holding_duration_s", "had_divergence",
                   "source", "sentiment", "confidence", "divergence_type", "divergence_severity"]
        with open(self.filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        self._created = True
        print(f"[CSV] Trade log started: {self.filename}")

    def log_trade(self, trade_data):
        if not trade_data:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": trade_data.get("symbol", ""),
            "market": trade_data.get("market", ""),
            "side": trade_data.get("side", ""),
            "action": trade_data.get("action", ""),
            "entry_price": trade_data.get("entry_price", 0),
            "exit_price": trade_data.get("exit_price", 0),
            "quantity": trade_data.get("quantity", 0),
            "entry_time": trade_data.get("entry_time", ""),
            "exit_time": trade_data.get("exit_time", ""),
            "entry_signal": trade_data.get("entry_signal", ""),
            "exit_reason": trade_data.get("exit_reason", ""),
            "pnl": trade_data.get("pnl", 0),
            "pnl_pct": trade_data.get("pnl_pct", 0),
            "holding_duration_s": trade_data.get("holding_duration_s", 0),
            "had_divergence": trade_data.get("had_divergence", False),
            "source": trade_data.get("source", ""),
            "sentiment": trade_data.get("sentiment", ""),
            "confidence": trade_data.get("confidence", 0),
            "divergence_type": trade_data.get("divergence_type", ""),
            "divergence_severity": trade_data.get("divergence_severity", 0),
        }
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def log_open_trade(self, trade_data):
        if not trade_data:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": trade_data.get("symbol", ""),
            "market": trade_data.get("market", ""),
            "side": trade_data.get("side", ""),
            "action": trade_data.get("action", ""),
            "entry_price": trade_data.get("entry_price", 0),
            "exit_price": "",
            "quantity": trade_data.get("quantity", 0),
            "entry_time": trade_data.get("entry_time", ""),
            "exit_time": "",
            "entry_signal": trade_data.get("entry_signal", ""),
            "exit_reason": "",
            "pnl": "",
            "pnl_pct": "",
            "holding_duration_s": "",
            "had_divergence": trade_data.get("had_divergence", False),
            "source": trade_data.get("source", ""),
            "sentiment": trade_data.get("sentiment", ""),
            "confidence": trade_data.get("confidence", 0),
            "divergence_type": trade_data.get("divergence_type", ""),
            "divergence_severity": trade_data.get("divergence_severity", 0),
        }
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

# SENTIMENT ENGINE
class SentimentEngine:
    def __init__(self, model_path="./output/finbert-lora/sentiment_model.onnx",
                 tokenizer_path="./output/finbert-lora"):
        self.session=None; self.tokenizer=None; self.is_mock=True
        if ONNX_AVAILABLE and os.path.exists(model_path):
            try:
                opts=ort.SessionOptions()
                opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads=4
                self.session=ort.InferenceSession(model_path,opts,providers=["CPUExecutionProvider"])
                self.tokenizer=DistilBertTokenizerFast.from_pretrained(tokenizer_path)
                self.is_mock=False; self.predict("warmup")
                print("[MODEL] ONNX loaded (CPU)")
            except Exception as e:
                print(f"[MODEL] Failed: {e}")
        else:
            print("[MODEL] Simulated inference")

    def predict(self, text):
        t0=time.perf_counter_ns()
        if not self.is_mock:
            enc=self.tokenizer(text,padding="max_length",truncation=True,max_length=128,return_tensors="np")
            logits=self.session.run(["logits"],{"input_ids":enc["input_ids"].astype(np.int64),"attention_mask":enc["attention_mask"].astype(np.int64)})[0]
            e=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
            p=(e/e.sum(axis=-1,keepdims=True)).squeeze()
            scores={"BEARISH":float(p[0]),"NEUTRAL":float(p[1]),"BULLISH":float(p[2])}
        else:
            t=text.lower()
            bh=sum(1 for k in ["surge","rising","rally","gain","beat","record","high"] if k in t)
            brh=sum(1 for k in ["plunging","falling","sell","drop","low","miss","loss","crash"] if k in t)
            if bh>brh: base=[0.10,0.15,0.75]
            elif brh>bh: base=[0.75,0.15,0.10]
            else: base=[0.20,0.60,0.20]
            noise=np.random.dirichlet([10,10,10])
            m=np.array(base)*0.7+noise*0.3; m/=m.sum()
            scores={"BEARISH":float(m[0]),"NEUTRAL":float(m[1]),"BULLISH":float(m[2])}
        ms=(time.perf_counter_ns()-t0)/1e6
        return max(scores,key=scores.get), scores, ms

# DASHBOARD MANAGER
class DashManager:
    def __init__(self): self.conns=[]
    async def connect(self, ws): await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws):
        if ws in self.conns: self.conns.remove(ws)
    async def broadcast(self, msg):
        if not self.conns: return
        payload=json.dumps(msg,default=str); dead=[]
        for c in self.conns:
            try: await c.send_text(payload)
            except: dead.append(c)
        for c in dead: self.conns.remove(c)

# MARKET COMMENTARY GENERATOR
class MarketCommentary:
    def __init__(self):
        self.prices = {}
        self.last_prices = {}
        self.trends = {}
    
    def generate_commentary(self, symbol, price, change_pct=0):
        self.prices[symbol] = price
        last = self.last_prices.get(symbol, price)
        change = ((price - last) / last * 100) if last > 0 else 0
        self.last_prices[symbol] = price
        
        trend = self.trends.get(symbol, 0)
        self.trends[symbol] = trend * 0.9 + change * 0.1
        
        direction = "rising" if change > 0 else ("falling" if change < 0 else "stable")
        trend_desc = "surge" if trend > 1 else ("drop" if trend < -1 else direction)
        action = "buy" if change > 0 else ("sell" if change < 0 else "hold")
        
        crypto_prefix = {
            "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum", 
            "SOLUSDT": "Solana"
        }.get(symbol, symbol)
        
        return f"{crypto_prefix} trading at ${price:.2f}, {trend_desc} {abs(change):.2f}%. Market showing {action} pressure. {action.title()} signal active."
commentary_gen = MarketCommentary()

# INFERENCE HELPER
async def run_inference_and_broadcast(engine, detector, trader, manager, state,
                                       symbol, text, price, market, loop, csv_logger=None,
                                       source="live_feed"):
    """Shared inference + signal generation for both crypto and stocks."""
    global is_paused
    
    t0 = time.perf_counter_ns()
    current_price = state["prices"].get(symbol, price)
    label, scores, inference_ms = await loop.run_in_executor(None, engine.predict, text)
    div_signal = detector.update(symbol, scores, current_price)

    bull = scores.get("BULLISH", 0)
    bear = scores.get("BEARISH", 0)
    action = "BUY" if bull >= 0.45 else ("SELL" if bear >= 0.45 else "HOLD")

    if div_signal.divergence_type == DivergenceType.BEARISH_DIVERGENCE.value and div_signal.severity > 0.3:
        action = "SELL"
    elif div_signal.divergence_type == DivergenceType.BULLISH_DIVERGENCE.value and div_signal.severity > 0.3:
        action = "BUY"

    # Only process trades if not paused
    if not is_paused:
        trade_result = trader.process_signal(symbol, action, current_price,
                                              max(scores.values()), label, div_signal)
    else:
        trade_result = {"action_taken": "paused", "trade": None, "position": None}
        action = "HOLD"  # Override action when paused

    t1 = time.perf_counter_ns()
    latency_ms = (t1 - t0) / 1e6
    state["total_signals"] += 1

    await manager.broadcast({
        "type": "signal", "market": market,
        "id": state["total_signals"], "symbol": symbol,
        "action": action, "sentiment": label,
        "confidence": round(max(scores.values()), 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "text": text, "latency_ms": round(latency_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "time": datetime.now(timezone.utc).isoformat(),
        "is_mock": engine.is_mock,
        "divergence": div_signal.to_dict(),
        "trade_action": trade_result["action_taken"],
        "is_paused": is_paused,
    })

    # Broadcast trade execution events
    if trade_result["action_taken"] not in ("no_action", "halted"):
        trade_details = trade_result.get("trade") or trade_result.get("position") or {}
        await manager.broadcast({
            "type": "trade_execution", "market": market,
            "symbol": symbol,
            "action": trade_result["action_taken"],
            "details": trade_details,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        # Log to CSV
        if csv_logger and trade_details:
            csv_logger.log_open_trade({
                **trade_details,
                "market": market,
                "action": trade_result["action_taken"],
                "source": source,
                "sentiment": label,
                "confidence": round(max(scores.values()), 4),
                "divergence_type": div_signal.divergence_type,
                "divergence_severity": div_signal.severity,
            })
    # Check auto-exits (stop-loss, take-profit, trailing stop)
    valid_prices = {s: p for s, p in state["prices"].items() if p > 0}
    auto_exits = trader.update_prices(valid_prices)
    for exit_trade in auto_exits:
        await manager.broadcast({
            "type": "trade_exit", "market": market,
            "trade": exit_trade,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        # Log closed trade to CSV
        if csv_logger:
            csv_logger.log_trade({
                **exit_trade,
                "market": market,
                "source": source,
                "sentiment": label,
                "confidence": round(max(scores.values()), 4),
                "divergence_type": div_signal.divergence_type,
                "divergence_severity": div_signal.severity,
            })

    if div_signal.divergence_type != DivergenceType.NONE.value:
        await manager.broadcast({
            "type": "divergence", "market": market,
            "symbol": symbol, "divergence": div_signal.to_dict(),
            "time": datetime.now(timezone.utc).isoformat(),
        })

# BINANCE PIPELINE (price tracking + inference)
async def binance_pipeline(trader, manager, state, engine, detector):
    """Streams Binance price data for tracking, inference, and trading."""
    streams = "/".join(f"{s}@trade" for s in CRYPTO_SYMBOLS)
    streams += "/" + "/".join(f"{s}@miniTicker" for s in CRYPTO_SYMBOLS)
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    loop = asyncio.get_running_loop()
    last_inference = {s: 0 for s in CRYPTO_SYMBOLS}
    INFERENCE_INTERVAL = 2  # seconds between inferences

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[CRYPTO] Binance connected — tracking ({len(CRYPTO_SYMBOLS)} symbols) + inference")
                async for raw in ws:
                    state["total_messages"] += 1
                    data = json.loads(raw)
                    sid = data.get("stream", ""); pl = data.get("data", data)
                    sym = pl.get("s", sid.split("@")[0].upper())
                    sym_lower = sym.lower()

                    if "@trade" in sid:
                        price = float(pl.get("p", 0))
                        change_pct = float(pl.get("P", 0))
                        if price > 0:
                            state["prices"][sym] = price
                        await manager.broadcast({"type":"price","market":"crypto","symbol":sym,"price":price,"time":datetime.now(timezone.utc).isoformat(),"side":"sell" if pl.get("m") else "buy"})

                        # Run inference on price commentary
                        current_time = time.time()
                        if sym_lower in CRYPTO_SYMBOLS and current_time - last_inference.get(sym_lower, 0) >= INFERENCE_INTERVAL:
                            commentary = commentary_gen.generate_commentary(sym, price, change_pct)
                            await run_inference_and_broadcast(
                                engine, detector, trader, manager, state,
                                sym, commentary, price, "crypto", loop, csv_logger, "binance"
                            )
                            last_inference[sym_lower] = current_time

                    elif "@miniTicker" in sid:
                        price = float(pl.get("c", 0))
                        if price > 0:
                            state["prices"][sym] = price
                        await manager.broadcast({"type":"ticker","market":"crypto","symbol":sym,"price":price,"change_pct":float(pl.get("P",0))})

        except Exception as e:
            print(f"[CRYPTO] Error: {e}. Reconnecting...")
            await asyncio.sleep(3)

# ALPACA PIPELINE (price tracking + inference)
async def alpaca_pipeline(trader, manager, state, engine, detector):
    """Streams Alpaca stock price data for tracking, inference, and trading."""
    if not ALPACA_API_KEY:
        print("[STOCKS] No Alpaca keys — disabled"); return

    url = "wss://stream.data.alpaca.markets/v2/iex"
    loop = asyncio.get_running_loop()
    last_inference = {s: 0 for s in STOCK_SYMBOLS}
    INFERENCE_INTERVAL = 2

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                connected_msg = json.loads(await ws.recv())
                await ws.send(json.dumps({"action":"auth","key":ALPACA_API_KEY,"secret":ALPACA_SECRET_KEY}))
                resp = json.loads(await ws.recv())
                auth_ok = any(m.get("msg")=="authenticated" for m in (resp if isinstance(resp,list) else [resp]))
                if not auth_ok:
                    print(f"[STOCKS] Auth failed: {resp}")
                    await asyncio.sleep(30); continue

                await ws.send(json.dumps({"action":"subscribe","trades":STOCK_SYMBOLS}))
                await ws.recv()
                print(f"[STOCKS] Alpaca connected — tracking ({len(STOCK_SYMBOLS)} symbols) + inference")

                async for raw in ws:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list): msgs = [msgs]

                    for msg in msgs:
                        mt = msg.get("T",""); sym = msg.get("S","")
                        if not sym or mt != "t": continue

                        state["total_messages"] += 1
                        price = float(msg.get("p",0))
                        if price > 0:
                            state["prices"][sym] = price
                        await manager.broadcast({"type":"price","market":"stocks","symbol":sym,"price":price,"time":datetime.now(timezone.utc).isoformat()})
                        await manager.broadcast({"type":"ticker","market":"stocks","symbol":sym,"price":price,"change_pct":0})

                        # Run inference on price commentary
                        current_time = time.time()
                        if sym in STOCK_SYMBOLS and current_time - last_inference.get(sym, 0) >= INFERENCE_INTERVAL:
                            stock_commentary = f"{sym} stock trading at ${price:.2f}. Market analysis shows trading activity. {'Bullish' if price > state['prices'].get(sym + '_prev', price) else 'Bearish'} momentum."
                            state[sym + '_prev'] = price
                            await run_inference_and_broadcast(
                                engine, detector, trader, manager, state,
                                sym, stock_commentary, price, "stocks", loop, csv_logger, "alpaca"
                            )
                            last_inference[sym] = current_time

        except Exception as e:
            print(f"[STOCKS] Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# REAL TEXT SENTIMENT CONSUMER
async def text_sentiment_consumer(router, engine, detector, trader, manager, state, csv_logger=None):
    """
    Consumes REAL text from Reddit and RSS feeds, runs inference, trades.
    
    Old (circular):  price tick → synthetic text → model → confirm what happened
    New (predictive): Reddit post → model → BULLISH/BEARISH → trade BEFORE price moves
    """
    loop = asyncio.get_running_loop()
    print("[SENTIMENT] Real text consumer started — waiting for Reddit/RSS data...")

    while True:
        item = await router.get_next()
        if item is None:
            continue

        text = item.text
        symbol = item.symbol
        market = item.market
        source = item.source

        if not text or not symbol:
            continue

        price = state["prices"].get(symbol, 0)
        if price <= 0:
            continue  # Can't trade without a live price

        # Run inference on REAL human text
        await run_inference_and_broadcast(
            engine, detector, trader, manager, state,
            symbol, text, price, market, loop, csv_logger, f"{item.source_type}:{source}"
        )

        # Broadcast source info for dashboard
        await manager.broadcast({
            "type": "ingestion",
            "source": source,
            "source_type": item.source_type,
            "market": market,
            "symbol": symbol,
            "text_preview": text[:150],
            "score": item.score,
            "url": item.url,
            "time": datetime.now(timezone.utc).isoformat(),
        })


# PERIODIC PRICE CHECK (stop-loss/take-profit)
async def periodic_price_check(trader, manager, state):
    """Check stop-loss and take-profit every 2 seconds using latest prices."""
    while True:
        await asyncio.sleep(2)
        valid_prices = {s: p for s, p in state["prices"].items() if p > 0}
        if valid_prices:
            auto_exits = trader.update_prices(valid_prices)
            for exit_trade in auto_exits:
                market = "crypto" if any(c.upper() in (exit_trade.get("symbol","")) for c in CRYPTO_SYMBOLS) else "stocks"
                await manager.broadcast({
                    "type": "trade_exit", "market": market,
                    "trade": exit_trade,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

# PERIODIC STATS
async def periodic_stats(trader, manager, state):
    while True:
        await asyncio.sleep(5)
        metrics = trader.get_metrics()
        ingestion = state.get("router").get_stats() if state.get("router") else {}
        await manager.broadcast({"type":"stats","total_signals":state["total_signals"],
            "total_messages":state["total_messages"],
            "uptime_seconds":round(time.time()-state["start_time"],1),
            "msg_per_sec":round(state["total_messages"]/max(time.time()-state["start_time"],1),1),
            "ingestion": ingestion})
        await manager.broadcast({"type":"portfolio","metrics":metrics.to_dict(),
            "positions":trader.get_open_positions(),"equity_curve":trader.equity_curve[-100:],
            "starting_capital": trader.starting_capital})

# FASTAPI
app = FastAPI(title="LZ-Quant Dual Market")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if os.path.exists("logos"):
    app.mount("/logos", StaticFiles(directory="logos"), name="logos")

mgr = DashManager()
state = {"total_signals":0,"total_messages":0,"prices":{},"start_time":time.time(),"router":None}
engine=None; detector=None; trader=None; router=None; csv_logger=None
is_paused=False

DASH = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="logos/favicon.png" type="image/png">
<link rel="shortcut icon" href="logos/favicon.png" type="image/png">
<title>LZ-Quant Dual Market</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#04080f;--sf:#0a1118;--bd:#162236;--tx:#94a8c4;--dm:#4a5e78;--br:#e2ecf8;--bull:#00e676;--bear:#ff1744;--neut:#ffab00;--acc:#448aff;--div:#e040fb;--crypto:#f7931a;--stock:#4fc3f7}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,'Segoe UI',sans-serif;padding:12px;font-size:13px}
.mono{font-family:'JetBrains Mono',monospace}
.hd{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid var(--bd);margin-bottom:8px}
.tt{font-size:18px;font-weight:800;color:var(--br)}.tt span{color:var(--acc)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}.dot.on{background:var(--bull);box-shadow:0 0 6px var(--bull)}.dot.off{background:var(--bear)}
.sg{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;margin-bottom:8px}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:6px;padding:8px}
.sl{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--dm)}
.sv{font-size:16px;font-weight:700;color:var(--br);font-family:'JetBrains Mono',monospace}.sv .u{font-size:10px;color:var(--dm)}
.dual{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mp{display:flex;flex-direction:column;gap:6px}
.mh{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;padding:6px 10px;border-radius:4px;display:flex;align-items:center;gap:6px}
.mh-c{background:#f7931a12;color:var(--crypto);border:1px solid #f7931a25}
.mh-s{background:#4fc3f712;color:var(--stock);border:1px solid #4fc3f725}
.pn{background:var(--sf);border:1px solid var(--bd);border-radius:6px;padding:10px}
.ph{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--dm);margin-bottom:6px;display:flex;justify-content:space-between}
.pg{display:grid;gap:4px}
.pc{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:6px 10px;display:flex;justify-content:space-between;align-items:center}
.ps{font-weight:700;font-size:12px;color:var(--br)}
.pp{font-size:14px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--br)}
.ch{font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace}
.fd{max-height:40vh;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
.si{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:6px 8px;font-size:11px;border-left:3px solid var(--dm);animation:fi .2s ease}
.si.buy{border-left-color:var(--bull)}.si.sell{border-left-color:var(--bear)}.si.hold{border-left-color:var(--neut)}
.st{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.sa{font-weight:800;font-family:'JetBrains Mono',monospace;font-size:11px}
.sa.buy{color:var(--bull)}.sa.sell{color:var(--bear)}.sa.hold{color:var(--neut)}
.sx{color:var(--dm);font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
.sb{display:flex;height:4px;border-radius:2px;overflow:hidden}
.ba{display:inline-block;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:700;font-family:'JetBrains Mono',monospace}
.bt{display:inline-block;padding:6px 14px;border-radius:6px;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;cursor:pointer;border:none}
.bt-pause{background:rgba(255,23,68,0.15);color:#ff1744;border:1px solid rgba(255,23,68,0.4)}
.bt-resume{background:rgba(0,230,118,0.15);color:#00e676;border:1px solid rgba(0,230,118,0.4)}
.ft{margin-top:8px;padding-top:6px;border-top:1px solid var(--bd);font-size:9px;color:var(--dm);display:flex;justify-content:space-between}
@keyframes fi{from{opacity:0;transform:translateY(-2px)}to{opacity:1;transform:translateY(0)}}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
.an{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.ac{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px}
.al{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--dm);margin-bottom:6px}
.av{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--br)}
.at{font-size:10px;color:var(--dm);margin-top:4px}
.ex{grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}
.ep{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:16px}
.en{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--dm);margin-bottom:10px}
.eb{display:flex;flex-direction:column;gap:8px}
.er{display:flex;justify-content:space-between;font-size:12px;padding:4px 0}
.er-l{color:var(--dm)}.er-v{font-family:'JetBrains Mono',monospace;color:var(--br);font-weight:600}
.sgauge{display:flex;align-items:center;gap:16px;margin-top:10px}
.gbar{flex:1;height:20px;background:var(--sf);border-radius:10px;overflow:hidden;display:flex}
.gb{height:100%;transition:width .3s ease}
.gb-b{background:var(--bull)}.gb-n{background:var(--neut)}.gb-r{background:var(--bear)}
.gstats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;flex:1}
.gs{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:12px;text-align:center}
.gs-v{font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace}
.gs-l{font-size:10px;text-transform:uppercase;color:var(--dm);margin-top:4px}
.sett{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.se{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px}
.sl2{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--dm);margin-bottom:10px}
.si2{display:flex;align-items:center;gap:8px}
.sr{width:100%;height:6px;border-radius:3px;background:var(--bd);appearance:none;cursor:pointer}
.sr::-webkit-slider-thumb{appearance:none;width:16px;height:16px;border-radius:50%;background:var(--acc);cursor:pointer}
.sv2{font-size:13px;font-family:'JetBrains Mono',monospace;color:var(--br);min-width:40px;text-align:right}
.sb2{margin-top:12px;background:var(--acc);color:#fff;border:none;padding:12px 20px;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;width:100%}
.tk{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px;margin-top:10px}
.tr2{font-size:12px;padding:6px 0;border-bottom:1px solid var(--bd)}
.tr2:last-child{border:none}
.tr-l{color:var(--dm)}
.tr-v{font-family:'JetBrains Mono',monospace;color:var(--br);float:right;font-size:13px}
.toggle{cursor:pointer;background:var(--sf);border:1px solid var(--bd);padding:8px 16px;border-radius:6px;font-size:11px;font-weight:700;color:var(--dm);margin-left:8px}
.toggle.active{background:var(--acc);color:#fff;border-color:var(--acc)}
.toggle:hover{background:var(--bd)}
</style></head><body>
<div class="hd"><div style="display:flex;align-items:center;gap:10px"><div class="tt">⚡ LZ-QUANT <span style="color:var(--dm);font-size:11px;font-weight:400">DUAL MARKET</span></div><div style="display:flex;align-items:center;gap:4px;font-size:11px;color:var(--dm)"><div class="dot off" id="dot"></div><span id="st">CONNECTING</span></div><span id="pausedBadge" style="display:none;background:rgba(255,23,68,0.15);color:#ff1744;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700">⏸ PAUSED</span></div><div style="display:flex;align-items:center;gap:10px"><button id="pauseBtn" class="bt bt-pause" onclick="togglePause()">⏸ Pause</button><span class="mono" style="font-size:11px;color:var(--dm)" id="ck"></span></div></div>
<div class="sg">
<div class="sc"><div class="sl">Signals</div><div class="sv" id="nS">0</div></div>
<div class="sc"><div class="sl">Latency</div><div class="sv" id="nL">—<span class="u">ms</span></div></div>
<div class="sc"><div class="sl">Throughput</div><div class="sv" id="nT">—<span class="u">/s</span></div></div>
<div class="sc"><div class="sl">Equity</div><div class="sv" id="nE">$10,000</div></div>
<div class="sc"><div class="sl">P&L</div><div class="sv" id="nP">$0</div></div>
<div class="sc"><div class="sl">Win Rate</div><div class="sv" id="nW">—</div></div>
<div class="sc"><div class="sl">Trades</div><div class="sv" id="nTr">0</div></div>
<div class="sc"><div class="sl">Positions</div><div class="sv" id="nPo">0</div></div>
</div>
<div class="dual">
<div class="mp"><div class="mh mh-c">₿ CRYPTO — BINANCE <span class="mono" style="margin-left:auto;font-size:8px" id="cN">0</span></div><div class="pn"><div class="ph"><span>PRICES</span></div><div class="pg" id="cP"></div></div><div class="pn" style="flex:1"><div class="ph"><span>SIGNALS</span><span class="mono" id="cS">0</span></div><div class="fd" id="cF"></div></div></div>
<div class="mp"><div class="mh mh-s">📈 STOCKS — NASDAQ <span class="mono" style="margin-left:auto;font-size:8px" id="sN">0</span></div><div class="pn"><div class="ph"><span>PRICES</span></div><div class="pg" id="sP"></div></div><div class="pn" style="flex:1"><div class="ph"><span>SIGNALS</span><span class="mono" id="sS">0</span></div><div class="fd" id="sF"></div></div></div>
</div>

<div class="sgauge" id="sentPanel">
<div style="min-width:80px"><div class="al">MARKET SENTIMENT</div><div class="gstats"><div class="gs"><div class="gs-v" id="sBull" style="color:var(--bull)">—</div><div class="gs-l">Bull</div></div><div class="gs"><div class="gs-v" id="sNeut" style="color:var(--neut)">—</div><div class="gs-l">Neut</div></div><div class="gs"><div class="gs-v" id="sBear" style="color:var(--bear)">—</div><div class="gs-l">Bear</div></div></div></div>
<div class="gbar"><div class="gb gb-b" id="gBull" style="width:33%"></div><div class="gb gb-n" id="gNeut" style="width:33%"></div><div class="gb gb-r" id="gBear" style="width:34%"></div></div>
</div>

<div class="tk" id="tkPanel">
<div class="en">TRADE ANALYTICS</div>
<div class="tr2"><span class="tr-l">Win Rate</span><span class="tr-v" id="taWR">—</span></div>
<div class="tr2"><span class="tr-l">Profit Factor</span><span class="tr-v" id="taPF">—</span></div>
<div class="tr2"><span class="tr-l">Avg Holding Time</span><span class="tr-v" id="taAHT">—</span></div>
<div class="tr2"><span class="tr-l">Best Symbol</span><span class="tr-v" id="taBS">—</span></div>
<div class="tr2"><span class="tr-l">Worst Symbol</span><span class="tr-v" id="taWS">—</span></div>
<div class="tr2"><span class="tr-l">Total P&L</span><span class="tr-v" id="taPnL">—</span></div>
</div>

<div class="an" id="anPanel">
<div class="ac"><div class="al">Sharpe Ratio</div><div class="av" id="sharpe">—</div><div class="at">Risk-adjusted return</div></div>
<div class="ac"><div class="al">Profit Factor</div><div class="av" id="pf">—</div><div class="at">Gross profit / loss</div></div>
<div class="ac"><div class="al">Avg Trade</div><div class="av" id="avgT">—</div><div class="at">Per closed trade</div></div>
<div class="ac"><div class="al">Best Trade</div><div class="av" id="bestT">—</div><div class="at">Highest single trade</div></div>
<div class="ac"><div class="al">Worst Trade</div><div class="av" id="worstT">—</div><div class="at">Lowest single trade</div></div>
<div class="ac"><div class="al">Avg Hold Time</div><div class="av" id="avgH">—</div><div class="at">Position duration</div></div>
<div class="ac"><div class="al">Max Drawdown</div><div class="av" id="maxDD">—</div><div class="at">Peak-to-trough loss</div></div>
<div class="ac"><div class="al">Calmar Ratio</div><div class="av" id="calmar">—</div><div class="at">Return / max drawdown</div></div>
</div>

<div class="ex" id="riskPanel">
<div class="ep"><div class="en">RISK METRICS</div><div class="eb"><div class="er2"><span class="tr-l">Total Exposure</span><span class="tr-v" id="tExp">—</span></div><div class="er2"><span class="tr-l">Crypto Exposure</span><span class="tr-v" id="cExp">—</span></div><div class="er2"><span class="tr-l">Stock Exposure</span><span class="tr-v" id="sExp">—</span></div><div class="er2"><span class="tr-l">Open Positions</span><span class="tr-v" id="oPos">—</span></div><div class="er2"><span class="tr-l">Max Positions</span><span class="tr-v" id="mPos">—</span></div></div></div>
<div class="ep"><div class="en">POSITION BREAKDOWN</div><div class="eb" id="posBreak"></div></div>
<div class="ep"><div class="en">TOP SYMBOLS</div><div class="eb" id="topSym"></div></div>
</div>

<div id="settPanel" style="display:none">
<div class="sett">
<div class="se"><div class="sl2">MIN CONFIDENCE</div><div class="si2"><input type="range" class="sr" id="sConf" min="30" max="90" value="45"><span class="sv2" id="sConfV">45%</span></div></div>
<div class="se"><div class="sl2">RISK PER TRADE</div><div class="si2"><input type="range" class="sr" id="sRisk" min="1" max="50" value="25"><span class="sv2" id="sRiskV">25%</span></div></div>
<div class="se"><div class="sl2">STOP LOSS</div><div class="si2"><input type="range" class="sr" id="sSL" min="1" max="10" value="15"><span class="sv2" id="sSLV">1.5%</span></div></div>
<div class="se"><div class="sl2">TAKE PROFIT</div><div class="si2"><input type="range" class="sr" id="sTP" min="1" max="20" value="25"><span class="sv2" id="sTPV">2.5%</span></div></div>
<div class="se"><div class="sl2">MAX POSITIONS</div><div class="si2"><input type="range" class="sr" id="sMP" min="1" max="10" value="6"><span class="sv2" id="sMPV">6</span></div></div>
<div class="se"><div class="sl2">MAX DRAWDOWN</div><div class="si2"><input type="range" class="sr" id="sDD" min="5" max="50" value="30"><span class="sv2" id="sDDV">30%</span></div></div>
<div class="se"><div class="sl2">DIVERGENCE BOOST</div><div class="si2"><input type="range" class="sr" id="sDiv" min="10" max="30" value="20"><span class="sv2" id="sDivV">2.0x</span></div></div>
<div class="se"><div class="sl2">REQUIRE DIVERGENCE</div><div class="si2" style="justify-content:center"><button class="toggle" id="sReqD" onclick="toggleReqDiv()">OFF</button></div></div>
</div>
<button class="sb2" onclick="saveSettings()">APPLY SETTINGS</button>
</div>

<div class="tk" id="tkPanel" style="display:none">
<div class="en">TRADE ANALYTICS</div>
<div class="tr2"><span class="tr-l">Win Rate</span><span class="tr-v" id="taWR">—</span></div>
<div class="tr2"><span class="tr-l">Profit Factor</span><span class="tr-v" id="taPF">—</span></div>
<div class="tr2"><span class="tr-l">Avg Holding Time</span><span class="tr-v" id="taAHT">—</span></div>
<div class="tr2"><span class="tr-l">Best Symbol</span><span class="tr-v" id="taBS">—</span></div>
<div class="tr2"><span class="tr-l">Worst Symbol</span><span class="tr-v" id="taWS">—</span></div>
<div class="tr2"><span class="tr-l">Total P&L</span><span class="tr-v" id="taPnL">—</span></div>
</div>

<div class="ft"><button class="toggle" id="btnSett" onclick="togglePanel('sett')" style="margin-left:0">⚙ SETTINGS</button><span style="flex:1"></span><span>LZ-QUANT — CRYPTO (BINANCE) + STOCKS (ALPACA/NASDAQ)</span><span id="mi" style="margin-left:20px">Model: Loading...</span></div>
<script>
const W=`ws://${location.host}/ws/dashboard`;
const CC={BTCUSDT:'#f7931a',ETHUSDT:'#627eea',SOLUSDT:'#00ffa3'};
const SC={AAPL:'#555',MSFT:'#00a4ef',GOOGL:'#4285f4',NVDA:'#76b900',TSLA:'#c00',AMD:'#ed1c24',META:'#0668E1',AMZN:'#ff9900',SPY:'#b8860b',QQQ:'#8b008b'};
let ls=[],cSg=0,sSg=0,tSg=0;
let bullCnt=0,neutCnt=0,bearCnt=0;
let tradeHistory=[],closedTrades=[];
let startCap=10000;
const $=id=>document.getElementById(id);
setInterval(()=>{$('ck').textContent=new Date().toLocaleTimeString('en-US',{hour12:false})},1000);

function togglePanel(p){
  const panel=$('settPanel');
  const btn=$('btnSett');
  if(p==='sett'){
    const isHidden=panel.style.display==='none';
    panel.style.display=isHidden?'grid':'none';
    btn.className='toggle'+(isHidden?' active':'');
  }
}

function toggleReqDiv(){
  const btn=$('sReqD');
  btn.textContent=btn.textContent==='OFF'?'ON':'OFF';
  btn.className=btn.textContent==='ON'?'toggle active':'toggle';
}

document.querySelectorAll('.sr').forEach(sr=>{
  sr.addEventListener('input',()=>{
    const id=sr.id;
    if(id==='sConf')$('sConfV').textContent=sr.value+'%';
    if(id==='sRisk')$('sRiskV').textContent=sr.value+'%';
    if(id==='sSL')$('sSLV').textContent=(sr.value/10).toFixed(1)+'%';
    if(id==='sTP')$('sTPV').textContent=(sr.value/10).toFixed(1)+'%';
    if(id==='sMP')$('sMPV').textContent=sr.value;
    if(id==='sDD')$('sDDV').textContent=sr.value+'%';
    if(id==='sDiv')$('sDivV').textContent=(sr.value/10).toFixed(1)+'x';
  });
});

function saveSettings(){
  const cfg={
    min_confidence:$('sConf').value/100,
    risk_per_trade_pct:parseFloat($('sRisk').value),
    stop_loss_pct:parseFloat($('sSL').value)/10,
    take_profit_pct:parseFloat($('sTP').value)/10,
    max_open_positions:parseInt($('sMP').value),
    max_drawdown_pct:parseFloat($('sDD').value),
    divergence_boost:parseFloat($('sDiv').value)/10,
    require_divergence:$('sReqD').textContent==='ON'
  };
  fetch(location.protocol+'//'+location.host+'/api/trading/set-config',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(cfg)
  }).then(r=>r.json()).then(d=>{
    if(d.status==='ok'){
      const btn=document.querySelector('.sb2');
      btn.textContent='SAVED ✓';
      btn.style.background='#00e676';
      setTimeout(()=>{btn.textContent='APPLY SETTINGS';btn.style.background='';},1500);
    }
  });
}

function updateAnalytics(){
  if(closedTrades.length<2){
    $('sharpe').textContent='—';
    $('pf').textContent='—';
    $('avgT').textContent='—';
    $('bestT').textContent='—';
    $('worstT').textContent='—';
    $('avgH').textContent='—';
    $('maxDD').textContent='—';
    $('calmar').textContent='—';
    return;
  }
  const pnls=closedTrades.map(t=>t.pnl);
  const wins=pnls.filter(p=>p>0);
  const losses=pnls.filter(p=>p<=0);
  const totalWin=wins.reduce((a,b)=>a+b,0);
  const totalLoss=Math.abs(losses.reduce((a,b)=>a+b,0));
  const pf=totalLoss>0?(totalWin/totalLoss).toFixed(2):totalWin>0?'∞':'—';
  const avgT=(pnls.reduce((a,b)=>a+b,0)/pnls.length).toFixed(2);
  const bestT=Math.max(...pnls).toFixed(2);
  const worstT=Math.min(...pnls).toFixed(2);
  const holdTimes=closedTrades.filter(t=>t.holding_duration_s>0).map(t=>t.holding_duration_s);
  const avgH=holdTimes.length>0?(holdTimes.reduce((a,b)=>a+b,0)/holdTimes.length/60).toFixed(1)+'m':'—';
  const peak=Math.max(...closedTrades.map((t,i)=>pnls.slice(0,i+1).reduce((a,b)=>a+b,0)));
  const maxDD=closedTrades.reduce((max,_,i)=>{
    const cumm=pnls.slice(0,i+1).reduce((a,b)=>a+b,0);
    return Math.max(max,peak-cumm);
  },0);
  const maxDDpct=(maxDD/startCap*100).toFixed(1)+'%';
  const totalReturn=pnls.reduce((a,b)=>a+b,0);
  const calmar=maxDD>0?(totalReturn/maxDD).toFixed(2):'—';
  const returns=pnls.map(p=>p/startCap);
  const meanR=returns.reduce((a,b)=>a+b,0)/returns.length;
  const stdR=Math.sqrt(returns.map(r=>Math.pow(r-meanR,2)).reduce((a,b)=>a+b,0)/returns.length);
  const sharpe=stdR>0?(meanR/stdR*Math.sqrt(252)).toFixed(2):'—';
  $('sharpe').textContent=sharpe;
  $('pf').textContent=pf;
  $('avgT').textContent='$'+avgT;
  $('bestT').textContent='$'+bestT;
  $('worstT').textContent='$'+worstT;
  $('avgH').textContent=avgH;
  $('maxDD').textContent=maxDDpct;
  $('calmar').textContent=calmar;
}

function updateRiskMetrics(pf){
  const pos=(pf.positions||[]);
  const totalExp=pos.reduce((s,p)=>s+(p.notional||0),0);
  const cExp=pos.filter(p=>['BTCUSDT','ETHUSDT','SOLUSDT'].includes(p.symbol)).reduce((s,p)=>s+(p.notional||0),0);
  const sExp=totalExp-cExp;
  const equity=pf.equity||startCap;
  $('tExp').textContent=(totalExp/equity*100).toFixed(0)+'%';
  $('cExp').textContent=(cExp/equity*100).toFixed(0)+'%';
  $('sExp').textContent=(sExp/equity*100).toFixed(0)+'%';
  $('oPos').textContent=pos.length;
  $('mPos').textContent=pf.max_open_positions||6;
  const breakEl=$('posBreak');
  breakEl.innerHTML=pos.length===0?'<div class="er2"><span class="tr-l">No positions</span></div>':
    pos.slice(0,5).map(p=>`<div class="er2"><span class="tr-l">${(p.symbol||'').replace('USDT','')}</span><span class="tr-v">$${(p.notional||0).toFixed(0)}</span></div>`).join('');
  const symCounts={};
  closedTrades.forEach(t=>{
    if(!symCounts[t.symbol])symCounts[t.symbol]={count:0,pnl:0};
    symCounts[t.symbol].count++;
    symCounts[t.symbol].pnl+=(t.pnl||0);
  });
  const topSym=Object.entries(symCounts).sort((a,b)=>b[1].count-a[1].count).slice(0,4);
  const topEl=$('topSym');
  topEl.innerHTML=topSym.length===0?'<div class="er2"><span class="tr-l">No trades yet</span></div>':
    topSym.map(([sym,d])=>`<div class="er2"><span class="tr-l">${sym.replace('USDT','')}</span><span class="tr-v" style="color:${d.pnl>=0?'var(--bull)':'var(--bear)'}">$${d.pnl.toFixed(0)}</span></div>`).join('');
}

function updateSentiment(){
  const total=bullCnt+neutCnt+bearCnt;
  if(total===0)return;
  const bPct=(bullCnt/total*100).toFixed(0);
  const nPct=(neutCnt/total*100).toFixed(0);
  const rPct=(bearCnt/total*100).toFixed(0);
  $('sBull').textContent=bPct+'%';
  $('sNeut').textContent=nPct+'%';
  $('sBear').textContent=rPct+'%';
  $('gBull').style.width=bPct+'%';
  $('gNeut').style.width=nPct+'%';
  $('gBear').style.width=rPct+'%';
}

function updateTradeAnalytics(){
  if(closedTrades.length===0){
    $('taWR').textContent='—';
    $('taPF').textContent='—';
    $('taAHT').textContent='—';
    $('taBS').textContent='—';
    $('taWS').textContent='—';
    $('taPnL').textContent='—';
    return;
  }
  const wins=closedTrades.filter(t=>(t.pnl||0)>0).length;
  const wr=(wins/closedTrades.length*100).toFixed(0)+'%';
  const pnls=closedTrades.map(t=>t.pnl||0);
  const wins2=pnls.filter(p=>p>0);
  const losses=pnls.filter(p=>p<=0);
  const pf=losses.length>0?(wins2.reduce((a,b)=>a+b,0)/Math.abs(losses.reduce((a,b)=>a+b,0))).toFixed(2):'∞';
  const holdTimes=closedTrades.filter(t=>(t.holding_duration_s||0)>0).map(t=>t.holding_duration_s/60);
  const avgHT=holdTimes.length>0?(holdTimes.reduce((a,b)=>a+b,0)/holdTimes.length).toFixed(1)+'m':'—';
  const symPnL={};
  closedTrades.forEach(t=>{
    if(!symPnL[t.symbol])symPnL[t.symbol]=0;
    symPnL[t.symbol]+=(t.pnl||0);
  });
  const sortedSym=Object.entries(symPnL).sort((a,b)=>b[1]-a[1]);
  const bestSym=sortedSym.length>0?sortedSym[0][0].replace('USDT',''):'—';
  const worstSym=sortedSym.length>0?sortedSym[sortedSym.length-1][0].replace('USDT',''):'—';
  const totalPnL=pnls.reduce((a,b)=>a+b,0);
  $('taWR').textContent=wr;
  $('taPF').textContent=pf;
  $('taAHT').textContent=avgHT;
  $('taBS').textContent=bestSym;
  $('taWS').textContent=worstSym;
  $('taPnL').textContent=(totalPnL>=0?'+$':'-$')+Math.abs(totalPnL).toFixed(2);
  $('taPnL').style.color=totalPnL>=0?'var(--bull)':'var(--bear)';
}

function addSig(s){
tSg++;const m=s.market||'crypto',ic=m==='crypto';if(ic)cSg++;else sSg++;
$('cS').textContent=cSg;$('sS').textContent=sSg;$('nS').textContent=tSg;
if(s.latency_ms){ls.push(s.latency_ms);if(ls.length>100)ls.shift();$('nL').innerHTML=(ls.reduce((a,b)=>a+b,0)/ls.length).toFixed(1)+'<span class="u">ms</span>';}
const a=s.action||'HOLD',c=a.toLowerCase(),ik={BUY:'▲',SELL:'▼',HOLD:'●'}[a]||'●';
const cf=((s.confidence||0)*100).toFixed(0),lt=(s.latency_ms||0).toFixed(1);
const sc=s.scores||{},bu=(sc.BULLISH||0)*100,ne=(sc.NEUTRAL||0)*100,be=(sc.BEARISH||0)*100;
if(bu>ne&&bu>be){bullCnt++;}else if(be>ne&&be>bu){bearCnt++;}else{neutCnt++;}
updateSentiment();
const hd=s.divergence&&s.divergence.divergence_type&&s.divergence.divergence_type!=='NONE';
const pa=s.trade_action==='paused'||s.is_paused;
const cols=ic?CC:SC,scl=cols[s.symbol]||'var(--br)';
const ac={buy:'var(--bull)',sell:'var(--bear)',hold:'var(--neut)'}[c]||'var(--dm)';
const d=document.createElement('div');d.className='si '+c;
d.innerHTML=`<div class="st"><div style="display:flex;align-items:center;gap:3px"><span class="sa ${c}">${ik} ${a}</span><span style="color:${scl};font-weight:700;font-size:8px">${s.symbol||''}</span><span class="ba" style="background:${ac}22;color:${ac}">${cf}%</span>${hd?'<span class="ba" style="background:var(--div)22;color:var(--div)">DIV</span>':''}${pa?'<span class="ba" style="background:rgba(255,23,68,0.3);color:#ff1744">PAUSED</span>':''}</div><span style="color:var(--dm);font-size:6px;font-family:JetBrains Mono,monospace">${lt}ms</span></div><div class="sx">${s.text||''}</div><div class="sb"><div style="width:${bu}%;background:var(--bull)"></div><div style="width:${ne}%;background:var(--neut)"></div><div style="width:${be}%;background:var(--bear)"></div></div>`;
const f=$(ic?'cF':'sF');f.insertBefore(d,f.firstChild);while(f.children.length>40)f.removeChild(f.lastChild);
$(ic?'cN':'sN').textContent=ic?cSg:sSg;}

function uP(m){const ic=m.market==='crypto',cid=ic?'cP':'sP',sym=m.symbol,p=parseFloat(m.price||0),cols=ic?CC:SC,cl=cols[sym]||'var(--br)';
let e=document.getElementById('p_'+sym);if(!e){e=document.createElement('div');e.className='pc';e.id='p_'+sym;
e.innerHTML=`<div style="display:flex;align-items:center;gap:4px"><div style="width:5px;height:5px;border-radius:50%;background:${cl}"></div><span class="ps">${sym.replace('USDT','')}</span></div><div style="display:flex;align-items:center;gap:6px"><span class="ch" id="c_${sym}"></span><span class="pp" id="v_${sym}">—</span></div>`;
$(cid).appendChild(e);}
const v=$('v_'+sym);if(v)v.textContent='$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}

function uT(m){uP(m);const c=$('c_'+m.symbol);if(c&&m.change_pct!==undefined){const v=parseFloat(m.change_pct||0);c.textContent=(v>=0?'▲+':'▼')+Math.abs(v).toFixed(2)+'%';c.style.color=v>=0?'var(--bull)':'var(--bear)';}}

function uPf(m){const p=m.metrics||{},pnl=p.total_pnl||0;if(m.starting_capital)startCap=m.starting_capital;
$('nE').textContent='$'+(p.equity||startCap).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});$('nE').style.color=pnl>=0?'var(--bull)':'var(--bear)';
$('nP').textContent=(pnl>=0?'+$':'−$')+Math.abs(pnl).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});$('nP').style.color=pnl>=0?'var(--bull)':'var(--bear)';
$('nW').textContent=((p.win_rate||0)*100).toFixed(0)+'%';$('nTr').textContent=p.total_trades||0;
$('nPo').textContent=(m.positions||[]).length;
if(m.positions)$('nPo').textContent=m.positions.length;
updateRiskMetrics(m);
if(m.positions&&m.positions.length>0)$('oPos').textContent=m.positions.length;
if(m.max_open_positions)$('mPos').textContent=m.max_open_positions;
}

function uSt(d){if(d.msg_per_sec)$('nT').innerHTML=d.msg_per_sec.toFixed(1)+'<span class="u">/s</span>';}

function handleTrade(t){
  if(t.type==='trade_execution'){
    tradeHistory.push({...t,time:Date.now()});
  }
  if(t.type==='trade_exit'){
    const existing=tradeHistory.find(h=>h.symbol===t.symbol&&!h.closed);
    if(existing){
      existing.closed=true;
      existing.exit_price=t.exit_price;
      existing.pnl=t.pnl;
      existing.exit_time=Date.now();
      existing.holding_duration_s=(existing.exit_time-existing.time)/1000;
      closedTrades.push({...existing,...t});
      updateAnalytics();
      updateTradeAnalytics();
    }
  }
}

let isPaused=false;
function togglePause(){
  const endpoint=isPaused?'/api/trading/resume':'/api/trading/pause';
  fetch(location.protocol+'//'+location.host+endpoint,{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.status==='paused'){isPaused=true;$('pauseBtn').textContent='▶ Resume';$('pauseBtn').className='bt bt-resume';$('pausedBadge').style.display='inline-block';}
    if(d.status==='resumed'){isPaused=false;$('pauseBtn').textContent='⏸ Pause';$('pauseBtn').className='bt bt-pause';$('pausedBadge').style.display='none';}
  });
}

function conn(){const ws=new WebSocket(W);
ws.onopen=()=>{$('dot').className='dot on';$('st').textContent='LIVE';};
ws.onclose=()=>{$('dot').className='dot off';$('st').textContent='RECONNECTING';setTimeout(conn,3000);};
ws.onerror=()=>ws.close();
ws.onmessage=e=>{try{const m=JSON.parse(e.data);
if(m.type==='trading_status'){if(m.is_paused){isPaused=true;$('pauseBtn').textContent='▶ Resume';$('pauseBtn').className='bt bt-resume';$('pausedBadge').style.display='inline-block';}else{isPaused=false;$('pauseBtn').textContent='⏸ Pause';$('pauseBtn').className='bt bt-pause';$('pausedBadge').style.display='none';}}
if(m.type==='signal'){addSig(m);$('mi').textContent='Model: '+(m.is_mock===false?'ONNX DistilBERT-LoRA':'Simulated');}
if(m.type==='price')uP(m);if(m.type==='ticker')uT(m);
if(m.type==='stats')uSt(m);if(m.type==='portfolio')uPf(m);
if(m.type==='trade_execution'||m.type==='trade_exit')handleTrade(m);
if(m.type==='snapshot')(m.signals||[]).forEach(addSig);
if(m.trades&&m.trades.forEach){(m.trades||[]).forEach(t=>{if(t.action&&t.action.includes('closed'))closedTrades.push(t);});updateAnalytics();updateTradeAnalytics();}
}catch(e){}};}
conn();
</script></body></html>"""

@app.on_event("startup")
async def startup():
    global engine, detector, trader, router, csv_logger
    engine = SentimentEngine()
    detector = DivergenceDetector(DIV_CFG)
    trader = PaperTradingEngine(TRADE_CFG)
    csv_logger = TradeCSVLogger()

    # Real text ingestion (Reddit + RSS)
    router = TextRouter()
    state["router"] = router
    await router.start()

    # Price tracking + inference pipelines
    asyncio.create_task(binance_pipeline(trader, mgr, state, engine, detector))
    asyncio.create_task(alpaca_pipeline(trader, mgr, state, engine, detector))

    # Sentiment consumer: reads real text → inference → trading signals
    asyncio.create_task(text_sentiment_consumer(router, engine, detector, trader, mgr, state, csv_logger))

    # Periodic tasks
    asyncio.create_task(periodic_stats(trader, mgr, state))
    asyncio.create_task(periodic_price_check(trader, mgr, state))

    has_alpaca = bool(ALPACA_API_KEY)
    print(f"\n[SERVER] http://localhost:{SERVER_PORT}")
    print(f"[SERVER] Crypto prices: Binance ({len(CRYPTO_SYMBOLS)})")
    print(f"[SERVER] Stock prices: {'Alpaca ('+str(len(STOCK_SYMBOLS))+')' if has_alpaca else 'DISABLED'}")
    print(f"[SERVER] Sentiment: Reddit + RSS → ONNX inference")
    print(f"[SERVER] Capital: ${TRADE_CFG.starting_capital:,.2f}\n")

@app.get("/",response_class=HTMLResponse)
async def dashboard(): return DASH

@app.get("/api/status")
async def status():
    capital = trader.starting_capital if trader else 10000
    metrics = trader.get_metrics() if trader else {}
    ing = router.get_stats() if router else {}
    return {
        "status": "running",
        "model": "real" if engine and not engine.is_mock else "simulated",
        "crypto": len(CRYPTO_SYMBOLS),
        "stocks": len(STOCK_SYMBOLS) if ALPACA_API_KEY else 0,
        "capital": capital,
        "equity": metrics.equity if hasattr(metrics, "equity") else capital,
        "ingestion": ing,
    }

@app.get("/api/ingestion")
async def ingestion_stats():
    if not router: return {"status": "not started"}
    return router.get_stats()

@app.get("/api/trades")
async def trades():
    return {"trades":[t.to_dict() for t in trader.closed_trades[-50:]] if trader else [],"total":len(trader.closed_trades) if trader else 0}

@app.post("/api/trading/set-capital")
async def set_capital(request: dict):
    """Set starting capital and reset all trades."""
    amount = float(request.get("capital", 10000))
    if amount < 100:
        return {"error": "Minimum capital is $100"}
    if amount > 10_000_000:
        return {"error": "Maximum capital is $10,000,000"}
    if trader:
        trader.starting_capital = amount
        trader.config.starting_capital = amount
        trader.config.max_position_pct = 50.0
        trader.reset()
        print(f"[CAPITAL] Set to ${amount:,.2f} — trader reset")
    return {"status": "ok", "capital": amount}

@app.post("/api/trading/reset")
async def reset():
    if trader: trader.reset()
    return {"status": "reset", "capital": trader.starting_capital if trader else 10000}

@app.post("/api/trading/pause")
async def pause():
    global is_paused
    is_paused = True
    print(f"[TRADING] Paused at {datetime.now(timezone.utc).isoformat()}")
    await mgr.broadcast({"type": "trading_status", "is_paused": True, "time": datetime.now(timezone.utc).isoformat()})
    return {"status": "paused", "paused_at": datetime.now(timezone.utc).isoformat()}

@app.post("/api/trading/resume")
async def resume():
    global is_paused
    is_paused = False
    print(f"[TRADING] Resumed at {datetime.now(timezone.utc).isoformat()}")
    await mgr.broadcast({"type": "trading_status", "is_paused": False, "time": datetime.now(timezone.utc).isoformat()})
    return {"status": "resumed", "resumed_at": datetime.now(timezone.utc).isoformat()}

@app.get("/api/trading/status")
async def trading_status():
    return {"is_paused": is_paused}

@app.post("/api/trading/set-config")
async def set_config(req: dict):
    global TRADE_CFG
    if "min_confidence" in req: TRADE_CFG.min_confidence = req["min_confidence"]
    if "risk_per_trade_pct" in req: TRADE_CFG.risk_per_trade_pct = req["risk_per_trade_pct"]
    if "stop_loss_pct" in req: TRADE_CFG.stop_loss_pct = req["stop_loss_pct"]
    if "take_profit_pct" in req: TRADE_CFG.take_profit_pct = req["take_profit_pct"]
    if "max_open_positions" in req: TRADE_CFG.max_open_positions = req["max_open_positions"]
    if "max_drawdown_pct" in req: TRADE_CFG.max_drawdown_pct = req["max_drawdown_pct"]
    if "divergence_boost" in req: TRADE_CFG.divergence_boost = req["divergence_boost"]
    if "require_divergence" in req: TRADE_CFG.require_divergence = req["require_divergence"]
    if trader: trader.config = TRADE_CFG
    print(f"[CONFIG] Updated: confidence={TRADE_CFG.min_confidence}, risk={TRADE_CFG.risk_per_trade_pct}%, stop={TRADE_CFG.stop_loss_pct}%, profit={TRADE_CFG.take_profit_pct}%, max_pos={TRADE_CFG.max_open_positions}, div_boost={TRADE_CFG.divergence_boost}x, req_div={TRADE_CFG.require_divergence}")
    return {"status": "ok", "config": {"min_confidence": TRADE_CFG.min_confidence, "risk_per_trade_pct": TRADE_CFG.risk_per_trade_pct, "stop_loss_pct": TRADE_CFG.stop_loss_pct, "take_profit_pct": TRADE_CFG.take_profit_pct, "max_open_positions": TRADE_CFG.max_open_positions, "max_drawdown_pct": TRADE_CFG.max_drawdown_pct, "divergence_boost": TRADE_CFG.divergence_boost, "require_divergence": TRADE_CFG.require_divergence}}

@app.websocket("/ws/dashboard")
async def ws_dash(ws:WebSocket):
    await mgr.connect(ws)
    try:
        while True: await ws.receive_text()
    except: mgr.disconnect(ws)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LZ-Quant Dual Market Engine")
    parser.add_argument("--capital", "--capacity", dest="capital", type=float, default=10_000, help="Starting capital (default: 10000)")
    parser.add_argument("--risk", type=float, default=25.0, help="Risk per trade %% (default: 25)")
    parser.add_argument("--max-pos", type=int, default=6, help="Max open positions (default: 6)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    args = parser.parse_args()

    # Override config with user input
    TRADE_CFG.starting_capital = args.capital
    TRADE_CFG.risk_per_trade_pct = args.risk
    TRADE_CFG.max_open_positions = args.max_pos
    TRADE_CFG.max_position_pct = min(args.risk * 2, 80.0)  # Cap at 80%
    SERVER_PORT = args.port

    print(f"\n  LZ-QUANT — DUAL MARKET ENGINE\n")
    print(f"  Capital   : ${args.capital:,.2f}")
    print(f"  Risk      : {args.risk}% per trade (${args.capital * args.risk / 100:,.2f})")
    print(f"  Max Pos   : {args.max_pos}")
    print(f"  Prices    : Binance ({', '.join(s.upper() for s in CRYPTO_SYMBOLS)})")
    print(f"            : Alpaca ({', '.join(STOCK_SYMBOLS)})")
    print(f"  Sentiment : 10 Reddit subs + 6 RSS news feeds")
    print(f"  Server    : http://localhost:{SERVER_PORT}")
    if not ALPACA_API_KEY: print(f"\n  !!No ALPACA keys — stock prices disabled!!")
    uvicorn.run(app,host="0.0.0.0",port=SERVER_PORT,log_level="warning")