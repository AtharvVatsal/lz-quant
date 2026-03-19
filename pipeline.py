"""
REAL-TIME DATA PIPELINE — The "Nervous System"
Target Hardware : Intel i5-13th Gen H | 32GB DDR5 | NVIDIA RTX 3050 (4GB VRAM)
Framework       : Python asyncio + websockets + aiohttp
Data Source     : Binance Public WebSocket API (FREE, no API key, no account)

This script connects to multiple live Binance streams simultaneously:
  1. Trade stream     — Every individual trade (BTC, ETH, SOL) as it happens
  2. Ticker stream    — 24h rolling price statistics, updated every second
  3. Depth stream     — Order book changes (bid/ask pressure)

All streams run concurrently via asyncio. Nothing blocks. The i5-13th Gen
H-series handles the async event loop on a single core while leaving the
rest free for model inference.

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Binance WebSocket API                         │
  │          wss://stream.binance.com:9443/ws/...                    │
  └──────┬──────────────┬───────────────────┬────────────────────────┘
         │              │                   │
    Trade Stream   Ticker Stream      Depth Stream
         │              │                   │
  ┌──────▼──────────────▼───────────────────▼───────────────────────┐
  │              AsyncIO Event Loop (single thread)                 │
  │                                                                 │
  │   ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐     │
  │   │ TradeHandler │  │ TickerHandler│  │  DepthHandler     │     │
  │   │  parse JSON  │  │  parse JSON  │  │  parse JSON       │     │
  │   │  format msg  │  │  format msg  │  │  format msg       │     │
  │   └──────┬───────┘  └──────┬───────┘  └──────┬────────────┘     │
  │          │                 │                 │                  │
  │          ▼                 ▼                 ▼                  │
  │   ┌─────────────────────────────────────────────────────┐       │
  │   │           Sentiment Queue (asyncio.Queue)           │       │
  │   │ (inference.py will pull from here for inference)    │       │
  │   └─────────────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────────────────┘

Binance WebSocket docs:
  https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams

Install:
  pip install websockets aiohttp

Run:
  python pipeline.py
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    import websockets
except ImportError:
    print("Missing dependency. Install with: pip install websockets")
    sys.exit(1)

try:
    import aiohttp
except ImportError:
    print("Missing dependency. Install with: pip install aiohttp")
    sys.exit(1)

# 1. CONFIGURATION
class StreamType(Enum):
    TRADE  = "trade"
    TICKER = "ticker"
    DEPTH  = "depth"


@dataclass
class PipelineConfig:
    """
    All pipeline settings in one place.

    Binance public WebSocket endpoints require NO api key and NO account.
    Rate limits are generous: 5 messages/sec outbound, no limit inbound.
    Connections auto-close after 24h; we handle reconnection automatically.
    """
    # Binance WebSocket base URL
    # This is the public endpoint. No authentication needed.
    ws_base_url: str = "wss://stream.binance.com:9443/ws"

    # Symbols to track
    # Binance uses lowercase symbol names in stream subscriptions.
    # Format: "<symbol>@<stream_type>"
    symbols: list = field(default_factory=lambda: ["btcusdt", "ethusdt", "solusdt"])

    # Stream toggles
    enable_trades: bool = True      # Individual trades (high frequency)
    enable_tickers: bool = True     # 24h rolling stats (1/sec per symbol)
    enable_depth: bool = True       # Order book updates (top 10 levels)

    # Connection resilience
    reconnect_delay_base: float = 1.0    # Initial reconnect wait (seconds)
    reconnect_delay_max: float = 60.0    # Max backoff cap
    reconnect_max_attempts: int = 50     # Give up after this many consecutive failures
    ping_interval: int = 20              # Send ping every N seconds (keep-alive)
    ping_timeout: int = 10               # Close if no pong within N seconds

    # Queue
    # The sentiment queue bridges this script and inference.
    # Messages accumulate here;  inference's consumer pulls them for scoring.
    queue_maxsize: int = 10_000    # Backpressure: drop oldest if queue fills up

    # Display
    colorize: bool = True          # ANSI color codes in terminal output
    show_timestamp: bool = True    # Prefix each message with UTC time


CONFIG = PipelineConfig()

# 2. ANSI COLORS — Makes the terminal output actually readable
class Color:
    """ANSI escape codes. Disabled automatically if colorize=False or piped."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    @classmethod
    def disable(cls):
        for attr in ["RESET","BOLD","DIM","RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN","WHITE"]:
            setattr(cls, attr, "")


if not CONFIG.colorize or not sys.stdout.isatty():
    Color.disable()

# 3. MESSAGE MODELS — Structured data from raw WebSocket JSON
@dataclass
class TradeMessage:
    """Parsed from Binance trade stream."""
    symbol: str
    price: float
    quantity: float
    trade_time: datetime
    is_buyer_maker: bool    # True = sell (taker sold), False = buy (taker bought)
    trade_id: int

    @classmethod
    def from_binance(cls, data: dict) -> "TradeMessage":
        return cls(
            symbol=data["s"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            trade_time=datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
            is_buyer_maker=data["m"],
            trade_id=data["t"],
        )

    def format(self) -> str:
        side_color = Color.RED if self.is_buyer_maker else Color.GREEN
        side_label = "SELL" if self.is_buyer_maker else "BUY "
        usd_value = self.price * self.quantity
        return (
            f"{Color.BOLD}[TRADE]{Color.RESET} "
            f"{Color.CYAN}{self.symbol:<10}{Color.RESET} "
            f"{side_color}{side_label}{Color.RESET} "
            f"${self.price:>12,.2f} × {self.quantity:<14.6f} "
            f"{Color.DIM}(${usd_value:>12,.2f}){Color.RESET}"
        )


@dataclass
class TickerMessage:
    """Parsed from Binance 24h mini-ticker stream."""
    symbol: str
    last_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    price_change_pct: float

    @classmethod
    def from_binance(cls, data: dict) -> "TickerMessage":
        return cls(
            symbol=data["s"],
            last_price=float(data["c"]),
            high_24h=float(data["h"]),
            low_24h=float(data["l"]),
            volume_24h=float(data["v"]),
            price_change_pct=float(data["P"]),
        )

    def format(self) -> str:
        pct_color = Color.GREEN if self.price_change_pct >= 0 else Color.RED
        arrow = "▲" if self.price_change_pct >= 0 else "▼"
        return (
            f"{Color.BOLD}[TICKR]{Color.RESET} "
            f"{Color.CYAN}{self.symbol:<10}{Color.RESET} "
            f"${self.last_price:>12,.2f}  "
            f"{pct_color}{arrow} {self.price_change_pct:>+7.2f}%{Color.RESET}  "
            f"{Color.DIM}H: ${self.high_24h:,.2f}  L: ${self.low_24h:,.2f}  "
            f"Vol: {self.volume_24h:,.1f}{Color.RESET}"
        )


@dataclass
class DepthMessage:
    """Parsed from Binance partial book depth stream (top 10 levels)."""
    symbol: str
    best_bid: float
    best_ask: float
    bid_depth: float     # Total quantity across top 10 bid levels
    ask_depth: float     # Total quantity across top 10 ask levels
    spread: float
    spread_bps: float    # Spread in basis points

    @classmethod
    def from_binance(cls, data: dict, symbol: str) -> "DepthMessage":
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2 if (best_ask + best_bid) > 0 else 1
        spread_bps = (spread / mid) * 10_000

        return cls(
            symbol=symbol.upper(),
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread=spread,
            spread_bps=spread_bps,
        )

    def format(self) -> str:
        # Bid/ask imbalance: >1 means more buying pressure
        imbalance = self.bid_depth / self.ask_depth if self.ask_depth > 0 else 0
        imb_color = Color.GREEN if imbalance > 1.1 else (Color.RED if imbalance < 0.9 else Color.YELLOW)
        return (
            f"{Color.BOLD}[DEPTH]{Color.RESET} "
            f"{Color.CYAN}{self.symbol:<10}{Color.RESET} "
            f"Bid: ${self.best_bid:>12,.2f}  Ask: ${self.best_ask:>12,.2f}  "
            f"Spread: {self.spread_bps:.1f}bps  "
            f"{imb_color}B/A Ratio: {imbalance:.2f}{Color.RESET}"
        )

# 4. PIPELINE STATISTICS — Track throughput and connection health
@dataclass
class PipelineStats:
    """Real-time counters for monitoring pipeline health."""
    trades_received: int = 0
    tickers_received: int = 0
    depth_updates: int = 0
    errors: int = 0
    reconnections: int = 0
    start_time: float = field(default_factory=time.monotonic)
    queue_drops: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def total_messages(self) -> int:
        return self.trades_received + self.tickers_received + self.depth_updates

    def format_summary(self) -> str:
        uptime = self.uptime_seconds
        rate = self.total_messages / uptime if uptime > 0 else 0
        return (
            f"\n{Color.BOLD}{'─' * 70}\n"
            f"  Pipeline Stats  │  Uptime: {uptime:.0f}s  │  {rate:.1f} msg/sec\n"
            f"  Trades: {self.trades_received:,}  │  Tickers: {self.tickers_received:,}  │  "
            f"Depth: {self.depth_updates:,}  │  Errors: {self.errors}  │  "
            f"Reconnects: {self.reconnections}  │  Queue drops: {self.queue_drops}\n"
            f"{'─' * 70}{Color.RESET}"
        )

# 5. CORE STREAM HANDLERS — One async task per stream type
class BinanceStreamHandler:
    """
    Manages WebSocket connections to Binance with:
      - Automatic reconnection with exponential backoff
      - Message parsing into structured dataclasses
      - Queue forwarding for downstream consumption
      - Per-stream statistics tracking
    """

    def __init__(self, config: PipelineConfig, stats: PipelineStats,
                 sentiment_queue: asyncio.Queue):
        self.config = config
        self.stats = stats
        self.queue = sentiment_queue
        self._running = True

    async def _connect_with_retry(self, url: str, stream_name: str):
        """
        Connect to a WebSocket URL with exponential backoff on failure.

        Backoff sequence: 1s → 2s → 4s → 8s → ... → 60s (capped)
        This prevents hammering Binance servers during an outage.
        """
        attempt = 0
        delay = self.config.reconnect_delay_base

        while self._running and attempt < self.config.reconnect_max_attempts:
            try:
                ws = await websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=5,
                )
                if attempt > 0:
                    self.stats.reconnections += 1
                    print(f"{Color.YELLOW}[RECONN] {stream_name} reconnected "
                          f"(attempt {attempt + 1}){Color.RESET}")
                else:
                    print(f"{Color.GREEN}[CONN] {stream_name} connected{Color.RESET}")

                return ws

            except (websockets.exceptions.WebSocketException,
                    OSError, ConnectionRefusedError) as e:
                attempt += 1
                print(f"{Color.RED}[ERROR] {stream_name} connection failed "
                      f"(attempt {attempt}): {e}{Color.RESET}")
                print(f"{Color.DIM}[RETRY] Waiting {delay:.1f}s before retry...{Color.RESET}")
                await asyncio.sleep(delay)
                # Exponential backoff with cap
                delay = min(delay * 2, self.config.reconnect_delay_max)

        print(f"{Color.RED}[FATAL] {stream_name} exceeded max reconnection attempts. "
              f"Giving up.{Color.RESET}")
        return None

    async def _enqueue(self, message: dict):
        """
        Push a parsed message onto the sentiment queue.
        If the queue is full, drop the oldest message (backpressure).
        """
        if self.queue.full():
            try:
                self.queue.get_nowait()  # Drop oldest
                self.stats.queue_drops += 1
            except asyncio.QueueEmpty:
                pass
        await self.queue.put(message)

    def _timestamp_prefix(self) -> str:
        if self.config.show_timestamp:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            return f"{Color.DIM}{now}{Color.RESET} "
        return ""

    # Trade Stream

    async def stream_trades(self):
        """
        Connect to individual trade streams for all configured symbols.

        Binance combined stream URL format:
          wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade

        Each message contains:
          - s : Symbol (e.g., "BTCUSDT")
          - p : Price as string
          - q : Quantity as string
          - T : Trade time (Unix ms)
          - m : Is buyer the maker? (True = sell aggressor, False = buy aggressor)
          - t : Trade ID
        """
        streams = "/".join(f"{s}@trade" for s in self.config.symbols)
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"

        while self._running:
            ws = await self._connect_with_retry(url, "TRADES")
            if ws is None:
                return

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break

                    data = json.loads(raw_msg)
                    # Combined stream wraps data in {"stream": "...", "data": {...}}
                    trade_data = data.get("data", data)
                    trade = TradeMessage.from_binance(trade_data)
                    self.stats.trades_received += 1

                    print(f"{self._timestamp_prefix()}{trade.format()}")

                    # Forward to queue
                    await self._enqueue({
                        "type": "trade",
                        "symbol": trade.symbol,
                        "price": trade.price,
                        "quantity": trade.quantity,
                        "side": "sell" if trade.is_buyer_maker else "buy",
                        "timestamp": trade.trade_time.isoformat(),
                    })

            except websockets.exceptions.ConnectionClosed as e:
                print(f"{Color.YELLOW}[WARN] Trade stream closed: {e}. Reconnecting...{Color.RESET}")
                self.stats.errors += 1
            except Exception as e:
                print(f"{Color.RED}[ERROR] Trade stream error: {e}{Color.RESET}")
                self.stats.errors += 1
                await asyncio.sleep(1)

    # Ticker Stream

    async def stream_tickers(self):
        """
        Connect to 24h mini-ticker streams. Updates every ~1 second per symbol.

        Lighter than full ticker — gives you price, volume, and 24h change
        without flooding your terminal at trade-level frequency.

        Key fields:
          - s : Symbol
          - c : Close price (last price)
          - h : 24h high
          - l : 24h low
          - v : 24h base asset volume
          - P : 24h price change percent
        """
        streams = "/".join(f"{s}@miniTicker" for s in self.config.symbols)
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"

        while self._running:
            ws = await self._connect_with_retry(url, "TICKERS")
            if ws is None:
                return

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break

                    data = json.loads(raw_msg)
                    ticker_data = data.get("data", data)
                    ticker = TickerMessage.from_binance(ticker_data)
                    self.stats.tickers_received += 1

                    print(f"{self._timestamp_prefix()}{ticker.format()}")

                    await self._enqueue({
                        "type": "ticker",
                        "symbol": ticker.symbol,
                        "price": ticker.last_price,
                        "change_pct": ticker.price_change_pct,
                        "volume_24h": ticker.volume_24h,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            except websockets.exceptions.ConnectionClosed as e:
                print(f"{Color.YELLOW}[WARN] Ticker stream closed: {e}. Reconnecting...{Color.RESET}")
                self.stats.errors += 1
            except Exception as e:
                print(f"{Color.RED}[ERROR] Ticker stream error: {e}{Color.RESET}")
                self.stats.errors += 1
                await asyncio.sleep(1)

    # Depth (Order Book) Stream

    async def stream_depth(self):
        """
        Connect to partial book depth streams (top 10 levels, updates every 1s).

        This gives you bid/ask imbalance data — critical for signal generation. If bids massively outweigh asks, buying pressure
        is building even before the price moves.

        URL format: <symbol>@depth10@1000ms (top 10 levels, 1s updates)

        Key fields:
          - bids : [[price, qty], ...] top 10 bid levels
          - asks : [[price, qty], ...] top 10 ask levels
        """
        streams = "/".join(f"{s}@depth10@1000ms" for s in self.config.symbols)
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"

        while self._running:
            ws = await self._connect_with_retry(url, "DEPTH")
            if ws is None:
                return

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break

                    data = json.loads(raw_msg)
                    stream_name = data.get("stream", "")
                    depth_data = data.get("data", data)

                    # Extract symbol from stream name (e.g., "btcusdt@depth10@1000ms")
                    symbol = stream_name.split("@")[0] if stream_name else "unknown"
                    depth = DepthMessage.from_binance(depth_data, symbol)
                    self.stats.depth_updates += 1

                    print(f"{self._timestamp_prefix()}{depth.format()}")

                    await self._enqueue({
                        "type": "depth",
                        "symbol": depth.symbol,
                        "best_bid": depth.best_bid,
                        "best_ask": depth.best_ask,
                        "spread_bps": depth.spread_bps,
                        "bid_ask_ratio": (depth.bid_depth / depth.ask_depth
                                          if depth.ask_depth > 0 else 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            except websockets.exceptions.ConnectionClosed as e:
                print(f"{Color.YELLOW}[WARN] Depth stream closed: {e}. Reconnecting...{Color.RESET}")
                self.stats.errors += 1
            except Exception as e:
                print(f"{Color.RED}[ERROR] Depth stream error: {e}{Color.RESET}")
                self.stats.errors += 1
                await asyncio.sleep(1)

    def shutdown(self):
        """Signal all stream loops to exit gracefully."""
        self._running = False

# 6. STATS REPORTER — Periodic health check printed to console
async def stats_reporter(stats: PipelineStats, interval: int = 30):
    """Print pipeline health stats every N seconds."""
    while True:
        await asyncio.sleep(interval)
        print(stats.format_summary())

# 7. QUEUE CONSUMER STUB
async def sentiment_consumer(queue: asyncio.Queue):
    """
    INTEGRATION POINT
    This consumer pulls messages from the sentiment queue and will
    eventually run them through the DistilBERT model.

    For now, it just logs queue depth every 100 messages so you can
    verify data is flowing through the pipeline end-to-end.

    Here, this function becomes:
        1. Pull message from queue
        2. Tokenize the text (or construct a text summary from trade data)
        3. Run through ONNX model
        4. Emit signal if confidence > threshold
    """
    processed = 0
    while True:
        message = await queue.get()
        processed += 1

        if processed % 100 == 0:
            print(
                f"{Color.MAGENTA}[QUEUE] Processed {processed:,} messages  │  "
                f"Queue depth: {queue.qsize():,}{Color.RESET}"
            )

        queue.task_done()

# 8. MAIN — Launch all streams concurrently
async def main():
    print(f"\n{'='*70}")
    print(f"  QUANTITATIVE FINANCE SENTIMENT ENGINE: Live Pipeline")
    print(f"{'='*70}")
    print(f"  Symbols : {', '.join(s.upper() for s in CONFIG.symbols)}")
    print(f"  Streams : trades={CONFIG.enable_trades}  tickers={CONFIG.enable_tickers}  "
          f"depth={CONFIG.enable_depth}")
    print(f"  Queue   : max {CONFIG.queue_maxsize:,} messages")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"{'='*70}\n")

    stats = PipelineStats()
    sentiment_queue = asyncio.Queue(maxsize=CONFIG.queue_maxsize)
    handler = BinanceStreamHandler(CONFIG, stats, sentiment_queue)

    # Build task list based on enabled streams
    tasks = []

    if CONFIG.enable_trades:
        tasks.append(asyncio.create_task(handler.stream_trades()))

    if CONFIG.enable_tickers:
        tasks.append(asyncio.create_task(handler.stream_tickers()))

    if CONFIG.enable_depth:
        tasks.append(asyncio.create_task(handler.stream_depth()))

    # Stats reporter runs alongside the streams
    tasks.append(asyncio.create_task(stats_reporter(stats, interval=30)))

    # Queue consumer (integration point)
    tasks.append(asyncio.create_task(sentiment_consumer(sentiment_queue)))

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        print(f"\n{Color.YELLOW}[SHUTDOWN] Ctrl+C received. Closing streams...{Color.RESET}")
        handler.shutdown()
        shutdown_event.set()
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Fall back to signal.signal
        signal.signal(signal.SIGINT, lambda s, f: _signal_handler())
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler())
    # Wait for shutdown signal
    await shutdown_event.wait()
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    # Final stats
    print(stats.format_summary())
    print(f"{Color.GREEN}[DONE] Pipeline shut down cleanly.{Color.RESET}\n")
if __name__ == "__main__":
    asyncio.run(main())