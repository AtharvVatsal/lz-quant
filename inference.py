"""
REAL-TIME INFERENCE ENGINE
Target Hardware : Intel i5-13th Gen H | 32GB DDR5 | NVIDIA RTX 3050 (4GB VRAM)
Runtime         : ONNX Runtime (GPU or CPU fallback)
Data Source     : Binance Public WebSocket (FREE, no API key)
Model           : DistilBERT LoRA fine-tuned → ONNX

This script is the fusion point. It:
  1. Loads the ONNX sentiment model exported
  2. Connects to live Binance WebSocket streams
  3. Constructs natural-language "market sentences" from raw trade data
  4. Runs inference on every sentence through the ONNX model
  5. Measures tick-to-trade latency (message arrival → signal output) in ms
  6. Generates BUY / SELL / HOLD signals based on sentiment + market context

Data Flow:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     Binance WebSocket                               │
  └──────────────┬──────────────────────────────────────────────────────┘
                 │
                 ▼  t₀ = message arrives
  ┌──────────────────────────────────────────────────────────────────┐
  │  Text Constructor                                                │
  │  Raw trade/ticker JSON → natural language sentence               │
  │  "BTC surged $450 (+0.52%) on heavy volume with strong bids"     │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼  t₁ = tokenization starts
  ┌──────────────────────────────────────────────────────────────────┐
  │  Tokenizer (DistilBertTokenizerFast — Rust backend)              │
  │  Sentence → input_ids + attention_mask (128 tokens, padded)      │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼  t₂ = inference starts
  ┌──────────────────────────────────────────────────────────────────┐
  │  ONNX Runtime InferenceSession                                   │
  │  GPU (CUDAExecutionProvider) or CPU fallback                     │
  │  Forward pass → logits → softmax → [Bearish, Neutral, Bullish]   │
  └──────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼  t₃ = signal emitted
  ┌──────────────────────────────────────────────────────────────────┐
  │  Signal Generator                                                │
  │  Sentiment + price momentum + order book imbalance → SIGNAL      │
  │                                                                  │
  │  Latency = t₃ - t₀  (target: < 15ms on GPU, < 50ms on CPU)       │
  └──────────────────────────────────────────────────────────────────┘

Install:
  pip install onnxruntime-gpu websockets aiohttp transformers numpy

  (Use onnxruntime instead of onnxruntime-gpu if you want CPU-only)

Run:
  python inference.py
  python inference.py --cpu          # Force CPU inference
  python inference.py --mock-model   # Run without a trained model (demo mode)
"""

import asyncio
import argparse
import json
import signal
import sys
import time
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("Missing: pip install websockets")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Missing: pip install onnxruntime-gpu  (or onnxruntime for CPU)")

try:
    from transformers import DistilBertTokenizerFast
except ImportError:
    sys.exit("Missing: pip install transformers")

# 1. CONFIGURATION

@dataclass
class InferenceConfig:
    onnx_model_path: str = "./output/finbert-lora/sentiment_model.onnx"
    tokenizer_path: str = "./output/finbert-lora"

    # Inference settings
    max_seq_length: int = 128
    force_cpu: bool = False         # Override: run on CPU even if GPU available
    mock_model: bool = False        # Demo mode: random scores instead of real model

    # Labels
    label_map: dict = field(default_factory=lambda: {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH",
    })

    # Signal thresholds
    bullish_threshold: float = 0.65     # Minimum bullish confidence for BUY signal
    bearish_threshold: float = 0.65     # Minimum bearish confidence for SELL signal
    imbalance_weight: float = 0.15      # How much order book imbalance affects score

    # Binance streams
    symbols: list = field(default_factory=lambda: ["btcusdt", "ethusdt", "solusdt"])
    ws_base_url: str = "wss://stream.binance.com:9443/stream?streams="

    # Latency tracking
    latency_window: int = 500           # Rolling window size for latency stats

    # Display
    colorize: bool = True
    # Throttle ticker/depth prints — only log inference for these every N messages
    # (trades always print because they carry the most alpha)
    ticker_print_interval: int = 5
    depth_print_interval: int = 10

# 2. ANSI COLORS

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"

    @classmethod
    def disable(cls):
        for a in ["RESET","BOLD","DIM","RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN"]:
            setattr(cls, a, "")

# 3. ONNX SENTIMENT MODEL — Wraps tokenizer + ONNX session

class SentimentModel:
    """
    Loads the ONNX model and provides a single .predict() method
    that takes a string and returns (label, confidence_scores, inference_time_ms).

    ONNX Runtime is 2-3x faster than raw PyTorch for inference because:
      - Graph optimizations (constant folding, operator fusion)
      - No autograd overhead (inference only, no gradient tracking)
      - Native CUDA kernel selection tuned for inference batch sizes
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.label_map = config.label_map

        # Load tokenizer
        print(f"[MODEL] Loading tokenizer from {config.tokenizer_path}...")

        if config.mock_model:
            # In mock mode, we still need a tokenizer for realistic latency measurement
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.session = None
            print(f"[MODEL] {C.YELLOW}Mock mode — using random scores{C.RESET}")
            return

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.tokenizer_path)

        # Configure ONNX Runtime execution providers
        # Priority order: CUDA GPU → CPU
        # ONNX Runtime will automatically pick the best available provider.
        providers = []
        if not config.force_cpu:
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                # Limit ONNX Runtime GPU memory so it doesn't fight with
                # any other GPU processes. 512MB is plenty for DistilBERT inference.
                "gpu_mem_limit": 512 * 1024 * 1024,
            }))
        providers.append("CPUExecutionProvider")

        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use 4 threads for CPU inference (leaves cores free for the async loop)
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        print(f"[MODEL] Loading ONNX model from {config.onnx_model_path}...")
        self.session = ort.InferenceSession(
            config.onnx_model_path,
            sess_options=sess_options,
            providers=providers,
        )

        active_provider = self.session.get_providers()[0]
        device_label = "GPU (CUDA)" if "CUDA" in active_provider else "CPU"
        print(f"[MODEL] Inference device: {C.GREEN}{device_label}{C.RESET} "
              f"({active_provider})")

        # Warmup run
        # First inference is always slow (CUDA kernel compilation, memory allocation).
        # Run a dummy prediction so the real ones are fast from the start.
        print("[MODEL] Running warmup inference...")
        _ = self.predict("warmup sentence for model initialization")
        print(f"[MODEL] {C.GREEN}Ready for real-time inference{C.RESET}")

    def predict(self, text: str) -> tuple:
        """
        Run sentiment inference on a single text string.

        Returns:
            (label_str, scores_dict, inference_ms)
            e.g. ("BULLISH", {"BEARISH": 0.05, "NEUTRAL": 0.10, "BULLISH": 0.85}, 3.2)
        """
        t_start = time.perf_counter_ns()

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="np",    # NumPy arrays for ONNX (no PyTorch dependency here)
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Inference
        if self.session is not None:
            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
            )[0]   # Shape: (1, 3)
        else:
            # Mock mode: random logits for pipeline testing
            logits = np.random.randn(1, 3).astype(np.float32)

        # Softmax → probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_logits / exp_logits.sum(axis=-1, keepdims=True)).squeeze()

        t_end = time.perf_counter_ns()
        inference_ms = (t_end - t_start) / 1_000_000

        pred_idx = int(np.argmax(probs))
        label = self.label_map[pred_idx]
        scores = {self.label_map[i]: float(probs[i]) for i in range(len(probs))}

        return label, scores, inference_ms

# 4. TEXT CONSTRUCTOR — Turns raw market data into sentences for the model
class MarketTextConstructor:
    """
    Converts raw Binance WebSocket data into natural language sentences
    that our financial-sentiment DistilBERT model can understand.

    Why not just feed raw numbers?
    The model was trained on Financial PhraseBank — English sentences about
    markets. It understands "revenue surged" and "prices tumbled" but not
    raw JSON. By constructing sentences that mirror financial news language,
    we get dramatically better sentiment signal than feeding raw data.

    This is essentially a feature engineering step — the text IS the feature.
    """

    def __init__(self):
        # Rolling price cache for computing short-term momentum
        self._last_prices: dict[str, deque] = {}
        self._last_tickers: dict[str, dict] = {}
        self._last_depths: dict[str, dict] = {}

    def _get_momentum(self, symbol: str, current_price: float) -> tuple:
        """Calculate short-term price momentum from last N trades."""
        if symbol not in self._last_prices:
            self._last_prices[symbol] = deque(maxlen=50)

        history = self._last_prices[symbol]
        history.append(current_price)

        if len(history) < 2:
            return 0.0, "steady"

        oldest = history[0]
        if oldest == 0:
            return 0.0, "steady"

        change_pct = ((current_price - oldest) / oldest) * 100

        if change_pct > 0.5:
            direction = "surging"
        elif change_pct > 0.1:
            direction = "rising"
        elif change_pct < -0.5:
            direction = "plunging"
        elif change_pct < -0.1:
            direction = "falling"
        else:
            direction = "steady"

        return change_pct, direction

    def from_trade(self, data: dict) -> str:
        """
        Construct sentence from a trade event.

        Example outputs:
          "BTC price surging at $67,450.00, up 0.35% on aggressive buying"
          "ETH price falling to $3,210.50, down 0.22% as sellers dominate"
        """
        symbol = data["s"]
        price = float(data["p"])
        qty = float(data["q"])
        is_sell = data["m"]     # True = buyer is maker = taker sold
        usd_value = price * qty

        change_pct, direction = self._get_momentum(symbol, price)

        side_phrase = "as sellers dominate" if is_sell else "on aggressive buying"
        size_phrase = ""
        if usd_value > 100_000:
            size_phrase = " in a large block trade"
        elif usd_value > 50_000:
            size_phrase = " with significant volume"

        # Include order book context if available
        book_phrase = ""
        if symbol in self._last_depths:
            depth = self._last_depths[symbol]
            ratio = depth.get("ratio", 1.0)
            if ratio > 1.3:
                book_phrase = " with strong bid support"
            elif ratio < 0.7:
                book_phrase = " against heavy ask pressure"

        sign = "up" if change_pct >= 0 else "down"

        return (
            f"{symbol} price {direction} at ${price:,.2f}, "
            f"{sign} {abs(change_pct):.2f}% {side_phrase}"
            f"{size_phrase}{book_phrase}"
        )

    def from_ticker(self, data: dict) -> str:
        """
        Construct sentence from 24h ticker update.

        Example:
          "BTC 24h change +2.45%, trading at $67,450 between high $68,100 and low $65,900 on volume of 12,450 BTC"
        """
        symbol = data["s"]
        price = float(data["c"])
        high = float(data["h"])
        low = float(data["l"])
        volume = float(data["v"])
        change = float(data["P"])

        self._last_tickers[symbol] = {"price": price, "change": change, "volume": volume}

        trend = "gaining" if change > 1 else ("declining" if change < -1 else "flat")

        return (
            f"{symbol} is {trend} with 24h change of {change:+.2f}%, "
            f"currently at ${price:,.2f} between high ${high:,.2f} and low ${low:,.2f} "
            f"on volume of {volume:,.1f}"
        )

    def from_depth(self, data: dict, symbol: str) -> str:
        """
        Construct sentence from order book depth snapshot.

        Example:
          "BTC order book shows strong buying pressure with bid-ask ratio 1.45, spread 0.8 basis points"
        """
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        bid_depth = sum(float(b[1]) for b in bids) if bids else 0
        ask_depth = sum(float(a[1]) for a in asks) if asks else 1
        ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0

        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 1
        spread_bps = ((best_ask - best_bid) / mid) * 10_000

        sym_upper = symbol.upper()
        self._last_depths[sym_upper] = {"ratio": ratio, "spread_bps": spread_bps}

        if ratio > 1.3:
            pressure = "strong buying pressure"
        elif ratio > 1.1:
            pressure = "moderate buying interest"
        elif ratio < 0.7:
            pressure = "heavy selling pressure"
        elif ratio < 0.9:
            pressure = "moderate selling interest"
        else:
            pressure = "balanced order flow"

        return (
            f"{sym_upper} order book shows {pressure} with "
            f"bid-ask ratio {ratio:.2f}, spread {spread_bps:.1f} basis points"
        )

# 5. LATENCY TRACKER — Measures tick-to-trade time in milliseconds
class LatencyTracker:
    """
    Tracks end-to-end latency from message arrival to signal emission.

    Tick-to-trade breakdown:
      t₀  Message arrives from WebSocket           ─┐
      t₁  Text construction from raw data           │  ~0.01ms
      t₂  Tokenization (DistilBertTokenizerFast)    │  ~0.5ms
      t₃  ONNX inference forward pass               │  ~2-8ms (GPU) / ~15-40ms (CPU)
      t₄  Signal generation + output                │  ~0.01ms
                                                   ─┘
      Total: ~3-10ms (GPU) / ~15-45ms (CPU)

    For context, most HFT systems target < 1μs. We're targeting < 15ms,
    which is plenty fast for sentiment-driven crypto signals where the
    alpha decays over seconds, not microseconds.
    """

    def __init__(self, window_size: int = 500):
        self.latencies = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.total_predictions = 0

    def record(self, total_latency_ms: float, inference_ms: float):
        self.latencies.append(total_latency_ms)
        self.inference_times.append(inference_ms)
        self.total_predictions += 1

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies, 99)) if self.latencies else 0.0

    @property
    def avg_inference_ms(self) -> float:
        return float(np.mean(self.inference_times)) if self.inference_times else 0.0

    def format_report(self) -> str:
        if not self.latencies:
            return "[LATENCY] No data yet"
        return (
            f"\n{C.BOLD}{'─'*78}\n"
            f"  LATENCY REPORT  │  Last {len(self.latencies)} predictions  │  "
            f"Total: {self.total_predictions:,}\n"
            f"{'─'*78}{C.RESET}\n"
            f"  End-to-End │  avg: {self.avg_latency_ms:>7.2f}ms  │  "
            f"p50: {self.p50_latency_ms:>7.2f}ms  │  p99: {self.p99_latency_ms:>7.2f}ms\n"
            f"  Inference  │  avg: {self.avg_inference_ms:>7.2f}ms  │  "
            f"min: {min(self.inference_times):>7.2f}ms  │  max: {max(self.inference_times):>7.2f}ms\n"
            f"  Overhead   │  avg: {self.avg_latency_ms - self.avg_inference_ms:>7.2f}ms  "
            f"(text construction + tokenization + signal logic)\n"
            f"{C.BOLD}{'─'*78}{C.RESET}"
        )

# 6. SIGNAL GENERATOR — Combines sentiment + market data → trading signal
@dataclass
class TradingSignal:
    symbol: str
    action: str                # "BUY", "SELL", "HOLD"
    sentiment_label: str       # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float          # 0.0 – 1.0
    scores: dict               # Full score breakdown
    input_text: str            # The sentence that was analyzed
    latency_ms: float          # End-to-end tick-to-trade
    inference_ms: float        # Model inference only
    timestamp: str

    def format(self) -> str:
        # Color the action
        if self.action == "BUY":
            action_str = f"{C.GREEN}{C.BOLD}▲ BUY {C.RESET}"
        elif self.action == "SELL":
            action_str = f"{C.RED}{C.BOLD}▼ SELL{C.RESET}"
        else:
            action_str = f"{C.YELLOW}{C.BOLD}● HOLD{C.RESET}"

        # Color the sentiment
        sent_colors = {"BULLISH": C.GREEN, "BEARISH": C.RED, "NEUTRAL": C.YELLOW}
        sent_color = sent_colors.get(self.sentiment_label, C.RESET)

        # Score bar visualization
        bar_len = 30
        bull_bar = int(self.scores.get("BULLISH", 0) * bar_len)
        neut_bar = int(self.scores.get("NEUTRAL", 0) * bar_len)
        bear_bar = bar_len - bull_bar - neut_bar

        score_bar = (
            f"{C.GREEN}{'█' * bull_bar}{C.RESET}"
            f"{C.YELLOW}{'█' * neut_bar}{C.RESET}"
            f"{C.RED}{'█' * bear_bar}{C.RESET}"
        )

        return (
            f"\n  {action_str}  {C.CYAN}{self.symbol:<10}{C.RESET}"
            f"  {sent_color}{self.sentiment_label}{C.RESET} "
            f"({self.confidence:.1%})  "
            f"│  {score_bar}  │  "
            f"{C.DIM}latency: {self.latency_ms:.2f}ms  "
            f"(inference: {self.inference_ms:.2f}ms){C.RESET}\n"
            f"  {C.DIM}└─ \"{self.input_text[:90]}{'...' if len(self.input_text) > 90 else ''}\"{C.RESET}"
        )


def generate_signal(
    symbol: str,
    label: str,
    scores: dict,
    config: InferenceConfig,
    depth_context: Optional[dict] = None,
) -> str:
    """
    Determine trading action from sentiment + optional market microstructure.

    Logic:
      1. Base signal from sentiment confidence
      2. Boost/dampen based on order book imbalance (if available)
      3. Apply threshold to decide BUY / SELL / HOLD
    """
    bullish_score = scores.get("BULLISH", 0)
    bearish_score = scores.get("BEARISH", 0)

    # Adjust scores with order book context
    if depth_context:
        ratio = depth_context.get("ratio", 1.0)
        # If bids > asks, boost bullish score slightly (and vice versa)
        imbalance = (ratio - 1.0) * config.imbalance_weight
        bullish_score = min(1.0, bullish_score + imbalance)
        bearish_score = min(1.0, bearish_score - imbalance)

    # Decision
    if bullish_score >= config.bullish_threshold:
        return "BUY"
    elif bearish_score >= config.bearish_threshold:
        return "SELL"
    else:
        return "HOLD"

# 7. CORE INFERENCE LOOP — Async consumer that runs the model
async def inference_consumer(
    queue: asyncio.Queue,
    model: SentimentModel,
    text_constructor: MarketTextConstructor,
    latency_tracker: LatencyTracker,
    config: InferenceConfig,
):
    """
    Pulls messages from the async queue, constructs text, runs inference,
    and emits trading signals. This is the hot path — every millisecond counts.

    Runs in the same asyncio event loop as the WebSocket handlers.
    ONNX Runtime releases the GIL during inference, so it doesn't block
    the event loop despite being compute-intensive.
    """
    ticker_count = 0
    depth_count = 0

    while True:
        msg = await queue.get()

        # t₀: Message arrival timestamp 
        t0 = time.perf_counter_ns()

        stream_type = msg.get("_type")
        symbol = msg.get("_symbol", "UNKNOWN")

        # Construct text from market data
        if stream_type == "trade":
            text = text_constructor.from_trade(msg["_raw"])
        elif stream_type == "ticker":
            ticker_count += 1
            text = text_constructor.from_ticker(msg["_raw"])
            # Throttle ticker output
            if ticker_count % config.ticker_print_interval != 0:
                queue.task_done()
                continue
        elif stream_type == "depth":
            depth_count += 1
            text = text_constructor.from_depth(msg["_raw"], msg.get("_stream_symbol", ""))
            if depth_count % config.depth_print_interval != 0:
                queue.task_done()
                continue
        else:
            queue.task_done()
            continue

        # Run inference
        # run_in_executor offloads the blocking ONNX call to a thread pool
        # so it doesn't freeze the async event loop while the model computes.
        loop = asyncio.get_running_loop()
        label, scores, inference_ms = await loop.run_in_executor(
            None, model.predict, text
        )

        # Generate signal
        depth_ctx = text_constructor._last_depths.get(symbol)
        action = generate_signal(symbol, label, scores, config, depth_ctx)

        # t₃: Total latency
        t3 = time.perf_counter_ns()
        total_latency_ms = (t3 - t0) / 1_000_000

        latency_tracker.record(total_latency_ms, inference_ms)

        # Emit signal
        sig = TradingSignal(
            symbol=symbol,
            action=action,
            sentiment_label=label,
            confidence=max(scores.values()),
            scores=scores,
            input_text=text,
            latency_ms=total_latency_ms,
            inference_ms=inference_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        print(sig.format())

        queue.task_done()

# 8. WEBSOCKET STREAM HANDLERS
async def stream_listener(
    url: str,
    stream_name: str,
    stream_type: str,
    queue: asyncio.Queue,
    symbols: list,
    running: asyncio.Event,
):
    """
    Generic WebSocket listener with reconnection.
    Pushes raw messages onto the inference queue with metadata tags.
    """
    reconnect_delay = 1.0

    while running.is_set():
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                print(f"  {C.GREEN}✓{C.RESET} {stream_name} connected")
                reconnect_delay = 1.0  # Reset backoff on success

                async for raw_msg in ws:
                    if not running.is_set():
                        break

                    data = json.loads(raw_msg)
                    stream_id = data.get("stream", "")
                    payload = data.get("data", data)

                    # Extract symbol from stream name
                    stream_symbol = stream_id.split("@")[0] if stream_id else ""
                    symbol_upper = payload.get("s", stream_symbol.upper())

                    await queue.put({
                        "_type": stream_type,
                        "_symbol": symbol_upper,
                        "_stream_symbol": stream_symbol,
                        "_raw": payload,
                        "_received_ns": time.perf_counter_ns(),
                    })

        except websockets.exceptions.ConnectionClosed:
            print(f"  {C.YELLOW}⚠{C.RESET} {stream_name} disconnected. Reconnecting in {reconnect_delay:.0f}s...")
        except Exception as e:
            print(f"  {C.RED}✗{C.RESET} {stream_name} error: {e}")

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60.0)

# 9. PERIODIC REPORTS
async def latency_reporter(tracker: LatencyTracker, interval: int = 30):
    """Print latency statistics every N seconds."""
    while True:
        await asyncio.sleep(interval)
        if tracker.total_predictions > 0:
            print(tracker.format_report())


# 10. MAIN — Orchestrate everything
def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Real-Time Sentiment Inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--mock-model", action="store_true",
                        help="Run without trained model (random scores for pipeline testing)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to ONNX model file")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer directory")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to track (e.g., btcusdt ethusdt)")
    args = parser.parse_args()
    config = InferenceConfig()

    if args.cpu:
        config.force_cpu = True
    if args.mock_model:
        config.mock_model = True
    if args.model_path:
        config.onnx_model_path = args.model_path
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
    if args.symbols:
        config.symbols = [s.lower() for s in args.symbols]

    return config


async def main():
    config = parse_args()

    if not config.colorize or not sys.stdout.isatty():
        C.disable()

    print(f"\n{C.BOLD}{'='*78}")
    print(f"  QUANTITATIVE FINANCE SENTIMENT ENGINE: Real-Time Inference")
    print(f"{'='*78}{C.RESET}")
    print(f"  Symbols      : {', '.join(s.upper() for s in config.symbols)}")
    print(f"  Model        : {config.onnx_model_path}")
    print(f"  Device       : {'CPU (forced)' if config.force_cpu else 'GPU (auto)'}")
    print(f"  Mock mode    : {config.mock_model}")
    print(f"  Thresholds   : BUY > {config.bullish_threshold:.0%}  │  SELL > {config.bearish_threshold:.0%}")
    print(f"  Press Ctrl+C to stop\n")

    # Validate model exists
    if not config.mock_model and not os.path.exists(config.onnx_model_path):
        print(f"{C.YELLOW}[WARN] Model not found at {config.onnx_model_path}")
        print(f"[WARN] Falling back to mock mode. Train first for real inference.{C.RESET}")
        config.mock_model = True

    # Initialize components
    model = SentimentModel(config)
    text_constructor = MarketTextConstructor()
    latency_tracker = LatencyTracker(window_size=config.latency_window)
    inference_queue = asyncio.Queue(maxsize=10_000)
    running = asyncio.Event()
    running.set()

    # Build stream URLs
    trade_streams = "/".join(f"{s}@trade" for s in config.symbols)
    ticker_streams = "/".join(f"{s}@miniTicker" for s in config.symbols)
    depth_streams = "/".join(f"{s}@depth10@1000ms" for s in config.symbols)

    base = "wss://stream.binance.com:9443/stream?streams="

    print(f"  Connecting to Binance streams...")

    # Launch all tasks
    tasks = [
        asyncio.create_task(stream_listener(
            f"{base}{trade_streams}", "Trades", "trade",
            inference_queue, config.symbols, running,
        )),
        asyncio.create_task(stream_listener(
            f"{base}{ticker_streams}", "Tickers", "ticker",
            inference_queue, config.symbols, running,
        )),
        asyncio.create_task(stream_listener(
            f"{base}{depth_streams}", "Depth", "depth",
            inference_queue, config.symbols, running,
        )),
        asyncio.create_task(inference_consumer(
            inference_queue, model, text_constructor, latency_tracker, config,
        )),
        asyncio.create_task(latency_reporter(latency_tracker, interval=30)),
    ]

    # Graceful shutdown
    shutdown = asyncio.Event()

    def _handle_signal():
        print(f"\n{C.YELLOW}[SHUTDOWN] Stopping...{C.RESET}")
        running.clear()
        shutdown.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_signal)
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: _handle_signal())

    await shutdown.wait()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Final report
    print(latency_tracker.format_report())
    print(f"\n{C.GREEN}[DONE] Real-Time Inference shut down cleanly.{C.RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())