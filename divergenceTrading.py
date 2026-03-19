"""
DIVERGENCE DETECTION + PAPER TRADING ENGINE
Target Hardware : Intel i5-13th Gen H | 32GB DDR5 | NVIDIA RTX 3050 (4GB VRAM)

This module adds two production-grade systems to the sentiment engine:

  1. SENTIMENT-PRICE DIVERGENCE DETECTOR
     Detects when sentiment and price action disagree — the #1 signal
     that a reversal is brewing. Uses rolling Z-scores on both sentiment
     and price momentum to catch statistically significant divergences.

     Example:
       Price is climbing +2% over 30 trades, but rolling sentiment has
       shifted from 0.7 bullish to 0.4 bullish → BEARISH DIVERGENCE
       → Model says "smart money is getting nervous while retail chases"

  2. PAPER TRADING ENGINE
     Simulates real position management with:
       - Entry/exit tracking with timestamps
       - Position sizing (fixed fractional or Kelly criterion)
       - Stop-loss and take-profit automation
       - Running P&L with drawdown tracking
       - Win rate, Sharpe ratio, profit factor metrics
       - Full trade journal exportable as JSON

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Inference Engine (existing)                                        │
  │  Emits: { symbol, sentiment, scores, price, action }                │
  └────────────────┬────────────────────────────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DivergenceDetector                                                 │
  │   ├─ Rolling sentiment Z-score (configurable window)                │
  │   ├─ Rolling price momentum Z-score                                 │
  │   ├─ Correlation tracker (sentiment vs returns)                     │
  │   └─ Divergence signal with severity + confidence                   │
  └────────────────┬────────────────────────────────────────────────────┘
                   │ enhanced signal (original + divergence metadata)
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PaperTradingEngine                                                 │
  │   ├─ Position manager (long/short/flat per symbol)                  │
  │   ├─ Risk manager (stop-loss, take-profit, max drawdown)            │
  │   ├─ P&L tracker (realized + unrealized, per-trade + cumulative)    │
  │   └─ Performance analytics (Sharpe, win rate, profit factor)        │
  └────────────────┬────────────────────────────────────────────────────┘
                   │ trade executions + P&L updates
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Dashboard (broadcast via WebSocket)                                │
  └─────────────────────────────────────────────────────────────────────┘

Usage:
  from divergenceTrading import DivergenceDetector, PaperTradingEngine

  detector = DivergenceDetector()
  trader = PaperTradingEngine(starting_capital=10_000)

  # On every signal from the inference engine:
  divergence = detector.update(symbol, sentiment_scores, current_price)
  trade_result = trader.process_signal(symbol, action, current_price, divergence)
"""

import json
import time
import math
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

# 1. CONFIGURATION

@dataclass
class DivergenceConfig:
    """
    Tuning parameters for divergence detection.

    These defaults are calibrated for crypto markets on ~1s tick intervals.
    For equities or slower feeds, increase the window sizes.
    """
    # Rolling windows
    sentiment_window: int = 50       # Trades to average sentiment over
    price_window: int = 50           # Trades to compute momentum over
    zscore_window: int = 100         # History for Z-score normalization
    correlation_window: int = 30     # Window for sentiment-return correlation

    # Divergence thresholds
    # A divergence fires when sentiment Z-score and price Z-score have
    # opposite signs AND both exceed these thresholds in magnitude.
    sentiment_zscore_threshold: float = 1.2    # ~88th percentile
    price_zscore_threshold: float = 1.2
    min_divergence_score: float = 0.5          # Minimum severity to report

    # Cooldown
    # After firing a divergence, wait N ticks before firing again for the
    # same symbol. Prevents signal spam during sustained divergences.
    cooldown_ticks: int = 30


@dataclass
class TradingConfig:
    """
    Paper trading parameters.
    Conservative defaults suitable for a sentiment-driven strategy.
    """
    # Capital
    starting_capital: float = 10_000.0

    # Position sizing
    # Fixed fractional: risk this % of current equity per trade
    risk_per_trade_pct: float = 2.0       # 2% of equity per position
    max_position_pct: float = 10.0        # Never put >10% of equity in one position
    max_open_positions: int = 3           # Across all symbols

    # Risk management
    stop_loss_pct: float = 1.5            # Close at 1.5% loss
    take_profit_pct: float = 3.0          # Close at 3.0% gain
    trailing_stop_pct: float = 1.0        # Trail 1% below high-water mark
    max_drawdown_pct: float = 15.0        # Halt trading if equity drops 15% from peak

    # Signal requirements
    min_confidence: float = 0.65          # Minimum sentiment confidence to trade
    require_divergence: bool = False       # If True, only trade on divergence signals
    divergence_boost: float = 1.5         # Multiply position size on divergence signals

    # Logging
    trade_journal_path: str = "./output/trade_journal.json"

# 2. DIVERGENCE DETECTOR

class DivergenceType(Enum):
    NONE = "NONE"
    BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"   # Price falling, sentiment rising
    BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"   # Price rising, sentiment falling


@dataclass
class DivergenceSignal:
    """Output of the divergence detector."""
    divergence_type: str             # NONE / BULLISH_DIVERGENCE / BEARISH_DIVERGENCE
    severity: float                  # 0.0 to 1.0 (how extreme the divergence is)
    sentiment_zscore: float          # Current sentiment Z-score
    price_zscore: float              # Current price momentum Z-score
    correlation: float               # Rolling sentiment-return correlation
    sentiment_avg: float             # Rolling average sentiment (bullish - bearish)
    price_momentum_pct: float        # Rolling price change %
    description: str                 # Human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


class DivergenceDetector:
    """
    Detects sentiment-price divergences using rolling Z-scores.

    How it works:
    1. Maintain rolling averages of:
       - Net sentiment  = bullish_score - bearish_score  (range: -1 to +1)
       - Price momentum = % change over last N trades

    2. Compute Z-scores for both:
       z_sentiment = (current_avg - long_run_mean) / long_run_std
       z_price     = (current_momentum - long_run_mean) / long_run_std

    3. Divergence fires when:
       - z_sentiment > +threshold AND z_price < -threshold → BEARISH DIVERGENCE
         (sentiment bullish but price dropping = smart money exiting)
       - z_sentiment < -threshold AND z_price > +threshold → BULLISH DIVERGENCE
         (sentiment bearish but price rising = accumulation happening)

    4. Severity = geometric mean of the two Z-score magnitudes, normalized to [0, 1]

    5. Correlation tracker: if the rolling correlation between sentiment and
       returns drops below 0, that's additional confirmation that the
       normal relationship has broken down.
    """

    def __init__(self, config: Optional[DivergenceConfig] = None):
        self.config = config or DivergenceConfig()
        self._state: dict[str, _SymbolDivergenceState] = {}

    def _get_state(self, symbol: str) -> "_SymbolDivergenceState":
        if symbol not in self._state:
            self._state[symbol] = _SymbolDivergenceState(self.config)
        return self._state[symbol]

    def update(self, symbol: str, scores: dict, price: float) -> DivergenceSignal:
        """
        Feed a new data point and return the current divergence status.
        Args:
            symbol: e.g., "BTCUSDT"
            scores: {"BULLISH": 0.7, "NEUTRAL": 0.2, "BEARISH": 0.1}
            price:  current price as float
        Returns:
            DivergenceSignal with type, severity, and explanation
        """
        state = self._get_state(symbol)
        return state.update(scores, price)

    def get_all_states(self) -> dict:
        """Get divergence status for all tracked symbols."""
        return {
            sym: state.last_signal.to_dict() if state.last_signal else None
            for sym, state in self._state.items()
        }


class _SymbolDivergenceState:
    """Internal per-symbol state for divergence tracking."""

    def __init__(self, config: DivergenceConfig):
        self.config = config

        # Rolling sentiment values (net = bullish - bearish)
        self.sentiment_history: deque = deque(maxlen=config.zscore_window)
        self.sentiment_window: deque = deque(maxlen=config.sentiment_window)

        # Rolling price values
        self.price_history: deque = deque(maxlen=config.zscore_window)
        self.price_window: deque = deque(maxlen=config.price_window)
        self.momentum_history: deque = deque(maxlen=config.zscore_window)

        # Correlation tracking
        self.returns_history: deque = deque(maxlen=config.correlation_window)
        self.sent_for_corr: deque = deque(maxlen=config.correlation_window)

        self.last_price: Optional[float] = None
        self.cooldown_counter: int = 0
        self.last_signal: Optional[DivergenceSignal] = None
        self.tick_count: int = 0

    def update(self, scores: dict, price: float) -> DivergenceSignal:
        self.tick_count += 1

        # Compute net sentiment
        net_sentiment = scores.get("BULLISH", 0) - scores.get("BEARISH", 0)
        self.sentiment_history.append(net_sentiment)
        self.sentiment_window.append(net_sentiment)
        sentiment_avg = float(np.mean(self.sentiment_window))

        # Compute price momentum
        self.price_window.append(price)
        self.price_history.append(price)

        if len(self.price_window) >= 2:
            oldest = self.price_window[0]
            momentum_pct = ((price - oldest) / oldest) * 100 if oldest > 0 else 0.0
        else:
            momentum_pct = 0.0
        self.momentum_history.append(momentum_pct)

        # Track returns for correlation
        if self.last_price is not None and self.last_price > 0:
            ret = (price - self.last_price) / self.last_price
            self.returns_history.append(ret)
            self.sent_for_corr.append(net_sentiment)
        self.last_price = price

        # Need minimum data before detecting
        if len(self.sentiment_history) < 20 or len(self.momentum_history) < 20:
            self.last_signal = DivergenceSignal(
                divergence_type=DivergenceType.NONE.value,
                severity=0.0,
                sentiment_zscore=0.0,
                price_zscore=0.0,
                correlation=0.0,
                sentiment_avg=sentiment_avg,
                price_momentum_pct=momentum_pct,
                description="Warming up — collecting baseline data",
            )
            return self.last_signal

        # Compute Z-scores
        sent_arr = np.array(self.sentiment_history)
        sent_mean, sent_std = float(np.mean(sent_arr)), float(np.std(sent_arr))
        sent_zscore = (sentiment_avg - sent_mean) / sent_std if sent_std > 1e-8 else 0.0

        mom_arr = np.array(self.momentum_history)
        mom_mean, mom_std = float(np.mean(mom_arr)), float(np.std(mom_arr))
        price_zscore = (momentum_pct - mom_mean) / mom_std if mom_std > 1e-8 else 0.0

        # Compute correlation
        correlation = 0.0
        if len(self.returns_history) >= 10:
            r = np.array(self.returns_history)
            s = np.array(self.sent_for_corr)
            r_std, s_std = np.std(r), np.std(s)
            if r_std > 1e-10 and s_std > 1e-10:
                correlation = float(np.corrcoef(r, s)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0

        # Detect divergence
        div_type = DivergenceType.NONE
        severity = 0.0
        description = "No divergence — sentiment and price are aligned"

        # Cooldown management
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        sent_thresh = self.config.sentiment_zscore_threshold
        price_thresh = self.config.price_zscore_threshold

        if self.cooldown_counter == 0:
            # BEARISH DIVERGENCE: sentiment bullish (z > 0) but price falling (z < 0)
            if sent_zscore > sent_thresh and price_zscore < -price_thresh:
                div_type = DivergenceType.BEARISH_DIVERGENCE
                severity = self._compute_severity(sent_zscore, price_zscore)
                description = (
                    f"BEARISH DIVERGENCE: Sentiment is unusually bullish "
                    f"(z={sent_zscore:+.2f}) but price momentum is negative "
                    f"(z={price_zscore:+.2f}, {momentum_pct:+.2f}%). "
                    f"Smart money may be exiting while sentiment lags."
                )
                if correlation < 0:
                    description += f" Sentiment-return correlation has inverted ({correlation:.2f})."
                    severity = min(1.0, severity * 1.2)

            # BULLISH DIVERGENCE: sentiment bearish (z < 0) but price rising (z > 0)
            elif sent_zscore < -sent_thresh and price_zscore > price_thresh:
                div_type = DivergenceType.BULLISH_DIVERGENCE
                severity = self._compute_severity(sent_zscore, price_zscore)
                description = (
                    f"BULLISH DIVERGENCE: Sentiment is unusually bearish "
                    f"(z={sent_zscore:+.2f}) but price momentum is positive "
                    f"(z={price_zscore:+.2f}, {momentum_pct:+.2f}%). "
                    f"Accumulation may be occurring beneath negative sentiment."
                )
                if correlation < 0:
                    description += f" Sentiment-return correlation has inverted ({correlation:.2f})."
                    severity = min(1.0, severity * 1.2)

            # Apply cooldown if we fired
            if div_type != DivergenceType.NONE:
                self.cooldown_counter = self.config.cooldown_ticks

        # Filter weak signals
        if severity < self.config.min_divergence_score:
            div_type = DivergenceType.NONE
            severity = 0.0
            description = f"Minor divergence below threshold ({severity:.2f} < {self.config.min_divergence_score})"

        self.last_signal = DivergenceSignal(
            divergence_type=div_type.value,
            severity=round(severity, 4),
            sentiment_zscore=round(sent_zscore, 4),
            price_zscore=round(price_zscore, 4),
            correlation=round(correlation, 4),
            sentiment_avg=round(sentiment_avg, 4),
            price_momentum_pct=round(momentum_pct, 4),
            description=description,
        )
        return self.last_signal

    def _compute_severity(self, sent_z: float, price_z: float) -> float:
        """
        Severity = geometric mean of Z-score magnitudes, squashed to [0, 1].
        Uses sigmoid-like squashing so extreme divergences saturate near 1.0
        instead of growing unbounded.
        """
        raw = math.sqrt(abs(sent_z) * abs(price_z))
        # Sigmoid squash: maps [0, ∞) → [0, 1), with inflection around raw=2
        return 2.0 / (1.0 + math.exp(-raw + 1.0)) - 1.0

# 3. PAPER TRADING ENGINE
class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """An open paper trade position."""
    symbol: str
    side: str                   # "LONG" or "SHORT"
    entry_price: float
    quantity: float             # In base asset units
    entry_time: str
    entry_signal: str           # "BUY" / "SELL"
    entry_confidence: float
    entry_sentiment: str
    had_divergence: bool        # Was this triggered by a divergence?
    # Tracking
    high_water_mark: float = 0.0    # Highest price since entry (for trailing stop)
    low_water_mark: float = 999_999_999.0

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == PositionSide.LONG.value:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG.value:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


@dataclass
class ClosedTrade:
    """A completed paper trade with full P&L accounting."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    entry_signal: str
    exit_reason: str          # "take_profit", "stop_loss", "trailing_stop", "signal_reversal", "manual"
    pnl: float                # Absolute P&L in USD
    pnl_pct: float            # Percentage return
    holding_duration_s: float
    had_divergence: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Real-time portfolio performance analytics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0       # Gross profits / gross losses
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time_s: float = 0.0
    divergence_trade_winrate: float = 0.0
    is_halted: bool = False           # True if max drawdown breached

    def to_dict(self) -> dict:
        return asdict(self)


class PaperTradingEngine:
    """
    Simulates real position management and P&L tracking.
    Entry rules:
      1. Signal is BUY or SELL with confidence >= min_confidence
      2. Not already in a position for this symbol
      3. Haven't exceeded max open positions
      4. Not halted due to max drawdown breach
    Position sizing:
      Base size = equity × risk_per_trade_pct%
      If divergence signal present, multiply by divergence_boost
    Exit rules (checked on every price update):
      1. Stop-loss:     unrealized P&L < -stop_loss_pct%
      2. Take-profit:   unrealized P&L > +take_profit_pct%
      3. Trailing stop:  price drops trailing_stop_pct% from high-water mark
      4. Signal reversal: opposite signal received for this symbol
    Risk management:
      If equity drops max_drawdown_pct% below peak, ALL positions close
      and trading halts until manually reset.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.starting_capital = self.config.starting_capital
        self.cash = self.config.starting_capital
        self.positions: dict[str, Position] = {}       # symbol → Position
        self.closed_trades: list[ClosedTrade] = []
        self.returns_history: list[float] = []         # Per-trade returns for Sharpe
        self.equity_curve: list[dict] = []
        self.peak_equity = self.config.starting_capital
        self.is_halted = False
        self._trade_count = 0
        self._prices: dict[str, float] = {}            # symbol → latest price

    # Public API

    def process_signal(
        self,
        symbol: str,
        action: str,              # "BUY", "SELL", "HOLD"
        price: float,
        confidence: float,
        sentiment: str,
        divergence: Optional[DivergenceSignal] = None,
    ) -> dict:
        """
        Process a trading signal. Returns a dict describing what happened.

        Possible outcomes:
          - "opened_long" / "opened_short"
          - "closed_reversal" (opposite signal closed existing position)
          - "no_action" (HOLD, or requirements not met)
          - "halted" (max drawdown breached)
        """
        result = {
            "action_taken": "no_action",
            "reason": "",
            "trade": None,
            "position": None,
            "metrics": None,
        }
        if price > 0:
            self._prices[symbol] = price

        # Check risk limits first
        if self.is_halted:
            result["action_taken"] = "halted"
            result["reason"] = f"Trading halted: max drawdown ({self.config.max_drawdown_pct}%) breached"
            result["metrics"] = self.get_metrics(price).to_dict()
            return result

        # Update existing positions (stop-loss, take-profit, trailing stop)
        exit_results = self._check_exits(symbol, price)
        if exit_results:
            result["closed_trades"] = exit_results

        # Process new signal
        has_divergence = (divergence is not None and
                          divergence.divergence_type != DivergenceType.NONE.value)

        if self.config.require_divergence and not has_divergence:
            result["reason"] = "No divergence signal (required by config)"
            result["metrics"] = self.get_metrics(price).to_dict()
            return result

        if action == "HOLD" or confidence < self.config.min_confidence:
            result["reason"] = f"Below confidence threshold ({confidence:.2f} < {self.config.min_confidence})"
            result["metrics"] = self.get_metrics(price).to_dict()
            return result

        # Signal reversal: close existing opposite position
        if symbol in self.positions:
            pos = self.positions[symbol]
            should_close = (
                (action == "BUY" and pos.side == PositionSide.SHORT.value) or
                (action == "SELL" and pos.side == PositionSide.LONG.value)
            )
            if should_close:
                trade = self._close_position(symbol, price, "signal_reversal")
                result["action_taken"] = "closed_reversal"
                result["trade"] = trade.to_dict()
            else:
                result["reason"] = f"Already {pos.side} on {symbol}"
                result["metrics"] = self.get_metrics(price).to_dict()
                return result

        # Open new position
        if symbol not in self.positions:
            if len(self.positions) >= self.config.max_open_positions:
                result["reason"] = f"Max positions ({self.config.max_open_positions}) reached"
                result["metrics"] = self.get_metrics(price).to_dict()
                return result

            position = self._open_position(
                symbol, action, price, confidence, sentiment, has_divergence
            )

            if position:
                result["action_taken"] = f"opened_{'long' if action == 'BUY' else 'short'}"
                result["position"] = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "notional": position.notional_value,
                    "divergence": has_divergence,
                }

        # Update equity curve
        current_equity = self._compute_equity(price)
        self.equity_curve.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2),
            "cash": round(self.cash, 2),
            "positions": len(self.positions),
        })

        # Check max drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        if drawdown_pct >= self.config.max_drawdown_pct:
            self._halt_trading(price)
            result["action_taken"] = "halted"
            result["reason"] = f"Max drawdown breached: {drawdown_pct:.1f}%"

        result["metrics"] = self.get_metrics(price).to_dict()
        return result

    def update_prices(self, prices: dict[str, float]) -> list:
        """
        Update all positions with latest prices.
        Checks stop-loss, take-profit, and trailing stops.
        Returns list of any auto-closed trades.
        IMPORTANT: Skips prices that are 0 or negative to prevent
        false stop-loss triggers on symbols that haven't streamed yet.
        """
        valid_prices = {s: p for s, p in prices.items() if p > 0}
        self._prices.update(valid_prices)
        all_exits = []
        for symbol, price in valid_prices.items():
            exits = self._check_exits(symbol, price)
            all_exits.extend(exits)
        return all_exits

    def get_metrics(self, current_price: float = 0) -> PerformanceMetrics:
        """Compute full performance analytics."""
        # Always use _compute_equity which reads per-symbol prices from self._prices
        equity = self._compute_equity(current_price)
        unrealized = sum(
            pos.unrealized_pnl(self._prices.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )

        total = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        drawdown_pct = 0.0
        if self.peak_equity > 0:
            drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100
        # Sharpe ratio (annualized, assuming ~8760 trades/year for crypto)
        sharpe = 0.0
        if len(self.returns_history) >= 5:
            ret_arr = np.array(self.returns_history)
            mean_ret = float(np.mean(ret_arr))
            std_ret = float(np.std(ret_arr))
            if std_ret > 1e-10:
                sharpe = (mean_ret / std_ret) * math.sqrt(8760)
        # Divergence trade performance
        div_trades = [t for t in self.closed_trades if t.had_divergence]
        div_wins = [t for t in div_trades if t.pnl > 0]
        div_winrate = len(div_wins) / len(div_trades) if div_trades else 0.0
        holding_times = [t.holding_duration_s for t in self.closed_trades]
        return PerformanceMetrics(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / total if total > 0 else 0.0,
            total_pnl=round(gross_profit - gross_loss, 2),
            unrealized_pnl=round(unrealized, 2),
            equity=round(equity, 2),
            peak_equity=round(self.peak_equity, 2),
            max_drawdown_pct=round(drawdown_pct, 2),
            current_drawdown_pct=round(drawdown_pct, 2),
            sharpe_ratio=round(sharpe, 4),
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
            avg_win=round(np.mean([t.pnl for t in wins]), 2) if wins else 0.0,
            avg_loss=round(np.mean([t.pnl for t in losses]), 2) if losses else 0.0,
            largest_win=round(max((t.pnl for t in wins), default=0.0), 2),
            largest_loss=round(min((t.pnl for t in losses), default=0.0), 2),
            avg_holding_time_s=round(float(np.mean(holding_times)), 1) if holding_times else 0.0,
            divergence_trade_winrate=round(div_winrate, 4),
            is_halted=self.is_halted,
        )
    def get_open_positions(self) -> list:
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "notional": p.notional_value,
                "entry_time": p.entry_time,
                "divergence": p.had_divergence,
            }
            for p in self.positions.values()
        ]

    def export_journal(self) -> str:
        """Export full trade journal as JSON."""
        journal = {
            "config": asdict(self.config),
            "starting_capital": self.starting_capital,
            "trades": [t.to_dict() for t in self.closed_trades],
            "final_metrics": self.get_metrics().to_dict(),
            "equity_curve": self.equity_curve[-500:],  # Last 500 points
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(self.config.trade_journal_path) or ".", exist_ok=True)
        with open(self.config.trade_journal_path, "w") as f:
            json.dump(journal, f, indent=2)
        return self.config.trade_journal_path

    def reset(self):
        """Reset the engine to initial state (clear all positions and history)."""
        self.cash = self.starting_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.returns_history.clear()
        self.equity_curve.clear()
        self.peak_equity = self.starting_capital
        self.is_halted = False
        self._trade_count = 0
        self._prices.clear()

    # Internal methods

    def _open_position(
        self, symbol: str, action: str, price: float,
        confidence: float, sentiment: str, has_divergence: bool,
    ) -> Optional[Position]:
        """Calculate position size and open a new position."""
        equity = self._compute_equity(price)

        # Position sizing: fixed fractional
        risk_amount = equity * (self.config.risk_per_trade_pct / 100)
        max_amount = equity * (self.config.max_position_pct / 100)
        position_value = min(risk_amount, max_amount)

        # Boost on divergence
        if has_divergence:
            position_value *= self.config.divergence_boost
            position_value = min(position_value, max_amount)

        if position_value <= 0 or price <= 0:
            return None

        quantity = position_value / price
        side = PositionSide.LONG.value if action == "BUY" else PositionSide.SHORT.value

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_signal=action,
            entry_confidence=confidence,
            entry_sentiment=sentiment,
            had_divergence=has_divergence,
            high_water_mark=price,
            low_water_mark=price,
        )

        self.positions[symbol] = position
        self.cash -= position_value
        self._trade_count += 1

        return position

    def _close_position(self, symbol: str, price: float, reason: str) -> ClosedTrade:
        """Close an existing position and record the trade."""
        pos = self.positions.pop(symbol)

        pnl = pos.unrealized_pnl(price)
        pnl_pct = pos.unrealized_pnl_pct(price)

        # Return capital + P&L to cash
        self.cash += pos.notional_value + pnl

        entry_dt = datetime.fromisoformat(pos.entry_time)
        exit_dt = datetime.now(timezone.utc)
        duration = (exit_dt - entry_dt).total_seconds()

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=exit_dt.isoformat(),
            entry_signal=pos.entry_signal,
            exit_reason=reason,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            holding_duration_s=round(duration, 1),
            had_divergence=pos.had_divergence,
        )

        self.closed_trades.append(trade)
        self.returns_history.append(pnl_pct / 100)  # As decimal for Sharpe

        return trade

    def _check_exits(self, symbol: str, price: float) -> list:
        """Check stop-loss, take-profit, and trailing stop for a symbol."""
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        exits = []

        # Update high/low water marks
        pos.high_water_mark = max(pos.high_water_mark, price)
        pos.low_water_mark = min(pos.low_water_mark, price)

        pnl_pct = pos.unrealized_pnl_pct(price)

        # Stop-loss
        if pnl_pct <= -self.config.stop_loss_pct:
            trade = self._close_position(symbol, price, "stop_loss")
            exits.append(trade.to_dict())
            return exits

        # Take-profit
        if pnl_pct >= self.config.take_profit_pct:
            trade = self._close_position(symbol, price, "take_profit")
            exits.append(trade.to_dict())
            return exits

        # Trailing stop (only for positions in profit)
        if pnl_pct > 0:
            if pos.side == PositionSide.LONG.value:
                trail_price = pos.high_water_mark * (1 - self.config.trailing_stop_pct / 100)
                if price <= trail_price:
                    trade = self._close_position(symbol, price, "trailing_stop")
                    exits.append(trade.to_dict())
            else:  # SHORT
                trail_price = pos.low_water_mark * (1 + self.config.trailing_stop_pct / 100)
                if price >= trail_price:
                    trade = self._close_position(symbol, price, "trailing_stop")
                    exits.append(trade.to_dict())

        return exits

    def _compute_equity(self, current_price: float = 0) -> float:
        """Total equity = cash + market value of all open positions.
        Uses per-symbol price tracking to avoid cross-symbol contamination."""
        positions_value = 0.0
        for sym, pos in self.positions.items():
            # Use the correct price for THIS symbol
            sym_price = self._prices.get(sym, pos.entry_price)
            positions_value += pos.notional_value + pos.unrealized_pnl(sym_price)
        return self.cash + positions_value

    def _halt_trading(self, current_price: float):
        """Emergency: close all positions using each symbol's correct price."""
        self.is_halted = True
        symbols = list(self.positions.keys())
        for symbol in symbols:
            # Use the tracked price for THIS symbol, not the triggering symbol's price
            sym_price = self._prices.get(symbol, current_price)
            self._close_position(symbol, sym_price, "max_drawdown_halt")

# 4. INTEGRATION EXAMPLE

def demo():
    """
    Demonstrates the full divergence + trading pipeline with simulated data.
    Run this to see the system in action without needing live market data.
    """
    print("\n" + "=" * 70)
    print(" DEMO: Divergence Detection + Paper Trading")
    print("=" * 70 + "\n")
    detector = DivergenceDetector()
    trader = PaperTradingEngine(TradingConfig(
        starting_capital=10_000,
        risk_per_trade_pct=2.0,
        stop_loss_pct=1.5,
        take_profit_pct=3.0,
    ))
    # Simulate a market scenario: BTC rallies while sentiment cools
    np.random.seed(42)
    price = 67_000.0
    sentiment_trend = 0.6  # Starts bullish
    print("  Simulating 200 ticks of BTC with diverging sentiment...\n")
    for tick in range(200):
        # Price trending up
        price += np.random.normal(15, 50)
        price = max(price, 50_000)

        # But sentiment slowly turning bearish (divergence building)
        if tick > 80:
            sentiment_trend -= 0.003
        sentiment_trend = max(-0.5, min(0.9, sentiment_trend))
        bull = max(0.05, min(0.95, sentiment_trend + np.random.normal(0, 0.08)))
        bear = max(0.05, min(0.95, (1 - sentiment_trend) * 0.6 + np.random.normal(0, 0.05)))
        neut = max(0.05, 1 - bull - bear)
        total = bull + bear + neut
        scores = {"BULLISH": bull / total, "BEARISH": bear / total, "NEUTRAL": neut / total}
        # Run divergence detection
        div_signal = detector.update("BTCUSDT", scores, price)
        # Determine action from sentiment
        if scores["BULLISH"] > 0.65:
            action = "BUY"
        elif scores["BEARISH"] > 0.65:
            action = "SELL"
        else:
            action = "HOLD"
        # Override action on divergence
        if div_signal.divergence_type == "BEARISH_DIVERGENCE":
            action = "SELL"
        elif div_signal.divergence_type == "BULLISH_DIVERGENCE":
            action = "BUY"
        confidence = max(scores.values())
        # Run paper trading
        result = trader.process_signal(
            "BTCUSDT", action, price, confidence, "BULLISH", div_signal
        )
        # Print key events
        if div_signal.divergence_type != "NONE":
            print(f"  Tick {tick:3d} │ ${price:>10,.2f} │ "
                  f"\033[91m{div_signal.divergence_type}\033[0m "
                  f"(severity: {div_signal.severity:.2f})")
            print(f"          │ {div_signal.description[:75]}...")
        if result["action_taken"] not in ("no_action", "halted"):
            color = "\033[92m" if "opened_long" in result["action_taken"] else "\033[91m"
            print(f"  Tick {tick:3d} │ ${price:>10,.2f} │ "
                  f"{color}{result['action_taken'].upper()}\033[0m")
    # Final report
    metrics = trader.get_metrics(price)
    print(f"\n{'─' * 70}")
    print(f"  PAPER TRADING RESULTS")
    print(f"{'─' * 70}")
    print(f"  Starting Capital : ${trader.starting_capital:>10,.2f}")
    print(f"  Final Equity     : ${metrics.equity:>10,.2f}")
    print(f"  Total P&L        : ${metrics.total_pnl:>10,.2f} "
          f"({'+'if metrics.total_pnl >= 0 else ''}{(metrics.total_pnl/trader.starting_capital)*100:.2f}%)")
    print(f"  Total Trades     : {metrics.total_trades}")
    print(f"  Win Rate         : {metrics.win_rate:.1%}")
    print(f"  Profit Factor    : {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio     : {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown     : {metrics.max_drawdown_pct:.2f}%")
    print(f"  Avg Win          : ${metrics.avg_win:>8,.2f}")
    print(f"  Avg Loss         : ${metrics.avg_loss:>8,.2f}")
    print(f"  Largest Win      : ${metrics.largest_win:>8,.2f}")
    print(f"  Largest Loss     : ${metrics.largest_loss:>8,.2f}")
    print(f"  Divergence Trades: WR {metrics.divergence_trade_winrate:.1%}")
    print(f"{'─' * 70}")

    # Export journal
    path = trader.export_journal()
    print(f"\n  Trade journal exported to: {path}\n")

if __name__ == "__main__":
    demo()