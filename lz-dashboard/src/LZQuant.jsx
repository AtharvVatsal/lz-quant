import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { AreaChart, Area, BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";

// ═══════════════════════════════════════════════════════════════════
// LZ-QUANT DUAL MARKET DASHBOARD
// Auto-detects server: connects LIVE if running, falls back to DEMO
// ═══════════════════════════════════════════════════════════════════

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8765/ws/dashboard";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8765";
const CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
const STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN", "SPY", "QQQ"];

const SYM = {
  BTCUSDT: { c: "#f7931a", n: "Bitcoin" }, ETHUSDT: { c: "#627eea", n: "Ethereum" }, SOLUSDT: { c: "#00ffa3", n: "Solana" },
  AAPL: { c: "#a2aaad", n: "Apple" }, MSFT: { c: "#00a4ef", n: "Microsoft" }, GOOGL: { c: "#4285f4", n: "Alphabet" },
  NVDA: { c: "#76b900", n: "NVIDIA" }, TSLA: { c: "#e82127", n: "Tesla" }, AMD: { c: "#ed1c24", n: "AMD" },
  META: { c: "#0668E1", n: "Meta" }, AMZN: { c: "#ff9900", n: "Amazon" }, SPY: { c: "#c9a84c", n: "S&P 500" }, QQQ: { c: "#7b2d8e", n: "Nasdaq 100" },
};

const bg = "#020408", sf = "#080e18", sf2 = "#0c1522", bd = "#111e32", bdB = "#18304a";
const tx = "#7a96b8", dm = "#3a5068", br = "#dce8f4";
const bull = "#00e676", bear = "#ff1744", neut = "#ffc400", acc = "#2979ff", dv = "#d500f9";

// ── DEMO ENGINE ────────────────────────────────────────────────────
function createDemo(startCap = 10000) {
  const prices = { BTCUSDT: 87400, ETHUSDT: 2050, SOLUSDT: 138, AAPL: 178, MSFT: 421, GOOGL: 175, NVDA: 876, TSLA: 245, AMD: 162, META: 505, AMZN: 186, SPY: 524, QQQ: 456 };
  const vol = { BTCUSDT: 60, ETHUSDT: 6, SOLUSDT: 0.9, AAPL: 0.6, MSFT: 1.2, GOOGL: 0.5, NVDA: 3, TSLA: 2.5, AMD: 1, META: 1.5, AMZN: 0.7, SPY: 0.4, QQQ: 0.5 };
  let id = 0, eq = startCap, pk = startCap, tr = 0, wi = 0;
  const eqH = [{ t: "00:00", v: startCap }];
  const demoTrades = [];
  return {
    tick() {
      const allS = [...CRYPTO, ...STOCKS];
      const sym = allS[Math.floor(Math.random() * allS.length)];
      const isCrypto = CRYPTO.includes(sym);
      const d = (Math.random() - 0.48) * (vol[sym] || 1);
      prices[sym] = Math.max(prices[sym] + d, 0.01);
      const p = prices[sym]; id++;
      const chg = d / (p - d) * 100;
      let bu = 0.3 + Math.random() * 0.4, be = 0.1 + Math.random() * 0.3;
      if (chg > 0.1) { bu += 0.2; be -= 0.1; } if (chg < -0.1) { be += 0.2; bu -= 0.1; }
      bu = Math.max(0.02, Math.min(0.98, bu)); be = Math.max(0.02, Math.min(0.98, be));
      let ne = Math.max(0.02, 1 - bu - be); const tot = bu + be + ne; bu /= tot; be /= tot; ne /= tot;
      const sent = bu > be && bu > ne ? "BULLISH" : be > bu && be > ne ? "BEARISH" : "NEUTRAL";
      const conf = Math.max(bu, be, ne);
      let action = "HOLD"; if (bu > 0.65) action = "BUY"; else if (be > 0.65) action = "SELL";
      const lat = isCrypto ? 40 + Math.random() * 30 : 50 + Math.random() * 40;
      const dir = chg > 0.3 ? "rising" : chg < -0.3 ? "falling" : "trading flat";
      const side = Math.random() > 0.5 ? "sell-side" : "buy-side";
      const text = isCrypto ? `${sym} ${dir} at $${p.toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}, ${chg >= 0 ? "up" : "down"} ${Math.abs(chg).toFixed(2)}% on ${side} trade` : `${sym} stock ${dir} at $${p.toFixed(2)}, ${chg >= 0 ? "up" : "down"} ${Math.abs(chg).toFixed(2)}%`;
      const hasDiv = Math.random() > 0.94;
      const divType = hasDiv ? (Math.random() > 0.5 ? "BEARISH_DIVERGENCE" : "BULLISH_DIVERGENCE") : "NONE";
      let tradeAction = "no_action";
      const tradeExec = { type: "trade_execution", market: isCrypto ? "crypto" : "stocks", symbol: sym, price: p, action: action, time: new Date().toISOString() };
      const tradeExit = { type: "trade_exit", market: isCrypto ? "crypto" : "stocks", symbol: sym, exit_price: p, time: new Date().toISOString() };
      if (action !== "HOLD" && conf > 0.65 && Math.random() > 0.75) { 
        tradeAction = action === "BUY" ? "opened_long" : "opened_short"; tr++; 
        const pnl = (Math.random() - 0.42) * eq * 0.015; eq += pnl; pk = Math.max(pk, eq); if (pnl > 0) wi++;
        tradeExec.trade_action = tradeAction;
        tradeExec.pnl = pnl;
        tradeExit.trade = { symbol: sym, exit_price: p, pnl: pnl, exit_reason: pnl > 0 ? "take_profit" : "stop_loss", side: tradeAction === "opened_long" ? "LONG" : "SHORT" };
        demoTrades.push({ symbol: sym, pnl: pnl, holding_duration_s: 60 + Math.random() * 300 });
      }
      eqH.push({ t: new Date().toLocaleTimeString("en", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }), v: +eq.toFixed(2) }); if (eqH.length > 80) eqH.shift();
      return {
        signal: { type: "signal", market: isCrypto ? "crypto" : "stocks", id, symbol: sym, action, sentiment: sent, confidence: +conf.toFixed(4), scores: { BULLISH: +bu.toFixed(4), BEARISH: +be.toFixed(4), NEUTRAL: +ne.toFixed(4) }, text, latency_ms: +lat.toFixed(2), inference_ms: +(lat * 0.85).toFixed(2), time: new Date().toISOString(), is_mock: true, divergence: { divergence_type: divType, severity: hasDiv ? +(0.5 + Math.random() * 0.4).toFixed(3) : 0 }, trade_action: tradeAction },
        ticker: { type: "ticker", market: isCrypto ? "crypto" : "stocks", symbol: sym, price: p, change_pct: +(chg * (2 + Math.random() * 10)).toFixed(2) },
        portfolio: { type: "portfolio", metrics: { equity: +eq.toFixed(2), total_pnl: +(eq - startCap).toFixed(2), total_trades: tr, winning_trades: wi, losing_trades: tr - wi, win_rate: tr > 0 ? +(wi / tr).toFixed(3) : 0, peak_equity: +pk.toFixed(2), current_drawdown_pct: pk > 0 ? +((pk - eq) / pk * 100).toFixed(2) : 0, sharpe_ratio: +(0.3 + Math.random() * 1.8).toFixed(2), profit_factor: tr > 2 ? +(0.7 + Math.random() * 1.5).toFixed(2) : 0, is_halted: false }, equity_curve: [...eqH], positions: [] },
        stats: { type: "stats", total_signals: id, total_messages: id * 8, msg_per_sec: +(8 + Math.random() * 8).toFixed(1) },
        tradeExec: tradeAction !== "no_action" ? tradeExec : null,
        tradeExit: tradeAction !== "no_action" ? tradeExit : null,
        demoTrades: demoTrades.length > 0 ? [...demoTrades] : null,
      };
    }
  };
}

// ── SPARKLINE ──────────────────────────────────────────────────────
const Spark = ({ data, color, w = 60, h = 18 }) => {
  if (!data || data.length < 2) return <div style={{ width: w, height: h }} />;
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`).join(" ");
  const uid = `sp${color.replace("#", "")}${data.length}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: "visible", flexShrink: 0 }}>
      <defs><linearGradient id={uid} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.25} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs>
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill={`url(#${uid})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  );
};

// ── SIGNAL CARD ────────────────────────────────────────────────────
const SignalCard = ({ sig }) => {
  const a = sig.action || "HOLD";
  const c = { BUY: bull, SELL: bear, HOLD: neut }[a] || dm;
  const ico = { BUY: "▲", SELL: "▼", HOLD: "◆" }[a];
  const conf = ((sig.confidence || 0) * 100).toFixed(0);
  const hasDiv = sig.divergence?.divergence_type && sig.divergence.divergence_type !== "NONE";
  const hasTrade = sig.trade_action && sig.trade_action !== "no_action" && sig.trade_action !== "paused";
  const isPaused = sig.trade_action === "paused" || sig.is_paused;
  const symCol = SYM[sig.symbol]?.c || br;
  const symName = SYM[sig.symbol]?.n || sig.symbol;
  const latency = (sig.latency_ms || 0).toFixed(0);

  return (
    <div style={{
      background: bg, border: `1px solid ${hasDiv ? dv + "40" : bd}`, borderRadius: 6,
      padding: "10px 12px", borderLeft: `3px solid ${c}`, position: "relative", overflow: "hidden",
    }}>
      {hasDiv && <div style={{ position: "absolute", top: 0, right: 0, width: 50, height: 50, background: `radial-gradient(circle at top right, ${dv}15, transparent)` }} />}
      
      {/* Row 1: Action + Symbol + Badges */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ color: c, fontWeight: 800, fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>{ico} {a}</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
            <span style={{ color: symCol, fontWeight: 700, fontSize: 13 }}>{(sig.symbol || "").replace("USDT", "")}</span>
            <span style={{ color: dm, fontSize: 9 }}>{symName}</span>
          </div>
          <span style={{ background: c + "18", color: c, padding: "1px 7px", borderRadius: 3, fontSize: 10, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{conf}%</span>
          {hasDiv && <span style={{ background: dv + "18", color: dv, padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>DIV</span>}
          {hasTrade && <span style={{ background: acc + "18", color: acc, padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{sig.trade_action.replace("opened_", "").toUpperCase()}</span>}
          {isPaused && <span style={{ background: bear + "18", color: bear, padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>PAUSED</span>}
        </div>
        <span style={{ color: tx, fontSize: 9, fontFamily: "'IBM Plex Mono', monospace", flexShrink: 0 }}>{latency}ms</span>
      </div>

      {/* Row 2: Description text */}
      <div style={{ color: br, fontSize: 11, lineHeight: 1.4, marginBottom: 6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", opacity: 0.75 }}>
        {sig.text}
      </div>

      {/* Row 3: Sentiment bar + score labels */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, display: "flex", height: 4, borderRadius: 2, overflow: "hidden", gap: 1 }}>
          <div style={{ width: `${(sig.scores?.BULLISH || 0) * 100}%`, background: bull, borderRadius: 2, transition: "width 0.3s" }} />
          <div style={{ width: `${(sig.scores?.NEUTRAL || 0) * 100}%`, background: neut, borderRadius: 2, transition: "width 0.3s" }} />
          <div style={{ width: `${(sig.scores?.BEARISH || 0) * 100}%`, background: bear, borderRadius: 2, transition: "width 0.3s" }} />
        </div>
        <div style={{ display: "flex", gap: 6, fontSize: 9, fontFamily: "'IBM Plex Mono', monospace", flexShrink: 0 }}>
          <span style={{ color: bull }}>{((sig.scores?.BULLISH || 0) * 100).toFixed(0)}</span>
          <span style={{ color: neut }}>{((sig.scores?.NEUTRAL || 0) * 100).toFixed(0)}</span>
          <span style={{ color: bear }}>{((sig.scores?.BEARISH || 0) * 100).toFixed(0)}</span>
        </div>
      </div>
    </div>
  );
};

// ── PRICE ROW ──────────────────────────────────────────────────────
const PriceRow = ({ symbol, ticker, spark }) => {
  const chg = ticker?.change_pct || 0;
  const s = SYM[symbol] || { c: br, n: symbol };
  const price = ticker?.price || 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 10px", background: bg, border: `1px solid ${bd}`, borderRadius: 6, transition: "all 0.2s", minHeight: 36 }}>
      <div style={{ width: 7, height: 7, borderRadius: "50%", background: s.c, flexShrink: 0, boxShadow: `0 0 5px ${s.c}40` }} />
      <div style={{ minWidth: 58, flexShrink: 0 }}>
        <div style={{ fontWeight: 700, fontSize: 11, color: br, lineHeight: 1.2 }}>{symbol.replace("USDT", "")}</div>
        <div style={{ fontSize: 8, color: dm, lineHeight: 1.2 }}>{s.n}</div>
      </div>
      <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
        <Spark data={spark || []} color={s.c} w={55} h={16} />
      </div>
      <div style={{ textAlign: "right", minWidth: 70, flexShrink: 0 }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, fontWeight: 700, color: br, lineHeight: 1.2 }}>
          ${price.toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, fontWeight: 600, color: chg >= 0 ? bull : bear, lineHeight: 1.2 }}>
          {chg >= 0 ? "▲" : "▼"} {Math.abs(chg).toFixed(2)}%
        </div>
      </div>
    </div>
  );
};

// ── MARKET PANEL ───────────────────────────────────────────────────
const MarketPanel = ({ title, icon, accent, symbols, panelSignals, tickers, sparkData }) => {
  const signalContainerRef = useRef(null);
  const isAtBottomRef = useRef(true);

  useEffect(() => {
    const el = signalContainerRef.current;
    if (!el) return;
    if (isAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [panelSignals.length]);

  const handleScroll = () => {
    const el = signalContainerRef.current;
    if (!el) return;
    const threshold = 50;
    isAtBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  };

  const counts = { BULLISH: 0, BEARISH: 0, NEUTRAL: 0 };
  panelSignals.forEach(s => counts[s.sentiment] = (counts[s.sentiment] || 0) + 1);
  const total = panelSignals.length || 1;

  return (
    <div style={{ background: sf, border: `1px solid ${bd}`, borderRadius: 8, padding: 10, display: "flex", flexDirection: "column", gap: 8, flex: 1, minWidth: 0, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 10px", background: accent + "06", border: `1px solid ${accent}15`, borderRadius: 6 }}>
        <span style={{ fontSize: 14 }}>{icon}</span>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1.2, textTransform: "uppercase", color: accent }}>{title}</span>
        <span style={{ marginLeft: "auto", fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: dm }}>{panelSignals.length} sig</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 3, maxHeight: symbols.length > 4 ? 220 : "none", overflowY: symbols.length > 4 ? "auto" : "visible", paddingRight: symbols.length > 4 ? 3 : 0 }}>
        {symbols.map(s => <PriceRow key={s} symbol={s} ticker={tickers[s]} spark={sparkData[s]} />)}
      </div>

      {panelSignals.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 9 }}>
          <div style={{ flex: 1, display: "flex", height: 4, borderRadius: 2, overflow: "hidden", gap: 1 }}>
            <div style={{ width: `${counts.BULLISH / total * 100}%`, background: bull, borderRadius: 2, transition: "width 0.4s" }} />
            <div style={{ width: `${counts.NEUTRAL / total * 100}%`, background: neut, borderRadius: 2, transition: "width 0.4s" }} />
            <div style={{ width: `${counts.BEARISH / total * 100}%`, background: bear, borderRadius: 2, transition: "width 0.4s" }} />
          </div>
          <span style={{ color: bull, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: 9 }}>{(counts.BULLISH / total * 100).toFixed(0)}%</span>
          <span style={{ color: neut, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: 9 }}>{(counts.NEUTRAL / total * 100).toFixed(0)}%</span>
          <span style={{ color: bear, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: 9 }}>{(counts.BEARISH / total * 100).toFixed(0)}%</span>
        </div>
      )}

      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", color: dm }}>Signal Feed</div>
      <div ref={signalContainerRef} onScroll={handleScroll} style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6, minHeight: 0, maxHeight: 320 }}>
        {panelSignals.length === 0 && (
          <div style={{ textAlign: "center", padding: 16, color: dm, fontSize: 11, lineHeight: 1.5 }}>
            {title.includes("STOCK") ? "Waiting for US market hours\n9:30–16:00 ET · 7:00 PM–1:30 AM IST" : "Waiting for signals..."}
          </div>
        )}
        {panelSignals.slice(-10).reverse().map((s, i) => <SignalCard key={s.id || i} sig={s} />)}
      </div>
    </div>
  );
};

// ── MAIN ───────────────────────────────────────────────────────────
export default function LZQuant() {
  const [phase, setPhase] = useState("setup"); // "setup" | "running"
  const [capitalInput, setCapitalInput] = useState("10000");
  const [mode, setMode] = useState("connecting");
  const [signals, setSignals] = useState([]);
  const [tickers, setTickers] = useState({});
  const [portfolio, setPortfolio] = useState(null);
  const [stats, setStats] = useState({});
  const [latencies, setLatencies] = useState([]);
  const [sparkData, setSparkData] = useState({});
  const [activeChart, setActiveChart] = useState("equity");
  const [divAlerts, setDivAlerts] = useState([]);
  const [tradeLog, setTradeLog] = useState([]);
  const [positions, setPositions] = useState([]);
  const [startingCapital, setStartingCapital] = useState(10000);
  const [time, setTime] = useState(new Date());
  const [isPaused, setIsPaused] = useState(false);
  const [activePanel, setActivePanel] = useState("analytics");
  const [closedTrades, setClosedTrades] = useState([]);
  const [sentimentTotals, setSentimentTotals] = useState({ bull: 0, neut: 0, bear: 0 });
  const [settings, setSettings] = useState({ minConfidence: 45, riskPerTrade: 25, stopLoss: 1.5, takeProfit: 2.5, maxPositions: 6, maxDrawdown: 30, divBoost: 2.0, requireDiv: false });
  const [savedMsg, setSavedMsg] = useState("");

  useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);

  const handleStart = async () => {
    const amount = parseFloat(capitalInput);
    if (isNaN(amount) || amount < 100) { alert("Minimum capital is $100"); return; }
    if (amount > 10_000_000) { alert("Maximum capital is $10,000,000"); return; }
    setStartingCapital(amount);

    // Try to set capital on server (if running)
    try {
      await fetch(`${API_URL}/api/trading/set-capital`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ capital: amount }),
      });
    } catch {
      // Server not running — demo mode will use this capital
    }

    setPhase("running");
  };

  // Setup screen JSX (rendered conditionally below, AFTER all hooks)

  const handlePauseResume = async () => {
    const endpoint = isPaused ? "/api/trading/resume" : "/api/trading/pause";
    try {
      const res =       await fetch(`${API_URL}${endpoint}`, { method: "POST" });
      const data = await res.json();
      if (data.status === "paused" || data.status === "resumed") {
        setIsPaused(data.status === "paused");
      }
    } catch {
      // Demo mode - toggle locally
      setIsPaused(!isPaused);
    }
  };

  const process = useCallback((msg) => {
    if (msg.type === "trading_status") {
      setIsPaused(msg.is_paused);
    }
    if (msg.type === "signal") {
      setSignals(p => [...p.slice(-99), msg]);
      setLatencies(p => [...p.slice(-59), { t: p.length, l: msg.latency_ms, i: msg.inference_ms }]);
      if (msg.divergence?.divergence_type && msg.divergence.divergence_type !== "NONE")
        setDivAlerts(p => [...p.slice(-4), { ...msg.divergence, symbol: msg.symbol, time: msg.time }]);
      if (msg.is_paused !== undefined) setIsPaused(msg.is_paused);
      // Track sentiment
      const bu = msg.scores?.BULLISH || 0, be = msg.scores?.BEARISH || 0, ne = msg.scores?.NEUTRAL || 0;
      if (bu > be && bu > ne) setSentimentTotals(p => ({ ...p, bull: p.bull + 1 }));
      else if (be > bu && be > ne) setSentimentTotals(p => ({ ...p, bear: p.bear + 1 }));
      else setSentimentTotals(p => ({ ...p, neut: p.neut + 1 }));
    }
    if (msg.type === "ticker" && msg.symbol) {
      setTickers(p => ({ ...p, [msg.symbol]: msg }));
      setSparkData(p => ({ ...p, [msg.symbol]: [...(p[msg.symbol] || []).slice(-29), msg.price] }));
    }
    if (msg.type === "portfolio") {
      if (msg.equity_curve) {
        msg.equity_curve = msg.equity_curve.map(pt => ({
          t: pt.t || (pt.time ? new Date(pt.time).toLocaleTimeString("en", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : ""),
          v: pt.v ?? pt.equity ?? 0,
        }));
      }
      setPortfolio(msg);
      setPositions(msg.positions || []);
      // Sync starting capital from server
      if (msg.starting_capital && msg.starting_capital > 0) {
        setStartingCapital(msg.starting_capital);
      }
    }
    if (msg.type === "trade_execution") {
      setTradeLog(p => [...p.slice(-29), {
        type: "open",
        symbol: msg.symbol,
        action: msg.action,
        market: msg.market,
        details: msg.details,
        time: msg.time,
      }]);
    }
    if (msg.type === "trade_exit") {
      const t = msg.trade || {};
      setTradeLog(p => [...p.slice(-29), {
        type: "close",
        symbol: t.symbol || msg.symbol,
        exit_reason: t.exit_reason,
        pnl: t.pnl,
        pnl_pct: t.pnl_pct,
        side: t.side,
        market: msg.market,
        time: msg.time,
      }]);
      if (t.pnl !== undefined) {
        setClosedTrades(p => [...p.slice(-99), {
          symbol: t.symbol || msg.symbol,
          pnl: t.pnl,
          holding_duration_s: t.holding_duration_s || 0,
        }]);
      }
    }
    if (msg.type === "stats") setStats(msg);
  }, []);

  // ── AUTO-DETECT ──────────────────────────────────────────────────
  // ── AUTO-DETECT (only starts after user clicks Start Trading) ──
  useEffect(() => {
    if (phase !== "running") return; // Don't connect until setup is done

    let ws, demoTimer, cancelled = false;

    const startDemo = () => {
      if (cancelled) return;
      setMode("demo");
      const eng = createDemo(startingCapital);
      demoTimer = setInterval(() => {
        const d = eng.tick();
        process(d.signal); process(d.ticker); process(d.portfolio); process(d.stats);
        if (d.tradeExec) process(d.tradeExec);
        if (d.tradeExit) process(d.tradeExit);
        if (d.demoTrades) {
          d.demoTrades.forEach(t => {
            setClosedTrades(p => [...p.slice(-99), t]);
          });
        }
      }, 650);
    };

    try {
      ws = new WebSocket(WS_URL);
      const timeout = setTimeout(() => { if (ws.readyState !== WebSocket.OPEN) { ws.close(); startDemo(); } }, 3000);
      ws.onopen = () => { clearTimeout(timeout); if (!cancelled) setMode("live"); };
      ws.onmessage = (e) => { try { const m = JSON.parse(e.data); if (m.type === "snapshot") (m.signals || []).forEach(process); else process(m); } catch {} };
      ws.onclose = () => { clearTimeout(timeout); if (!cancelled) { if (demoTimer) clearInterval(demoTimer); startDemo(); } };
      ws.onerror = () => ws.close();
    } catch { startDemo(); }

    return () => { cancelled = true; if (ws) ws.close(); if (demoTimer) clearInterval(demoTimer); };
  }, [phase, startingCapital, process]);

  const m = portfolio?.metrics || {};
  const pnl = m.total_pnl || 0;
  const pnlCol = pnl >= 0 ? bull : bear;
  const eqCurve = portfolio?.equity_curve || [];
  const chartTT = { background: sf2, border: `1px solid ${bdB}`, borderRadius: 6, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" };
  const P = { background: sf, border: `1px solid ${bd}`, borderRadius: 8, padding: 10, display: "flex", flexDirection: "column" };

  const sentHistory = useMemo(() => {
    const w = 12, res = [];
    for (let i = Math.max(0, signals.length - 50); i < signals.length; i++) {
      const sl = signals.slice(Math.max(0, i - w + 1), i + 1);
      const a = { b: 0, n: 0, r: 0 };
      sl.forEach(s => { a.b += s.scores?.BULLISH || 0; a.n += s.scores?.NEUTRAL || 0; a.r += s.scores?.BEARISH || 0; });
      const n = sl.length;
      res.push({ i, b: +(a.b / n).toFixed(3), n: +(a.n / n).toFixed(3), r: +(a.r / n).toFixed(3) });
    }
    return res;
  }, [signals]);

  const cryptoSignals = signals.filter(s => s.market === "crypto").slice(-40);
  const stockSignals = signals.filter(s => s.market === "stocks").slice(-40);

  // Analytics calculations
  const analytics = useMemo(() => {
    if (closedTrades.length < 2) return { sharpe: null, profitFactor: null, avgTrade: null, bestTrade: null, worstTrade: null, avgHold: null, maxDD: null, calmar: null };
    const pnls = closedTrades.map(t => t.pnl || 0);
    const wins = pnls.filter(p => p > 0);
    const losses = pnls.filter(p => p <= 0);
    const totalWin = wins.reduce((a, b) => a + b, 0);
    const totalLoss = Math.abs(losses.reduce((a, b) => a + b, 0));
    const pf = totalLoss > 0 ? (totalWin / totalLoss).toFixed(2) : totalWin > 0 ? "∞" : "—";
    const avgT = (pnls.reduce((a, b) => a + b, 0) / pnls.length).toFixed(2);
    const bestT = Math.max(...pnls).toFixed(2);
    const worstT = Math.min(...pnls).toFixed(2);
    const holds = closedTrades.filter(t => (t.holding_duration_s || 0) > 0).map(t => t.holding_duration_s);
    const avgH = holds.length > 0 ? (holds.reduce((a, b) => a + b, 0) / holds.length / 60).toFixed(1) : null;
    const peak = Math.max(...closedTrades.map((t, i) => pnls.slice(0, i + 1).reduce((a, b) => a + b, 0)));
    let maxDD = 0;
    closedTrades.forEach((_, i) => {
      const cumm = pnls.slice(0, i + 1).reduce((a, b) => a + b, 0);
      maxDD = Math.max(maxDD, peak - cumm);
    });
    const maxDDpct = (maxDD / startingCapital * 100).toFixed(1);
    const totalReturn = pnls.reduce((a, b) => a + b, 0);
    const calmar = maxDD > 0 ? (totalReturn / maxDD).toFixed(2) : "—";
    const returns = pnls.map(p => p / startingCapital);
    const meanR = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdR = Math.sqrt(returns.map(r => Math.pow(r - meanR, 2)).reduce((a, b) => a + b, 0) / returns.length);
    const sharpe = stdR > 0 ? (meanR / stdR * Math.sqrt(252)).toFixed(2) : "—";
    return { sharpe, profitFactor: pf, avgTrade: avgT, bestTrade: bestT, worstTrade: worstT, avgHold: avgH, maxDD: maxDDpct, calmar };
  }, [closedTrades, startingCapital]);

  const riskMetrics = useMemo(() => {
    const pos = positions || [];
    const totalExp = pos.reduce((s, p) => s + (p.notional || 0), 0);
    const cExp = pos.filter(p => ["BTCUSDT", "ETHUSDT", "SOLUSDT"].includes(p.symbol)).reduce((s, p) => s + (p.notional || 0), 0);
    const sExp = totalExp - cExp;
    const equity = m.equity || startingCapital;
    const symPnL = {};
    closedTrades.forEach(t => {
      if (!symPnL[t.symbol]) symPnL[t.symbol] = 0;
      symPnL[t.symbol] += t.pnl || 0;
    });
    const topSym = Object.entries(symPnL).sort((a, b) => b[1] - a[1]).slice(0, 4);
    return {
      totalExp: equity > 0 ? (totalExp / equity * 100).toFixed(0) : 0,
      cExp: equity > 0 ? (cExp / equity * 100).toFixed(0) : 0,
      sExp: equity > 0 ? (sExp / equity * 100).toFixed(0) : 0,
      openPos: pos.length,
      maxPos: settings.maxPositions,
      posBreak: pos.slice(0, 5),
      topSym,
    };
  }, [positions, closedTrades, startingCapital, m.equity, settings.maxPositions]);

  const tradeAnalytics = useMemo(() => {
    if (closedTrades.length === 0) return { winRate: null, profitFactor: null, avgHold: null, bestSym: null, worstSym: null, totalPnL: null };
    const pnls = closedTrades.map(t => t.pnl || 0);
    const wins = pnls.filter(p => p > 0).length;
    const wr = (wins / closedTrades.length * 100).toFixed(0);
    const wins2 = pnls.filter(p => p > 0);
    const losses = pnls.filter(p => p <= 0);
    const pf = losses.length > 0 ? (wins2.reduce((a, b) => a + b, 0) / Math.abs(losses.reduce((a, b) => a + b, 0))).toFixed(2) : "∞";
    const holds = closedTrades.filter(t => (t.holding_duration_s || 0) > 0).map(t => t.holding_duration_s / 60);
    const avgH = holds.length > 0 ? (holds.reduce((a, b) => a + b, 0) / holds.length).toFixed(1) : null;
    const symPnL = {};
    closedTrades.forEach(t => {
      if (!symPnL[t.symbol]) symPnL[t.symbol] = 0;
      symPnL[t.symbol] += t.pnl || 0;
    });
    const sortedSym = Object.entries(symPnL).sort((a, b) => b[1] - a[1]);
    const bestSym = sortedSym.length > 0 ? sortedSym[0][0].replace("USDT", "") : null;
    const worstSym = sortedSym.length > 0 ? sortedSym[sortedSym.length - 1][0].replace("USDT", "") : null;
    const totalPnL = pnls.reduce((a, b) => a + b, 0);
    return { winRate: wr, profitFactor: pf, avgHold: avgH, bestSym, worstSym, totalPnL };
  }, [closedTrades]);

  const sentiment = useMemo(() => {
    const total = sentimentTotals.bull + sentimentTotals.neut + sentimentTotals.bear;
    if (total === 0) return { bull: "—", neut: "—", bear: "—", bullPct: 33, neutPct: 33, bearPct: 34 };
    return {
      bull: (sentimentTotals.bull / total * 100).toFixed(0) + "%",
      neut: (sentimentTotals.neut / total * 100).toFixed(0) + "%",
      bear: (sentimentTotals.bear / total * 100).toFixed(0) + "%",
      bullPct: sentimentTotals.bull / total * 100,
      neutPct: sentimentTotals.neut / total * 100,
      bearPct: sentimentTotals.bear / total * 100,
    };
  }, [sentimentTotals]);

  const saveSettings = async () => {
    const cfg = {
      min_confidence: settings.minConfidence / 100,
      risk_per_trade_pct: settings.riskPerTrade,
      stop_loss_pct: settings.stopLoss,
      take_profit_pct: settings.takeProfit,
      max_open_positions: settings.maxPositions,
      max_drawdown_pct: settings.maxDrawdown,
      divergence_boost: settings.divBoost,
      require_divergence: settings.requireDiv,
    };
    try {
      await fetch(`${API_URL}/api/trading/set-config`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      setSavedMsg("Saved!");
      setTimeout(() => setSavedMsg(""), 2000);
    } catch {}
  };

  // ── SETUP SCREEN ──────────────────────────────────────────────
  if (phase === "setup") return (
    <div style={{ background: bg, color: tx, minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Outfit', -apple-system, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet" />
      <div style={{ width: 420, padding: 40, background: sf, border: `1px solid ${bd}`, borderRadius: 16, textAlign: "center" }}>
        <img src="/favicon.png" alt="LZ-Quant" style={{ width: 48, height: 48, borderRadius: 10, marginBottom: 12 }} />
        <div style={{ fontSize: 28, fontWeight: 800, color: br, marginBottom: 6 }}>
          <span style={{ color: acc }}>⚡</span> LZ-QUANT
        </div>
        <div style={{ fontSize: 12, color: dm, marginBottom: 32, letterSpacing: 1, textTransform: "uppercase" }}>Dual Market Sentiment Engine</div>
        <div style={{ textAlign: "left", marginBottom: 20 }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: tx, textTransform: "uppercase", letterSpacing: 1, display: "block", marginBottom: 8 }}>Starting Capital (USD)</label>
          <div style={{ position: "relative" }}>
            <span style={{ position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)", fontSize: 20, fontWeight: 700, color: dm }}>$</span>
            <input type="number" value={capitalInput} onChange={e => setCapitalInput(e.target.value)} onKeyDown={e => e.key === "Enter" && handleStart()}
              style={{ width: "100%", padding: "16px 16px 16px 36px", fontSize: 24, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", background: bg, color: br, border: `2px solid ${bdB}`, borderRadius: 10, outline: "none" }}
              min="100" max="10000000" step="100" autoFocus />
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6, marginBottom: 24 }}>
          {[1000, 5000, 10000, 50000].map(amt => (
            <button key={amt} onClick={() => setCapitalInput(String(amt))}
              style={{ background: capitalInput === String(amt) ? acc + "20" : bg, color: capitalInput === String(amt) ? acc : dm, border: `1px solid ${capitalInput === String(amt) ? acc + "40" : bd}`, borderRadius: 6, padding: "8px 4px", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace" }}>
              ${amt.toLocaleString()}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 20, fontSize: 11, color: dm, textAlign: "left" }}>
          <div style={{ flex: 1, background: bg, border: `1px solid ${bd}`, borderRadius: 6, padding: 10 }}>
            <div style={{ color: bull, fontWeight: 600, marginBottom: 2 }}>Risk per trade</div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace" }}>25% · ${(parseFloat(capitalInput || 0) * 0.25).toLocaleString()}</div>
          </div>
          <div style={{ flex: 1, background: bg, border: `1px solid ${bd}`, borderRadius: 6, padding: 10 }}>
            <div style={{ color: bear, fontWeight: 600, marginBottom: 2 }}>Max drawdown</div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace" }}>30% · ${(parseFloat(capitalInput || 0) * 0.30).toLocaleString()}</div>
          </div>
        </div>
        <button onClick={handleStart}
          style={{ width: "100%", padding: "14px 24px", fontSize: 16, fontWeight: 700, background: `linear-gradient(135deg, ${acc}, #00e5ff)`, color: "white", border: "none", borderRadius: 10, cursor: "pointer", fontFamily: "'Outfit', sans-serif", boxShadow: `0 4px 20px ${acc}40` }}>
          ⚡  Start Trading
        </button>
        <div style={{ marginTop: 16, fontSize: 10, color: dm, lineHeight: 1.5 }}>
          Paper trading only — no real money at risk.<br />
          Auto-connects to server if running, otherwise runs in demo mode.
        </div>
      </div>
      <style>{`
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
        input[type=number] { -moz-appearance: textfield; }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );

  // ── DASHBOARD ─────────────────────────────────────────────────
  return (
    <div style={{ background: bg, color: tx, height: "100vh", fontFamily: "'Outfit', -apple-system, sans-serif", padding: 6, display: "flex", flexDirection: "column", gap: 5, overflow: "hidden" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet" />

      {/* HEADER */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 4, borderBottom: `1px solid ${bd}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <img src="/favicon.png" alt="LZ-Quant" style={{ width: 22, height: 22, borderRadius: 4 }} />
          <span style={{ fontSize: 13, fontWeight: 800, color: br, letterSpacing: -0.5 }}>
            <span style={{ color: acc }}></span> LZ-QUANT
            <span style={{ fontSize: 8, fontWeight: 400, color: dm, marginLeft: 5 }}>DUAL MARKET</span>
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: mode === "live" ? bull : mode === "connecting" ? neut : acc, boxShadow: `0 0 5px ${mode === "live" ? bull : mode === "connecting" ? neut : acc}50`, animation: "pulse 2s ease infinite" }} />
            <span style={{ fontSize: 8, fontWeight: 600, color: mode === "live" ? bull : mode === "connecting" ? neut : dm }}>{mode === "live" ? "LIVE" : mode === "connecting" ? "CONNECTING..." : "DEMO"}</span>
          </div>
          {mode === "demo" && <span style={{ background: neut + "10", color: neut, padding: "1px 8px", borderRadius: 4, fontSize: 8, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>Simulated</span>}
          {divAlerts.length > 0 && <span style={{ background: dv + "10", color: dv, padding: "1px 8px", borderRadius: 4, fontSize: 8, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", animation: "pulse 1.5s ease infinite" }}>⚡ {divAlerts.length} DIV</span>}
          {isPaused && <span style={{ background: bear + "12", color: bear, padding: "1px 8px", borderRadius: 4, fontSize: 8, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", animation: "pulse 1.5s ease infinite" }}>⏸ PAUSED</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={handlePauseResume} style={{ background: isPaused ? bull + "12" : bear + "12", color: isPaused ? bull : bear, border: `1px solid ${isPaused ? bull : bear}35`, borderRadius: 5, padding: "3px 10px", fontSize: 9, fontWeight: 700, cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace" }}>
            {isPaused ? "▶ Resume" : "⏸ Pause"}
          </button>
          <span style={{ fontSize: 10, color: dm, fontFamily: "'IBM Plex Mono', monospace" }}>{time.toLocaleTimeString("en", { hour12: false })}</span>
        </div>
      </div>

      {/* STATS */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 5 }}>
        {[
          { l: "Signals", v: stats.total_signals || signals.length },
          { l: "Latency", v: latencies.length ? (latencies.reduce((a, b) => a + b.l, 0) / latencies.length).toFixed(0) + "ms" : "—", c: acc },
          { l: "Throughput", v: stats.msg_per_sec ? stats.msg_per_sec.toFixed(0) + "/s" : "—" },
          { l: "Equity", v: "$" + (m.equity || startingCapital).toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 }), c: pnlCol },
          { l: "P&L", v: (pnl >= 0 ? "+$" : "−$") + Math.abs(pnl).toFixed(2), c: pnlCol },
          { l: "Win Rate", v: m.total_trades > 0 ? ((m.win_rate || 0) * 100).toFixed(0) + "%" : "—", c: (m.win_rate || 0) >= 0.5 ? bull : bear },
          { l: "Trades", v: m.total_trades || 0 },
          { l: "Drawdown", v: (m.current_drawdown_pct || 0).toFixed(1) + "%", c: (m.current_drawdown_pct || 0) > 5 ? bear : bull },
        ].map(({ l, v, c }, i) => (
          <div key={i} style={{ ...P, padding: "6px 8px", gap: 2, borderRadius: 6 }}>
            <span style={{ fontSize: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, color: dm }}>{l}</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: c || br, fontFamily: "'IBM Plex Mono', monospace" }}>{v}</span>
          </div>
        ))}
      </div>

      {/* CHART */}
      <div style={{ ...P, borderRadius: 7, minHeight: 130 }}>
        <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
          {[["equity", "EQUITY"], ["sentiment", "SENTIMENT"], ["latency", "LATENCY"]].map(([k, l]) => (
            <button key={k} onClick={() => setActiveChart(k)} style={{ background: activeChart === k ? acc + "10" : "transparent", color: activeChart === k ? acc : dm, border: `1px solid ${activeChart === k ? acc + "20" : bd}`, borderRadius: 3, padding: "2px 8px", fontSize: 7, fontWeight: 700, cursor: "pointer", letterSpacing: 0.5, fontFamily: "'IBM Plex Mono', monospace", transition: "all 0.2s" }}>{l}</button>
          ))}
          <div style={{ flex: 1 }} />
          {activeChart === "equity" && <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 700, color: pnlCol, alignSelf: "center" }}>${(m.equity || startingCapital).toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>}
        </div>
        <ResponsiveContainer width="100%" height={115}>
          {activeChart === "equity" ? (
            <AreaChart data={eqCurve.slice(-60)} margin={{ top: 2, right: 2, bottom: 0, left: 2 }}>
              <defs><linearGradient id="gEq" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={pnlCol} stopOpacity={0.2} /><stop offset="100%" stopColor={pnlCol} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke={bd} />
              <XAxis dataKey="t" tick={{ fontSize: 6, fill: dm }} tickLine={false} axisLine={{ stroke: bd }} interval="preserveStartEnd" />
              <YAxis domain={[d => Math.floor(d * 0.998), d => Math.ceil(d * 1.002)]} tick={{ fontSize: 6, fill: dm }} tickLine={false} axisLine={false} tickFormatter={v => `$${v.toLocaleString()}`} width={45} />
              <ReferenceLine y={startingCapital} stroke={dm} strokeDasharray="3 3" label={{ value: "Start", fill: dm, fontSize: 6 }} />
              <Tooltip contentStyle={chartTT} />
              <Area type="monotone" dataKey="v" stroke={pnlCol} strokeWidth={2} fill="url(#gEq)" dot={false} isAnimationActive={false} />
            </AreaChart>
          ) : activeChart === "sentiment" ? (
            <AreaChart data={sentHistory} margin={{ top: 2, right: 2, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="gSB" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={bull} stopOpacity={0.2} /><stop offset="100%" stopColor={bull} stopOpacity={0} /></linearGradient>
                <linearGradient id="gSR" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={bear} stopOpacity={0.2} /><stop offset="100%" stopColor={bear} stopOpacity={0} /></linearGradient>
              </defs>
              <YAxis domain={[0, 1]} tick={{ fontSize: 6, fill: dm }} tickLine={false} axisLine={false} width={15} />
              <ReferenceLine y={0.5} stroke={bd} strokeDasharray="3 3" />
              <Tooltip contentStyle={chartTT} />
              <Area type="monotone" dataKey="b" stroke={bull} strokeWidth={1.5} fill="url(#gSB)" dot={false} isAnimationActive={false} />
              <Area type="monotone" dataKey="r" stroke={bear} strokeWidth={1.5} fill="url(#gSR)" dot={false} isAnimationActive={false} />
              <Area type="monotone" dataKey="n" stroke={neut} strokeWidth={1} fill="none" dot={false} isAnimationActive={false} strokeDasharray="4 2" />
            </AreaChart>
          ) : (
            <BarChart data={latencies.slice(-40)} margin={{ top: 2, right: 2, bottom: 0, left: 0 }}>
              <YAxis tick={{ fontSize: 6, fill: dm }} tickLine={false} axisLine={false} width={18} />
              <Tooltip contentStyle={chartTT} formatter={v => [`${v.toFixed(1)}ms`]} />
              <Bar dataKey="l" radius={[2, 2, 0, 0]} isAnimationActive={false}>{latencies.slice(-40).map((e, i) => <Cell key={i} fill={e.l < 30 ? bull + "60" : e.l < 80 ? neut + "60" : bear + "60"} />)}</Bar>
              <Bar dataKey="i" radius={[2, 2, 0, 0]} isAnimationActive={false} fill={acc + "35"} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* DUAL PANELS */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, minHeight: 0 }}>
        <MarketPanel title="CRYPTO — BINANCE" icon="₿" accent="#f7931a" symbols={CRYPTO} panelSignals={cryptoSignals} tickers={tickers} sparkData={sparkData} />
        <MarketPanel title="STOCKS — NASDAQ" icon="📈" accent="#4fc3f7" symbols={STOCKS} panelSignals={stockSignals} tickers={tickers} sparkData={sparkData} />
      </div>

      {/* POSITIONS & TRADE LOG */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, minHeight: 0 }}>
        {/* Open Positions */}
        <div style={{ ...P, borderRadius: 8, gap: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", color: dm }}>Open Positions</span>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 700, color: positions.length > 0 ? br : dm }}>{positions.length}</span>
          </div>
          {positions.length === 0 ? (
            <div style={{ textAlign: "center", padding: 12, color: dm, fontSize: 11 }}>No open positions</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {positions.map((pos, i) => {
                const isLong = pos.side === "LONG";
                const sideCol = isLong ? bull : bear;
                const symCol = SYM[pos.symbol]?.c || br;
                return (
                  <div key={i} style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 6, padding: "8px 12px", borderLeft: `3px solid ${sideCol}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{ width: 7, height: 7, borderRadius: "50%", background: symCol }} />
                      <span style={{ fontWeight: 700, fontSize: 12, color: br }}>{(pos.symbol || "").replace("USDT", "")}</span>
                      <span style={{ background: sideCol + "15", color: sideCol, padding: "1px 6px", borderRadius: 3, fontSize: 10, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{pos.side}</span>
                      {pos.divergence && <span style={{ background: dv + "15", color: dv, padding: "1px 5px", borderRadius: 3, fontSize: 9, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>DIV</span>}
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, fontWeight: 700, color: br }}>@ ${(pos.entry_price || 0).toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: dm }}>${(pos.notional || 0).toFixed(2)}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Trade Log */}
        <div style={{ ...P, borderRadius: 8, gap: 6, overflow: "hidden" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", color: dm }}>Trade Log</span>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 700, color: tradeLog.length > 0 ? br : dm }}>{tradeLog.length}</span>
          </div>
          {tradeLog.length === 0 ? (
            <div style={{ textAlign: "center", padding: 12, color: dm, fontSize: 11 }}>No trades yet</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0, overflowY: "auto" }}>
              {tradeLog.slice().reverse().map((t, i) => {
                const isOpen = t.type === "open";
                const isProfit = (t.pnl || 0) >= 0;
                const symCol = SYM[t.symbol]?.c || br;
                const actionLabel = isOpen
                  ? (t.action || "").replace("opened_", "").replace("closed_", "").toUpperCase()
                  : (t.exit_reason || "CLOSED").replace("_", " ").toUpperCase();
                const borderCol = isOpen ? acc : (isProfit ? bull : bear);
                const timeStr = t.time ? new Date(t.time).toLocaleTimeString("en", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";

                return (
                  <div key={i} style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 6, padding: "6px 10px", borderLeft: `3px solid ${borderCol}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ color: borderCol, fontWeight: 800, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 }}>
                        {isOpen ? "↗" : "↘"} {actionLabel}
                      </span>
                      <span style={{ color: symCol, fontWeight: 700, fontSize: 11 }}>{(t.symbol || "").replace("USDT", "")}</span>
                      {t.side && <span style={{ background: (t.side === "LONG" ? bull : bear) + "15", color: t.side === "LONG" ? bull : bear, padding: "1px 5px", borderRadius: 3, fontSize: 8, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{t.side}</span>}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      {!isOpen && t.pnl !== undefined && (
                        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, fontWeight: 700, color: isProfit ? bull : bear }}>
                          {isProfit ? "+" : ""}${(t.pnl || 0).toFixed(2)}
                        </span>
                      )}
                      {!isOpen && t.pnl_pct !== undefined && (
                        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: isProfit ? bull : bear }}>
                          ({isProfit ? "+" : ""}{(t.pnl_pct || 0).toFixed(2)}%)
                        </span>
                      )}
                      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, color: dm }}>{timeStr}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* MARKET SENTIMENT - Always Visible */}
      <div style={{ ...P, marginTop: 6, borderRadius: 10 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, color: dm, marginBottom: 10 }}>Market Sentiment</div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", gap: 10 }}>
            {[{ l: "Bullish", v: sentiment.bull, c: bull }, { l: "Neutral", v: sentiment.neut, c: neut }, { l: "Bearish", v: sentiment.bear, c: bear }].map(({ l, v, c }, i) => (
              <div key={i} style={{ textAlign: "center", background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: "10px 14px", minWidth: 80 }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: c, fontFamily: "'IBM Plex Mono', monospace" }}>{v}</div>
                <div style={{ fontSize: 9, textTransform: "uppercase", color: dm, marginTop: 3 }}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{ flex: 1, height: 18, background: sf, borderRadius: 10, overflow: "hidden", display: "flex" }}>
            <div style={{ width: sentiment.bullPct + "%", background: bull, transition: "width 0.3s" }} />
            <div style={{ width: sentiment.neutPct + "%", background: neut, transition: "width 0.3s" }} />
            <div style={{ width: sentiment.bearPct + "%", background: bear, transition: "width 0.3s" }} />
          </div>
        </div>
      </div>

      {/* ANALYTICS PANELS BUTTONS */}
      <div style={{ display: "flex", gap: 6 }}>
        {["analytics", "risk", "tradeanalytics"].map(p => (
          <button key={p} onClick={() => setActivePanel(activePanel === p ? "" : p)} style={{
            background: activePanel === p ? acc : sf, color: activePanel === p ? "#fff" : dm,
            border: `1px solid ${activePanel === p ? acc : bd}`, borderRadius: 6, padding: "6px 12px",
            fontSize: 10, fontWeight: 700, cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace",
          }}>{p === "analytics" ? "ANALYTICS" : p === "tradeanalytics" ? "TRADE ANALYTICS" : p.toUpperCase()}</button>
        ))}
        <div style={{ flex: 1 }} />
        <button onClick={() => setActivePanel(activePanel === "settings" ? "" : "settings")} style={{
          background: activePanel === "settings" ? acc : sf, color: activePanel === "settings" ? "#fff" : dm,
          border: `1px solid ${activePanel === "settings" ? acc : bd}`, borderRadius: 6, padding: "6px 12px",
          fontSize: 10, fontWeight: 700, cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace",
        }}>⚙ SETTINGS</button>
      </div>

      {/* ANALYTICS */}
      {activePanel === "analytics" && (
        <div style={{ ...P, marginTop: 6, borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
            {[
              { l: "Sharpe Ratio", v: analytics.sharpe || "—", sub: "Risk-adjusted return" },
              { l: "Profit Factor", v: analytics.profitFactor || "—", sub: "Gross profit / loss" },
              { l: "Avg Trade", v: analytics.avgTrade ? "$" + analytics.avgTrade : "—", sub: "Per closed trade" },
              { l: "Best Trade", v: analytics.bestTrade ? "$" + analytics.bestTrade : "—", sub: "Highest single trade" },
              { l: "Worst Trade", v: analytics.worstTrade ? "$" + analytics.worstTrade : "—", sub: "Lowest single trade" },
              { l: "Avg Hold Time", v: analytics.avgHold ? analytics.avgHold + "m" : "—", sub: "Position duration" },
              { l: "Max Drawdown", v: analytics.maxDD ? analytics.maxDD + "%" : "—", sub: "Peak-to-trough" },
              { l: "Calmar Ratio", v: analytics.calmar || "—", sub: "Return / max DD" },
            ].map(({ l, v, sub }, i) => (
              <div key={i} style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 12 }}>
                <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: 0.8, color: dm, marginBottom: 4 }}>{l}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: br, fontFamily: "'IBM Plex Mono', monospace" }}>{v}</div>
                <div style={{ fontSize: 9, color: dm, marginTop: 3 }}>{sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* RISK METRICS */}
      {activePanel === "risk" && (
        <div style={{ ...P, marginTop: 6, borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, color: dm, marginBottom: 10 }}>Risk Metrics</div>
              {[{ l: "Total Exposure", v: riskMetrics.totalExp + "%" }, { l: "Crypto Exposure", v: riskMetrics.cExp + "%" }, { l: "Stock Exposure", v: riskMetrics.sExp + "%" }, { l: "Open Positions", v: riskMetrics.openPos }, { l: "Max Positions", v: riskMetrics.maxPos }].map(({ l, v }, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "5px 0", borderBottom: i < 4 ? `1px solid ${bd}` : "none" }}>
                  <span style={{ color: dm }}>{l}</span>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, color: br }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, color: dm, marginBottom: 10 }}>Position Breakdown</div>
              {riskMetrics.posBreak.length === 0 ? (
                <div style={{ color: dm, fontSize: 11 }}>No positions</div>
              ) : riskMetrics.posBreak.map((p, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "5px 0", borderBottom: i < riskMetrics.posBreak.length - 1 ? `1px solid ${bd}` : "none" }}>
                  <span style={{ color: dm }}>{(p.symbol || "").replace("USDT", "")}</span>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, color: br }}>${(p.notional || 0).toFixed(0)}</span>
                </div>
              ))}
            </div>
            <div style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, color: dm, marginBottom: 10 }}>Top Symbols</div>
              {riskMetrics.topSym.length === 0 ? (
                <div style={{ color: dm, fontSize: 11 }}>No trades yet</div>
              ) : riskMetrics.topSym.map(([sym, d], i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "5px 0", borderBottom: i < riskMetrics.topSym.length - 1 ? `1px solid ${bd}` : "none" }}>
                  <span style={{ color: dm }}>{sym.replace("USDT", "")}</span>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, color: d[1] >= 0 ? bull : bear }}>${d[1].toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TRADE ANALYTICS */}
      {activePanel === "tradeanalytics" && (
        <div style={{ ...P, marginTop: 6, borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
            {[
              { l: "Win Rate", v: tradeAnalytics.winRate ? tradeAnalytics.winRate + "%" : "—", c: (tradeAnalytics.winRate || 0) >= 50 ? bull : bear },
              { l: "Profit Factor", v: tradeAnalytics.profitFactor || "—", c: (tradeAnalytics.profitFactor || 0) >= 1 ? bull : bear },
              { l: "Avg Holding", v: tradeAnalytics.avgHold ? tradeAnalytics.avgHold + "m" : "—", c: br },
              { l: "Best Symbol", v: tradeAnalytics.bestSym || "—", c: bull },
              { l: "Worst Symbol", v: tradeAnalytics.worstSym || "—", c: bear },
              { l: "Total P&L", v: tradeAnalytics.totalPnL !== null ? (tradeAnalytics.totalPnL >= 0 ? "+$" : "-$") + Math.abs(tradeAnalytics.totalPnL).toFixed(2) : "—", c: (tradeAnalytics.totalPnL || 0) >= 0 ? bull : bear },
            ].map(({ l, v, c }, i) => (
              <div key={i} style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 12, textAlign: "center" }}>
                <div style={{ fontSize: 9, textTransform: "uppercase", color: dm, marginBottom: 6 }}>{l}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: c, fontFamily: "'IBM Plex Mono', monospace" }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SETTINGS */}
      {activePanel === "settings" && (
        <div style={{ ...P, marginTop: 6, borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 12 }}>
            {[
              { l: "Min Confidence", id: "minConfidence", min: 30, max: 90, step: 5, suffix: "%" },
              { l: "Risk Per Trade", id: "riskPerTrade", min: 1, max: 50, step: 1, suffix: "%" },
              { l: "Stop Loss", id: "stopLoss", min: 0.5, max: 5, step: 0.5, suffix: "%" },
              { l: "Take Profit", id: "takeProfit", min: 1, max: 10, step: 0.5, suffix: "%" },
              { l: "Max Positions", id: "maxPositions", min: 1, max: 10, step: 1, suffix: "" },
              { l: "Max Drawdown", id: "maxDrawdown", min: 5, max: 50, step: 5, suffix: "%" },
              { l: "Div Boost", id: "divBoost", min: 1, max: 3, step: 0.5, suffix: "x" },
            ].map(({ l, id, min, max, step, suffix }, i) => (
              <div key={i} style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 10 }}>
                <div style={{ fontSize: 9, textTransform: "uppercase", color: dm, marginBottom: 8 }}>{l}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input type="range" min={min} max={max} step={step} value={settings[id]} onChange={e => setSettings(p => ({ ...p, [id]: parseFloat(e.target.value) }))} style={{ flex: 1, accentColor: acc, height: 5 }} />
                  <span style={{ fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: br, minWidth: 40, textAlign: "right" }}>{settings[id]}{suffix}</span>
                </div>
              </div>
            ))}
            <div style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 9, textTransform: "uppercase", color: dm, marginBottom: 8 }}>Require Divergence</div>
              <button onClick={() => setSettings(p => ({ ...p, requireDiv: !p.requireDiv }))} style={{
                width: "100%", padding: 8, background: settings.requireDiv ? acc : "transparent", color: settings.requireDiv ? "#fff" : dm,
                border: `1px solid ${settings.requireDiv ? acc : bd}`, borderRadius: 5, fontSize: 11, fontWeight: 700, cursor: "pointer",
              }}>{settings.requireDiv ? "ON" : "OFF"}</button>
            </div>
          </div>
          <button onClick={saveSettings} style={{
            width: "100%", padding: 10, background: savedMsg ? bull : acc, color: "#fff", border: "none",
            borderRadius: 6, fontSize: 12, fontWeight: 700, cursor: "pointer",
          }}>{savedMsg || "APPLY SETTINGS"}</button>
        </div>
      )}

      {/* FOOTER */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 9, color: dm, paddingTop: 6, borderTop: `1px solid ${bd}` }}>
        <span>LZ-QUANT — DUAL MARKET · DIVERGENCE DETECTION · PAPER TRADING</span>
        <span>{mode === "live" ? "🟢 Server connected" : mode === "demo" ? "🟡 Demo — server offline" : "⏳ Connecting..."} · Model: {signals[signals.length - 1]?.is_mock === false ? "ONNX DistilBERT-LoRA" : "Simulated"}</span>
      </div>

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: ${bg}; }
        ::-webkit-scrollbar-thumb { background: ${bd}; border-radius: 3px; }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}