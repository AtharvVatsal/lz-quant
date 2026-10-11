import { useState, useEffect, useCallback, useMemo } from "react";
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import "./LZQuant.css";

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

// Chart mark colors (mirrors LZQuant.css tokens; validated for the dark surface)
const bull = "#14A89A", bear = "#D9473F", flat = "#7F8DA6", acc = "#6E9BFF";
const lineCol = "#243553", lineStrong = "#33496F", ink = "#E8EEF8", ink3 = "#8E9EB9", surface2 = "#192741";

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

// ── FORMATTING (display only) ──────────────────────────────────────
const isNum = v => typeof v === "number" && Number.isFinite(v);
const toNum = v => { const n = typeof v === "string" ? parseFloat(v) : v; return isNum(n) ? n : null; };
const short = s => (s || "").replace("USDT", "");
const round = (v, dp) => Math.round(v * 10 ** dp) / 10 ** dp;
const fmt = (v, dp) => Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
const money = (v, dp = 2) => isNum(v) ? (round(v, dp) < 0 ? "−" : "") + "$" + fmt(v, dp) : "—";
const signedMoney = (v, dp = 2) => { if (!isNum(v)) return "—"; const r = round(v, dp); return (r > 0 ? "+" : r < 0 ? "−" : "") + "$" + fmt(v, dp); };
const signedPct = (v, dp = 2) => { if (!isNum(v)) return "—"; const r = round(v, dp); return (r > 0 ? "+" : r < 0 ? "−" : "") + fmt(v, dp) + "%"; };
// Color follows the value as displayed (rounded), so "$0.00" is never red or green
const tone = (v, dp = 2) => { if (!isNum(v)) return "flat"; const r = round(v, dp); return r > 0 ? "up" : r < 0 ? "down" : "flat"; };
const priceFmt = v => isNum(v) && v > 0 ? "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: v < 1 ? 4 : 2 }) : "—";
const qtyFmt = v => isNum(v) ? v.toLocaleString("en-US", { maximumFractionDigits: Math.abs(v) < 1 ? 6 : 4 }) : "—";
const clock = iso => {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
};
const duration = s => {
  if (!isNum(s)) return "—";
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const mins = Math.floor(s / 60);
  if (mins < 60) return `${mins}m ${s % 60}s`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
};
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

const ACTIONS = {
  BUY: { cls: "buy", label: "Buy", icon: "▲" },
  SELL: { cls: "sell", label: "Sell", icon: "▼" },
  HOLD: { cls: "hold", label: "Hold", icon: "■" },
};
const isTradeAction = ta => typeof ta === "string" && (ta.startsWith("opened") || ta.startsWith("closed"));
const tradeOutcome = ta => {
  if (!ta || ta === "no_action") return null;
  if (ta === "opened_long") return { cls: "trade", text: "Opened long" };
  if (ta === "opened_short") return { cls: "trade", text: "Opened short" };
  if (ta === "closed_reversal") return { cls: "trade", text: "Closed on reversal" };
  if (ta === "paused") return { cls: "warn", text: "Not traded: paused" };
  if (ta === "halted") return { cls: "warn", text: "Not traded: trading halted" };
  return { cls: "trade", text: ta.replace(/_/g, " ") };
};
const sideOfAction = a => a === "opened_long" ? "LONG" : a === "opened_short" ? "SHORT" : null;
const EXIT_REASONS = {
  take_profit: "Take profit", stop_loss: "Stop loss", trailing_stop: "Trailing stop",
  signal_reversal: "Reversal", max_drawdown_halt: "Drawdown halt", manual: "Manual close",
};
const exitLabel = r => EXIT_REASONS[r] || (r ? r.replace(/_/g, " ") : "Closed");
const divLabel = t => t === "BULLISH_DIVERGENCE" ? "Bullish divergence" : t === "BEARISH_DIVERGENCE" ? "Bearish divergence" : "Divergence";

const chartTip = {
  contentStyle: { background: surface2, border: `1px solid ${lineStrong}`, borderRadius: 8, fontSize: 13, color: ink, padding: "8px 12px" },
  labelStyle: { color: ink3, marginBottom: 4 },
  itemStyle: { color: ink, padding: 0 },
  cursor: { stroke: lineStrong, fill: "rgba(110,155,255,0.06)" },
};
const axisTick = { fontSize: 12, fill: ink3 };

// ── SMALL COMPONENTS ───────────────────────────────────────────────
const Panel = ({ title, aside, children }) => (
  <section className="lzq-panel">
    {(title || aside) && (
      <div className="lzq-panel-head">
        {title && <h2>{title}</h2>}
        {aside && <span className="aside">{aside}</span>}
      </div>
    )}
    {children}
  </section>
);

const Empty = ({ title, children }) => (
  <div className="lzq-empty">{title && <strong>{title}</strong>}{children}</div>
);

const Metric = ({ label, value, sub, cls }) => (
  <div className="lzq-metric">
    <dt>{label}</dt>
    <dd className={value === "—" ? undefined : cls}>{value}{sub && value !== "—" && <small>{sub}</small>}</dd>
  </div>
);

const Seg = ({ label, options, value, onChange }) => (
  <div className="lzq-seg" role="group" aria-label={label}>
    {options.map(([k, l]) => (
      <button key={k} type="button" aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>
    ))}
  </div>
);

const Spark = ({ data, color, id, w = 72, h = 24 }) => {
  if (!data || data.length < 2) return <svg width={w} height={h} aria-hidden="true" />;
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - 2 - ((v - min) / range) * (h - 4)}`).join(" ");
  const gid = `spark-${id}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true" style={{ overflow: "visible" }}>
      <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.22} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs>
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill={`url(#${gid})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.75} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
};

const SentimentSplit = ({ b, n, r, thin }) => (
  <div className={`lzq-split${thin ? " thin" : ""}`} role="img" aria-label={`Bullish ${b.toFixed(0)}%, neutral ${n.toFixed(0)}%, bearish ${r.toFixed(0)}%`}>
    <div style={{ width: `${b}%`, background: "var(--up)" }} />
    <div style={{ width: `${n}%`, background: "var(--flat)" }} />
    <div style={{ width: `${r}%`, background: "var(--down)" }} />
  </div>
);

const SignalCard = ({ sig }) => {
  const a = ACTIONS[sig.action] || ACTIONS.HOLD;
  const sc = sig.scores || {};
  const b = (sc.BULLISH || 0) * 100, n = (sc.NEUTRAL || 0) * 100, r = (sc.BEARISH || 0) * 100;
  const trade = tradeOutcome(sig.trade_action);
  const dvType = sig.divergence?.divergence_type;
  const hasDiv = dvType && dvType !== "NONE";
  const meta = SYM[sig.symbol];
  return (
    <article className={`lzq-sig ${a.cls}`}>
      <div className="lzq-sig-top">
        <span className={`lzq-pill ${a.cls}`}><span aria-hidden="true">{a.icon}</span>{a.label}</span>
        <span className="lzq-sig-sym">{short(sig.symbol)}</span>
        {meta && <span className="lzq-sig-name">{meta.n}</span>}
        <span className="lzq-sig-meta">
          <span>{clock(sig.time)}</span>
          {isNum(sig.latency_ms) && <span>{Math.round(sig.latency_ms)} ms</span>}
        </span>
      </div>
      {sig.text && <p className="lzq-sig-text" title={sig.text}>{sig.text}</p>}
      <div className="lzq-sig-foot">
        <SentimentSplit b={b} n={n} r={r} thin />
        <div className="lzq-sig-scores">
          <span>Bullish <b>{b.toFixed(0)}%</b></span>
          <span>Neutral <b>{n.toFixed(0)}%</b></span>
          <span>Bearish <b>{r.toFixed(0)}%</b></span>
        </div>
      </div>
      <div className="lzq-sig-tags">
        {trade && <span className={`lzq-pill ${trade.cls}`}>{trade.text}</span>}
        {hasDiv && <span className="lzq-pill div">{divLabel(dvType)}{isNum(sig.divergence.severity) ? `, severity ${sig.divergence.severity.toFixed(2)}` : ""}</span>}
      </div>
    </article>
  );
};

const PriceRow = ({ symbol, ticker, spark }) => {
  const meta = SYM[symbol] || { c: ink3, n: symbol };
  const p = ticker?.price;
  const has = isNum(p) && p > 0;
  const first = spark && spark.length > 1 ? spark[0] : null;
  const last = spark && spark.length > 1 ? spark[spark.length - 1] : null;
  const chg = first > 0 && isNum(last) ? (last - first) / first * 100 : null;
  const t = tone(chg);
  return (
    <div className={`lzq-price${has ? "" : " waiting"}`}>
      <span className="lzq-dot" style={{ background: meta.c }} aria-hidden="true" />
      <div style={{ minWidth: 0 }}>
        <div className="sym">{short(symbol)}</div>
        <div className="name">{meta.n}</div>
      </div>
      <Spark data={spark} id={symbol} color={t === "up" ? bull : t === "down" ? bear : flat} />
      <div>
        <div className="px">{has ? priceFmt(p) : "No price yet"}</div>
        {has && <div className={`chg ${t}`} title="Change across the last 30 price updates">{chg === null ? "—" : signedPct(chg)}</div>}
      </div>
    </div>
  );
};

const PriceList = ({ symbols, tickers, sparkData }) => (
  <div className="lzq-prices">
    {symbols.map(s => <PriceRow key={s} symbol={s} ticker={tickers[s]} spark={sparkData[s]} />)}
  </div>
);

const MarketPanel = ({ title, source, symbols, panelSignals, tickers, sparkData, emptyText }) => {
  const counts = { BULLISH: 0, BEARISH: 0, NEUTRAL: 0 };
  panelSignals.forEach(s => counts[s.sentiment] = (counts[s.sentiment] || 0) + 1);
  const total = panelSignals.length || 1;
  return (
    <Panel title={title} aside={source}>
      <PriceList symbols={symbols} tickers={tickers} sparkData={sparkData} />
      <div style={{ margin: "18px 0 16px" }}>
        <div className="lzq-panel-head" style={{ marginBottom: 8 }}>
          <h2 style={{ fontSize: 14.5 }}>Sentiment</h2>
          <span className="aside">{panelSignals.length ? `Last ${panelSignals.length} signals` : "No signals yet"}</span>
        </div>
        <SentimentSplit b={counts.BULLISH / total * 100} n={counts.NEUTRAL / total * 100} r={counts.BEARISH / total * 100} />
        <div className="lzq-split-labels">
          <span><span className="lzq-swatch" style={{ background: bull }} />Bullish <b>{panelSignals.length ? (counts.BULLISH / total * 100).toFixed(0) + "%" : "—"}</b></span>
          <span><span className="lzq-swatch" style={{ background: flat }} />Neutral <b>{panelSignals.length ? (counts.NEUTRAL / total * 100).toFixed(0) + "%" : "—"}</b></span>
          <span><span className="lzq-swatch" style={{ background: bear }} />Bearish <b>{panelSignals.length ? (counts.BEARISH / total * 100).toFixed(0) + "%" : "—"}</b></span>
        </div>
      </div>
      <div className="lzq-panel-head" style={{ marginBottom: 8 }}>
        <h2 style={{ fontSize: 14.5 }}>Signal feed</h2>
        <span className="aside">Newest first</span>
      </div>
      {panelSignals.length === 0
        ? <Empty>{emptyText}</Empty>
        : <div className="lzq-feed scroll" style={{ maxHeight: 520 }}>{panelSignals.slice(-15).reverse().map((s, i) => <SignalCard key={s.id || i} sig={s} />)}</div>}
    </Panel>
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
  const [activePanel, setActivePanel] = useState("overview");
  const [closedTrades, setClosedTrades] = useState([]);
  const [sentimentTotals, setSentimentTotals] = useState({ bull: 0, neut: 0, bear: 0 });
  const [settings, setSettings] = useState({ minConfidence: 45, riskPerTrade: 25, stopLoss: 1.5, takeProfit: 2.5, maxPositions: 6, maxDrawdown: 30, divBoost: 2.0, requireDiv: false });
  const [savedMsg, setSavedMsg] = useState("");
  const [sigMarket, setSigMarket] = useState("all");
  const [sigAction, setSigAction] = useState("all");
  const [sigTradedOnly, setSigTradedOnly] = useState(false);

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

  // ── VIEW-ONLY DERIVED DATA ────────────────────────────────────
  const filteredSignals = useMemo(() => signals.filter(s =>
    (sigMarket === "all" || s.market === sigMarket) &&
    (sigAction === "all" || s.action === sigAction) &&
    (!sigTradedOnly || isTradeAction(s.trade_action))
  ).slice().reverse(), [signals, sigMarket, sigAction, sigTradedOnly]);

  const latencyStats = useMemo(() => {
    const ls = latencies.map(x => x.l).filter(isNum).sort((a, b) => a - b);
    if (!ls.length) return null;
    const rank = p => ls[Math.max(0, Math.ceil(p * ls.length) - 1)];
    const inf = latencies.map(x => x.i).filter(isNum);
    return { n: ls.length, avg: mean(ls), p50: rank(0.5), p95: rank(0.95), inf: inf.length ? mean(inf) : null };
  }, [latencies]);

  const latencyChart = useMemo(() => latencies.slice(-40).map((e, k) => ({
    k: k + 1, inf: isNum(e.i) ? e.i : 0, rest: Math.max((e.l || 0) - (e.i || 0), 0),
  })), [latencies]);

  const symbolPnl = useMemo(() => {
    const acc = {};
    closedTrades.forEach(t => { if (t.symbol) acc[t.symbol] = (acc[t.symbol] || 0) + (t.pnl || 0); });
    return Object.entries(acc).sort((a, b) => b[1] - a[1]);
  }, [closedTrades]);

  // ── SETUP SCREEN ──────────────────────────────────────────────
  if (phase === "setup") {
    const cap = parseFloat(capitalInput);
    return (
      <div className="lzq">
        <div className="lzq-setup">
          <div className="lzq-setup-card">
            <header>
              <img src="/favicon.png" alt="" />
              <div>
                <h1>LZ-Quant</h1>
                <div className="muted">Sentiment-driven paper trading</div>
              </div>
            </header>
            <p className="lede">Streams crypto and US stock prices, reads news and Reddit posts, scores the text with a sentiment model, and trades a simulated account.</p>
            <label className="lzq-field-label" htmlFor="lzq-capital">Starting capital (USD)</label>
            <div className="lzq-money">
              <span aria-hidden="true">$</span>
              <input id="lzq-capital" type="number" value={capitalInput} onChange={e => setCapitalInput(e.target.value)} onKeyDown={e => e.key === "Enter" && handleStart()}
                min="100" max="10000000" step="100" autoFocus />
            </div>
            <div className="lzq-presets">
              {[1000, 5000, 10000, 50000].map(amt => (
                <button key={amt} type="button" aria-pressed={capitalInput === String(amt)} onClick={() => setCapitalInput(String(amt))}>${amt.toLocaleString()}</button>
              ))}
            </div>
            <dl className="lzq-setup-facts">
              <div><dt>Size per trade (25%)</dt><dd>{isNum(cap) ? money(cap * 0.25, 0) : "—"}</dd></div>
              <div><dt>Halts after a 30% drop</dt><dd>{isNum(cap) ? money(cap * 0.30, 0) : "—"}</dd></div>
            </dl>
            <button type="button" className="lzq-btn primary" onClick={handleStart}>Start trading</button>
            <footer>
              Paper trading only: no real money and no real orders. Amounts above use the server's default settings.
              If the server at <span className="code">{API_URL}</span> is running, starting resets its paper account to this amount. If it isn't, you'll see simulated demo data.
            </footer>
          </div>
        </div>
      </div>
    );
  }

  // ── DASHBOARD ─────────────────────────────────────────────────
  const equity = isNum(m.equity) ? m.equity : startingCapital;
  const eqChange = equity - startingCapital;
  const eqChangePct = startingCapital > 0 ? eqChange / startingCapital * 100 : null;
  const lastSig = signals[signals.length - 1];
  const modelLabel = !lastSig ? "Waiting for first signal" : lastSig.is_mock === false ? "DistilBERT (ONNX)" : "Simulated";
  const modeLabel = mode === "live" ? "Live" : mode === "demo" ? "Demo data" : "Connecting";
  const modeColor = mode === "live" ? bull : mode === "demo" ? "#F2B84B" : ink3;
  const tradingState = m.is_halted ? "Halted" : isPaused ? "Paused" : "Active";
  const ingestion = stats.ingestion && typeof stats.ingestion === "object" ? stats.ingestion : null;

  const positionView = pos => {
    const mark = tickers[pos.symbol]?.price;
    const ok = isNum(mark) && mark > 0 && isNum(pos.entry_price) && isNum(pos.quantity);
    const upnl = ok ? (pos.side === "SHORT" ? (pos.entry_price - mark) : (mark - pos.entry_price)) * pos.quantity : null;
    const base = isNum(pos.entry_price) && isNum(pos.quantity) ? pos.entry_price * pos.quantity : 0;
    const held = pos.entry_time ? (time - new Date(pos.entry_time)) / 1000 : null;
    return { mark: ok ? mark : null, upnl, upct: upnl !== null && base > 0 ? upnl / base * 100 : null, held: isNum(held) ? held : null };
  };

  const TABS = [
    ["overview", "Overview"], ["markets", "Markets"], ["signals", "Signals"], ["trades", "Trades", positions.length],
    ["performance", "Performance"], ["system", "System"], ["settings", "Settings"],
  ];
  const tabKeys = TABS.map(t => t[0]);
  const onTabKey = e => {
    const i = tabKeys.indexOf(activePanel);
    let next = null;
    if (e.key === "ArrowRight") next = tabKeys[(i + 1) % tabKeys.length];
    if (e.key === "ArrowLeft") next = tabKeys[(i - 1 + tabKeys.length) % tabKeys.length];
    if (e.key === "Home") next = tabKeys[0];
    if (e.key === "End") next = tabKeys[tabKeys.length - 1];
    if (next) { e.preventDefault(); setActivePanel(next); document.getElementById(`lzq-tab-${next}`)?.focus(); }
  };

  const kpis = [
    { l: "Equity", v: money(equity), s: `${signedPct(eqChangePct)} since start`, sc: tone(eqChangePct) },
    { l: "Realized P&L", v: signedMoney(pnl), vc: tone(pnl), s: "From closed trades" },
    { l: "Win rate", v: m.total_trades > 0 ? ((m.win_rate || 0) * 100).toFixed(0) + "%" : "—", s: `${m.winning_trades || 0} won, ${m.losing_trades || 0} lost` },
    { l: "Closed trades", v: m.total_trades || 0, s: `${positions.length} open now` },
    { l: "Drawdown", v: (isNum(m.current_drawdown_pct) ? m.current_drawdown_pct : 0).toFixed(1) + "%", s: "Below peak equity" },
    { l: "Signals", v: (stats.total_signals || signals.length).toLocaleString("en-US"), s: isNum(stats.msg_per_sec) ? `${stats.msg_per_sec.toFixed(1)} market msgs/s` : "Waiting for data" },
    { l: "Avg latency", v: latencyStats ? `${Math.round(latencyStats.avg)} ms` : "—", s: latencyStats && isNum(latencyStats.inf) ? `Model ${Math.round(latencyStats.inf)} ms` : "Per signal" },
  ];

  const divList = divAlerts.slice().reverse();
  const renderDivAlerts = limit => divList.length === 0
    ? <Empty>None yet. An alert appears when sentiment and price move strongly in opposite directions.</Empty>
    : (
      <div className="lzq-feed">
        {divList.slice(0, limit).map((d, i) => (
          <div key={i} className="lzq-sig" style={{ borderLeftColor: "var(--div)" }}>
            <div className="lzq-sig-top">
              <span className="lzq-pill div">{divLabel(d.divergence_type)}</span>
              <span className="lzq-sig-sym">{short(d.symbol)}</span>
              <span className="lzq-sig-meta"><span>{clock(d.time)}</span></span>
            </div>
            <p className="lzq-sig-text" style={{ marginBottom: 0 }} title={d.description || ""}>
              {d.description || (d.divergence_type === "BULLISH_DIVERGENCE" ? "Sentiment turned unusually bearish while price momentum rose." : "Sentiment turned unusually bullish while price momentum fell.")}
              {isNum(d.severity) && <span className="muted"> Severity {d.severity.toFixed(2)} of 1.</span>}
            </p>
          </div>
        ))}
      </div>
    );

  const settingDefs = [
    { l: "Minimum confidence", id: "minConfidence", min: 30, max: 90, step: 5, suffix: "%", d: "A trade opens only when the model's top score is at least this high." },
    { l: "Risk per trade", id: "riskPerTrade", min: 1, max: 50, step: 1, suffix: "%", d: "Position size as a share of current equity." },
    { l: "Stop loss", id: "stopLoss", min: 0.5, max: 5, step: 0.5, suffix: "%", d: "Close a position once it is down this much." },
    { l: "Take profit", id: "takeProfit", min: 1, max: 10, step: 0.5, suffix: "%", d: "Close a position once it is up this much." },
    { l: "Max positions", id: "maxPositions", min: 1, max: 10, step: 1, suffix: "", d: "Most positions open at once, across every symbol." },
    { l: "Max drawdown", id: "maxDrawdown", min: 5, max: 50, step: 5, suffix: "%", d: "Close everything and halt trading if equity falls this far below its peak." },
    { l: "Divergence boost", id: "divBoost", min: 1, max: 3, step: 0.5, suffix: "×", d: "Multiply position size when a divergence is detected (still capped by the max position size)." },
  ];

  const pnlBarMax = symbolPnl.reduce((mx, [, v]) => Math.max(mx, Math.abs(v)), 0) || 1;

  return (
    <div className="lzq">
      {/* TOP BAR */}
      <header className="lzq-top">
        <div className="lzq-bar">
          <div className="lzq-brand">
            <img src="/favicon.png" alt="" />
            <div>
              <h1>LZ-Quant</h1>
              <p>Sentiment-driven paper trading for crypto and US stocks</p>
            </div>
          </div>
          <div className="lzq-status">
            <span className="lzq-chip" title={mode === "live" ? `Connected to ${WS_URL}` : mode === "demo" ? "Server unreachable, showing simulated data" : "Trying to reach the server"}>
              <span className={`lzq-dot${mode === "live" ? " lzq-pulse" : ""}`} style={{ background: modeColor }} aria-hidden="true" />
              <b>{modeLabel}</b>
            </span>
            <span className="lzq-chip">Model <b>{modelLabel}</b></span>
            {divAlerts.length > 0 && (
              <button type="button" className="lzq-chip is-div" onClick={() => setActivePanel("signals")}>
                {divAlerts.length} divergence {divAlerts.length === 1 ? "alert" : "alerts"}
              </button>
            )}
            {isPaused && <span className="lzq-chip is-paused">Trading paused</span>}
            {m.is_halted && <span className="lzq-chip is-halt">Trading halted</span>}
          </div>
          <span className="lzq-clock" aria-label="Local time">{time.toLocaleTimeString("en-GB", { hourCycle: "h23" })}</span>
          <button type="button" className={`lzq-btn ${isPaused ? "resume" : "pause"}`} onClick={handlePauseResume}>
            {isPaused ? "▶ Resume trading" : "❚❚ Pause trading"}
          </button>
        </div>

        {mode === "demo" && (
          <div className="lzq-banner demo" role="status">
            <strong>You're looking at simulated demo data.</strong> The dashboard couldn't reach the trading server at <span className="code">{WS_URL}</span>.
            Start it with <span className="code">python dual.py</span>, then reload this page.
          </div>
        )}
        {m.is_halted && (
          <div className="lzq-banner halt" role="alert">
            <strong>Trading is halted.</strong> Equity fell past the max drawdown limit, so every position was closed.
            Restart the server or call <span className="code">POST /api/trading/reset</span> to trade again.
          </div>
        )}

        <div className="lzq-kpis">
          {kpis.map(k => (
            <div key={k.l} className="lzq-kpi">
              <div className="lzq-kpi-label">{k.l}</div>
              <div className={`lzq-kpi-value${k.vc ? " " + k.vc : ""}`}>{k.v}</div>
              <div className={`lzq-kpi-sub${k.sc ? " " + k.sc : ""}`}>{k.s}</div>
            </div>
          ))}
        </div>

        <nav className="lzq-tabs" role="tablist" aria-label="Dashboard sections" onKeyDown={onTabKey}>
          {TABS.map(([k, l, count]) => (
            <button key={k} id={`lzq-tab-${k}`} type="button" role="tab" className="lzq-tab"
              aria-selected={activePanel === k} aria-controls="lzq-tabpanel" tabIndex={activePanel === k ? 0 : -1}
              onClick={() => setActivePanel(k)}>
              {l}{count > 0 && <span className="lzq-count">{count}</span>}
            </button>
          ))}
        </nav>
      </header>

      <main id="lzq-tabpanel" className="lzq-main" role="tabpanel" aria-labelledby={`lzq-tab-${activePanel}`}>

        {/* ── OVERVIEW ── */}
        {activePanel === "overview" && (<>
          <div className="lzq-row r-2-1">
            <section className="lzq-panel">
              <div className="lzq-hero">
                <div>
                  <div className="lzq-hero-label">Paper account equity</div>
                  <div className="lzq-hero-value">{money(equity)}</div>
                  <div className={`lzq-hero-sub ${tone(eqChange)}`}>
                    {signedMoney(eqChange)} ({signedPct(eqChangePct)}) <span className="muted">since the {money(startingCapital, 0)} start</span>
                  </div>
                </div>
                <Seg label="Chart" value={activeChart} onChange={setActiveChart}
                  options={[["equity", "Equity"], ["sentiment", "Sentiment"], ["latency", "Latency"]]} />
              </div>

              {activeChart === "equity" && (eqCurve.length < 2
                ? <Empty title="No equity history yet">The curve starts once the server processes its first signals.</Empty>
                : (<>
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={eqCurve} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
                      <defs><linearGradient id="lzqEq" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={pnlCol} stopOpacity={0.22} /><stop offset="100%" stopColor={pnlCol} stopOpacity={0} /></linearGradient></defs>
                      <CartesianGrid vertical={false} stroke={lineCol} />
                      <XAxis dataKey="t" tick={axisTick} tickLine={false} axisLine={{ stroke: lineStrong }} minTickGap={48} />
                      <YAxis domain={[d => Math.floor(d * 0.998), d => Math.ceil(d * 1.002)]} tick={axisTick} tickLine={false} axisLine={false} width={78}
                        tickFormatter={v => "$" + Math.round(v).toLocaleString("en-US")} />
                      <ReferenceLine y={startingCapital} stroke={ink3} strokeDasharray="4 4" label={{ value: "Start", position: "insideTopLeft", fill: ink3, fontSize: 12 }} />
                      <Tooltip {...chartTip} formatter={v => [money(v), "Equity"]} />
                      <Area type="monotone" dataKey="v" name="Equity" stroke={pnlCol} strokeWidth={2} fill="url(#lzqEq)" dot={false} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                  <p className="lzq-note">Cash plus the value of open positions, sampled as the server processes signals. The dashed line is your starting capital.</p>
                </>))}

              {activeChart === "sentiment" && (sentHistory.length < 2
                ? <Empty title="No sentiment history yet">This chart fills in as signals arrive.</Empty>
                : (<>
                  <div className="lzq-legend">
                    <span><span className="lzq-swatch line" style={{ background: bull }} />Bullish</span>
                    <span><span className="lzq-swatch line" style={{ background: flat }} />Neutral (dashed)</span>
                    <span><span className="lzq-swatch line" style={{ background: bear }} />Bearish</span>
                  </div>
                  <ResponsiveContainer width="100%" height={262}>
                    <AreaChart data={sentHistory} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
                      <defs>
                        <linearGradient id="lzqSB" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={bull} stopOpacity={0.2} /><stop offset="100%" stopColor={bull} stopOpacity={0} /></linearGradient>
                        <linearGradient id="lzqSR" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={bear} stopOpacity={0.2} /><stop offset="100%" stopColor={bear} stopOpacity={0} /></linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} stroke={lineCol} />
                      <XAxis dataKey="i" hide />
                      <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={axisTick} tickLine={false} axisLine={false} width={44} tickFormatter={v => `${Math.round(v * 100)}%`} />
                      <Tooltip {...chartTip} labelFormatter={() => "Rolling average"} formatter={(v, name) => [`${Math.round(v * 100)}%`, name]} />
                      <Area type="monotone" dataKey="b" name="Bullish" stroke={bull} strokeWidth={2} fill="url(#lzqSB)" dot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="r" name="Bearish" stroke={bear} strokeWidth={2} fill="url(#lzqSR)" dot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="n" name="Neutral" stroke={flat} strokeWidth={2} strokeDasharray="5 4" fill="none" dot={false} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                  <p className="lzq-note">Average model scores over the previous 12 signals, across all symbols. Shows the last 50 signals.</p>
                </>))}

              {activeChart === "latency" && (latencyChart.length === 0
                ? <Empty title="No latency data yet">Each processed signal adds a bar here.</Empty>
                : (<>
                  <div className="lzq-legend">
                    <span><span className="lzq-swatch" style={{ background: acc }} />Model inference</span>
                    <span><span className="lzq-swatch" style={{ background: "#3A5A99" }} />Rest of the pipeline</span>
                  </div>
                  <ResponsiveContainer width="100%" height={262}>
                    <BarChart data={latencyChart} margin={{ top: 8, right: 8, bottom: 0, left: 4 }} barCategoryGap={2}>
                      <CartesianGrid vertical={false} stroke={lineCol} />
                      <XAxis dataKey="k" hide />
                      <YAxis tick={axisTick} tickLine={false} axisLine={false} width={52} tickFormatter={v => `${v} ms`} />
                      <Tooltip {...chartTip} labelFormatter={k => `Signal ${k} of the last ${latencyChart.length}`} formatter={(v, name) => [`${v.toFixed(1)} ms`, name]} />
                      <Bar dataKey="inf" name="Model inference" stackId="lat" fill={acc} stroke="#131E33" strokeWidth={1} isAnimationActive={false} />
                      <Bar dataKey="rest" name="Rest of the pipeline" stackId="lat" fill="#3A5A99" stroke="#131E33" strokeWidth={1} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="lzq-note">Time for each of the last {latencyChart.length} signals, from receiving the text to finishing the trade decision.</p>
                </>))}
            </section>

            <div className="lzq-stack">
              <Panel title="Market mood" aside={`${(sentimentTotals.bull + sentimentTotals.neut + sentimentTotals.bear).toLocaleString("en-US")} signals seen by this page`}>
                <SentimentSplit b={sentiment.bullPct} n={sentiment.neutPct} r={sentiment.bearPct} />
                <div className="lzq-split-labels">
                  <span><span className="lzq-swatch" style={{ background: bull }} />Bullish <b>{sentiment.bull}</b></span>
                  <span><span className="lzq-swatch" style={{ background: flat }} />Neutral <b>{sentiment.neut}</b></span>
                  <span><span className="lzq-swatch" style={{ background: bear }} />Bearish <b>{sentiment.bear}</b></span>
                </div>
                <p className="lzq-note" style={{ marginTop: 10 }}>Share of this session's signals where each sentiment scored highest.</p>
              </Panel>
              <Panel title="Divergence alerts" aside={divList.length > 3 ? <button type="button" className="lzq-link" onClick={() => setActivePanel("signals")}>See all {divList.length}</button> : null}>
                {renderDivAlerts(3)}
              </Panel>
              <Panel title="Open positions" aside={<button type="button" className="lzq-link" onClick={() => setActivePanel("trades")}>Trades</button>}>
                {positions.length === 0
                  ? <Empty title="No open positions">{mode === "demo" ? "Demo mode doesn't simulate open positions." : "A position opens when a signal passes the confidence and position limits."}</Empty>
                  : (
                    <div className="lzq-prices">
                      {positions.map((pos, i) => {
                        const v = positionView(pos);
                        return (
                          <div key={pos.symbol || i} className="lzq-price" style={{ gridTemplateColumns: "10px minmax(0,1fr) auto" }}>
                            <span className="lzq-dot" style={{ background: SYM[pos.symbol]?.c || ink3 }} aria-hidden="true" />
                            <div style={{ minWidth: 0 }}>
                              <div className="sym">{short(pos.symbol)} <span className={`lzq-pill ${pos.side === "LONG" ? "buy" : "sell"}`} style={{ marginLeft: 4 }}>{pos.side === "LONG" ? "Long" : "Short"}</span></div>
                              <div className="name">Entry {priceFmt(pos.entry_price)}{v.held !== null ? `, held ${duration(v.held)}` : ""}</div>
                            </div>
                            <div>
                              <div className={`px ${tone(v.upnl)}`}>{signedMoney(v.upnl)}</div>
                              <div className={`chg ${tone(v.upct)}`}>{signedPct(v.upct)}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                {positions.length > 0 && <p className="lzq-note" style={{ marginTop: 8 }}>Unrealized P&amp;L, estimated from the latest price on this page.</p>}
              </Panel>
            </div>
          </div>

          <div className="lzq-row r-1-1">
            <Panel title="Prices" aside="Change over last 30 updates">
              <PriceList symbols={CRYPTO} tickers={tickers} sparkData={sparkData} />
              <div className="lzq-panel-head" style={{ margin: "16px 0 4px" }}><h2 style={{ fontSize: 14.5 }}>US stocks</h2></div>
              <PriceList symbols={STOCKS} tickers={tickers} sparkData={sparkData} />
            </Panel>
            <Panel title="Latest signals" aside={<button type="button" className="lzq-link" onClick={() => setActivePanel("signals")}>All signals</button>}>
              {signals.length === 0
                ? <Empty title="Waiting for signals">They appear as the server scores news, posts and price moves.</Empty>
                : <div className="lzq-feed">{signals.slice(-6).reverse().map((s, i) => <SignalCard key={s.id || i} sig={s} />)}</div>}
            </Panel>
          </div>
        </>)}

        {/* ── MARKETS ── */}
        {activePanel === "markets" && (<>
          <div className="lzq-row r-1-1">
            <MarketPanel title="Crypto" source="Binance, live trades" symbols={CRYPTO} panelSignals={cryptoSignals} tickers={tickers} sparkData={sparkData}
              emptyText="Waiting for crypto signals." />
            <MarketPanel title="US stocks" source="Alpaca IEX feed" symbols={STOCKS} panelSignals={stockSignals} tickers={tickers} sparkData={sparkData}
              emptyText="No stock signals yet. Stocks stream during US market hours (9:30 AM to 4:00 PM ET) when the server has Alpaca API keys." />
          </div>
          <p className="lzq-note">Change compares the latest price with the oldest of the last 30 price updates on this page. Sentiment covers each market's last 40 signals.</p>
        </>)}

        {/* ── SIGNALS ── */}
        {activePanel === "signals" && (
          <div className="lzq-row r-2-1">
            <Panel title="Signal feed" aside={`${filteredSignals.length} of ${signals.length} shown`}>
              <div className="lzq-filters">
                <Seg label="Market" value={sigMarket} onChange={setSigMarket} options={[["all", "All markets"], ["crypto", "Crypto"], ["stocks", "Stocks"]]} />
                <Seg label="Action" value={sigAction} onChange={setSigAction} options={[["all", "Any action"], ["BUY", "Buy"], ["SELL", "Sell"], ["HOLD", "Hold"]]} />
                <label className="check"><input type="checkbox" checked={sigTradedOnly} onChange={e => setSigTradedOnly(e.target.checked)} />Only signals that traded</label>
              </div>
              {filteredSignals.length === 0
                ? <Empty title={signals.length ? "No signals match these filters" : "Waiting for signals"}>{signals.length ? "Try a different market or action." : "They appear as the server scores news, posts and price moves."}</Empty>
                : <div className="lzq-feed scroll">{filteredSignals.map((s, i) => <SignalCard key={s.id || i} sig={s} />)}</div>}
              <p className="lzq-note" style={{ marginTop: 10 }}>This page keeps the latest 100 signals. Hover over a signal's text to read all of it.</p>
            </Panel>
            <div className="lzq-stack">
              <Panel title="How a signal is made">
                <ol className="lzq-explain">
                  <li>The text is a news headline, a Reddit post, or a sentence the server writes about the latest price move.</li>
                  <li>The sentiment model scores it bullish, neutral and bearish. The three scores add up to 100%.</li>
                  <li>It becomes <b className="up">Buy</b> if bullish is at least 45%, <b className="down">Sell</b> if bearish is at least 45%, and <b className="flat">Hold</b> otherwise. A detected divergence overrides this.</li>
                  <li>The paper trader opens a position only if the top score meets the minimum confidence and the position limit allows it. An opposite signal closes an open position.</li>
                </ol>
                {mode === "demo" && <p className="lzq-note" style={{ marginTop: 10 }}>This describes the live server. Demo signals are random.</p>}
              </Panel>
              <Panel title="Divergence alerts" aside="Latest 5">
                {renderDivAlerts(5)}
              </Panel>
            </div>
          </div>
        )}

        {/* ── TRADES ── */}
        {activePanel === "trades" && (<>
          <Panel title="Open positions" aside={`${positions.length} open`}>
            {positions.length === 0
              ? <Empty title="No open positions">{mode === "demo" ? "Demo mode doesn't simulate open positions." : "A position opens when a signal passes the confidence and position limits."}</Empty>
              : (
                <div className="lzq-table-wrap">
                  <table className="lzq-table">
                    <thead><tr><th>Symbol</th><th>Side</th><th className="r">Entry</th><th className="r">Latest price</th><th className="r">Quantity</th><th className="r">Size</th><th className="r">Unrealized P&amp;L</th><th className="r">Held for</th></tr></thead>
                    <tbody>
                      {positions.map((pos, i) => {
                        const v = positionView(pos);
                        return (
                          <tr key={pos.symbol || i}>
                            <td><span className="sym"><span className="lzq-dot" style={{ background: SYM[pos.symbol]?.c || ink3 }} aria-hidden="true" />{short(pos.symbol)}{pos.divergence && <span className="lzq-pill div">Divergence</span>}</span></td>
                            <td><span className={`lzq-pill ${pos.side === "LONG" ? "buy" : "sell"}`}>{pos.side === "LONG" ? "Long" : "Short"}</span></td>
                            <td className="r">{priceFmt(pos.entry_price)}</td>
                            <td className="r">{priceFmt(v.mark)}</td>
                            <td className="r">{qtyFmt(pos.quantity)}</td>
                            <td className="r">{money(pos.notional)}</td>
                            <td className={`r ${tone(v.upnl)}`}>{signedMoney(v.upnl)} <span className="muted" style={{ fontSize: 13 }}>{signedPct(v.upct)}</span></td>
                            <td className="r">{duration(v.held)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            {positions.length > 0 && <p className="lzq-note" style={{ marginTop: 10 }}>Unrealized P&amp;L is estimated in your browser from the latest price shown on this page.</p>}
          </Panel>

          <Panel title="Trade log" aside={tradeLog.length ? `Latest ${tradeLog.length} events, newest first` : null}>
            {tradeLog.length === 0
              ? <Empty title="No trades yet">Opened and closed positions are listed here as they happen.</Empty>
              : (
                <div className="lzq-table-wrap">
                  <table className="lzq-table">
                    <thead><tr><th>Time</th><th>Symbol</th><th>Event</th><th>Side</th><th className="r">P&amp;L</th><th className="r">Return</th></tr></thead>
                    <tbody>
                      {tradeLog.slice().reverse().map((t, i) => {
                        const isOpen = t.type === "open";
                        const side = isOpen ? sideOfAction(t.action) : t.side;
                        const entry = isOpen && side && t.details?.side === side ? t.details.entry_price : null;
                        const event = isOpen ? (tradeOutcome(t.action)?.text || "Opened") : exitLabel(t.exit_reason);
                        return (
                          <tr key={i}>
                            <td className="muted">{clock(t.time)}</td>
                            <td><span className="sym"><span className="lzq-dot" style={{ background: SYM[t.symbol]?.c || ink3 }} aria-hidden="true" />{short(t.symbol)}</span></td>
                            <td>{event}{isNum(entry) && <span className="muted"> at {priceFmt(entry)}</span>}</td>
                            <td>{side ? <span className={`lzq-pill ${side === "LONG" ? "buy" : "sell"}`}>{side === "LONG" ? "Long" : "Short"}</span> : <span className="muted">—</span>}</td>
                            <td className={`r ${isOpen ? "" : tone(t.pnl)}`}>{isOpen ? <span className="muted">—</span> : signedMoney(t.pnl)}</td>
                            <td className={`r ${isOpen ? "" : tone(t.pnl_pct)}`}>{isOpen || !isNum(t.pnl_pct) ? <span className="muted">—</span> : signedPct(t.pnl_pct)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            <p className="lzq-note" style={{ marginTop: 10 }}>This log lists the trade events received by this page. The totals in the header and the Performance tab come from the server.</p>
          </Panel>
        </>)}

        {/* ── PERFORMANCE ── */}
        {activePanel === "performance" && (<>
          <div className="lzq-row r-1-1">
            <Panel title="Account" aside="From the server">
              <dl className="lzq-metrics">
                <Metric label="Equity" value={money(equity)} />
                <Metric label="Starting capital" value={money(startingCapital)} />
                <Metric label="Realized P&L" value={signedMoney(pnl)} cls={tone(pnl)} />
                <Metric label="Unrealized P&L" value={signedMoney(m.unrealized_pnl)} cls={tone(m.unrealized_pnl)} />
                <Metric label="Peak equity" value={money(m.peak_equity)} />
                <Metric label="Below peak" value={isNum(m.current_drawdown_pct) ? m.current_drawdown_pct.toFixed(2) + "%" : "—"} />
                <Metric label="Trading" value={tradingState} cls={m.is_halted ? "down" : undefined} />
              </dl>
            </Panel>
            <Panel title="Closed trades" aside="From the server">
              <dl className="lzq-metrics">
                <Metric label="Trades closed" value={m.total_trades || 0} />
                <Metric label="Won / lost" value={`${m.winning_trades || 0} / ${m.losing_trades || 0}`} />
                <Metric label="Win rate" value={m.total_trades > 0 ? ((m.win_rate || 0) * 100).toFixed(1) + "%" : "—"} />
                <Metric label="Profit factor" value={isNum(m.profit_factor) && m.profit_factor > 0 ? m.profit_factor.toFixed(2) : "—"} sub="profit ÷ loss" />
                <Metric label="Average win" value={isNum(m.avg_win) && m.winning_trades > 0 ? signedMoney(m.avg_win) : "—"} cls="up" />
                <Metric label="Average loss" value={isNum(m.avg_loss) && m.losing_trades > 0 ? signedMoney(m.avg_loss) : "—"} cls="down" />
                <Metric label="Largest win" value={isNum(m.largest_win) && m.winning_trades > 0 ? signedMoney(m.largest_win) : "—"} />
                <Metric label="Largest loss" value={isNum(m.largest_loss) && m.losing_trades > 0 ? signedMoney(m.largest_loss) : "—"} />
                <Metric label="Average hold" value={isNum(m.avg_holding_time_s) && m.total_trades > 0 ? duration(m.avg_holding_time_s) : "—"} />
                <Metric label="Divergence trades won" value={isNum(m.divergence_trade_winrate) && m.total_trades > 0 ? (m.divergence_trade_winrate * 100).toFixed(0) + "%" : "—"} sub={m.divergence_trade_winrate === 0 ? "or none yet" : undefined} />
              </dl>
            </Panel>
          </div>

          <div className="lzq-row r-1-1">
            <Panel title="This session" aside={`${closedTrades.length} closed ${closedTrades.length === 1 ? "trade" : "trades"} seen by this page`}>
              {closedTrades.length < 2
                ? <Empty title="Not enough trades yet">These figures need at least two closed trades seen by this page.</Empty>
                : (
                  <dl className="lzq-metrics">
                    <Metric label="Total P&L" value={signedMoney(tradeAnalytics.totalPnL)} cls={tone(tradeAnalytics.totalPnL)} />
                    <Metric label="Average trade" value={signedMoney(toNum(analytics.avgTrade))} cls={tone(toNum(analytics.avgTrade))} />
                    <Metric label="Best trade" value={signedMoney(toNum(analytics.bestTrade))} />
                    <Metric label="Worst trade" value={signedMoney(toNum(analytics.worstTrade))} />
                    <Metric label="Max drawdown" value={analytics.maxDD !== null ? analytics.maxDD + "%" : "—"} sub="of capital" />
                    <Metric label="Sharpe ratio" value={analytics.sharpe || "—"} />
                    <Metric label="Calmar ratio" value={analytics.calmar || "—"} />
                    <Metric label="Average hold" value={analytics.avgHold ? analytics.avgHold + " min" : "—"} />
                    <Metric label="Best symbol" value={tradeAnalytics.bestSym || "—"} />
                    <Metric label="Worst symbol" value={tradeAnalytics.worstSym || "—"} />
                  </dl>
                )}
              <p className="lzq-note" style={{ marginTop: 10 }}>Calculated in your browser from trades closed while this page was open, so they can differ from the server totals. Sharpe annualizes per-trade returns with √252, so treat it as a rough guide.</p>
            </Panel>

            <Panel title="Exposure and P&L by symbol">
              <dl className="lzq-metrics" style={{ marginBottom: 18 }}>
                <Metric label="Total exposure" value={riskMetrics.totalExp + "%"} sub="of equity" />
                <Metric label="Crypto" value={riskMetrics.cExp + "%"} sub="of equity" />
                <Metric label="Stocks" value={riskMetrics.sExp + "%"} sub="of equity" />
                <Metric label="Open positions" value={riskMetrics.openPos} sub={`limit ${riskMetrics.maxPos}`} />
              </dl>
              {symbolPnl.length === 0
                ? <Empty>Realized P&amp;L per symbol appears after the first closed trade.</Empty>
                : (
                  <div className="lzq-bars" role="list" aria-label="Realized P&L by symbol">
                    {symbolPnl.map(([sym, v]) => {
                      const w = Math.abs(v) / pnlBarMax * 50;
                      return (
                        <div key={sym} className="lzq-barrow" role="listitem">
                          <span className="lbl">{short(sym)}</span>
                          <div className="lzq-bartrack">
                            <div style={{ left: v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? bull : bear }} />
                          </div>
                          <span className={`val ${tone(v)}`}>{signedMoney(v)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              <p className="lzq-note" style={{ marginTop: 10 }}>Exposure is the entry value of open positions. The position limit is the Max positions setting on this page.</p>
            </Panel>
          </div>
        </>)}

        {/* ── SYSTEM ── */}
        {activePanel === "system" && (<>
          <div className="lzq-row r-3">
            <Panel title="Connection">
              <dl className="lzq-kv">
                <div><dt>Status</dt><dd style={{ color: modeColor }}>{modeLabel}</dd></div>
                <div><dt>Sentiment model</dt><dd>{modelLabel}</dd></div>
                <div><dt>Trading</dt><dd className={m.is_halted ? "down" : undefined}>{tradingState}</dd></div>
                <div><dt>WebSocket</dt><dd><span className="code">{WS_URL}</span></dd></div>
                <div><dt>REST API</dt><dd><span className="code">{API_URL}</span></dd></div>
              </dl>
            </Panel>
            <Panel title="Throughput">
              <dl className="lzq-kv">
                <div><dt>Market messages per second</dt><dd className="num">{isNum(stats.msg_per_sec) ? stats.msg_per_sec.toFixed(1) : "—"}</dd></div>
                <div><dt>Market messages received</dt><dd className="num">{isNum(stats.total_messages) ? stats.total_messages.toLocaleString("en-US") : "—"}</dd></div>
                <div><dt>Signals generated</dt><dd className="num">{(stats.total_signals || signals.length).toLocaleString("en-US")}</dd></div>
                <div><dt>Server uptime</dt><dd className="num">{isNum(stats.uptime_seconds) ? duration(stats.uptime_seconds) : "—"}</dd></div>
              </dl>
            </Panel>
            <Panel title="Latency" aside={latencyStats ? `Last ${latencyStats.n} signals` : null}>
              <dl className="lzq-kv">
                <div><dt>Average</dt><dd className="num">{latencyStats ? `${latencyStats.avg.toFixed(1)} ms` : "—"}</dd></div>
                <div><dt>Median</dt><dd className="num">{latencyStats ? `${latencyStats.p50.toFixed(1)} ms` : "—"}</dd></div>
                <div><dt>95th percentile</dt><dd className="num">{latencyStats ? `${latencyStats.p95.toFixed(1)} ms` : "—"}</dd></div>
                <div><dt>Model inference (average)</dt><dd className="num">{latencyStats && isNum(latencyStats.inf) ? `${latencyStats.inf.toFixed(1)} ms` : "—"}</dd></div>
              </dl>
            </Panel>
          </div>

          <div className="lzq-row r-2-1">
            <Panel title="How the pipeline works">
              <ol className="lzq-steps">
                <li><b>Prices</b>Binance streams crypto trades and Alpaca streams US stock trades.</li>
                <li><b>Text</b>News feeds and Reddit are checked every 45 to 60 seconds. Each item is tagged with the ticker it mentions.</li>
                <li><b>Sentiment</b>A fine-tuned DistilBERT model, run with ONNX, scores each text bullish, neutral or bearish.</li>
                <li><b>Divergence</b>Rolling z-scores flag moments when sentiment and price momentum disagree.</li>
                <li><b>Paper trading</b>Signals open and close simulated positions with stop-loss, take-profit and trailing stops.</li>
                <li><b>Dashboard</b>Every event is pushed to this page over a WebSocket as it happens.</li>
              </ol>
              <p className="lzq-note" style={{ marginTop: 12 }}>Paper trading only. No real orders are ever placed.</p>
            </Panel>
            <Panel title="Text sources" aside="After removing duplicates">
              {!ingestion
                ? <Empty>{mode === "demo" ? "Text-source counts come from the live server and aren't available in demo mode." : "Waiting for the server's first stats update."}</Empty>
                : (
                  <dl className="lzq-kv">
                    <div><dt>Reddit posts</dt><dd className="num">{isNum(ingestion.reddit) ? ingestion.reddit.toLocaleString("en-US") : "—"}</dd></div>
                    <div><dt>News articles (RSS)</dt><dd className="num">{isNum(ingestion.rss) ? ingestion.rss.toLocaleString("en-US") : "—"}</dd></div>
                    <div><dt>Items per minute</dt><dd className="num">{isNum(ingestion.items_per_min) ? ingestion.items_per_min.toFixed(1) : "—"}</dd></div>
                    <div><dt>Waiting in queue</dt><dd className="num">{isNum(ingestion.queue_size) ? ingestion.queue_size : "—"}</dd></div>
                  </dl>
                )}
            </Panel>
          </div>
        </>)}

        {/* ── SETTINGS ── */}
        {activePanel === "settings" && (
          <Panel title="Trading settings" aside={mode === "live" ? "Applies to the running server" : "Server not connected"}>
            <p className="lzq-note" style={{ marginBottom: 14 }}>
              Changes take effect on the server as soon as you apply them, and reset when it restarts. The sliders start at the server's default values; this page doesn't read the server's current settings.
            </p>
            <div className="lzq-settings">
              {settingDefs.map(({ l, id, min, max, step, suffix, d }) => (
                <div key={id} className="lzq-setting">
                  <div className="lzq-setting-head">
                    <label htmlFor={`lzq-set-${id}`}>{l}</label>
                    <output htmlFor={`lzq-set-${id}`}>{settings[id]}{suffix}</output>
                  </div>
                  <input id={`lzq-set-${id}`} type="range" min={min} max={max} step={step} value={settings[id]}
                    onChange={e => setSettings(p => ({ ...p, [id]: parseFloat(e.target.value) }))} />
                  <div className="range-ends"><span>{min}{suffix}</span><span>{max}{suffix}</span></div>
                  <p>{d}</p>
                </div>
              ))}
              <div className="lzq-setting">
                <div className="lzq-setting-head">
                  <span className="t" id="lzq-set-reqdiv">Require divergence</span>
                  <button type="button" role="switch" className="lzq-switch" aria-checked={settings.requireDiv} aria-labelledby="lzq-set-reqdiv"
                    onClick={() => setSettings(p => ({ ...p, requireDiv: !p.requireDiv }))} />
                </div>
                <p>{settings.requireDiv ? "On: " : "Off: "}only trade when a divergence between sentiment and price is detected.</p>
              </div>
            </div>
            <div className="lzq-apply">
              <button type="button" className="lzq-btn primary" onClick={saveSettings} disabled={mode !== "live"}>Apply to server</button>
              <span className={savedMsg ? "up" : "muted"} role="status">
                {savedMsg ? "Settings applied to the server." : mode !== "live" ? "Connect to the live server to apply settings." : ""}
              </span>
            </div>
          </Panel>
        )}
      </main>
    </div>
  );
}
