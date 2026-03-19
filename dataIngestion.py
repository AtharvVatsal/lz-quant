"""
LZ-QUANT — REAL TEXT INGESTION

Streams real human text from Reddit and RSS news feeds.
No API keys needed — uses public endpoints only.

Sources:
  Reddit: r/CryptoCurrency, r/Bitcoin, r/ethereum, r/solana,
          r/wallstreetbets, r/stocks, r/investing, r/options
  RSS:    CoinDesk, CoinTelegraph, Decrypt, MarketWatch, Yahoo Finance, CNBC

Usage:
  from data_ingestion import TextRouter
  router = TextRouter()
  await router.start()
  item = await router.get_next()  # TextItem(text, symbol, market, source, ...)

Test standalone:
  python data_ingestion.py

"""

import asyncio
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


# CONFIG

REDDIT_SOURCES = [
    ("CryptoCurrency", "crypto", 15), ("Bitcoin", "crypto", 15),
    ("ethereum", "crypto", 15), ("solana", "crypto", 15),
    ("CryptoMarkets", "crypto", 15),
    ("wallstreetbets", "stocks", 15), ("stocks", "stocks", 15),
    ("investing", "stocks", 15), ("options", "stocks", 15),
    ("StockMarket", "stocks", 15),
]

RSS_SOURCES = [
    ("https://www.coindesk.com/arc/outboundfeeds/rss/", "crypto", 45),
    ("https://cointelegraph.com/rss", "crypto", 45),
    ("https://decrypt.co/feed", "crypto", 45),
    ("https://feeds.content.dowjones.io/public/rss/mw_topstories", "stocks", 45),
    ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "stocks", 45),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "stocks", 45),
]

CRYPTO_SYMBOLS = {
    "BTCUSDT": [r"\bBTC\b", r"\bbitcoin\b", r"\bBitcoin\b"],
    "ETHUSDT": [r"\bETH\b", r"\bethereum\b", r"\bEthereum\b"],
    "SOLUSDT": [r"\bSOL\b", r"\bsolana\b", r"\bSolana\b"],
}
STOCK_SYMBOLS = {
    "AAPL": [r"\bAAPL\b", r"\b[Aa]pple\b"], "MSFT": [r"\bMSFT\b", r"\b[Mm]icrosoft\b"],
    "GOOGL": [r"\bGOOGL?\b", r"\b[Gg]oogle\b", r"\bAlphabet\b"],
    "NVDA": [r"\bNVDA\b", r"\b[Nn]vidia\b", r"\bNVIDIA\b"],
    "TSLA": [r"\bTSLA\b", r"\b[Tt]esla\b"], "AMD": [r"\bAMD\b"],
    "META": [r"\bMETA\b", r"\bMeta\b"], "AMZN": [r"\bAMZN\b", r"\b[Aa]mazon\b"],
    "SPY": [r"\bSPY\b", r"\bS&P\s*500\b"], "QQQ": [r"\bQQQ\b", r"\b[Nn]asdaq\b"],
}
CRYPTO_KW = [r"\bcrypto\b", r"\bcryptocurrency\b", r"\bdefi\b", r"\bblockchain\b",
             r"\btoken\b", r"\bhodl\b", r"\bmoon\b", r"\brug\s?pull\b"]
STOCK_KW = [r"\bstock\b", r"\bearnings\b", r"\bFed\b", r"\brate\b",
            r"\bmarket\b", r"\bDow\b", r"\brally\b", r"\bcrash\b"]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LZ-Quant/1.0"


# DATA TYPES

@dataclass
class TextItem:
    text: str
    symbol: str          # "BTCUSDT", "AAPL", "SPY", etc.
    market: str          # "crypto" or "stocks"
    source: str          # "reddit:Bitcoin", "rss:coindesk"
    source_type: str     # "reddit" or "rss"
    timestamp: float = 0.0
    url: str = ""
    score: int = 0       # Reddit upvotes


# SYMBOL DETECTOR

class SymbolDetector:
    def __init__(self):
        self._cp = {s: [re.compile(p) for p in ps] for s, ps in CRYPTO_SYMBOLS.items()}
        self._sp = {s: [re.compile(p) for p in ps] for s, ps in STOCK_SYMBOLS.items()}
        self._ck = [re.compile(p, re.IGNORECASE) for p in CRYPTO_KW]
        self._sk = [re.compile(p, re.IGNORECASE) for p in STOCK_KW]

    def detect(self, text, hint=""):
        results = []
        for sym, pats in self._cp.items():
            if any(p.search(text) for p in pats): results.append((sym, "crypto"))
        for sym, pats in self._sp.items():
            if any(p.search(text) for p in pats): results.append((sym, "stocks"))
        if not results:
            hc = any(p.search(text) for p in self._ck)
            hs = any(p.search(text) for p in self._sk)
            if hc and not hs: results.append(("BTCUSDT", "crypto"))
            elif hs and not hc: results.append(("SPY", "stocks"))
            elif hc and hs: results.extend([("BTCUSDT", "crypto"), ("SPY", "stocks")])
            elif hint == "crypto": results.append(("BTCUSDT", "crypto"))
            elif hint == "stocks": results.append(("SPY", "stocks"))
        return results


# REDDIT STREAM

class RedditStream:
    def __init__(self):
        self._seen = deque(maxlen=5000)
        self._det = SymbolDetector()

    def _fetch(self, url):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except: return {}

    def _clean(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"[#*_~`>]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:500]

    async def poll(self, sub, market_hint, queue, interval=30):
        urls = [
            f"https://www.reddit.com/r/{sub}/new.json?limit=15",
            f"https://www.reddit.com/r/{sub}/hot.json?limit=10",
        ]
        loop = asyncio.get_running_loop()
        print(f"[REDDIT] r/{sub} every {interval}s")

        while True:
            for url in urls:
                try:
                    data = await loop.run_in_executor(None, self._fetch, url)
                    for post in data.get("data", {}).get("children", []):
                        p = post.get("data", {})
                        pid = p.get("id", "")
                        if pid in self._seen: continue
                        self._seen.append(pid)

                        title = p.get("title", "")
                        body = p.get("selftext", "")[:300]
                        text = self._clean(f"{title}. {body}" if body else title)
                        if len(text) < 15: continue

                        matches = self._det.detect(text, market_hint)
                        if not matches:
                            matches = [("BTCUSDT" if market_hint == "crypto" else "SPY", market_hint)]

                        score = p.get("ups", 0) - p.get("downs", 0)
                        for sym, mkt in matches:
                            await queue.put(TextItem(
                                text=text, symbol=sym, market=mkt,
                                source=f"reddit:{sub}", source_type="reddit",
                                timestamp=p.get("created_utc", time.time()),
                                url=f"https://reddit.com{p.get('permalink','')}",
                                score=score,
                            ))
                except: pass
            await asyncio.sleep(interval)


# RSS STREAM

class RSSStream:
    def __init__(self):
        self._seen = deque(maxlen=3000)
        self._det = SymbolDetector()

    def _fetch(self, url):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=15) as r:
                return r.read().decode("utf-8", errors="replace")
        except: return ""

    def _parse(self, xml):
        items = []
        try: root = ET.fromstring(xml)
        except: return items
        for it in root.iter("item"):
            items.append({
                "title": it.findtext("title", ""),
                "desc": it.findtext("description", ""),
                "link": it.findtext("link", ""),
                "guid": it.findtext("guid", it.findtext("link", "")),
            })
        if not items:  # Try Atom
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                t = entry.findtext("{http://www.w3.org/2005/Atom}title", "")
                l = ""
                for lnk in entry.iter("{http://www.w3.org/2005/Atom}link"):
                    l = lnk.get("href", ""); break
                s = entry.findtext("{http://www.w3.org/2005/Atom}summary", "")
                items.append({"title": t, "desc": s, "link": l, "guid": l or t})
        return items

    def _strip_html(self, text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&\w+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:400]

    async def poll(self, url, market_hint, queue, interval=60):
        name = url.split("//")[1].split("/")[0].replace("www.", "").split(".")[0]
        loop = asyncio.get_running_loop()
        print(f"[RSS] {name} every {interval}s")

        while True:
            try:
                xml = await loop.run_in_executor(None, self._fetch, url)
                if not xml: await asyncio.sleep(interval); continue

                for item in self._parse(xml)[:15]:
                    guid = item.get("guid", "")
                    uid = hashlib.md5(guid.encode()).hexdigest()
                    if uid in self._seen: continue
                    self._seen.append(uid)

                    title = item.get("title", "").strip()
                    desc = self._strip_html(item.get("desc", ""))
                    text = (f"{title}. {desc}" if desc and desc != title else title)[:500]
                    if len(text) < 20: continue

                    matches = self._det.detect(text, market_hint)
                    if not matches:
                        matches = [("BTCUSDT" if market_hint == "crypto" else "SPY", market_hint)]

                    for sym, mkt in matches:
                        await queue.put(TextItem(
                            text=text, symbol=sym, market=mkt,
                            source=f"rss:{name}", source_type="rss",
                            timestamp=time.time(), url=item.get("link", ""),
                        ))
            except: pass
            await asyncio.sleep(interval)


# TEXT ROUTER

class TextRouter:
    """Central hub: collects from all sources, deduplicates, routes to inference."""

    def __init__(self, max_queue=500):
        self.queue = asyncio.Queue(maxsize=max_queue)
        self._reddit = RedditStream()
        self._rss = RSSStream()
        self._hashes = deque(maxlen=2000)
        self._stats = {"reddit": 0, "rss": 0, "total": 0, "start": time.time()}

    async def start(self):
        tasks = []
        for sub, mkt, interval in REDDIT_SOURCES:
            tasks.append(asyncio.create_task(self._reddit.poll(sub, mkt, self.queue, interval)))
        for url, mkt, interval in RSS_SOURCES:
            tasks.append(asyncio.create_task(self._rss.poll(url, mkt, self.queue, interval)))
        tasks.append(asyncio.create_task(self._report()))
        print(f"\n[INGESTION] {len(REDDIT_SOURCES)} Reddit + {len(RSS_SOURCES)} RSS feeds started")
        return tasks

    async def get_next(self) -> Optional[TextItem]:
        while True:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                return None
            h = hashlib.md5(item.text[:200].lower().encode()).hexdigest()
            if h in self._hashes: continue
            self._hashes.append(h)
            self._stats["reddit" if item.source_type == "reddit" else "rss"] += 1
            self._stats["total"] += 1
            return item

    async def _report(self):
        while True:
            await asyncio.sleep(60)
            elapsed = max(time.time() - self._stats["start"], 1)
            rate = self._stats["total"] / elapsed * 60
            print(f"[INGESTION] Reddit: {self._stats['reddit']}, RSS: {self._stats['rss']}, "
                  f"Rate: {rate:.1f}/min, Queue: {self.queue.qsize()}")

    def get_stats(self):
        elapsed = max(time.time() - self._stats["start"], 1)
        return {**self._stats, "items_per_min": round(self._stats["total"] / elapsed * 60, 1),
                "queue_size": self.queue.qsize()}


# STANDALONE TEST

async def _test():
    router = TextRouter()
    await router.start()
    print("\nWaiting for text...\n")
    n = 0
    while n < 30:
        item = await router.get_next()
        if item:
            n += 1
            c = "\033[93m" if item.market == "crypto" else "\033[96m"
            print(f"[{n:3d}] {c}{item.symbol:10s}\033[0m ({item.source:25s}) {item.text[:120]}...")
            print()
    print(f"\nStats: {router.get_stats()}")

if __name__ == "__main__":
    asyncio.run(_test())