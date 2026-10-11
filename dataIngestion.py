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
import html
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


# CONFIG

# (subreddit, market). The third field is unused: all subreddits share one feed
# polled every REDDIT_POLL_INTERVAL seconds to stay under Reddit's rate limit.
REDDIT_SOURCES = [
    ("CryptoCurrency", "crypto", 60), ("Bitcoin", "crypto", 60),
    ("ethereum", "crypto", 60), ("solana", "crypto", 60),
    ("CryptoMarkets", "crypto", 60),
    ("wallstreetbets", "stocks", 60), ("stocks", "stocks", 60),
    ("investing", "stocks", 60), ("options", "stocks", 60),
    ("StockMarket", "stocks", 60),
]
REDDIT_POLL_INTERVAL = 60

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
# Reddit asks for a descriptive "<platform>:<app>:<version>" User-Agent
REDDIT_UA = "python:lz-quant:v1.0 (paper-trading research)"


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
# Reddit blocks the logged-out .json endpoints (HTTP 403) and allows only a few
# logged-out requests per minute (HTTP 429). Its Atom (.rss) feeds still work, so
# all subreddits are fetched as ONE combined feed (r/a+b+c/new/.rss) per poll.

ATOM = "{http://www.w3.org/2005/Atom}"


class RedditStream:
    def __init__(self):
        self._seen = deque(maxlen=5000)
        self._det = SymbolDetector()

    def _fetch(self, url):
        """Returns (xml_text, retry_after_seconds). retry_after is set on HTTP 429."""
        try:
            req = Request(url, headers={"User-Agent": REDDIT_UA})
            with urlopen(req, timeout=15) as r:
                return r.read().decode("utf-8", errors="replace"), 0
        except HTTPError as e:
            if e.code == 429:
                reset = e.headers.get("x-ratelimit-reset") or e.headers.get("Retry-After") or 60
                try: return "", max(float(reset), 30)
                except ValueError: return "", 60
            print(f"[REDDIT] HTTP {e.code}")
            return "", 0
        except Exception as e:
            print(f"[REDDIT] Fetch failed: {e}")
            return "", 0

    def _clean(self, text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"submitted by\s+/u/\S+.*$", "", text, flags=re.DOTALL)
        text = re.sub(r"This post contains content not supported on old Reddit\.[^.]*\.?", "", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[#*_~`>]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:500]

    def _parse(self, xml):
        try: root = ET.fromstring(xml)
        except ET.ParseError: return []
        posts = []
        for e in root.iter(f"{ATOM}entry"):
            cat = e.find(f"{ATOM}category")
            link = e.find(f"{ATOM}link")
            posts.append({
                "id": e.findtext(f"{ATOM}id", ""),
                "sub": cat.get("term", "") if cat is not None else "",
                "title": html.unescape(e.findtext(f"{ATOM}title", "")),
                "body": e.findtext(f"{ATOM}content", ""),
                "url": link.get("href", "") if link is not None else "",
            })
        return posts

    async def poll(self, sources, queue, interval=60):
        markets = {sub.lower(): mkt for sub, mkt, _ in sources}
        names = {sub.lower(): sub for sub, _, _ in sources}
        url = f"https://www.reddit.com/r/{'+'.join(sub for sub, _, _ in sources)}/new/.rss?limit=100"
        loop = asyncio.get_running_loop()
        print(f"[REDDIT] {len(sources)} subreddits as one RSS feed every {interval}s")

        while True:
            xml, retry_after = await loop.run_in_executor(None, self._fetch, url)
            if retry_after:
                print(f"[REDDIT] Rate limited — retrying in {retry_after:.0f}s")
                await asyncio.sleep(retry_after)
                continue

            for p in self._parse(xml):
                if not p["id"] or p["id"] in self._seen: continue
                self._seen.append(p["id"])

                body = self._clean(p["body"])[:300]
                text = self._clean(f"{p['title']}. {body}" if body else p["title"])
                if len(text) < 15: continue

                sub = names.get(p["sub"].lower(), p["sub"])
                market_hint = markets.get(p["sub"].lower(), "crypto")
                matches = self._det.detect(text, market_hint)
                if not matches:
                    matches = [("BTCUSDT" if market_hint == "crypto" else "SPY", market_hint)]

                for sym, mkt in matches:
                    await queue.put(TextItem(
                        text=text, symbol=sym, market=mkt,
                        source=f"reddit:{sub}", source_type="reddit",
                        timestamp=time.time(), url=p["url"],
                    ))
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
        tasks = [asyncio.create_task(self._reddit.poll(REDDIT_SOURCES, self.queue, REDDIT_POLL_INTERVAL))]
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