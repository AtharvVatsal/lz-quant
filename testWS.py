import asyncio, json
import websockets

async def test():
    url = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade"
    async with websockets.connect(url, ping_interval=20) as ws:
        for i in range(5):
            msg = json.loads(await ws.recv())
            data = msg.get("data", msg)
            symbol = data["s"]
            price = float(data["p"])
            qty = float(data["q"])
            print(f"Trade {i+1}: {symbol} ${price:,.2f} x {qty:.4f}")
    print("WebSocket works fine.")

asyncio.run(test())