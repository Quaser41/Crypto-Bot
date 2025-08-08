# test_websocket.py
from data_fetcher import start_coinbase_ws, track_symbol, LIVE_PRICES
import time
import threading

def delayed_subscribe():
    time.sleep(3)  # Give WebSocket time to connect
    track_symbol("ETC")
    track_symbol("ETH")

# Start WebSocket in background
start_coinbase_ws()

# Subscribe after short delay
threading.Thread(target=delayed_subscribe).start()

# Monitor prices
for _ in range(30):
    print(LIVE_PRICES)
    time.sleep(2)
