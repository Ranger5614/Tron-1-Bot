
import csv
from datetime import datetime
import os

def log_trade_to_csv(pair, action, price, quantity, pnl=None, pnl_pct=None, filename="trade_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "pair", "action", "price", "quantity", "pnl", "pnl_pct"])
        writer.writerow([
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            pair,
            action,
            round(price, 6),
            round(quantity, 6),
            round(pnl, 6) if pnl is not None else "",
            round(pnl_pct, 4) if pnl_pct is not None else ""
        ])
