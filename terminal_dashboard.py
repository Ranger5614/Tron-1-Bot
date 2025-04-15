#!/usr/bin/env python
"""
HYPERION Trading System - Simple Terminal Dashboard
"""

import os
import sys
import time
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.utils.database import get_trades, get_latest_status

# ANSI color codes
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print(f"{BLUE}{BOLD}{'='*50}")
    print(f"HYPERION TRADING SYSTEM")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}{RESET}")

def print_metrics(trades_df):
    if trades_df.empty:
        print(f"{YELLOW}No trading data available{RESET}")
        return

    total_trades = len(trades_df)
    total_pnl = trades_df['net_profit'].sum() if 'net_profit' in trades_df.columns else 0
    win_rate = (trades_df['net_profit'] > 0).mean() * 100 if 'net_profit' in trades_df.columns else 0

    print(f"\n{WHITE}Trading Metrics:")
    print(f"Total Trades: {total_trades}")
    pnl_color = GREEN if total_pnl >= 0 else RED
    print(f"Total PnL: {pnl_color}${total_pnl:.2f}{RESET}")
    print(f"Win Rate: {win_rate:.1f}%")

def print_trades(trades_df):
    if trades_df.empty:
        print(f"\n{YELLOW}No trades available{RESET}")
        return

    print(f"\n{WHITE}Recent Trades:")
    print(f"{'Time':<20} {'Pair':<10} {'Action':<6} {'PnL':<10}")
    print(f"{'-'*50}")

    # Get last 5 trades
    recent_trades = trades_df.tail(5)
    for _, trade in recent_trades.iterrows():
        pnl = trade.get('net_profit', 0)
        pnl_color = GREEN if pnl >= 0 else RED
        print(f"{str(trade.get('timestamp', '')):<20} "
              f"{str(trade.get('pair', '')):<10} "
              f"{str(trade.get('action', '')):<6} "
              f"{pnl_color}${pnl:.2f}{RESET}")

def print_status(status):
    print(f"\n{WHITE}System Status:")
    status_color = GREEN if status.get('status') == 'RUNNING' else RED
    print(f"Bot Status: {status_color}{status.get('status', 'UNKNOWN')}{RESET}")
    print(f"Account Value: ${status.get('account_value', 0):.2f}")

def print_footer():
    print(f"\n{BLUE}{'='*50}")
    print(f"HYPERION TRADING SYSTEM v1.0")
    print(f"Press Ctrl+C to exit")
    print(f"{'='*50}{RESET}")

def main():
    try:
        while True:
            clear_screen()
            print_header()
            
            # Get latest data
            trades_df = get_trades()
            status = get_latest_status()
            
            print_metrics(trades_df)
            print_trades(trades_df)
            print_status(status)
            print_footer()
            
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{RED}Shutting down HYPERION Trading System...{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main() 