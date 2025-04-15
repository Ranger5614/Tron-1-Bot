#!/usr/bin/env python
"""
HYPERION Trading System - Minimal Terminal Dashboard
A very simple terminal interface with no external dependencies.
"""

import os
import sys
import time
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import the database functions
try:
    from src.utils.database import get_trades, get_latest_status
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Database module not found. Using sample data.")

# Simple color codes (may not work in all terminals)
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the dashboard header"""
    print(f"{BLUE}{BOLD}{'='*50}")
    print(f"HYPERION TRADING SYSTEM")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}{RESET}")

def get_sample_data():
    """Generate sample data if database is not available"""
    # Create sample trades
    trades = []
    pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    actions = ['BUY', 'SELL']
    
    # Generate 10 sample trades
    for i in range(10):
        pair = pairs[i % len(pairs)]
        action = actions[i % len(actions)]
        price = 100 + i * 100
        quantity = 0.1 + i * 0.01
        net_profit = 100 - i * 20 if action == 'SELL' else 0
        
        trades.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pair': pair,
            'action': action,
            'price': price,
            'quantity': quantity,
            'net_profit': net_profit
        })
    
    return trades

def print_metrics(trades):
    """Print trading metrics"""
    if not trades:
        print(f"{YELLOW}No trading data available{RESET}")
        return

    total_trades = len(trades)
    total_pnl = sum(trade.get('net_profit', 0) for trade in trades)
    win_count = sum(1 for trade in trades if trade.get('net_profit', 0) > 0)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    print(f"\n{WHITE}Trading Metrics:")
    print(f"Total Trades: {total_trades}")
    pnl_color = GREEN if total_pnl >= 0 else RED
    print(f"Total PnL: {pnl_color}${total_pnl:.2f}{RESET}")
    print(f"Win Rate: {win_rate:.1f}%")

def print_trades(trades):
    """Print recent trades"""
    if not trades:
        print(f"\n{YELLOW}No trades available{RESET}")
        return

    print(f"\n{WHITE}Recent Trades:")
    print(f"{'Time':<20} {'Pair':<10} {'Action':<6} {'PnL':<10}")
    print(f"{'-'*50}")

    # Get last 5 trades
    recent_trades = trades[-5:] if len(trades) > 5 else trades
    for trade in recent_trades:
        pnl = trade.get('net_profit', 0)
        pnl_color = GREEN if pnl >= 0 else RED
        print(f"{str(trade.get('timestamp', '')):<20} "
              f"{str(trade.get('pair', '')):<10} "
              f"{str(trade.get('action', '')):<6} "
              f"{pnl_color}${pnl:.2f}{RESET}")

def print_status(status):
    """Print bot status"""
    print(f"\n{WHITE}System Status:")
    status_color = GREEN if status.get('status') == 'RUNNING' else RED
    print(f"Bot Status: {status_color}{status.get('status', 'UNKNOWN')}{RESET}")
    print(f"Account Value: ${status.get('account_value', 0):.2f}")

def print_footer():
    """Print the dashboard footer"""
    print(f"\n{BLUE}{'='*50}")
    print(f"HYPERION TRADING SYSTEM v1.0")
    print(f"Press Ctrl+C to exit")
    print(f"{'='*50}{RESET}")

def main():
    """Main function to run the dashboard"""
    try:
        while True:
            clear_screen()
            print_header()
            
            # Get data
            if HAS_DATABASE:
                try:
                    trades_df = get_trades()
                    # Convert DataFrame to list of dictionaries if needed
                    if hasattr(trades_df, 'to_dict'):
                        trades = trades_df.to_dict('records')
                    else:
                        trades = trades_df
                    status = get_latest_status()
                except Exception as e:
                    print(f"{RED}Error accessing database: {str(e)}{RESET}")
                    trades = get_sample_data()
                    status = {'status': 'RUNNING', 'account_value': 10000.00}
            else:
                trades = get_sample_data()
                status = {'status': 'RUNNING', 'account_value': 10000.00}
            
            print_metrics(trades)
            print_trades(trades)
            print_status(status)
            print_footer()
            
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{RED}Shutting down HYPERION Trading System...{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{RED}Error: {str(e)}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main() 