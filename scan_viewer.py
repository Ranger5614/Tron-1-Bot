"""
Utility to view and analyze trading bot scan logs.
"""

import os
import csv
import sys
from datetime import datetime, timedelta
import config

def list_log_files():
    """List all available scan log files"""
    log_files = []
    
    try:
        for filename in os.listdir(config.SCAN_LOGS_DIR):
            if filename.startswith("scan_log_") and filename.endswith(".csv"):
                date_str = filename.replace("scan_log_", "").replace(".csv", "")
                log_files.append(date_str)
    except FileNotFoundError:
        print(f"Log directory not found: {config.SCAN_LOGS_DIR}")
    
    return sorted(log_files)

def load_scan_log(date_str=None):
    """
    Load scan log data for a specific date
    
    Args:
        date_str: Date string in YYYY-MM-DD format or None for today
    
    Returns:
        List of dictionaries containing scan log entries
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    log_file = os.path.join(config.SCAN_LOGS_DIR, f"scan_log_{date_str}.csv")
    
    if not os.path.exists(log_file):
        print(f"No log file found for date: {date_str}")
        return []
    
    data = []
    try:
        with open(log_file, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse indicators
                indicators = {}
                if "indicators" in row and row["indicators"]:
                    for item in row["indicators"].split(";"):
                        if ":" in item:
                            key, value = item.split(":", 1)
                            try:
                                indicators[key.strip()] = float(value.strip())
                            except ValueError:
                                indicators[key.strip()] = value.strip()
                
                # Add parsed row to data
                entry = {
                    "timestamp": row["timestamp"],
                    "pair": row["pair"],
                    "interval": row["interval"],
                    "signal": row["signal"],
                    "price": float(row["price"]) if row["price"] else 0.0,
                    "volume": float(row["volume"]) if row["volume"] else 0.0,
                    "strategy": row.get("strategy", ""),
                    "indicators": indicators
                }
                data.append(entry)
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
    
    return data

def summarize_scans(scans):
    """Generate a summary of scan data"""
    if not scans:
        return {
            "total_scans": 0,
            "signals": {},
            "pairs": {},
            "strategies": {}
        }
    
    # Initialize counters
    signals = {}
    pairs = {}
    strategies = {}
    
    # Count signals, pairs, and strategies
    for scan in scans:
        signal = scan["signal"].lower()
        pair = scan["pair"]
        strategy = scan["strategy"]
        
        # Count signals
        signals[signal] = signals.get(signal, 0) + 1
        
        # Count pairs
        if pair not in pairs:
            pairs[pair] = {"total": 0, "buy": 0, "sell": 0, "hold": 0}
        pairs[pair]["total"] += 1
        signal_type = signal if signal in ["buy", "sell"] else "hold"
        pairs[pair][signal_type] += 1
        
        # Count strategies
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    return {
        "total_scans": len(scans),
        "signals": signals,
        "pairs": pairs,
        "strategies": strategies
    }

def print_summary(summary):
    """Print a readable summary of scan data"""
    print("\n===== SCAN LOG SUMMARY =====")
    print(f"Total scans: {summary['total_scans']}")
    
    print("\nSignals:")
    for signal, count in summary["signals"].items():
        percentage = (count / summary["total_scans"] * 100) if summary["total_scans"] > 0 else 0
        print(f"  {signal}: {count} ({percentage:.2f}%)")
    
    print("\nTrading pairs:")
    for pair, data in summary["pairs"].items():
        buy_pct = (data["buy"] / data["total"] * 100) if data["total"] > 0 else 0
        sell_pct = (data["sell"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  {pair}: {data['total']} scans")
        print(f"    Buy: {data['buy']} ({buy_pct:.2f}%), Sell: {data['sell']} ({sell_pct:.2f}%)")
    
    print("\nStrategies:")
    for strategy, count in summary["strategies"].items():
        percentage = (count / summary["total_scans"] * 100) if summary["total_scans"] > 0 else 0
        print(f"  {strategy}: {count} ({percentage:.2f}%)")
    
    print("=============================\n")

def filter_scans(scans, pair=None, signal=None, strategy=None):
    """Filter scans based on criteria"""
    filtered = scans.copy()
    
    if pair:
        filtered = [scan for scan in filtered if scan["pair"] == pair]
    
    if signal:
        filtered = [scan for scan in filtered if scan["signal"].lower() == signal.lower()]
    
    if strategy:
        filtered = [scan for scan in filtered if scan["strategy"] == strategy]
    
    return filtered

def print_scans(scans, limit=20):
    """Print scan details in a readable format"""
    if not scans:
        print("No scan data to display.")
        return
    
    # Limit the number of scans to display
    display_scans = scans[-limit:] if limit else scans
    
    print("\n===== SCAN DETAILS =====")
    for scan in display_scans:
        signal_icon = "⚪"  # default/hold
        if scan["signal"].lower() == "buy":
            signal_icon = "🟢"
        elif scan["signal"].lower() == "sell":
            signal_icon = "🔴"
        
        print(f"{signal_icon} [{scan['timestamp']}] {scan['pair']} ({scan['interval']})")
        print(f"  Signal: {scan['signal']}, Price: {scan['price']:.2f}, Strategy: {scan['strategy']}")
        
        if scan["indicators"]:
            print("  Indicators:")
            for key, value in scan["indicators"].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")
        print("")
    
    if limit and len(scans) > limit:
        print(f"Showing {limit} of {len(scans)} scans. Use --limit 0 to see all.")
    
    print("========================\n")

def print_help():
    """Print usage information"""
    print("\nTrading Bot Scan Log Viewer")
    print("==========================")
    print("\nUsage:")
    print("  python scan_viewer.py [options]")
    print("\nOptions:")
    print("  --date YYYY-MM-DD    Show scans for a specific date (default: today)")
    print("  --pair SYMBOL        Filter by trading pair (e.g., BTCUSDT)")
    print("  --signal TYPE        Filter by signal type (buy, sell, hold)")
    print("  --strategy NAME      Filter by strategy (e.g., RSI, SMA, COMBINED)")
    print("  --summary            Show only the summary, not individual scans")
    print("  --list               List available log dates")
    print("  --limit N            Limit output to N scans (default: 20, 0 for all)")
    print("  --help               Show this help message")
    print("\nExamples:")
    print("  python scan_viewer.py --list")
    print("  python scan_viewer.py --date 2023-04-12")
    print("  python scan_viewer.py --pair BTCUSDT --signal buy")
    print("  python scan_viewer.py --strategy COMBINED --summary")
    print("")

def main():
    """Main function to run the scan viewer"""
    args = sys.argv[1:]
    
    # Parse command line arguments
    date_str = None
    pair = None
    signal = None
    strategy = None
    summary_only = False
    list_dates = False
    limit = 20
    
    i = 0
    while i < len(args):
        if args[i] == "--date" and i+1 < len(args):
            date_str = args[i+1]
            i += 2
        elif args[i] == "--pair" and i+1 < len(args):
            pair = args[i+1]
            i += 2
        elif args[i] == "--signal" and i+1 < len(args):
            signal = args[i+1]
            i += 2
        elif args[i] == "--strategy" and i+1 < len(args):
            strategy = args[i+1]
            i += 2
        elif args[i] == "--limit" and i+1 < len(args):
            try:
                limit = int(args[i+1])
            except ValueError:
                print(f"Invalid limit value: {args[i+1]}")
            i += 2
        elif args[i] == "--summary":
            summary_only = True
            i += 1
        elif args[i] == "--list":
            list_dates = True
            i += 1
        elif args[i] == "--help":
            print_help()
            return
        else:
            print(f"Unknown option: {args[i]}")
            print_help()
            return
    
    # List available dates
    if list_dates:
        dates = list_log_files()
        if dates:
            print("\nAvailable log dates:")
            for date in dates:
                print(f"  {date}")
        else:
            print("\nNo log files found.")
        return
    
    # Load scan data
    scans = load_scan_log(date_str)
    
    if not scans:
        return
    
    # Filter scans if needed
    if pair or signal or strategy:
        scans = filter_scans(scans, pair, signal, strategy)
    
    # Generate and print summary
    summary = summarize_scans(scans)
    print_summary(summary)
    
    # Print scan details unless summary only
    if not summary_only:
        print_scans(scans, limit)

if __name__ == "__main__":
    main()