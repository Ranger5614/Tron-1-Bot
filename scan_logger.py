"""
Enhanced logging for cryptocurrency trading bot scans.
Creates structured CSV logs for easier analysis.
"""

import csv
import os
from datetime import datetime
import logging
import config

class ScanLogger:
    """
    A specialized logger for trading bot scans that creates structured, 
    easy-to-read logs in CSV format.
    """
    
    def __init__(self):
        # Use config paths
        self.log_dir = config.SCAN_LOGS_DIR
        
        # Set up date-based logging
        self.today = datetime.utcnow().strftime("%Y-%m-%d")
        self.csv_file = os.path.join(self.log_dir, f"scan_log_{self.today}.csv")
        
        # Get a logger instance
        self.logger = logging.getLogger("trading_bot")
        
        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "pair", "interval", "signal", 
                    "price", "volume", "strategy", "indicators"
                ])
    
    def log_scan(self, pair, interval, signal, price, volume, strategy=None, indicators=None):
        """
        Log a trading bot scan with structured data
        
        Args:
            pair (str): Trading pair (e.g., 'BTCUSDT')
            interval (str): Timeframe interval (e.g., '15m', '1h')
            signal (str): Signal detected (e.g., 'buy', 'sell', 'hold')
            price (float): Current price
            volume (float): Trading volume
            strategy (str): Trading strategy used (e.g., 'RSI', 'SMA', 'COMBINED')
            indicators (dict): Dictionary of indicator values
        """
        # Skip if detailed logging is disabled
        if not getattr(config, 'DETAILED_SCAN_LOGGING', True):
            return
            
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format indicators as a string
        indicators_str = ""
        if indicators and isinstance(indicators, dict):
            # Convert indicators dictionary to a string format
            indicators_str = "; ".join([f"{k}:{v}" for k, v in indicators.items()])
        
        # Use the strategy from config if not provided
        if not strategy:
            strategy = getattr(config, 'STRATEGY', 'UNKNOWN')
            
        # Add emoji based on signal
        signal_icon = "⚪"  # default/hold
        if signal.lower() in ['buy', 'long']:
            signal_icon = "🟢"
        elif signal.lower() in ['sell', 'short']:
            signal_icon = "🔴"
            
        # Log to standard logger (summary only)
        log_msg = (f"SCAN: {pair} ({interval}) - {strategy} - Signal: {signal} - "
                  f"Price: {price:.2f}")
        self.logger.info(f"{signal_icon} {log_msg}")
            
        # Log to CSV file
        try:
            with open(self.csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    pair,
                    interval,
                    signal,
                    round(price, 6),
                    round(volume, 2),
                    strategy,
                    indicators_str
                ])
        except Exception as e:
            self.logger.error(f"Failed to write scan log to CSV: {str(e)}")

# Singleton instance
_scan_logger = None

def get_scan_logger():
    """Get or create the scan logger instance"""
    global _scan_logger
    if _scan_logger is None:
        _scan_logger = ScanLogger()
    return _scan_logger


# Example usage
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create logger instance
    scan_logger = get_scan_logger()
    
    # Log a sample scan
    scan_logger.log_scan(
        pair="BTCUSDT",
        interval="15m",
        signal="buy",
        price=42350.75,
        volume=1250000.45,
        strategy="COMBINED",
        indicators={
            "rsi": 22.5,
            "short_sma": 42100.50,
            "long_sma": 41950.25
        }
    )
    
    print(f"Scan log created at: {scan_logger.csv_file}")