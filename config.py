# In config.py - Add any missing parameters that might be hardcoded elsewhere
"""
Configuration settings for the cryptocurrency trading bot.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API credentials
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

# Trading settings
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]  # Trading pairs to monitor
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"
LOG_FILE = "bot.log"  # Log file path

# Strategy selection
STRATEGY = "COMBINED"  # Options: SMA, RSI, COMBINED

# Risk management parameters
MAX_RISK_PER_TRADE = 8.0  # ~ $10.88 per trade at $136 balance
STOP_LOSS_PERCENTAGE = 2.0
TAKE_PROFIT_PERCENTAGE = 6.0
MAX_TRADES_PER_DAY = 12  # Increased from 5 to 12
DAILY_LOSS_LIMIT = 10.0

# SMA Strategy params (faster signals)
SHORT_WINDOW = 7
LONG_WINDOW = 15

# RSI Strategy params (crypto-friendly & aggressive)
RSI_PERIOD = 10
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25

# Trade intervals (frequent evaluations)
DEFAULT_INTERVAL = 900  # 15 minutes
STATUS_UPDATE_INTERVAL = 1800  # 30 minutes

# Fixed minimums for testing - these were hardcoded elsewhere
MIN_ORDER_VALUES = {
    'BTCUSDT': 15.0,
    'ETHUSDT': 15.0,
    'SOLUSDT': 15.0,
    'XRPUSDT': 15.0,
    'DEFAULT': 15.0
}

# Fixed test sizes - these were hardcoded in your risk_manager.py
TESTING_SIZES = {
    'BTCUSDT': 0.001,  # Minimum for BTC
    'ETHUSDT': 0.01,   # Minimum for ETH
    'SOLUSDT': 0.1,    # Minimum for SOL
    'XRPUSDT': 10.0,   # Minimum for XRP
    'DEFAULT': 0.01    # Default for other pairs
}

# Testing mode flag
TESTING_MODE = True  # Set to False in production