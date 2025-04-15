"""
Configuration settings for the cryptocurrency trading bot.
Optimized for focused capital allocation and improved profitability.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API credentials
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

# Trading settings
# EXPANDED number of trading pairs for better opportunities
TRADING_PAIRS = [
    "BTCUSDT",  # Major pair - 20% allocation
    "ETHUSDT",  # Major pair - 20% allocation
    "BNBUSDT",  # Altcoin - 20% allocation
    "SOLUSDT",  # Altcoin - 20% allocation
    "AVAXUSDT"  # Altcoin - 20% allocation
]
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"

# File paths
BASE_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "All Bots", "Crypto Bot")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
TRADE_LOGS_DIR = BASE_DIR  # Store trade logs in the main bot directory

# Strategy selection
STRATEGY = "TRON11"  # Options: SMA, RSI, COMBINED, SMALL, TRON11

# Risk management parameters - OPTIMIZED for $500 account
MAX_RISK_PER_TRADE = 2.0  # 2% risk per trade ($10 on $500)
STOP_LOSS_PERCENTAGE = {
    'BTCUSDT': 2.0,  # 2% for BTC
    'ETHUSDT': 2.0,  # 2% for ETH
    'DEFAULT': 1.5   # 1.5% for altcoins
}
TAKE_PROFIT_PERCENTAGE = {
    'BTCUSDT': 3.0,  # 3% for BTC
    'ETHUSDT': 3.0,  # 3% for ETH
    'DEFAULT': 5.0   # 5% for altcoins
}
MAX_TRADES_PER_DAY = 8  # Increased for more opportunities
DAILY_LOSS_LIMIT = 5.0  # 5% daily loss limit ($25 on $500)

# Fee management settings
FEE_RATE = 0.001  # 0.1% standard Binance fee
MIN_FEE_MULTIPLIER = 5  # Increased from 3 to ensure trades are meaningful after fees
USE_BNB_FOR_FEES = True  # Changed to True to reduce fees (make sure you have BNB)
BNB_FEE_DISCOUNT = 0.25  # 25% discount when paying fees with BNB
MAKER_FEE_RATE = 0.0009  # 0.09% for limit orders (maker)
TAKER_FEE_RATE = 0.001   # 0.1% for market orders (taker)

# Profit threshold settings
MIN_PROFIT_MULTIPLIER = 2  # Reduced to 2 for more frequent trades

# SMA Strategy params (faster signals)
SHORT_WINDOW = 7
LONG_WINDOW = 15

# RSI Strategy params (crypto-friendly & aggressive)
RSI_PERIOD = 10
RSI_OVERBOUGHT = 70  # Adjusted to be less aggressive
RSI_OVERSOLD = 30    # Adjusted to be less aggressive

# TRON 1.1 Strategy params - OPTIMIZED for more active trading
TRON_RSI_PERIOD = 14
TRON_RSI_OVERBOUGHT = 65  # Adjusted from 60 
TRON_RSI_OVERSOLD = 35    # Adjusted from 40
TRON_SHORT_MA = 8         # Increased from 5 for better trend confirmation
TRON_LONG_MA = 21
TRON_MACD_SIGNAL = 9      # Increased from 7 for fewer false signals

# Trade intervals - ADDED multiple timeframes
INTERVALS = {
    'SCALP': '1m',    # For scalping
    'SHORT': '5m',    # For short-term trades
    'MEDIUM': '15m',  # For medium-term trades
    'LONG': '1h'      # For trend confirmation
}
DEFAULT_INTERVAL = '5m'  # Default timeframe
STATUS_UPDATE_INTERVAL = 1800  # 30 minutes

# Chart interval for API calls (must use valid Binance interval)
CHART_INTERVAL = "5m"  # Valid Binance interval: 5 minutes

# Minimum order values for each trading pair (in USDT)
# ADJUSTED for $500 account
MIN_ORDER_VALUES = {
   'BTCUSDT': 15.0,  # Reduced minimum for BTC
   'ETHUSDT': 15.0,  # Reduced minimum for ETH
   'BNBUSDT': 10.0,  # Lower minimum for altcoins
   'SOLUSDT': 10.0,
   'AVAXUSDT': 10.0,
   'DEFAULT': 10.0
}

# Calculate and store fee-adjusted minimum values
FEE_ADJUSTED_MIN_VALUES = {
    symbol: max(value, value * FEE_RATE * MIN_FEE_MULTIPLIER) 
    for symbol, value in MIN_ORDER_VALUES.items()
}

# Position sizing based on conviction level
USE_CONVICTION_SIZING = True  # Enable position sizing based on signal strength
MIN_CONVICTION_THRESHOLD = 0.65  # Only take trades with conviction above this value

# CONVICTION MULTIPLIERS for position sizing
CONVICTION_MULTIPLIERS = {
    0.65: 0.5,  # 50% of standard position size for minimum conviction
    0.75: 0.75, # 75% of standard position size for medium conviction
    0.85: 1.0,  # 100% of standard position size for high conviction
    1.0: 1.2    # 120% of standard position size for maximum conviction
}

# Testing mode flag
TESTING_MODE = True  # Set to False in production

# Enhanced logging configuration
BASE_LOG_DIR = os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "."
SCAN_LOGS_DIR = os.path.join(BASE_LOG_DIR, "scans")
TRADE_LOGS_DIR = BASE_DIR  # Store in bot main directory for easier access

# Ensure log directories exist
for directory in [BASE_LOG_DIR, SCAN_LOGS_DIR, TRADE_LOGS_DIR]:
   os.makedirs(directory, exist_ok=True)

# Detailed logging settings
LOG_LEVEL = "INFO"
DETAILED_SCAN_LOGGING = True  # Enable detailed scan logging

# Market condition filters - ADJUSTED for more opportunities
MARKET_VOLATILITY_MIN = 0.15  # Lowered minimum volatility to catch more opportunities
MARKET_VOLATILITY_MAX = 6.0   # Increased maximum volatility for altcoins

# Grid Trading Parameters
USE_GRID_TRADING = True
GRID_LEVELS = 5  # Number of grid levels
GRID_SPACING_PERCENT = 1.0  # 1% between grid levels
MAX_GRID_ORDERS = 3  # Maximum number of open grid orders per pair

# Scalping Parameters
ENABLE_SCALPING = True
SCALP_PROFIT_TARGET = 0.5  # 0.5% profit target for scalping
SCALP_STOP_LOSS = 0.3     # 0.3% stop loss for scalping
MAX_SCALP_TRADES = 3      # Maximum number of concurrent scalp trades