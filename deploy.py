import sys
sys.path.insert(0, '.')  # Ensure the current directory is in the path

import os
from dotenv import load_dotenv
from config import (
    API_KEY, API_SECRET, STRATEGY, LOG_FILE, USE_TESTNET, 
    DEFAULT_INTERVAL, TRADING_PAIRS, MAX_RISK_PER_TRADE,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)
from src.strategies.bot_manager import BotManager
from src.utils.bot_monitor import BotMonitor
import logging
from binance.client import Client
from src.utils.logger import console_message, get_logger
from src.utils.logger_utils import log_bot_status

# Load environment variables from .env file
load_dotenv()

# Ensure these variables are loaded
if not API_KEY or not API_SECRET:
    console_message("❌ Error: API Key and Secret are required.")
    exit(1)

# Suppress all urllib3 logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("discord_webhook").setLevel(logging.WARNING)

# Strategy validation
valid_strategies = ['SMA', 'RSI', 'COMBINED', 'SMALL', 'TRON11']
if STRATEGY not in valid_strategies:
    console_message(f"❌ Error: Unknown strategy: {STRATEGY}. Valid options are {', '.join(valid_strategies)}.")
    exit(1)

# Initialize bot
try:
    console_message(f"🚀 Starting Advanced Trading Bot...")
    
    # Try to connect to Binance with time synchronization
    binance_client = Client(API_KEY, API_SECRET, tld='com')
    
    # Get server time for diagnostics
    server_time = binance_client.get_server_time()
    import time
    from datetime import datetime
    
    # Calculate time difference
    server_timestamp = server_time['serverTime']
    local_timestamp = int(time.time() * 1000)
    time_diff = server_timestamp - local_timestamp
    
    # Show time sync information
    server_time_str = datetime.fromtimestamp(server_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    console_message(f"📡 Connected to Binance API (Time offset: {time_diff} ms)")
    
    # Suppress the account info dump
    account_info = binance_client.get_account()
    
    # Display account balances
    usdt_balance = next((float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'), 0)
    console_message(f"💵 USDT Balance: ${usdt_balance:.2f}")
    
except Exception as e:
    console_message(f"❌ Error connecting to Binance: {str(e)}")
    exit(1)

# Monitor Bot
try:
    monitor = BotMonitor()
except Exception as e:
    console_message(f"❌ Error initializing bot monitor: {str(e)}")
    exit(1)

try:
    console_message(f"⚙️ Initializing advanced trading systems...")
    bot = BotManager()  # Initialize the new BotManager

    console_message(f"✅ Advanced Trading Bot is now running!")
    console_message(f"📊 Trading pairs: {', '.join(TRADING_PAIRS)}")
    console_message(f"💰 Risk per trade: {MAX_RISK_PER_TRADE}% of account")
    console_message(f"🛑 Stop-loss: {STOP_LOSS_PERCENTAGE}% | ⭐ Take-profit: {TAKE_PROFIT_PERCENTAGE}%")
    console_message(f"⏱️ Scanning markets every minute")
    console_message(f"📝 Full logs available in: {LOG_FILE}")
    
    # Log initial bot status
    try:
        log_bot_status(
            status="STARTED",
            account_value=usdt_balance,
            active_pairs=TRADING_PAIRS,
            message="Bot started with advanced trading strategies"
        )
    except Exception as e:
        console_message(f"⚠️ Warning: Could not log initial bot status: {e}")
    
    console_message("\n💰 Bot is trading... You can minimize this window.")
    
    # Run the bot
    bot.run()
    
except KeyboardInterrupt:
    console_message("\n🛑 Bot stopped by user.")
    try:
        log_bot_status(
            status="STOPPED",
            message="Bot stopped by user"
        )
    except:
        pass
except Exception as e:
    console_message(f"❌ Error: {e}")
    try:
        log_bot_status(
            status="ERROR",
            message=f"Bot error: {str(e)}"
        )
    except:
        pass