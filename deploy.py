import sys
sys.path.insert(0, '.')  # Ensure the current directory is in the path

import os
from dotenv import load_dotenv
from config import API_KEY, API_SECRET, STRATEGY, LOG_FILE, USE_TESTNET
from src.trading_bot import TradingBot
from bot_monitor import BotMonitor
import logging
from binance.client import Client  # Add this import for the Binance Client

# Load environment variables from .env file
load_dotenv()

# Get the API Key and Secret from environment variables

# Ensure these variables are loaded
if not API_KEY or not API_SECRET:
    print("API Key and Secret are required.")
    exit(1)

# Log file path

# Example strategy name if required
print(f"API Key: {API_KEY}")
print(f"API Secret: {API_SECRET}")

# Initialize logger
logger = logging.getLogger('crypto_bot')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Placeholder for the strategy selection
if STRATEGY not in ['SMA', 'RSI', 'COMBINED']:
    logger.error(f"Unknown strategy: {STRATEGY}. Valid options are 'SMA', 'RSI', 'COMBINED'.")
    exit(1)

# Initialize bot
try:
    logger.info("Bot started!")
    binance_client = Client(API_KEY, API_SECRET)
    account_info = binance_client.get_account()
    logger.info(f"Account Info: {account_info}")
except Exception as e:
    logger.error(f"Error connecting to Binance: {str(e)}")
    exit(1)

# Monitor Bot
try:
    monitor = BotMonitor()
except Exception as e:
    logger.error(f"Error initializing bot monitor: {str(e)}")
    exit(1)


try:
    logger.info("Initializing bot...")
    bot = TradingBot(api_key=API_KEY, api_secret=API_SECRET, testnet=USE_TESTNET)

    if bot.initialize():
        logger.info("Running single trading cycle...")
        bot.run_continuous(interval_seconds=900)  # Now running every 15 minutes (900 seconds)
    else:
        logger.error("Bot initialization failed.")
except Exception as e:
    logger.error(f"Fatal error running bot: {e}")
    print(f"❌ Error: {e}")