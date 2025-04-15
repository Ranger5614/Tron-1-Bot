"""
Script to run the trading bot.
"""

import os
import sys
from src.trading_bot import TradingBot
from src.utils.logger import get_logger

logger = get_logger()

def main():
    try:
        # Initialize the trading bot
        bot = TradingBot()
        
        # Run the bot continuously
        bot.run_continuous()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise

if __name__ == "__main__":
    main() 