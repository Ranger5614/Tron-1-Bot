"""
Test script to verify Binance API connection.
"""

import os
import sys
import time

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_api import BinanceAPI
from src.logger import get_logger

logger = get_logger()

def test_binance_api():
    """
    Test Binance API connection and basic functionality.
    """
    logger.info("Starting Binance API test")
    
    # Initialize API client with testnet
    api = BinanceAPI(testnet=True)
    
    # Test server connection
    server_time = api.get_server_time()
    if not server_time:
        logger.error("Failed to connect to Binance server")
        return False
    
    logger.info(f"Connected to Binance server. Server time: {server_time}")
    
    # Test getting ticker price
    btc_price = api.get_ticker_price('BTCUSDT')
    if not btc_price:
        logger.error("Failed to get BTCUSDT price")
        return False
    
    logger.info(f"Current BTC price: ${btc_price}")
    
    # Test getting historical data
    btc_history = api.get_historical_klines('BTCUSDT', '1h', '1 day ago UTC')
    if btc_history is None or len(btc_history) == 0:
        logger.error("Failed to get historical data for BTCUSDT")
        return False
    
    logger.info(f"Retrieved {len(btc_history)} historical klines for BTCUSDT")
    logger.info(f"Last candle: {btc_history.iloc[-1].to_dict()}")
    
    # If we're using testnet, we can test account info if API keys are set
    if api.testnet and api.api_key and api.api_secret:
        try:
            account_info = api.get_account_info()
            if not account_info:
                logger.warning("Failed to get account information")
            else:
                logger.info(f"Account status: {account_info['accountType']}")
                
                # Get balances
                balances = api.get_account_balance()
                if balances:
                    logger.info(f"Account balances: {balances}")
        except Exception as e:
            logger.warning(f"Could not test account info: {e} (this is normal if API keys are not set)")
    else:
        logger.info("Skipping account info test (API keys not set)")
    
    logger.info("Binance API test completed successfully")
    return True

if __name__ == "__main__":
    success = test_binance_api()
    if success:
        print("✅ Binance API test passed")
    else:
        print("❌ Binance API test failed")
