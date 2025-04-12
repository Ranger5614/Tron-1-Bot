"""
Test script to verify risk management functionality.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_api import BinanceAPI
from src.risk_manager import RiskManager
from src.logger import get_logger
from config import config

logger = get_logger()

def test_risk_management():
    """
    Test risk management functionality.
    """
    logger.info("Starting risk management test")
    
    # Initialize API client with testnet
    api = BinanceAPI(testnet=True)
    
    # Initialize risk manager
    risk_manager = RiskManager(api_client=api)
    
    # Test position sizing
    symbol = 'BTCUSDT'
    entry_price = 50000.0  # Example entry price
    stop_loss_price = 49000.0  # Example stop loss price (2% below entry)
    
    logger.info(f"Testing position sizing for {symbol} with entry price {entry_price} and stop loss {stop_loss_price}")
    
    position_size = risk_manager.calculate_position_size(symbol, entry_price, stop_loss_price)
    logger.info(f"Calculated position size: {position_size} {symbol} (value: ${position_size * entry_price:.2f})")
    
    # Test stop loss calculation
    stop_loss = risk_manager.calculate_stop_loss(entry_price, 'BUY')
    logger.info(f"Calculated stop loss for BUY at {entry_price}: {stop_loss}")
    
    # Test take profit calculation
    take_profit = risk_manager.calculate_take_profit(entry_price, 'BUY')
    logger.info(f"Calculated take profit for BUY at {entry_price}: {take_profit}")
    
    # Test daily limits
    can_trade = risk_manager.can_trade_today()
    logger.info(f"Can trade today: {can_trade}")
    
    # Test recording trades
    risk_manager.record_trade()
    logger.info(f"Recorded trade, trades today: {risk_manager.trades_today}")
    
    # Test maximum drawdown
    initial_balance = 1000.0
    current_balance = 900.0  # 10% drawdown
    max_drawdown_exceeded = risk_manager.check_max_drawdown(initial_balance, current_balance, 15.0)
    logger.info(f"Max drawdown exceeded (10% < 15%): {max_drawdown_exceeded}")
    
    max_drawdown_exceeded = risk_manager.check_max_drawdown(initial_balance, current_balance, 5.0)
    logger.info(f"Max drawdown exceeded (10% > 5%): {max_drawdown_exceeded}")
    
    logger.info("Risk management test completed")
    return True

if __name__ == "__main__":
    success = test_risk_management()
    if success:
        print("✅ Risk management test passed")
    else:
        print("❌ Risk management test failed")
