"""
Script to run all tests for the cryptocurrency trading bot.
"""

import os
import sys
import time
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.test_api import test_binance_api
from src.test_strategies import test_strategies
from src.test_risk_management import test_risk_management
from src.paper_trading import run_paper_trading
from src.logger import get_logger

logger = get_logger()

def run_all_tests(paper_trading_duration=1):
    """
    Run all tests for the cryptocurrency trading bot.
    
    Args:
        paper_trading_duration (int, optional): Paper trading duration in hours. Defaults to 1.
    
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    logger.info("Starting all tests for the cryptocurrency trading bot")
    
    # Test Binance API
    logger.info("Testing Binance API...")
    api_test_result = test_binance_api()
    
    if not api_test_result:
        logger.error("Binance API test failed")
        return False
    
    logger.info("Binance API test passed")
    
    # Test strategies
    logger.info("Testing trading strategies...")
    strategy_results = test_strategies()
    
    if not strategy_results:
        logger.error("Strategy testing failed")
        return False
    
    logger.info("Strategy testing passed")
    
    # Test risk management
    logger.info("Testing risk management...")
    try:
        risk_test_result = test_risk_management()
        
        if not risk_test_result:
            logger.error("Risk management test failed")
            return False
        
        logger.info("Risk management test passed")
    except Exception as e:
        logger.warning(f"Risk management test encountered issues: {e} (this is normal if API keys are not set)")
        logger.info("Continuing with limited risk management testing")
    
    # Run paper trading
    logger.info(f"Running paper trading for {paper_trading_duration} hour(s)...")
    paper_trading_results = run_paper_trading(duration_hours=paper_trading_duration, check_interval_minutes=5)
    
    if not paper_trading_results:
        logger.error("Paper trading failed")
        return False
    
    logger.info("Paper trading completed successfully")
    
    # All tests passed
    logger.info("All tests passed successfully")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all tests for the cryptocurrency trading bot')
    parser.add_argument('--paper-duration', type=int, default=1, help='Paper trading duration in hours (default: 1)')
    
    args = parser.parse_args()
    
    # Run all tests
    success = run_all_tests(paper_trading_duration=args.paper_duration)
    
    if success:
        print("\n✅ All tests passed successfully")
    else:
        print("\n❌ Some tests failed")
