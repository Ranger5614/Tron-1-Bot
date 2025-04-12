"""
Paper trading script for the cryptocurrency trading bot.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_api import BinanceAPI
from src.trading_bot import TradingBot
from src.logger import get_logger
from config import config

logger = get_logger()

def run_paper_trading(duration_hours=24, check_interval_minutes=15):
    """
    Run paper trading for a specified duration.
    
    Args:
        duration_hours (int, optional): Duration in hours. Defaults to 24.
        check_interval_minutes (int, optional): Check interval in minutes. Defaults to 15.
    
    Returns:
        dict: Paper trading results.
    """
    logger.info(f"Starting paper trading for {duration_hours} hours with {check_interval_minutes} minute intervals")
    
    # Initialize trading bot with testnet
    bot = TradingBot(testnet=True)
    
    try:
        if not bot.initialize():
            logger.error("Failed to initialize trading bot")
            return False
    except Exception as e:
        logger.warning(f"Could not fully initialize trading bot: {e} (this is normal if API keys are not set)")
        logger.info("Continuing with limited functionality for demonstration purposes")
    
    # Record start time and initial balance
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    initial_balance = bot.risk_manager.get_account_value()
    
    logger.info(f"Paper trading started at {start_time}")
    logger.info(f"Initial account value: ${initial_balance:.2f}")
    logger.info(f"Paper trading will end at {end_time}")
    
    # Track trades and performance
    trades = []
    check_interval_seconds = check_interval_minutes * 60
    
    try:
        # Run until duration is reached
        while datetime.now() < end_time:
            logger.info(f"Running trading cycle at {datetime.now()}")
            
            # Run one trading cycle
            result = bot.run_once()
            
            # Record trades
            if result.get('status') == 'success' and 'results' in result:
                for symbol, trade_result in result['results'].items():
                    if 'action' in trade_result:
                        trades.append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': trade_result['action'],
                            'price': trade_result['price'],
                            'profit_loss': trade_result.get('profit_loss', 0),
                            'profit_loss_pct': trade_result.get('profit_loss_pct', 0)
                        })
                        
                        logger.info(f"Recorded trade: {symbol} {trade_result['action']} at {trade_result['price']}")
            
            # Check if trading was stopped due to risk management
            if result.get('status') == 'stopped':
                logger.warning("Paper trading stopped due to risk management rules")
                break
            
            # Sleep until next check
            logger.info(f"Sleeping for {check_interval_minutes} minutes")
            time.sleep(check_interval_seconds)
    
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
    
    finally:
        # Calculate final results
        end_time = datetime.now()
        duration = end_time - start_time
        final_balance = bot.risk_manager.get_account_value()
        
        profit_loss = final_balance - initial_balance
        profit_loss_pct = (final_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
        
        # Prepare results
        results = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'num_trades': len(trades),
            'trades': trades
        }
        
        # Print summary
        logger.info("Paper Trading Summary:")
        logger.info(f"  Start time: {start_time}")
        logger.info(f"  End time: {end_time}")
        logger.info(f"  Duration: {duration}")
        logger.info(f"  Initial balance: ${initial_balance:.2f}")
        logger.info(f"  Final balance: ${final_balance:.2f}")
        logger.info(f"  Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        logger.info(f"  Number of trades: {len(trades)}")
        
        # Save results to CSV
        if trades:
            trades_df = pd.DataFrame(trades)
            os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
            trades_file = f"{config.DATA_DIRECTORY}/paper_trading_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved paper trading trades to {trades_file}")
        
        return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run paper trading for the cryptocurrency trading bot')
    parser.add_argument('--duration', type=int, default=24, help='Duration in hours (default: 24)')
    parser.add_argument('--interval', type=int, default=15, help='Check interval in minutes (default: 15)')
    
    args = parser.parse_args()
    
    # Run paper trading
    results = run_paper_trading(duration_hours=args.duration, check_interval_minutes=args.interval)
    
    if results:
        print("\n✅ Paper trading completed successfully")
        print(f"Profit/Loss: ${results['profit_loss']:.2f} ({results['profit_loss_pct']:.2f}%)")
    else:
        print("\n❌ Paper trading failed")
