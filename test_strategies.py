"""
Test script to evaluate trading strategies with historical data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_api import BinanceAPI
from src.strategies import get_strategy
from src.backtester import Backtester
from src.logger import get_logger
from config import config

logger = get_logger()

def test_strategies():
    """
    Test and compare different trading strategies with historical data.
    """
    logger.info("Starting strategy testing")
    
    # Initialize API client
    api = BinanceAPI(testnet=True)
    
    # Initialize backtester
    backtester = Backtester(api_client=api, initial_balance=1000.0)
    
    # Define test parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '30 days ago UTC'
    
    # Test strategies
    strategies = ['SMA', 'RSI', 'COMBINED']
    results = {}
    
    for strategy_name in strategies:
        logger.info(f"Testing {strategy_name} strategy")
        
        # Get strategy
        strategy = get_strategy(strategy_name, api)
        
        # Run backtest
        result = backtester.run_backtest(
            strategy=strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            plot=True
        )
        
        if result:
            results[strategy_name] = result
    
    # Compare strategies
    if results:
        compare_strategies(results)
    
    return results

def compare_strategies(results):
    """
    Compare performance of different strategies.
    
    Args:
        results (dict): Dictionary of backtest results.
    """
    # Create comparison table
    comparison = pd.DataFrame({
        'Strategy': [],
        'Total Return': [],
        'Annualized Return': [],
        'Max Drawdown': [],
        'Sharpe Ratio': [],
        'Number of Trades': []
    })
    
    for strategy_name, result in results.items():
        comparison = pd.concat([comparison, pd.DataFrame({
            'Strategy': [strategy_name],
            'Total Return': [f"{result['total_return']:.2%}"],
            'Annualized Return': [f"{result['annualized_return']:.2%}"],
            'Max Drawdown': [f"{result['max_drawdown']:.2%}"],
            'Sharpe Ratio': [f"{result['sharpe_ratio']:.2f}"],
            'Number of Trades': [result['num_trades']]
        })], ignore_index=True)
    
    # Log comparison table
    logger.info("Strategy Comparison:")
    logger.info("\n" + comparison.to_string(index=False))
    
    # Save comparison to CSV
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
    comparison_file = f"{config.DATA_DIRECTORY}/strategy_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    logger.info(f"Saved strategy comparison to {comparison_file}")
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    
    for strategy_name, result in results.items():
        df = result['data']
        plt.plot(df['timestamp'], df['portfolio'], label=strategy_name)
    
    plt.title('Strategy Comparison: Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    comparison_plot = f"{config.DATA_DIRECTORY}/strategy_comparison.png"
    plt.savefig(comparison_plot)
    logger.info(f"Saved strategy comparison plot to {comparison_plot}")
    plt.close()

if __name__ == "__main__":
    results = test_strategies()
    
    # Determine best strategy based on Sharpe ratio
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        print(f"\n✅ Strategy testing completed. Best strategy: {best_strategy}")
    else:
        print("\n❌ Strategy testing failed")
