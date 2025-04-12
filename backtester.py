"""
Backtesting framework for the cryptocurrency trading bot.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_api import BinanceAPI
from src.strategies import get_strategy
from src.logger import get_logger
from config import config

logger = get_logger()

class Backtester:
    """
    Backtesting framework for trading strategies.
    """
    
    def __init__(self, api_client=None, initial_balance=1000.0, commission=0.001):
        """
        Initialize the backtester.
        
        Args:
            api_client (BinanceAPI, optional): Binance API client. Defaults to None.
            initial_balance (float, optional): Initial balance in USDT. Defaults to 1000.0.
            commission (float, optional): Commission rate. Defaults to 0.001 (0.1%).
        """
        self.api = api_client or BinanceAPI()
        self.initial_balance = initial_balance
        self.commission = commission
        
        logger.info(f"Initialized backtester with initial_balance={initial_balance}, commission={commission}")
    
    def run_backtest(self, strategy, symbol, interval, start_date, end_date=None, plot=True):
        """
        Run a backtest for a strategy.
        
        Args:
            strategy (Strategy): Trading strategy.
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_date (str): Start date in format 'YYYY-MM-DD' or '30 days ago UTC'.
            end_date (str, optional): End date. Defaults to None (current time).
            plot (bool, optional): Whether to plot the results. Defaults to True.
        
        Returns:
            dict: Backtest results.
        """
        logger.info(f"Running backtest for {strategy.name} on {symbol} ({interval}) from {start_date} to {end_date or 'now'}")
        
        # Generate signals
        df = strategy.generate_signals(symbol, interval, start_date, end_date)
        if df is None or len(df) == 0:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Initialize portfolio metrics
        df['position'] = 0
        df['cash'] = self.initial_balance
        df['holdings'] = 0.0
        df['portfolio'] = self.initial_balance
        
        # Track trades
        trades = []
        
        # Simulate trading
        position = 0
        cash = self.initial_balance
        holdings = 0.0
        
        for i in range(1, len(df)):
            # Get current and previous signals
            current_signal = df.iloc[i]['signal']
            prev_signal = df.iloc[i-1]['signal']
            
            # Get current price
            price = df.iloc[i]['close']
            
            # Update position based on signal changes
            if current_signal == 'BUY' and prev_signal != 'BUY' and position == 0:
                # Buy with all available cash
                position_size = cash * 0.99  # Keep some cash for fees
                position_size_after_commission = position_size * (1 - self.commission)
                holdings = position_size_after_commission / price
                cash -= position_size
                position = 1
                
                # Record trade
                trade = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'type': 'BUY',
                    'price': price,
                    'quantity': holdings,
                    'value': holdings * price,
                    'commission': position_size * self.commission
                }
                trades.append(trade)
                
                logger.info(f"BUY signal at {df.iloc[i]['timestamp']}: {holdings} {symbol} at {price}")
            
            elif current_signal == 'SELL' and prev_signal != 'SELL' and position == 1:
                # Sell all holdings
                position_size = holdings * price
                position_size_after_commission = position_size * (1 - self.commission)
                cash += position_size_after_commission
                
                # Record trade
                trade = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'type': 'SELL',
                    'price': price,
                    'quantity': holdings,
                    'value': holdings * price,
                    'commission': position_size * self.commission
                }
                trades.append(trade)
                
                logger.info(f"SELL signal at {df.iloc[i]['timestamp']}: {holdings} {symbol} at {price}")
                
                holdings = 0.0
                position = 0
            
            # Update portfolio value
            portfolio = cash + (holdings * price)
            
            # Update dataframe
            df.at[df.index[i], 'position'] = position
            df.at[df.index[i], 'cash'] = cash
            df.at[df.index[i], 'holdings'] = holdings
            df.at[df.index[i], 'portfolio'] = portfolio
        
        # Calculate returns
        df['returns'] = df['portfolio'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Calculate performance metrics
        initial_value = df.iloc[0]['portfolio']
        final_value = df.iloc[-1]['portfolio']
        total_return = (final_value / initial_value) - 1
        
        # Calculate annualized return
        days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0
        
        # Calculate drawdown
        df['drawdown'] = 1 - df['portfolio'] / df['portfolio'].cummax()
        max_drawdown = df['drawdown'].max()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(df) > 1:
            sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()
        else:
            sharpe_ratio = 0
        
        # Prepare results
        results = {
            'strategy': strategy.name,
            'symbol': symbol,
            'interval': interval,
            'start_date': df.iloc[0]['timestamp'],
            'end_date': df.iloc[-1]['timestamp'],
            'initial_balance': self.initial_balance,
            'final_balance': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades),
            'trades': trades,
            'data': df
        }
        
        # Log results
        logger.info(f"Backtest results for {strategy.name} on {symbol}:")
        logger.info(f"  Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"  Final balance: ${final_value:.2f}")
        logger.info(f"  Total return: {total_return:.2%}")
        logger.info(f"  Annualized return: {annualized_return:.2%}")
        logger.info(f"  Max drawdown: {max_drawdown:.2%}")
        logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Number of trades: {len(trades)}")
        
        # Plot results
        if plot:
            self._plot_backtest_results(results)
        
        return results
    
    def _plot_backtest_results(self, results):
        """
        Plot backtest results.
        
        Args:
            results (dict): Backtest results.
        """
        df = results['data']
        
        # Create figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Format dates
        date_format = DateFormatter('%Y-%m-%d')
        
        # Plot price and signals
        ax1.plot(df['timestamp'], df['close'], label='Price')
        ax1.scatter(df[df['signal'] == 'BUY']['timestamp'], df[df['signal'] == 'BUY']['close'], 
                   marker='^', color='g', label='Buy Signal')
        ax1.scatter(df[df['signal'] == 'SELL']['timestamp'], df[df['signal'] == 'SELL']['close'], 
                   marker='v', color='r', label='Sell Signal')
        
        # Plot moving averages if available
        if 'short_ma' in df.columns and 'long_ma' in df.columns:
            ax1.plot(df['timestamp'], df['short_ma'], label=f'Short MA ({results["strategy"].short_window})', alpha=0.7)
            ax1.plot(df['timestamp'], df['long_ma'], label=f'Long MA ({results["strategy"].long_window})', alpha=0.7)
        
        # Plot RSI if available
        if 'rsi' in df.columns:
            ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('RSI')
            ax3.legend()
        
        # Plot portfolio value
        ax2.plot(df['timestamp'], df['portfolio'], label='Portfolio Value', color='blue')
        
        # Set labels and title
        ax1.set_title(f'Backtest Results: {results["strategy"]} on {results["symbol"]} ({results["interval"]})')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        
        ax3.set_xlabel('Date')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Add performance metrics as text
        textstr = '\n'.join((
            f'Initial Balance: ${results["initial_balance"]:.2f}',
            f'Final Balance: ${results["final_balance"]:.2f}',
            f'Total Return: {results["total_return"]:.2%}',
            f'Annualized Return: {results["annualized_return"]:.2%}',
            f'Max Drawdown: {results["max_drawdown"]:.2%}',
            f'Sharpe Ratio: {results["sharpe_ratio"]:.2f}',
            f'Number of Trades: {results["num_trades"]}'
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save plot
        plt.tight_layout()
        
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
        
        # Save plot to file
        filename = f"{config.DATA_DIRECTORY}/backtest_{results['strategy'].replace(' ', '_')}_{results['symbol']}_{results['interval']}.png"
        plt.savefig(filename)
        logger.info(f"Saved backtest plot to {filename}")
        
        plt.close(fig)
