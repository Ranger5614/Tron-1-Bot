"""
Bot manager to coordinate between different trading strategies.
Handles strategy selection, position management, and risk control.
"""

import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional

from src.api.binance_api import BinanceAPI
from src.strategies.advanced_strategies import GridStrategy, ScalpingStrategy, AdvancedTRONStrategy
from src.utils.logger import get_logger
from src.utils.logger_utils import send_trade_notification
import config

logger = get_logger()

class BotManager:
    """
    Manages multiple trading strategies and coordinates their execution.
    """
    
    def __init__(self):
        self.api = BinanceAPI()
        self.strategies = {
            'GRID': GridStrategy(self.api),
            'SCALP': ScalpingStrategy(self.api),
            'TRON': AdvancedTRONStrategy(self.api)
        }
        self.active_trades = {}  # Track active trades
        self.daily_stats = {
            'trades': 0,
            'profit_loss': 0.0,
            'win_rate': 0.0
        }
        
    def select_strategy(self, symbol: str, market_regime: str) -> str:
        """
        Select the most appropriate strategy based on market conditions.
        """
        if market_regime == 'RANGING':
            return 'GRID'
        elif market_regime == 'VOLATILE':
            return 'SCALP'
        else:
            return 'TRON'
            
    def check_risk_limits(self) -> bool:
        """
        Check if we're within risk limits.
        """
        # Check daily loss limit
        if self.daily_stats['profit_loss'] < -config.DAILY_LOSS_LIMIT:
            logger.warning("Daily loss limit reached. Stopping trading.")
            return False
            
        # Check number of trades
        if self.daily_stats['trades'] >= config.MAX_TRADES_PER_DAY:
            logger.warning("Maximum daily trades reached.")
            return False
            
        return True
        
    def calculate_position_size(self, symbol: str, strategy: str, 
                              conviction: float = 1.0) -> float:
        """
        Calculate position size based on strategy and risk parameters.
        """
        # Get base position size from strategy
        if strategy == 'TRON':
            base_size = conviction * config.MAX_RISK_PER_TRADE
        elif strategy == 'GRID':
            base_size = config.MAX_RISK_PER_TRADE * 0.5  # Reduced risk for grid
        else:  # SCALP
            base_size = config.MAX_RISK_PER_TRADE * 0.3  # Minimal risk for scalping
            
        # Adjust for symbol-specific limits
        min_value = config.MIN_ORDER_VALUES.get(symbol, config.MIN_ORDER_VALUES['DEFAULT'])
        
        # Get account balance
        balance = float(self.api.get_asset_balance('USDT')['free'])
        
        # Calculate maximum position size
        max_position = balance * (base_size / 100)
        
        # Ensure position size meets minimum requirements
        position_size = max(min_value, max_position)
        
        return position_size
        
    def execute_trade(self, symbol: str, signal: str, strategy: str, 
                     position_size: float) -> bool:
        """
        Execute a trade based on the signal.
        """
        try:
            if signal == 'BUY':
                # Place buy order
                order = self.api.create_market_buy_order(
                    symbol=symbol,
                    quantity=position_size
                )
                
                if order:
                    # Set stop loss and take profit
                    entry_price = float(order['price'])
                    stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENTAGE[symbol] / 100)
                    take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENTAGE[symbol] / 100)
                    
                    # Place stop loss and take profit orders
                    self.api.create_stop_loss_order(symbol, stop_loss)
                    self.api.create_take_profit_order(symbol, take_profit)
                    
                    # Track the trade
                    self.active_trades[order['orderId']] = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'strategy': strategy,
                        'timestamp': datetime.now()
                    }
                    
                    # Update stats
                    self.daily_stats['trades'] += 1
                    
                    # Send notification
                    send_trade_notification(
                        f"BUY {symbol}",
                        f"Strategy: {strategy}\nSize: {position_size}\nPrice: {entry_price}"
                    )
                    
                    return True
                    
            elif signal == 'SELL':
                # Close any open positions for this symbol
                for order_id, trade in self.active_trades.items():
                    if trade['symbol'] == symbol:
                        # Place sell order
                        order = self.api.create_market_sell_order(
                            symbol=symbol,
                            quantity=trade['position_size']
                        )
                        
                        if order:
                            # Calculate profit/loss
                            exit_price = float(order['price'])
                            pnl = (exit_price - trade['entry_price']) * trade['position_size']
                            
                            # Update stats
                            self.daily_stats['profit_loss'] += pnl
                            
                            # Remove from active trades
                            del self.active_trades[order_id]
                            
                            # Send notification
                            send_trade_notification(
                                f"SELL {symbol}",
                                f"Strategy: {strategy}\nPnL: {pnl:.2f} USDT"
                            )
                            
                            return True
                            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
            
    def run(self):
        """
        Main bot loop.
        """
        logger.info("Starting bot manager...")
        
        while True:
            try:
                # Check risk limits
                if not self.check_risk_limits():
                    logger.info("Risk limits reached. Waiting for reset...")
                    time.sleep(3600)  # Wait an hour
                    continue
                    
                # Process each trading pair
                for symbol in config.TRADING_PAIRS:
                    # Get market data
                    df = self.api.get_historical_klines(
                        symbol, 
                        config.DEFAULT_INTERVAL,
                        "1 day ago UTC"
                    )
                    
                    if df is None or len(df) == 0:
                        continue
                        
                    # Detect market regime
                    strategy = self.strategies['TRON']
                    regime = strategy.detect_market_regime(df)
                    
                    # Select appropriate strategy
                    strategy_name = self.select_strategy(symbol, regime)
                    strategy = self.strategies[strategy_name]
                    
                    # Generate signals
                    signals = strategy.generate_signals(
                        symbol,
                        config.DEFAULT_INTERVAL,
                        "1 day ago UTC"
                    )
                    
                    if signals is None or len(signals) == 0:
                        continue
                        
                    # Get latest signal
                    latest = signals.iloc[-1]
                    signal = latest['signal']
                    
                    # Calculate position size
                    conviction = latest.get('conviction', 1.0)
                    position_size = self.calculate_position_size(
                        symbol,
                        strategy_name,
                        conviction
                    )
                    
                    # Execute trade if signal is not HOLD
                    if signal != 'HOLD':
                        self.execute_trade(symbol, signal, strategy_name, position_size)
                        
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
                
if __name__ == "__main__":
    bot = BotManager()
    bot.run() 