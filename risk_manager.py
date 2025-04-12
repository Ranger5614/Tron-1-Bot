"""
Risk management module for the cryptocurrency trading bot.
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api import BinanceAPI
from logger import get_logger
import config

logger = get_logger()

class RiskManager:
    """
    Risk management for trading operations.
    """
    
    def __init__(self, api_client=None, 
                 max_risk_per_trade=None, 
                 stop_loss_pct=None, 
                 take_profit_pct=None,
                 max_trades_per_day=None,
                 daily_loss_limit_pct=None):
        """
        Initialize the risk manager.
        
        Args:
            api_client (BinanceAPI, optional): Binance API client. Defaults to None.
            max_risk_per_trade (float, optional): Maximum risk per trade as percentage of account. Defaults to config value.
            stop_loss_pct (float, optional): Stop loss percentage. Defaults to config value.
            take_profit_pct (float, optional): Take profit percentage. Defaults to config value.
            max_trades_per_day (int, optional): Maximum number of trades per day. Defaults to config value.
            daily_loss_limit_pct (float, optional): Daily loss limit as percentage of account. Defaults to config value.
        """
        self.api = api_client or BinanceAPI()
        
        # ENHANCED: Default values if not provided in config
        self.max_risk_per_trade = max_risk_per_trade or getattr(config, 'MAX_RISK_PER_TRADE', 2.0)  # Increased from 1%
        self.stop_loss_pct = stop_loss_pct or getattr(config, 'STOP_LOSS_PERCENTAGE', 5.0)  # Larger stop loss
        self.take_profit_pct = take_profit_pct or getattr(config, 'TAKE_PROFIT_PERCENTAGE', 10.0)  # Larger take profit
        self.max_trades_per_day = max_trades_per_day or getattr(config, 'MAX_TRADES_PER_DAY', 10)  # More trades allowed
        self.daily_loss_limit_pct = daily_loss_limit_pct or getattr(config, 'DAILY_LOSS_LIMIT', 5.0)  # Typical value
        
        # Initialize trade tracking
        self.trades_today = 0
        self.daily_loss = 0.0
        self.initial_daily_balance = 0.0
        self.last_day_reset = datetime.now().date()
        
        logger.info(f"Initialized risk manager with max_risk_per_trade={self.max_risk_per_trade}%, "
                   f"stop_loss_pct={self.stop_loss_pct}%, take_profit_pct={self.take_profit_pct}%, "
                   f"max_trades_per_day={self.max_trades_per_day}, daily_loss_limit_pct={self.daily_loss_limit_pct}%")
    
    def reset_daily_limits(self):
        """
        Reset daily trading limits.
        """
        today = datetime.now().date()
        
        if today > self.last_day_reset:
            self.trades_today = 0
            self.daily_loss = 0.0
            self.last_day_reset = today
            
            # Get current account balance
            balance = self.get_account_value()
            if balance > 0:
                self.initial_daily_balance = balance
            
            logger.info(f"Reset daily limits. Initial balance: ${self.initial_daily_balance:.2f}")
    
    def get_account_value(self):
        """
        Get total account value in USDT.
        
        Returns:
            float: Account value in USDT.
        """
        try:
            # Get account balances
            balances = self.api.get_account_balance()
            
            if not balances:
                logger.error("Failed to get account balances")
                return 0.0
            
            # Calculate total value in USDT
            total_value = 0.0
            
            for asset, amount in balances.items():
                if amount <= 0:
                    continue
                
                if asset == 'USDT':
                    total_value += amount
                else:
                    # Get asset price in USDT
                    symbol = f"{asset}USDT"
                    price = self.api.get_ticker_price(symbol)
                    
                    if price:
                        asset_value = amount * price
                        total_value += asset_value
            
            logger.info(f"Total account value: ${total_value:.2f}")
            return total_value
        
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price):
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            entry_price (float): Entry price.
            stop_loss_price (float): Stop loss price.
        
        Returns:
            float: Position size in base currency.
        """
        try:
            # Reset daily limits if needed
            self.reset_daily_limits()
            
            # Get account value
            account_value = self.get_account_value()
            
            if account_value <= 0:
                logger.error("Account value is zero or negative")
                return 0.0
            
            # Calculate risk amount
            risk_amount = account_value * (self.max_risk_per_trade / 100.0)
            logger.info(f"Risk amount based on account value: ${risk_amount:.2f}")
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                logger.error("Risk per unit is zero or negative")
                return 0.0
            
            # Calculate position size in quote currency
            position_size_quote = risk_amount / risk_per_unit
            
            # Convert to base currency
            position_size_base = position_size_quote / entry_price
            logger.info(f"Position size based on risk calculation: {position_size_base} {symbol[:3]} (${position_size_base * entry_price:.2f})")
            
            # ENHANCED: Set minimum order sizes by trading pair
            # These are more aggressive minimums to ensure orders execute
            min_order_values = {
                'BTCUSDT': 15.0,  # $15 minimum for BTC
                'ETHUSDT': 15.0,  # $15 minimum for ETH
                'SOLUSDT': 15.0,  # $15 minimum for SOL
                'XRPUSDT': 15.0,  # $15 minimum for XRP
                'DEFAULT': 15.0    # Default minimum for other pairs
            }
            
            # Get minimum order value for this symbol
            min_order_value = min_order_values.get(symbol, min_order_values['DEFAULT'])
            
            # Calculate minimum position size based on minimum order value
            min_position_base = min_order_value / entry_price
            logger.info(f"Minimum position size for {symbol}: {min_position_base} (${min_order_value:.2f})")
            
            # Check if calculated position size is too small
            if position_size_base < min_position_base:
                logger.warning(f"Risk-based position size {position_size_base} is below minimum. Using minimum size: {min_position_base}")
                position_size_base = min_position_base
            
            # TESTING MODE: Force a minimum practical position size to ensure execution
            # Uncomment this for testing to force trades to execute
            testing_mode = True  # Set to False in production
            if testing_mode:
                # Force minimum sizes that are guaranteed to work on Binance
                testing_sizes = {
                    'BTCUSDT': 0.001,  # Minimum for BTC
                    'ETHUSDT': 0.01,   # Minimum for ETH
                    'SOLUSDT': 0.1,    # Minimum for SOL
                    'XRPUSDT': 10.0,   # Minimum for XRP
                    'DEFAULT': 0.01     # Default for other pairs
                }
                test_size = testing_sizes.get(symbol, testing_sizes['DEFAULT'])
                logger.info(f"TESTING MODE: Forcing minimum position size to {test_size} for {symbol}")
                position_size_base = test_size
            
            # Get symbol info for minimum quantity and step size
            symbol_info = self.api.get_symbol_info(symbol)
            
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
            
            # Find the LOT_SIZE filter
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                min_qty = float(lot_size_filter['minQty'])
                step_size = float(lot_size_filter['stepSize'])
                
                logger.info(f"Exchange min quantity: {min_qty}, step size: {step_size}")
                
                # Round down to step size
                original_size = position_size_base
                position_size_base = self._round_step_size(position_size_base, step_size)
                
                # Log if rounding changed the value
                if original_size != position_size_base:
                    logger.info(f"Rounded position size to match step size: {position_size_base}")
                
                # Ensure minimum quantity
                if position_size_base < min_qty:
                    logger.info(f"Position size {position_size_base} below exchange minimum {min_qty}, using exchange minimum")
                    position_size_base = min_qty
            
            # Check minimum notional value (MIN_NOTIONAL filter)
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if min_notional_filter:
                min_notional = float(min_notional_filter.get('minNotional', 0))
                order_value = position_size_base * entry_price
                
                logger.info(f"Order value: ${order_value:.2f}, Exchange minimum notional: ${min_notional:.2f}")
                
                if order_value < min_notional:
                    # Recalculate position size to meet MIN_NOTIONAL
                    needed_position = min_notional / entry_price
                    # Round to step size
                    if lot_size_filter:
                        needed_position = self._round_step_size(needed_position, step_size)
                    
                    logger.warning(f"Order value ${order_value:.2f} below exchange minimum ${min_notional:.2f}. Increasing to {needed_position}")
                    position_size_base = needed_position
            
            # Final check that position size is valid
            if position_size_base <= 0:
                logger.error("Final position size is zero or negative, cannot execute trade")
                return 0.0
            
            # Log final position size
            final_value = position_size_base * entry_price
            logger.info(f"Final position size for {symbol}: {position_size_base} (value: ${final_value:.2f})")
            
            return position_size_base
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _round_step_size(self, quantity, step_size):
        """
        Round quantity down to step size.
        
        Args:
            quantity (float): Quantity to round.
            step_size (float): Step size.
        
        Returns:
            float: Rounded quantity.
        """
        precision = int(round(-math.log10(step_size)))
        return math.floor(quantity * 10**precision) / 10**precision
    
    def calculate_stop_loss(self, entry_price, side):
        """
        Calculate stop loss price.
        
        Args:
            entry_price (float): Entry price.
            side (str): Order side ('BUY' or 'SELL').
        
        Returns:
            float: Stop loss price.
        """
        if side == 'BUY':
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100.0)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100.0)
        
        logger.info(f"Calculated stop loss for {side} at {entry_price}: {stop_loss} ({self.stop_loss_pct}%)")
        return stop_loss
    
    def calculate_take_profit(self, entry_price, side):
        """
        Calculate take profit price.
        
        Args:
            entry_price (float): Entry price.
            side (str): Order side ('BUY' or 'SELL').
        
        Returns:
            float: Take profit price.
        """
        if side == 'BUY':
            take_profit = entry_price * (1 + self.take_profit_pct / 100.0)
        else:
            take_profit = entry_price * (1 - self.take_profit_pct / 100.0)
        
        logger.info(f"Calculated take profit for {side} at {entry_price}: {take_profit} ({self.take_profit_pct}%)")
        return take_profit
    
    def can_trade_today(self):
        """
        Check if trading is allowed today based on daily limits.
        
        Returns:
            bool: True if trading is allowed, False otherwise.
        """
        # Reset daily limits if needed
        self.reset_daily_limits()
        
        # Check number of trades
        if self.trades_today >= self.max_trades_per_day:
            logger.warning(f"Maximum number of trades per day reached ({self.max_trades_per_day})")
            return False
        
        # Check daily loss limit
        if self.initial_daily_balance > 0:
            current_balance = self.get_account_value()
            daily_loss_pct = (self.initial_daily_balance - current_balance) / self.initial_daily_balance * 100.0
            
            if daily_loss_pct >= self.daily_loss_limit_pct:
                logger.warning(f"Daily loss limit reached ({daily_loss_pct:.2f}% > {self.daily_loss_limit_pct}%)")
                return False
        
        return True
    
    def record_trade(self, is_profitable=None):
        """
        Record a trade for daily limits tracking.
        
        Args:
            is_profitable (bool, optional): Whether the trade was profitable. Defaults to None.
        """
        self.trades_today += 1
        logger.info(f"Recorded trade #{self.trades_today} for today")
    
    def place_order_with_risk_management(self, symbol, side, entry_price=None):
        """
        Place an order with risk management.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            side (str): Order side ('BUY' or 'SELL').
            entry_price (float, optional): Entry price. Defaults to current market price.
        
        Returns:
            dict: Order information including main order and stop loss/take profit orders.
        """
        try:
            logger.info(f"Starting order placement for {symbol} {side} with risk management")
            
            # Check if trading is allowed today
            logger.info("Checking if trading is allowed today...")
            if not self.can_trade_today():
                logger.warning("Trading not allowed today due to daily limits")
                return None
            logger.info("Trading is allowed today")
            
            # Get current price if entry price not provided
            if not entry_price:
                logger.info(f"Getting current price for {symbol}...")
                entry_price = self.api.get_ticker_price(symbol)
                
                if not entry_price:
                    logger.error(f"Failed to get current price for {symbol}")
                    return None
                logger.info(f"Got current price: {entry_price}")
            else:
                logger.info(f"Using provided entry price: {entry_price}")
            
            # Calculate stop loss price
            logger.info(f"Calculating stop loss price for {side} at {entry_price}...")
            stop_loss_price = self.calculate_stop_loss(entry_price, side)
            logger.info(f"Calculated stop loss price: {stop_loss_price}")
            
            # Calculate position size
            logger.info(f"Calculating position size for {symbol} (entry: {entry_price}, stop: {stop_loss_price})...")
            position_size = self.calculate_position_size(symbol, entry_price, stop_loss_price)
            logger.info(f"Calculated position size: {position_size}")
            
            if position_size <= 0:
                logger.error("Calculated position size is zero or negative")
                return None
            
            # DEBUG: Check account balance
            balance = self.api.get_account_balance('USDT')
            logger.info(f"USDT balance before placing order: {balance}")
            value = position_size * entry_price
            logger.info(f"Order value: ${value:.2f} (position size: {position_size} at {entry_price})")
            
            # Place main order
            main_order = None
            
            # DEBUG: Force a small position size for testing if necessary
            # This can be useful if your balance is small and the calculated position would be too large
            # position_size = 0.01  # Minimum size for most crypto pairs
            
            logger.info(f"Placing {side} market order for {symbol}, size: {position_size}...")
            if side == 'BUY':
                try:
                    main_order = self.api.place_market_order(symbol, side, position_size)
                    logger.info(f"Market order result: {main_order}")
                except Exception as market_error:
                    logger.error(f"Error placing market order: {market_error}")
                    return None
            else:
                try:
                    main_order = self.api.place_market_order(symbol, side, position_size)
                    logger.info(f"Market order result: {main_order}")
                except Exception as market_error:
                    logger.error(f"Error placing market order: {market_error}")
                    return None
            
            if not main_order:
                logger.error(f"Failed to place {side} order for {symbol} - market order returned None")
                return None
            
            # Record the trade
            logger.info(f"Recording trade for {symbol} {side}...")
            self.record_trade()
            
            # Calculate take profit price
            logger.info(f"Calculating take profit price for {side} at {entry_price}...")
            take_profit_price = self.calculate_take_profit(entry_price, side)
            logger.info(f"Calculated take profit price: {take_profit_price}")
            
            # Place stop loss order
            stop_loss_order = None
            take_profit_order = None
            
            if side == 'BUY':
                # For buy orders, place sell stop loss and take profit
                logger.info(f"Placing stop loss order for {symbol}, size: {position_size}, price: {stop_loss_price}...")
                try:
                    stop_loss_order = self.api.place_stop_loss_order(symbol, position_size, stop_loss_price)
                    logger.info(f"Stop loss order result: {stop_loss_order}")
                except Exception as sl_error:
                    logger.error(f"Error placing stop loss order: {sl_error}")
                
                logger.info(f"Placing take profit order for {symbol}, size: {position_size}, price: {take_profit_price}...")
                try:
                    take_profit_order = self.api.place_take_profit_order(symbol, position_size, take_profit_price)
                    logger.info(f"Take profit order result: {take_profit_order}")
                except Exception as tp_error:
                    logger.error(f"Error placing take profit order: {tp_error}")
            else:
                # For sell orders, we would need to implement buy stop loss and take profit
                # This is more complex and depends on the specific requirements
                logger.warning("Stop loss and take profit for SELL orders not implemented")
            
            # Return order information
            order_info = {
                'main_order': main_order,
                'stop_loss_order': stop_loss_order,
                'take_profit_order': take_profit_order,
                'symbol': symbol,
                'side': side,
                'position_size': position_size,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            }
            
            logger.info(f"Successfully placed {side} order for {symbol} with risk management: "
                      f"size={position_size}, entry={entry_price}, "
                      f"stop_loss={stop_loss_price}, take_profit={take_profit_price}")
            
            return order_info
        
        except Exception as e:
            logger.error(f"Error placing order with risk management: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def check_max_drawdown(self, initial_balance, current_balance, max_drawdown_pct=25.0):
        """
        Check if maximum drawdown has been exceeded.
        
        Args:
            initial_balance (float): Initial account balance.
            current_balance (float): Current account balance.
            max_drawdown_pct (float, optional): Maximum allowed drawdown percentage. Defaults to 25.0.
        
        Returns:
            bool: True if maximum drawdown has been exceeded, False otherwise.
        """
        if initial_balance <= 0:
            return False
        
        drawdown_pct = (initial_balance - current_balance) / initial_balance * 100.0
        
        if drawdown_pct >= max_drawdown_pct:
            logger.warning(f"Maximum drawdown exceeded: {drawdown_pct:.2f}% > {max_drawdown_pct}%")
            return True
        
        return False