"""
Risk management module for the trading bot.
"""

import os
import sys
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.binance_api import BinanceAPI
from src.utils.logger import get_logger
import config

logger = get_logger()

class RiskManager:
    """
    Risk management for trading operations with enhanced conviction-based sizing.
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
        
        # Use config values directly instead of hardcoded defaults
        self.max_risk_per_trade = max_risk_per_trade or config.MAX_RISK_PER_TRADE
        self.stop_loss_pct = stop_loss_pct or config.STOP_LOSS_PERCENTAGE
        self.take_profit_pct = take_profit_pct or config.TAKE_PROFIT_PERCENTAGE
        self.max_trades_per_day = max_trades_per_day or config.MAX_TRADES_PER_DAY
        self.daily_loss_limit_pct = daily_loss_limit_pct or config.DAILY_LOSS_LIMIT
        
        # Initialize trade tracking
        self.trades_today = 0
        self.daily_loss = 0.0
        self.initial_daily_balance = 0.0
        self.last_day_reset = datetime.now().date()
        
        # Fee settings from config
        self.fee_rate = config.TAKER_FEE_RATE  # Use taker fee rate for market orders
        self.maker_fee_rate = config.MAKER_FEE_RATE  # Use maker fee rate for limit orders
        self.min_fee_multiplier = config.MIN_FEE_MULTIPLIER
        self.use_bnb_for_fees = config.USE_BNB_FOR_FEES
        self.min_profit_multiplier = config.MIN_PROFIT_MULTIPLIER
        
        # TRON 1.1: Conviction-based position sizing
        self.use_conviction_sizing = getattr(config, 'USE_CONVICTION_SIZING', True)
        
        # Minimum conviction threshold
        self.min_conviction_threshold = getattr(config, 'MIN_CONVICTION_THRESHOLD', 0.65)
        
        # Custom conviction multipliers from config
        self.conviction_multipliers = getattr(config, 'CONVICTION_MULTIPLIERS', {
            0.65: 0.5,  # 50% of standard position size for minimum conviction
            0.75: 0.75, # 75% of standard position size for medium conviction
            0.85: 1.0,  # 100% of standard position size for high conviction
            1.0: 1.2    # 120% of standard position size for maximum conviction
        })
        
        # Update the fee rate if using BNB for fees
        if self.use_bnb_for_fees:
            self.fee_rate = self.fee_rate * (1 - config.BNB_FEE_DISCOUNT)
            self.maker_fee_rate = self.maker_fee_rate * (1 - config.BNB_FEE_DISCOUNT)
        
        logger.info(f"Initialized risk manager with max_risk_per_trade={self.max_risk_per_trade}%, "
                   f"stop_loss_pct={self.stop_loss_pct}%, take_profit_pct={self.take_profit_pct}%, "
                   f"max_trades_per_day={self.max_trades_per_day}, daily_loss_limit_pct={self.daily_loss_limit_pct}%")
        logger.info(f"Fee settings: rate={self.fee_rate*100}%, maker_rate={self.maker_fee_rate*100}%, "
                   f"min_multiplier={self.min_fee_multiplier}x, profit_multiplier={self.min_profit_multiplier}x, "
                   f"use_bnb={self.use_bnb_for_fees}")
        logger.info(f"TRON 1.1: Using conviction-based position sizing: {self.use_conviction_sizing}")
        logger.info(f"Minimum conviction threshold: {self.min_conviction_threshold}")
        logger.info(f"Custom conviction multipliers: {self.conviction_multipliers}")
    
    def calculate_fee(self, order_value, is_maker=False):
        """
        Calculate fee for a trade.
        
        Args:
            order_value (float): Order value in USDT.
            is_maker (bool, optional): Whether this is a maker order. Defaults to False.
        
        Returns:
            float: Fee amount in USDT.
        """
        fee_rate = self.maker_fee_rate if is_maker else self.fee_rate
        return order_value * fee_rate
    
    def is_trade_viable(self, order_value, is_maker=False):
        """
        Check if a trade is viable after fees.
        
        Args:
            order_value (float): Order value in USDT.
            is_maker (bool, optional): Whether this is a maker order. Defaults to False.
        
        Returns:
            bool: True if the trade is viable, False otherwise.
        """
        fee = self.calculate_fee(order_value, is_maker)
        min_viable_trade = fee * self.min_fee_multiplier
        return order_value >= min_viable_trade
    
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
    
    def track_account_growth(self, initial_balance):
        """
        Track account growth and adjust risk parameters accordingly.
        
        Args:
            initial_balance (float): Initial account balance.
            
        Returns:
            dict: Account growth metrics.
        """
        if initial_balance <= 0:
            logger.warning("Invalid initial balance for tracking growth")
            return {'growth_pct': 0, 'risk_factor': 1.0}
        
        # Get current account value
        current_balance = self.get_account_value()
        
        if current_balance <= 0:
            logger.warning("Invalid current balance for tracking growth")
            return {'growth_pct': 0, 'risk_factor': 1.0}
        
        # Calculate growth
        growth_pct = ((current_balance / initial_balance) - 1) * 100
        
        # Adjust risk factor based on growth
        # More conservative approach - start with lower risk and increase slowly
        risk_factor = 0.8  # Base risk factor (slightly reduced)
        
        if growth_pct >= 100:  # Double or more
            risk_factor = 1.2  # Less aggressive than before
        elif growth_pct >= 50:  # 50% or more growth
            risk_factor = 1.1  # Less aggressive than before  
        elif growth_pct >= 20:  # 20% or more growth
            risk_factor = 1.0  # Standard risk
        elif growth_pct <= -10:  # 10% or more loss
            risk_factor = 0.6  # More conservative when losing
        elif growth_pct <= -20:  # 20% or more loss
            risk_factor = 0.4  # Much more conservative when losing badly
        
        logger.info(f"Account growth: ${current_balance:.2f} ({growth_pct:.2f}%), Risk adjustment factor: {risk_factor}")
        
        return {
            'current_balance': current_balance,
            'growth_pct': growth_pct,
            'risk_factor': risk_factor
        }
    
    def calculate_drawdown(self, current_balance, max_drawdown_pct=15.0):
        """
        Calculate current drawdown percentage.
        
        Args:
            current_balance (float): Current account balance.
            max_drawdown_pct (float, optional): Maximum allowed drawdown percentage. Defaults to 15.0.
        
        Returns:
            float: Current drawdown percentage.
        """
        self.max_drawdown_pct = max_drawdown_pct
        
        if not hasattr(self, 'highest_balance'):
            self.highest_balance = current_balance
        else:
            self.highest_balance = max(self.highest_balance, current_balance)
        
        if self.highest_balance <= 0 or current_balance <= 0:
            return 0.0
        
        drawdown_pct = ((self.highest_balance - current_balance) / self.highest_balance) * 100
        
        return drawdown_pct
    
    def get_conviction_multiplier(self, conviction_strength):
        """
        Get position size multiplier based on signal conviction using config settings.
        
        Args:
            conviction_strength (float): Signal conviction strength (0.0-1.0).
            
        Returns:
            float: Position size multiplier.
        """
        # Return 0 if below minimum threshold
        if conviction_strength < self.min_conviction_threshold:
            return 0.0
            
        # Use custom conviction multipliers from config if available
        if hasattr(self, 'conviction_multipliers') and self.conviction_multipliers:
            # Find the closest key in conviction_multipliers that's less than or equal to conviction_strength
            applicable_threshold = 0
            multiplier = 0
            
            for threshold, mult in sorted(self.conviction_multipliers.items()):
                threshold_float = float(threshold)
                if conviction_strength >= threshold_float and threshold_float > applicable_threshold:
                    applicable_threshold = threshold_float
                    multiplier = mult
            
            logger.info(f"Using conviction multiplier {multiplier} for strength {conviction_strength} (threshold: {applicable_threshold})")
            return multiplier
        else:
            # Fallback to simple linear scaling
            return min(conviction_strength, 1.0)
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, conviction_multiplier=1.0):
        """
        Calculate position size based on risk management rules, with optional conviction multiplier.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            entry_price (float): Entry price.
            stop_loss_price (float): Stop loss price.
            conviction_multiplier (float, optional): Multiplier for position size based on signal conviction. Defaults to 1.0.
        
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
            
            # TRON 1.1: Apply conviction multiplier to risk amount if enabled
            base_risk = self.max_risk_per_trade / 100.0
            
            if self.use_conviction_sizing and conviction_multiplier != 1.0:
                # Use custom multiplier lookup instead of direct multiplication
                pos_multiplier = self.get_conviction_multiplier(conviction_multiplier)
                adjusted_risk = base_risk * pos_multiplier
                logger.info(f"Adjusting risk based on conviction: {base_risk*100:.2f}% -> {adjusted_risk*100:.2f}% (strength: {conviction_multiplier}, multiplier: {pos_multiplier})")
                risk_pct = adjusted_risk
            else:
                risk_pct = base_risk
            
            # If conviction is below minimum threshold, return 0
            if conviction_multiplier < self.min_conviction_threshold:
                logger.info(f"Conviction {conviction_multiplier} below minimum threshold {self.min_conviction_threshold}, skipping trade")
                return 0.0
            
            # Calculate risk amount
            risk_amount = account_value * risk_pct
            logger.info(f"Risk amount based on account value and conviction: ${risk_amount:.2f}")
            
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
            
            # Get minimum order value for this symbol from config
            min_order_value = config.MIN_ORDER_VALUES.get(symbol, config.MIN_ORDER_VALUES['DEFAULT'])
            
            # Calculate minimum position size based on minimum order value
            min_position_base = min_order_value / entry_price
            logger.info(f"Minimum position size for {symbol}: {min_position_base} (${min_order_value:.2f})")
            
            # Check if calculated position size is too small
            if position_size_base < min_position_base:
                logger.warning(f"Risk-based position size {position_size_base} is below minimum. Using minimum size: {min_position_base}")
                position_size_base = min_position_base
            
            # Check if position value is sufficient considering fees
            position_value = position_size_base * entry_price
            estimated_fee = self.calculate_fee(position_value)
            min_viable_trade = estimated_fee * self.min_fee_multiplier
            
            logger.info(f"Position value: ${position_value:.2f}, Estimated fee: ${estimated_fee:.2f}, Min viable: ${min_viable_trade:.2f}")
            
            # Use fee-adjusted minimum values from config if available
            fee_adjusted_min_value = getattr(config, 'FEE_ADJUSTED_MIN_VALUES', {}).get(symbol, 
                                     getattr(config, 'FEE_ADJUSTED_MIN_VALUES', {}).get('DEFAULT', min_order_value))
            fee_adjusted_min_position = fee_adjusted_min_value / entry_price
            
            # If min_order_value isn't sufficient to cover fees adequately, use fee-adjusted minimum
            if position_value < min_viable_trade or min_order_value < min_viable_trade:
                logger.warning(f"Using fee-adjusted minimum position: {fee_adjusted_min_position} (${fee_adjusted_min_value:.2f})")
                
                # If our calculated position is below the fee-adjusted minimum, use the fee-adjusted minimum
                if position_size_base < fee_adjusted_min_position:
                    logger.warning(f"Risk-based position size {position_size_base} is below fee-adjusted minimum. Using adjusted minimum: {fee_adjusted_min_position}")
                    position_size_base = fee_adjusted_min_position
            
            # TESTING MODE: Use testing sizes if enabled
            if hasattr(config, 'TESTING_MODE') and config.TESTING_MODE:
                # Get testing size for this symbol from config
                test_size = getattr(config, 'TESTING_SIZES', {}).get(symbol, 
                            getattr(config, 'TESTING_SIZES', {}).get('DEFAULT', position_size_base))
                
                # TRON 1.1: Apply conviction multiplier to test size if enabled
                if self.use_conviction_sizing and conviction_multiplier != 1.0:
                    # Use custom multiplier lookup instead of direct multiplication
                    pos_multiplier = self.get_conviction_multiplier(conviction_multiplier) 
                    adjusted_test_size = test_size * pos_multiplier
                    logger.info(f"TESTING MODE: Adjusting test size based on conviction: {test_size} -> {adjusted_test_size} (strength: {conviction_multiplier}, multiplier: {pos_multiplier})")
                    test_size = adjusted_test_size
                else:
                    logger.info(f"TESTING MODE: Using standard position size of {test_size} for {symbol}")
                
                # Check if even the testing size is viable for fees
                test_value = test_size * entry_price
                test_fee = self.calculate_fee(test_value)
                if test_value < test_fee * self.min_fee_multiplier:
                    logger.warning(f"Testing size value (${test_value:.2f}) too small compared to fees (${test_fee:.2f}). Position may not be profitable.")
                
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
            
            # Final fee check after all adjustments
            final_value = position_size_base * entry_price
            final_fee = self.calculate_fee(final_value)
            logger.info(f"Final position size for {symbol}: {position_size_base} (value: ${final_value:.2f}, fee: ${final_fee:.2f})")
            
            # Check if the final position is viable considering fees
            if not self.is_trade_viable(final_value):
                logger.warning(f"Final position value (${final_value:.2f}) may be too small compared to fees (${final_fee:.2f}). Trade may not be profitable.")
            
            return position_size_base
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    # Remaining methods unchanged
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
    
    def calculate_profit_after_fees(self, entry_price, current_price, position_size, side='BUY'):
        """
        Calculate profit after fees.
        
        Args:
            entry_price (float): Entry price.
            current_price (float): Current price.
            position_size (float): Position size.
            side (str, optional): Order side ('BUY' or 'SELL'). Defaults to 'BUY'.
        
        Returns:
            dict: Profit information.
        """
        entry_value = position_size * entry_price
        current_value = position_size * current_price
        
        buy_fee = self.calculate_fee(entry_value)
        sell_fee = self.calculate_fee(current_value)
        total_fees = buy_fee + sell_fee
        
        if side == 'BUY':
            gross_profit = current_value - entry_value
        else:
            gross_profit = entry_value - current_value
        
        actual_profit = gross_profit - total_fees
        profit_percentage = (actual_profit / entry_value) * 100 if entry_value > 0 else 0
        
        return {
            'gross_profit': gross_profit,
            'total_fees': total_fees,
            'actual_profit': actual_profit,
            'profit_percentage': profit_percentage,
            'is_profitable': actual_profit > 0
        }
    
    def is_profit_viable(self, entry_price, current_price, position_size, side='BUY'):
        """
        Check if a trade would result in viable profit after fees.
        
        Args:
            entry_price (float): Entry price.
            current_price (float): Current price.
            position_size (float): Position size.
            side (str, optional): Order side ('BUY' or 'SELL'). Defaults to 'BUY'.
        
        Returns:
            bool: True if the profit is viable, False otherwise.
        """
        profit_info = self.calculate_profit_after_fees(entry_price, current_price, position_size, side)
        
        # Check if the profit is greater than the fees multiplied by the minimum profit multiplier
        return profit_info['actual_profit'] > 0 and profit_info['actual_profit'] >= (profit_info['total_fees'] * self.min_profit_multiplier)
    
    def place_order_with_risk_management(self, symbol, side, entry_price=None, conviction_multiplier=1.0):
        """
        Place an order with risk management and optional conviction-based sizing.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            side (str): Order side ('BUY' or 'SELL').
            entry_price (float, optional): Entry price. Defaults to current market price.
            conviction_multiplier (float, optional): Multiplier for position size based on signal conviction. Defaults to 1.0.
        
        Returns:
            dict: Order information including main order and stop loss/take profit orders.
        """
        try:
            logger.info(f"Starting order placement for {symbol} {side} with risk management (conviction: {conviction_multiplier})")
            
            # Check if conviction is below minimum threshold
            if conviction_multiplier < self.min_conviction_threshold:
                logger.warning(f"Conviction {conviction_multiplier} below minimum threshold {self.min_conviction_threshold}, skipping trade")
                return None
            
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
            
            # Calculate position size with conviction multiplier
            logger.info(f"Calculating position size for {symbol} with conviction {conviction_multiplier}...")
            position_size = self.calculate_position_size(symbol, entry_price, stop_loss_price, conviction_multiplier)
            logger.info(f"Calculated position size: {position_size}")
            
            if position_size <= 0:
                logger.error("Calculated position size is zero or negative")
                return None
            
            # Check if position size creates a viable trade considering fees
            position_value = position_size * entry_price
            if not self.is_trade_viable(position_value):
                estimated_fee = self.calculate_fee(position_value)
                min_viable_trade = estimated_fee * self.min_fee_multiplier
                logger.warning(f"Position value (${position_value:.2f}) too small compared to fees (est: ${estimated_fee:.2f}, min: ${min_viable_trade:.2f}). Skipping trade.")
                return None
            
            # DEBUG: Check account balance
            balance = self.api.get_account_balance('USDT')
            logger.info(f"USDT balance before placing order: {balance}")
            logger.info(f"Order value: ${position_value:.2f} (position size: {position_size} at {entry_price})")
            
            # Place main order
            main_order = None
            
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
                'take_profit_price': take_profit_price,
                'estimated_fee': self.calculate_fee(position_value),
                'conviction_multiplier': conviction_multiplier  # TRON 1.1: Add conviction info
            }
            
            logger.info(f"Successfully placed {side} order for {symbol} with risk management: "
                      f"size={position_size}, entry={entry_price}, "
                      f"stop_loss={stop_loss_price}, take_profit={take_profit_price}, "
                      f"conviction={conviction_multiplier}")
            
            return order_info
        
        except Exception as e:
            logger.error(f"Error placing order with risk management: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def check_max_drawdown(self, initial_balance, current_balance, max_drawdown_pct=None):
        """
        Check if maximum drawdown has been exceeded.
        
        Args:
            initial_balance (float): Initial account balance.
            current_balance (float): Current account balance.
            max_drawdown_pct (float, optional): Maximum allowed drawdown percentage. Defaults to config value.
        
        Returns:
            bool: True if maximum drawdown has been exceeded, False otherwise.
        """
        if initial_balance <= 0:
            return False
        
        # Use config value if available
        if max_drawdown_pct is None and hasattr(config, 'MAX_DRAWDOWN_PERCENT'):
            max_drawdown_pct = config.MAX_DRAWDOWN_PERCENT
        elif max_drawdown_pct is None:
            max_drawdown_pct = 15.0  # Default value
        
        drawdown_pct = (initial_balance - current_balance) / initial_balance * 100.0
        
        if drawdown_pct >= max_drawdown_pct:
            logger.warning(f"Maximum drawdown exceeded: {drawdown_pct:.2f}% > {max_drawdown_pct}%")
            return True
        
        return False