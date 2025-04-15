"""
Binance API client for the cryptocurrency trading bot.
Optimized for better error handling and performance.
"""

import os
import time
import math
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from datetime import datetime
import sys
import json

# Add the parent directory to the path to import config and logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils.logger import get_logger
from src.utils.logger_utils import log_api_error  # Import API error logging function

logger = get_logger()

class BinanceAPI:
    """
    Wrapper class for Binance API interactions.
    """
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialize the Binance API client.
        
        Args:
            api_key (str, optional): Binance API key. Defaults to config value.
            api_secret (str, optional): Binance API secret. Defaults to config value.
            testnet (bool, optional): Whether to use testnet. Defaults to config value.
        """
        self.api_key = api_key or config.API_KEY
        self.api_secret = api_secret or config.API_SECRET
        self.testnet = testnet if testnet is not None else config.USE_TESTNET
        
        # Track time offset for API calls
        self.time_offset = 0
        
        try:
            # Initialize the Binance client with appropriate tld
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                tld='com'  # Add this parameter - use 'us' for Binance.US or 'com' for Binance.com
            )
            
            # Synchronize time with Binance servers to avoid timestamp issues
            self._sync_time()
            
            logger.info(f"Binance API client initialized (testnet: {self.testnet})")
        except Exception as e:
            logger.error(f"Error initializing Binance API client: {e}")
            raise
    
    def _sync_time(self):
        """
        Synchronize local time with Binance server time to avoid timestamp errors.
        """
        try:
            server_time = self.client.get_server_time()
            server_timestamp = server_time['serverTime']
            local_timestamp = int(time.time() * 1000)  # Local time in milliseconds
            self.time_offset = server_timestamp - local_timestamp
            logger.info(f"Time synchronized with Binance server. Offset: {self.time_offset} ms")
        except Exception as e:
            logger.error(f"Failed to synchronize time with Binance server: {e}")
    
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
            dict: Account information.
        """
        try:
            account_info = self.client.get_account()
            logger.info("Retrieved account information")
            return account_info
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error("ACCOUNT", "get_account", str(e))
            logger.error(f"Error getting account information: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    account_info = self.client.get_account()
                    logger.info("Retrieved account information after time sync")
                    return account_info
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def get_account_balance(self, asset=None):
        """
        Get account balance for a specific asset or all assets.
        
        Args:
            asset (str, optional): Asset symbol (e.g., 'BTC'). Defaults to None.
        
        Returns:
            dict or float: Balance information for all assets or specific asset.
        """
        try:
            account_info = self.client.get_account()
            balances = account_info['balances']
            
            if asset:
                for balance in balances:
                    if balance['asset'] == asset:
                        free_balance = float(balance['free'])
                        logger.info(f"Retrieved balance for {asset}: {free_balance}")
                        return free_balance
                logger.warning(f"Asset {asset} not found in account")
                return 0.0
            else:
                # Return all non-zero balances
                non_zero_balances = {
                    balance['asset']: float(balance['free'])
                    for balance in balances
                    if float(balance['free']) > 0
                }
                logger.info(f"Retrieved balances for all assets: {non_zero_balances}")
                return non_zero_balances
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error("ACCOUNT", "get_account_balance", str(e), {"asset": asset})
            logger.error(f"Error getting account balance: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    return self.get_account_balance(asset)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def get_symbol_info(self, symbol):
        """
        Get information about a trading symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
        
        Returns:
            dict: Symbol information.
        """
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if symbol_info:
                logger.info(f"Retrieved information for symbol {symbol}")
                return symbol_info
            else:
                logger.warning(f"Symbol {symbol} not found on Binance")
                return None
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error(symbol, "get_symbol_info", str(e))
            logger.error(f"Error getting symbol information for {symbol}: {e}")
            return None
    
    def get_ticker_price(self, symbol):
        """
        Get current price for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
        
        Returns:
            float: Current price.
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.info(f"Retrieved current price for {symbol}: {price}")
            return price
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error(symbol, "get_ticker_price", str(e))
            logger.error(f"Error getting ticker price for {symbol}: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    logger.info(f"Retrieved current price for {symbol} after time sync: {price}")
                    return price
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Get historical klines (candlestick data).
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_str (str): Start time in format 'YYYY-MM-DD' or '1 day ago UTC'.
                             Can also be 'lookback_period' for backward compatibility.
            end_str (str, optional): End time. Defaults to None (current time).
        
        Returns:
            pandas.DataFrame: Historical klines data.
        """
        try:
            # Use CHART_INTERVAL from config instead of passed interval
            # This ensures we always use a valid interval format accepted by Binance
            interval_to_use = config.CHART_INTERVAL
            
            # Handle the case where start_str is actually a lookback_period
            # This provides compatibility with the old method signature
            if end_str is None and isinstance(start_str, str) and 'ago' in start_str:
                # We're likely being called with the lookback_period parameter
                # so keep the original start_str
                lookback_period = start_str
                end_time = None
            else:
                # We're being called with the standard parameters
                lookback_period = start_str
                end_time = end_str
                
            logger.info(f"Getting historical klines for {symbol} with interval {interval_to_use}, lookback: {lookback_period}")
            
            # Add retry logic for klines retrieval
            max_retries = 5  # Increased from 3
            retry_delay = 5  # Increased from 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Increase timeout for this specific API call
                    self.client.session.request_timeout = 30  # Increased from default 10
                    
                    klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval_to_use,
                        start_str=lookback_period,
                        end_str=end_time
                    )
                    
                    # Reset timeout to default after successful call
                    self.client.session.request_timeout = 10
                    
                    if not klines:
                        logger.warning(f"No klines returned for {symbol}")
                        return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    logger.info(f"Retrieved {len(df)} historical klines for {symbol} ({interval_to_use})")
                    return df
                    
                except (BinanceAPIException, BinanceRequestException) as e:
                    # Only retry for specific error codes
                    if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                        # Timestamp error - sync time and retry
                        self._sync_time()
                        logger.warning(f"Timestamp error, retrying attempt {attempt+1}/{max_retries} after time sync...")
                        time.sleep(retry_delay)
                    elif attempt < max_retries - 1:
                        # Exponential backoff for other errors
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"API error, retrying attempt {attempt+1}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed
                        log_api_error(symbol, "get_historical_klines", str(e))
                        logger.error(f"Error getting historical klines for {symbol} after {max_retries} attempts: {e}")
                        return None
                        
                    # Reset timeout to default after failed call
                    self.client.session.request_timeout = 10
                    
                except Exception as e:
                    # For network timeouts and other exceptions
                    if "Read timed out" in str(e) and attempt < max_retries - 1:
                        # Exponential backoff for timeout errors
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Timeout error, retrying attempt {attempt+1}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
                        
                        # Reset timeout to default and try with an even longer timeout
                        self.client.session.request_timeout = 10 + (10 * attempt)
                    else:
                        log_api_error(symbol, "get_historical_klines", str(e))
                        logger.error(f"Unexpected error getting historical klines for {symbol} on attempt {attempt+1}: {e}")
                        if attempt >= max_retries - 1:
                            return None
                            
                        # Exponential backoff for other errors
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Unexpected error, retrying attempt {attempt+1}/{max_retries} after {wait_time}s...")
                        time.sleep(wait_time)
            
            # If we get here, all retries failed
            return None
                
        except Exception as e:
            log_api_error(symbol, "get_historical_klines", str(e))
            logger.error(f"Unexpected error getting historical klines for {symbol}: {e}")
            return None
    
    def validate_and_adjust_quantity(self, symbol, quantity):
        """
        Validate and adjust a quantity to ensure it meets Binance's requirements.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            quantity (float): Original quantity to validate
            
        Returns:
            float or None: Adjusted quantity that meets requirements, or None if impossible
        """
        try:
            # First, get symbol info to find precision requirements
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
            
            # Log the original quantity for debugging
            logger.info(f"Original quantity for {symbol}: {quantity}")
            
            # Check if quantity is too small (below 0.00001)
            if quantity < 0.00001:
                logger.error(f"Quantity {quantity} for {symbol} is too small (min: 0.00001)")
                return None
                
            # Find the LOT_SIZE filter for quantity precision
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                min_qty = float(lot_size_filter['minQty'])
                max_qty = float(lot_size_filter['maxQty'])
                step_size = float(lot_size_filter['stepSize'])
                
                # Check minimum quantity
                if quantity < min_qty:
                    logger.error(f"Quantity {quantity} is below minimum {min_qty} for {symbol}")
                    return None
                    
                # Check maximum quantity
                if quantity > max_qty:
                    logger.error(f"Quantity {quantity} is above maximum {max_qty} for {symbol}")
                    quantity = max_qty
                    logger.info(f"Adjusted quantity to maximum: {quantity}")
                
                # Adjust to step size precision
                precision = int(round(-math.log10(step_size)))
                
                # Use proper rounding to stepsize
                quantity = math.floor(quantity * 10**precision) / 10**precision
                
                logger.info(f"Adjusted quantity to step size precision ({precision} decimals): {quantity}")
                
                # Check again if quantity is now below min after rounding
                if quantity < min_qty:
                    logger.error(f"Adjusted quantity {quantity} is below minimum {min_qty} for {symbol}")
                    return None
            
            # Check MIN_NOTIONAL value
            current_price = self.get_ticker_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return None
                
            order_value = quantity * current_price
            
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if min_notional_filter:
                min_notional = float(min_notional_filter['minNotional'])
                if order_value < min_notional:
                    logger.error(f"Order value ${order_value} is below minimum notional ${min_notional} for {symbol}")
                    
                    # Try to calculate a valid quantity based on min notional
                    suggested_qty = min_notional / current_price
                    
                    # Adjust suggested quantity to step size
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                        precision = int(round(-math.log10(step_size)))
                        suggested_qty = math.ceil(suggested_qty * 10**precision) / 10**precision
                        
                    logger.info(f"Suggested minimum quantity to meet min notional: {suggested_qty}")
                    return None
            
            # Check with min fee multiplier for profitability
            # Ensure the order is large enough that fees don't eat all potential profit
            fee_rate = getattr(config, 'TAKER_FEE_RATE', 0.001)  # Default to 0.1% if not in config
            min_fee_multiplier = getattr(config, 'MIN_FEE_MULTIPLIER', 5)  # Default to 5x if not in config
            
            estimated_fee = order_value * fee_rate
            min_viable_trade = estimated_fee * min_fee_multiplier
            
            if order_value < min_viable_trade:
                logger.warning(f"Order value (${order_value:.2f}) may be too small compared to fees (${estimated_fee:.2f})")
                logger.warning(f"Recommended minimum: ${min_viable_trade:.2f} for profitability")
                # Continue anyway, but log the warning
            
            return quantity
        
        except Exception as e:
            log_api_error(symbol, "validate_and_adjust_quantity", str(e))
            logger.error(f"Error validating quantity for {symbol}: {e}")
            return None
    
    def place_stop_loss_order(self, symbol, quantity, stop_price):
        """
        Place a stop loss order with price precision handling.
        """
        try:
            # Validate and adjust quantity first
            adjusted_quantity = self.validate_and_adjust_quantity(symbol, quantity)
            if not adjusted_quantity:
                logger.error(f"Cannot place stop loss order for {symbol}: Invalid quantity {quantity}")
                return None
            
            # Get symbol info to find price precision
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            # Find the PRICE_FILTER for price precision
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                # Get minimum notional value
                min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                min_notional = float(min_notional_filter['minNotional']) if min_notional_filter else 10.0
                
                # Round to appropriate precision
                precision = int(round(-math.log10(tick_size)))
                stop_price = round(stop_price, precision)
                limit_price = round(stop_price * 0.99, precision)  # Slightly below stop price
                
                logger.info(f"Rounded stop price to {precision} decimals: {stop_price}, limit price: {limit_price}")
            else:
                # If no price filter found, use a default precision
                stop_price = round(stop_price, 2)
                limit_price = round(stop_price * 0.99, 2)
                logger.info(f"No price filter found, using default precision. Stop price: {stop_price}, limit price: {limit_price}")
            
            # Check if order meets minimum notional value
            order_value = adjusted_quantity * limit_price
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if min_notional_filter and order_value < float(min_notional_filter['minNotional']):
                logger.error(f"Order value ${order_value} is below minimum notional ${float(min_notional_filter['minNotional'])}")
                return None
                
            # Try to place the order with retries for timestamp errors
            max_retries = 2
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    order = self.client.create_order(
                        symbol=symbol,
                        side='SELL',
                        type='STOP_LOSS_LIMIT',
                        timeInForce='GTC',
                        quantity=adjusted_quantity,
                        price=limit_price,
                        stopPrice=stop_price
                    )
                    logger.info(f"Placed stop loss order for {adjusted_quantity} {symbol} at {stop_price}: {order['orderId']}")
                    return order
                    
                except (BinanceAPIException, BinanceRequestException) as e:
                    # Handle timestamp errors specifically
                    if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022) and attempt < max_retries - 1:
                        self._sync_time()
                        logger.warning(f"Timestamp error, retrying after time sync...")
                        time.sleep(retry_delay)
                    else:
                        log_api_error(symbol, "place_stop_loss_order", str(e))
                        logger.error(f"Error placing stop loss order for {symbol}: {e}")
                        return None
            
            # If we get here, all retries failed
            return None
            
        except Exception as e:
            log_api_error(symbol, "place_stop_loss_order", str(e))
            logger.error(f"Unexpected error placing stop loss order for {symbol}: {e}")
            return None
    
    def place_take_profit_order(self, symbol, quantity, take_profit_price):
        """
        Place a take profit order with price precision handling.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            quantity (float): Order quantity.
            take_profit_price (float): Take profit price.
        
        Returns:
            dict: Order information.
        """
        try:
            # Validate and adjust quantity first
            adjusted_quantity = self.validate_and_adjust_quantity(symbol, quantity)
            if not adjusted_quantity:
                logger.error(f"Cannot place take profit order for {symbol}: Invalid quantity {quantity}")
                return None
            
            # Get symbol info to find price precision
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            # Find the PRICE_FILTER for price precision
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                # Round to appropriate precision
                precision = int(round(-math.log10(tick_size)))
                take_profit_price = round(take_profit_price, precision)
                
                logger.info(f"Rounded take profit price to {precision} decimals: {take_profit_price}")
            else:
                # If no price filter found, use a default precision
                take_profit_price = round(take_profit_price, 2)
                logger.info(f"No price filter found, using default precision. Take profit price: {take_profit_price}")
            
            # Try to place the order with retries for timestamp errors
            max_retries = 2
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    order = self.client.create_order(
                        symbol=symbol,
                        side='SELL',
                        type='LIMIT',
                        timeInForce='GTC',
                        quantity=adjusted_quantity,
                        price=take_profit_price
                    )
                    logger.info(f"Placed take profit order for {adjusted_quantity} {symbol} at {take_profit_price}: {order['orderId']}")
                    return order
                    
                except (BinanceAPIException, BinanceRequestException) as e:
                    # Handle timestamp errors specifically
                    if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022) and attempt < max_retries - 1:
                        self._sync_time()
                        logger.warning(f"Timestamp error, retrying after time sync...")
                        time.sleep(retry_delay)
                    else:
                        log_api_error(symbol, "place_take_profit_order", str(e))
                        logger.error(f"Error placing take profit order for {symbol}: {e}")
                        return None
            
            # If we get here, all retries failed
            return None
            
        except Exception as e:
            log_api_error(symbol, "place_take_profit_order", str(e))
            logger.error(f"Error placing take profit order for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, side, quantity):
        """
        Place a market order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Order quantity.
        
        Returns:
            dict: Order information.
        """
        # Get original quantity for logging
        original_quantity = quantity
        logger.info(f"Attempting to place {side} order for {symbol} with quantity {quantity}")
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        # Get current price for value calculations
        current_price = self.get_ticker_price(symbol)
        if not current_price:
            logger.error(f"Failed to get current price for {symbol}")
            return None
            
        # Get minimum notional value requirement
        min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
        min_notional = float(min_notional_filter['minNotional']) if min_notional_filter else 10.0
        logger.info(f"Minimum notional value for {symbol}: ${min_notional}")
        
        # Get minimum quantity
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log10(step_size)))
            
            # Round quantity to meet step size requirements
            quantity = math.floor(quantity * 10**precision) / 10**precision
            logger.info(f"Adjusted quantity to meet step size: {quantity}")
            
            # Check if the quantity meets minimum requirements
            if quantity < min_qty:
                logger.warning(f"Quantity ({quantity}) below minimum ({min_qty}) for {symbol}. Skipping order.")
                return None
        
        # Check if order meets minimum notional value
        order_value = quantity * current_price
        if order_value < min_notional:
            logger.warning(f"Order value (${order_value:.2f}) below minimum (${min_notional}). Skipping order.")
            return None
        
        # Check if SELL order would be profitable after fees
        if side == 'SELL':
            # Get fee rate from config
            fee_rate = getattr(config, 'TAKER_FEE_RATE', 0.001)
            min_profit_multiplier = getattr(config, 'MIN_PROFIT_MULTIPLIER', 3)
            
            # Estimate fees
            estimated_fee = order_value * fee_rate
            
            # Check if profitable after fees
            min_viable_trade = estimated_fee * min_profit_multiplier
            if order_value < min_viable_trade:
                logger.warning(f"Trade may not be profitable - value ${order_value:.2f}, fees ~${estimated_fee:.2f}")
                logger.warning(f"Recommended min value: ${min_viable_trade:.2f} (fee x {min_profit_multiplier})")
                # Continue anyway, but log the warning
        
        # Place the order with the adjusted quantity
        try:
            # Try to place the order with retries for timestamp errors
            max_retries = 2
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Placing {side} market order for {quantity} {symbol} at approx. ${current_price}")
                    order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=quantity
                    )
                    logger.info(f"Successfully placed {side} market order: {order['orderId']}")
                    
                    # Add value information to the log
                    filled_qty = sum(float(fill['qty']) for fill in order.get('fills', []))
                    if filled_qty > 0:
                        avg_price = sum(float(fill['qty']) * float(fill['price']) for fill in order.get('fills', [])) / filled_qty
                        total_value = filled_qty * avg_price
                        logger.info(f"Order filled: {filled_qty} {symbol} at avg price ${avg_price} (value: ${total_value:.2f})")
                    
                    return order
                    
                except (BinanceAPIException, BinanceRequestException) as e:
                    # Handle timestamp errors specifically
                    if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022) and attempt < max_retries - 1:
                        self._sync_time()
                        logger.warning(f"Timestamp error, retrying after time sync...")
                        time.sleep(retry_delay)
                    else:
                        log_api_error(symbol, "place_market_order", str(e), {"side": side, "quantity": quantity})
                        
                        # Add specific handling for common errors
                        if isinstance(e, BinanceAPIException):
                            if e.code == -1013:  # Invalid quantity or notional
                                logger.error(f"Binance rejected quantity. Raw error: {e.message}")
                                
                                # Try to extract the actual requirements from the error message
                                if "LOT_SIZE" in e.message:
                                    logger.error("Quantity doesn't meet LOT_SIZE filter requirements")
                                elif "NOTIONAL" in e.message:
                                    logger.error("Order value doesn't meet minimum notional value requirement")
                            elif e.code == -2010:  # Insufficient balance
                                logger.error(f"Insufficient balance for order. Raw error: {e.message}")
                        
                        return None
            
            # If we get here, all retries failed
            return None
            
        except Exception as e:
            log_api_error(symbol, "place_market_order", str(e), {"side": side, "quantity": quantity})
            logger.error(f"Unexpected error placing {side} market order for {symbol}: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """
        Get open orders.
        
        Args:
            symbol (str, optional): Trading symbol (e.g., 'BTCUSDT'). Defaults to None.
        
        Returns:
            list: Open orders.
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
                logger.info(f"Retrieved {len(orders)} open orders for {symbol}")
            else:
                orders = self.client.get_open_orders()
                logger.info(f"Retrieved {len(orders)} open orders for all symbols")
            return orders
        except (BinanceAPIException, BinanceRequestException) as e:
            endpoint = f"get_open_orders/{symbol}" if symbol else "get_open_orders/all"
            log_api_error(symbol or "ALL", endpoint, str(e))
            logger.error(f"Error getting open orders: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    return self.get_open_orders(symbol)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def cancel_order(self, symbol, order_id):
        """
        Cancel an order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            order_id (int): Order ID.
        
        Returns:
            dict: Cancellation result.
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error(symbol, "cancel_order", str(e), {"order_id": order_id})
            logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    return self.cancel_order(symbol, order_id)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            # If order doesn't exist, consider it successfully cancelled
            if isinstance(e, BinanceAPIException) and e.code == -2011:
                logger.info(f"Order {order_id} not found - may already be filled or cancelled")
                return {"status": "CANCELED", "message": "Order not found or already cancelled"}
            
            return None
    
    def get_order_status(self, symbol, order_id):
        """
        Get order status.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            order_id (int): Order ID.
        
        Returns:
            dict: Order information.
        """
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            logger.info(f"Retrieved status for order {order_id} for {symbol}: {order['status']}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error(symbol, "get_order_status", str(e), {"order_id": order_id})
            logger.error(f"Error getting order status for {order_id} for {symbol}: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    return self.get_order_status(symbol, order_id)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def get_all_orders(self, symbol):
        """
        Get all orders for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
        
        Returns:
            list: All orders.
        """
        try:
            orders = self.client.get_all_orders(symbol=symbol)
            logger.info(f"Retrieved {len(orders)} orders for {symbol}")
            return orders
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error(symbol, "get_all_orders", str(e))
            logger.error(f"Error getting all orders for {symbol}: {e}")
            
            # Handle timestamp errors by re-syncing time and retrying once
            if isinstance(e, BinanceAPIException) and (e.code == -1021 or e.code == -1022):
                logger.info("Timestamp error detected. Re-syncing time and retrying...")
                self._sync_time()
                try:
                    return self.get_all_orders(symbol)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            
            return None
    
    def get_server_time(self):
        """
        Get Binance server time.
        
        Returns:
            datetime: Server time.
        """
        try:
            server_time = self.client.get_server_time()
            server_time_dt = datetime.fromtimestamp(server_time['serverTime'] / 1000)
            logger.info(f"Retrieved server time: {server_time_dt}")
            return server_time_dt
        except (BinanceAPIException, BinanceRequestException) as e:
            log_api_error("SYSTEM", "get_server_time", str(e))
            logger.error(f"Error getting server time: {e}")
            return None

    def is_dust_position(self, symbol, quantity):
        """
        Check if a position is dust (too small to trade).
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            quantity (float): Position quantity
            
        Returns:
            bool: True if the position is dust, False otherwise
        """
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return False  # If we can't determine, assume it's not dust
            
            # Get current price
            current_price = self.get_ticker_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return False
            
            # Get minimum quantity
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                min_qty = float(lot_size_filter['minQty'])
                if quantity < min_qty:
                    logger.info(f"Position is dust: {quantity} {symbol} (below min: {min_qty})")
                    return True
            
            # Get minimum notional value
            order_value = quantity * current_price
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if min_notional_filter:
                min_notional = float(min_notional_filter['minNotional'])
                if order_value < min_notional:
                    logger.info(f"Position is dust: {quantity} {symbol} (value: ${order_value:.2f}, below min: ${min_notional})")
                    return True
            
            # Check fee viability from config
            fee_rate = getattr(config, 'TAKER_FEE_RATE', 0.001)
            min_fee_multiplier = getattr(config, 'MIN_FEE_MULTIPLIER', 5)
            
            estimated_fee = order_value * fee_rate
            min_viable_trade = estimated_fee * min_fee_multiplier
            
            if order_value < min_viable_trade:
                logger.info(f"Position is dust due to fees: {quantity} {symbol} (value: ${order_value:.2f}, fees: ${estimated_fee:.2f}, min viable: ${min_viable_trade:.2f})")
                return True
            
            # If we got here, it's not dust
            return False
        
        except Exception as e:
            log_api_error(symbol, "is_dust_position", str(e), {"quantity": quantity})
            logger.error(f"Error checking if position is dust for {symbol}: {e}")
            return False  # If error, assume not dust to be safe