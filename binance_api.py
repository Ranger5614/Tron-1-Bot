"""
Binance API client for the cryptocurrency trading bot.
"""

import os
import time
import math  # Added this import
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from datetime import datetime
import sys

# Add the parent directory to the path to import config and logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from logger import get_logger

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
        
    # Initialize the Binance client with appropriate tld
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
            tld='com'  # Add this parameter - use 'us' for Binance.US or 'com' for Binance.com
        )
        
        logger.info(f"Binance API client initialized (testnet: {self.testnet})")
    
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
            logger.error(f"Error getting account information: {e}")
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
            logger.error(f"Error getting account balance: {e}")
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
            logger.info(f"Retrieved information for symbol {symbol}")
            return symbol_info
        except (BinanceAPIException, BinanceRequestException) as e:
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
            logger.error(f"Error getting ticker price for {symbol}: {e}")
            return None
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Get historical klines (candlestick data).
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            start_str (str): Start time in format 'YYYY-MM-DD' or '1 day ago UTC'.
            end_str (str, optional): End time. Defaults to None (current time).
        
        Returns:
            pandas.DataFrame: Historical klines data.
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            
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
            
            logger.info(f"Retrieved {len(df)} historical klines for {symbol} ({interval})")
            return df
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting historical klines for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, side, quantity):
        """
        Place a market order with enhanced debugging.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            side (str): Order side ('BUY' or 'SELL').
            quantity (float): Order quantity.
        
        Returns:
            dict: Order information.
        """
        try:
            logger.info(f"Preparing to place {side} market order for {quantity} {symbol}")
            
            # Format the quantity according to the symbol's precision requirements
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            # Find the LOT_SIZE filter for quantity precision
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                min_qty = float(lot_size_filter['minQty'])
                
                logger.info(f"LOT_SIZE filter - step_size: {step_size}, min_qty: {min_qty}")
                
                # Ensure quantity meets minimum
                if quantity < min_qty:
                    logger.warning(f"Quantity {quantity} is less than minimum {min_qty}. Setting to minimum.")
                    quantity = min_qty
                
                # Round to step size
                precision = int(round(-math.log10(step_size)))
                quantity = math.floor(quantity * 10**precision) / 10**precision
                
                logger.info(f"Adjusted quantity: {quantity} (rounded to {precision} decimal places)")
            
            # Check if quantity is too small
            if quantity <= 0:
                logger.error(f"Quantity {quantity} is too small to execute")
                return None
            
            # Check account balance for BUY orders
            if side == 'BUY':
                quote_asset = symbol[len(symbol)-4:]
                if quote_asset != 'USDT':
                    quote_asset = 'USDT'  # Default to USDT if unable to determine quote asset
                
                logger.info(f"Checking {quote_asset} balance before BUY")
                balance = self.get_account_balance(quote_asset)
                price = self.get_ticker_price(symbol)
                
                if balance and price:
                    order_value = quantity * price
                    logger.info(f"Order value: {order_value} {quote_asset}, Available balance: {balance} {quote_asset}")
                    
                    if order_value > balance:
                        logger.error(f"Insufficient balance. Need {order_value} {quote_asset}, have {balance} {quote_asset}")
                        return None
            
            # Execute the order
            logger.info(f"Executing {side} market order for {quantity} {symbol}")
            try:
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                logger.info(f"Order executed successfully: {order}")
                logger.info(f"Placed {side} market order for {quantity} {symbol}: {order['orderId']}")
                return order
            except Exception as order_error:
                logger.error(f"Binance API error placing order: {str(order_error)}")
                # Include more details about the error if available
                if hasattr(order_error, 'code') and hasattr(order_error, 'message'):
                    logger.error(f"Error code: {order_error.code}, message: {order_error.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def place_stop_loss_order(self, symbol, quantity, stop_price):
        """
        Place a stop loss order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            quantity (float): Order quantity.
            stop_price (float): Stop price.
        
        Returns:
            dict: Order information.
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=stop_price * 0.99,  # Slightly below stop price to ensure execution
                stopPrice=stop_price
            )
            logger.info(f"Placed stop loss order for {quantity} {symbol} at {stop_price}: {order['orderId']}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error placing stop loss order for {symbol}: {e}")
            return None
    
    def place_take_profit_order(self, symbol, quantity, take_profit_price):
        """
        Place a take profit order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            quantity (float): Order quantity.
            take_profit_price (float): Take profit price.
        
        Returns:
            dict: Order information.
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=take_profit_price
            )
            logger.info(f"Placed take profit order for {quantity} {symbol} at {take_profit_price}: {order['orderId']}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error placing take profit order for {symbol}: {e}")
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
            logger.error(f"Error getting open orders: {e}")
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
            logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
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
            logger.error(f"Error getting order status for {order_id} for {symbol}: {e}")
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
            logger.error(f"Error getting all orders for {symbol}: {e}")
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
            logger.error(f"Error getting server time: {e}")
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
                    for balance in balances if float(balance['free']) > 0
                }
                logger.info("Retrieved non-zero balances")
                return non_zero_balances
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error getting balance: {e}")
            return None