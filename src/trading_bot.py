"""
Main trading bot implementation that integrates all components.
"""

import os
import sys
import time
from notifier import send_trade_notification
import pandas as pd
from datetime import datetime, timedelta
from logger_utils import log_trade_to_csv

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api import BinanceAPI
from strategies import get_strategy
from risk_manager import RiskManager
from logger import get_logger
from notifier import send_trade_notification, send_status_update
import config

logger = get_logger()

class TradingBot:
    """
    Main trading bot class that integrates API, strategies, and risk management.
    """
    
    def __init__(self, api_key=None, api_secret=None, testnet=None):
        """
        Initialize the trading bot.
        
        Args:
            api_key (str, optional): Binance API key. Defaults to config value.
            api_secret (str, optional): Binance API secret. Defaults to config value.
            testnet (bool, optional): Whether to use testnet. Defaults to config value.
        """
        # Initialize API client
        self.api = BinanceAPI(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet if testnet is not None else config.USE_TESTNET
        )
        
        # Initialize strategy
        self.strategy = get_strategy(config.STRATEGY, self.api)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.api)
        
        # Trading parameters
        self.trading_pairs = config.TRADING_PAIRS
        self.interval = '15m'  # Updated interval for aggressive growth
        self.lookback_period = '2 days ago UTC'  # Shorter lookback for faster signals
        
        # Trading state
        self.active_trades = {}
        self.initial_balance = 0.0
        self.start_time = datetime.now()
        
        logger.info(f"Initialized trading bot with strategy: {self.strategy.name}, "
                   f"trading pairs: {self.trading_pairs}, testnet: {self.api.testnet}")
    
    def initialize(self):
        """
        Initialize the trading bot.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Check API connection
            server_time = self.api.get_server_time()
            if not server_time:
                logger.error("Failed to connect to Binance API")
                return False
            
            logger.info(f"Connected to Binance API. Server time: {server_time}")
            
            # Get initial account balance
            account_value = self.risk_manager.get_account_value()
            if account_value > 0:
                self.initial_balance = account_value
                logger.info(f"Initial account value: ${self.initial_balance:.2f}")
            else:
                logger.warning("Failed to get initial account value")
            
            # Check trading pairs
            for pair in self.trading_pairs:
                symbol_info = self.api.get_symbol_info(pair)
                if not symbol_info:
                    logger.error(f"Invalid trading pair: {pair}")
                    return False
                
                logger.info(f"Validated trading pair: {pair}")
            
            logger.info("Trading bot initialization completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing trading bot: {e}")
            return False
    
    def run_once(self):
        """
        Run a single trading cycle with enhanced debugging.
        
        Returns:
            dict: Trading results.
        """
        results = {}
        
        try:
            # Check if trading is allowed
            if not self.risk_manager.can_trade_today():
                logger.warning("Trading not allowed today due to daily limits")
                return {'status': 'limited', 'message': 'Trading not allowed due to daily limits'}
            
            # Process each trading pair
            for pair in self.trading_pairs:
                logger.info(f"Processing trading pair: {pair}")
                
                # Get current signal
                self.strategy.generate_signals(pair, self.interval, self.lookback_period)
                signal = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                
                # Get current price
                current_price = self.api.get_ticker_price(pair)
                if not current_price:
                    logger.error(f"Failed to get current price for {pair}")
                    continue
                
                logger.info(f"Current signal for {pair}: {signal}, price: {current_price}")
                
                # Check if we have an active trade for this pair
                active_trade = self.active_trades.get(pair)
                
                # DEBUG: Check USDT balance before attempting to trade
                usdt_balance = self.api.get_account_balance('USDT')
                logger.info(f"Available USDT balance before trade decision: {usdt_balance}")
                
                if signal == 'BUY' and not active_trade:
                    logger.info(f"BUY SIGNAL DETECTED - Attempting to execute trade for {pair}")
                    
                    # DEBUG: Check risk parameters
                    entry_price = current_price
                    stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, 'BUY')
                    logger.info(f"Entry price: {entry_price}, Stop loss price: {stop_loss_price}")
                    
                    # DEBUG: Calculate position size separately to see the value
                    position_size = self.risk_manager.calculate_position_size(pair, entry_price, stop_loss_price)
                    logger.info(f"Calculated position size: {position_size} (value: ${position_size * entry_price:.2f})")
                    
                    # Place buy order with risk management
                    logger.info(f"Calling place_order_with_risk_management for {pair}")
                    order_info = self.risk_manager.place_order_with_risk_management(pair, 'BUY', current_price)
                    
                    if order_info:
                        logger.info(f"ORDER INFO RECEIVED: {order_info}")
                        self.active_trades[pair] = order_info
                        results[pair] = {'action': 'BUY', 'price': current_price, 'order': order_info}
                        logger.info(f"Successfully placed BUY order for {pair} at {current_price}")
                        
                        # DEBUG: Verify notification is being sent
                        logger.info(f"Sending trade notification for {pair}")
                        notification_result = send_trade_notification(pair, 'BUY', current_price, order_info['position_size'])
                        logger.info(f"Notification result: {notification_result}")

                        # ✅ Log BUY trade to CSV
                        log_trade_to_csv(pair, "BUY", current_price, order_info['position_size'])
                    else:
                        logger.error(f"Failed to place order - order_info is None for {pair}")
                
                elif signal == 'SELL' and active_trade:
                    logger.info(f"SELL SIGNAL DETECTED - Attempting to close position for {pair}")
                    
                    # Close position
                    if active_trade['side'] == 'BUY':
                        # Cancel existing stop loss and take profit orders
                        if active_trade.get('stop_loss_order'):
                            cancel_result = self.api.cancel_order(pair, active_trade['stop_loss_order']['orderId'])
                            logger.info(f"Stop loss cancellation result: {cancel_result}")
                        
                        if active_trade.get('take_profit_order'):
                            cancel_result = self.api.cancel_order(pair, active_trade['take_profit_order']['orderId'])
                            logger.info(f"Take profit cancellation result: {cancel_result}")
                        
                        # Place sell order
                        logger.info(f"Placing SELL market order for {pair}, size: {active_trade['position_size']}")
                        order = self.api.place_market_order(pair, 'SELL', active_trade['position_size'])
                        
                        if order:
                            profit_loss = (current_price - active_trade['entry_price']) * active_trade['position_size']
                            profit_loss_pct = (current_price / active_trade['entry_price'] - 1) * 100
                            
                            results[pair] = {
                                'action': 'SELL',
                                'price': current_price,
                                'profit_loss': profit_loss,
                                'profit_loss_pct': profit_loss_pct,
                                'order': order
                            }
                            
                            logger.info(f"Closed position for {pair} at {current_price}. PL: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                            
                            # DEBUG: Verify notification is being sent
                            logger.info(f"Sending SELL notification for {pair}")
                            notification_result = send_trade_notification(
                                pair=pair,
                                action='SELL',
                                price=current_price,
                                quantity=active_trade['position_size'],
                                profit_loss=profit_loss,
                                profit_loss_pct=profit_loss_pct
                            )
                            logger.info(f"Notification result: {notification_result}")

                            # ✅ Log SELL trade to CSV
                            log_trade_to_csv(
                                pair,
                                "SELL",
                                current_price,
                                active_trade["position_size"],
                                profit_loss,
                                profit_loss_pct
                            )
                            
                            # Remove from active trades
                            del self.active_trades[pair]
                
                elif signal == 'SELL' and not active_trade:
                    logger.info(f"SELL signal for {pair}, but no active trade exists to close.")
                
                # Check for stop loss or take profit hits for active trades
                elif active_trade and active_trade['side'] == 'BUY':
                    # DEBUG: Log active trade monitoring
                    logger.info(f"Monitoring active trade for {pair} - Entry: {active_trade['entry_price']}, Current: {current_price}")
                    logger.info(f"Stop loss: {active_trade['stop_loss_price']}, Take profit: {active_trade['take_profit_price']}")
                    
                    # Check if stop loss or take profit was hit
                    if current_price <= active_trade['stop_loss_price']:
                        logger.info(f"Stop loss hit for {pair} at {current_price}")
                        
                        # Position should be closed automatically by the stop loss order
                        # Just update our tracking
                        profit_loss = (current_price - active_trade['entry_price']) * active_trade['position_size']
                        profit_loss_pct = ((current_price / active_trade['entry_price']) - 1) * 100
                        
                        results[pair] = {
                            'action': 'STOP_LOSS',
                            'price': current_price,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct
                        }
                        
                        # Remove from active trades
                        del self.active_trades[pair]
                    
                    elif current_price >= active_trade['take_profit_price']:
                        logger.info(f"Take profit hit for {pair} at {current_price}")
                        
                        # Position should be closed automatically by the take profit order
                        # Just update our tracking
                        profit_loss = (current_price - active_trade['entry_price']) * active_trade['position_size']
                        profit_loss_pct = ((current_price / active_trade['entry_price']) - 1) * 100
                        
                        results[pair] = {
                            'action': 'TAKE_PROFIT',
                            'price': current_price,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct
                        }
                        
                        # Remove from active trades
                        del self.active_trades[pair]
            
            # Check for maximum drawdown
            current_balance = self.risk_manager.get_account_value()
            if current_balance > 0 and self.initial_balance > 0:
                if self.risk_manager.check_max_drawdown(self.initial_balance, current_balance):
                    logger.warning("Maximum drawdown exceeded. Stopping trading.")
                    return {'status': 'stopped', 'message': 'Maximum drawdown exceeded'}
            
            return {'status': 'success', 'results': results}
        
        except Exception as e:
            logger.error(f"Error running trading bot: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_continuous(self, interval_seconds=config.DEFAULT_INTERVAL):
        """
        Run the trading bot continuously.
        
        Args:
            interval_seconds (int, optional): Interval between runs in seconds. Defaults to 3600 (1 hour).
        """
        logger.info(f"Starting continuous trading with interval: {interval_seconds} seconds")
        
        if not self.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        # Variables for status updates
        last_status_update = 0
        status_update_interval = config.STATUS_UPDATE_INTERVAL  # Optimized for organic growth
        current_signals = {}
        current_prices = {}
        
        try:
            # Send initial status update when starting
            for pair in self.trading_pairs:
                try:
                    # Get current signal
                    self.strategy.generate_signals(pair, self.interval, self.lookback_period)
                    signal = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                    current_signals[pair] = signal
                    
                    # Get current price
                    price = self.api.get_ticker_price(pair)
                    current_prices[pair] = price
                except Exception as e:
                    logger.error(f"Error getting initial data for {pair}: {e}")
                    current_signals[pair] = "ERROR"
                    current_prices[pair] = "N/A"
            
            # Send initial status update
            send_status_update(self.trading_pairs, current_signals, current_prices)
            last_status_update = time.time()
            logger.info("Sent initial status update to Discord")
            
            while True:
                logger.info(f"Running trading cycle at {datetime.now()}")
                
                # Collect current signals and prices for all trading pairs
                current_signals = {}
                current_prices = {}
                
                for pair in self.trading_pairs:
                    try:
                        # Get current signal and price before running the cycle
                        signal = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                        current_signals[pair] = signal
                        
                        price = self.api.get_ticker_price(pair)
                        current_prices[pair] = price
                    except Exception as e:
                        logger.error(f"Error getting data for {pair}: {e}")
                        current_signals[pair] = "ERROR"
                        current_prices[pair] = "N/A"
                
                # Run the trading cycle
                result = self.run_once()
                logger.info(f"Trading cycle result: {result}")
                
                # Check if it's time for a status update
                current_time = time.time()
                if current_time - last_status_update >= status_update_interval:
                    try:
                        send_status_update(self.trading_pairs, current_signals, current_prices)
                        last_status_update = current_time
                        logger.info("Sent periodic status update to Discord")
                    except Exception as e:
                        logger.error(f"Error sending status update: {e}")
                
                if result.get('status') == 'stopped':
                    logger.warning("Trading stopped due to risk management rules")
                    break
                
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")
        
        finally:
            # Send final status update when stopping
            try:
                send_status_update(self.trading_pairs, current_signals, current_prices)
                logger.info("Sent final status update to Discord")
            except Exception as e:
                logger.error(f"Error sending final status update: {e}")
                
            self.print_summary()
    
    def print_summary(self):
        """
        Print trading summary.
        """
        try:
            # Get final account value
            final_balance = self.risk_manager.get_account_value()
            
            # Calculate performance
            if self.initial_balance > 0 and final_balance > 0:
                profit_loss = final_balance - self.initial_balance
                profit_loss_pct = (final_balance / self.initial_balance - 1) * 100
                
                logger.info("Trading Summary:")
                logger.info(f"  Start time: {self.start_time}")
                logger.info(f"  End time: {datetime.now()}")
                logger.info(f"  Duration: {datetime.now() - self.start_time}")
                logger.info(f"  Initial balance: ${self.initial_balance:.2f}")
                logger.info(f"  Final balance: ${final_balance:.2f}")
                logger.info(f"  Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            else:
                logger.warning("Unable to calculate performance (missing initial or final balance)")
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")