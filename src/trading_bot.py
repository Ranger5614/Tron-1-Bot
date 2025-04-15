"""
Main trading bot implementation that integrates all components.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from src.utils.notifier import send_trade_notification, send_status_update
from src.utils.logger_utils import log_trade_to_csv

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.binance_api import BinanceAPI
from src.strategies.strategies import get_strategy
from src.strategies.risk_manager import RiskManager
from src.utils.logger import get_logger
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
        # Initialize the logger
        self.logger = logger  # Use the global logger
        
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
        
        # Trading parameters - TRON 1.1 optimized settings
        self.trading_pairs = config.TRADING_PAIRS
        self.interval = config.CHART_INTERVAL if hasattr(config, 'CHART_INTERVAL') else '10m'  # TRON 1.1 optimal
        self.lookback_period = config.LOOKBACK_PERIOD if hasattr(config, 'LOOKBACK_PERIOD') else '2 days ago UTC'
        
        # Trading state
        self.active_trades = {}
        self.initial_balance = 0.0
        self.start_time = datetime.now()
        
        # Fee settings
        self.fee_rate = config.FEE_RATE if hasattr(config, 'FEE_RATE') else 0.001  # 0.1% standard fee rate
        self.min_fee_multiplier = config.MIN_FEE_MULTIPLIER if hasattr(config, 'MIN_FEE_MULTIPLIER') else 3
        
        # Profit threshold settings
        self.min_profit_multiplier = config.MIN_PROFIT_MULTIPLIER if hasattr(config, 'MIN_PROFIT_MULTIPLIER') else 2
        
        # TRON 1.1: Min conviction threshold for trades
        self.min_conviction_threshold = config.MIN_CONVICTION_THRESHOLD if hasattr(config, 'MIN_CONVICTION_THRESHOLD') else 0.2
        
        logger.info(f"Initialized trading bot with strategy: {self.strategy.name}, "
                   f"trading pairs: {self.trading_pairs}, testnet: {self.api.testnet}")
        logger.info(f"Using TRON 1.1 optimized settings - interval: {self.interval}, lookback: {self.lookback_period}")
    
    def should_ignore_dust_position(self, pair, actual_balance, current_price):
        """
        Determine if a position is too small (dust) and should be ignored
        
        Args:
            pair (str): The trading pair (e.g., 'BTCUSDT')
            actual_balance (float): The actual balance of the asset
            current_price (float): Current price of the asset
            
        Returns:
            bool: True if this is a dust position that should be ignored
        """
        # Calculate USD value of the position
        usd_value = actual_balance * current_price
        
        # Define minimum value from config if available
        min_value = config.DUST_THRESHOLD if hasattr(config, 'DUST_THRESHOLD') else 10.0
        
        # Return True if position is smaller than minimum value
        if usd_value < min_value:
            self.logger.info(f"Ignoring dust position for {pair}: {actual_balance} (${usd_value:.2f})")
            return True
        
        return False
    
    def check_market_conditions(self, symbol, interval, lookback_period):
        """
        Check market conditions with improved boolean handling.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            lookback_period (str): Lookback period (e.g., '7 days ago UTC').
            
        Returns:
            bool: True if market conditions are favorable, False otherwise.
        """
        try:
            # If the strategy has an is_market_favorable method, use it
            if hasattr(self.strategy, 'is_market_favorable'):
                # Get the result and explicitly convert to boolean to avoid type issues
                result = self.strategy.is_market_favorable(symbol, interval, lookback_period)
                
                # Use direct value checking instead of isinstance
                if result is True:
                    return True
                elif result is False:
                    return False
                else:
                    # Convert to boolean as a fallback
                    bool_result = bool(result)
                    logger.info(f"Converting is_market_favorable result to boolean: {bool_result}")
                    return bool_result
            else:
                # If method doesn't exist, assume conditions are favorable
                logger.info(f"Strategy doesn't have is_market_favorable method, assuming favorable conditions")
                return True
        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {str(e)}")
            return False  # Default to unfavorable if there's an error
    
    def load_current_holdings(self):
        """
        Load current holdings from the Binance account and update active_trades
        to reflect the actual positions rather than relying on internal tracking.
        """
        logger.info("Loading current account holdings...")
        
        try:
            # Get all non-zero balances
            balances = self.api.get_account_balance()
            if not balances:
                logger.error("Failed to get account balances")
                return False
                
            logger.info(f"Current account balances: {balances}")
            
            # For each trading pair, check if we have a position
            for pair in self.trading_pairs:
                # Extract the base asset (e.g., 'BTC' from 'BTCUSDT')
                base_asset = pair[:-4]  # Assumes all pairs end with 'USDT'
                
                # Check if we have a balance for this asset
                if base_asset in balances and balances[base_asset] > 0:
                    # Get current price
                    current_price = self.api.get_ticker_price(pair)
                    if not current_price:
                        logger.error(f"Failed to get current price for {pair}")
                        continue
                    
                    # Create a simulated active trade entry
                    position_size = balances[base_asset]
                    position_value = position_size * current_price
                    
                    # Only consider positions with significant value
                    min_value = config.MIN_ORDER_VALUES.get(pair, config.MIN_ORDER_VALUES['DEFAULT'])
                    if position_value >= min_value:
                        logger.info(f"Found existing position for {pair}: {position_size} {base_asset} (${position_value:.2f})")
                        
                        # Calculate simulated stop loss and take profit levels
                        stop_loss_price = current_price * (1 - config.STOP_LOSS_PERCENTAGE / 100)
                        take_profit_price = current_price * (1 + config.TAKE_PROFIT_PERCENTAGE / 100)
                        
                        # Add to active trades with TRON 1.1 fields
                        self.active_trades[pair] = {
                            'side': 'BUY',
                            'entry_price': current_price,  # We don't know the actual entry price, use current as estimate
                            'position_size': position_size,
                            'stop_loss_price': stop_loss_price,
                            'take_profit_price': take_profit_price,
                            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'imported': True,  # Flag to indicate this was imported from account balance
                            'conviction_multiplier': 1.0,  # Default conviction for imported positions
                            'market_regime': 'unknown'  # Default market regime for imported positions
                        }
            
            logger.info(f"Loaded {len(self.active_trades)} active positions from account")
            return True
            
        except Exception as e:
            logger.error(f"Error loading current holdings: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
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
            
            # Load current holdings from account
            self.load_current_holdings()
            
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
            # Track account growth metrics for risk adjustment
            growth_metrics = self.risk_manager.track_account_growth(self.initial_balance)
            
            # Process each trading pair
            for pair in self.trading_pairs:
                logger.info(f"Processing trading pair: {pair}")
                
                # Extract base asset from pair (e.g., 'BTC' from 'BTCUSDT')
                base_asset = pair[:-4]  # Assumes all pairs end with 'USDT'
                
                # Check market conditions first
                logger.info(f"Checking market conditions for {pair}")
                market_favorable = self.check_market_conditions(pair, self.interval, self.lookback_period)
                
                if not market_favorable:
                    logger.info(f"Market conditions unfavorable for {pair}, skipping")
                    continue
                
                # Get current signal with conviction level if available (TRON 1.1)
                try:
                    # First generate signals
                    self.strategy.generate_signals(pair, self.interval, self.lookback_period)
                    
                    # Check if strategy supports conviction-based sizing (TRON 1.1)
                    if hasattr(self.strategy, 'get_current_signal') and callable(getattr(self.strategy, 'get_current_signal')):
                        # New method might return tuple (signal, conviction)
                        signal_result = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                        
                        # Check if we got a tuple (signal, conviction) or just a signal
                        if isinstance(signal_result, tuple) and len(signal_result) == 2:
                            signal, conviction_multiplier = signal_result
                            logger.info(f"Using conviction-based sizing - signal: {signal}, conviction: {conviction_multiplier}")
                        else:
                            signal = signal_result
                            conviction_multiplier = 1.0
                            logger.info(f"Using standard sizing - signal: {signal}")
                    else:
                        # Fallback for older strategies
                        signal = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                        conviction_multiplier = 1.0
                except Exception as e:
                    logger.error(f"Error getting signal for {pair}: {e}")
                    signal = 'HOLD'
                    conviction_multiplier = 0.0
                
                # Get current price
                current_price = self.api.get_ticker_price(pair)
                if not current_price:
                    logger.error(f"Failed to get current price for {pair}")
                    continue
                
                logger.info(f"Current signal for {pair}: {signal}, price: {current_price}")
                
                # Check if we have an active trade for this pair
                active_trade = self.active_trades.get(pair)
                
                # Also check if we actually hold any of this asset regardless of tracking
                actual_balance = self.api.get_account_balance(base_asset)
                logger.info(f"Actual balance for {base_asset}: {actual_balance}")
                
                # DEBUG: Check USDT balance before attempting to trade
                usdt_balance = self.api.get_account_balance('USDT')
                logger.info(f"Available USDT balance before trade decision: {usdt_balance}")
                
                # Process BUY signal
                if signal == 'BUY' and not active_trade:
                    logger.info(f"BUY SIGNAL DETECTED - Attempting to execute trade for {pair} with conviction {conviction_multiplier}")
                    
                    # Skip if conviction is too low (using config or default)
                    if conviction_multiplier < self.min_conviction_threshold:
                        logger.info(f"Conviction too low ({conviction_multiplier}), skipping trade")
                        continue
                    
                    # DEBUG: Check risk parameters
                    entry_price = current_price
                    stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, 'BUY')
                    logger.info(f"Entry price: {entry_price}, Stop loss price: {stop_loss_price}")
                    
                    # DEBUG: Calculate position size separately to see the value
                    position_size = self.risk_manager.calculate_position_size(
                        pair, entry_price, stop_loss_price, conviction_multiplier
                    )
                    logger.info(f"Calculated position size: {position_size} (value: ${position_size * entry_price:.2f})")
                    
                    # Check if the position value is worth more than fees
                    position_value = position_size * entry_price
                    estimated_fee = position_value * self.fee_rate
                    min_viable_trade = estimated_fee * self.min_fee_multiplier
                    
                    if position_value <= min_viable_trade:
                        logger.warning(f"Position value (${position_value:.2f}) too small compared to fees (est: ${estimated_fee:.2f}). Skipping BUY for {pair}.")
                        continue
                    
                    # Place buy order with risk management and conviction sizing
                    logger.info(f"Calling place_order_with_risk_management for {pair} with conviction {conviction_multiplier}")
                    order_info = self.risk_manager.place_order_with_risk_management(
                        pair, 'BUY', current_price, conviction_multiplier
                    )
                    
                    if order_info:
                        logger.info(f"ORDER INFO RECEIVED: {order_info}")
                        
                        # TRON 1.1: Store conviction multiplier in order_info for future reference
                        order_info['conviction_multiplier'] = conviction_multiplier
                        
                        # Get market regime if available from strategy
                        if hasattr(self.strategy, 'detect_market_regime'):
                            try:
                                # Generate data and get market regime
                                df = self.strategy.generate_signals(pair, self.interval, self.lookback_period)
                                if df is not None and 'market_regime' in df.columns:
                                    market_regime = df['market_regime'].iloc[-1]
                                    order_info['market_regime'] = market_regime
                                    logger.info(f"Market regime for {pair}: {market_regime}")
                            except Exception as e:
                                logger.error(f"Error getting market regime: {e}")
                                order_info['market_regime'] = 'unknown'
                        
                        self.active_trades[pair] = order_info
                        results[pair] = {'action': 'BUY', 'price': current_price, 'order': order_info}
                        logger.info(f"Successfully placed BUY order for {pair} at {current_price}")
                        
                        # DEBUG: Verify notification is being sent
                        logger.info(f"Sending trade notification for {pair}")
                        notification_result = send_trade_notification(
                            pair, 'BUY', current_price, order_info['position_size'], 
                            conviction=conviction_multiplier
                        )
                        logger.info(f"Notification result: {notification_result}")

                        # ✅ Log BUY trade to CSV with conviction
                        log_trade_to_csv(
                            pair, "BUY", current_price, order_info['position_size'], 
                            conviction=conviction_multiplier
                        )
                    else:
                        logger.error(f"Failed to place order - order_info is None for {pair}")
                
                # Process SELL signal - ENHANCED with profit threshold check
                elif signal == 'SELL':
                    # Check if we have an actual balance regardless of active_trades tracking
                    if actual_balance and actual_balance > 0:
                        # Check if this is a dust position that should be ignored
                        if self.should_ignore_dust_position(pair, actual_balance, current_price):
                            logger.warning(f"Skipping dust position for {pair}")
                            continue    

                        logger.info(f"SELL SIGNAL DETECTED - Found {actual_balance} {base_asset} to sell")
                        
                        # Calculate position value
                        position_value = actual_balance * current_price
                        estimated_fee = position_value * self.fee_rate
                        
                        # Get entry details if available
                        if active_trade:
                            logger.info(f"Using tracked trade info for {pair}")
                            position_size = active_trade['position_size']
                            entry_price = active_trade['entry_price']
                            
                            # Get the conviction level if it was stored
                            original_conviction = active_trade.get('conviction_multiplier', 1.0)
                            
                            # Calculate actual profit after all fees
                            entry_value = position_size * entry_price
                            current_value = position_size * current_price
                            buy_fee = entry_value * self.fee_rate
                            sell_fee = current_value * self.fee_rate
                            total_fees = buy_fee + sell_fee
                            gross_profit = current_value - entry_value
                            actual_profit = gross_profit - total_fees
                            profit_percentage = (actual_profit / entry_value) * 100
                            
                            logger.info(f"PROFIT ANALYSIS - Gross: ${gross_profit:.2f}, Fees: ${total_fees:.2f}, Net: ${actual_profit:.2f} ({profit_percentage:.2f}%)")
                            
                            # Only proceed with sell if there's meaningful profit after fees
                            if actual_profit <= 0:
                                logger.warning(f"Trade would result in loss after fees (${actual_profit:.2f}). Skipping SELL for {pair}.")
                                continue
                            elif actual_profit < (total_fees * self.min_profit_multiplier):
                                logger.warning(f"Profit (${actual_profit:.2f}) too small compared to fees (${total_fees:.2f}). Skipping SELL for {pair}.")
                                continue
                            
                            # Cancel existing stop loss and take profit orders if they exist
                            if active_trade.get('stop_loss_order'):
                                cancel_result = self.api.cancel_order(pair, active_trade['stop_loss_order']['orderId'])
                                logger.info(f"Stop loss cancellation result: {cancel_result}")
                            
                            if active_trade.get('take_profit_order'):
                                cancel_result = self.api.cancel_order(pair, active_trade['take_profit_order']['orderId'])
                                logger.info(f"Take profit cancellation result: {cancel_result}")
                        else:
                            # No active trade tracking, use actual balance and current values
                            logger.info(f"No tracked trade found for {pair}, using actual balance")
                            position_size = actual_balance
                            original_conviction = 1.0
                            
                            # Since we don't know the entry price, we have to estimate based on position value
                            # We'll use a minimum viable trade check instead of profit check
                            min_viable_trade = estimated_fee * self.min_fee_multiplier
                            if position_value <= min_viable_trade:
                                logger.warning(f"Position value (${position_value:.2f}) too small compared to fees (est: ${estimated_fee:.2f}). Skipping SELL for {pair}.")
                                continue
                            
                            # Set entry price to current for reporting
                            entry_price = current_price
                        
                        # Place sell order
                        logger.info(f"Placing SELL market order for {pair}, size: {position_size}")
                        order = self.api.place_market_order(pair, 'SELL', position_size)
                        
                        if order:
                            profit_loss = (current_price - entry_price) * position_size
                            profit_loss_pct = (current_price / entry_price - 1) * 100
                            
                            results[pair] = {
                                'action': 'SELL',
                                'price': current_price,
                                'profit_loss': profit_loss,
                                'profit_loss_pct': profit_loss_pct,
                                'order': order,
                                'conviction_multiplier': original_conviction
                            }
                            
                            logger.info(f"Closed position for {pair} at {current_price}. PL: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                            
                            # DEBUG: Verify notification is being sent
                            logger.info(f"Sending SELL notification for {pair}")
                            notification_result = send_trade_notification(
                                pair=pair,
                                action='SELL',
                                price=current_price,
                                quantity=position_size,
                                profit_loss=profit_loss,
                                profit_loss_pct=profit_loss_pct,
                                conviction=original_conviction
                            )
                            logger.info(f"Notification result: {notification_result}")

                            # ✅ Log SELL trade to CSV with conviction
                            log_trade_to_csv(
                                pair,
                                "SELL",
                                current_price,
                                position_size,
                                profit_loss,
                                profit_loss_pct,
                                conviction=original_conviction
                            )
                            
                            # Remove from active trades if it exists
                            if pair in self.active_trades:
                                del self.active_trades[pair]
                        else:
                            logger.error(f"Failed to place SELL order for {pair}")
                    else:
                        logger.info(f"SELL signal for {pair}, but no actual balance found to sell.")
                
                # Process stop loss / take profit checks for active trades
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
            drawdown = self.risk_manager.calculate_drawdown(current_balance)
            logger.info(f"Current drawdown: {drawdown:.2f}% (limit: {self.risk_manager.max_drawdown_pct}%)")
            
            return {'status': 'success', 'results': results}
        
        except Exception as e:
            logger.error(f"Error running trading bot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def run_continuous(self, interval_seconds=config.DEFAULT_INTERVAL):
        """
        Run the trading bot continuously.

        Args:
            interval_seconds (int, optional): Interval between runs in seconds. Defaults to config.DEFAULT_INTERVAL.
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
        current_convictions = {}  # TRON 1.1: Track conviction levels for status updates

        # Add cycle counter for tracking and notifications
        cycle_counter = 0

        try:
            # Send initial status update when starting
            for pair in self.trading_pairs:
                try:
                    # Get current signal
                    self.strategy.generate_signals(pair, self.interval, self.lookback_period)
                    signal_result = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                    
                    # Extract signal and conviction if it returns a tuple (TRON 1.1)
                    if isinstance(signal_result, tuple) and len(signal_result) == 2:
                        signal, conviction = signal_result[0], signal_result[1]
                        current_convictions[pair] = conviction
                    else:
                        signal = signal_result
                        current_convictions[pair] = 1.0
                        
                    current_signals[pair] = signal

                    # Get current price
                    price = self.api.get_ticker_price(pair)
                    current_prices[pair] = price
                except Exception as e:
                    logger.error(f"Error getting initial data for {pair}: {e}")
                    current_signals[pair] = "ERROR"
                    current_prices[pair] = "N/A"
                    current_convictions[pair] = 0.0

            # Send initial status update
            # TRON 1.1: Enhanced status update to include conviction levels
            if hasattr(send_status_update, "__code__") and "current_convictions" in send_status_update.__code__.co_varnames:
                # New version of status update accepts conviction
                send_status_update(self.trading_pairs, current_signals, current_prices, current_convictions)
            else:
                # Backward compatibility
                send_status_update(self.trading_pairs, current_signals, current_prices)
                
            last_status_update = time.time()
            logger.info("Sent initial status update to Discord")

            while True:
                # Increment cycle counter
                cycle_counter += 1

                logger.info(f"Running trading cycle #{cycle_counter} at {datetime.now()}")

                # Send cycle update notification
                try:
                    from notifier import send_cycle_update
                    send_cycle_update(self.trading_pairs, cycle_counter)
                    logger.info(f"Sent notification for cycle #{cycle_counter}")
                except Exception as e:
                    logger.error(f"Error sending cycle update: {e}")

                # Collect current signals and prices for all trading pairs
                current_signals = {}
                current_prices = {}
                current_convictions = {}  # TRON 1.1: Track conviction levels

                for pair in self.trading_pairs:
                    try:
                        # Get current signal and price before running the cycle
                        signal_result = self.strategy.get_current_signal(pair, self.interval, self.lookback_period)
                        
                        # Extract signal and conviction if it returns a tuple (TRON 1.1)
                        if isinstance(signal_result, tuple) and len(signal_result) == 2:
                            signal, conviction = signal_result[0], signal_result[1]
                            current_convictions[pair] = conviction
                        else:
                            signal = signal_result
                            current_convictions[pair] = 1.0
                            
                        current_signals[pair] = signal

                        price = self.api.get_ticker_price(pair)
                        current_prices[pair] = price
                    except Exception as e:
                        logger.error(f"Error getting data for {pair}: {e}")
                        current_signals[pair] = "ERROR"
                        current_prices[pair] = "N/A"
                        current_convictions[pair] = 0.0

                # Run the trading cycle
                result = self.run_once()
                logger.info(f"Trading cycle result: {result}")

                # Check if it's time for a status update
                current_time = time.time()
                if current_time - last_status_update >= status_update_interval:
                    try:
                        # TRON 1.1: Enhanced status update to include conviction levels
                        if hasattr(send_status_update, "__code__") and "current_convictions" in send_status_update.__code__.co_varnames:
                            # New version of status update accepts conviction
                            send_status_update(self.trading_pairs, current_signals, current_prices, current_convictions)
                        else:
                            # Backward compatibility
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
                # TRON 1.1: Enhanced status update to include conviction levels
                if hasattr(send_status_update, "__code__") and "current_convictions" in send_status_update.__code__.co_varnames:
                    # New version of status update accepts conviction
                    send_status_update(self.trading_pairs, current_signals, current_prices, current_convictions)
                else:
                    # Backward compatibility
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
            
    def force_sell_all_positions(self):
        """
        Force sell all cryptocurrency positions in the account.
        Useful for emergency liquidation or testing.
        
        Returns:
            dict: Results of all sell operations
        """
        results = {}
        logger.info("EMERGENCY SELL - Selling all positions")
        
        try:
            # Get all non-zero balances
            balances = self.api.get_account_balance()
            if not balances:
                logger.error("Failed to get account balances")
                return {'status': 'error', 'message': 'Failed to get account balances'}
                
            logger.info(f"Current account balances: {balances}")
            
            # For each trading pair, check if we have a position and sell it
            for pair in self.trading_pairs:
                # Extract the base asset (e.g., 'BTC' from 'BTCUSDT')
                base_asset = pair[:-4]  # Assumes all pairs end with 'USDT'
                
                # Check if we have a balance for this asset
                if base_asset in balances and balances[base_asset] > 0:
                    position_size = balances[base_asset]
                    
                    # Get current price
                    current_price = self.api.get_ticker_price(pair)
                    if not current_price:
                        logger.error(f"Failed to get current price for {pair}")
                        results[pair] = {'status': 'error', 'message': 'Failed to get price'}
                        continue
                    
                    # Calculate position value and check if it's worth more than fees
                    position_value = position_size * current_price
                    estimated_fee = position_value * self.fee_rate
                    min_viable_trade = estimated_fee * self.min_fee_multiplier
                    
                    if position_value <= min_viable_trade:
                        logger.warning(f"Position value (${position_value:.2f}) too small compared to fees (est: ${estimated_fee:.2f}). Skipping emergency sell for {pair}.")
                        results[pair] = {'status': 'skipped', 'message': 'Position too small compared to fees'}
                        continue
                    
                    # Place sell order
                    logger.info(f"EMERGENCY SELL: Placing SELL market order for {pair}, size: {position_size}")
                    order = self.api.place_market_order(pair, 'SELL', position_size)
                    
                    if order:
                        logger.info(f"Successfully sold {position_size} {base_asset} at {current_price}")
                        results[pair] = {
                            'status': 'success',
                            'action': 'EMERGENCY_SELL',
                            'price': current_price,
                            'quantity': position_size,
                            'order': order
                        }
                        
                        # Remove from active trades if it exists
                        if pair in self.active_trades:
                            del self.active_trades[pair]
                    else:
                        logger.error(f"Failed to place EMERGENCY SELL order for {pair}")
                        results[pair] = {'status': 'error', 'message': 'Failed to place order'}
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"Error in emergency sell: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}