import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.binance_api import BinanceAPI
from src.utils.logger import get_logger
from src.utils.logger_utils import log_latest_scan
from src.utils.notifier import send_trade_notification
import config

logger = get_logger()

class Strategy:
   """
   Base strategy class.
   """
   
   def __init__(self, api_client=None):
       """
       Initialize the strategy.
       
       Args:
           api_client (BinanceAPI, optional): Binance API client. Defaults to None.
       """
       self.api = api_client or BinanceAPI()
       self.name = "Base Strategy"
   
   def generate_signals(self, symbol, interval, lookback_period):
       """
       Generate trading signals.
       
       Args:
           symbol (str): Trading symbol (e.g., 'BTCUSDT').
           interval (str): Kline interval (e.g., '1h', '4h', '1d').
           lookback_period (str): Lookback period (e.g., '7 days ago UTC').
       
       Returns:
           pandas.DataFrame: DataFrame with signals.
       """
       raise NotImplementedError("Subclasses must implement generate_signals method")
   
   def get_current_signal(self, symbol, interval, lookback_period):
       """
       Get the current trading signal.
       
       Args:
           symbol (str): Trading symbol (e.g., 'BTCUSDT').
           interval (str): Kline interval (e.g., '1h', '4h', '1d').
           lookback_period (str): Lookback period (e.g., '7 days ago UTC').
       
       Returns:
           str: Trading signal ('BUY', 'SELL', or 'HOLD').
       """
       df = self.generate_signals(symbol, interval, lookback_period)
       if df is None or len(df) == 0:
           logger.error(f"No data available for {symbol}")
           return 'HOLD'
       
       # Get the last signal
       last_signal = df.iloc[-1]['signal']
       
       # Log the latest scan to a file for analysis
       self.log_to_file(symbol, df, interval)
       
       return last_signal
   
   def log_to_file(self, symbol, df, interval):
       """
       Log the latest scan results to a text file.
       
       Args:
           symbol (str): Trading symbol
           df (pandas.DataFrame): DataFrame with signals
           interval (str): Kline interval
       """
       try:
           # First check if df is valid
           if df is None or len(df) == 0:
               logger.error(f"Cannot log scan for {symbol}: DataFrame is empty")
               # Log this error to the scan file
               log_latest_scan(
                   symbol=symbol,
                   signal="ERROR",
                   price=0.0,
                   indicators={"error": "No data available"},
                   strategy=self.name,
                   interval=interval
               )
               return
           
           # Debug info
           logger.info(f"Logging scan for {symbol} with {len(df)} data points")
           logger.info(f"DataFrame columns: {df.columns.tolist()}")
           
           # Get the last row
           last_row = df.iloc[-1]
           
           # Debug output
           logger.info(f"Last row index: {last_row.name}")
           logger.info(f"Signal: {last_row.get('signal', 'N/A')}")
           
           # Extract relevant indicators based on strategy type
           indicators = {}
           
           # Add common fields
           if 'close' in last_row:
               indicators['close'] = last_row['close']
           else:
               logger.warning(f"'close' not in DataFrame for {symbol}")
               
           if 'volume' in last_row:
               indicators['volume'] = last_row['volume']
               
           # Add strategy-specific indicators
           if 'rsi' in last_row:
               indicators['rsi'] = last_row['rsi']
               
           if 'short_ma' in last_row and 'long_ma' in last_row:
               indicators['short_ma'] = last_row['short_ma']
               indicators['long_ma'] = last_row['long_ma']
               indicators['ma_diff'] = last_row['short_ma'] - last_row['long_ma']
           
           # For combined strategy
           if 'sma_signal' in last_row and 'rsi_signal' in last_row:
               indicators['sma_signal'] = last_row['sma_signal']
               indicators['rsi_signal'] = last_row['rsi_signal']
               
           # Log the current price for reference
           current_price = last_row.get('close', 0.0)
           
           # Get the signal
           signal = last_row.get('signal', 'UNKNOWN')
           
           # Debug output
           logger.info(f"About to call log_latest_scan for {symbol} with signal {signal}")
           
           # Log to file using the logger_utils function
           log_latest_scan(
               symbol=symbol,
               signal=signal,
               price=current_price,
               indicators=indicators,
               strategy=self.name,
               interval=interval
           )
           
           logger.info(f"Successfully logged scan for {symbol}")
           
       except Exception as e:
           import traceback
           error_msg = f"Error logging scan to file: {str(e)}\n{traceback.format_exc()}"
           logger.error(error_msg)
           
           # Try to log the error
           try:
               log_latest_scan(
                   symbol=symbol,
                   signal="ERROR",
                   price=0.0,
                   indicators={"error": str(e)},
                   strategy=self.name,
                   interval=interval
               )
           except:
               pass


class SimpleMovingAverageStrategy(Strategy):
   """
   Simple Moving Average (SMA) crossover strategy.
   
   This strategy generates buy signals when the short-term moving average crosses above
   the long-term moving average, and sell signals when the short-term moving average
   crosses below the long-term moving average.
   """
   
   def __init__(self, api_client=None, short_window=None, long_window=None):
       """
       Initialize the SMA strategy.
       
       Args:
           api_client (BinanceAPI, optional): Binance API client. Defaults to None.
           short_window (int, optional): Short moving average window. Defaults to config value.
           long_window (int, optional): Long moving average window. Defaults to config value.
       """
       super().__init__(api_client)
       self.name = "Simple Moving Average"
       self.short_window = short_window or config.SHORT_WINDOW
       self.long_window = long_window or config.LONG_WINDOW
       
       logger.info(f"Initialized SMA strategy with short_window={self.short_window}, long_window={self.long_window}")
   
   def generate_signals(self, symbol, interval, lookback_period):
       """
       Generate trading signals based on SMA crossover.
       
       Args:
           symbol (str): Trading symbol (e.g., 'BTCUSDT').
           interval (str): Kline interval (e.g., '1h', '4h', '1d').
           lookback_period (str): Lookback period (e.g., '7 days ago UTC').
       
       Returns:
           pandas.DataFrame: DataFrame with signals.
       """
       # Get historical klines
       df = self.api.get_historical_klines(symbol, interval, lookback_period)
       if df is None or len(df) == 0:
           logger.error(f"No data available for {symbol}")
           return None
       
       # Calculate moving averages
       df['short_ma'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
       df['long_ma'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()
       
       # Initialize signals
       df['signal'] = 'HOLD'
       
       # ENHANCED: Generate both BUY and SELL signals based on crossover
       df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 'BUY'
       df.loc[df['short_ma'] < df['long_ma'], 'signal'] = 'SELL'
       
       # Generate signal changes
       df['prev_signal'] = df['signal'].shift(1)
       df['signal_change'] = (df['signal'] != df['prev_signal']).astype(int)
       
       # Log signal changes
       signal_changes = df[df['signal_change'] == 1]
       for idx, row in signal_changes.iterrows():
           logger.info(f"Signal change at {idx}: {row['prev_signal']} -> {row['signal']} (price: {row['close']})")
       
       return df


class RSIStrategy(Strategy):
   """
   Relative Strength Index (RSI) strategy.
   
   This strategy generates buy signals when the RSI falls below the oversold threshold
   and sell signals when the RSI rises above the overbought threshold.
   """
   
   def __init__(self, api_client=None, period=None, overbought=None, oversold=None):
       """
       Initialize the RSI strategy.
       
       Args:
           api_client (BinanceAPI, optional): Binance API client. Defaults to None.
           period (int, optional): RSI period. Defaults to config value.
           overbought (int, optional): Overbought threshold. Defaults to config value.
           oversold (int, optional): Oversold threshold. Defaults to config value.
       """
       super().__init__(api_client)
       self.name = "Relative Strength Index"
       self.period = period or config.RSI_PERIOD
       self.overbought = overbought or config.RSI_OVERBOUGHT
       self.oversold = oversold or config.RSI_OVERSOLD
       
       logger.info(f"Initialized RSI strategy with period={self.period}, overbought={self.overbought}, oversold={self.oversold}")
   
   def calculate_rsi(self, data, window=14):
       """
       Calculate Relative Strength Index.
       
       Args:
           data (pandas.Series): Price data.
           window (int, optional): RSI period. Defaults to 14.
       
       Returns:
           pandas.Series: RSI values.
       """
       # Calculate price changes
       delta = data.diff()
       
       # Separate gains and losses
       gain = delta.where(delta > 0, 0)
       loss = -delta.where(delta < 0, 0)
       
       # Calculate average gain and loss
       avg_gain = gain.rolling(window=window, min_periods=1).mean()
       avg_loss = loss.rolling(window=window, min_periods=1).mean()
       
       # Calculate RS
       rs = avg_gain / avg_loss
       
       # Calculate RSI
       rsi = 100 - (100 / (1 + rs))
       
       return rsi
   
   def generate_signals(self, symbol, interval, lookback_period):
       """
       Generate trading signals based on RSI.
       
       Args:
           symbol (str): Trading symbol (e.g., 'BTCUSDT').
           interval (str): Kline interval (e.g., '1h', '4h', '1d').
           lookback_period (str): Lookback period (e.g., '7 days ago UTC').
       
       Returns:
           pandas.DataFrame: DataFrame with signals.
       """
       # Get historical klines
       df = self.api.get_historical_klines(symbol, interval, lookback_period)
       if df is None or len(df) == 0:
           logger.error(f"No data available for {symbol}")
           return None
       
       # Calculate RSI
       df['rsi'] = self.calculate_rsi(df['close'], window=self.period)
       
       # Initialize signals
       df['signal'] = 'HOLD'
       
       # Generate signals based on RSI thresholds
       df.loc[df['rsi'] < self.oversold, 'signal'] = 'BUY'
       df.loc[df['rsi'] > self.overbought, 'signal'] = 'SELL'
       
       # ENHANCED: Add trend following logic for more active signals
       # If RSI is trending up from the middle range, consider it a BUY
       df['rsi_change'] = df['rsi'].diff(3)  # 3-period change in RSI
       middle_threshold = (self.overbought + self.oversold) / 2
       
       # If RSI is below middle but rising rapidly, consider BUY
       rising_condition = (df['rsi'] > self.oversold) & (df['rsi'] < middle_threshold) & (df['rsi_change'] > 10)
       df.loc[rising_condition, 'signal'] = 'BUY'
       
       # If RSI is above middle but falling rapidly, consider SELL
       falling_condition = (df['rsi'] < self.overbought) & (df['rsi'] > middle_threshold) & (df['rsi_change'] < -10)
       df.loc[falling_condition, 'signal'] = 'SELL'
       
       # Generate signal changes
       df['prev_signal'] = df['signal'].shift(1)
       df['signal_change'] = (df['signal'] != df['prev_signal']).astype(int)
       
       # Log signal changes
       signal_changes = df[df['signal_change'] == 1]
       for idx, row in signal_changes.iterrows():
           if 'rsi_change' in row and abs(row['rsi_change']) > 10:
               logger.info(f"Signal change at {idx}: {row['prev_signal']} -> {row['signal']} (RSI: {row['rsi']:.2f}, RSI change: {row['rsi_change']:.2f}, price: {row['close']})")
           else:
               logger.info(f"Signal change at {idx}: {row['prev_signal']} -> {row['signal']} (RSI: {row['rsi']:.2f}, price: {row['close']})")
       
       return df


class CombinedStrategy(Strategy):
   """
   Combined strategy using both SMA and RSI.
   
   ENHANCED: This strategy is now less restrictive, requiring only one condition to be met for signals
   while using the other indicator for confirmation.
   """
   
   def __init__(self, api_client=None):
       """
       Initialize the combined strategy.
       
       Args:
           api_client (BinanceAPI, optional): Binance API client. Defaults to None.
       """
       super().__init__(api_client)
       self.name = "Combined SMA and RSI"
       self.sma_strategy = SimpleMovingAverageStrategy(api_client)
       self.rsi_strategy = RSIStrategy(api_client)
       
       logger.info("Initialized Combined strategy with SMA and RSI")
   
   def generate_signals(self, symbol, interval, lookback_period):
       """
       Generate trading signals based on combined SMA and RSI strategies.
       
       Args:
           symbol (str): Trading symbol (e.g., 'BTCUSDT').
           interval (str): Kline interval (e.g., '1h', '4h', '1d').
           lookback_period (str): Lookback period (e.g., '7 days ago UTC').
       
       Returns:
           pandas.DataFrame: DataFrame with signals.
       """
       # Get signals from individual strategies
       sma_df = self.sma_strategy.generate_signals(symbol, interval, lookback_period)
       rsi_df = self.rsi_strategy.generate_signals(symbol, interval, lookback_period)
       
       if sma_df is None or rsi_df is None:
           logger.error(f"No data available for {symbol}")
           return None
       
       # Combine signals
       df = sma_df.copy()
       df['sma_signal'] = sma_df['signal']
       df['rsi_signal'] = rsi_df['signal']
       df['rsi'] = rsi_df['rsi']
       
       # Initialize signals
       df['signal'] = 'HOLD'
       
       # ENHANCED: More active signal generation - less restrictive
       # BUY if either SMA says BUY and RSI is not overbought, or RSI says BUY and SMA is not bearish
       buy_condition = ((df['sma_signal'] == 'BUY') & (df['rsi'] < 70)) | \
                       ((df['rsi_signal'] == 'BUY') & (df['sma_signal'] != 'SELL'))
       
       # SELL if either SMA says SELL and RSI is not oversold, or RSI says SELL and SMA is not bullish
       sell_condition = ((df['sma_signal'] == 'SELL') & (df['rsi'] > 30)) | \
                        ((df['rsi_signal'] == 'SELL') & (df['sma_signal'] != 'BUY'))
       
       df.loc[buy_condition, 'signal'] = 'BUY'
       df.loc[sell_condition, 'signal'] = 'SELL'
       
       # Generate signal changes
       df['prev_signal'] = df['signal'].shift(1)
       df['signal_change'] = (df['signal'] != df['prev_signal']).astype(int)
       
       # Log signal changes
       signal_changes = df[df['signal_change'] == 1]
       for idx, row in signal_changes.iterrows():
           logger.info(f"Signal change at {idx}: {row['prev_signal']} -> {row['signal']} (SMA: {row['sma_signal']}, RSI: {row['rsi_signal']} at {row['rsi']:.2f}, price: {row['close']})")
       
       return df


class SmallAccountStrategyV1_1(Strategy):
    """
    TRON 1.1 - Upgraded Small Account Growth Strategy.
    
    This strategy is optimized for more aggressive trading with reasonable risk controls.
    It combines momentum, volatility assessment, trend direction, and pullback detection.
    """
    
    def __init__(self, api_client=None):
        """
        Initialize the upgraded Small Account Growth strategy.
        """
        super().__init__(api_client)
        self.name = "TRON 1.1 - Aggressive Growth"
        
        # Use optimized parameters from config
        self.rsi_period = config.TRON_RSI_PERIOD
        self.rsi_overbought = config.TRON_RSI_OVERBOUGHT
        self.rsi_oversold = config.TRON_RSI_OVERSOLD
        self.short_ma_period = config.TRON_SHORT_MA
        self.long_ma_period = config.TRON_LONG_MA
        self.macd_fast_period = 12
        self.macd_slow_period = 26
        self.macd_signal_period = config.TRON_MACD_SIGNAL
        
        # Pullback detection parameters
        self.pullback_threshold = 3.0  # Minimum pullback percentage
        self.volume_spike_threshold = 1.5  # Volume > 150% of average
        
        # Market regime detection
        self.regime_ma_short = 20
        self.regime_ma_long = 50
        
        # Minimum conviction threshold from config
        self.min_conviction_threshold = config.MIN_CONVICTION_THRESHOLD if hasattr(config, 'MIN_CONVICTION_THRESHOLD') else 0.65
        
        logger.info(f"Initialized TRON 1.1 Strategy with optimized parameters from config")
        logger.info(f"Using RSI parameters: period={self.rsi_period}, overbought={self.rsi_overbought}, oversold={self.rsi_oversold}")
        logger.info(f"Using MA parameters: short={self.short_ma_period}, long={self.long_ma_period}")
        logger.info(f"Using minimum conviction threshold: {self.min_conviction_threshold}")
    
    def calculate_rsi(self, data, window=14):
        """
        Calculate Relative Strength Index.
        
        Args:
            data (pandas.Series): Price data.
            window (int, optional): RSI period. Defaults to 14.
        
        Returns:
            pandas.Series: RSI values.
        """
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        """
        # Calculate EMAs
        ema_fast = data.ewm(span=fast_period, min_periods=fast_period).mean()
        ema_slow = data.ewm(span=slow_period, min_periods=slow_period).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def detect_pullbacks(self, df):
        """
        Detect price pullbacks in an uptrend.
        
        Args:
            df (pandas.DataFrame): Price data with indicators.
            
        Returns:
            pandas.Series: Boolean series with True for pullback points.
        """
        # Calculate recent highs (rolling 5-period high)
        df['recent_high'] = df['close'].rolling(5).max()
        
        # Calculate pullback percentage
        df['pullback_pct'] = (df['recent_high'] - df['close']) / df['recent_high'] * 100
        
        # Detect uptrend - price above long MA and short MA above long MA
        uptrend = (df['close'] > df['long_ma']) & (df['short_ma'] > df['long_ma'])
        
        # Pullback condition - pullback percentage exceeds threshold in an uptrend
        pullback = uptrend & (df['pullback_pct'] > self.pullback_threshold) & (df['pullback_pct'] < 10)
        
        return pullback
    
    def detect_volume_spikes(self, df):
        """
        Detect unusual volume spikes.
        
        Args:
            df (pandas.DataFrame): Price data with volume.
            
        Returns:
            pandas.Series: Boolean series with True for volume spike points.
        """
        # Calculate the average volume (5-period)
        df['avg_volume'] = df['volume'].rolling(5).mean()
        
        # Volume spike condition - volume exceeds threshold of average
        volume_spike = df['volume'] > (df['avg_volume'] * self.volume_spike_threshold)
        
        return volume_spike
    
    def detect_market_regime(self, df):
        """
        Detect the market regime (bull, bear, sideways).
        
        Args:
            df (pandas.DataFrame): Price data.
            
        Returns:
            pandas.Series: Market regime ('bull', 'bear', or 'sideways').
        """
        # Calculate medium and long-term moving averages
        df['regime_ma_short'] = df['close'].rolling(window=self.regime_ma_short, min_periods=1).mean()
        df['regime_ma_long'] = df['close'].rolling(window=self.regime_ma_long, min_periods=1).mean()
        
        # Initialize regime
        df['market_regime'] = 'sideways'
        
        # Bull market: price > short MA > long MA
        bull_condition = (df['close'] > df['regime_ma_short']) & (df['regime_ma_short'] > df['regime_ma_long'])
        df.loc[bull_condition, 'market_regime'] = 'bull'
        
        # Bear market: price < short MA < long MA
        bear_condition = (df['close'] < df['regime_ma_short']) & (df['regime_ma_short'] < df['regime_ma_long'])
        df.loc[bear_condition, 'market_regime'] = 'bear'
        
        return df['market_regime']
    
    def generate_signals(self, symbol, interval, lookback_period):
        """
        Generate trading signals based on the upgraded strategy.
        """
        # Get historical klines
        df = self.api.get_historical_klines(symbol, interval, lookback_period)
        if df is None or len(df) == 0:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Calculate indicators
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], window=self.rsi_period)
        
        # Moving Averages
        df['short_ma'] = df['close'].rolling(window=self.short_ma_period, min_periods=1).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_ma_period, min_periods=1).mean()
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(
            df['close'], 
            fast_period=self.macd_fast_period, 
            slow_period=self.macd_slow_period, 
            signal_period=self.macd_signal_period
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        # Add pullback detection
        df['pullback'] = self.detect_pullbacks(df)
        
        # Add volume spike detection
        df['volume_spike'] = self.detect_volume_spikes(df)
        
        # Add market regime detection
        df['market_regime'] = self.detect_market_regime(df)
        
        # Add volatility calculation
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * 100  # 20-day volatility in percentage
        
        # Initialize signals
        df['signal'] = 'HOLD'
        # Initialize signal_strength as float to avoid the FutureWarning
        df['signal_strength'] = 0.0
        
        # Generate buy signals with different conviction levels
        
        # Strong buy: Classic conditions with increased RSI threshold
        strong_buy = (
            (df['rsi'] < self.rsi_oversold) &     # Oversold condition (RSI < config threshold)
            (df['close'] > df['short_ma']) &      # Price above short MA
            (df['macd_hist'] > 0) &               # MACD histogram positive
            (df['market_regime'] != 'bear')       # Not in a bear market
        )
        # Fix: Initialize signal_strength as float and use appropriate method to set values
        df.loc[strong_buy, 'signal'] = 'BUY'
        # Using float values directly instead of integers to avoid FutureWarning
        df.loc[strong_buy, 'signal_strength'] = 0.9  # High conviction
        
        # Buy on pullbacks in uptrend with volume confirmation
        pullback_buy = (
            df['pullback'] & 
            df['volume_spike'] &
            (df['market_regime'] == 'bull')
        )
        df.loc[pullback_buy, 'signal'] = 'BUY'
        df.loc[pullback_buy, 'signal_strength'] = 0.8  # Medium-high conviction
        
        # Buy on RSI bounce from oversold
        rsi_bounce_buy = (
            (df['rsi'] < 45) &                    # Near oversold
            (df['rsi'].diff(3) > 5) &             # RSI rising
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD histogram increasing
            (df['market_regime'] != 'bear')       # Not in a bear market
        )
        df.loc[rsi_bounce_buy, 'signal'] = 'BUY'
        df.loc[rsi_bounce_buy, 'signal_strength'] = 0.7  # Medium conviction
        
        # Generate sell signals
        
        # Strong sell: RSI overbought or negative momentum
        strong_sell_1 = (df['rsi'] > self.rsi_overbought)  # Overbought condition (RSI > config threshold)
        
        strong_sell_2 = (
            (df['close'] < df['short_ma']) &   # Price below short MA
            (df['macd_hist'] < 0) &            # MACD histogram negative
            (df['market_regime'] != 'bull')    # Not in a bull market
        )
        
        # Combined sell conditions
        df.loc[strong_sell_1 | strong_sell_2, 'signal'] = 'SELL'
        df.loc[strong_sell_1 | strong_sell_2, 'signal_strength'] = 0.8  # Medium-high conviction for sells
        
        # Overriding rules
        
        # If in a clear bear market, be more cautious with buys
        bear_market = (df['market_regime'] == 'bear') & (df['close'] < df['regime_ma_long'])
        df.loc[bear_market & (df['signal'] == 'BUY') & (df['signal_strength'] < 0.9), 'signal'] = 'HOLD'
        
        # Check market volatility limits from config
        if hasattr(config, 'MARKET_VOLATILITY_MIN') and hasattr(config, 'MARKET_VOLATILITY_MAX'):
            # Calculate 20-period volatility
            volatility = df['returns'].iloc[-20:].std() * 100  # Convert to percentage
            
            # If volatility is outside acceptable range, override signals
            if volatility < config.MARKET_VOLATILITY_MIN or volatility > config.MARKET_VOLATILITY_MAX:
                logger.info(f"Volatility {volatility:.2f}% outside acceptable range ({config.MARKET_VOLATILITY_MIN}% - {config.MARKET_VOLATILITY_MAX}%)")
                # Reduce signal strength for BUY signals - multiply by 0.5
                df.loc[df['signal'] == 'BUY', 'signal_strength'] *= 0.5
        
        # Apply minimum conviction threshold - signals below threshold become HOLD
        if self.min_conviction_threshold > 0:
            low_conviction = (df['signal_strength'] < self.min_conviction_threshold)
            df.loc[low_conviction, 'signal'] = 'HOLD'
            df.loc[low_conviction, 'signal_strength'] = 0.0
        
        # Log signals with conviction levels
        for idx, row in df[df['signal'] != 'HOLD'].iterrows():
            conditions = []
            
            if row['signal'] == 'BUY':
                if row['rsi'] < self.rsi_oversold:
                    conditions.append(f"RSI={row['rsi']:.1f}")
                if row['pullback']:
                    conditions.append(f"Pullback={row['pullback_pct']:.1f}%")
                if row['volume_spike']:
                    conditions.append(f"VolSpike={row['volume']/row['avg_volume']:.1f}x")
                if row['macd_hist'] > 0:
                    conditions.append(f"MACD+")
                    
                logger.info(f"BUY signal at {idx} - price: {row['close']} - strength: {row['signal_strength']} - conditions: {', '.join(conditions)}")
            
            elif row['signal'] == 'SELL':
                if row['rsi'] > self.rsi_overbought:
                    conditions.append(f"RSI={row['rsi']:.1f}")
                if row['close'] < row['short_ma']:
                    conditions.append(f"price<MA{self.short_ma_period}")
                if row['macd_hist'] < 0:
                    conditions.append(f"MACD-")
                    
                logger.info(f"SELL signal at {idx} - price: {row['close']} - strength: {row['signal_strength']} - conditions: {', '.join(conditions)}")
        
        return df
    
    def get_current_signal(self, symbol, interval, lookback_period):
        """
        Override to include position sizing recommendation based on signal strength.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1h', '4h', '1d').
            lookback_period (str): Lookback period (e.g., '7 days ago UTC').
        
        Returns:
            tuple: (signal, position_size_multiplier)
        """
        df = self.generate_signals(symbol, interval, lookback_period)
        if df is None or len(df) == 0:
            logger.error(f"No data available for {symbol}")
            return 'HOLD', 0
        
        # Get the last signal and its strength
        last_row = df.iloc[-1]
        signal = last_row.get('signal', 'HOLD')
        strength = last_row.get('signal_strength', 0)
        
        # Don't trade if conviction is below threshold
        if strength < self.min_conviction_threshold:
            signal = 'HOLD'
            strength = 0
            logger.info(f"Signal strength {strength:.2f} below minimum threshold {self.min_conviction_threshold}, changing to HOLD")
        
        # Calculate position size multiplier based on custom multipliers from config if available
        if hasattr(config, 'CONVICTION_MULTIPLIERS') and isinstance(config.CONVICTION_MULTIPLIERS, dict):
            # Find the closest key in CONVICTION_MULTIPLIERS that's less than or equal to strength
            position_size_multiplier = 0
            for threshold, multiplier in sorted(config.CONVICTION_MULTIPLIERS.items()):
                if strength >= float(threshold):
                    position_size_multiplier = multiplier
        else:
            # Fallback to simple calculation
            position_size_multiplier = min(strength, 1.0)
        
        # Log the latest scan with position sizing information
        self.log_to_file(symbol, df, interval)
        
        # Include market regime in the log
        market_regime = last_row.get('market_regime', 'unknown')
        logger.info(f"Current market regime for {symbol}: {market_regime}")
        logger.info(f"Signal for {symbol}: {signal} with strength {strength:.2f} (size multiplier: {position_size_multiplier:.2f})")
        
        return signal, position_size_multiplier
    
    def is_market_favorable(self, symbol, interval, lookback_period):
        """
        Enhanced check if market conditions are favorable for trading.
        """
        try:
            # Get historical klines
            df = self.api.get_historical_klines(symbol, interval, lookback_period)
            if df is None or len(df) == 0:
                logger.error(f"No data available for {symbol}")
                return False
            
            # Calculate volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * 100  # Convert to percentage
            
            # Get volatility thresholds from config
            min_volatility = config.MARKET_VOLATILITY_MIN if hasattr(config, 'MARKET_VOLATILITY_MIN') else 0.2
            max_volatility = config.MARKET_VOLATILITY_MAX if hasattr(config, 'MARKET_VOLATILITY_MAX') else 5.0
            
            # Detect market regime
            self.detect_market_regime(df)
            current_regime = df['market_regime'].iloc[-1]
            
            # Market is favorable if:
            # 1. Volatility is within acceptable range, OR
            # 2. We're in a bull market with decent volatility
            is_favorable = (
                ((volatility >= min_volatility) and (volatility <= max_volatility)) or
                ((current_regime == 'bull') and (volatility >= min_volatility/2))
            )
            
            logger.info(f"Market conditions for {symbol}: Volatility = {volatility:.2f}% - Regime = {current_regime} - {'Favorable' if is_favorable else 'Unfavorable'}")
            
            return is_favorable
            
        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {str(e)}")
            return False  # Default to unfavorable if there's an error


def get_strategy(strategy_name, api_client=None):
    """
    Returns the appropriate strategy class based on the strategy name.
    """
    if strategy_name == 'SMA':
        return SimpleMovingAverageStrategy(api_client)
    elif strategy_name == 'RSI':
        return RSIStrategy(api_client)
    elif strategy_name == 'COMBINED':
        return CombinedStrategy(api_client)
    elif strategy_name == 'SMALL':
        return SmallAccountStrategy(api_client)
    elif strategy_name == 'TRON11':  # New strategy name
        return SmallAccountStrategyV1_1(api_client)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid options are 'SMA', 'RSI', 'COMBINED', 'SMALL', 'TRON11'.")