import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api import BinanceAPI
from logger import get_logger
from notifier import send_trade_notification
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
        return last_signal


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


def get_strategy(strategy_name, api_client=None):
    """
    Returns the appropriate strategy class based on the strategy name.

    Args:
        strategy_name (str): The name of the strategy (e.g., 'SMA', 'RSI', 'COMBINED').
        api_client (BinanceAPI, optional): Binance API client. Defaults to None.

    Returns:
        Strategy: The appropriate strategy class.
    """
    if strategy_name == 'SMA':
        return SimpleMovingAverageStrategy(api_client)
    elif strategy_name == 'RSI':
        return RSIStrategy(api_client)
    elif strategy_name == 'COMBINED':
        return CombinedStrategy(api_client)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid options are 'SMA', 'RSI', 'COMBINED'.")