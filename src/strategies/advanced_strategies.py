"""
Advanced trading strategies including grid trading and scalping.
Optimized for $500 account with multiple trading pairs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from src.strategies.strategies import Strategy
from src.api.binance_api import BinanceAPI
from src.utils.logger import get_logger
import config

logger = get_logger()

class GridStrategy(Strategy):
    """
    Grid trading strategy that places orders at regular price intervals.
    Optimized for sideways markets.
    """
    
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.name = "Grid Trading"
        self.grid_levels = {}  # Store grid levels for each symbol
        self.active_orders = {}  # Track active grid orders
        
    def calculate_grid_levels(self, symbol: str, current_price: float) -> List[float]:
        """Calculate grid levels around current price."""
        spacing = current_price * (config.GRID_SPACING_PERCENT / 100)
        levels = []
        
        # Calculate levels above and below current price
        for i in range(-config.GRID_LEVELS, config.GRID_LEVELS + 1):
            level = current_price + (i * spacing)
            levels.append(level)
            
        return sorted(levels)
    
    def generate_signals(self, symbol: str, interval: str, lookback_period: str) -> pd.DataFrame:
        """Generate grid trading signals."""
        df = self.api.get_historical_klines(symbol, interval, lookback_period)
        if df is None or len(df) == 0:
            return None
            
        current_price = df['close'].iloc[-1]
        
        # Initialize or update grid levels
        if symbol not in self.grid_levels:
            self.grid_levels[symbol] = self.calculate_grid_levels(symbol, current_price)
            
        # Find nearest grid levels
        buy_levels = [level for level in self.grid_levels[symbol] if level < current_price]
        sell_levels = [level for level in self.grid_levels[symbol] if level > current_price]
        
        # Generate signals based on price crossing grid levels
        df['signal'] = 'HOLD'
        
        # Buy signal when price crosses below a grid level
        for level in buy_levels:
            df.loc[df['low'] <= level, 'signal'] = 'BUY'
            
        # Sell signal when price crosses above a grid level
        for level in sell_levels:
            df.loc[df['high'] >= level, 'signal'] = 'SELL'
            
        return df

class ScalpingStrategy(Strategy):
    """
    Scalping strategy for quick profits on small price movements.
    Uses multiple timeframes for confirmation.
    """
    
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.name = "Scalping"
        
    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators for scalping."""
        # RSI for momentum
        df['rsi'] = self.calculate_rsi(df['close'], window=7)
        
        # MACD for trend
        df['macd'], df['signal'], df['hist'] = self.calculate_macd(
            df['close'], 
            fast_period=6, 
            slow_period=13, 
            signal_period=4
        )
        
        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
        
    def generate_signals(self, symbol: str, interval: str, lookback_period: str) -> pd.DataFrame:
        """Generate scalping signals."""
        # Get data for multiple timeframes
        df_1m = self.api.get_historical_klines(symbol, '1m', lookback_period)
        df_5m = self.api.get_historical_klines(symbol, '5m', lookback_period)
        
        if df_1m is None or df_5m is None:
            return None
            
        # Calculate indicators for both timeframes
        df_1m = self.calculate_momentum(df_1m)
        df_5m = self.calculate_momentum(df_5m)
        
        # Generate signals based on multiple conditions
        df_1m['signal'] = 'HOLD'
        
        # Buy conditions
        buy_condition = (
            (df_1m['rsi'] < 30) &  # Oversold on 1m
            (df_1m['macd'] > df_1m['signal']) &  # MACD crossover
            (df_1m['volume_ratio'] > 1.5) &  # High volume
            (df_5m['rsi'] < 40)  # Confirmation from 5m
        )
        
        # Sell conditions
        sell_condition = (
            (df_1m['rsi'] > 70) |  # Overbought on 1m
            (df_1m['macd'] < df_1m['signal']) |  # MACD crossover
            (df_1m['volume_ratio'] < 0.5)  # Low volume
        )
        
        df_1m.loc[buy_condition, 'signal'] = 'BUY'
        df_1m.loc[sell_condition, 'signal'] = 'SELL'
        
        return df_1m

class AdvancedTRONStrategy(Strategy):
    """
    Enhanced version of TRON strategy with market regime detection
    and dynamic position sizing.
    """
    
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.name = "Advanced TRON"
        
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime (trending, ranging, volatile)
        """
        # Calculate volatility
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Calculate trend strength
        df['adx'] = self.calculate_adx(df, period=14)
        trend_strength = df['adx'].iloc[-1]
        
        # Determine regime
        if trend_strength > 25:
            return 'TRENDING'
        elif volatility > 0.02:  # 2% daily volatility
            return 'VOLATILE'
        else:
            return 'RANGING'
            
    def calculate_position_size(self, symbol: str, regime: str, conviction: float) -> float:
        """
        Calculate position size based on market regime and conviction
        """
        base_size = config.MAX_RISK_PER_TRADE / 100  # Convert percentage to decimal
        
        # Adjust for market regime
        regime_multipliers = {
            'TRENDING': 1.0,
            'RANGING': 0.7,
            'VOLATILE': 0.5
        }
        
        # Adjust for conviction
        conviction_multiplier = 1.0
        for threshold, multiplier in config.CONVICTION_MULTIPLIERS.items():
            if conviction >= threshold:
                conviction_multiplier = multiplier
                break
                
        return base_size * regime_multipliers[regime] * conviction_multiplier
        
    def generate_signals(self, symbol: str, interval: str, lookback_period: str) -> pd.DataFrame:
        """Generate enhanced TRON signals."""
        df = self.api.get_historical_klines(symbol, interval, lookback_period)
        if df is None or len(df) == 0:
            return None
            
        # Calculate basic indicators
        df['rsi'] = self.calculate_rsi(df['close'], window=config.TRON_RSI_PERIOD)
        df['short_ma'] = df['close'].rolling(window=config.TRON_SHORT_MA).mean()
        df['long_ma'] = df['close'].rolling(window=config.TRON_LONG_MA).mean()
        df['macd'], df['signal'], df['hist'] = self.calculate_macd(
            df['close'],
            fast_period=12,
            slow_period=26,
            signal_period=config.TRON_MACD_SIGNAL
        )
        
        # Detect market regime
        regime = self.detect_market_regime(df)
        
        # Calculate conviction level (0-1)
        df['conviction'] = (
            (df['rsi'] < 30).astype(int) * 0.3 +  # RSI oversold
            (df['macd'] > df['signal']).astype(int) * 0.3 +  # MACD bullish
            (df['short_ma'] > df['long_ma']).astype(int) * 0.4  # MA alignment
        )
        
        # Generate signals
        df['signal'] = 'HOLD'
        
        # Buy conditions
        buy_condition = (
            (df['rsi'] < config.TRON_RSI_OVERSOLD) &
            (df['macd'] > df['signal']) &
            (df['short_ma'] > df['long_ma']) &
            (df['conviction'] >= config.MIN_CONVICTION_THRESHOLD)
        )
        
        # Sell conditions
        sell_condition = (
            (df['rsi'] > config.TRON_RSI_OVERBOUGHT) |
            (df['macd'] < df['signal']) |
            (df['short_ma'] < df['long_ma'])
        )
        
        df.loc[buy_condition, 'signal'] = 'BUY'
        df.loc[sell_condition, 'signal'] = 'SELL'
        
        # Add position size recommendation
        df['position_size'] = df.apply(
            lambda x: self.calculate_position_size(symbol, regime, x['conviction']),
            axis=1
        )
        
        return df 