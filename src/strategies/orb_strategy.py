"""Open Range Breakout (ORB) Strategy."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Any

from src.strategies.base import BaseStrategy, TradingSignal, SignalDirection
from src.config import get_config
from src.indicators import atr

logger = logging.getLogger(__name__)


class ORBStrategy(BaseStrategy):
    """
    Open Range Breakout Strategy.
    
    Strategy:
    - Define range from first N minutes of zone
    - Enter on breakout beyond range with ATR buffer
    - Use ATR-based stops
    - Time-based exit
    
    Based on research showing breakout effectiveness around session transitions.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or get_config()
        self.strategy_config = self.config.strategy
        
        # ORB Parameters
        self.range_minutes = self.strategy_config.orb_range_minutes
        self.entry_buffer_atr = self.strategy_config.orb_entry_buffer_atr
        self.stop_atr = self.strategy_config.orb_stop_atr
        self.time_stop_minutes = self.strategy_config.orb_time_stop_minutes
        
        # State
        self._range_high: Optional[float] = None
        self._range_low: Optional[float] = None
        self._range_start_time: Optional[pd.Timestamp] = None
        self._atr_value: float = 0
        self._entry_bar_time: Optional[pd.Timestamp] = None
        
    def compute_signal(
        self,
        df: pd.DataFrame,
        position: int,
        zone_info: Any = None
    ) -> Optional[TradingSignal]:
        """
        Compute ORB signal.
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            position: Current position
            zone_info: Current zone info
        
        Returns:
            TradingSignal if breakout detected
        """
        if not self.validate_data(df):
            return None
        
        # Need enough bars
        if len(df) < self.range_minutes + 2:
            return None
        
        # Get current bar
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        current_time = df.index[-1]
        
        # Reset range on first bar of zone
        if zone_info and zone_info.is_first_bar:
            self._range_start_time = zone_info.start_time or df.index[0]
            self._reset_range(df)

        # Initialize range if not set
        if self._range_high is None:
            if zone_info:
                self._range_start_time = zone_info.start_time or df.index[0]
            self._reset_range(df)

        # Range may still be unavailable if the current zone just started and we don't have
        # enough in-zone bars yet (even if the overall dataframe is long enough).
        if self._range_high is None or self._range_low is None:
            return None
        
        # Calculate ATR
        self._atr_value = atr(
            df['high'], 
            df['low'], 
            df['close'], 
            self.strategy_config.atr_length
        ).iloc[-1]
        
        if pd.isna(self._atr_value) or self._atr_value <= 0:
            return None
        
        # Calculate buffers
        entry_buffer = self.entry_buffer_atr * self._atr_value
        stop_distance = self.stop_atr * self._atr_value
        
        # Check for breakout
        signal = None
        
        if position == 0:
            # Long breakout
            if current_price > self._range_high + entry_buffer:
                # Check trend filter (price above MA for longs)
                ma_length = self.strategy_config.trend_ma_length
                if len(df) >= ma_length:
                    ma = df['close'].rolling(ma_length).mean().iloc[-1]
                    if current_price > ma:  # Uptrend filter
                        stop_loss = current_price - stop_distance
                        
                        signal = TradingSignal(
                            direction=SignalDirection.LONG,
                            price=current_price,
                            stop_loss=stop_loss,
                            contracts=1,  # Sized by risk manager
                            confidence=0.8,
                            reason=f"ORB long breakout above {self._range_high:.2f}",
                            metadata={
                                'range_high': self._range_high,
                                'range_low': self._range_low,
                                'atr': self._atr_value,
                                'buffer': entry_buffer
                            }
                        )
                        self._entry_bar_time = current_time
            
            # Short breakout  
            elif current_price < self._range_low - entry_buffer:
                # Check trend filter (price below MA for shorts)
                ma_length = self.strategy_config.trend_ma_length
                if len(df) >= ma_length:
                    ma = df['close'].rolling(ma_length).mean().iloc[-1]
                    if current_price < ma:  # Downtrend filter
                        stop_loss = current_price + stop_distance
                        
                        signal = TradingSignal(
                            direction=SignalDirection.SHORT,
                            price=current_price,
                            stop_loss=stop_loss,
                            contracts=1,
                            confidence=0.8,
                            reason=f"ORB short breakout below {self._range_low:.2f}",
                            metadata={
                                'range_high': self._range_high,
                                'range_low': self._range_low,
                                'atr': self._atr_value,
                                'buffer': entry_buffer
                            }
                        )
                        self._entry_bar_time = current_time
        
        # Time-based exit check
        if position != 0 and self._entry_bar_time is not None:
            minutes_held = (current_time - self._entry_bar_time).total_seconds() / 60.0
            if minutes_held > self.time_stop_minutes:
                # Time stop triggered
                logger.info(f"ORB time stop triggered after {minutes_held:.1f} minutes")
                return TradingSignal(
                    direction=SignalDirection.FLAT,
                    price=current_price,
                    reason="time_stop"
                )
        
        if signal:
            self._last_signal = signal
            self._bars_since_signal = 0
        else:
            self._bars_since_signal += 1
            
        return signal
    
    def _reset_range(self, df: pd.DataFrame):
        """Reset the open range."""
        if self._range_start_time is not None:
            zone_bars = df[df.index >= self._range_start_time]
        else:
            zone_bars = df

        range_bars = zone_bars.iloc[:self.range_minutes]
        if len(range_bars) < self.range_minutes:
            return

        self._range_high = range_bars['high'].max()
        self._range_low = range_bars['low'].min()
        self._range_start_time = range_bars.index[0]
        logger.debug(f"ORB range reset: high={self._range_high}, low={self._range_low}")
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._range_high = None
        self._range_low = None
        self._range_start_time = None
        self._atr_value = 0
        self._entry_bar_time = None
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {
            'range_minutes': self.range_minutes,
            'entry_buffer_atr': self.entry_buffer_atr,
            'stop_atr': self.stop_atr,
            'time_stop_minutes': self.time_stop_minutes,
        }
