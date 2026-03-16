"""VWAP Trend Strategy - Trend continuation using VWAP."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Any

from src.strategies.base import BaseStrategy, TradingSignal, SignalDirection
from src.config import get_config
from src.indicators import atr, vwap, vwap_deviation

logger = logging.getLogger(__name__)


class VWAPTrendStrategy(BaseStrategy):
    """
    VWAP Trend Continuation Strategy.
    
    Strategy:
    - Identify trend via MA slope
    - Enter on pullback to VWAP in direction of trend
    - Exit at profit target or trend reversal
    
    Based on research showing VWAP as "fair price" anchor.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or get_config()
        self.strategy_config = self.config.strategy
        
        # Parameters
        self.ma_length = self.strategy_config.trend_ma_length
        self.ma_type = self.strategy_config.trend_ma_type  # EMA or SMA
        self.vwap_dev_threshold = self.strategy_config.vwap_deviation_threshold
        self.vwap_confirmation_bars = self.strategy_config.vwap_confirmation_bars
        
        # State
        self._vwap_value: float = 0
        self._atr_value: float = 0
        self._ma_value: float = 0
        
    def compute_signal(
        self,
        df: pd.DataFrame,
        position: int,
        zone_info: Any = None
    ) -> Optional[TradingSignal]:
        """
        Compute VWAP trend signal.
        
        Args:
            df: DataFrame with OHLCV data
            position: Current position
            zone_info: Current zone info
        
        Returns:
            TradingSignal if entry/exit signal
        """
        if not self.validate_data(df):
            return None
        
        if len(df) < max(self.ma_length, 20):
            return None
        
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        previous_close = df['close'].iloc[-2]
        
        # Calculate indicators
        self._atr_value = atr(
            df['high'],
            df['low'],
            df['close'],
            self.strategy_config.atr_length
        ).iloc[-1]
        
        self._vwap_value = vwap(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            self.strategy_config.vwap_source
        ).iloc[-1]
        
        # Calculate MA
        if self.ma_type == "EMA":
            self._ma_value = df['close'].ewm(span=self.ma_length, adjust=False).mean().iloc[-1]
        else:
            self._ma_value = df['close'].rolling(self.ma_length).mean().iloc[-1]
        
        if pd.isna(self._vwap_value) or pd.isna(self._ma_value) or pd.isna(self._atr_value):
            return None
        
        # Calculate deviation from VWAP
        dev = vwap_deviation(current_price, self._vwap_value, self._atr_value)
        
        # Determine trend
        ma_slope = (self._ma_value - df['close'].iloc[-self.ma_length]) / self.ma_length
        confirmation_window = min(self.vwap_confirmation_bars, len(df))
        recent_closes = df['close'].iloc[-confirmation_window:]
        recent_vwap = vwap(
            df['high'].iloc[-confirmation_window:],
            df['low'].iloc[-confirmation_window:],
            df['close'].iloc[-confirmation_window:],
            df['volume'].iloc[-confirmation_window:],
            self.strategy_config.vwap_source
        )
        
        # Check for entry signals (flat)
        if position == 0:
            # Long: uptrend (price > MA) + pullback to VWAP
            if current_price > self._ma_value and ma_slope > 0:
                # Check if price pulled back to VWAP
                if (
                    dev >= -self.vwap_dev_threshold
                    and dev <= 0
                    and current_price > previous_close
                    and (recent_closes >= recent_vwap.ffill()).all()
                ):
                    # Enter long on bounce from VWAP
                    stop_loss = self._vwap_value - (self._atr_value * 2)
                    take_profit = current_price + (self._atr_value * 1.5)
                    
                    return TradingSignal(
                        direction=SignalDirection.LONG,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        contracts=1,
                        confidence=0.75,
                        reason=f"VWAP trend long: price {dev:.2f} ATR from VWAP",
                        metadata={
                            'vwap': self._vwap_value,
                            'ma': self._ma_value,
                            'ma_slope': ma_slope,
                            'atr': self._atr_value,
                            'deviation': dev
                        }
                    )
            
            # Short: downtrend (price < MA) + pullback to VWAP
            elif current_price < self._ma_value and ma_slope < 0:
                # Check if price pulled back to VWAP
                if (
                    dev <= self.vwap_dev_threshold
                    and dev >= 0
                    and current_price < previous_close
                    and (recent_closes <= recent_vwap.ffill()).all()
                ):
                    stop_loss = self._vwap_value + (self._atr_value * 2)
                    take_profit = current_price - (self._atr_value * 1.5)
                    
                    return TradingSignal(
                        direction=SignalDirection.SHORT,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        contracts=1,
                        confidence=0.75,
                        reason=f"VWAP trend short: price {dev:.2f} ATR from VWAP",
                        metadata={
                            'vwap': self._vwap_value,
                            'ma': self._ma_value,
                            'ma_slope': ma_slope,
                            'atr': self._atr_value,
                            'deviation': dev
                        }
                    )
        
        # Exit logic (in position)
        elif position != 0:
            # Exit on trend reversal
            if position > 0 and current_price < self._ma_value:
                return TradingSignal(
                    direction=SignalDirection.FLAT,
                    price=current_price,
                    reason="trend_reversal"
                )
            elif position < 0 and current_price > self._ma_value:
                return TradingSignal(
                    direction=SignalDirection.FLAT,
                    price=current_price,
                    reason="trend_reversal"
                )
            
        return None
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {
            'ma_length': self.ma_length,
            'ma_type': self.ma_type,
            'vwap_dev_threshold': self.vwap_dev_threshold,
            'vwap_confirmation_bars': self.vwap_confirmation_bars,
        }
