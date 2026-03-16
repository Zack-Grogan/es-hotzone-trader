"""VWAP Mean Reversion Strategy."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Any

from src.strategies.base import BaseStrategy, TradingSignal, SignalDirection
from src.config import get_config
from src.indicators import atr, vwap, rsi

logger = logging.getLogger(__name__)


class VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.
    
    Strategy:
    - Fade deviations from VWAP when RSI indicates extension
    - Use strict time stops
    - Exit at VWAP
    
    Best suited for midday when volatility is lower.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or get_config()
        self.strategy_config = self.config.strategy
        
        # Parameters
        self.rsi_length = self.strategy_config.mr_rsi_length
        self.rsi_oversold = self.strategy_config.mr_rsi_oversold
        self.rsi_overbought = self.strategy_config.mr_rsi_overbought
        self.band_deviation = self.strategy_config.mr_band_deviation
        self.time_stop_minutes = self.strategy_config.mr_time_stop_minutes
        self.exit_at_vwap = self.strategy_config.mr_exit_at_vwap
        
        # State
        self._vwap_value: float = 0
        self._atr_value: float = 0
        self._rsi_value: float = 50
        self._entry_bar_time: Optional[pd.Timestamp] = None

    def _flat_signal(self, price: float, reason: str) -> TradingSignal:
        """Emit a flat signal and clear entry timing state."""
        self._entry_bar_time = None
        return TradingSignal(
            direction=SignalDirection.FLAT,
            price=price,
            reason=reason,
        )
        
    def compute_signal(
        self,
        df: pd.DataFrame,
        position: int,
        zone_info: Any = None
    ) -> Optional[TradingSignal]:
        """
        Compute VWAP mean reversion signal.
        
        Args:
            df: DataFrame with OHLCV data
            position: Current position
            zone_info: Current zone info
        
        Returns:
            TradingSignal if entry/exit signal
        """
        if not self.validate_data(df):
            return None
        
        if len(df) < max(self.rsi_length, 20):
            return None
        
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        current_time = df.index[-1]

        if zone_info and position == 0 and zone_info.minutes_remaining <= 10:
            return None
        
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
        
        self._rsi_value = rsi(df['close'], self.rsi_length).iloc[-1]
        bollinger_mid = df['close'].rolling(20).mean().iloc[-1]
        bollinger_std = df['close'].rolling(20).std().iloc[-1]
        upper_band = bollinger_mid + (bollinger_std * 2)
        lower_band = bollinger_mid - (bollinger_std * 2)
        
        if (
            pd.isna(self._vwap_value)
            or pd.isna(self._atr_value)
            or pd.isna(self._rsi_value)
            or pd.isna(upper_band)
            or pd.isna(lower_band)
        ):
            return None
        
        # Calculate deviation in ATR
        dev_atr = (current_price - self._vwap_value) / self._atr_value
        
        # Check for entry signals (flat)
        if position == 0:
            # Long: price below VWAP + RSI oversold
            if (
                dev_atr <= -self.band_deviation
                and self._rsi_value <= self.rsi_oversold
                and current_price <= lower_band
            ):
                stop_loss = current_price - (self._atr_value * 2)
                self._entry_bar_time = current_time
                
                return TradingSignal(
                    direction=SignalDirection.LONG,
                    price=current_price,
                    stop_loss=stop_loss,
                    contracts=1,
                    confidence=0.7,
                    reason=f"VWAP MR long: dev={dev_atr:.2f} ATR, RSI={self._rsi_value:.1f}",
                    metadata={
                        'vwap': self._vwap_value,
                        'atr': self._atr_value,
                        'rsi': self._rsi_value,
                        'deviation_atr': dev_atr
                    }
                )
            
            # Short: price above VWAP + RSI overbought
            elif (
                dev_atr >= self.band_deviation
                and self._rsi_value >= self.rsi_overbought
                and current_price >= upper_band
            ):
                stop_loss = current_price + (self._atr_value * 2)
                self._entry_bar_time = current_time
                
                return TradingSignal(
                    direction=SignalDirection.SHORT,
                    price=current_price,
                    stop_loss=stop_loss,
                    contracts=1,
                    confidence=0.7,
                    reason=f"VWAP MR short: dev={dev_atr:.2f} ATR, RSI={self._rsi_value:.1f}",
                    metadata={
                        'vwap': self._vwap_value,
                        'atr': self._atr_value,
                        'rsi': self._rsi_value,
                        'deviation_atr': dev_atr
                    }
                )
        
        # Exit logic (in position)
        elif position != 0:
            # Time-based exit
            if self._entry_bar_time is not None:
                minutes_held = (current_time - self._entry_bar_time).total_seconds() / 60.0
                if minutes_held > self.time_stop_minutes:
                    return self._flat_signal(current_price, "time_stop")
            
            # VWAP mean reversion target exit
            if self.exit_at_vwap:
                if position > 0 and current_price >= self._vwap_value:
                    return self._flat_signal(current_price, "vwap_target")
                elif position < 0 and current_price <= self._vwap_value:
                    return self._flat_signal(current_price, "vwap_target")
            
            # RSI neutralization exit
            if position > 0 and self._rsi_value >= 50:
                return self._flat_signal(current_price, "rsi_neutral")
            elif position < 0 and self._rsi_value <= 50:
                return self._flat_signal(current_price, "rsi_neutral")
        
        return None
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._vwap_value = 0
        self._atr_value = 0
        self._rsi_value = 50
        self._entry_bar_time = None
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {
            'rsi_length': self.rsi_length,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'band_deviation': self.band_deviation,
            'time_stop_minutes': self.time_stop_minutes,
            'exit_at_vwap': self.exit_at_vwap,
        }
