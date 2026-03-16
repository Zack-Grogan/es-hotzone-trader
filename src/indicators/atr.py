"""ATR (Average True Range) indicator."""
import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    use_wilder: bool = True
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        period: ATR period
        use_wilders: Use Wilder's smoothing (default True)
    
    Returns:
        ATR series
    """
    tr = calculate_true_range(high, low, close)
    
    if use_wilder:
        # Wilder's smoothing - EMA with alpha = 1/period
        alpha = 1.0 / period
        atr_values = pd.Series(index=tr.index, dtype=float)
        atr_values.iloc[period - 1] = tr.iloc[:period].mean()
        
        for i in range(period, len(tr)):
            atr_values.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * atr_values.iloc[i - 1]
        
        return atr_values
    else:
        return tr.rolling(window=period).mean()


def atr_current(high: float, low: float, close: float, prev_close: Optional[float] = None) -> float:
    """Calculate current bar TR and return simplified ATR estimate."""
    tr = max(
        high - low,
        abs(high - (prev_close or close)),
        abs(low - (prev_close or close))
    )
    return tr


def normalized_atr(price: float, atr_value: float) -> float:
    """Get ATR as percentage of price."""
    if price == 0:
        return 0
    return (atr_value / price) * 100
