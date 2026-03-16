"""RSI (Relative Strength Index) indicator."""
import pandas as pd
import numpy as np
from typing import Optional, Literal


def rsi(
    close: pd.Series,
    period: int = 14,
    source: Literal["close"] = "close"
) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        close: Close prices
        period: RSI period
        source: Source series (for compatibility)
    
    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    rsi_values = pd.Series(index=close.index, dtype=float)
    if len(close) <= period:
        return rsi_values

    avg_gain = gains.iloc[1 : period + 1].mean()
    avg_loss = losses.iloc[1 : period + 1].mean()
    if avg_loss == 0:
        rsi_values.iloc[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values.iloc[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(close)):
        avg_gain = ((avg_gain * (period - 1)) + gains.iloc[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses.iloc[i]) / period
        if avg_loss == 0:
            rsi_values.iloc[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values.iloc[i] = 100 - (100 / (1 + rs))

    return rsi_values


def rsi_classic(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate RSI using classic (non-Wilder) smoothing.
    
    Args:
        close: Close prices
        period: RSI period
    
    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def rsi_ma(
    rsi_values: pd.Series,
    ma_period: int = 9
) -> pd.Series:
    """
    Calculate RSI moving average for signal smoothing.
    
    Args:
        rsi_values: RSI series
        ma_period: MA period
    
    Returns:
        RSI MA series
    """
    return rsi_values.rolling(window=ma_period).mean()


def is_oversold(rsi: float, threshold: float = 30) -> bool:
    """Check if RSI indicates oversold condition."""
    return rsi <= threshold


def is_overbought(rsi: float, threshold: float = 70) -> bool:
    """Check if RSI indicates overbought condition."""
    return rsi >= threshold


def rsi_divergence(
    price: pd.Series,
    rsi_values: pd.Series,
    lookback: int = 20
) -> Optional[str]:
    """
    Detect RSI divergence.
    
    Args:
        price: Price series
        rsi_values: RSI series
        lookback: Bars to look back
    
    Returns:
        'bullish', 'bearish', or None
    """
    if len(price) < lookback:
        return None
    
    recent_price = price.iloc[-lookback:]
    recent_rsi = rsi_values.iloc[-lookback:]
    
    price_lower = recent_price.iloc[-1] == recent_price.min()
    prior_rsi_low = recent_rsi.iloc[:-1].min() if len(recent_rsi) > 1 else recent_rsi.iloc[-1]
    rsi_higher = recent_rsi.iloc[-1] > prior_rsi_low
    
    if price_lower and rsi_higher:
        return 'bullish'
    
    price_higher = recent_price.iloc[-1] == recent_price.max()
    prior_rsi_high = recent_rsi.iloc[:-1].max() if len(recent_rsi) > 1 else recent_rsi.iloc[-1]
    rsi_lower = recent_rsi.iloc[-1] < prior_rsi_high
    
    if price_higher and rsi_lower:
        return 'bearish'
    
    return None
