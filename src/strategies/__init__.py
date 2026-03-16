"""Strategies package."""
from .base import BaseStrategy, TradingSignal, SignalDirection, StrategyRegistry
from .orb_strategy import ORBStrategy
from .vwap_trend import VWAPTrendStrategy
from .vwap_mr import VWAPMeanReversionStrategy
from .flatten_strategy import FlattenStrategy
from src.config import get_config


def register_default_strategies(config=None) -> None:
    """Rebuild the default strategy registry for the active config."""
    cfg = config or get_config()
    StrategyRegistry.clear()
    StrategyRegistry.register("ORB_BREAKOUT", ORBStrategy(cfg))
    StrategyRegistry.register("VWAP_TREND", VWAPTrendStrategy(cfg))
    StrategyRegistry.register("VWAP_MEAN_REVERSION", VWAPMeanReversionStrategy(cfg))
    StrategyRegistry.register("FLATTEN_ONLY", FlattenStrategy(cfg))


register_default_strategies()

__all__ = [
    'BaseStrategy',
    'TradingSignal',
    'SignalDirection',
    'StrategyRegistry',
    'ORBStrategy',
    'VWAPTrendStrategy',
    'VWAPMeanReversionStrategy',
    'FlattenStrategy',
    'register_default_strategies',
]
