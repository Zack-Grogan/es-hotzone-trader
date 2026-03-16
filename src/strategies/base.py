"""Base strategy class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    direction: SignalDirection
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    contracts: int = 1
    confidence: float = 1.0  # 0-1
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Base class for trading strategies.
    
    All strategies should inherit from this and implement compute_signal.
    """
    
    def __init__(self, config=None):
        """Initialize strategy."""
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
        
        # Strategy state
        self._last_signal: Optional[TradingSignal] = None
        self._bars_since_signal: int = 0
        
    @abstractmethod
    def compute_signal(
        self,
        df: pd.DataFrame,
        position: int,
        zone_info: Any = None
    ) -> Optional[TradingSignal]:
        """
        Compute trading signal from current data.
        
        Args:
            df: DataFrame with OHLCV data
            position: Current position (0=flat, +n=long, -n=short)
            zone_info: Current zone information
        
        Returns:
            TradingSignal if signal generated, None otherwise
        """
        pass
    
    def reset(self):
        """Reset strategy state."""
        self._last_signal = None
        self._bars_since_signal = 0
        self._initialized = False
        logger.debug(f"Strategy {self.name} state reset")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters for logging."""
        return {}
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that df has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required)
    
    def get_last_signal(self) -> Optional[TradingSignal]:
        """Get last generated signal."""
        return self._last_signal


class StrategyRegistry:
    """Registry for strategy instances."""
    
    _strategies: Dict[str, BaseStrategy] = {}
    
    @classmethod
    def register(cls, name: str, strategy: BaseStrategy):
        """Register a strategy."""
        cls._strategies[name] = strategy
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseStrategy]:
        """Get strategy by name."""
        return cls._strategies.get(name)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registered strategies."""
        cls._strategies.clear()
