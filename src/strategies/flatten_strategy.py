"""Flatten-Only Strategy - No new entries, manage exits only."""
import pandas as pd
import logging
from typing import Optional, Any

from src.strategies.base import BaseStrategy, TradingSignal, SignalDirection

logger = logging.getLogger(__name__)


class FlattenStrategy(BaseStrategy):
    """
    Flatten Only Strategy.
    
    Used for:
    - Micro-windows near zone end (12:45-1:00)
    - High-risk periods
    - News blackout periods
    
    Always returns FLAT signal if in position, None otherwise.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "FlattenOnly"
        
    def compute_signal(
        self,
        df: pd.DataFrame,
        position: int,
        zone_info: Any = None
    ) -> Optional[TradingSignal]:
        """
        Compute flatten signal.
        
        Args:
            df: DataFrame (not used)
            position: Current position
            zone_info: Current zone info
        
        Returns:
            FLAT signal if in position, None otherwise
        """
        if position != 0:
            return TradingSignal(
                direction=SignalDirection.FLAT,
                price=df.iloc[-1]['close'] if len(df) > 0 else 0,
                reason="flatten_only_mode"
            )
        
        return None
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {'mode': 'flatten_only'}
