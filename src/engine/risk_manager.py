"""Risk Manager - Position sizing and risk controls."""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional, List
from collections import deque

import pytz

from src.config import get_config, RiskConfig
from src.observability import get_observability_store

logger = logging.getLogger(__name__)


class RiskState(Enum):
    """Risk state machine."""
    NORMAL = "normal"
    REDUCED = "reduced"      # Reduced size
    CIRCUIT_BREAKER = "circuit_breaker"  # No trading


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1 for long, -1 for short
    contracts: int
    entry_price: float
    exit_price: float
    pnl: float
    zone: str
    strategy: str
    regime: str = ""
    event_tags: list[str] | None = None


@dataclass 
class RiskMetrics:
    """Current risk metrics."""
    daily_pnl: float = 0
    max_daily_loss: float = 0
    consecutive_losses: int = 0
    trades_today: int = 0
    trades_this_hour: int = 0
    trades_this_zone: int = 0
    current_position: int = 0
    current_position_pnl: float = 0
    risk_state: RiskState = RiskState.NORMAL


class RiskManager:
    """
    Manages position sizing and risk controls.
    
    Enforces:
    - Daily loss limits
    - Position loss limits
    - Consecutive loss circuit breaker
    - Trade frequency limits
    - Position sizing based on volatility
    """
    
    def __init__(self, config=None):
        """Initialize risk manager."""
        self.config = config or get_config()
        self.risk_config: RiskConfig = self.config.risk
        self.account_config = self.config.account
        self.observability = get_observability_store()
        
        # State
        self._daily_pnl: float = 0
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._last_trade_time: Optional[datetime] = None
        self._trades_this_hour: deque = deque()  # Timestamps of recent trades
        self._trades_this_zone: int = 0
        self._current_zone_name: str = ""
        self._current_regime: str = ""
        self._current_event_tags: list[str] = []
        self._current_strategy: str = "WEIGHTED_SCORE_MATRIX"
        self._blackout_active: bool = False
        self._blackout_reason: str = ""
        self._volatility_history: deque = deque(maxlen=20)
        self._volatility_circuit_active: bool = False
        
        # Trade history
        self._trade_history: List[TradeRecord] = []
        self._daily_date: Optional[date] = None
        self._clock_time: Optional[datetime] = None
        self._session_timezone = pytz.timezone(self.risk_config.session_timezone)
        self._market_price_stale_seconds = max(int(getattr(self.config.watchdog, "feed_stale_seconds", 15)), 1)
        
        # Risk state
        self._risk_state: RiskState = RiskState.NORMAL
        
        # Position tracking
        self._current_position: int = 0
        self._position_entry_price: float = 0
        self._position_entry_time: Optional[datetime] = None
        self._current_position_pnl: float = 0
        self._last_market_price: Optional[float] = None
        self._last_market_price_time: Optional[datetime] = None
        
        logger.info("RiskManager initialized with max_daily_loss=$%s", self.risk_config.max_daily_loss)

    def _record_event(
        self,
        *,
        event_type: str,
        payload: Optional[dict] = None,
        event_time: Optional[datetime] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> None:
        self.observability.record_event(
            category="risk",
            event_type=event_type,
            source=__name__,
            payload=payload or {},
            event_time=event_time or self._clock_time,
            symbol=self.config.symbols[0] if self.config.symbols else None,
            zone=zone if zone is not None else self._current_zone_name,
            action=action,
            reason=reason,
            risk_state=self._risk_state.value,
        )
    
    def _coerce_time(self, current_time: Optional[datetime] = None) -> datetime:
        """Convert a timestamp into the configured session timezone."""
        candidate = current_time or self._clock_time
        if candidate is None:
            return datetime.now(self._session_timezone)
        if candidate.tzinfo is None:
            return self._session_timezone.localize(candidate)
        return candidate.astimezone(self._session_timezone)

    def _session_date(self, current_time: Optional[datetime] = None):
        """Return the trading-session date keyed off the CME session boundary."""
        local_time = self._coerce_time(current_time)
        reset_offset = timedelta(
            hours=self.risk_config.session_reset_hour,
            minutes=self.risk_config.session_reset_minute,
        )
        return (local_time - reset_offset).date()

    def observe_time(self, current_time: Optional[datetime]) -> None:
        """Track the current engine/replay clock."""
        if current_time is None:
            return
        self._clock_time = self._coerce_time(current_time)

    def observe_market_price(self, market_price: Optional[float], observed_at: Optional[datetime] = None) -> None:
        """Track the most recent tradeable market price."""
        if market_price is None:
            return
        price = float(market_price)
        if price > 0:
            self._last_market_price = price
            self._last_market_price_time = self._coerce_time(observed_at)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL without mutating other state."""
        if self._current_position == 0:
            return 0.0
        direction = 1 if self._current_position > 0 else -1
        contracts = abs(self._current_position)
        if direction > 0:
            return (current_price - self._position_entry_price) * 50 * contracts
        return (self._position_entry_price - current_price) * 50 * contracts

    def _reset_daily(self, current_time: Optional[datetime] = None):
        """Reset daily counters if new day."""
        today = self._session_date(current_time)
        if self._daily_date != today:
            self._daily_pnl = 0
            self._daily_trades = 0
            self._consecutive_losses = 0
            self._trades_this_hour.clear()
            self._trades_this_zone = 0
            self._daily_date = today
            self._risk_state = RiskState.NORMAL
            logger.info("Daily counters reset")
            self._record_event(
                event_type="daily_reset",
                payload={"session_date": today.isoformat()},
                event_time=self._coerce_time(current_time),
                action="reset",
                reason="daily_reset",
            )
    
    def _clean_hourly_trades(self, current_time: Optional[datetime] = None):
        """Remove trades older than 1 hour from hourly counter."""
        now = self._coerce_time(current_time)
        while self._trades_this_hour and (now - self._coerce_time(self._trades_this_hour[0])).total_seconds() > 3600:
            self._trades_this_hour.popleft()
    
    def can_trade(self, zone_name: str, current_time: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Check if new trade is allowed.
        
        Returns:
            (allowed, reason)
        """
        self.observe_time(current_time)
        self._reset_daily(current_time)
        self._clean_hourly_trades(current_time)
        
        # Check circuit breaker state
        if self._risk_state == RiskState.CIRCUIT_BREAKER:
            self._record_event(
                event_type="trade_blocked",
                payload={"zone_name": zone_name},
                event_time=self._coerce_time(current_time),
                action="block_trade",
                reason="circuit_breaker_active",
                zone=zone_name,
            )
            return False, "circuit_breaker_active"

        if self._blackout_active:
            self._record_event(
                event_type="trade_blocked",
                payload={"zone_name": zone_name},
                event_time=self._coerce_time(current_time),
                action="block_trade",
                reason=self._blackout_reason or "blackout_active",
                zone=zone_name,
            )
            return False, self._blackout_reason or "blackout_active"
        
        # Check daily loss limit
        if self._daily_pnl <= -self.risk_config.max_daily_loss:
            self._risk_state = RiskState.CIRCUIT_BREAKER
            logger.warning("Daily loss limit hit: $%.2f", self._daily_pnl)
            self._record_event(
                event_type="risk_state_changed",
                payload={"daily_pnl": self._daily_pnl, "max_daily_loss": self.risk_config.max_daily_loss},
                event_time=self._coerce_time(current_time),
                action="set_circuit_breaker",
                reason="daily_loss_limit",
                zone=zone_name,
            )
            return False, "daily_loss_limit"
        
        # Check consecutive losses
        if self._consecutive_losses >= self.risk_config.max_consecutive_losses:
            self._risk_state = RiskState.CIRCUIT_BREAKER
            logger.warning("Consecutive loss limit hit: %s", self._consecutive_losses)
            self._record_event(
                event_type="risk_state_changed",
                payload={"consecutive_losses": self._consecutive_losses, "max_consecutive_losses": self.risk_config.max_consecutive_losses},
                event_time=self._coerce_time(current_time),
                action="set_circuit_breaker",
                reason="consecutive_loss_limit",
                zone=zone_name,
            )
            return False, "consecutive_loss_limit"
        
        # Check trades per hour
        if len(self._trades_this_hour) >= self.risk_config.max_trades_per_hour:
            self._record_event(
                event_type="trade_blocked",
                payload={"zone_name": zone_name, "trades_this_hour": len(self._trades_this_hour)},
                event_time=self._coerce_time(current_time),
                action="block_trade",
                reason="hourly_trade_limit",
                zone=zone_name,
            )
            return False, "hourly_trade_limit"
        
        # Check trades per zone
        if zone_name != self._current_zone_name:
            self._trades_this_zone = 0
            self._current_zone_name = zone_name
        
        if self._trades_this_zone >= self.risk_config.max_trades_per_zone:
            self._record_event(
                event_type="trade_blocked",
                payload={"zone_name": zone_name, "trades_this_zone": self._trades_this_zone},
                event_time=self._coerce_time(current_time),
                action="block_trade",
                reason="zone_trade_limit",
                zone=zone_name,
            )
            return False, "zone_trade_limit"
        
        # Check total daily trades
        if self._daily_trades >= self.risk_config.max_daily_trades:
            self._record_event(
                event_type="trade_blocked",
                payload={"zone_name": zone_name, "daily_trades": self._daily_trades},
                event_time=self._coerce_time(current_time),
                action="block_trade",
                reason="daily_trade_limit",
                zone=zone_name,
            )
            return False, "daily_trade_limit"
        
        return True, "ok"

    def set_blackout(self, active: bool, reason: str = "news_blackout") -> None:
        """Enable or disable a trading blackout."""
        self._blackout_active = active
        self._blackout_reason = reason if active else ""
        self._record_event(
            event_type="blackout_changed",
            payload={"active": active},
            event_time=self._clock_time,
            action="set_blackout",
            reason=reason if active else "blackout_cleared",
        )

    def update_volatility(self, atr_value: float) -> None:
        """Track ATR history and raise a circuit breaker on volatility spikes."""
        if atr_value <= 0:
            return

        self._volatility_history.append(atr_value)
        if len(self._volatility_history) < 5:
            return

        baseline_values = list(self._volatility_history)[:-1]
        if len(baseline_values) < 4:
            return
        baseline = sum(baseline_values) / len(baseline_values)
        threshold = baseline * self.risk_config.vol_spike_threshold

        if self.risk_config.enable_circuit_breakers and atr_value >= threshold:
            if not self._volatility_circuit_active or self._risk_state != RiskState.CIRCUIT_BREAKER:
                logger.warning(
                    "risk_circuit_breaker_activated reason=volatility_spike atr=%.4f baseline=%.4f threshold=%.4f",
                    atr_value,
                    baseline,
                    threshold,
                )
                self._record_event(
                    event_type="risk_state_changed",
                    payload={"atr_value": atr_value, "baseline": baseline, "threshold": threshold},
                    event_time=self._clock_time,
                    action="set_circuit_breaker",
                    reason="volatility_spike",
                    zone=self._current_zone_name,
                )
            self._volatility_circuit_active = True
            self._risk_state = RiskState.CIRCUIT_BREAKER
        elif self._volatility_circuit_active and atr_value < (baseline * 0.9):
            self._volatility_circuit_active = False
            if self._risk_state == RiskState.CIRCUIT_BREAKER and self._daily_pnl > -self.risk_config.max_daily_loss:
                self._risk_state = RiskState.REDUCED
                logger.info(
                    "risk_circuit_breaker_cleared reason=volatility_normalized atr=%.4f baseline=%.4f new_state=%s",
                    atr_value,
                    baseline,
                    self._risk_state.value,
                )
                self._record_event(
                    event_type="risk_state_changed",
                    payload={"atr_value": atr_value, "baseline": baseline, "new_state": self._risk_state.value},
                    event_time=self._clock_time,
                    action="set_reduced_risk",
                    reason="volatility_normalized",
                    zone=self._current_zone_name,
                )
    
    def calculate_position_size(
        self,
        atr_value: float,
        direction: int = 1
    ) -> int:
        """
        Calculate position size based on volatility and risk parameters.
        
        Args:
            atr_value: Current ATR value
            direction: Trade direction (1=long, -1=short)
        
        Returns:
            Number of contracts (1-5)
        """
        if atr_value <= 0:
            return self.account_config.default_contracts
        
        # Target risk per contract
        risk_per_contract = self.account_config.risk_per_contract
        
        # Risk per point = ATR * $50 (ES multiplier)
        risk_per_point = atr_value * 50
        
        # Contracts based on risk
        contracts = int(risk_per_contract / risk_per_point) if risk_per_point > 0 else 1
        
        # Clamp to limits
        contracts = max(1, min(contracts, self.account_config.max_contracts))
        
        # Check if we should reduce size due to risk state
        if self._risk_state == RiskState.REDUCED:
            contracts = min(contracts, 2)  # Max 2 contracts in reduced state
        elif self._risk_state == RiskState.CIRCUIT_BREAKER:
            contracts = 0
        
        return contracts
    
    def calculate_stop_distance(self, atr_value: float, multiplier: float = 2.0) -> float:
        """
        Calculate stop distance in points.
        
        Args:
            atr_value: Current ATR
            multiplier: ATR multiplier for stop
    
        Returns:
            Stop distance in points
        """
        return atr_value * multiplier
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update metrics."""
        self._trade_history.append(trade)
        self._daily_trades += 1
        self._trades_this_hour.append(trade.exit_time)
        self._trades_this_zone += 1
        self._last_trade_time = trade.exit_time
        self._daily_pnl += trade.pnl
        
        if trade.pnl < 0:
            self._consecutive_losses += 1
            logger.info("Loss recorded. Consecutive losses: %s", self._consecutive_losses)
        else:
            self._consecutive_losses = 0
        self._record_event(
            event_type="trade_recorded",
            payload={
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "contracts": trade.contracts,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "direction": trade.direction,
                "zone": trade.zone,
                "strategy": trade.strategy,
                "regime": trade.regime,
                "event_tags": list(trade.event_tags or []),
            },
            event_time=trade.exit_time,
            action="record_trade",
            reason="trade_recorded",
            zone=trade.zone,
        )
        if getattr(self.config.observability, "persist_completed_trades", True):
            self.observability.record_completed_trade(trade)
        
        # Reset zone trade count if zone changed
        if trade.zone != self._current_zone_name:
            self._trades_this_zone = 0
            self._current_zone_name = trade.zone

    def _build_trade_record(self, contracts: int, exit_price: float, exit_time: datetime) -> TradeRecord:
        direction = 1 if self._current_position > 0 else -1
        pnl_per_contract = ((exit_price - self._position_entry_price) * 50) if direction > 0 else ((self._position_entry_price - exit_price) * 50)
        return TradeRecord(
            entry_time=self._position_entry_time or exit_time,
            exit_time=exit_time,
            direction=direction,
            contracts=contracts,
            entry_price=self._position_entry_price,
            exit_price=exit_price,
            pnl=pnl_per_contract * contracts,
            zone=self._current_zone_name,
            strategy=self._current_strategy,
            regime=self._current_regime,
            event_tags=list(self._current_event_tags),
        )

    def _clear_position_tracking(self) -> None:
        self._current_position = 0
        self._position_entry_price = 0
        self._position_entry_time = None
        self._current_position_pnl = 0
        self._current_regime = ""
        self._current_event_tags = []
        self._current_strategy = "WEIGHTED_SCORE_MATRIX"
    
    def open_position(
        self,
        contracts: int,
        entry_price: float,
        direction: int,
        zone: str,
        regime: str = "",
        event_tags: Optional[list[str]] = None,
        strategy: str = "WEIGHTED_SCORE_MATRIX",
        current_time: Optional[datetime] = None,
    ):
        """Record position opening."""
        self.observe_time(current_time)
        self._current_position = contracts * direction
        self._position_entry_price = entry_price
        self._position_entry_time = self._coerce_time(current_time)
        self._current_zone_name = zone
        self._current_regime = regime
        self._current_event_tags = list(event_tags or [])
        self._current_strategy = strategy
        self._current_position_pnl = 0
        logger.info("Position opened: %s contracts %s at %s", contracts, "long" if direction > 0 else "short", entry_price)
        self._record_event(
            event_type="position_opened",
            payload={"contracts": contracts, "entry_price": entry_price, "direction": direction, "regime": regime, "event_tags": list(event_tags or []), "strategy": strategy},
            event_time=self._coerce_time(current_time),
            action="open_position",
            reason="position_opened",
            zone=zone,
        )
    
    def close_position(self, exit_price: float, current_time: Optional[datetime] = None) -> Optional[TradeRecord]:
        """Record position closing and return trade record."""
        if self._current_position == 0:
            return None
        self.observe_time(current_time)
        exit_time = self._coerce_time(current_time)
        trade = self._build_trade_record(abs(self._current_position), exit_price, exit_time)
        self.record_trade(trade)
        self._clear_position_tracking()
        return trade

    def sync_position(
        self,
        signed_position: int,
        entry_price: float,
        transition_price: float,
        zone: str,
        regime: str = "",
        event_tags: Optional[list[str]] = None,
        strategy: str = "WEIGHTED_SCORE_MATRIX",
        current_time: Optional[datetime] = None,
    ) -> list[TradeRecord]:
        self.observe_time(current_time)
        current_time_value = self._coerce_time(current_time)
        completed: list[TradeRecord] = []
        prior_position = self._current_position

        if prior_position == signed_position:
            if signed_position != 0 and entry_price > 0:
                self._position_entry_price = entry_price
            return completed

        if prior_position == 0:
            if signed_position != 0:
                self.open_position(
                    contracts=abs(signed_position),
                    entry_price=entry_price if entry_price > 0 else transition_price,
                    direction=1 if signed_position > 0 else -1,
                    zone=zone,
                    regime=regime,
                    event_tags=event_tags,
                    strategy=strategy,
                    current_time=current_time_value,
                )
            return completed

        if transition_price <= 0:
            transition_price = entry_price if entry_price > 0 else self._position_entry_price

        prior_direction = 1 if prior_position > 0 else -1
        new_direction = 1 if signed_position > 0 else (-1 if signed_position < 0 else 0)
        prior_contracts = abs(prior_position)
        new_contracts = abs(signed_position)

        if new_direction == 0:
            completed_trade = self._build_trade_record(prior_contracts, transition_price, current_time_value)
            self.record_trade(completed_trade)
            completed.append(completed_trade)
            self._clear_position_tracking()
            return completed

        if new_direction != prior_direction:
            completed_trade = self._build_trade_record(prior_contracts, transition_price, current_time_value)
            self.record_trade(completed_trade)
            completed.append(completed_trade)
            self._clear_position_tracking()
            self.open_position(
                contracts=new_contracts,
                entry_price=entry_price if entry_price > 0 else transition_price,
                direction=new_direction,
                zone=zone,
                regime=regime,
                event_tags=event_tags,
                strategy=strategy,
                current_time=current_time_value,
            )
            return completed

        if new_contracts < prior_contracts:
            completed_trade = self._build_trade_record(prior_contracts - new_contracts, transition_price, current_time_value)
            self.record_trade(completed_trade)
            completed.append(completed_trade)
            self._current_position = signed_position
            if entry_price > 0:
                self._position_entry_price = entry_price
            self._current_position_pnl = 0
            return completed

        self._current_position = signed_position
        if entry_price > 0:
            self._position_entry_price = entry_price
        if self._position_entry_time is None:
            self._position_entry_time = current_time_value
        return completed
    
    def update_position_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL."""
        if self._current_position == 0:
            return 0
        pnl = self._calculate_unrealized_pnl(current_price)
        self._current_position_pnl = pnl
        return pnl

    def should_flatten_position(self, current_price: Optional[float] = None, current_time: Optional[datetime] = None) -> tuple[bool, str]:
        """Return whether the current position should be flattened for risk reasons."""
        if self._current_position == 0:
            return False, "flat"

        self.observe_time(current_time)
        effective_price = current_price
        if effective_price is None:
            if self._last_market_price is None or self._last_market_price_time is None:
                return False, "no_market_price"
            market_age = abs((self._coerce_time(current_time) - self._coerce_time(self._last_market_price_time)).total_seconds())
            if market_age > self._market_price_stale_seconds:
                return False, "stale_market_price"
            effective_price = self._last_market_price
        if effective_price is None or effective_price <= 0:
            return False, "no_market_price"

        current_pnl = self.update_position_pnl(effective_price)
        if current_pnl <= -self.risk_config.max_position_loss:
            self._risk_state = RiskState.REDUCED
            self._record_event(
                event_type="flatten_required",
                payload={"current_pnl": current_pnl, "max_position_loss": self.risk_config.max_position_loss, "current_price": effective_price},
                event_time=self._coerce_time(current_time),
                action="flatten",
                reason="max_position_loss",
            )
            return True, "max_position_loss"

        if self._blackout_active:
            self._record_event(
                event_type="flatten_required",
                payload={"current_price": effective_price},
                event_time=self._coerce_time(current_time),
                action="flatten",
                reason=self._blackout_reason or "blackout_active",
            )
            return True, self._blackout_reason or "blackout_active"

        if self._risk_state == RiskState.CIRCUIT_BREAKER:
            self._record_event(
                event_type="flatten_required",
                payload={"current_price": effective_price},
                event_time=self._coerce_time(current_time),
                action="flatten",
                reason="circuit_breaker_active",
            )
            return True, "circuit_breaker_active"

        return False, "ok"
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return RiskMetrics(
            daily_pnl=self._daily_pnl,
            max_daily_loss=self.risk_config.max_daily_loss,
            consecutive_losses=self._consecutive_losses,
            trades_today=self._daily_trades,
            trades_this_hour=len(self._trades_this_hour),
            trades_this_zone=self._trades_this_zone,
            current_position=self._current_position,
            current_position_pnl=self._current_position_pnl,
            risk_state=self._risk_state
        )
    
    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self._risk_state
    
    def reduce_risk(self):
        """Reduce risk state (e.g., after partial loss)."""
        if self._risk_state == RiskState.NORMAL:
            self._risk_state = RiskState.REDUCED
            logger.warning("Risk state reduced")
            self._record_event(
                event_type="risk_state_changed",
                payload={},
                event_time=self._clock_time,
                action="set_reduced_risk",
                reason="reduce_risk",
            )
    
    def reset_risk(self):
        """Reset risk state to normal."""
        self._risk_state = RiskState.NORMAL
        logger.info("Risk state reset to normal")
        self._record_event(
            event_type="risk_state_changed",
            payload={},
            event_time=self._clock_time,
            action="set_normal_risk",
            reason="reset_risk",
        )

    def reset_state(self, clear_history: bool = True):
        """Reset runtime risk state for a fresh session or replay."""
        self._daily_pnl = 0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._last_trade_time = None
        self._trades_this_hour.clear()
        self._trades_this_zone = 0
        self._current_zone_name = ""
        self._current_regime = ""
        self._current_event_tags = []
        self._current_strategy = "WEIGHTED_SCORE_MATRIX"
        self._blackout_active = False
        self._blackout_reason = ""
        self._volatility_history.clear()
        self._volatility_circuit_active = False
        self._daily_date = None
        self._clock_time = None
        self._risk_state = RiskState.NORMAL
        self._current_position = 0
        self._position_entry_price = 0
        self._position_entry_time = None
        self._current_position_pnl = 0
        self._last_market_price = None
        self._last_market_price_time = None
        if clear_history:
            self._trade_history = []

    def get_trade_history(self) -> List[TradeRecord]:
        """Return completed trades."""
        return list(self._trade_history)

    def is_reduced_risk(self) -> bool:
        """Return whether size should be reduced."""
        return self._risk_state == RiskState.REDUCED


# Global risk manager instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager(force_recreate: bool = False) -> RiskManager:
    """Get global risk manager instance."""
    global _risk_manager
    if force_recreate:
        _risk_manager = None
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
