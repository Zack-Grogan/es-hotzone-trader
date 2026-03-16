"""Order execution module."""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import OrderExecutionConfig, ReplayExecutionConfig, WatchdogConfig, get_config
from src.market.topstep_client import MarketData, TopstepClient, get_client
from src.observability import get_observability_store

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Per-order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionState(Enum):
    """Local execution state machine."""

    PENDING_SUBMIT = "PENDING_SUBMIT"
    ACK_PENDING = "ACK_PENDING"
    WORKING = "WORKING"
    FILLED = "FILLED"
    PROTECTED = "PROTECTED"
    FLATTENING = "FLATTENING"
    FLAT = "FLAT"
    ERROR = "ERROR"


@dataclass
class Order:
    """Order information."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    remaining_quantity: int = 0
    created_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    filled_time: Optional[datetime] = None
    activation_time: Optional[datetime] = None
    is_protective: bool = False
    role: str = "entry"

    def __post_init__(self) -> None:
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in {OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}


class OrderExecutor:
    """Order execution manager for live and replay/mock flows."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.exec_config: OrderExecutionConfig = self.config.order_execution
        self.replay_config: ReplayExecutionConfig = self.config.replay_execution
        self.watchdog_config: WatchdogConfig = self.config.watchdog
        self.client: TopstepClient = get_client()
        self.observability = get_observability_store()

        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: List[Order] = []
        self._protective_orders: Dict[str, Dict[str, Any]] = {}
        self._mock_mode: bool = False
        self._mock_position: int = 0
        self._mock_avg_price: float = 0.0
        self._mock_trade_clock: Optional[datetime] = None
        self._lifecycle_state: ExecutionState = ExecutionState.FLAT
        self._last_ack_time: Optional[datetime] = None
        self._last_action_time: Optional[datetime] = None
        self._protection_requested_at: Dict[str, datetime] = {}
        self._last_protective_fill_reason: Optional[str] = None
        self._recent_fills: List[Dict[str, Any]] = []

    def _record_event(
        self,
        *,
        event_type: str,
        payload: Optional[dict] = None,
        event_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        self.observability.record_event(
            category="execution",
            event_type=event_type,
            source=__name__,
            payload=payload or {},
            event_time=event_time,
            symbol=symbol,
            action=action,
            reason=reason,
            order_id=order_id,
            risk_state=None,
        )

    def enable_mock_mode(self) -> None:
        """Enable mock mode for testing and replay."""
        self._mock_mode = True
        self.client.enable_mock_mode()
        logger.info("Executor mock mode enabled")
        self._record_event(
            event_type="mock_mode_enabled",
            payload={},
            event_time=datetime.now(UTC),
            action="enable_mock_mode",
            reason="mock_mode_enabled",
        )

    def reset_state(self, mock_mode: Optional[bool] = None) -> None:
        """Reset executor runtime state."""
        self._pending_orders = {}
        self._filled_orders = []
        self._protective_orders = {}
        self._mock_mode = bool(mock_mode) if mock_mode is not None else False
        self._mock_position = 0
        self._mock_avg_price = 0.0
        self._mock_trade_clock = None
        self._lifecycle_state = ExecutionState.FLAT
        self._last_ack_time = None
        self._last_action_time = None
        self._protection_requested_at = {}
        self._last_protective_fill_reason = None
        self._recent_fills = []

    def process_market_data(self, market_data: MarketData) -> bool:
        """Advance mock pending orders against the latest quote."""
        if not self._mock_mode:
            return False

        self._mock_trade_clock = market_data.timestamp
        changed = False
        for order in list(self._pending_orders.values()):
            if not order.is_active:
                continue
            if order.activation_time and market_data.timestamp < order.activation_time:
                continue
            if self._try_fill_mock_order(order, market_data):
                changed = True

        return changed

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        use_limit_fallback: bool = True,
        is_protective: bool = False,
        role: str = "entry",
    ) -> Optional[Order]:
        """Place an order."""
        if quantity <= 0:
            return None

        created_time = self._current_time(symbol)
        self._last_action_time = created_time
        self._lifecycle_state = ExecutionState.PENDING_SUBMIT
        self._record_event(
            event_type="order_submission_requested",
            payload={"quantity": quantity, "side": side, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "is_protective": is_protective, "role": role},
            event_time=created_time,
            symbol=symbol,
            action="submit_order",
            reason=role,
        )

        if self._mock_mode:
            return self._place_mock_order(symbol, quantity, side, order_type, limit_price, stop_price, is_protective, role)

        self._lifecycle_state = ExecutionState.ACK_PENDING
        order_id = self.client.place_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        if order_id:
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                status=OrderStatus.OPEN,
                created_time=created_time,
                updated_time=created_time,
                activation_time=created_time,
                is_protective=is_protective,
                role=role,
            )
            self._pending_orders[order_id] = order
            self._last_ack_time = created_time
            self._lifecycle_state = ExecutionState.WORKING
            self._record_event(
                event_type="order_submitted",
                payload={"quantity": quantity, "side": side, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "is_protective": is_protective, "role": role, "lifecycle_state": self._lifecycle_state.value},
                event_time=created_time,
                symbol=symbol,
                action="submit_order",
                reason=role,
                order_id=order_id,
            )
            return order

        self._lifecycle_state = ExecutionState.ERROR
        self._record_event(
            event_type="order_submission_failed",
            payload={"quantity": quantity, "side": side, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "is_protective": is_protective, "role": role, "fallback_enabled": use_limit_fallback and order_type == "limit" and self.exec_config.market_order_fallback},
            event_time=created_time,
            symbol=symbol,
            action="submit_order",
            reason="submission_failed",
        )
        if use_limit_fallback and order_type == "limit" and self.exec_config.market_order_fallback:
            logger.info("Limit order failed, falling back to market order")
            self._record_event(
                event_type="order_submission_fallback",
                payload={"original_order_type": order_type, "fallback_order_type": "market", "quantity": quantity, "side": side, "is_protective": is_protective, "role": role},
                event_time=created_time,
                symbol=symbol,
                action="fallback_to_market",
                reason="limit_order_failed",
            )
            return self.place_order(symbol, quantity, side, "market", None, None, False, is_protective, role)
        return None

    def _place_mock_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        is_protective: bool,
        role: str,
    ) -> Order:
        """Place a mock order with simple replay-aware fills."""
        created_time = self._current_time(symbol)
        order = Order(
            order_id=f"MOCK_{int(created_time.timestamp() * 1000)}_{len(self._pending_orders) + len(self._filled_orders)}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.OPEN,
            created_time=created_time,
            updated_time=created_time,
            activation_time=None,
            is_protective=is_protective,
            role=role,
        )
        self._pending_orders[order.order_id] = order
        self._last_ack_time = created_time
        self._lifecycle_state = ExecutionState.WORKING
        self._record_event(
            event_type="order_submitted",
            payload={"quantity": quantity, "side": side, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "is_protective": is_protective, "role": role, "mock_mode": True, "lifecycle_state": self._lifecycle_state.value},
            event_time=created_time,
            symbol=symbol,
            action="submit_order",
            reason=role,
            order_id=order.order_id,
        )

        market_data = self.client.get_market_data(symbol)
        if market_data is not None:
            self.process_market_data(market_data)
        return order

    def _sample_latency_ms(self, explicit_latency_ms: int) -> int:
        if explicit_latency_ms > 0:
            return explicit_latency_ms
        jitter = max(int(self.replay_config.latency_ms_jitter), 0)
        base = max(int(self.replay_config.latency_ms_default), 0)
        if jitter == 0:
            return base
        return max(0, base + random.randint(-jitter, jitter))

    def _current_time(self, symbol: str = "ES") -> datetime:
        market_data = self.client.get_market_data(symbol)
        if market_data is not None and market_data.timestamp is not None:
            return market_data.timestamp.astimezone(UTC) if market_data.timestamp.tzinfo else market_data.timestamp.replace(tzinfo=UTC)
        if self._mock_trade_clock is not None:
            return self._mock_trade_clock.astimezone(UTC) if self._mock_trade_clock.tzinfo else self._mock_trade_clock.replace(tzinfo=UTC)
        return datetime.now(UTC)

    def _try_fill_mock_order(self, order: Order, market_data: MarketData) -> bool:
        fillable = False
        fill_price: Optional[float] = None
        available_size = order.remaining_quantity

        if order.order_type == "market":
            fillable = True
            fill_price = self._mock_market_fill_price(order.side, market_data)
            available_size = order.remaining_quantity
        elif order.order_type == "limit":
            if order.side == "buy" and market_data.ask and order.limit_price is not None and market_data.ask <= order.limit_price:
                fillable = True
                fill_price = min(order.limit_price, market_data.ask)
                available_size = int(market_data.ask_size or order.remaining_quantity)
            elif order.side == "sell" and market_data.bid and order.limit_price is not None and market_data.bid >= order.limit_price:
                fillable = True
                fill_price = max(order.limit_price, market_data.bid)
                available_size = int(market_data.bid_size or order.remaining_quantity)
        elif order.order_type == "stop":
            if order.side == "sell" and order.stop_price is not None and (market_data.bid or market_data.last) <= order.stop_price:
                fillable = True
                fill_price = min(order.stop_price, market_data.bid or market_data.last)
                available_size = order.remaining_quantity
            elif order.side == "buy" and order.stop_price is not None and (market_data.ask or market_data.last) >= order.stop_price:
                fillable = True
                fill_price = max(order.stop_price, market_data.ask or market_data.last)
                available_size = order.remaining_quantity

        if not fillable or fill_price is None:
            return False

        if not self.replay_config.allow_partial_fills or available_size <= 0:
            fill_qty = order.remaining_quantity
        else:
            fill_qty = max(1, min(order.remaining_quantity, available_size))

        order.filled_quantity += fill_qty
        order.remaining_quantity = max(order.quantity - order.filled_quantity, 0)
        order.filled_price = fill_price
        order.filled_time = market_data.timestamp
        order.updated_time = market_data.timestamp
        order.status = OrderStatus.FILLED if order.remaining_quantity == 0 else OrderStatus.PARTIALLY_FILLED
        self._recent_fills.append(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": fill_qty,
                "filled_price": fill_price,
                "filled_time": market_data.timestamp,
                "is_protective": order.is_protective,
                "role": order.role,
            }
        )

        self._apply_mock_fill(order.side, fill_qty, fill_price)
        self._last_ack_time = market_data.timestamp
        self._lifecycle_state = ExecutionState.FILLED if not order.is_protective else ExecutionState.PROTECTED

        if order.status == OrderStatus.FILLED:
            self._filled_orders.append(order)
            self._pending_orders.pop(order.order_id, None)
            if order.is_protective:
                self._last_protective_fill_reason = order.role
                self._cancel_sibling_protection(order.symbol, order.order_id)
        self._record_event(
            event_type="order_fill",
            payload={"filled_quantity": fill_qty, "filled_price": fill_price, "remaining_quantity": order.remaining_quantity, "is_protective": order.is_protective, "role": order.role, "status": order.status.value, "mock_mode": True},
            event_time=market_data.timestamp,
            symbol=order.symbol,
            action="fill_order",
            reason=order.role,
            order_id=order.order_id,
        )
        return True

    def _mock_market_fill_price(self, side: str, market_data: MarketData) -> float:
        tick_size = 0.25
        slippage_ticks = self.replay_config.market_slippage_ticks
        slip = tick_size * slippage_ticks
        anchor = market_data.ask if side == "buy" else market_data.bid
        anchor = anchor or market_data.last or market_data.mid
        return anchor + slip if side == "buy" else anchor - slip

    def _apply_mock_fill(self, side: str, quantity: int, fill_price: float) -> None:
        prior_position = self._mock_position
        if side == "buy":
            new_position = prior_position + quantity
        else:
            new_position = prior_position - quantity

        increasing_same_side = (
            (prior_position > 0 and side == "buy")
            or (prior_position < 0 and side == "sell")
        )
        if prior_position == 0:
            self._mock_avg_price = fill_price if new_position != 0 else 0.0
        elif increasing_same_side:
            total_contracts = abs(prior_position) + quantity
            self._mock_avg_price = (
                (abs(prior_position) * self._mock_avg_price) + (quantity * fill_price)
            ) / total_contracts
        elif new_position == 0:
            self._mock_avg_price = 0.0
        elif (prior_position > 0) != (new_position > 0):
            self._mock_avg_price = fill_price

        self._mock_position = new_position
        if self._mock_position == 0:
            self._lifecycle_state = ExecutionState.FLAT

    def close_position(self, symbol: str, quantity: Optional[int] = None, order_type: str = "market") -> Optional[Order]:
        """Close position (partial or full)."""
        position = self.get_position(symbol)
        if position == 0:
            return None
        qty = abs(quantity) if quantity else abs(position)
        side = "sell" if position > 0 else "buy"
        return self.place_order(symbol, qty, side, order_type)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._pending_orders.get(order_id)
        if self._mock_mode:
            if order is None:
                return False
            order.status = OrderStatus.CANCELLED
            order.updated_time = self._current_time(order.symbol)
            self._pending_orders.pop(order_id, None)
            self._record_event(
                event_type="order_cancelled",
                payload={"mock_mode": True, "is_protective": order.is_protective, "role": order.role},
                event_time=order.updated_time,
                symbol=order.symbol,
                action="cancel_order",
                reason="cancelled",
                order_id=order_id,
            )
            return True

        if not self.client.cancel_order(order_id):
            self._record_event(
                event_type="order_cancel_failed",
                payload={"mock_mode": False},
                event_time=self._current_time(order.symbol if order is not None else "ES"),
                symbol=order.symbol if order is not None else "ES",
                action="cancel_order",
                reason="broker_cancel_failed",
                order_id=order_id,
            )
            return False

        self._last_ack_time = self._current_time()
        if order is not None:
            order.status = OrderStatus.CANCELLED
            order.updated_time = self._last_ack_time
            self._pending_orders.pop(order_id, None)
            self._record_event(
                event_type="order_cancelled",
                payload={"mock_mode": False, "is_protective": order.is_protective, "role": order.role},
                event_time=order.updated_time,
                symbol=order.symbol,
                action="cancel_order",
                reason="cancelled",
                order_id=order_id,
            )
        return True

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        for order_id in list(self._pending_orders.keys()):
            if self.cancel_order(order_id):
                cancelled += 1
        return cancelled

    def get_active_orders(
        self,
        symbol: Optional[str] = None,
        role: Optional[str] = None,
        is_protective: Optional[bool] = None,
    ) -> Dict[str, Order]:
        """Return only active local orders, optionally filtered."""
        orders: Dict[str, Order] = {}
        for order_id, order in self._pending_orders.items():
            if not order.is_active:
                continue
            if symbol is not None and order.symbol != symbol:
                continue
            if role is not None and order.role != role:
                continue
            if is_protective is not None and order.is_protective != is_protective:
                continue
            orders[order_id] = order
        return orders

    def has_active_entry_order(self, symbol: str = "ES") -> bool:
        """Return whether a non-protective entry/flatten order is still active."""
        return bool(self.get_active_orders(symbol=symbol, is_protective=False))

    def get_position(self, symbol: str = "ES") -> int:
        """Get current signed position."""
        if self._mock_mode:
            return self._mock_position
        position = self.client.get_position(symbol)
        return position.quantity

    def get_average_price(self) -> float:
        """Get the current average entry price."""
        if self._mock_mode:
            return self._mock_avg_price
        return self.client.get_position().entry_price

    def reconcile_pending_orders(self) -> int:
        """Cancel stale pending orders after the configured timeout."""
        timeout_seconds = int(self.watchdog_config.stale_order_seconds or self.exec_config.cancel_timeout_seconds)
        now = self._current_time()
        cancelled = 0
        for order_id, order in list(self._pending_orders.items()):
            age = (now - order.created_time).total_seconds()
            if age < timeout_seconds or not order.is_active:
                continue
            logger.warning(
                "Cancelling stale %s order %s after %.1fs (symbol=%s role=%s protective=%s)",
                order.order_type,
                order_id,
                age,
                order.symbol,
                order.role,
                order.is_protective,
            )
            self._record_event(
                event_type="stale_order_detected",
                payload={"order_type": order.order_type, "age_seconds": age, "role": order.role, "is_protective": order.is_protective},
                event_time=now,
                symbol=order.symbol,
                action="cancel_stale_order",
                reason="stale_order",
                order_id=order_id,
            )
            if self.cancel_order(order_id):
                cancelled += 1
            else:
                logger.warning("Unable to cancel stale order %s; it will remain locally active until broker state updates", order_id)
        return cancelled

    def ensure_protection(
        self,
        symbol: str,
        quantity: int,
        direction: int,
        stop_price: Optional[float],
        take_profit: Optional[float],
    ) -> int:
        """Place or refresh local protective orders."""
        if quantity <= 0 or (stop_price is None and take_profit is None):
            return 0

        existing = self._protective_orders.get(symbol)
        same_stop = (
            existing is not None
            and (
                existing.get("stop_price") is None
                and stop_price is None
                or (
                    existing.get("stop_price") is not None
                    and stop_price is not None
                    and math.isclose(float(existing.get("stop_price")), float(stop_price), rel_tol=0.0, abs_tol=1e-6)
                )
            )
        )
        same_target = (
            existing is not None
            and (
                existing.get("take_profit") is None
                and take_profit is None
                or (
                    existing.get("take_profit") is not None
                    and take_profit is not None
                    and math.isclose(float(existing.get("take_profit")), float(take_profit), rel_tol=0.0, abs_tol=1e-6)
                )
            )
        )
        if existing and existing.get("direction") == direction and same_stop and same_target:
            self._lifecycle_state = ExecutionState.PROTECTED
            self._record_event(
                event_type="protection_unchanged",
                payload={"quantity": quantity, "direction": direction, "stop_price": stop_price, "take_profit": take_profit, "orders": list(existing.get("orders", []))},
                event_time=self._current_time(symbol),
                symbol=symbol,
                action="ensure_protection",
                reason="protection_already_matches",
            )
            return len(existing.get("orders", []))

        self.clear_protection(symbol)
        self._protection_requested_at[symbol] = self._current_time(symbol)
        exit_side = "sell" if direction > 0 else "buy"
        orders: list[str] = []

        if stop_price is not None:
            stop_order = self.place_order(
                symbol=symbol,
                quantity=quantity,
                side=exit_side,
                order_type="stop",
                stop_price=stop_price,
                use_limit_fallback=False,
                is_protective=True,
                role="stop_loss",
            )
            if stop_order is not None:
                orders.append(stop_order.order_id)

        if take_profit is not None:
            target_order = self.place_order(
                symbol=symbol,
                quantity=quantity,
                side=exit_side,
                order_type="limit",
                limit_price=take_profit,
                use_limit_fallback=False,
                is_protective=True,
                role="take_profit",
            )
            if target_order is not None:
                orders.append(target_order.order_id)

        if orders:
            self._protective_orders[symbol] = {
                "orders": orders,
                "direction": direction,
                "stop_price": stop_price,
                "take_profit": take_profit,
            }
            self._lifecycle_state = ExecutionState.PROTECTED
            self._record_event(
                event_type="protection_placed",
                payload={"orders": list(orders), "direction": direction, "stop_price": stop_price, "take_profit": take_profit, "quantity": quantity},
                event_time=self._current_time(symbol),
                symbol=symbol,
                action="ensure_protection",
                reason="protection_placed",
            )
        else:
            self._lifecycle_state = ExecutionState.ERROR
            self._record_event(
                event_type="protection_failed",
                payload={"direction": direction, "stop_price": stop_price, "take_profit": take_profit, "quantity": quantity},
                event_time=self._current_time(symbol),
                symbol=symbol,
                action="ensure_protection",
                reason="protection_placement_failed",
            )
        return len(orders)

    def _cancel_sibling_protection(self, symbol: str, filled_order_id: str) -> None:
        record = self._protective_orders.get(symbol)
        if not record:
            return
        siblings = [order_id for order_id in record.get("orders", []) if order_id != filled_order_id]
        unresolved: list[str] = []
        for order_id in siblings:
            if not self.cancel_order(order_id):
                unresolved.append(order_id)
        if unresolved:
            record["orders"] = unresolved
            self._protective_orders[symbol] = record
        else:
            self._protective_orders.pop(symbol, None)
        self._record_event(
            event_type="sibling_protection_cancelled",
            payload={"filled_order_id": filled_order_id, "remaining_orders": unresolved},
            event_time=self._current_time(symbol),
            symbol=symbol,
            action="cancel_sibling_protection",
            reason="protective_fill",
            order_id=filled_order_id,
        )

    def clear_protection(self, symbol: str) -> int:
        """Cancel active protective orders for a symbol."""
        cancelled = 0
        record = self._protective_orders.pop(symbol, None)
        self._protection_requested_at.pop(symbol, None)
        if not record:
            return 0
        for order_id in record.get("orders", []):
            if self.cancel_order(order_id):
                cancelled += 1
        self._record_event(
            event_type="protection_cleared",
            payload={"cancelled_orders": cancelled, "requested_orders": list(record.get("orders", []))},
            event_time=self._current_time(symbol),
            symbol=symbol,
            action="clear_protection",
            reason="protection_cleared",
        )
        return cancelled

    def protection_pending_too_long(self, symbol: str, current_time: datetime, timeout_seconds: int) -> bool:
        """Return True when protection should exist but is still not active."""
        requested_at = self._protection_requested_at.get(symbol)
        if requested_at is None:
            return False
        if self._protective_orders.get(symbol):
            return False
        return (current_time - requested_at).total_seconds() >= timeout_seconds

    def is_protected(self, symbol: str) -> bool:
        """Return whether protective orders are active for a symbol."""
        record = self._protective_orders.get(symbol)
        return bool(record and record.get("orders"))

    def get_orders(self) -> Dict[str, Order]:
        """Return a copy of active orders."""
        return self._pending_orders.copy()

    def consume_fills(self, symbol: str = "ES") -> List[Dict[str, Any]]:
        fills = [fill for fill in self._recent_fills if fill.get("symbol") == symbol]
        if fills:
            self._recent_fills = [fill for fill in self._recent_fills if fill.get("symbol") != symbol]
        return fills

    def update_order_status(self, order_id: str, status: OrderStatus, fill_info: Optional[Dict[str, Any]] = None) -> None:
        """Update order status from an external callback."""
        if order_id not in self._pending_orders:
            return

        order = self._pending_orders[order_id]
        prior_filled_quantity = order.filled_quantity
        order.status = status
        order.updated_time = self._current_time(order.symbol)
        if fill_info:
            order.filled_quantity = int(fill_info.get("filled_quantity", order.filled_quantity))
            order.remaining_quantity = max(order.quantity - order.filled_quantity, 0)
            order.filled_price = float(fill_info.get("filled_price", order.filled_price))
            order.filled_time = self._current_time(order.symbol)
            filled_delta = max(order.filled_quantity - prior_filled_quantity, 0)
            if filled_delta > 0:
                self._recent_fills.append(
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": filled_delta,
                        "filled_price": order.filled_price,
                        "filled_time": order.filled_time,
                        "is_protective": order.is_protective,
                        "role": order.role,
                    }
                )
        if status == OrderStatus.FILLED:
            self._filled_orders.append(order)
            self._pending_orders.pop(order_id, None)
            self._last_ack_time = order.updated_time
            self._lifecycle_state = ExecutionState.FILLED if not order.is_protective else ExecutionState.PROTECTED
            if order.is_protective:
                self._last_protective_fill_reason = order.role
                self._cancel_sibling_protection(order.symbol, order.order_id)
        elif status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            self._pending_orders.pop(order_id, None)
            self._last_ack_time = order.updated_time
            if not self.get_active_orders(symbol=order.symbol, is_protective=False):
                self._lifecycle_state = ExecutionState.FLAT
        self._record_event(
            event_type="order_status_updated",
            payload={"status": status.value, "filled_quantity": order.filled_quantity, "filled_price": order.filled_price, "remaining_quantity": order.remaining_quantity, "is_protective": order.is_protective, "role": order.role, "lifecycle_state": self._lifecycle_state.value},
            event_time=order.updated_time,
            symbol=order.symbol,
            action=status.value,
            reason=order.role,
            order_id=order_id,
        )

    def flatten(self, symbol: str = "ES") -> bool:
        """Flatten all positions."""
        self.clear_protection(symbol)
        self._lifecycle_state = ExecutionState.FLATTENING
        self._record_event(
            event_type="flatten_requested",
            payload={"position": self.get_position(symbol), "mock_mode": self._mock_mode},
            event_time=self._current_time(symbol),
            symbol=symbol,
            action="flatten",
            reason="flatten_requested",
        )
        if self._mock_mode:
            position = self.get_position(symbol)
            if position == 0:
                self._lifecycle_state = ExecutionState.FLAT
                self._record_event(
                    event_type="flatten_skipped",
                    payload={"position": position, "mock_mode": True},
                    event_time=self._current_time(symbol),
                    symbol=symbol,
                    action="flatten",
                    reason="already_flat",
                )
                return True
            side = "sell" if position > 0 else "buy"
            order = self.place_order(symbol, abs(position), side, "market", is_protective=False, role="flatten")
            return order is not None
        success = self.client.flatten_all(symbol)
        if success:
            self._last_ack_time = self._current_time(symbol)
        else:
            self._lifecycle_state = ExecutionState.ERROR
        self._record_event(
            event_type="flatten_result",
            payload={"success": success, "mock_mode": self._mock_mode, "lifecycle_state": self._lifecycle_state.value},
            event_time=self._current_time(symbol),
            symbol=symbol,
            action="flatten",
            reason="flatten_completed" if success else "flatten_failed",
        )
        return success

    def mark_position_open(self) -> None:
        """Mark the execution lifecycle after broker position confirmation."""
        if self._lifecycle_state in {ExecutionState.FILLED, ExecutionState.WORKING}:
            self._lifecycle_state = ExecutionState.FILLED
            self._record_event(
                event_type="lifecycle_state_changed",
                payload={"lifecycle_state": self._lifecycle_state.value},
                event_time=self._current_time(),
                action="mark_position_open",
                reason="position_open",
            )

    def mark_position_flat(self) -> None:
        """Mark the execution lifecycle as flat after broker confirmation."""
        self._lifecycle_state = ExecutionState.FLAT
        self._record_event(
            event_type="lifecycle_state_changed",
            payload={"lifecycle_state": self._lifecycle_state.value},
            event_time=self._current_time(),
            action="mark_position_flat",
            reason="position_flat",
        )

    def get_lifecycle_state(self) -> str:
        """Return the current execution lifecycle state."""
        return self._lifecycle_state.value

    def get_last_ack_time(self) -> Optional[datetime]:
        """Return the last broker/local acknowledgement time."""
        return self._last_ack_time

    def get_watchdog_snapshot(self, symbol: str = "ES") -> dict[str, Any]:
        """Expose order-protection and acknowledgement state for diagnostics."""
        active_orders = self.get_active_orders(symbol=symbol)
        active_entry_orders = self.get_active_orders(symbol=symbol, is_protective=False)
        return {
            "execution_state": self._lifecycle_state.value,
            "pending_orders": len(active_orders),
            "tracked_orders": len(self._pending_orders),
            "active_entry_orders": len(active_entry_orders),
            "protected": self.is_protected(symbol),
            "last_ack_time": self._last_ack_time.isoformat() if self._last_ack_time else None,
        }

    def pop_last_protective_fill_reason(self) -> Optional[str]:
        """Return and clear the last protective fill reason."""
        reason = self._last_protective_fill_reason
        self._last_protective_fill_reason = None
        return reason


_executor: Optional[OrderExecutor] = None


def get_executor(force_recreate: bool = False) -> OrderExecutor:
    """Get global executor instance."""
    global _executor
    if force_recreate:
        _executor = None
    if _executor is None:
        _executor = OrderExecutor()
    return _executor
