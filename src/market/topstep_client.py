"""TopstepX API Client."""
import asyncio
import json
import logging
import os
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import threading
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
import requests
import websockets
import pandas as pd
from dotenv import load_dotenv

from src.config import APIConfig, get_config
from src.observability import get_observability_store

# Load .env file
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    bid: float = 0
    ask: float = 0
    last: float = 0
    volume: int = 0
    volume_is_cumulative: bool = True
    quote_is_synthetic: bool = False
    bid_size: float = 0
    ask_size: float = 0
    last_size: float = 0
    trade_side: str = ""
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.bid and self.ask else 0


@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: int = 0
    entry_price: float = 0
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    authoritative: bool = True
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def direction(self) -> int:
        return 1 if self.quantity > 0 else (-1 if self.quantity < 0 else 0)


@dataclass
class Account:
    """Account information."""
    account_id: str
    name: str = ""
    balance: float = 0
    equity: float = 0
    available: float = 0
    margin_used: float = 0
    open_pnl: float = 0
    realized_pnl: float = 0
    is_practice: bool = False


class TopstepClient:
    """
    TopstepX API Client.
    
    Uses ProjectX-style SignalR API for real-time data.
    Handles:
    - Authentication (OAuth/JWT)
    - Market data streaming via SignalR
    - Order placement and management
    - Position and account queries
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize TopstepX client."""
        root_config = get_config()
        self.config = config or root_config.api
        self.account_config = root_config.account
        self.safety_config = getattr(root_config, "safety", None)
        self.observability = get_observability_store()
        self.base_url = self.config.base_url
        self.ws_url = self.config.ws_url
        # SignalR hubs
        self.user_hub_url = "wss://rtc.topstepx.com/hubs/user"
        self.market_hub_url = "wss://rtc.topstepx.com/hubs/market"
        
        # Auth
        self._access_token: Optional[str] = None
        self._token_expires: float = 0
        self._refresh_token: Optional[str] = None
        self._account_id: Optional[int] = None
        
        # Session
        self._session: Optional[requests.Session] = None
        self._contract_cache: Dict[str, str] = {}
        
        # State
        self._connected: bool = False
        self._mock_mode: bool = False
        self._market_data: Dict[str, MarketData] = {}
        self._positions: Dict[str, Position] = {}
        self._account: Optional[Account] = None
        self._state_lock = threading.RLock()
        self._stream_error: Optional[str] = None
        self._stream_ready = threading.Event()
        self._next_invocation_id: int = 1
        self._active_contract_id: Optional[str] = None
        self._stream_symbol: str = "ES"
        
        # WebSocket
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_market_data: Optional[Callable] = None
        self._on_order_update: Optional[Callable] = None
        self._on_position_update: Optional[Callable] = None
        self._agent_debug_quote_count: int = 0

        # Persistent user-hub listener (orders/positions/trades)
        self._user_hub_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._user_hub_loop: Optional[asyncio.AbstractEventLoop] = None
        self._user_hub_thread: Optional[threading.Thread] = None
        self._user_hub_connected: bool = False
        self._user_hub_stop_requested: bool = False
        self._user_hub_reconnect_delay: float = 5.0
        self._user_hub_max_reconnect_delay: float = 60.0
        self._user_hub_error: Optional[str] = None

    def _record_event(
        self,
        *,
        category: str,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        event_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        self.observability.record_event(
            category=category,
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

    def _is_practice_account(self, account: Dict[str, Any]) -> bool:
        """Return True when an account is obviously a practice account."""
        preferred_match = (
            getattr(self.safety_config, "preferred_account_match", None)
            or self.account_config.preferred_id_match
            or "PRAC"
        ).upper()
        candidate_fields = [
            account.get("id"),
            account.get("name"),
            account.get("accountId"),
            account.get("description"),
        ]

        if any(preferred_match in str(value).upper() for value in candidate_fields if value is not None):
            return True

        return bool(account.get("simulated"))

    def _practice_account_required(self) -> bool:
        return bool(
            getattr(self.safety_config, "prac_only", False)
            or self.account_config.require_preferred_account
        )

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _account_summary(self, account: Dict[str, Any]) -> Account:
        """Map a raw API account payload into the internal representation."""
        balance = self._coerce_float(
            account.get("balance", account.get("cashBalance", account.get("accountBalance", 0))),
            0.0,
        )
        equity = self._coerce_float(
            account.get("equity", account.get("netLiq", account.get("netLiquidationValue", balance))),
            balance,
        )
        available = self._coerce_float(
            account.get("available", account.get("availableBalance", account.get("availableFunds", balance))),
            balance,
        )
        margin_used = self._coerce_float(
            account.get("marginUsed", account.get("marginRequirement", account.get("initialMargin", 0))),
            0.0,
        )
        open_pnl = self._coerce_float(
            account.get("openPnl", account.get("openPnL", account.get("unrealizedPnl", account.get("profitAndLoss", 0)))),
            0.0,
        )
        realized_pnl = self._coerce_float(
            account.get("realizedPnl", account.get("realizedPnL", account.get("closedProfitAndLoss", account.get("realizedProfitAndLoss", 0)))),
            0.0,
        )
        return Account(
            account_id=str(account.get("id")),
            name=str(account.get("name", account.get("accountId", account.get("id", "")))),
            balance=balance,
            equity=equity,
            available=available,
            margin_used=margin_used,
            open_pnl=open_pnl,
            realized_pnl=realized_pnl,
            is_practice=self._is_practice_account(account),
        )

    def _select_account(self, accounts: list[Dict[str, Any]]) -> Optional[Account]:
        """Choose the safest/default tradable account."""
        tradable_accounts = [acc for acc in accounts if acc.get("canTrade")]
        if not tradable_accounts:
            return None

        require_practice = self._practice_account_required()
        practice_accounts = [acc for acc in tradable_accounts if self._is_practice_account(acc)]

        preferred_account_id = os.getenv("PREFERRED_ACCOUNT_ID")
        if preferred_account_id:
            for acc in tradable_accounts:
                if str(acc.get("id")) == preferred_account_id:
                    if require_practice and not self._is_practice_account(acc):
                        logger.error(
                            "Configured PREFERRED_ACCOUNT_ID=%s is not a practice account; refusing startup.",
                            preferred_account_id,
                        )
                        return None
                    return self._account_summary(acc)

        if practice_accounts:
            return self._account_summary(practice_accounts[0])

        if require_practice:
            logger.error(
                "No practice account matched preferred marker '%s'; refusing to select a non-practice account.",
                getattr(self.safety_config, "preferred_account_match", None)
                or self.account_config.preferred_id_match
                or "PRAC",
            )
            return None

        target_balance = os.getenv("TARGET_ACCOUNT_BALANCE", "50000")
        try:
            target = float(target_balance)
        except ValueError:
            target = 50000.0

        for acc in tradable_accounts:
            if abs(acc.get("balance", 0) - target) < 100:
                return self._account_summary(acc)

        for acc in tradable_accounts:
            if acc.get("isVisible"):
                return self._account_summary(acc)

        return self._account_summary(tradable_accounts[0])
        
    def authenticate(self, client_id: Optional[str] = None, client_secret: Optional[str] = None) -> bool:
        """
        Authenticate with TopstepX API.
        
        If no credentials provided, loads from environment variables:
        - EMAIL (used as username)
        - TOPSTEP_API_KEY
        
        Args:
            client_id: Not used (kept for API compatibility)
            client_secret: Not used (kept for API compatibility)
        
        Returns:
            True if authenticated successfully
        """
        # Load from environment
        email = os.getenv("EMAIL")
        api_key = os.getenv("TOPSTEP_API_KEY")
        
        if not email or not api_key:
            logger.error("Missing EMAIL or TOPSTEP_API_KEY in environment")
            self._record_event(
                category="system",
                event_type="authentication_failed",
                payload={"reason": "missing_credentials"},
                event_time=datetime.now(timezone.utc),
                action="authenticate",
                reason="missing_credentials",
            )
            return False
        
        # Use loginKey endpoint with email as username
        try:
            auth_url = f"{self.base_url}/api/Auth/loginKey"
            attempts = max(int(getattr(self.config, "retry_attempts", 1) or 1), 1)
            response = None
            for attempt in range(attempts):
                try:
                    response = requests.post(
                        auth_url,
                        json={"userName": email, "apiKey": api_key},
                        headers={"Content-Type": "application/json"},
                        timeout=self.config.timeout,
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == attempts - 1:
                        raise
                    time.sleep(min(0.5 * (attempt + 1), 2.0))
            assert response is not None
            
            data = response.json()
            if data.get("success") and data.get("token"):
                self._mock_mode = False
                self._access_token = data["token"]
                # Tokens don't seem to expire in the same way
                self._token_expires = time.time() + 86400 * 7  # Assume 7 days
                
                logger.info("Successfully authenticated with TopstepX")
                self._record_event(
                    category="system",
                    event_type="authenticated",
                    payload={"base_url": self.base_url},
                    event_time=datetime.now(timezone.utc),
                    action="authenticate",
                    reason="authentication_succeeded",
                )
                return True
            else:
                error_code = data.get("errorCode", -1)
                error_msgs = {
                    1: "User not found",
                    2: "Password verification failed", 
                    3: "Invalid credentials",
                    4: "App not found",
                    9: "API subscription not found",
                    10: "API key authentication disabled"
                }
                logger.error(
                    "Authentication failed: %s",
                    error_msgs.get(error_code, f"Error {error_code}"),
                )
                self._record_event(
                    category="system",
                    event_type="authentication_failed",
                    payload={"error_code": error_code},
                    event_time=datetime.now(timezone.utc),
                    action="authenticate",
                    reason=error_msgs.get(error_code, f"Error {error_code}"),
                )
                return False
            
        except requests.RequestException as e:
            logger.error("Authentication failed: %s", e)
            self._record_event(
                category="system",
                event_type="authentication_failed",
                payload={"error": str(e)},
                event_time=datetime.now(timezone.utc),
                action="authenticate",
                reason="request_exception",
            )
            return False
    
    def _ensure_auth(self) -> bool:
        """Ensure we have valid authentication."""
        if not self._access_token or time.time() > self._token_expires - 60:
            logger.warning("Token expired or missing - re-authentication required")
            return False
        return True
    
    def _headers(self) -> Dict[str, str]:
        """Get auth headers."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }

    def _normalize_symbol(self, symbol: str) -> str:
        cleaned = str(symbol or "").strip().upper()
        return cleaned[1:] if cleaned.startswith("/") else cleaned

    def _build_hub_url(self, hub_url: str) -> str:
        parsed = urlparse(hub_url)
        scheme = "wss" if parsed.scheme in {"http", "https"} else (parsed.scheme or "wss")
        query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_items["access_token"] = str(self._access_token or "")
        return urlunparse(parsed._replace(scheme=scheme, query=urlencode(query_items)))

    def _lookup_market_snapshot(self, key: str) -> Optional[MarketData]:
        normalized = self._normalize_symbol(key)
        with self._state_lock:
            for candidate in (key, normalized, f"/{normalized}", self._contract_cache.get(normalized, "")):
                if candidate and candidate in self._market_data:
                    return self._market_data[candidate]
        return None

    def _resolve_contract_id(self, symbol: str) -> Optional[str]:
        normalized = self._normalize_symbol(symbol)
        if normalized.startswith("CON."):
            return normalized
        cached = self._contract_cache.get(normalized)
        if cached:
            return cached
        if not self._ensure_auth():
            return None

        queries = [normalized]
        if normalized == "ES":
            queries.extend(["/ES", "E-mini S&P 500"])

        candidates: list[Dict[str, Any]] = []
        for query in queries:
            for live_value in (True, False):
                try:
                    response = self._post_with_retry(
                        f"{self.base_url}/api/Contract/search",
                        {"live": live_value, "searchText": query},
                    )
                except requests.RequestException:
                    continue
                data = response.json()
                candidates.extend(data.get("contracts", []))
            if candidates:
                break

        if not candidates:
            logger.error("Unable to resolve active contract for symbol %s", symbol)
            self._record_event(
                category="market",
                event_type="contract_resolution_failed",
                payload={"symbol": symbol},
                event_time=datetime.now(timezone.utc),
                symbol=symbol,
                action="resolve_contract",
                reason="contract_not_found",
            )
            return None

        def score(contract: Dict[str, Any]) -> tuple[int, int]:
            contract_id = str(contract.get("id", ""))
            name = str(contract.get("name", ""))
            description = str(contract.get("description", ""))
            active = 1 if contract.get("activeContract") else 0
            text = " ".join([contract_id, name, description]).upper()
            rank = 0
            if normalized in text:
                rank += 10
            if name.upper().startswith(normalized):
                rank += 10
            if normalized == "ES" and ("S&P" in description.upper() or name.upper().startswith("ES")):
                rank += 15
            return (active, rank)

        selected = sorted(candidates, key=score, reverse=True)[0]
        contract_id = str(selected.get("id", ""))
        if not contract_id:
            logger.error("Contract search returned no usable contract ID for %s", symbol)
            self._record_event(
                category="market",
                event_type="contract_resolution_failed",
                payload={"symbol": symbol},
                event_time=datetime.now(timezone.utc),
                symbol=symbol,
                action="resolve_contract",
                reason="missing_contract_id",
            )
            return None
        self._contract_cache[normalized] = contract_id
        logger.info("Resolved %s to active contract %s", normalized, contract_id)
        self._record_event(
            category="market",
            event_type="contract_resolved",
            payload={"contract_id": contract_id},
            event_time=datetime.now(timezone.utc),
            symbol=normalized,
            action="resolve_contract",
            reason="contract_resolved",
        )
        return contract_id

    def _decode_signalr_frames(self, payload: str) -> list[Dict[str, Any]]:
        frames: list[Dict[str, Any]] = []
        for part in payload.split("\x1e"):
            part = part.strip()
            if not part:
                continue
            try:
                decoded = json.loads(part)
            except json.JSONDecodeError:
                logger.warning("Ignoring unparseable SignalR frame: %s", part[:200])
                continue
            if isinstance(decoded, dict):
                frames.append(decoded)
        return frames

    def _coerce_signalr_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    return item
        return {}

    async def _signalr_send(self, target: str, *arguments: Any) -> None:
        if not self._ws:
            raise RuntimeError("SignalR socket is not connected")
        invocation = {
            "type": 1,
            "target": target,
            "arguments": list(arguments),
            "invocationId": str(self._next_invocation_id),
        }
        self._next_invocation_id += 1
        await self._ws.send(json.dumps(invocation) + "\x1e")

    async def _signalr_handshake(self) -> None:
        if not self._ws:
            raise RuntimeError("SignalR socket is not connected")
        await self._ws.send(json.dumps({"protocol": "json", "version": 1}) + "\x1e")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=5)
        for frame in self._decode_signalr_frames(raw):
            if frame.get("error"):
                raise RuntimeError(str(frame["error"]))

    def wait_for_market_stream(self, timeout: float = 15.0) -> bool:
        deadline = time.time() + max(float(timeout), 1.0)
        while time.time() < deadline:
            if self._stream_ready.wait(timeout=0.25):
                return self._stream_error is None
            if self._stream_error is not None:
                return False
            if self._ws_thread and not self._ws_thread.is_alive():
                return False
        return self._stream_ready.is_set() and self._stream_error is None

    def get_last_stream_error(self) -> Optional[str]:
        return self._stream_error

    def is_mock_mode(self) -> bool:
        return self._mock_mode

    def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        """POST to the API with simple retry handling for transient failures."""
        attempts = max(int(getattr(self.config, "retry_attempts", 1) or 1), 1)
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retriable = status_code is None or status_code >= 500
                if attempt == attempts - 1 or not retriable:
                    raise
                time.sleep(min(0.5 * (attempt + 1), 2.0))

        assert last_error is not None
        raise last_error

    def list_accounts(self, only_active_accounts: bool = True) -> list[Dict[str, Any]]:
        """Return tradable accounts available to the authenticated user."""
        if not self._ensure_auth():
            return []
        try:
            url = f"{self.base_url}/api/Account/search"
            payload: Dict[str, Any] = {"onlyActiveAccounts": bool(only_active_accounts)}
            response = self._post_with_retry(url, payload)
            accounts = list(response.json().get("accounts", []))
            return [dict(account) for account in accounts if account.get("canTrade")]
        except requests.RequestException as exc:
            logger.error("Failed to list accounts: %s", exc)
            self._record_event(
                category="system",
                event_type="account_listing_failed",
                payload={"error": str(exc)},
                event_time=datetime.now(timezone.utc),
                action="list_accounts",
                reason="request_exception",
            )
            return []

    def select_account(self, account_id: str | int, *, enforce_practice: Optional[bool] = None) -> Optional[Account]:
        """Select an account by ID and make it active for subsequent broker operations."""
        if not self._ensure_auth():
            return None
        try:
            target_account_id = str(account_id).strip()
            accounts = self.list_accounts(only_active_accounts=False)
            if not accounts:
                return None
            account_match = next((account for account in accounts if str(account.get("id")) == target_account_id), None)
            if account_match is None:
                logger.error("Requested account %s not found in account search response.", target_account_id)
                return None
            require_practice = self._practice_account_required() if enforce_practice is None else bool(enforce_practice)
            if require_practice and not self._is_practice_account(account_match):
                logger.error("Refusing non-practice account switch to %s while practice-only policy is enabled.", target_account_id)
                return None

            selected = self._account_summary(account_match)
            self._account_id = int(selected.account_id)
            self._account = selected
            self._mock_mode = False
            self._record_event(
                category="system",
                event_type="account_selected",
                payload={
                    "account_id": selected.account_id,
                    "account_name": selected.name,
                    "balance": selected.balance,
                    "practice": selected.is_practice,
                    "selection_mode": "explicit",
                },
                event_time=datetime.now(timezone.utc),
                action="select_account",
                reason="account_selected",
            )
            return selected
        except Exception as exc:
            logger.error("Failed to select account %s: %s", account_id, exc)
            return None

    def get_active_account_id(self) -> Optional[int]:
        return self._account_id

    def search_orders(
        self,
        *,
        start_timestamp: str,
        end_timestamp: Optional[str] = None,
        account_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search order history for an account in a time range."""
        if not self._ensure_auth():
            return []
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            account = self.get_account()
            if account is None:
                return []
            lookup_account_id = int(account.account_id)
        payload: Dict[str, Any] = {"accountId": lookup_account_id, "startTimestamp": start_timestamp}
        if end_timestamp:
            payload["endTimestamp"] = end_timestamp
        try:
            response = self._post_with_retry(f"{self.base_url}/api/Order/search", payload)
            return list(response.json().get("orders", []))
        except requests.RequestException as exc:
            logger.error("Failed to search orders: %s", exc)
            return []

    def search_trades(
        self,
        *,
        start_timestamp: str,
        end_timestamp: Optional[str] = None,
        account_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search filled trades for an account in a time range."""
        if not self._ensure_auth():
            return []
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            account = self.get_account()
            if account is None:
                return []
            lookup_account_id = int(account.account_id)
        payload: Dict[str, Any] = {"accountId": lookup_account_id, "startTimestamp": start_timestamp}
        if end_timestamp:
            payload["endTimestamp"] = end_timestamp
        try:
            response = self._post_with_retry(f"{self.base_url}/api/Trade/search", payload)
            return list(response.json().get("trades", []))
        except requests.RequestException as exc:
            logger.error("Failed to search trades: %s", exc)
            return []

    def modify_order(
        self,
        order_id: str | int,
        *,
        size: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        account_id: Optional[int] = None,
    ) -> bool:
        """Modify an open order."""
        if not self._ensure_auth():
            return False
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            return False
        payload: Dict[str, Any] = {"accountId": lookup_account_id, "orderId": int(order_id)}
        if size is not None:
            payload["size"] = int(size)
        if limit_price is not None:
            payload["limitPrice"] = float(limit_price)
        if stop_price is not None:
            payload["stopPrice"] = float(stop_price)
        if trail_price is not None:
            payload["trailPrice"] = float(trail_price)
        try:
            response = self._post_with_retry(f"{self.base_url}/api/Order/modify", payload)
            data = response.json()
            return bool(data.get("success"))
        except requests.RequestException as exc:
            logger.error("Failed to modify order %s: %s", order_id, exc)
            return False

    def close_position(self, contract_id: str, *, account_id: Optional[int] = None) -> bool:
        """Close an open position for a specific contract."""
        if not self._ensure_auth():
            return False
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            return False
        payload = {"accountId": lookup_account_id, "contractId": str(contract_id)}
        try:
            response = self._post_with_retry(f"{self.base_url}/api/Position/closeContract", payload)
            data = response.json()
            return bool(data.get("success"))
        except requests.RequestException as exc:
            logger.error("Failed to close position for contract %s: %s", contract_id, exc)
            return False

    async def _invoke_user_hub(self, invocations: list[tuple[str, list[Any]]]) -> bool:
        if not self._ensure_auth():
            return False
        try:
            ws = await websockets.connect(
                self._build_hub_url(self.user_hub_url),
                open_timeout=self.config.timeout,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )
            await ws.send(json.dumps({"protocol": "json", "version": 1}) + "\x1e")
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            for frame in self._decode_signalr_frames(raw):
                if frame.get("error"):
                    raise RuntimeError(str(frame["error"]))
            invocation_id = 1
            for target, args in invocations:
                frame = {
                    "type": 1,
                    "target": target,
                    "arguments": list(args),
                    "invocationId": str(invocation_id),
                }
                invocation_id += 1
                await ws.send(json.dumps(frame) + "\x1e")
            await asyncio.sleep(0.1)
            await ws.close()
            return True
        except Exception as exc:
            logger.error("User hub invocation failed: %s", exc)
            return False

    def _run_async(self, coroutine) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return bool(asyncio.run(coroutine))

        result: dict[str, bool] = {"ok": False}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result["ok"] = bool(loop.run_until_complete(coroutine))
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=max(float(self.config.timeout), 5.0))
        return bool(result["ok"])

    def subscribe_user_updates(self, account_id: Optional[int] = None) -> bool:
        """Subscribe user-hub channels for accounts, orders, positions, and trades."""
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            account = self.get_account()
            if account is None:
                return False
            lookup_account_id = int(account.account_id)
        return self._run_async(
            self._invoke_user_hub(
                [
                    ("SubscribeAccounts", []),
                    ("SubscribeOrders", [lookup_account_id]),
                    ("SubscribePositions", [lookup_account_id]),
                    ("SubscribeTrades", [lookup_account_id]),
                ]
            )
        )

    def unsubscribe_user_updates(self, account_id: Optional[int] = None) -> bool:
        """Unsubscribe user-hub channels for accounts, orders, positions, and trades."""
        lookup_account_id = int(account_id) if account_id is not None else self._account_id
        if lookup_account_id is None:
            return False
        return self._run_async(
            self._invoke_user_hub(
                [
                    ("UnsubscribeAccounts", []),
                    ("UnsubscribeOrders", [lookup_account_id]),
                    ("UnsubscribePositions", [lookup_account_id]),
                    ("UnsubscribeTrades", [lookup_account_id]),
                ]
            )
        )

    async def _user_hub_connect(self) -> None:
        """Open persistent user-hub WebSocket, handshake, and subscribe to orders/positions."""
        if not self._ensure_auth():
            raise RuntimeError("User hub requires authentication")
        lookup_account_id = self._account_id
        if lookup_account_id is None:
            acc = self.get_account()
            if acc is None:
                raise RuntimeError("No account for user hub subscription")
            lookup_account_id = int(acc.account_id)
        self._user_hub_ws = await websockets.connect(
            self._build_hub_url(self.user_hub_url),
            open_timeout=self.config.timeout,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )
        await self._user_hub_ws.send(json.dumps({"protocol": "json", "version": 1}) + "\x1e")
        raw = await asyncio.wait_for(self._user_hub_ws.recv(), timeout=5)
        for frame in self._decode_signalr_frames(raw):
            if frame.get("error"):
                raise RuntimeError(str(frame["error"]))
        for target, args in [
            ("SubscribeAccounts", []),
            ("SubscribeOrders", [lookup_account_id]),
            ("SubscribePositions", [lookup_account_id]),
            ("SubscribeTrades", [lookup_account_id]),
        ]:
            inv = {"type": 1, "target": target, "arguments": list(args), "invocationId": str(id(self._user_hub_ws))}
            await self._user_hub_ws.send(json.dumps(inv) + "\x1e")
        await asyncio.sleep(0.2)
        self._user_hub_connected = True
        self._user_hub_error = None
        logger.info("User hub connected and subscribed for account %s", lookup_account_id)
        self._record_event(
            category="market",
            event_type="user_hub_connected",
            payload={"account_id": lookup_account_id},
            event_time=datetime.now(timezone.utc),
            action="user_hub_connect",
            reason="user_hub_connected",
        )

    def _handle_user_hub_message(self, data: Dict[str, Any]) -> None:
        """Dispatch user-hub invocation to order/position callbacks."""
        if data.get("type") != 1:
            return
        target = str(data.get("target", "")).lower()
        arguments = list(data.get("arguments", []))
        payload = self._coerce_signalr_payload(arguments[0] if arguments else data)
        if not payload:
            return
        order_targets = ("orderupdated", "orderupdate", "order", "onorderupdate")
        position_targets = ("positionupdated", "positionupdate", "position", "onpositionupdate")
        if target in order_targets and self._on_order_update:
            order_id = payload.get("orderId") or payload.get("id") or payload.get("orderID")
            status = payload.get("status") or payload.get("orderStatus") or payload.get("state")
            if order_id is not None:
                normalized = {
                    "orderId": order_id,
                    "id": order_id,
                    "status": str(status or "").lower(),
                    "filledQuantity": payload.get("filledQuantity", payload.get("filled_quantity", 0)),
                    "filledPrice": payload.get("filledPrice", payload.get("filled_price", 0.0)),
                }
                try:
                    self._on_order_update(normalized)
                except Exception as exc:
                    logger.warning("on_order_update callback error: %s", exc)
        if target in position_targets and self._on_position_update:
            try:
                self._on_position_update(payload)
            except Exception as exc:
                logger.warning("on_position_update callback error: %s", exc)

    async def _user_hub_listen(self) -> None:
        """Listen to user-hub WebSocket and dispatch messages."""
        if not self._user_hub_ws:
            return
        try:
            async for message in self._user_hub_ws:
                for frame in self._decode_signalr_frames(message):
                    if frame.get("type") == 7:
                        raise RuntimeError(str(frame.get("error", "SignalR closed")))
                    self._handle_user_hub_message(frame)
        except Exception as e:
            self._user_hub_connected = False
            self._user_hub_error = str(e)
            logger.warning("User hub listen error: %s", e)
            self._record_event(
                category="market",
                event_type="user_hub_listen_error",
                payload={"error": str(e)},
                event_time=datetime.now(timezone.utc),
                action="user_hub_listen",
                reason="user_hub_listen_error",
            )
            raise

    async def _user_hub_run_loop(self) -> None:
        """Connect and listen with reconnect and backoff."""
        delay = self._user_hub_reconnect_delay
        while not self._user_hub_stop_requested:
            try:
                await self._user_hub_connect()
                delay = self._user_hub_reconnect_delay
                await self._user_hub_listen()
            except Exception as e:
                self._user_hub_connected = False
                self._user_hub_error = str(e)
                if self._user_hub_stop_requested:
                    break
                logger.warning("User hub disconnected, reconnecting in %.1fs: %s", delay, e)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, self._user_hub_max_reconnect_delay)
        self._user_hub_connected = False

    def start_user_hub_listener(self) -> None:
        """Start persistent user-hub listener in a background thread."""
        if self._mock_mode or (self._on_order_update is None and self._on_position_update is None):
            return
        if self._user_hub_thread and self._user_hub_thread.is_alive():
            return
        self._user_hub_stop_requested = False
        self._user_hub_error = None

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._user_hub_loop = loop
            try:
                loop.run_until_complete(self._user_hub_run_loop())
            except Exception:
                logger.exception("User hub thread terminated")
            finally:
                self._user_hub_connected = False
                self._user_hub_loop = None

        self._user_hub_thread = threading.Thread(target=run, name="user-hub-listener", daemon=True)
        self._user_hub_thread.start()

    def stop_user_hub_listener(self) -> None:
        """Stop user-hub listener and close connection."""
        self._user_hub_stop_requested = True
        if self._user_hub_ws and self._user_hub_loop and self._user_hub_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._user_hub_ws.close(), self._user_hub_loop).result(timeout=5)
            except Exception:
                pass
        self._user_hub_ws = None
        if self._user_hub_thread:
            self._user_hub_thread.join(timeout=5)
            self._user_hub_thread = None
        self._user_hub_connected = False
        self._record_event(
            category="market",
            event_type="user_hub_stopped",
            payload={},
            event_time=datetime.now(timezone.utc),
            action="stop_user_hub",
            reason="user_hub_stopped",
        )

    def is_user_hub_connected(self) -> bool:
        """Return whether the persistent user-hub connection is active."""
        return self._user_hub_connected

    def get_user_hub_error(self) -> Optional[str]:
        """Return last user-hub error if any."""
        return self._user_hub_error

    def get_account(self) -> Optional[Account]:
        """Get account information."""
        if not self._ensure_auth():
            return None
        
        try:
            url = f"{self.base_url}/api/Account/search"
            response = self._post_with_retry(url, {})
            
            data = response.json()
            accounts = data.get("accounts", [])
            selected = self._select_account(accounts)
            if selected is None:
                self._record_event(
                    category="system",
                    event_type="account_selection_failed",
                    payload={"accounts_returned": len(accounts)},
                    event_time=datetime.now(timezone.utc),
                    action="get_account",
                    reason="no_tradable_account",
                )
                return None

            self._account_id = int(selected.account_id)
            self._account = selected
            self._mock_mode = False
            logger.info(
                "Using account: %s (%s, $%s, practice=%s)",
                selected.name,
                selected.account_id,
                selected.balance,
                selected.is_practice,
            )
            self._record_event(
                category="system",
                event_type="account_selected",
                payload={"account_id": selected.account_id, "account_name": selected.name, "balance": selected.balance, "practice": selected.is_practice},
                event_time=datetime.now(timezone.utc),
                action="get_account",
                reason="account_selected",
            )
            return self._account
            
        except requests.RequestException as e:
            logger.error("Failed to get account: %s", e)
            self._record_event(
                category="system",
                event_type="account_lookup_failed",
                payload={"error": str(e)},
                event_time=datetime.now(timezone.utc),
                action="get_account",
                reason="request_exception",
            )
            return None
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        positions, _ = self.get_positions_snapshot()
        return positions or {}

    def get_positions_snapshot(self) -> tuple[Optional[Dict[str, Position]], Optional[str]]:
        """Get broker position state with explicit success/failure status."""
        if not self._ensure_auth() or not self._account_id:
            return None, "auth_unavailable"
        
        try:
            url = f"{self.base_url}/api/Position/searchOpen"
            response = self._post_with_retry(url, {"accountId": self._account_id})
            
            data = response.json()
            positions = data.get("positions", [])
            new_positions: Dict[str, Position] = {}
            for pos in positions:
                raw_size = pos.get("size", 0)
                side = str(pos.get("type", pos.get("side", ""))).lower()
                signed_size = -abs(raw_size) if side in {"short", "sell", "ask", "-1"} else raw_size
                symbol = str(pos.get("contractId", "ES"))
                snapshot = self._lookup_market_snapshot(symbol)
                new_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=signed_size,
                    entry_price=pos.get("averagePrice", 0),
                    current_price=(snapshot.last if snapshot is not None else pos.get("averagePrice", 0)) or pos.get("averagePrice", 0),
                    unrealized_pnl=pos.get("profitAndLoss", 0),
                    realized_pnl=0
                )
            with self._state_lock:
                self._positions = new_positions
                return dict(self._positions), None
            
        except requests.RequestException as e:
            logger.error("Failed to get positions: %s", e)
            self._record_event(
                category="execution",
                event_type="position_lookup_failed",
                payload={"error": str(e), "account_id": self._account_id},
                event_time=datetime.now(timezone.utc),
                symbol=symbol if 'symbol' in locals() else self._stream_symbol,
                action="get_positions",
                reason="request_exception",
            )
            return None, str(e)

    def get_open_orders_snapshot(self, symbol: Optional[str] = None) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Get broker open orders with explicit success/failure status."""
        if not self._ensure_auth() or not self._account_id:
            return None, "auth_unavailable"

        payload: dict[str, Any] = {"accountId": self._account_id}
        contract_id: Optional[str] = None
        if symbol:
            contract_id = self._resolve_contract_id(symbol)
            if not contract_id:
                return None, "contract_resolution_failed"
            payload["contractId"] = contract_id

        try:
            url = f"{self.base_url}/api/Order/searchOpen"
            response = self._post_with_retry(url, payload)
            data = response.json()
            orders = list(data.get("orders", []))
            normalized = self._normalize_symbol(symbol or self._stream_symbol)
            filtered_orders: list[dict[str, Any]] = []
            for order in orders:
                order_contract_id = str(order.get("contractId", ""))
                if symbol:
                    if contract_id and order_contract_id and order_contract_id == contract_id:
                        filtered_orders.append(order)
                        continue
                    order_symbol = self._normalize_symbol(
                        str(order.get("symbol", order.get("symbolName", order.get("contractName", ""))))
                    )
                    if order_symbol and order_symbol == normalized:
                        filtered_orders.append(order)
                else:
                    filtered_orders.append(order)
            return filtered_orders, None
        except requests.RequestException as e:
            logger.error("Failed to get open orders: %s", e)
            self._record_event(
                category="execution",
                event_type="open_order_lookup_failed",
                payload={"error": str(e), "account_id": self._account_id, "symbol": symbol},
                event_time=datetime.now(timezone.utc),
                symbol=symbol or self._stream_symbol,
                action="get_open_orders",
                reason="request_exception",
            )
            return None, str(e)

    def get_position(self, symbol: str = "ES") -> Position:
        """Get position for specific symbol."""
        positions, error = self.get_positions_snapshot()
        if positions and symbol in positions:
            return positions[symbol]

        requested = symbol.upper()
        for contract_id, position in (positions or {}).items():
            normalized = str(contract_id).upper()
            if requested in normalized or normalized in requested:
                return position

        if error:
            with self._state_lock:
                cached_positions = dict(self._positions)
            if cached_positions:
                if symbol in cached_positions:
                    return replace(cached_positions[symbol], authoritative=False)
                for contract_id, position in cached_positions.items():
                    normalized = str(contract_id).upper()
                    if requested in normalized or normalized in requested:
                        return replace(position, authoritative=False)
            logger.warning("Position lookup failed for %s with no authoritative broker snapshot: %s", symbol, error)
            return Position(symbol=symbol, authoritative=False)

        return Position(symbol=symbol)
    
    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # "buy" or "sell"
        order_type: str = "limit",  # "limit" or "market"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Optional[str]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol (e.g., "ES" or contract ID)
            quantity: Number of contracts
            side: "buy" or "sell"
            order_type: "limit" or "market"
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: "day", "gtc", "ioc", "fok" (not used in this API)
        
        Returns:
            Order ID if successful, None otherwise
        """
        if not self._ensure_auth() or not self._account_id:
            # Try to get account first
            self.get_account()
            if not self._account_id:
                logger.error("No account ID available")
                return None

        if self._practice_account_required() and self._account is not None and not self._account.is_practice:
            logger.error(
                "Refusing order placement on non-practice account %s while practice-only safety is enabled.",
                self._account.account_id,
            )
            return None
        
        # Map order types to API enums
        type_map = {"limit": 1, "market": 2, "stoplimit": 3, "stop": 4}
        side_map = {"buy": 0, "sell": 1, "bid": 0, "ask": 1}
        
        contract_id = self._resolve_contract_id(symbol)
        if not contract_id:
            return None

        order_payload = {
            "accountId": self._account_id,
            "contractId": contract_id,
            "type": type_map.get(order_type.lower(), 1),
            "side": side_map.get(side.lower(), 0),
            "size": abs(quantity)
        }
        
        if order_type.lower() == "limit" and limit_price:
            order_payload["limitPrice"] = limit_price
        if stop_price:
            order_payload["stopPrice"] = stop_price
        
        try:
            url = f"{self.base_url}/api/Order/place"
            response = self._post_with_retry(url, order_payload)
            
            data = response.json()
            if data.get("success"):
                order_id = str(data.get("orderId"))
                logger.info(
                    "broker_order_submit order_id=%s side=%s quantity=%s symbol=%s order_type=%s limit_price=%s stop_price=%s account_id=%s",
                    order_id,
                    side,
                    quantity,
                    symbol,
                    order_type,
                    limit_price,
                    stop_price,
                    self._account_id,
                )
                self._record_event(
                    category="execution",
                    event_type="broker_order_submit",
                    payload={"side": side, "quantity": quantity, "symbol": symbol, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "account_id": self._account_id},
                    event_time=datetime.now(timezone.utc),
                    symbol=symbol,
                    action="submit_order",
                    reason="broker_order_submit",
                    order_id=order_id,
                )
                return order_id
            else:
                logger.error(
                    "broker_order_submit_failed error=%s side=%s quantity=%s symbol=%s order_type=%s limit_price=%s stop_price=%s account_id=%s",
                    data.get("errorMessage", "Unknown error"),
                    side,
                    quantity,
                    symbol,
                    order_type,
                    limit_price,
                    stop_price,
                    self._account_id,
                )
                self._record_event(
                    category="execution",
                    event_type="broker_order_submit_failed",
                    payload={"error": data.get("errorMessage", "Unknown error"), "side": side, "quantity": quantity, "symbol": symbol, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "account_id": self._account_id},
                    event_time=datetime.now(timezone.utc),
                    symbol=symbol,
                    action="submit_order",
                    reason="broker_order_submit_failed",
                )
                return None
            
        except requests.RequestException as e:
            logger.error(
                "broker_order_submit_failed error=%s side=%s quantity=%s symbol=%s order_type=%s limit_price=%s stop_price=%s account_id=%s",
                e,
                side,
                quantity,
                symbol,
                order_type,
                limit_price,
                stop_price,
                self._account_id,
            )
            self._record_event(
                category="execution",
                event_type="broker_order_submit_failed",
                payload={"error": str(e), "side": side, "quantity": quantity, "symbol": symbol, "order_type": order_type, "limit_price": limit_price, "stop_price": stop_price, "account_id": self._account_id},
                event_time=datetime.now(timezone.utc),
                symbol=symbol,
                action="submit_order",
                reason="request_exception",
            )
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._ensure_auth():
            return False
        
        try:
            url = f"{self.base_url}/api/Order/cancel"
            response = self._post_with_retry(
                url,
                {"orderId": int(order_id), "accountId": self._account_id},
            )
            
            logger.info("broker_order_cancelled order_id=%s account_id=%s", order_id, self._account_id)
            self._record_event(
                category="execution",
                event_type="broker_order_cancelled",
                payload={"account_id": self._account_id},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="cancel_order",
                reason="broker_order_cancelled",
                order_id=order_id,
            )
            return True
            
        except requests.RequestException as e:
            logger.error("broker_order_cancel_failed order_id=%s account_id=%s error=%s", order_id, self._account_id, e)
            self._record_event(
                category="execution",
                event_type="broker_order_cancel_failed",
                payload={"account_id": self._account_id, "error": str(e)},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="cancel_order",
                reason="request_exception",
                order_id=order_id,
            )
            return False
    
    def flatten_all(self, symbol: str = "ES") -> bool:
        """Flatten all positions for symbol."""
        position = self.get_position(symbol)
        
        if position.is_flat:
            return True
        
        side = "sell" if position.quantity > 0 else "buy"
        return self.place_order(symbol, abs(position.quantity), side, "market") is not None
    
    # Market Data Streaming
    
    async def _ws_connect(self):
        """Connect to WebSocket for market data."""
        if not self._ensure_auth():
            raise RuntimeError("Authentication required before starting market stream")
        
        try:
            contract_id = self._resolve_contract_id(self._stream_symbol)
            if not contract_id:
                raise RuntimeError(f"Unable to resolve live contract for {self._stream_symbol}")
            self._active_contract_id = contract_id
            self._ws = await websockets.connect(
                self._build_hub_url(self.market_hub_url),
                open_timeout=self.config.timeout,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )
            await self._signalr_handshake()
            self._connected = True
            logger.info("Market hub connected")
            self._record_event(
                category="market",
                event_type="market_hub_connected",
                payload={"contract_id": contract_id},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="connect_stream",
                reason="market_hub_connected",
            )
            await self._signalr_send("SubscribeContractQuotes", contract_id)
            await self._signalr_send("SubscribeContractTrades", contract_id)
            logger.info("Subscribed to live market data for contract %s", contract_id)
            self._record_event(
                category="market",
                event_type="market_subscriptions_ready",
                payload={"contract_id": contract_id},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="subscribe_stream",
                reason="subscriptions_ready",
            )
        except Exception as e:
            self._stream_error = str(e)
            self._stream_ready.set()
            logger.error("WebSocket connection failed: %s", e)
            self._connected = False
            self._record_event(
                category="market",
                event_type="market_hub_connect_failed",
                payload={"error": str(e)},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="connect_stream",
                reason="websocket_connection_failed",
            )
            raise
    
    async def _ws_listen(self):
        """Listen to WebSocket messages."""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                for frame in self._decode_signalr_frames(message):
                    await self._handle_ws_message(frame)
                
        except Exception as e:
            self._stream_error = str(e)
            self._stream_ready.set()
            logger.error("WebSocket listen error: %s", e)
            self._connected = False
            self._record_event(
                category="market",
                event_type="market_stream_error",
                payload={"error": str(e)},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="listen_stream",
                reason="websocket_listen_error",
            )
    
    async def _handle_ws_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        if data.get("type") == 6:
            return
        if data.get("type") == 7:
            raise RuntimeError(str(data.get("error", "SignalR closed by remote host")))
        if data.get("type") != 1:
            return

        target = str(data.get("target", ""))
        arguments = list(data.get("arguments", []))

        if target == "GatewayQuote" and len(arguments) >= 2:
            contract_id = str(arguments[0])
            payload = self._coerce_signalr_payload(arguments[1])
            symbol_name = str(payload.get("symbolName", payload.get("symbol", "ES")) or "ES")
            root_symbol = self._normalize_symbol(symbol_name or payload.get("symbol", "ES"))
            quote = MarketData(
                symbol=root_symbol,
                bid=float(payload.get("bestBid", payload.get("bid", 0)) or 0),
                ask=float(payload.get("bestAsk", payload.get("ask", 0)) or 0),
                last=float(payload.get("lastPrice", payload.get("last", 0)) or 0),
                volume=int(payload.get("volume", 0) or 0),
                bid_size=float(payload.get("bidSize", 0) or 0),
                ask_size=float(payload.get("askSize", 0) or 0),
                timestamp=pd.Timestamp(payload.get("timestamp") or payload.get("lastUpdated") or datetime.now(timezone.utc)).to_pydatetime(warn=False),
            )
            aliases = {contract_id, root_symbol, symbol_name}
            with self._state_lock:
                for alias in aliases:
                    if alias:
                        self._market_data[alias] = quote
            self.observability.record_market_tick(
                {
                    "run_id": self.observability.get_run_id(),
                    "symbol": root_symbol,
                    "contract_id": contract_id,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "last": quote.last,
                    "volume": quote.volume,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "last_size": quote.last_size,
                    "volume_is_cumulative": quote.volume_is_cumulative,
                    "quote_is_synthetic": quote.quote_is_synthetic,
                    "trade_side": quote.trade_side,
                    "latency_ms": quote.latency_ms,
                    "source": "GatewayQuote",
                    "timestamp": quote.timestamp,
                }
            )
            
            if not self._stream_ready.is_set():
                self._stream_error = None
                self._stream_ready.set()
                logger.info("Received first live quote for %s at %s", root_symbol, quote.last or quote.mid)
                self._record_event(
                    category="market",
                    event_type="first_live_quote",
                    payload={"price": quote.last or quote.mid, "contract_id": contract_id},
                    event_time=quote.timestamp,
                    symbol=root_symbol,
                    action="receive_quote",
                    reason="stream_ready",
                )

            if self._on_market_data:
                self._on_market_data(quote)
            self._agent_debug_quote_count += 1
            if self._agent_debug_quote_count % 300 == 0:
                logger.info(
                    "market_stream_heartbeat symbol=%s quotes=%s last_price=%s",
                    root_symbol,
                    self._agent_debug_quote_count,
                    quote.last or quote.mid,
                )

        elif target == "GatewayTrade" and len(arguments) >= 2:
            contract_id = str(arguments[0])
            payload = self._coerce_signalr_payload(arguments[1])
            symbol_id = str(payload.get("symbolId", self._stream_symbol))
            prior = self._lookup_market_snapshot(contract_id) or self._lookup_market_snapshot(self._stream_symbol)
            root_symbol = prior.symbol if prior is not None else self._normalize_symbol(symbol_id)
            prior = prior or MarketData(symbol=root_symbol)
            quote = MarketData(
                symbol=root_symbol or prior.symbol,
                bid=prior.bid,
                ask=prior.ask,
                last=float(payload.get("price", prior.last or 0) or 0),
                volume=int(payload.get("volume", prior.volume or 0) or 0),
                bid_size=prior.bid_size,
                ask_size=prior.ask_size,
                last_size=float(payload.get("size", payload.get("volume", prior.last_size or 0)) or 0),
                trade_side="buy" if int(payload.get("type", 0) or 0) == 0 else "sell",
                timestamp=pd.Timestamp(payload.get("timestamp") or datetime.now(timezone.utc)).to_pydatetime(warn=False),
            )
            with self._state_lock:
                for alias in {contract_id, root_symbol, quote.symbol}:
                    if alias:
                        self._market_data[alias] = quote
            self.observability.record_market_tick(
                {
                    "run_id": self.observability.get_run_id(),
                    "symbol": quote.symbol,
                    "contract_id": contract_id,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "last": quote.last,
                    "volume": quote.volume,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "last_size": quote.last_size,
                    "volume_is_cumulative": quote.volume_is_cumulative,
                    "quote_is_synthetic": quote.quote_is_synthetic,
                    "trade_side": quote.trade_side,
                    "latency_ms": quote.latency_ms,
                    "source": "GatewayTrade",
                    "timestamp": quote.timestamp,
                }
            )

            if not self._stream_ready.is_set():
                self._stream_error = None
                self._stream_ready.set()
                logger.info("Received first live trade for %s at %s", quote.symbol, quote.last)
                self._record_event(
                    category="market",
                    event_type="first_live_trade",
                    payload={"price": quote.last, "contract_id": contract_id, "trade_side": quote.trade_side},
                    event_time=quote.timestamp,
                    symbol=quote.symbol,
                    action="receive_trade",
                    reason="stream_ready",
                )

            if self._on_market_data:
                self._on_market_data(quote)
    
    def start_market_stream(
        self,
        symbol: str = "ES",
        on_market_data: Optional[Callable] = None,
        on_order_update: Optional[Callable] = None,
        on_position_update: Optional[Callable] = None
    ):
        """Start market data streaming in background thread."""
        self._on_market_data = on_market_data
        self._on_order_update = on_order_update
        self._on_position_update = on_position_update
        self._stream_symbol = self._normalize_symbol(symbol)
        self._stream_error = None
        self._stream_ready.clear()
        self._next_invocation_id = 1
        if self._ws_thread and self._ws_thread.is_alive():
            logger.info("Market stream already running")
            self._record_event(
                category="market",
                event_type="market_stream_already_running",
                payload={"symbol": self._stream_symbol},
                event_time=datetime.now(timezone.utc),
                symbol=self._stream_symbol,
                action="start_stream",
                reason="already_running",
            )
            return
        
        def run_ws():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ws_loop = loop
            try:
                loop.run_until_complete(self._ws_connect())
                loop.run_until_complete(self._ws_listen())
            except Exception:
                logger.exception("Market stream thread terminated")
            finally:
                self._connected = False
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
        
        self._ws_thread = threading.Thread(target=run_ws, name=f"market-stream-{symbol.lower()}", daemon=True)
        self._ws_thread.start()
        self.start_user_hub_listener()
        self._record_event(
            category="market",
            event_type="market_stream_started",
            payload={"symbol": self._stream_symbol},
            event_time=datetime.now(timezone.utc),
            symbol=self._stream_symbol,
            action="start_stream",
            reason="market_stream_started",
        )

    def stop_market_stream(self):
        """Stop market data streaming and user-hub listener."""
        self.stop_user_hub_listener()
        thread_alive_after_join = False
        if self._ws and self._ws_loop and self._ws_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._ws.close(), self._ws_loop)
            try:
                future.result(timeout=5)
            except FutureTimeoutError:
                logger.warning("Timed out waiting for websocket close during shutdown")
                self._record_event(
                    category="market",
                    event_type="market_stream_stop_timeout",
                    payload={"symbol": self._stream_symbol},
                    event_time=datetime.now(timezone.utc),
                    symbol=self._stream_symbol,
                    action="stop_stream",
                    reason="websocket_close_timeout",
                )
            except Exception as exc:
                logger.warning("Websocket close during shutdown failed: %s", exc)
                self._record_event(
                    category="market",
                    event_type="market_stream_stop_failed",
                    payload={"symbol": self._stream_symbol, "error": str(exc)},
                    event_time=datetime.now(timezone.utc),
                    symbol=self._stream_symbol,
                    action="stop_stream",
                    reason="websocket_close_failed",
                )
        elif self._ws:
            logger.warning("Closing websocket without a running stream loop; falling back to local event loop close")
            try:
                asyncio.run(self._ws.close())
            except Exception as exc:
                logger.warning("Fallback websocket close failed: %s", exc)
                self._record_event(
                    category="market",
                    event_type="market_stream_stop_failed",
                    payload={"symbol": self._stream_symbol, "error": str(exc)},
                    event_time=datetime.now(timezone.utc),
                    symbol=self._stream_symbol,
                    action="stop_stream",
                    reason="fallback_websocket_close_failed",
                )
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
            thread_alive_after_join = self._ws_thread.is_alive()
            if thread_alive_after_join:
                logger.warning("Market stream thread did not exit before shutdown timeout")
                self._record_event(
                    category="market",
                    event_type="market_stream_thread_join_timeout",
                    payload={"symbol": self._stream_symbol},
                    event_time=datetime.now(timezone.utc),
                    symbol=self._stream_symbol,
                    action="stop_stream",
                    reason="thread_join_timeout",
                )
            else:
                self._ws_thread = None
        self._ws = None
        self._ws_loop = None
        self._connected = False
        self._record_event(
            category="market",
            event_type="market_stream_stopped",
            payload={"symbol": self._stream_symbol, "thread_alive_after_join": thread_alive_after_join},
            event_time=datetime.now(timezone.utc),
            symbol=self._stream_symbol,
            action="stop_stream",
            reason="market_stream_stopped",
        )
    
    def get_market_data(self, symbol: str = "ES") -> Optional[MarketData]:
        """Get current market data for symbol."""
        return self._lookup_market_snapshot(symbol)
    
    # Offline execution (replay and tests only)

    def enable_mock_mode(self):
        """Enable offline execution: no real API calls; synthetic account and data. Used only by replay and tests. Practice account = real Topstep PRAC (use start, not this)."""
        logger.info("Offline execution enabled (replay/tests only)")
        self._mock_mode = True
        self._account = Account(account_id="SIM-PRAC", name="Practice Sim", balance=50000, is_practice=True)
        self._record_event(
            category="system",
            event_type="mock_mode_enabled",
            payload={"account_id": "SIM-PRAC"},
            event_time=datetime.now(timezone.utc),
            action="enable_mock_mode",
            reason="mock_mode_enabled",
        )
        with self._state_lock:
            self._market_data["ES"] = MarketData(
                symbol="ES",
                bid=5900.0,
                ask=5900.25,
                last=5900.0,
                volume=1000,
                bid_size=10,
                ask_size=10,
                last_size=1,
                timestamp=datetime.now(timezone.utc)
            )
    
    def update_mock_price(self, price: float):
        """Update mock price for testing."""
        with self._state_lock:
            self._market_data["ES"] = MarketData(
                symbol="ES",
                bid=price - 0.25,
                ask=price + 0.25,
                last=price,
                volume=1000,
                bid_size=10,
                ask_size=10,
                last_size=1,
                timestamp=datetime.now(timezone.utc)
            )

    def reset_mock_state(self):
        """Clear cached mock data and account state."""
        with self._state_lock:
            self._mock_mode = False
            self._market_data = {}
            self._positions = {}
            self._account = None
            self._account_id = None
            self._active_contract_id = None
            self._stream_error = None
            self._stream_ready.clear()


# Global client instance
_client: Optional[TopstepClient] = None


def get_client(force_recreate: bool = False) -> TopstepClient:
    """Get global TopstepX client instance."""
    global _client
    if force_recreate:
        _client = None
    if _client is None:
        _client = TopstepClient()
    return _client
