"""Configuration loader module."""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AccountConfig:
    capital: float = 50000
    max_contracts: int = 5
    default_contracts: int = 1
    risk_per_contract: float = 100
    preferred_id_match: str = "PRAC"
    require_preferred_account: bool = True


@dataclass
class HotZoneConfig:
    name: str
    start: str
    end: str
    timezone: str = "America/Chicago"
    enabled: bool = True
    mode: str = "active"


@dataclass
class SessionsConfig:
    timezone: str = "America/Chicago"
    eth_reset_hour: int = 17
    eth_reset_minute: int = 0
    rth_start_hour: int = 8
    rth_start_minute: int = 30
    rth_end_hour: int = 15
    rth_end_minute: int = 15


@dataclass
class StrategyConfig:
    zone_strategies: Dict[str, str] = field(default_factory=dict)
    atr_length: int = 14
    vwap_session: str = "RTH"
    vwap_source: str = "HLC3"
    orb_range_minutes: int = 15
    orb_entry_buffer_atr: float = 0.5
    orb_stop_atr: float = 2.0
    orb_time_stop_minutes: int = 45
    trend_ma_length: int = 100
    trend_ma_type: str = "EMA"
    vwap_deviation_threshold: float = 1.5
    vwap_confirmation_bars: int = 2
    mr_rsi_length: int = 14
    mr_rsi_oversold: int = 35
    mr_rsi_overbought: int = 65
    mr_band_deviation: float = 2.0
    mr_time_stop_minutes: int = 20
    mr_exit_at_vwap: bool = True
    trade_outside_hotzones: bool = False
    breakeven_trigger_atr: float = 0.5
    profit_lock_atr: float = 0.5
    trailing_stop_atr: float = 1.0
    dynamic_exit_update_cadence_seconds: float = 10.0


@dataclass
class RiskConfig:
    max_daily_loss: float = 600
    max_position_loss: float = 200
    max_consecutive_losses: int = 3
    max_trades_per_hour: int = 6
    max_trades_per_zone: int = 3
    max_daily_trades: int = 10
    use_volatility_sizing: bool = True
    target_daily_risk_pct: float = 1.0
    enable_circuit_breakers: bool = True
    vol_spike_threshold: float = 1.75
    session_timezone: str = "America/Chicago"
    session_reset_hour: int = 17
    session_reset_minute: int = 0


def _default_zone_weights() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Default weighted score matrix per zone."""
    return {
        "Pre-Open": {
            "long": {
                "orb_break": 1.8,
                "opening_drive": 1.55,
                "eth_vwap_distance_z": 0.85,
                "eth_vwap_slope": 0.85,
                "trend_state": 1.35,
                "ofi_zscore": 1.1,
                "execution_state": 0.7,
            },
            "short": {
                "orb_break": 1.9,
                "opening_drive": 1.35,
                "eth_vwap_distance_z": 0.75,
                "eth_vwap_slope": 0.75,
                "trend_state": 1.2,
                "ofi_zscore": 1.0,
                "execution_state": 0.65,
            },
            "flat": {
                "event_state": 1.0,
                "execution_state": 0.9,
                "spread_regime": 0.6,
                "regime_stress": 1.1,
            },
        },
        "Post-Open": {
            "long": {
                "trend_state": 1.75,
                "rth_vwap_slope": 1.0,
                "rth_vwap_distance_z": 0.95,
                "pullback_quality": 1.45,
                "ofi_zscore": 0.95,
                "spread_regime": 0.45,
                "execution_state": 0.55,
            },
            "short": {
                "trend_state": 1.55,
                "rth_vwap_slope": 0.95,
                "rth_vwap_distance_z": 0.8,
                "pullback_quality": 1.3,
                "ofi_zscore": 0.85,
                "spread_regime": 0.4,
                "execution_state": 0.5,
            },
            "flat": {
                "range_state": 0.7,
                "event_state": 1.0,
                "execution_state": 0.9,
                "regime_stress": 1.0,
            },
        },
        "Midday": {
            "long": {
                "extension_state": 1.8,
                "range_state": 1.4,
                "vwap_distance": 1.2,
                "wick_rejection": 1.2,
                "spread_regime": 0.5,
                "execution_state": 0.5,
            },
            "short": {
                "extension_state": 1.85,
                "range_state": 1.35,
                "vwap_distance": 1.25,
                "wick_rejection": 1.1,
                "spread_regime": 0.5,
                "execution_state": 0.5,
            },
            "flat": {
                "trend_state": 1.0,
                "event_state": 0.9,
                "execution_state": 0.8,
                "regime_stress": 1.0,
            },
        },
        "Close-Scalp": {
            "long": {},
            "short": {},
            "flat": {"event_state": 1.0, "execution_state": 1.0, "regime_stress": 1.0},
        },
        "Outside": {
            "long": {
                "trend_state": 1.6,
                "rth_vwap_slope": 1.0,
                "rth_vwap_distance_z": 0.9,
                "pullback_quality": 1.2,
                "ofi_zscore": 0.9,
                "spread_regime": 0.45,
                "execution_state": 0.55,
            },
            "short": {
                "trend_state": 1.6,
                "rth_vwap_slope": 1.0,
                "rth_vwap_distance_z": 0.9,
                "pullback_quality": 1.2,
                "ofi_zscore": 0.9,
                "spread_regime": 0.45,
                "execution_state": 0.55,
            },
            "flat": {
                "range_state": 0.7,
                "event_state": 1.0,
                "execution_state": 0.9,
                "regime_stress": 1.0,
            },
        },
    }


def _default_zone_vetoes() -> Dict[str, Dict[str, Any]]:
    """Default veto thresholds and feature gates per zone."""
    return {
        "Pre-Open": {
            "max_atr_accel": 0.75,
            "max_spread_ticks": 4,
            "reject_orb_middle": True,
            "require_execution_tradeable": True,
            "blocked_regimes": ["STRESS"],
        },
        "Post-Open": {
            "flat_vwap_threshold": 0.08,
            "flat_ema_threshold": 0.08,
            "max_spread_ticks": 4,
            "require_execution_tradeable": True,
            "blocked_regimes": ["RANGE", "STRESS"],
        },
        "Midday": {
            "max_ema_slope": 0.15,
            "max_atr_percentile": 0.7,
            "min_minutes_remaining": 10,
            "require_mean_reversion_confirmation": True,
            "require_execution_tradeable": True,
            "blocked_regimes": ["TREND", "STRESS"],
        },
        "Close-Scalp": {"flatten_only": True},
        "Outside": {
            "max_spread_ticks": 4,
            "require_execution_tradeable": True,
            "blocked_regimes": ["STRESS"],
        },
    }


@dataclass
class AlphaConfig:
    matrix_version: str = "hotzone-v3"
    decision_interval: str = "tick_and_bar"
    min_entry_score: float = 5.0
    min_score_gap: float = 2.0
    exit_decay_score: float = 1.5
    reverse_score_gap: float = 2.5
    full_size_score: float = 6.5
    flat_bias_buffer: float = 0.5
    stop_loss_atr: float = 1.2
    take_profit_atr: Dict[str, float] = field(
        default_factory=lambda: {
            "Pre-Open": 2.2,
            "Post-Open": 2.0,
            "Midday": 1.3,
            "Close-Scalp": 0.8,
            "Outside": 1.8,
        }
    )
    max_hold_minutes: Dict[str, int] = field(
        default_factory=lambda: {
            "Pre-Open": 40,
            "Post-Open": 55,
            "Midday": 20,
            "Close-Scalp": 5,
            "Outside": 30,
        }
    )
    zone_weights: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=_default_zone_weights)
    zone_vetoes: Dict[str, Dict[str, Any]] = field(default_factory=_default_zone_vetoes)


@dataclass
class OrderExecutionConfig:
    use_limit_orders: bool = True
    limit_offset_ticks: int = 1
    market_order_fallback: bool = True
    max_slippage_ticks: int = 3
    cancel_timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class BlackoutConfig:
    pre_minutes: int = 5
    post_minutes: int = 10
    news_times: list = field(default_factory=list)


@dataclass
class VolumeProfileConfig:
    tick_size: float = 0.25
    value_area_pct: float = 0.7
    source: str = "close"
    min_bars: int = 10


@dataclass
class OrderFlowConfig:
    window_seconds: int = 300
    zscore_window: int = 120
    volume_window: int = 60
    quote_rate_baseline: float = 20.0
    stress_spread_ticks: float = 6.0


@dataclass
class RegimeConfig:
    enabled: bool = True
    trend_slope_threshold: float = 0.22
    trend_ofi_threshold: float = 0.6
    range_slope_threshold: float = 0.1
    stress_spread_ticks: float = 5.0
    stress_vol_ratio: float = 1.8
    stress_quote_rate: float = 5.0
    event_cooldown_minutes: int = 10


@dataclass
class WatchdogConfig:
    enabled: bool = True
    feed_stale_seconds: int = 15
    broker_ack_stale_seconds: int = 20
    protection_ack_seconds: int = 5
    stale_order_seconds: int = 60


@dataclass
class ReplayExecutionConfig:
    market_slippage_ticks: float = 0.5
    latency_ms_default: int = 50
    latency_ms_jitter: int = 25
    allow_partial_fills: bool = True
    limit_touch_fill_ratio: float = 1.0
    dsr_trials: int = 20
    volume_mode: str = "auto"
    commission_per_contract: float = 2.5
    exchange_fee_per_contract: float = 1.25
    stress_slippage_ticks: float = 2.0
    limit_fill_penalty_ticks: float = 0.5


def _default_benchmark_zone_strategies() -> Dict[str, str]:
    return {
        "Pre-Open": "ORB",
        "Post-Open": "VWAP_TREND",
        "Midday": "VWAP_MR",
        "Close-Scalp": "FLATTEN_ONLY",
    }


@dataclass
class ValidationConfig:
    benchmarks_enabled: bool = True
    benchmark_zone_strategies: Dict[str, str] = field(default_factory=_default_benchmark_zone_strategies)
    max_features_per_side: int = 7
    walk_forward_train_bars: int = 120
    walk_forward_test_bars: int = 60
    min_positive_test_window_ratio: float = 0.5
    max_zone_pnl_share: float = 0.8
    synthetic_quote_policy: str = "reject"
    max_prac_fill_drift_ticks: float = 2.0


@dataclass
class SafetyConfig:
    prac_only: bool = True
    preferred_account_match: str = "PRAC"
    allow_non_prac_override_env: str = "ALLOW_NON_PRAC_ACCOUNT"


@dataclass
class EventProviderConfig:
    provider: str = "local_file"
    calendar_path: str = "config/events.yaml"
    refresh_seconds: int = 60
    emergency_halt_path: str = "config/emergency_halt.flag"
    emergency_halt_max_age_minutes: int = 1440
    default_timezone: str = "America/New_York"
    impact_windows: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "low": {"pre": 0, "post": 5},
            "medium": {"pre": 5, "post": 10},
            "high": {"pre": 15, "post": 20},
        }
    )


@dataclass
class APIConfig:
    base_url: str = "https://api.topstepx.com"
    ws_url: str = "wss://realtime.topstepx.com/api"
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class ServerConfig:
    health_port: int = 8080
    debug_port: int = 8081
    host: str = "127.0.0.1"
    mcp_enabled: bool = True
    mcp_path: str = "/mcp"
    railway_mcp_url: Optional[str] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/trading.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    console_colors: bool = True


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    sqlite_path: str = "logs/observability.db"
    flush_interval_ms: int = 250
    batch_size: int = 100
    queue_max_size: int = 10000
    retention_days: int = 14
    capture_run_provenance: bool = True
    persist_completed_trades: bool = True
    backfill_missing_trade_records: bool = True
    railway_ingest_url: str = ""
    railway_ingest_api_key: str = ""
    bridge_interval_seconds: float = 30.0
    outbox_path: str = "logs/railway_outbox.db"
    bridge_retry_attempts: int = 5
    bridge_retry_base_seconds: float = 2.0


@dataclass
class ReplayConfig:
    mode: str = "projectx_feed_replay"
    primary_granularity: str = "tick_bbo_if_available"
    fallback_granularity: str = "1m_bars"
    segment_size: int = 2500


@dataclass
class Config:
    account: AccountConfig = field(default_factory=AccountConfig)
    symbols: list = field(default_factory=lambda: ["ES"])
    hot_zones: list = field(default_factory=list)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    order_execution: OrderExecutionConfig = field(default_factory=OrderExecutionConfig)
    blackout: BlackoutConfig = field(default_factory=BlackoutConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    order_flow: OrderFlowConfig = field(default_factory=OrderFlowConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    replay_execution: ReplayExecutionConfig = field(default_factory=ReplayExecutionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    event_provider: EventProviderConfig = field(default_factory=EventProviderConfig)
    api: APIConfig = field(default_factory=APIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


def _dict_to_dataclass(d: Dict[str, Any], cls: type) -> Any:
    """Convert a dictionary to the requested dataclass."""
    if not isinstance(d, dict):
        return d

    field_types = {field_def.name: field_def.type for field_def in dataclass_fields(cls)}
    kwargs = {}
    for key, value in d.items():
        if key not in field_types:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = field_types[key]
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(value, field_type)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_path = config_dir / "default.yaml"

    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    hot_zones = [HotZoneConfig(**zone) for zone in data.get("hot_zones", [])]

    return Config(
        account=_dict_to_dataclass(data.get("account", {}), AccountConfig),
        symbols=data.get("symbols", ["ES"]),
        hot_zones=hot_zones,
        sessions=_dict_to_dataclass(data.get("sessions", {}), SessionsConfig),
        strategy=_dict_to_dataclass(data.get("strategy", {}), StrategyConfig),
        alpha=_dict_to_dataclass(data.get("alpha", {}), AlphaConfig),
        risk=_dict_to_dataclass(data.get("risk", {}), RiskConfig),
        order_execution=_dict_to_dataclass(data.get("order_execution", {}), OrderExecutionConfig),
        blackout=_dict_to_dataclass(data.get("blackout_events", {}), BlackoutConfig),
        volume_profile=_dict_to_dataclass(data.get("volume_profile", {}), VolumeProfileConfig),
        order_flow=_dict_to_dataclass(data.get("order_flow", {}), OrderFlowConfig),
        regime=_dict_to_dataclass(data.get("regime", {}), RegimeConfig),
        watchdog=_dict_to_dataclass(data.get("watchdog", {}), WatchdogConfig),
        replay_execution=_dict_to_dataclass(data.get("replay_execution", {}), ReplayExecutionConfig),
        validation=_dict_to_dataclass(data.get("validation", {}), ValidationConfig),
        safety=_dict_to_dataclass(data.get("safety", {}), SafetyConfig),
        event_provider=_dict_to_dataclass(data.get("event_provider", {}), EventProviderConfig),
        api=_dict_to_dataclass(data.get("api", {}), APIConfig),
        server=_dict_to_dataclass(data.get("server", {}), ServerConfig),
        logging=_dict_to_dataclass(data.get("logging", {}), LoggingConfig),
        observability=_dict_to_dataclass(data.get("observability", {}), ObservabilityConfig),
        replay=_dict_to_dataclass(data.get("replay", {}), ReplayConfig),
    )


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
