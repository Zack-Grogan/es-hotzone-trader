"""Hot-Zone Scheduler - Time-gated trading window management."""
import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from enum import Enum
from typing import Optional, List
import pytz

from src.config import HotZoneConfig, get_config

logger = logging.getLogger(__name__)


class ZoneState(Enum):
    """Hot-zone state machine states."""
    INACTIVE = "inactive"       # Outside all zones
    ACTIVE = "active"          # Trading allowed
    FLATTEN_ONLY = "flatten_only"  # No new entries, manage exits only
    CLOSING = "closing"         # Near zone end, tightening exits


@dataclass
class ZoneInfo:
    """Information about current zone."""
    name: str
    state: ZoneState
    start_time: datetime
    end_time: datetime
    minutes_remaining: float
    is_first_bar: bool
    is_last_bar: bool


class HotZoneScheduler:
    """
    Manages hot-zone time windows for trading.
    
    Implements time-gated trading based on configured zones.
    Handles timezone conversion and state transitions.
    """
    
    def __init__(self, config=None):
        """Initialize scheduler with configuration."""
        self.config = config or get_config()
        self.hot_zones: List[HotZoneConfig] = self.config.hot_zones
        self._current_zone: Optional[ZoneInfo] = None
        self._previous_zone: Optional[str] = None
        self._zone_entry_time: Optional[datetime] = None
        self._bars_in_zone: int = 0
        self._last_zone_observation: Optional[datetime] = None
        
        # Track if we've warned about zone end
        self._zone_end_warning_issued: bool = False
        self._validate_overlaps()
        
    def _parse_time(self, time_str: str, tz: pytz.timezone) -> dt_time:
        """Parse time string to time object."""
        return datetime.strptime(time_str, "%H:%M").time()
    
    def _get_current_time_in_zone(self, zone: HotZoneConfig) -> datetime:
        """Get current time in zone's timezone."""
        zone_tz = pytz.timezone(zone.timezone)
        now = datetime.now(zone_tz)
        return now

    def _validate_overlaps(self) -> None:
        """Log explicit overlap resolution for enabled zones."""
        enabled_zones = [zone for zone in self.hot_zones if zone.enabled]
        for idx, left in enumerate(enabled_zones):
            left_start = self._parse_time(left.start, left.timezone)
            left_end = self._parse_time(left.end, left.timezone)
            for right in enabled_zones[idx + 1 :]:
                if left.timezone != right.timezone:
                    continue
                right_start = self._parse_time(right.start, right.timezone)
                right_end = self._parse_time(right.end, right.timezone)
                if left_end <= left_start or right_end <= right_start:
                    continue
                overlap = left_start < right_end and right_start < left_end
                if overlap:
                    winner = left.name if left_start >= right_start else right.name
                    logger.warning(
                        "Overlapping hot zones detected: %s and %s. Scheduler will prioritize %s by latest start time.",
                        left.name,
                        right.name,
                        winner,
                    )
    
    def get_current_zone(self, current_time: Optional[datetime] = None) -> Optional[ZoneInfo]:
        """
        Get current zone information.
        
        Args:
            current_time: Optional time to check (defaults to now)
        
        Returns:
            ZoneInfo if inside a zone, None otherwise
        """
        if current_time is None:
            # Use config timezone (first zone's timezone)
            if not self.hot_zones:
                return None
            default_tz = pytz.timezone(self.hot_zones[0].timezone)
            current_time = datetime.now(default_tz)

        # Find all matching zones, then pick the one with latest start time (most specific)
        matching_zones = []
        for zone in self.hot_zones:
            if not zone.enabled:
                continue

            zone_tz = pytz.timezone(zone.timezone)
            start_time = self._parse_time(zone.start, zone.timezone)
            end_time = self._parse_time(zone.end, zone.timezone)

            if current_time.tzinfo is not None:
                zone_time = current_time.astimezone(zone_tz)
            else:
                zone_time = current_time
            zone_date = zone_time.date()
            zone_time_only = zone_time.time()

            if start_time <= end_time:
                start_dt = datetime.combine(zone_date, start_time)
                end_dt = datetime.combine(zone_date, end_time)
                if current_time.tzinfo is not None:
                    start_dt = zone_tz.localize(start_dt)
                    end_dt = zone_tz.localize(end_dt)
                in_zone = start_dt <= zone_time < end_dt
            else:
                if zone_time_only >= start_time:
                    start_date = zone_date
                    end_date = zone_date + timedelta(days=1)
                else:
                    start_date = zone_date - timedelta(days=1)
                    end_date = zone_date
                start_dt = datetime.combine(start_date, start_time)
                end_dt = datetime.combine(end_date, end_time)
                if current_time.tzinfo is not None:
                    start_dt = zone_tz.localize(start_dt)
                    end_dt = zone_tz.localize(end_dt)
                in_zone = start_dt <= zone_time < end_dt

            if in_zone:
                matching_zones.append((zone, start_dt, end_dt, zone_time))

        if not matching_zones:
            # Outside all zones
            self._previous_zone = None
            self._zone_entry_time = None
            self._bars_in_zone = 0
            self._last_zone_observation = None
            self._zone_end_warning_issued = False
            return None

        # Pick zone with latest start time (most specific for overlaps)
        zone, start_dt, end_dt, zone_time = max(matching_zones, key=lambda x: x[1])

        # Determine state
        if zone.mode == "flatten_only":
            state = ZoneState.FLATTEN_ONLY
        else:
            state = ZoneState.ACTIVE

        minutes_remaining = (end_dt - zone_time).total_seconds() / 60

        # Track zone transitions
        if self._previous_zone != zone.name:
            self._previous_zone = zone.name
            self._zone_entry_time = zone_time
            self._bars_in_zone = 1
            self._last_zone_observation = zone_time
            self._zone_end_warning_issued = False
        elif self._last_zone_observation != zone_time:
            self._bars_in_zone += 1
            self._last_zone_observation = zone_time

        # Determine if first/last bars
        is_first_bar = self._bars_in_zone <= 5
        is_last_bar = minutes_remaining <= 5

        # Warn about zone end
        if is_last_bar and not self._zone_end_warning_issued:
            logger.info(f"Zone {zone.name} ending in {minutes_remaining:.1f} minutes")
            self._zone_end_warning_issued = True

        return ZoneInfo(
            name=zone.name,
            state=state,
            start_time=start_dt,
            end_time=end_dt,
            minutes_remaining=minutes_remaining,
            is_first_bar=is_first_bar,
            is_last_bar=is_last_bar
        )
    
    def is_trading_allowed(self) -> bool:
        """Check if new entries are allowed."""
        zone = self.get_current_zone()
        if zone is None:
            return False
        return zone.state == ZoneState.ACTIVE
    
    def is_flatten_only(self) -> bool:
        """Check if only exits are allowed (no new entries)."""
        zone = self.get_current_zone()
        if zone is None:
            return True  # Outside zones = flatten
        return zone.state == ZoneState.FLATTEN_ONLY
    
    def should_flatten(self) -> bool:
        """Check if positions should be flattened."""
        zone = self.get_current_zone()
        if zone is None:
            return True  # Outside zones = flatten
        return zone.is_last_bar
    
    def get_strategy_for_zone(self, zone_name: str) -> str:
        """Get strategy name configured for zone."""
        return self.config.strategy.zone_strategies.get(zone_name, "FLATTEN_ONLY")
    
    def get_zone_stats(self) -> dict:
        """Get statistics about current zone."""
        zone = self.get_current_zone()
        if zone is None:
            return {
                "active": False,
                "state": "inactive",
                "zone_name": None,
                "minutes_remaining": 0,
                "bars_in_zone": 0
            }
        
        time_in_zone = 0
        if self._zone_entry_time:
            default_tz = pytz.timezone(self.hot_zones[0].timezone)
            now = datetime.now(default_tz)
            time_in_zone = (now - self._zone_entry_time).total_seconds() / 60
        
        return {
            "active": True,
            "state": zone.state.value,
            "zone_name": zone.name,
            "minutes_remaining": zone.minutes_remaining,
            "bars_in_zone": self._bars_in_zone,
            "time_in_zone_minutes": time_in_zone,
            "is_first_bar": zone.is_first_bar,
            "is_last_bar": zone.is_last_bar
        }


# Global scheduler instance
_scheduler: Optional[HotZoneScheduler] = None


def get_scheduler(force_recreate: bool = False) -> HotZoneScheduler:
    """Get global scheduler instance."""
    global _scheduler
    if force_recreate:
        _scheduler = None
    if _scheduler is None:
        _scheduler = HotZoneScheduler()
    return _scheduler
