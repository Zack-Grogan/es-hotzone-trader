"""Local event calendar provider and emergency halt interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import yaml

from src.config import BlackoutConfig, EventProviderConfig


@dataclass
class EventContext:
    """Active event/blackout state."""

    blackout_active: bool = False
    reason: str = ""
    active_tags: list[str] = field(default_factory=list)
    minutes_to_next_event: float | None = None
    post_event_cooling: bool = False


class LocalEventProvider:
    """File-backed economic calendar with manual emergency halt support."""

    def __init__(self, config: EventProviderConfig, fallback: BlackoutConfig, root: Path):
        self.config = config
        self.fallback = fallback
        self.root = root
        self._calendar_cache: list[dict[str, Any]] = []
        self._last_refresh: datetime | None = None
        self._calendar_mtime: float | None = None
        self._tz = pytz.timezone(self.config.default_timezone)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.root / path

    def _load_calendar(self) -> list[dict[str, Any]]:
        calendar_path = self._resolve_path(self.config.calendar_path)
        if not calendar_path.exists():
            return []
        stat = calendar_path.stat()
        if self._calendar_mtime == stat.st_mtime and self._calendar_cache:
            return self._calendar_cache
        with calendar_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or []
        self._calendar_cache = payload if isinstance(payload, list) else payload.get("events", [])
        self._calendar_mtime = stat.st_mtime
        return self._calendar_cache

    def _fallback_events(self, current_time: datetime) -> list[dict[str, Any]]:
        current_et = current_time.astimezone(self._tz)
        event_date = current_et.date()
        fallback_events = []
        for time_str in self.fallback.news_times:
            event_time = datetime.strptime(time_str, "%H:%M").time()
            event_dt = self._tz.localize(datetime.combine(event_date, event_time))
            fallback_events.append({"name": f"static_{time_str}", "timestamp": event_dt.isoformat(), "impact": "medium"})
        return fallback_events

    def _iter_events(self, current_time: datetime) -> list[dict[str, Any]]:
        now = datetime.now(tz=current_time.tzinfo or pytz.UTC)
        if (
            self._last_refresh is None
            or (now - self._last_refresh).total_seconds() >= max(int(self.config.refresh_seconds), 1)
        ):
            self._last_refresh = now
            loaded = self._load_calendar()
            if loaded:
                return loaded
        return self._calendar_cache or self._fallback_events(current_time)

    def get_context(self, current_time: datetime) -> EventContext:
        """Return the active event context for the supplied time."""
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        current_local = current_time.astimezone(self._tz)

        emergency_path = self._resolve_path(self.config.emergency_halt_path)
        if emergency_path.exists():
            max_age_minutes = max(int(getattr(self.config, "emergency_halt_max_age_minutes", 1440)), 1)
            halt_age_seconds = current_time.timestamp() - emergency_path.stat().st_mtime
            if halt_age_seconds <= max_age_minutes * 60:
                return EventContext(True, "manual_emergency_halt", ["manual_halt"], 0.0, True)

        blackout_active = False
        reason = ""
        active_tags: list[str] = []
        minutes_to_next: float | None = None
        post_event_cooling = False

        for raw_event in self._iter_events(current_time):
            raw_ts = raw_event.get("timestamp")
            if not raw_ts:
                continue
            event_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if event_ts.tzinfo is None:
                event_ts = self._tz.localize(event_ts)
            else:
                event_ts = event_ts.astimezone(self._tz)

            impact = str(raw_event.get("impact", "medium")).lower()
            impact_window = self.config.impact_windows.get(impact, self.config.impact_windows.get("medium", {"pre": 5, "post": 10}))
            pre_minutes = int(raw_event.get("pre_minutes", impact_window.get("pre", self.fallback.pre_minutes)))
            post_minutes = int(raw_event.get("post_minutes", impact_window.get("post", self.fallback.post_minutes)))
            window_start = event_ts - timedelta(minutes=pre_minutes)
            window_end = event_ts + timedelta(minutes=post_minutes)
            minutes_until = (event_ts - current_local).total_seconds() / 60.0

            if minutes_until >= 0 and (minutes_to_next is None or minutes_until < minutes_to_next):
                minutes_to_next = minutes_until

            if window_start <= current_local <= window_end:
                blackout_active = True
                reason = f"event_{raw_event.get('name', 'calendar')}"
                active_tags.append(str(raw_event.get("name", "event")))
                if current_local > event_ts:
                    post_event_cooling = True

        return EventContext(
            blackout_active=blackout_active,
            reason=reason,
            active_tags=active_tags,
            minutes_to_next_event=minutes_to_next,
            post_event_cooling=post_event_cooling,
        )
