from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class Segment:
    minutes: int
    drift_per_min: float
    vol: float


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_rows(
    *,
    start_utc: datetime,
    minutes: int,
    symbol: str,
    start_price: float,
    seed_volume: int,
    segments: list[Segment],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    price = float(start_price)
    cumulative_volume = int(seed_volume)
    t = start_utc

    # Create non-trivial behavior across multiple hot-zones: drift + oscillation + occasional shocks.
    seg_idx = 0
    seg_remaining = segments[0].minutes if segments else minutes

    for i in range(minutes):
        if seg_remaining <= 0 and seg_idx + 1 < len(segments):
            seg_idx += 1
            seg_remaining = segments[seg_idx].minutes
        seg_remaining -= 1

        seg = segments[seg_idx] if segments else Segment(minutes=minutes, drift_per_min=0.0, vol=1.0)

        base_osc = math.sin(i / 7.0) * 0.6 + math.sin(i / 29.0) * 1.2
        shock = 0.0
        if i in {35, 78, 146, 210}:
            shock = (1 if (i % 2 == 0) else -1) * (3.5 + (i % 5) * 0.25)

        price += seg.drift_per_min + base_osc * (seg.vol / 10.0) + shock * (seg.vol / 10.0)

        # Keep within a plausible ES range and on a quarter-tick grid.
        price = max(3000.0, min(8000.0, price))
        last = round(price * 4) / 4

        spread = 0.25
        bid = last - spread
        ask = last + spread

        # Volume: cumulative by default (ReplayRunner treats "volume" as cumulative when present).
        step = int(40 + abs(base_osc) * 25 + (5 if shock else 0))
        cumulative_volume += max(step, 1)

        rows.append(
            {
                "timestamp": _iso_z(t),
                "symbol": symbol,
                "last": f"{last:.2f}",
                "bid": f"{bid:.2f}",
                "ask": f"{ask:.2f}",
                "volume": str(cumulative_volume),
                "latency_ms": "15",
            }
        )
        t += timedelta(minutes=1)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic replay CSV for es-hotzone-trader.")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--symbol", default="ES", help="Symbol (default: ES)")
    parser.add_argument("--start", default="2026-03-16T11:20:00Z", help="UTC start timestamp (ISO-8601)")
    parser.add_argument("--minutes", type=int, default=340, help="Number of minutes to generate")
    parser.add_argument("--start-price", type=float, default=5200.0, help="Starting price")
    args = parser.parse_args()

    start_str = str(args.start).replace("Z", "+00:00")
    start_utc = datetime.fromisoformat(start_str).astimezone(timezone.utc)

    # Designed to span the configured hot-zones in America/Chicago:
    # 06:30-08:30 (Pre-Open), 09:00-11:00 (Post-Open), 12:00-13:00 (Midday/Close-Scalp).
    segments = [
        Segment(minutes=90, drift_per_min=0.06, vol=0.9),   # pre-open style drift
        Segment(minutes=150, drift_per_min=0.10, vol=1.4),  # post-open trend + higher vol
        Segment(minutes=100, drift_per_min=-0.03, vol=0.8), # midday chop / mean reversion
    ]

    rows = generate_rows(
        start_utc=start_utc,
        minutes=int(args.minutes),
        symbol=str(args.symbol),
        start_price=float(args.start_price),
        seed_volume=10_000,
        segments=segments,
    )

    out_path = args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Time window: {rows[0]['timestamp']} -> {rows[-1]['timestamp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

