# ES Hot-Zone Trader

CLI-based day trading system for ES hot zones. Execution and Topstep API run locally; telemetry is pushed to Railway for analytics and tooling.

## Operator interface

The only operator interface is the **CLI** (no TUI). From the project root:

- `es-trade` — show help and commands
- `es-trade start` — start the trading engine (live)
- `es-trade stop` / `es-trade restart` — lifecycle
- `es-trade status` — one-screen status (running, zone, position, PnL, risk)
- `es-trade debug` / `es-trade events` / `es-trade config` / `es-trade balance` / `es-trade health` / `es-trade replay <path>`

See the repo **docs/** for full operator and architecture docs:

- [docs/OPERATOR.md](../docs/OPERATOR.md) — commands, MCP setup, compliance
- [docs/Architecture-Overview.md](../docs/Architecture-Overview.md) — what runs where, data flow, Railway services
- [docs/Compliance-Boundaries.md](../docs/Compliance-Boundaries.md) — Topstep/CME boundaries and pre-migration gate

## Layout

- **`src/cli/`** — CLI commands and entrypoint
- **`src/engine/`** — Trading engine, strategy, reconciliation
- **`src/execution/`** — Order executor
- **`src/market/`** — Topstep client (market data, orders, positions)
- **`src/observability/`** — SQLite store (events, runs, trades)
- **`src/bridge/`** — Data bridge and outbox (Mac → Railway ingest)
- **`src/server/`** — Debug server (health + debug HTTP; MCP runs on Railway)
- **`config/default.yaml`** — Default configuration

## Railway

Cloud services live in the repo under **railway/** (ingest, analytics, mcp, web). They are analytics and tooling only; no execution. Deploy to the G-Trade Railway project and set `observability.railway_ingest_url` and API key so the local bridge can send data.
