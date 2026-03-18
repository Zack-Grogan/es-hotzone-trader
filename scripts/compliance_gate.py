#!/usr/bin/env python3
"""Print compliance gate checklist before TUI Sunset / Railway migration. See docs/Compliance-Boundaries.md."""
from __future__ import annotations

import sys


def main() -> int:
    print("Compliance gate (TUI Sunset / Railway migration)")
    print("See docs/Compliance-Boundaries.md for full boundaries.")
    print("")
    print("Before Phase 1+, confirm:")
    print("  [ ] Trader is stopped (es-trade stop)")
    print("  [ ] Topstep: execution stays on Mac; no execution from Railway")
    print("  [ ] CME/data: cloud data use is personal, private, no redistribution")
    print("  [ ] State reset runbook reviewed; no stuck/unresolved state")
    print("")
    print("Set COMPLIANCE_GATE_ACK=1 to pass this script (e.g. for CI).")
    ack = __import__("os").environ.get("COMPLIANCE_GATE_ACK", "")
    if ack == "1":
        print("COMPLIANCE_GATE_ACK=1 set — gate passed.")
        return 0
    print("Complete the checklist and re-run, or set COMPLIANCE_GATE_ACK=1 to acknowledge.")
    return 0  # informational only; does not block


if __name__ == "__main__":
    sys.exit(main())
