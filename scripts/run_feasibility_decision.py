#!/usr/bin/env python3
"""GO/REVISE/DROP from NAS/PCG smoke and isolated reports. No LLM. Thresholds unchanged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.feasibility.decision import build_decision

_ART = ROOT / "artifacts/controlled_attribution"
DEFAULT_NAS_SMOKE = _ART / "nas201/nas201_lookup_smoke.json"
DEFAULT_NAS_ISOLATED = _ART / "nas201/nas201_isolated.json"
DEFAULT_PCG_SMOKE = _ART / "pcg/pcg_smoke.json"
DEFAULT_PCG_ISOLATED = _ART / "pcg/pcg_isolated.json"
DEFAULT_OUT = _ART / "feasibility_decision.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nas-smoke", type=Path, default=DEFAULT_NAS_SMOKE)
    parser.add_argument("--nas-isolated", type=Path, default=DEFAULT_NAS_ISOLATED)
    parser.add_argument("--pcg-smoke", type=Path, default=DEFAULT_PCG_SMOKE)
    parser.add_argument("--pcg-isolated", type=Path, default=DEFAULT_PCG_ISOLATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    decision = build_decision(
        nas_smoke=_load(args.nas_smoke),
        nas_isolated=_load(args.nas_isolated),
        pcg_smoke=_load(args.pcg_smoke),
        pcg_isolated=_load(args.pcg_isolated),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "nas201": decision["nas201"]["decision"],
        "pcg_sokoban": decision["pcg_sokoban"]["decision"],
        "pcg_zelda": decision["pcg_zelda"]["decision"],
        "pcg_family": decision["pcg_family"]["decision"],
        "confirmatory_tasks_frozen": decision["shortlist"]["confirmatory_tasks_frozen"],
        "next_stage": decision["shortlist"]["next_stage"],
        "needs_replacement_family": decision["shortlist"]["needs_replacement_family"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
