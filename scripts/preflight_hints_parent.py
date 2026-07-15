#!/usr/bin/env python3
"""Pre-flight check for Path 4D parent-hints pilot: parent vs surrogate fitness gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.archive import load_and_collapse_jsonl
from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.illuminators.scheduler import load_scheduler
from worldspace.surrogate.checkpoint_io import load_surrogate_checkpoint
from worldspace.surrogate.surrogate import SurrogateFacade, build_surrogate_facade

DEFAULT_ARCHIVE = (
    ROOT / "artifacts/experiments/q1-full/hints/seed_0/map_elites_archive.jsonl"
)
DEFAULT_CHECKPOINT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
DEFAULT_SCHEDULER = (
    ROOT
    / "worldspace/specs/map_elites_scheduler_nightly_llm_hints_parent.yaml"
)
DEFAULT_GO_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument(
        "--go-threshold",
        type=float,
        default=DEFAULT_GO_THRESHOLD,
        help="Minimum median |parent_fitness - surrogate_fitness| for GO",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on archive rows (0 = all collapsed elites)",
    )
    return parser.parse_args()


def _build_facade(checkpoint: Path, scheduler_path: Path) -> SurrogateFacade:
    config = load_scheduler(scheduler_path)
    model = load_surrogate_checkpoint(checkpoint)
    return build_surrogate_facade(
        model,
        uncertainty_fallback=config.surrogate_stub_uncertainty,
        calibration_path=config.surrogate_calibration,
        use_soft_extinction=config.surrogate_use_soft_extinction,
        extinction_gate_threshold=config.surrogate_extinction_gate_threshold,
    )


def _collect_deltas(
    archive_path: Path,
    facade: SurrogateFacade,
    *,
    max_rows: int,
) -> list[float]:
    archive = load_and_collapse_jsonl(archive_path)
    deltas: list[float] = []
    for cell_id in range(archive.n_cells):
        if max_rows > 0 and len(deltas) >= max_rows:
            break
        elite = archive.get_cell(cell_id)
        if elite is None or elite.world_spec is None:
            continue
        spec = elite.world_spec
        apply_canonical_seed(spec)
        prediction = facade.predict(spec)
        deltas.append(abs(float(elite.fitness) - float(prediction.fitness)))
    return deltas


def main() -> int:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    scheduler = args.scheduler.expanduser().resolve()
    if not archive.is_file():
        print(f"Archive not found: {archive}", file=sys.stderr)
        return 2
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    facade = _build_facade(checkpoint, scheduler)
    deltas = _collect_deltas(archive, facade, max_rows=args.max_rows)
    if not deltas:
        print("No archive elites with world_spec found.")
        return 2
    array = np.asarray(deltas, dtype=float)
    median = float(np.median(array))
    p95 = float(np.quantile(array, 0.95))
    payload = {
        "archive": str(archive),
        "checkpoint": str(checkpoint),
        "row_count": int(array.size),
        "median_abs_fitness_delta": median,
        "p95_abs_fitness_delta": p95,
        "go_threshold": float(args.go_threshold),
        "go": median >= float(args.go_threshold),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["go"]:
        print("VERDICT: GO — parent vs surrogate fitness differs enough for pilot run.")
        return 0
    print("VERDICT: NO-GO — parent vs surrogate fitness too similar; pilot run optional.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
