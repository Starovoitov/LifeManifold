#!/usr/bin/env python3
"""Pre-flight check for Path 5E direction-hints pilot: surrogate gradient strength."""

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
from worldspace.surrogate.direction_hints import compute_composed_fitness_gradient
from worldspace.surrogate.feature_extractor import extract

DEFAULT_ARCHIVE = (
    ROOT / "artifacts/experiments/q1-full/hints/seed_0/map_elites_archive.jsonl"
)
DEFAULT_CHECKPOINT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
DEFAULT_SCHEDULER = (
    ROOT / "worldspace/specs/map_elites_scheduler_nightly_llm_hints_direction.yaml"
)
DEFAULT_GO_THRESHOLD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument(
        "--go-threshold",
        type=float,
        default=DEFAULT_GO_THRESHOLD,
        help="Minimum median max |∂fit/∂x| over actionable dims for GO",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Cap on archive rows for quick preflight (0 = all collapsed elites)",
    )
    return parser.parse_args()


def _collect_gradient_strengths(
    archive_path: Path,
    model,
    *,
    use_soft_extinction: bool,
    extinction_gate_threshold: float,
    max_rows: int,
) -> list[float]:
    archive = load_and_collapse_jsonl(archive_path)
    strengths: list[float] = []
    for cell_id in range(archive.n_cells):
        if max_rows > 0 and len(strengths) >= max_rows:
            break
        elite = archive.get_cell(cell_id)
        if elite is None or elite.world_spec is None:
            continue
        spec = elite.world_spec
        apply_canonical_seed(spec)
        features = extract(spec)
        gradient, _ = compute_composed_fitness_gradient(
            model,
            features,
            use_soft_extinction=use_soft_extinction,
            extinction_gate_threshold=extinction_gate_threshold,
        )
        strengths.append(float(np.max(np.abs(gradient))))
    return strengths


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
    config = load_scheduler(scheduler)
    model = load_surrogate_checkpoint(checkpoint)
    strengths = _collect_gradient_strengths(
        archive,
        model,
        use_soft_extinction=config.surrogate_use_soft_extinction,
        extinction_gate_threshold=config.surrogate_extinction_gate_threshold,
        max_rows=args.max_rows,
    )
    if not strengths:
        print("No archive elites with world_spec found.")
        return 2
    array = np.asarray(strengths, dtype=float)
    median = float(np.median(array))
    p95 = float(np.quantile(array, 0.95))
    payload = {
        "archive": str(archive),
        "checkpoint": str(checkpoint),
        "row_count": int(array.size),
        "median_max_abs_gradient": median,
        "p95_max_abs_gradient": p95,
        "go_threshold": float(args.go_threshold),
        "go": median >= float(args.go_threshold),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["go"]:
        print("VERDICT: GO — surrogate gradients strong enough for direction pilot.")
        return 0
    print("VERDICT: NO-GO — gradients too flat; pilot run optional.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
