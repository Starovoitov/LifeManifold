"""Generate local surrogate buffer + checkpoints for dashboard and MAP-Elites.

Writes under ``artifacts/surrogate/`` (gitignored). Uses deterministic synthetic
buffer rows so no long illuminator run is required for local dev.

Examples::

    uv run python scripts/bootstrap_surrogate_artifacts.py
    uv run python scripts/bootstrap_surrogate_artifacts.py --quick
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.evaluation import MIN_TRAIN_SAMPLES_FULL, MIN_TRAIN_SAMPLES_MICRO
from worldspace.surrogate.synthetic_buffer import write_synthetic_buffer

_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_surrogate.py"
_DEFAULT_SURROGATE_ROOT = _REPO_ROOT / "artifacts" / "surrogate"
_DEFAULT_BUFFER = _DEFAULT_SURROGATE_ROOT / "buffer.jsonl"
_DEFAULT_CHECKPOINT_DIR = _DEFAULT_SURROGATE_ROOT / "checkpoints"
_MICRO_BUFFER_SAMPLES = 160
_FULL_BUFFER_SAMPLES = 2400


@dataclass(frozen=True)
class BootstrapPaths:
    """Output paths for one bootstrap profile."""

    buffer_path: Path
    micro_checkpoint: Path
    micro_summary: Path
    latest_checkpoint: Path
    latest_summary: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate surrogate buffer.jsonl and .pkl checkpoints locally.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only micro.pkl + small buffer (skip latest.pkl / full train)",
    )
    parser.add_argument(
        "--surrogate-root",
        type=Path,
        default=_DEFAULT_SURROGATE_ROOT,
        help="Root directory for buffer and checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic buffer rows",
    )
    parser.add_argument(
        "--model-type",
        choices=("lightgbm", "mlp"),
        default="lightgbm",
        help="Surrogate backend for training",
    )
    return parser.parse_args()


def resolve_paths(surrogate_root: Path) -> BootstrapPaths:
    checkpoint_dir = surrogate_root / "checkpoints"
    return BootstrapPaths(
        buffer_path=surrogate_root / "buffer.jsonl",
        micro_checkpoint=checkpoint_dir / "micro.pkl",
        micro_summary=checkpoint_dir / "micro.summary.json",
        latest_checkpoint=checkpoint_dir / "latest.pkl",
        latest_summary=checkpoint_dir / "latest.summary.json",
    )


def run_train(
    *,
    buffer_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
    model_type: str,
    micro: bool,
) -> None:
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--model-type",
        model_type,
        "--buffer-path",
        str(buffer_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--summary-path",
        str(summary_path),
    ]
    if micro:
        cmd.append("--micro")
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)


def bootstrap(*, paths: BootstrapPaths, quick: bool, seed: int, model_type: str) -> None:
    paths.buffer_path.parent.mkdir(parents=True, exist_ok=True)
    paths.micro_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if quick:
        write_synthetic_buffer(
            paths.buffer_path,
            n_samples=_MICRO_BUFFER_SAMPLES,
            seed=seed,
        )
        print(
            f"Wrote {paths.buffer_path} ({_MICRO_BUFFER_SAMPLES} synthetic rows, seed={seed})"
        )
        run_train(
            buffer_path=paths.buffer_path,
            checkpoint_path=paths.micro_checkpoint,
            summary_path=paths.micro_summary,
            model_type=model_type,
            micro=True,
        )
        print(f"Wrote {paths.micro_checkpoint}")
        return

    write_synthetic_buffer(
        paths.buffer_path,
        n_samples=_FULL_BUFFER_SAMPLES,
        seed=seed,
    )
    print(
        f"Wrote {paths.buffer_path} ({_FULL_BUFFER_SAMPLES} synthetic rows, seed={seed})"
    )

    micro_buffer = paths.buffer_path.with_name("buffer_micro.jsonl")
    write_synthetic_buffer(micro_buffer, n_samples=_MICRO_BUFFER_SAMPLES, seed=seed + 1)
    run_train(
        buffer_path=micro_buffer,
        checkpoint_path=paths.micro_checkpoint,
        summary_path=paths.micro_summary,
        model_type=model_type,
        micro=True,
    )
    print(f"Wrote {paths.micro_checkpoint}")

    run_train(
        buffer_path=paths.buffer_path,
        checkpoint_path=paths.latest_checkpoint,
        summary_path=paths.latest_summary,
        model_type=model_type,
        micro=False,
    )
    print(f"Wrote {paths.latest_checkpoint}")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.surrogate_root.expanduser())
    if not args.quick and _FULL_BUFFER_SAMPLES < MIN_TRAIN_SAMPLES_FULL:
        raise SystemExit(f"FULL_BUFFER_SAMPLES must be >= {MIN_TRAIN_SAMPLES_FULL}")
    if args.quick and _MICRO_BUFFER_SAMPLES < MIN_TRAIN_SAMPLES_MICRO:
        raise SystemExit(f"MICRO_BUFFER_SAMPLES must be >= {MIN_TRAIN_SAMPLES_MICRO}")
    bootstrap(
        paths=paths,
        quick=args.quick,
        seed=args.seed,
        model_type=args.model_type,
    )
    print()
    print("Surrogate artifacts ready:")
    print(f"  buffer:     {paths.buffer_path.resolve()}")
    if paths.micro_checkpoint.is_file():
        print(f"  micro:      {paths.micro_checkpoint.resolve()}")
    if paths.latest_checkpoint.is_file():
        print(f"  latest:     {paths.latest_checkpoint.resolve()}")
    print()
    print("Enable in scheduler YAML: surrogate.enabled: true")
    print("Dashboard paths: dashboard/config/config.yaml (paths.surrogate_*)")


if __name__ == "__main__":
    main()
