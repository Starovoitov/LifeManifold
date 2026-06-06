"""Record v2 feature-extractor eval memo from a training summary JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldspace.surrogate.feature_extractor import FEATURE_SCHEMA_VERSION
from worldspace.surrogate.genome_features import FEATURE_DIM

_DEFAULT_OUTPUT = (
    _REPO_ROOT / "artifacts" / "surrogate" / "feature_extractor_v2_eval.json"
)


def load_training_summary(path: Path) -> dict[str, object]:
    """Load a train summary JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Summary must be a JSON object: {path}"
        raise ValueError(msg)
    return payload


def resolve_checkpoint_from_summary(summary_path: Path, override: Path | None) -> Path:
    """Derive checkpoint path from summary filename when override is absent."""
    if override is not None:
        return override
    stem = summary_path.name.removesuffix(".summary.json")
    return summary_path.with_name(f"{stem}.pkl")


def build_eval_payload(
    summary: dict[str, object],
    *,
    checkpoint_path: Path,
    summary_path: Path,
    buffer_path: Path | None,
    archive_path: Path | None,
    dataset_source: str,
    notes: str,
) -> dict[str, object]:
    """Merge training summary fields into the E6.3 eval memo schema."""
    holdout = summary.get("holdout_metrics")
    holdout_metrics = holdout if isinstance(holdout, dict) else {}
    quality_passed = summary.get("quality_passed") is True
    llm_hints = "enabled" if quality_passed else "stub"
    acquisition_filter = "filter" if quality_passed else "off"
    return {
        "schema_version": "1.0",
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "feature_schema_version": summary.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        ),
        "feature_dim": summary.get("feature_dim", FEATURE_DIM),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "dataset": {
            "source": dataset_source,
            "archive_path": str(archive_path.resolve()) if archive_path else None,
            "buffer_path": str(buffer_path.resolve()) if buffer_path else None,
            "sample_count": summary.get("sample_count"),
            "train_count": summary.get("train_count"),
            "holdout_count": summary.get("holdout_count"),
        },
        "holdout_metrics": {
            "r2_fitness": holdout_metrics.get("r2_fitness"),
            "mae_fitness": holdout_metrics.get("mae_fitness"),
            "mae_stability": holdout_metrics.get("mae_stability"),
        },
        "quality_passed": quality_passed,
        "decisions": {
            "llm_hints": llm_hints,
            "acquisition_filter": acquisition_filter,
            "production_checkpoint": quality_passed,
        },
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write artifacts/surrogate/feature_extractor_v2_eval.json from a train summary.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Training summary JSON (e.g. nightly_v2.summary.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path (default: summary stem with .pkl).",
    )
    parser.add_argument("--buffer-path", type=str, default="")
    parser.add_argument("--archive-path", type=str, default="")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="backfill",
        help="Label for dataset.source (backfill, nightly_append, synthetic).",
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help="Output eval memo path.",
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).expanduser()
    summary = load_training_summary(summary_path)
    checkpoint_override = (
        Path(args.checkpoint).expanduser() if args.checkpoint.strip() else None
    )
    checkpoint = resolve_checkpoint_from_summary(summary_path, checkpoint_override)
    buffer_path = (
        Path(args.buffer_path).expanduser() if args.buffer_path.strip() else None
    )
    archive_path = (
        Path(args.archive_path).expanduser() if args.archive_path.strip() else None
    )
    output = Path(args.output).expanduser()
    payload = build_eval_payload(
        summary,
        checkpoint_path=checkpoint,
        summary_path=summary_path,
        buffer_path=buffer_path,
        archive_path=archive_path,
        dataset_source=args.dataset_source.strip() or "backfill",
        notes=args.notes.strip(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "quality_passed": payload["quality_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
