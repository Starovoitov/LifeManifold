"""Record surrogate improvement eval memo (schema v3) from training summaries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_OUTPUT = (
    _REPO_ROOT / "artifacts" / "surrogate" / "surrogate_improvement_eval.json"
)
_TEMPLATE_PATH = (
    _REPO_ROOT / "artifacts" / "surrogate" / "surrogate_improvement_eval.template.json"
)


def load_training_summary(path: Path) -> dict[str, Any]:
    """Load one training summary JSON object."""
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


def build_run_entry(
    summary: dict[str, Any],
    *,
    checkpoint_path: Path,
    summary_path: Path,
    buffer_path: Path | None,
    archive_path: Path | None,
    dataset_source: str,
    notes: str,
) -> dict[str, Any]:
    """Map one training summary into a v3 eval run block."""
    holdout = summary.get("holdout_metrics")
    holdout_metrics = holdout if isinstance(holdout, dict) else {}
    per_target = summary.get("per_target_holdout")
    return {
        "model_type": summary.get("model_type", "lightgbm"),
        "feature_schema_version": summary.get("feature_schema_version"),
        "feature_dim": summary.get("feature_dim"),
        "emitter_onehot": bool(summary.get("emitter_onehot")),
        "stratify_emitter": bool(summary.get("stratify_emitter")),
        "low_stability_weight": summary.get("low_stability_weight", 1.0),
        "consistency_weight": summary.get("consistency_weight", 0.0),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "dataset": {
            "source": dataset_source,
            "buffer_path": str(buffer_path.resolve()) if buffer_path else None,
            "archive_path": str(archive_path.resolve()) if archive_path else None,
            "sample_count": summary.get("sample_count"),
            "train_count": summary.get("train_count"),
            "holdout_count": summary.get("holdout_count"),
        },
        "holdout_metrics": {
            "r2_fitness": holdout_metrics.get("r2_fitness"),
            "r2_fitness_direct": holdout_metrics.get("r2_fitness_direct"),
            "mae_fitness": holdout_metrics.get("mae_fitness"),
            "mae_fitness_direct": holdout_metrics.get("mae_fitness_direct"),
            "mae_stability": holdout_metrics.get("mae_stability"),
        },
        "per_target_holdout": per_target if isinstance(per_target, list) else [],
        "hints_ok": summary.get("hints_ok"),
        "quality_passed": summary.get("quality_passed"),
        "notes": notes,
    }


def load_or_create_payload(output_path: Path) -> dict[str, Any]:
    """Load an existing eval memo or start from the v3 template."""
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"Eval memo must be a JSON object: {output_path}"
            raise ValueError(msg)
        return payload
    template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        msg = f"Template must be a JSON object: {_TEMPLATE_PATH}"
        raise ValueError(msg)
    return template


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one surrogate train run to the improvement eval memo JSON.",
    )
    parser.add_argument(
        "--summary", type=str, required=True, help="Training summary JSON"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Optional checkpoint path"
    )
    parser.add_argument("--buffer-path", type=str, default="")
    parser.add_argument("--archive-path", type=str, default="")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="backfill",
        help="Label for dataset.source",
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help="Eval memo output path",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace output file instead of appending a run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    payload = (
        load_or_create_payload(output)
        if not args.replace
        else {
            "schema_version": "3.0",
            "recorded_at": None,
            "notes": "",
            "runs": [],
            "comparisons": {
                "lgbm_vs_mlp": None,
                "emitter_onehot_ab": None,
                "hard_vs_soft_compose": None,
            },
        }
    )
    payload["schema_version"] = "3.0"
    payload["recorded_at"] = datetime.now(tz=UTC).isoformat()
    if args.notes.strip():
        payload["notes"] = args.notes.strip()
    runs = payload.get("runs")
    if not isinstance(runs, list):
        runs = []
    runs.append(
        build_run_entry(
            summary,
            checkpoint_path=checkpoint,
            summary_path=summary_path,
            buffer_path=buffer_path,
            archive_path=archive_path,
            dataset_source=args.dataset_source.strip() or "backfill",
            notes=args.notes.strip(),
        )
    )
    payload["runs"] = runs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "run_count": len(runs),
                "hints_ok": summary.get("hints_ok"),
                "quality_passed": summary.get("quality_passed"),
            }
        )
    )


if __name__ == "__main__":
    main()
