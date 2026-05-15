"""CLI: embedding scatter from metrics-trace JSONL; CA-step trace figures + summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m worldspace.visualizer",
        description=(
            "World-space visualizations: embedding scatter from ``--metrics-trace`` JSONL; "
            "or CA step trace plots from --ca-step-trace output."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_emb = sub.add_parser(
        "embedding",
        help="2D scatter from world-space JSONL (e.g. ``python -m worldspace --metrics-trace``).",
    )
    p_emb.add_argument(
        "jsonl",
        type=str,
        help="Path to metrics JSONL (one object per line).",
    )
    p_emb.add_argument(
        "--plot",
        type=str,
        required=True,
        help="Output PNG path for the embedding scatter.",
    )
    p_emb.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title (optional).",
    )

    p_ca = sub.add_parser(
        "ca-trace",
        help="Plots from --ca-step-trace JSONL (time-series + PCA and UMAP trajectories).",
    )
    p_ca.add_argument(
        "trace_jsonl",
        type=str,
        help="Path to CA step trace JSONL (from --ca-step-trace).",
    )
    p_ca.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for PNG outputs (created if missing).",
    )
    p_ca.add_argument(
        "--worlds",
        type=str,
        default="",
        help=(
            "Comma-separated yield_index values to plot; "
            "default: up to 8 distinct indices from the file."
        ),
    )
    p_ca.add_argument(
        "--metrics",
        type=str,
        default="interestingness,entropy,density_mean,oscillation_score",
        help="Comma-separated metric names for the time-series figure.",
    )
    p_ca.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Skip the metrics-vs-ca_step line plot.",
    )
    p_ca.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip the PCA trajectory plot.",
    )
    p_ca.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip the UMAP trajectory plot.",
    )
    p_ca.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print per-world pandas aggregate table (mean/std/min/max per metric) "
            "to stdout."
        ),
    )

    args = parser.parse_args(argv)

    if args.command == "embedding":
        from .plotting import plot_world_embedding_from_jsonl

        title = args.title.strip() or None
        plot_world_embedding_from_jsonl(args.jsonl, args.plot, title=title)
        return

    if args.command == "ca-trace":
        _run_ca_trace(args)
        return


def _run_ca_trace(args: argparse.Namespace) -> None:
    from .plotting import (
        load_ca_step_trace_jsonl,
        plot_ca_step_metrics_timeseries,
        plot_ca_step_pca_trajectories,
        plot_ca_step_umap_trajectories,
        summarize_ca_step_trace_by_world,
    )

    src = Path(args.trace_jsonl)
    if not src.is_file():
        print(f"Not a file: {src.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ca_step_trace_jsonl(src)
    if df.empty:
        print("Trace file is empty; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    worlds_arg = args.worlds.strip()
    if worlds_arg:
        yield_indices = [int(x.strip()) for x in worlds_arg.split(",") if x.strip()]
    else:
        uniq = sorted(df["yield_index"].unique().tolist())
        yield_indices = uniq[:8]

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.summary:
        summary = summarize_ca_step_trace_by_world(df)
        print(summary.to_string())

    if not args.no_timeseries:
        plot_ca_step_metrics_timeseries(
            df,
            yield_indices,
            out_dir / "ca_timeseries.png",
            metric_names=metrics,
            title="CA metrics vs step (selected worlds)",
        )
    if not args.no_pca:
        plot_ca_step_pca_trajectories(
            df,
            yield_indices,
            out_dir / "ca_pca_trajectories.png",
            title="PCA on metric snapshots (trajectories over ca_step)",
        )
    if not args.no_umap:
        plot_ca_step_umap_trajectories(
            df,
            yield_indices,
            out_dir / "ca_umap_trajectories.png",
            title="UMAP on metric snapshots (trajectories over ca_step)",
        )
