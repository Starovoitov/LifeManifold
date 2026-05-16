"""CLI: render all requested world-space figures under ``--output-dir``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m worldspace.visualizer",
        description=(
            "Render visualizations into ``--output-dir`` using fixed filenames. "
            "From ``--metrics-jsonl``: ``dominant_metric_delta.png``, ``pca.png``, "
            "``umap.png`` plus ``*_norm.png`` (z-scored layout; same cluster colors "
            "as raw). UMAP needs ≥3 worlds. "
            "From ``--ca-step-jsonl``: ``ca_step_timeseries.png``, "
            "``pca_trajectories.png`` / ``pca_trajectories_norm.png``, "
            "``umap_trajectories.png`` / ``umap_trajectories_norm.png``."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for PNG outputs (created if missing).",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=str,
        default="",
        help=(
            "Path to metrics JSONL (``--metrics-trace`` or any JSONL with per-line "
            "``metrics``). Writes raw and ``*_norm.png`` scatters (norm layout uses "
            "z-scored metrics; colors always ``cluster_id`` / k-means on raw 7D). "
            "``pca*.png``: ≥2 worlds; ``umap*.png``: ≥3."
        ),
    )
    parser.add_argument(
        "--k-clusters",
        type=int,
        default=4,
        help=(
            "Number of k-means clusters for metrics scatters when "
            "``cluster_id`` is absent from ``--metrics-jsonl`` (default matches pipeline)."
        ),
    )
    parser.add_argument(
        "--ca-step-jsonl",
        type=str,
        default="",
        help="Path to CA step trace JSONL from ``--ca-step-trace``.",
    )
    parser.add_argument(
        "--ca-trace-worlds",
        type=str,
        default="",
        help=(
            "Comma-separated ``yield_index`` values for CA-step plots; "
            "default: up to 8 distinct indices from the CA trace file."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mo_eoc_indicator,entropy,density_mean,oscillation_score",
        help="Comma-separated metric names for ``ca_step_timeseries.png`` (CA trace only).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print per-world pandas aggregate table (mean/std/min/max per metric) "
            "to stdout when a CA step trace is provided."
        ),
    )
    args = parser.parse_args(argv)

    metrics_path = args.metrics_jsonl.strip()
    ca_path = args.ca_step_jsonl.strip()
    if not metrics_path and not ca_path:
        parser.error("Provide at least one of --metrics-jsonl and/or --ca-step-jsonl.")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = False
    if metrics_path:
        ok |= _run_metrics_plots(
            metrics_path,
            out_dir,
            k_clusters=max(1, args.k_clusters),
            ca_also=bool(ca_path),
        )

    if ca_path:
        ok |= _run_ca_trace_plots(
            ca_path=ca_path,
            out_dir=out_dir,
            worlds_arg=args.ca_trace_worlds.strip(),
            metrics_csv=args.metrics,
            do_summary=args.summary,
        )
    if not ok:
        sys.exit(1)


def _run_metrics_plots(
    metrics_jsonl: str,
    out_dir: Path,
    *,
    k_clusters: int,
    ca_also: bool,
) -> bool:
    from .plotting import (
        plot_world_metrics_pca_scatter_from_jsonl,
        plot_world_metrics_umap_scatter_from_jsonl,
        plot_dominant_metric_delta_scatter_from_jsonl,
    )

    src = Path(metrics_jsonl)
    if not src.is_file():
        print(f"Not a file: {src.resolve()}", file=sys.stderr)
        return False

    # Drop prior metrics-trace figures so a failed step cannot leave a stale file
    # (e.g. old ``umap.png`` from when CA trajectories used that name).
    for name in (
        "dominant_metric_delta.png",
        "dominant_metric_delta_norm.png",
        "world_space.png",
        "pca.png",
        "pca_norm.png",
        "umap.png",
        "umap_norm.png",
        "embedding.png",
    ):
        stale = out_dir / name
        if stale.is_file():
            stale.unlink()

    ok = False
    metrics_stems = (
        ("dominant_metric_delta", plot_dominant_metric_delta_scatter_from_jsonl),
        ("pca", plot_world_metrics_pca_scatter_from_jsonl),
        ("umap", plot_world_metrics_umap_scatter_from_jsonl),
    )
    for stem, plot_fn in metrics_stems:
        for norm, suffix in ((False, ""), (True, "_norm")):
            out_name = f"{stem}{suffix}.png"
            try:
                plot_fn(
                    src,
                    out_dir / out_name,
                    title=None,
                    k_clusters=k_clusters,
                    standardize_metrics=norm,
                )
                ok = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"Skipping {out_name}: {exc}", file=sys.stderr)
    if not ok and not ca_also:
        print(
            "(Fix --metrics-jsonl or pass --ca-step-jsonl to render other plots.)",
            file=sys.stderr,
        )
    return ok


def _run_ca_trace_plots(
    *,
    ca_path: str,
    out_dir: Path,
    worlds_arg: str,
    metrics_csv: str,
    do_summary: bool,
) -> bool:
    from .plotting import (
        load_ca_step_trace_jsonl,
        plot_ca_step_metrics_timeseries,
        plot_ca_step_pca_trajectories,
        plot_ca_step_umap_trajectories,
        summarize_ca_step_trace_by_world,
    )

    src = Path(ca_path)
    if not src.is_file():
        print(f"Not a file: {src.resolve()}", file=sys.stderr)
        return False

    df = load_ca_step_trace_jsonl(src)
    if df.empty:
        print("CA step trace file is empty.", file=sys.stderr)
        return False

    if worlds_arg:
        yield_indices = [int(x.strip()) for x in worlds_arg.split(",") if x.strip()]
    else:
        uniq = sorted(df["yield_index"].unique().tolist())
        yield_indices = uniq[:8]

    metrics = [m.strip() for m in metrics_csv.split(",") if m.strip()]

    if do_summary:
        summary = summarize_ca_step_trace_by_world(df)
        print(summary.to_string())

    plot_ca_step_metrics_timeseries(
        df,
        yield_indices,
        out_dir / "ca_step_timeseries.png",
        metric_names=metrics,
        title="CA metrics vs step (selected worlds)",
    )
    for stem, plot_fn in (
        ("pca_trajectories", plot_ca_step_pca_trajectories),
        ("umap_trajectories", plot_ca_step_umap_trajectories),
    ):
        for norm, suffix in ((False, ""), (True, "_norm")):
            plot_fn(
                df,
                yield_indices,
                out_dir / f"{stem}{suffix}.png",
                title=None,
                standardize_metrics=norm,
            )
    return True
