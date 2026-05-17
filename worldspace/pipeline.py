from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TextIO

import numpy as np
from sklearn.decomposition import PCA

from . import math as ws_math
from .generators import WorldGenerator
from .metrics import (
    METRIC_KEYS,
    METRICS_VECTOR_DIM,
    metrics_vector_to_dict,
)
from .simulator import run_world
from .specs.spec import WorldSpec


@contextmanager
def memmap_workspace(n: int) -> Iterator[tuple[np.memmap, np.memmap]]:
    """Create metrics and label memory-mapped arrays; unlink backing files on exit."""
    tmp_metrics, tmp_labels = _memmap_temp_paths()
    mm: np.memmap | None = None
    labels: np.memmap | None = None
    try:
        mm = np.memmap(
            tmp_metrics, dtype=np.float32, mode="w+", shape=(n, METRICS_VECTOR_DIM)
        )
        labels = np.memmap(tmp_labels, dtype=np.int32, mode="w+", shape=(n,))
        labels[:] = 0
        yield mm, labels
    finally:
        _release_memmaps(mm, labels)
        _unlink_quiet((tmp_metrics, tmp_labels))


def stream_world_space_to_jsonl(
    generator: WorldGenerator,
    n_worlds: int,
    path: str | Path | None,
    k_clusters: int = 4,
    *,
    echo_stdout: bool = False,
    metrics_trace_path: str | Path | None = None,
    ca_step_trace_path: str | Path | None = None,
) -> None:
    """
    Run generator → simulate → metrics → 2D world-space layout → k-means.

    Each ``WorldSpec`` is kept in RAM for a second pass (small parameter structs) so JSON
    lines match the metrics rows without re-running ``iter_worlds`` (avoids duplicate LLM
    calls for ``LLMWorldGenerator`` / ``HybridGALlmWorldGenerator``).

    Per-world 2D layout ``dominant_metric_delta_xy`` (see :func:`dominant_metric_delta_xy_batch`): **x** is
    the batch's highest-variance metric minus its batch mean; **y** is sklearn PC1 on the
    other ``METRICS_VECTOR_DIM - 1`` metrics (centering only inside that PCA). Not the same as full-vector PCA/UMAP plots.

    Metrics rows live on a temporary memory-mapped file during k-means. If ``path`` is
    set, append each record as one JSON line to that file. If ``path`` is ``None``,
    main JSONL lines are printed only when ``echo_stdout`` is True (default False),
    e.g. trace-only runs stay quiet.

    If ``metrics_trace_path`` is set and ``n_worlds`` > 0, after layout and k-means
    write one JSON line per world: ``yield_index`` plus the same fields as the main
    record (``world``, ``metrics``, ``dominant_metric_delta_xy``, ``dominant_metric_delta_axis_labels``,
    ``cluster_id``).

    If ``ca_step_trace_path`` is set and ``n_worlds`` > 0, append one JSON line per CA
    timestep for each pipeline ``run_world`` (``yield_index``, ``ca_step``, ``metrics``).
    Does not trace extra ``run_world`` calls made inside generators (e.g. LLM scoring).
    """
    file_write = path is not None
    target = Path(path) if file_write else None
    if file_write and target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    n = n_worlds
    if n <= 0:
        # Return before opening trace files: avoids dangling handles when n is 0.
        if file_write and target is not None:
            target.write_text("")
        return

    trace_file: TextIO | None = None
    ca_trace_file: TextIO | None = None
    try:
        raw_trace = metrics_trace_path
        if raw_trace is not None and str(raw_trace).strip():
            tp = Path(str(raw_trace).strip()).expanduser()
            tp.parent.mkdir(parents=True, exist_ok=True)
            trace_file = tp.open("w", encoding="utf-8")

        raw_ca = ca_step_trace_path
        if raw_ca is not None and str(raw_ca).strip():
            cp = Path(str(raw_ca).strip()).expanduser()
            cp.parent.mkdir(parents=True, exist_ok=True)
            ca_trace_file = cp.open("w", encoding="utf-8")

        with memmap_workspace(n) as (mm, labels):
            worlds = _accumulate_metrics_memmap(generator, n, mm, ca_trace_file)
            labels.flush()

            x = np.asarray(mm[:n], dtype=np.float64)
            mean, dominant_idx, axis_name, pca = _fit_dominant_metric_orthogonal_pca(x)
            ws_math.kmeans_lloyd_on_memmap(mm, labels, n, k_clusters)

            point_dicts = list(
                _iter_space_point_dicts(
                    worlds, mm, labels, mean, dominant_idx, axis_name, pca
                )
            )
            lines = (json.dumps(row, ensure_ascii=True) for row in point_dicts)
            if file_write and target is not None:
                _write_jsonl_to_path(target, lines, echo_stdout)
            else:
                if echo_stdout:
                    for row in point_dicts:
                        print(json.dumps(row, ensure_ascii=True), flush=True)

            if trace_file is not None:
                for i, row in enumerate(point_dicts):
                    trace_file.write(
                        json.dumps({"yield_index": i, **row}, ensure_ascii=True) + "\n"
                    )
                trace_file.flush()

            if echo_stdout:
                _print_generator_fallback_summary(generator)
    finally:
        if trace_file is not None:
            trace_file.close()
        if ca_trace_file is not None:
            ca_trace_file.close()


def dominant_metric_delta_axis_labels(x_axis_metric: str) -> dict[str, str]:
    """Axis metadata for ``dominant_metric_delta_xy`` (written to ``--metrics-trace`` JSONL)."""
    om = METRICS_VECTOR_DIM - 1
    return {
        "x_metric": x_axis_metric,
        "x_label": f"Δ {x_axis_metric}",
        "y_label": f"PC1 of {om} metrics (excluding {x_axis_metric})",
    }


def dominant_metric_delta_xy_batch(X: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    """
    Batch 2D dominant-metric-delta layout for metric rows ``X`` of shape ``(n, d)`` with
    ``d = METRICS_VECTOR_DIM``.

    **x** (``dominant_metric_delta_xy[0]``): value of the batch's highest-variance metric minus
    that metric's batch mean (one coordinate, not PCA).

    **y** (``dominant_metric_delta_xy[1]``): sklearn ``PCA(n_components=1)`` on the **other
    ``d - 1``** columns only; sklearn centers those columns inside ``fit``/``transform``.
    The dominant metric is excluded from the PCA subspace entirely.

    For ``n < 2``, **y** is always ``0``. See ``docs/WORLDSPACE.md`` §6.1.
    """
    x = np.asarray(X, dtype=np.float64)
    mean, dominant_idx, axis_name, pca = _fit_dominant_metric_orthogonal_pca(x)
    n = int(x.shape[0])
    z = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        z[i, 0], z[i, 1] = _project_dominant_metric_orthogonal(
            x[i], mean, dominant_idx, pca
        )
    return z, dominant_metric_delta_axis_labels(axis_name)


def _unlink_quiet(paths: tuple[str, ...]) -> None:
    """Remove files if present; ignore OS errors (best-effort cleanup)."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _release_memmaps(mm: np.memmap | None, labels: np.memmap | None) -> None:
    """Drop mmap references so the backing files can be unlinked on Windows and elsewhere."""
    if mm is not None:
        del mm
    if labels is not None:
        del labels


def _memmap_temp_paths() -> tuple[str, str]:
    """Create two empty temp files and return their paths (file descriptors closed)."""
    fd_m, tmp_metrics = tempfile.mkstemp(suffix=".metrics.f32")
    fd_l, tmp_labels = tempfile.mkstemp(suffix=".labels.i4")
    os.close(fd_m)
    os.close(fd_l)
    return tmp_metrics, tmp_labels


def _accumulate_metrics_memmap(
    generator: WorldGenerator,
    n: int,
    mm: np.memmap,
    ca_step_trace_file: TextIO | None,
) -> list[WorldSpec]:
    """Simulate each world, write float32 metric rows to ``mm``, return specs for pass 2."""
    worlds: list[WorldSpec] = []
    for i, world in enumerate(generator.iter_worlds(n)):
        worlds.append(world)
        if ca_step_trace_file is not None:
            vec = run_world(
                world,
                ca_step_trace_file=ca_step_trace_file,
                ca_step_trace_yield_index=i,
            ).metrics.as_vector()
        else:
            vec = run_world(world).metrics.as_vector()
        mm[i] = vec.astype(np.float32)
    mm.flush()
    return worlds


def _fit_dominant_metric_orthogonal_pca(
    X: np.ndarray,
) -> tuple[np.ndarray, int, str, PCA | None]:
    """
    Horizontal axis: most variable metric in the batch minus its mean.

    Vertical axis: ``sklearn.decomposition.PCA(n_components=1)`` on the **raw**
    remaining ``METRICS_VECTOR_DIM - 1`` columns of ``X``. sklearn centers those columns internally
    (``mean_`` equals their column means);
    training rows are **not** pre-centered so centering happens only inside PCA.
    """
    mean = X.mean(axis=0, dtype=np.float64)
    var = X.var(axis=0, dtype=np.float64)
    j = int(np.argmax(var))
    axis_name = METRIC_KEYS[j]
    n = int(X.shape[0])
    if n < 2:
        return mean, j, axis_name, None
    x_rest = np.delete(X, j, axis=1)
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(x_rest)
    return mean, j, axis_name, pca


def _project_dominant_metric_orthogonal(
    vec: np.ndarray,
    mean: np.ndarray,
    dominant_index: int,
    pca: PCA | None,
) -> tuple[float, float]:
    v = vec.astype(np.float64)
    x = float(v[dominant_index] - mean[dominant_index])
    if pca is None:
        return x, 0.0
    v_rest = np.delete(v, dominant_index)
    y = float(pca.transform(v_rest.reshape(1, -1))[0, 0])
    return x, y


def _space_point_row(
    world: WorldSpec,
    vec: np.ndarray,
    mean: np.ndarray,
    dominant_index: int,
    x_axis_metric: str,
    pca: PCA | None,
    label: int,
) -> dict:
    """Build one JSON-serializable record (world + metrics + world-space xy + cluster)."""
    xy = _project_dominant_metric_orthogonal(vec, mean, dominant_index, pca)
    return {
        "world": world.to_json_dict(),
        "metrics": metrics_vector_to_dict(vec),
        "dominant_metric_delta_xy": [xy[0], xy[1]],
        "dominant_metric_delta_axis_labels": dominant_metric_delta_axis_labels(
            x_axis_metric
        ),
        "cluster_id": label,
    }


def _iter_space_point_dicts(
    worlds: Sequence[WorldSpec],
    mm: np.memmap,
    labels: np.memmap,
    mean: np.ndarray,
    dominant_index: int,
    x_axis_metric: str,
    pca: PCA | None,
) -> Iterator[dict]:
    """Yield one space record dict per cached world (metrics rows + k-means labels)."""
    for i, world in enumerate(worlds):
        vec = mm[i].astype(np.float64)
        lab = int(labels[i])
        yield _space_point_row(
            world, vec, mean, dominant_index, x_axis_metric, pca, lab
        )


def _print_generator_fallback_summary(generator: WorldGenerator) -> None:
    """After echoed JSONL lines, print fallback stats on stderr (keeps stdout JSON-clean)."""
    summary = json.dumps(
        {"generator_fallback_count": generator.fallback_count},
        ensure_ascii=True,
    )
    print(summary, file=sys.stderr, flush=True)


def _write_jsonl_to_path(
    target: Path,
    lines: Iterator[str],
    echo_stdout: bool,
) -> None:
    """Write each line to ``target``; optionally mirror lines to stdout."""
    with target.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line + "\n")
            if echo_stdout:
                print(line, flush=True)
