from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from . import math as ws_math
from .generators import WorldGenerator
from .metrics import (
    METRIC_INDEX_AVERAGE_LIFESPAN,
    METRICS_VECTOR_DIM,
    metrics_vector_to_dict,
)
from .simulator import run_world
from .spec import WorldSpec


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
    echo_stdout: bool = True,
) -> None:
    """
    Run generator → simulate → metrics → 2D embedding → k-means with **O(1) RAM** vs. batch size.

    The 2D embedding is **not** plain PCA on all seven metrics: horizontal axis is
    ``average_lifespan`` minus batch mean; vertical axis is the first sklearn PCA
    component of the raw six non-lifespan metrics (single internal centering;
    ``_fit_lifespan_orthogonal_pca``).

    Metrics rows live on a temporary memory-mapped file during k-means. If ``path`` is
    set, append each record as one JSON line to that file. If ``path`` is ``None``,
    records are only printed (``echo_stdout`` is treated as True).
    """
    file_write = path is not None
    target = Path(path) if file_write else None
    if file_write and target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    n = n_worlds
    if n <= 0:
        if file_write and target is not None:
            target.write_text("")
        return
    if not file_write:
        echo_stdout = True

    with memmap_workspace(n) as (mm, labels):
        _accumulate_metrics_memmap(generator, n, mm)
        labels.flush()

        x = np.asarray(mm[:n], dtype=np.float64)
        mean, lifespan_idx, pca = _fit_lifespan_orthogonal_pca(x)
        ws_math.kmeans_lloyd_on_memmap(mm, labels, n, k_clusters)

        lines = _iter_space_json_lines(
            generator, n, mm, labels, mean, lifespan_idx, pca
        )
        if file_write and target is not None:
            _write_jsonl_to_path(target, lines, echo_stdout)
        else:
            for line in lines:
                print(line, flush=True)


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
) -> None:
    """Simulate each world and write float32 metric rows to ``mm``."""
    for i, world in enumerate(generator.iter_worlds(n)):
        vec = run_world(world).metrics.as_vector()
        mm[i] = vec.astype(np.float32)
    mm.flush()


def _fit_lifespan_orthogonal_pca(X: np.ndarray) -> tuple[np.ndarray, int, PCA | None]:
    """
    Horizontal axis: ``average_lifespan`` minus batch mean.

    Vertical axis: ``sklearn.decomposition.PCA(n_components=1)`` on the **raw** six
    non-lifespan columns of ``X``. sklearn centers those columns internally
    (``mean_`` equals the column means, i.e. ``mean`` with the lifespan entry removed);
    training rows are **not** pre-centered so centering happens only inside PCA.
    """
    mean = X.mean(axis=0, dtype=np.float64)
    j = METRIC_INDEX_AVERAGE_LIFESPAN
    n = int(X.shape[0])
    if n < 2:
        return mean, j, None
    x_rest = np.delete(X, j, axis=1)
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(x_rest)
    return mean, j, pca


def _project_lifespan_orthogonal(
    vec: np.ndarray,
    mean: np.ndarray,
    lifespan_index: int,
    pca: PCA | None,
) -> tuple[float, float]:
    v = vec.astype(np.float64)
    x = float(v[lifespan_index] - mean[lifespan_index])
    if pca is None:
        return x, 0.0
    v_rest = np.delete(v, lifespan_index)
    y = float(pca.transform(v_rest.reshape(1, -1))[0, 0])
    return x, y


def _space_point_row(
    world: WorldSpec,
    vec: np.ndarray,
    mean: np.ndarray,
    lifespan_index: int,
    pca: PCA | None,
    label: int,
) -> dict:
    """Build one JSON-serializable record (world + metrics + 2D embedding + cluster)."""
    emb = _project_lifespan_orthogonal(vec, mean, lifespan_index, pca)
    return {
        "world": world.to_json_dict(),
        "metrics": metrics_vector_to_dict(vec),
        "embedding_2d": [emb[0], emb[1]],
        "cluster_id": label,
    }


def _iter_space_json_lines(
    generator: WorldGenerator,
    n: int,
    mm: np.memmap,
    labels: np.memmap,
    mean: np.ndarray,
    lifespan_index: int,
    pca: PCA | None,
) -> Iterator[str]:
    """Yield JSON lines for each world using metrics rows and k-means labels from mmap."""
    for i, world in enumerate(generator.iter_worlds(n)):
        vec = mm[i].astype(np.float64)
        lab = int(labels[i])
        row = _space_point_row(world, vec, mean, lifespan_index, pca, lab)
        yield json.dumps(row, ensure_ascii=True)


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
