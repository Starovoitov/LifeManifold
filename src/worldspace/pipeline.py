from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import math as ws_math
from .generators import WorldGenerator
from .metrics import WorldMetrics, compute_metrics
from .simulator import run_world
from .spec import WorldSpec


@dataclass
class SpacePoint:
    """One explored world mapped to metrics, embedding, and cluster."""

    world: WorldSpec
    metrics: WorldMetrics
    embedding_2d: tuple[float, float]
    cluster_id: int


def explore_world_space(generator: WorldGenerator, n_worlds: int, k_clusters: int = 4) -> list[SpacePoint]:
    """Run generator->simulation->metrics->embedding->clustering pipeline."""
    worlds = generator.generate(n_worlds)
    metrics_list: list[WorldMetrics] = []
    vectors = []

    for world in worlds:
        result = run_world(world)
        metrics = compute_metrics(result)
        metrics_list.append(metrics)
        vectors.append(metrics.as_vector())

    matrix = np.array(vectors, dtype=float)
    emb = ws_math.pca_2d(matrix)
    labels = ws_math.kmeans(matrix, k=k_clusters)

    points: list[SpacePoint] = []
    for world, metrics, xy, label in zip(worlds, metrics_list, emb, labels):
        points.append(
            SpacePoint(
                world=world,
                metrics=metrics,
                embedding_2d=(float(xy[0]), float(xy[1])),
                cluster_id=int(label),
            )
        )
    return points


def points_to_dicts(points: list[SpacePoint]) -> list[dict]:
    """Convert space points to JSON-serializable dictionaries."""
    return [
        {
            "world": p.world.to_json_dict(),
            "metrics": {
                "entropy": p.metrics.entropy,
                "stability": p.metrics.stability,
                "average_lifespan": p.metrics.average_lifespan,
                "density_mean": p.metrics.density_mean,
                "oscillation_score": p.metrics.oscillation_score,
                "diversity": p.metrics.diversity,
            },
            "embedding_2d": list(p.embedding_2d),
            "cluster_id": p.cluster_id,
        }
        for p in points
    ]


def save_points_jsonl(points: list[SpacePoint], path: str | Path) -> None:
    """Save explored world points as JSONL (one record per line)."""
    rows = points_to_dicts(points)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
