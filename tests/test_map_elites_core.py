"""Unit tests for MAP-Elites illuminator core (canonical seed, measures, fitness, bin)."""

from __future__ import annotations

import hashlib
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from worldspace.illuminators.evaluation import (
    ILLUMINATOR_MIN_STEPS,
    MEASURE_KEYS,
    EvalResult,
    apply_canonical_seed,
    bin_index,
    bin_index_from_measures,
    canonical_seed,
    compute_fitness,
    evaluate_candidate,
    extinction_probability,
    measures_from_metrics,
    topology_complexity,
)
from worldspace.metrics import METRICS_VECTOR_DIM, WorldMetrics
from worldspace.simulator import run_world
from worldspace.specs.spec import CANONICAL_CELL_TYPES, WorldSpec

_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":")}


def _canonical_json(spec: WorldSpec) -> str:
    return json.dumps(spec.to_canonical_dict(), **_CANONICAL_JSON_KWARGS)


class TestWorldSpecCanonicalDict(unittest.TestCase):
    def test_canonical_dict_excludes_seed(self) -> None:
        spec = WorldSpec(
            birth=[1],
            survival=[2, 3],
            noise=0.01,
            resource_regen=0.1,
            predation=0.2,
            cell_types=["life", "food"],
            seed=999,
        )
        canonical = spec.to_canonical_dict()
        self.assertNotIn("seed", canonical)
        self.assertEqual(spec.seed, 999)

    def test_canonical_dict_stable_json(self) -> None:
        a = WorldSpec(
            birth=[3, 1, 2],
            survival=[5, 4],
            noise=0.10000004,
            resource_regen=0.05,
            predation=0.25,
            cell_types=["empty", "life", "food"],
            grid_size=40,
            steps=250,
            seed=1,
        )
        b = WorldSpec(
            birth=[2, 3, 1],
            survival=[4, 5],
            noise=0.10000005,
            resource_regen=0.05,
            predation=0.25,
            cell_types=["food", "life", "empty"],
            grid_size=40,
            steps=250,
            seed=99,
        )
        self.assertEqual(_canonical_json(a), _canonical_json(b))

    def test_canonical_dict_sorts_rule_lists(self) -> None:
        spec = WorldSpec(
            birth=[3, 1, 2],
            survival=[8, 2, 5],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["life", "food"],
        )
        canonical = spec.to_canonical_dict()
        self.assertEqual(canonical["birth"], [1, 2, 3])
        self.assertEqual(canonical["survival"], [2, 5, 8])

    def test_canonical_dict_rounds_floats(self) -> None:
        a = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.10000004,
            resource_regen=0.20000004,
            predation=0.30000004,
            cell_types=["life", "food"],
        )
        b = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.10000005,
            resource_regen=0.20000005,
            predation=0.30000005,
            cell_types=["life", "food"],
        )
        self.assertEqual(_canonical_json(a), _canonical_json(b))
        canonical = a.to_canonical_dict()
        self.assertEqual(canonical["noise"], 0.1)
        self.assertEqual(canonical["resource_regen"], 0.2)
        self.assertEqual(canonical["predation"], 0.3)

    def test_canonical_dict_normalizes_cell_types(self) -> None:
        spec = WorldSpec(
            birth=[0],
            survival=[1],
            noise=0.0,
            resource_regen=0.0,
            predation=0.0,
            cell_types=["empty", "life", "food"],
        )
        self.assertEqual(
            spec.to_canonical_dict()["cell_types"], list(CANONICAL_CELL_TYPES)
        )

    def test_canonical_dict_differs_from_to_json_dict(self) -> None:
        spec = WorldSpec(
            birth=[2, 1],
            survival=[3],
            noise=0.05,
            resource_regen=0.1,
            predation=0.15,
            cell_types=["empty", "life", "food"],
            seed=42,
        )
        canonical = spec.to_canonical_dict()
        raw = spec.to_json_dict()
        self.assertNotIn("seed", canonical)
        self.assertIn("seed", raw)
        self.assertEqual(canonical["cell_types"], ["life", "food"])
        self.assertEqual(raw["cell_types"], ["empty", "life", "food"])
        self.assertEqual(canonical["birth"], [1, 2])
        self.assertEqual(raw["birth"], [2, 1])


_BASE_WORLD_SPEC = WorldSpec(
    birth=[1],
    survival=[2],
    noise=0.0,
    resource_regen=0.0,
    predation=0.0,
    cell_types=["life", "food"],
    grid_size=4,
    steps=300,
    seed=0,
)


def _minimal_world_spec(**overrides: object) -> WorldSpec:
    return replace(_BASE_WORLD_SPEC, **overrides)


class TestCanonicalSeed(unittest.TestCase):
    def test_canonical_seed_stable(self) -> None:
        a = _minimal_world_spec(seed=1)
        b = _minimal_world_spec(
            birth=[1],
            survival=[2],
            seed=99,
            cell_types=["empty", "life", "food"],
        )
        self.assertEqual(canonical_seed(a), canonical_seed(b))

    def test_canonical_seed_differs(self) -> None:
        a = _minimal_world_spec(grid_size=4)
        b = _minimal_world_spec(grid_size=8)
        self.assertNotEqual(canonical_seed(a), canonical_seed(b))

    def test_canonical_seed_ignores_spec_seed_field(self) -> None:
        a = _minimal_world_spec(seed=1)
        b = _minimal_world_spec(seed=99)
        self.assertEqual(canonical_seed(a), canonical_seed(b))

    def test_canonical_seed_range(self) -> None:
        seed = canonical_seed(_minimal_world_spec())
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_apply_canonical_seed_mutates_spec(self) -> None:
        spec = _minimal_world_spec(seed=0)
        applied = apply_canonical_seed(spec)
        self.assertEqual(spec.seed, applied)
        self.assertEqual(spec.seed, canonical_seed(spec))

    def test_canonical_seed_matches_tz_formula(self) -> None:
        spec = _minimal_world_spec()
        payload = json.dumps(
            spec.to_canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        expected = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % (
            2**32
        )
        self.assertEqual(canonical_seed(spec), expected)


class TestEarlyExtinction(unittest.TestCase):
    def test_early_extinct_at_init_zero_ca_steps(self) -> None:
        n = 4
        zeros = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        spec = _minimal_world_spec(steps=500)

        with patch(
            "worldspace.simulator._initial_grids",
            return_value=(zeros, zeros, ages),
        ):
            buf = io.StringIO()
            run_world(
                spec,
                early_extinction_step=200,
                ca_step_trace_file=buf,
            )
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 0)

    def test_early_extinct_before_threshold(self) -> None:
        n = 4
        life = np.ones((n, n), dtype=np.uint8)
        food = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        spec = _minimal_world_spec(steps=500)

        with (
            patch(
                "worldspace.simulator._initial_grids",
                return_value=(life, food, ages),
            ),
            patch(
                "worldspace.simulator._next_life_from_rules",
                return_value=np.zeros((n, n), dtype=np.uint8),
            ),
        ):
            buf = io.StringIO()
            run_world(
                spec,
                early_extinction_step=200,
                ca_step_trace_file=buf,
            )
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)

    def test_early_extinction_disabled_runs_full_steps(self) -> None:
        n = 4
        life = np.ones((n, n), dtype=np.uint8)
        food = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        steps = 30
        spec = _minimal_world_spec(steps=steps)

        with (
            patch(
                "worldspace.simulator._initial_grids",
                return_value=(life, food, ages),
            ),
            patch(
                "worldspace.simulator._next_life_from_rules",
                return_value=np.zeros((n, n), dtype=np.uint8),
            ),
        ):
            buf = io.StringIO()
            run_world(
                spec,
                early_extinction_step=None,
                ca_step_trace_file=buf,
            )
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), steps)

    def test_early_extinction_none_matches_default(self) -> None:
        spec = _minimal_world_spec(grid_size=6, steps=12, seed=7)
        explicit = run_world(spec, early_extinction_step=None).metrics.as_vector()
        default = run_world(spec).metrics.as_vector()
        np.testing.assert_allclose(explicit, default)


class TestMeasuresFromMetrics(unittest.TestCase):
    def test_measures_from_metrics_clips(self) -> None:
        metrics = _example_metrics(stability=-0.1, diversity=1.5)
        measures = measures_from_metrics(metrics)
        self.assertEqual(measures["stability"], 0.0)
        self.assertEqual(measures["diversity"], 1.0)

    def test_measures_from_metrics_keys(self) -> None:
        measures = measures_from_metrics(_example_metrics())
        self.assertEqual(tuple(measures.keys()), MEASURE_KEYS)

    def test_measures_from_metrics_identity_when_in_range(self) -> None:
        metrics = _example_metrics(stability=0.55, diversity=0.68)
        measures = measures_from_metrics(metrics)
        self.assertEqual(measures["stability"], 0.55)
        self.assertEqual(measures["diversity"], 0.68)

    def test_measures_after_early_extinct_run(self) -> None:
        n = 4
        zeros = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        spec = _minimal_world_spec(steps=500, seed=3)

        with patch(
            "worldspace.simulator._initial_grids",
            return_value=(zeros, zeros, ages),
        ):
            result = run_world(spec, early_extinction_step=200)

        measures = measures_from_metrics(result.metrics)
        self.assertEqual(set(measures.keys()), {"stability", "diversity"})
        for value in measures.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_measures_after_normal_run(self) -> None:
        spec = _minimal_world_spec(grid_size=8, steps=20, seed=11)
        result = run_world(spec)
        measures = measures_from_metrics(result.metrics)
        for key in MEASURE_KEYS:
            self.assertIn(key, measures)
            self.assertGreaterEqual(measures[key], 0.0)
            self.assertLessEqual(measures[key], 1.0)


class TestTopologyComplexity(unittest.TestCase):
    def test_topology_complexity_formula(self) -> None:
        metrics = _example_metrics(
            topology_interface_index=0.6,
            topology_window_heterogeneity=0.4,
        )
        self.assertAlmostEqual(topology_complexity(metrics), 0.5)

    def test_topology_complexity_clips(self) -> None:
        metrics = _example_metrics(
            topology_interface_index=2.0,
            topology_window_heterogeneity=2.0,
        )
        self.assertEqual(topology_complexity(metrics), 1.0)

    def test_topology_complexity_not_in_measures(self) -> None:
        measures = measures_from_metrics(_example_metrics())
        self.assertNotIn("topology_complexity", measures)


class TestExtinctionProbabilityAndFitness(unittest.TestCase):
    def test_extinction_probability_bounds(self) -> None:
        self.assertEqual(extinction_probability(1.0), 0.0)
        self.assertEqual(extinction_probability(0.0), 1.0)

    def test_compute_fitness_early_extinct_is_zero(self) -> None:
        measures = measures_from_metrics(_example_metrics(stability=0.8, diversity=0.9))
        fitness = compute_fitness(
            _example_metrics(),
            measures,
            early_extinct=True,
            final_density=0.0,
        )
        self.assertEqual(fitness, 0.0)

    def test_compute_fitness_formula(self) -> None:
        metrics = _example_metrics(
            oscillation_score=0.4,
            topology_interface_index=0.2,
            topology_window_heterogeneity=0.6,
        )
        measures = {"stability": 0.5, "diversity": 0.68}
        final_density = 0.3
        expected = 0.45 * 0.68 + 0.25 * (1.0 - 0.7) + 0.20 * 0.4 + 0.10 * 0.4
        fitness = compute_fitness(
            metrics,
            measures,
            early_extinct=False,
            final_density=final_density,
        )
        self.assertAlmostEqual(fitness, expected)

    def test_compute_fitness_integration_early_extinct(self) -> None:
        n = 4
        zeros = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        spec = _minimal_world_spec(steps=500, seed=5)

        with patch(
            "worldspace.simulator._initial_grids",
            return_value=(zeros, zeros, ages),
        ):
            result = run_world(spec, early_extinction_step=200)

        self.assertTrue(result.early_extinct)
        self.assertIsNotNone(result.final_life)
        measures = measures_from_metrics(result.metrics)
        fitness = compute_fitness(
            result.metrics,
            measures,
            early_extinct=result.early_extinct,
            final_density=float(result.final_life.mean()),
        )
        self.assertEqual(fitness, 0.0)

    def test_compute_fitness_integration_normal_run(self) -> None:
        spec = _minimal_world_spec(grid_size=8, steps=20, seed=13)
        result = run_world(spec, early_extinction_step=200)
        self.assertFalse(result.early_extinct)
        self.assertIsNotNone(result.final_life)
        measures = measures_from_metrics(result.metrics)
        fitness = compute_fitness(
            result.metrics,
            measures,
            early_extinct=result.early_extinct,
            final_density=float(result.final_life.mean()),
        )
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

    def test_metrics_vector_dim_unchanged(self) -> None:
        self.assertEqual(METRICS_VECTOR_DIM, 12)


class TestBinIndex(unittest.TestCase):
    def test_bin_index_unity_goes_to_last_cell(self) -> None:
        resolution = 50
        self.assertEqual(bin_index(1.0, 1.0, resolution), (49, 49))

    def test_bin_index_zero_at_origin(self) -> None:
        self.assertEqual(bin_index(0.0, 0.0, 50), (0, 0))

    def test_bin_index_clips_out_of_range(self) -> None:
        self.assertEqual(bin_index(-0.5, 1.5, 50), bin_index(0.0, 1.0, 50))

    def test_bin_index_from_measures_matches(self) -> None:
        measures = {"stability": 0.55, "diversity": 0.68}
        self.assertEqual(
            bin_index_from_measures(measures, 50),
            bin_index(0.55, 0.68, 50),
        )

    def test_bin_index_resolution_one(self) -> None:
        self.assertEqual(bin_index(0.5, 0.9, 1), (0, 0))


class TestEvaluateCandidate(unittest.TestCase):
    def test_evaluate_candidate_reproducible(self) -> None:
        spec_a = _minimal_world_spec(grid_size=8, steps=220, seed=1)
        spec_b = _minimal_world_spec(grid_size=8, steps=220, seed=99)
        result_a = evaluate_candidate(spec_a, resolution=20)
        result_b = evaluate_candidate(spec_b, resolution=20)
        self.assertEqual(result_a.world_spec.seed, result_b.world_spec.seed)
        self.assertEqual(result_a.fitness, result_b.fitness)
        self.assertEqual(result_a.measures, result_b.measures)
        self.assertEqual(result_a.bin, result_b.bin)
        np.testing.assert_allclose(
            result_a.metrics.as_vector(),
            result_b.metrics.as_vector(),
        )

    def test_evaluate_candidate_sets_seed(self) -> None:
        spec = _minimal_world_spec(seed=0)
        result = evaluate_candidate(spec, resolution=10)
        self.assertEqual(result.world_spec.seed, canonical_seed(result.world_spec))

    def test_evaluate_candidate_does_not_mutate_input(self) -> None:
        spec = _minimal_world_spec(steps=250, seed=42)
        original_steps = spec.steps
        original_seed = spec.seed
        evaluate_candidate(spec, resolution=10)
        self.assertEqual(spec.steps, original_steps)
        self.assertEqual(spec.seed, original_seed)

    def test_evaluate_candidate_enforces_min_steps(self) -> None:
        spec = _minimal_world_spec(steps=50, seed=7)
        result = evaluate_candidate(spec, resolution=10)
        self.assertGreaterEqual(result.world_spec.steps, ILLUMINATOR_MIN_STEPS)

    def test_evaluate_candidate_early_extinct(self) -> None:
        n = 4
        zeros = np.zeros((n, n), dtype=np.uint8)
        ages = np.zeros((n, n), dtype=np.int16)
        spec = _minimal_world_spec(steps=500, seed=8)

        with patch(
            "worldspace.simulator._initial_grids",
            return_value=(zeros, zeros, ages),
        ):
            result = evaluate_candidate(spec, resolution=15)

        self.assertTrue(result.early_extinct)
        self.assertEqual(result.fitness, 0.0)
        for value in result.measures.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_evaluate_candidate_bin_matches_measures(self) -> None:
        spec = _minimal_world_spec(grid_size=8, steps=220, seed=12)
        result = evaluate_candidate(spec, resolution=30)
        self.assertEqual(
            result.bin,
            bin_index_from_measures(result.measures, 30),
        )

    def test_evaluate_candidate_returns_eval_result(self) -> None:
        result = evaluate_candidate(_minimal_world_spec(steps=200), resolution=5)
        self.assertIsInstance(result, EvalResult)


def _example_metrics(**overrides: float) -> WorldMetrics:
    defaults: dict[str, float] = {
        "entropy": 0.5,
        "stability": 0.5,
        "average_lifespan": 1.0,
        "density_mean": 0.3,
        "oscillation_score": 0.2,
        "diversity": 0.4,
        "mo_eoc_indicator": 0.5,
        "topology_interface_index": 0.3,
        "topology_window_heterogeneity": 0.4,
        "compressibility_score": 0.5,
        "ecology_state_entropy_norm": 0.6,
        "ecology_resource_adjacency": 0.2,
    }
    defaults.update(overrides)
    return WorldMetrics(
        entropy=defaults["entropy"],
        stability=defaults["stability"],
        average_lifespan=defaults["average_lifespan"],
        density_mean=defaults["density_mean"],
        oscillation_score=defaults["oscillation_score"],
        diversity=defaults["diversity"],
        mo_eoc_indicator=defaults["mo_eoc_indicator"],
        topology_interface_index=defaults["topology_interface_index"],
        topology_window_heterogeneity=defaults["topology_window_heterogeneity"],
        compressibility_score=defaults["compressibility_score"],
        ecology_state_entropy_norm=defaults["ecology_state_entropy_norm"],
        ecology_resource_adjacency=defaults["ecology_resource_adjacency"],
    )


if __name__ == "__main__":
    unittest.main()
