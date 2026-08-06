"""Unit tests for H1 shuffled hint placebo."""

from __future__ import annotations

import unittest

import numpy as np

from worldspace.illuminators.emitters.llm_emitter import (
    LlmPreparedSlot,
    apply_batch_hint_placebo,
    remap_prepared_slot_prediction,
)
from worldspace.illuminators.scheduler import TargetCell, load_scheduler
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.model import TARGET_KEYS
from worldspace.surrogate.types import SurrogatePrediction

_TARGET = TargetCell(
    cell_id=6,
    target_stability=0.5,
    target_diversity=0.6,
    bin_ij=(1, 1),
)
_BASE = WorldSpec(
    birth=[1, 3],
    survival=[2, 3],
    noise=0.02,
    resource_regen=0.05,
    predation=0.1,
    cell_types=["life", "food"],
    grid_size=8,
    steps=200,
    seed=0,
)


def _pred(fitness: float, uncertainty: float) -> SurrogatePrediction:
    components = {key: float(fitness) for key in TARGET_KEYS}
    return SurrogatePrediction(
        components=components,
        measures={"stability": fitness, "diversity": fitness},
        fitness=fitness,
        uncertainty=uncertainty,
    )


def _slot(fitness: float, uncertainty: float, marker: str) -> LlmPreparedSlot:
    pred = _pred(fitness, uncertainty)
    user = (
        f"Target niche: stability ≈ 0.50 (±0.03), diversity ≈ 0.60 (±0.03)\n"
        f"Surrogate predicts fitness ≈ {fitness:.3f}, uncertainty = {uncertainty:.3f}\n"
        f"PARENT_MARKER={marker}\n"
    )
    return LlmPreparedSlot(
        target=_TARGET,
        parent_spec=_BASE,
        parent_id=marker,
        system_prompt="sys",
        user_prompt=user,
        prompt_version="test",
        grid_size=8,
        steps=200,
        surrogate_prediction=pred,
    )


class HintPlaceboTests(unittest.TestCase):
    def test_remap_preserves_parent_marker(self) -> None:
        slot = _slot(0.40, 0.10, "A")
        remapped = remap_prepared_slot_prediction(slot, _pred(0.80, 0.20))
        self.assertIn("PARENT_MARKER=A", remapped.user_prompt)
        self.assertIn(
            "Surrogate predicts fitness ≈ 0.800, uncertainty = 0.200",
            remapped.user_prompt,
        )
        self.assertNotIn(
            "Surrogate predicts fitness ≈ 0.400, uncertainty = 0.100",
            remapped.user_prompt,
        )
        assert remapped.surrogate_prediction is not None
        self.assertAlmostEqual(remapped.surrogate_prediction.fitness, 0.80)

    def test_batch_shuffle_preserves_pair_multiset(self) -> None:
        slots = [
            _slot(0.10, 0.01, "A"),
            _slot(0.20, 0.02, "B"),
            _slot(0.30, 0.03, "C"),
            _slot(0.40, 0.04, "D"),
        ]
        before = sorted(
            (s.surrogate_prediction.fitness, s.surrogate_prediction.uncertainty)  # type: ignore[union-attr]
            for s in slots
        )
        parents_before = [s.parent_id for s in slots]
        markers_before = [f"PARENT_MARKER={s.parent_id}" for s in slots]

        # Seed chosen so permutation is not identity for n=4 under numpy Generator.
        rng = np.random.default_rng(7)
        shuffled = apply_batch_hint_placebo(slots, rng)
        after = sorted(
            (s.surrogate_prediction.fitness, s.surrogate_prediction.uncertainty)  # type: ignore[union-attr]
            for s in shuffled
        )
        self.assertEqual(before, after)
        self.assertEqual([s.parent_id for s in shuffled], parents_before)
        for slot, marker in zip(shuffled, markers_before, strict=True):
            self.assertIn(marker, slot.user_prompt)
        # At least one slot should receive a non-original pair (with high probability;
        # force-check against original pairing).
        original_pairs = [
            (s.surrogate_prediction.fitness, s.surrogate_prediction.uncertainty)  # type: ignore[union-attr]
            for s in slots
        ]
        new_pairs = [
            (s.surrogate_prediction.fitness, s.surrogate_prediction.uncertainty)  # type: ignore[union-attr]
            for s in shuffled
        ]
        self.assertNotEqual(original_pairs, new_pairs)

    def test_scheduler_yaml_loads_shuffle_batch(self) -> None:
        cfg = load_scheduler(
            "worldspace/specs/map_elites_scheduler_nightly_llm_hints_placebo.yaml"
        )
        self.assertEqual(cfg.llm_hint_placebo, "shuffle_batch")
        self.assertTrue(cfg.surrogate_enabled)
        self.assertTrue(cfg.performance.llm_parallel_emit)


if __name__ == "__main__":
    unittest.main()
