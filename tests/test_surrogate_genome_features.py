from __future__ import annotations

import unittest

import numpy as np

from worldspace.illuminators.emitters.genetics import encode_world
from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract,
)
from worldspace.surrogate.genome_features import FEATURE_DIM, encode_world_spec_features


def _canonical_spec(**overrides: object) -> WorldSpec:
    spec = WorldSpec(
        birth=[3],
        survival=[2, 3],
        noise=0.1,
        resource_regen=0.2,
        predation=0.05,
        cell_types=["life", "food"],
        grid_size=30,
        steps=220,
        seed=0,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    apply_canonical_seed(spec)
    return spec


class TestSurrogateGenomeFeatures(unittest.TestCase):
    def test_encode_shape_and_dtype(self) -> None:
        spec = _canonical_spec()
        vector = encode_world_spec_features(spec)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertEqual(vector.dtype, np.float64)

    def test_encode_matches_genetics_encode_world(self) -> None:
        specs = [
            _canonical_spec(),
            _canonical_spec(birth=[1, 4, 7], survival=[0, 8]),
            _canonical_spec(birth=[2, 6], survival=[1, 3, 5], noise=0.05),
        ]
        for spec in specs:
            encoded = encode_world_spec_features(spec)
            expected = encode_world(spec)
            np.testing.assert_allclose(encoded, expected)

    def test_encode_is_deterministic(self) -> None:
        spec = _canonical_spec(birth=[0, 2, 5], survival=[1, 6, 8])
        first = encode_world_spec_features(spec)
        second = encode_world_spec_features(spec)
        np.testing.assert_array_equal(first, second)

    def test_bit_masks_follow_rule_indices(self) -> None:
        spec = _canonical_spec(birth=[3], survival=[2, 8])
        vector = encode_world_spec_features(spec)
        birth_bits = vector[0:9]
        survival_bits = vector[9:18]
        self.assertEqual(birth_bits.tolist(), [0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(survival_bits.tolist(), [0, 0, 1, 0, 0, 0, 0, 0, 1])

    def test_same_v1_density_specs_have_different_v2_vectors(self) -> None:
        left = _canonical_spec(birth=[2, 6], survival=[1, 7])
        right = _canonical_spec(birth=[1, 7], survival=[2, 6])
        left_vector = encode_world_spec_features(left)
        right_vector = encode_world_spec_features(right)
        self.assertNotEqual(left_vector.tolist(), right_vector.tolist())

    def test_feature_extractor_v21_contract(self) -> None:
        from worldspace.surrogate.genome_features import FEATURE_DIM_V21

        spec = _canonical_spec(birth=[2, 6], survival=[4])
        vector = extract(spec)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "2.1")
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM_V21)
        self.assertEqual(vector.shape, (FEATURE_DIM_V21,))

    def test_v21_count_and_overlap_features(self) -> None:
        from worldspace.surrogate.genome_features import (
            encode_world_spec_features_v21,
            rule_count_overlap_features,
        )

        spec = _canonical_spec(birth=[2, 6], survival=[2, 4])
        birth_count, survival_count, overlap = rule_count_overlap_features(spec)
        self.assertAlmostEqual(birth_count, 2 / 9)
        self.assertAlmostEqual(survival_count, 2 / 9)
        self.assertAlmostEqual(overlap, 1 / 9)
        vector = encode_world_spec_features_v21(spec)
        self.assertAlmostEqual(float(vector[-3]), birth_count)
        self.assertAlmostEqual(float(vector[-2]), survival_count)
        self.assertAlmostEqual(float(vector[-1]), overlap)


if __name__ == "__main__":
    unittest.main()
