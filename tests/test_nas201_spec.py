"""NAS-Bench-201 genotype parsing, hashing, and one-edge mutation."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from worldspace.nas201.emitters import mutate_one_edge, random_spec
from worldspace.nas201.spec import (
    Nas201Spec,
    hamming_ops,
    try_parse_arch_str,
    try_parse_ops_payload,
)

EXAMPLE = (
    "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|"
    "+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|"
)


class TestNas201Spec(unittest.TestCase):
    def test_official_string_roundtrip(self) -> None:
        spec = Nas201Spec.from_arch_str(EXAMPLE)
        self.assertEqual(spec.arch_str, EXAMPLE)
        self.assertEqual(Nas201Spec.from_arch_str(spec.arch_str), spec)
        self.assertEqual(len(spec.candidate_hash()), 16)
        self.assertEqual(len(spec.genotype_sha256()), 64)

    def test_ops_json_roundtrip(self) -> None:
        spec = Nas201Spec.from_arch_str(EXAMPLE)
        again = Nas201Spec.from_ops_json(spec.to_json_dict())
        self.assertEqual(again, spec)

    def test_hash_is_arch_string_not_json(self) -> None:
        spec = Nas201Spec.from_arch_str(EXAMPLE)
        self.assertNotEqual(spec.genotype_sha256(), spec.candidate_hash())
        self.assertTrue(spec.genotype_sha256().startswith(spec.candidate_hash()))

    def test_unknown_op_is_not_structurally_valid(self) -> None:
        with self.assertRaises(ValidationError):
            Nas201Spec(ops=("conv_5x5",) * 6)
        self.assertIsNone(
            try_parse_arch_str("|conv_5x5~0|+|none~0|none~1|+|none~0|none~1|none~2|")
        )
        self.assertIsNone(try_parse_ops_payload({"ops": ["none"] * 5}))
        self.assertIsNone(try_parse_ops_payload("not-json"))

    def test_garbage_string_does_not_parse(self) -> None:
        self.assertIsNone(try_parse_arch_str("resnet18"))
        self.assertIsNone(try_parse_arch_str(""))

    def test_random_specs_are_valid_and_hashes_unique_in_sample(self) -> None:
        rng = np.random.default_rng(7)
        specs = [random_spec(rng) for _ in range(80)]
        hashes = {item.genotype_sha256() for item in specs}
        self.assertEqual(len(hashes), len({item.arch_str for item in specs}))

    def test_mutation_changes_exactly_one_edge(self) -> None:
        rng = np.random.default_rng(11)
        parent = random_spec(rng)
        for _ in range(40):
            child = mutate_one_edge(parent, rng)
            self.assertEqual(hamming_ops(parent, child), 1)
            parent = child


if __name__ == "__main__":
    unittest.main()
