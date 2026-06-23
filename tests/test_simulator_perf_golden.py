"""Golden metrics vectors for ``run_world`` (Epic A / L1 perf baseline).

Captured from the pre-optimization simulator. Future numpy micro-opts must keep
``metrics.as_vector()`` bit-identical on this set.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

import numpy as np

from worldspace.metrics import METRICS_VECTOR_DIM
from worldspace.simulator import run_world
from worldspace.specs.spec import WorldSpec

_BASE_SMOKE = WorldSpec(
    birth=[3],
    survival=[2, 3],
    noise=0.01,
    resource_regen=0.02,
    predation=0.3,
    cell_types=["life", "food"],
    grid_size=64,
    steps=200,
    seed=0,
)


@dataclass(frozen=True)
class _GoldenCase:
    name: str
    spec: WorldSpec
    early_extinction_step: int | None
    expected: np.ndarray


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=float)


_GOLDEN_CASES: tuple[_GoldenCase, ...] = (
    _GoldenCase(
        "smoke_seed0",
        _BASE_SMOKE,
        200,
        _vec(
            0.14619820546268988,
            0.0,
            1.7223836068147915,
            0.020852050781249985,
            0.8957131096702596,
            0.0546875,
            -0.7684623387889384,
            0.020751953125,
            0.04150390625,
            0.9010009765625,
            0.6271116325863588,
            0.6395348837209303,
        ),
    ),
    _GoldenCase(
        "smoke_seed42",
        replace(_BASE_SMOKE, seed=42),
        200,
        _vec(
            0.1473490999925783,
            0.0,
            1.7498616053776197,
            0.021059570312499977,
            0.9081837923594133,
            0.078125,
            -0.7472037911416691,
            0.0235595703125,
            0.046142578125,
            0.8990478515625,
            0.6307539953274589,
            0.6014150943396226,
        ),
    ),
    _GoldenCase(
        "deterministic_ca",
        replace(_BASE_SMOKE, noise=0.0, predation=0.0, seed=7),
        200,
        _vec(
            0.4527360937742345,
            0.6569060831937564,
            2.551988065122916,
            0.09493652343750003,
            0.9625205329914969,
            0.15625,
            0.29675807404172283,
            0.079345703125,
            0.136474609375,
            0.897216796875,
            0.7160262449473043,
            0.2963362068965517,
        ),
    ),
    _GoldenCase(
        "high_noise",
        replace(_BASE_SMOKE, noise=0.15, predation=0.0, seed=11),
        200,
        _vec(
            0.9615282890713492,
            0.9713669995894273,
            1.9879706352667539,
            0.38504638671874997,
            0.13792298507111195,
            0.8671875,
            0.9583847493564013,
            0.467529296875,
            0.81982421875,
            0.8819580078125,
            0.737357709345109,
            0.037091301665638496,
        ),
    ),
    _GoldenCase(
        "high_predation",
        replace(_BASE_SMOKE, noise=0.0, predation=0.8, seed=13),
        200,
        _vec(
            0.2263875321602871,
            0.0,
            1.1642633228840125,
            0.03658040364583333,
            0.6937349694843067,
            0.0078125,
            -0.7453549689778541,
            0.0,
            0.0,
            0.9239501953125,
            0.7409410586950573,
            0.0,
        ),
    ),
    _GoldenCase(
        "grid32",
        replace(_BASE_SMOKE, grid_size=32, seed=17),
        200,
        _vec(
            0.13416823798333496,
            0.0,
            1.7025210084033613,
            0.018715820312500003,
            0.8811117549903369,
            0.0625,
            -0.780145171304728,
            0.0185546875,
            0.0361328125,
            0.88671875,
            0.6193041086333295,
            0.625,
        ),
    ),
    _GoldenCase(
        "grid50_legacy",
        replace(_BASE_SMOKE, grid_size=50, steps=300, seed=19),
        None,
        _vec(
            0.12069611986740139,
            0.0,
            1.7027000099631364,
            0.016396000000000004,
            0.8882028087829971,
            0.0703125,
            -0.7884072994992288,
            0.02419999986886978,
            0.048,
            0.8936,
            0.6440982104762557,
            0.6733870967741935,
        ),
    ),
    _GoldenCase(
        "birth_zero",
        replace(_BASE_SMOKE, birth=[0], survival=[2, 3], seed=23),
        200,
        _vec(
            0.8267021598903652,
            0.9416471403683563,
            1.650706660707613,
            0.2599707031250001,
            0.7532732075219435,
            0.578125,
            0.9708906285600982,
            0.3248291015625,
            0.556884765625,
            0.8726806640625,
            0.8089734567531158,
            0.12296260786193672,
        ),
    ),
    _GoldenCase(
        "survival_eight",
        replace(_BASE_SMOKE, birth=[3], survival=[8], seed=29),
        200,
        _vec(
            0.08713634534458654,
            0.25419929276608133,
            1.406997205258255,
            0.010966796874999997,
            0.4452589935954726,
            0.0625,
            -0.711098131590918,
            0.0157470703125,
            0.03125,
            0.9036865234375,
            0.6096384113651883,
            0.6325757575757576,
        ),
    ),
    _GoldenCase(
        "low_regen",
        replace(_BASE_SMOKE, resource_regen=0.001, seed=31),
        200,
        _vec(
            0.14123487633943688,
            0.0,
            1.3602881657762669,
            0.01996337890625001,
            0.8991204302505504,
            0.0546875,
            -0.7811516701535851,
            0.016845703125,
            0.03369140625,
            0.9437255859375,
            0.2926130311143239,
            0.04642857142857143,
        ),
    ),
    _GoldenCase(
        "high_regen",
        replace(_BASE_SMOKE, resource_regen=0.25, seed=37),
        200,
        _vec(
            0.1362410487489374,
            0.0,
            2.1872018424082906,
            0.01907958984375,
            0.8883284940974138,
            0.0703125,
            -0.7639470838058524,
            0.021728515625,
            0.043212890625,
            0.955810546875,
            0.18207594055731566,
            0.9666666666666667,
        ),
    ),
    _GoldenCase(
        "conway_like",
        replace(
            _BASE_SMOKE,
            birth=[3],
            survival=[2, 3],
            noise=0.0,
            predation=0.0,
            seed=41,
        ),
        None,
        _vec(
            0.50961857830422,
            0.8275887404430493,
            2.5904291822836933,
            0.11324218750000001,
            0.9230703922284516,
            0.21875,
            0.5824383160560329,
            0.13671875,
            0.23876953125,
            0.8814697265625,
            0.8560887330136859,
            0.16564039408866996,
        ),
    ),
    _GoldenCase(
        "sparse_rules",
        replace(_BASE_SMOKE, birth=[5], survival=[4, 5, 6], seed=43),
        200,
        _vec(
            0.08273717688582942,
            0.8408453562949849,
            1.4401850627891606,
            0.010294189453124996,
            0.09663366027833917,
            0.0625,
            -0.3846230489163258,
            0.0216064453125,
            0.043212890625,
            0.899169921875,
            0.6298521409211886,
            0.65,
        ),
    ),
    _GoldenCase(
        "full_steps_no_early",
        replace(_BASE_SMOKE, seed=47),
        None,
        _vec(
            0.12204497337434957,
            0.0,
            1.650990990990991,
            0.016624755859375003,
            0.8526282490831568,
            0.0546875,
            -0.8028428375308756,
            0.018798828125,
            0.037109375,
            0.9002685546875,
            0.624938495307852,
            0.65625,
        ),
    ),
    _GoldenCase(
        "short_run",
        replace(_BASE_SMOKE, grid_size=16, steps=50, seed=53),
        200,
        _vec(
            0.5144600898415479,
            0.24595241872273355,
            1.7792207792207793,
            0.11487926136363635,
            0.8702924587789556,
            0.0078125,
            -0.21359481414290193,
            0.0,
            0.0,
            0.865234375,
            0.8293289299279132,
            0.0,
        ),
    ),
)


class TestSimulatorPerfGolden(unittest.TestCase):
    def test_golden_case_count(self) -> None:
        self.assertGreaterEqual(len(_GOLDEN_CASES), 10)

    def test_metrics_vector_dim(self) -> None:
        for case in _GOLDEN_CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.expected.shape, (METRICS_VECTOR_DIM,))

    def test_run_world_metrics_match_golden(self) -> None:
        for case in _GOLDEN_CASES:
            with self.subTest(case=case.name):
                result = run_world(
                    case.spec,
                    early_extinction_step=case.early_extinction_step,
                )
                np.testing.assert_array_equal(
                    result.metrics.as_vector(),
                    case.expected,
                )


if __name__ == "__main__":
    unittest.main()
