"""Exact-match copy audit vs documented sokoban-v0 example grids.

P2.4 is zero-shot: prompts must not contain these grids. The batch still
flags a child that equals a README example.
"""

from __future__ import annotations

import json

from worldspace.pcg.spec import PcgSpec

# Integer grid from pcg_benchmark/probs/sokoban/README.md at pin
# cd0f55b26c412a26e8797193e5417f5e651cf6cd (Microban, David W Skinner).
README_SOKOBAN_V0_GRIDS: tuple[tuple[tuple[int, ...], ...], ...] = (
    (
        (0, 1, 4, 0, 0),
        (0, 1, 1, 0, 0),
        (0, 4, 3, 2, 1),
        (0, 1, 1, 3, 1),
        (0, 1, 1, 0, 0),
    ),
)


def compact_grid_json(grid: tuple[tuple[int, ...], ...]) -> str:
    return json.dumps([list(row) for row in grid], separators=(",", ":"))


def readme_example_compact_jsons() -> tuple[str, ...]:
    return tuple(compact_grid_json(grid) for grid in README_SOKOBAN_V0_GRIDS)


def copy_readme_example(spec: PcgSpec) -> bool:
    if spec.problem_name != "sokoban-v0":
        return False
    return spec.grid in README_SOKOBAN_V0_GRIDS
