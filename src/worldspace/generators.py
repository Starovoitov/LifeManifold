from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np

from .spec import WorldSpec

DEFAULT_CELL_TYPES = ["empty", "life", "food"]


def random_walk(
    value: float,
    scale: float,
    rng: np.random.Generator,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Move a scalar value by Gaussian noise and clamp it to bounds."""
    moved = value + rng.normal(0.0, scale)
    return float(np.clip(moved, low, high))


class WorldGenerator(ABC):
    """Abstract sequence of :class:`WorldSpec` for simulation batches or streaming pipelines."""

    @abstractmethod
    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate a list of world specifications."""
        raise NotImplementedError

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield worlds one at a time (same order as ``generate``; no full list kept)."""
        yield from self.generate(n_worlds)


class RandomWorldGenerator(WorldGenerator):
    def __init__(self, grid_size: int = 50, steps: int = 300):
        """Initialize random world generator defaults."""
        self.grid_size = grid_size
        self.steps = steps

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate independent random worlds."""
        return [self._make_world(seed=i) for i in range(n_worlds)]

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield worlds without allocating the full list."""
        for i in range(n_worlds):
            yield self._make_world(seed=i)

    def _make_world(self, seed: int) -> WorldSpec:
        """Create one random world with bounded parameters (reproducible for ``seed``)."""
        rng = np.random.default_rng(seed)
        birth = sorted(
            rng.choice(np.arange(9), size=rng.integers(1, 4), replace=False).tolist()
        )
        survival = sorted(
            rng.choice(np.arange(9), size=rng.integers(2, 5), replace=False).tolist()
        )
        return WorldSpec(
            birth=birth,
            survival=survival,
            noise=float(rng.uniform(0.0, 0.08)),
            resource_regen=float(rng.uniform(0.0, 0.2)),
            predation=float(rng.uniform(0.0, 0.5)),
            cell_types=DEFAULT_CELL_TYPES.copy(),
            grid_size=self.grid_size,
            steps=self.steps,
            seed=seed,
        )


class RandomWalkWorldGenerator(WorldGenerator):
    def __init__(self, start_world: WorldSpec, scale: float = 0.02):
        """Initialize random-walk generator from a starting world."""
        self.start_world = start_world
        self.scale = scale

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate a local trajectory through world space."""
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Walk from ``start_world`` without storing the full trajectory."""
        current = self.start_world
        yield current
        for seed in range(1, n_worlds):
            current = self._step(current, seed)
            yield current

    def _step(self, world: WorldSpec, seed: int) -> WorldSpec:
        """Apply one random-walk mutation step to a world."""
        rng = np.random.default_rng(seed)
        return replace(
            world,
            birth=self._mutate_rule_set(world.birth, rng),
            survival=self._mutate_rule_set(world.survival, rng),
            noise=random_walk(world.noise, self.scale, rng, 0.0, 0.2),
            resource_regen=random_walk(world.resource_regen, self.scale, rng, 0.0, 0.5),
            predation=random_walk(world.predation, self.scale, rng, 0.0, 1.0),
            seed=seed,
        )

    def _mutate_rule_set(
        self, rule_set: list[int], rng: np.random.Generator
    ) -> list[int]:
        """Randomly add or remove one rule value in 0..8."""
        values = set(rule_set)
        if rng.random() < 0.5 and len(values) > 1:
            values.discard(int(rng.choice(list(values))))
        else:
            values.add(int(rng.integers(0, 9)))
        return sorted(values)


class MarkovWorldGenerator(WorldGenerator, ABC):
    def __init__(self, start_world: WorldSpec):
        """Initialize Markov generator from an initial world."""
        self.start_world = start_world

    @abstractmethod
    def transition(
        self, world: WorldSpec, state: int, seed: int
    ) -> tuple[WorldSpec, int]:
        """Produce next world and hidden Markov state."""
        raise NotImplementedError

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate worlds by repeatedly applying Markov transitions."""
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield Markov chain worlds without a Python list of all specs."""
        yield self.start_world
        state = 0
        current = self.start_world
        for seed in range(1, n_worlds):
            current, state = self.transition(current, state, seed)
            yield current


class TwoStateNoiseMarkovGenerator(MarkovWorldGenerator):
    """State 0 = calm, state 1 = chaotic."""

    def __init__(
        self,
        start_world: WorldSpec,
        p_stay_calm: float = 0.9,
        p_stay_chaos: float = 0.75,
    ):
        """Initialize two-state Markov dynamics for noise level."""
        super().__init__(start_world)
        self.p_stay_calm = p_stay_calm
        self.p_stay_chaos = p_stay_chaos

    def transition(
        self, world: WorldSpec, state: int, seed: int
    ) -> tuple[WorldSpec, int]:
        """Switch calm/chaos state and adjust world noise accordingly."""
        rng = np.random.default_rng(seed)
        rnd = rng.random()
        if state == 0:
            next_state = 0 if rnd < self.p_stay_calm else 1
        else:
            next_state = 1 if rnd < self.p_stay_chaos else 0
        noise = world.noise * (0.9 if next_state == 0 else 1.2)
        next_world = replace(world, noise=float(np.clip(noise, 0.0, 0.2)), seed=seed)
        return next_world, next_state


class RuleBiasMarkovGenerator(MarkovWorldGenerator):
    """State 0 biases survival, state 1 biases reproduction."""

    def transition(
        self, world: WorldSpec, state: int, seed: int
    ) -> tuple[WorldSpec, int]:
        """Bias rule sets toward survival or reproduction regimes."""
        rng = np.random.default_rng(seed)
        next_state = int(rng.random() >= 0.6)
        if next_state == 0:
            survival = sorted(set(world.survival + [2, 3]))
            birth = sorted(set(v for v in world.birth if v <= 4))
        else:
            birth = sorted(set(world.birth + [3, 4]))
            survival = sorted(set(v for v in world.survival if v >= 2))
        next_world = replace(world, birth=birth, survival=survival, seed=seed)
        return next_world, next_state


class LLMWorldGenerator(WorldGenerator):
    """Future placeholder: LLM-driven generator."""

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Placeholder for future LLM generator implementation."""
        raise NotImplementedError("LLMWorldGenerator is a placeholder for future work")


def __getattr__(name: str) -> Any:
    """Lazy import so PyTorch is only pulled in when accessing ``NeuralWorldGenerator``."""
    if name == "NeuralWorldGenerator":
        from .neural_world import NeuralWorldGenerator as _NWG

        return _NWG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
