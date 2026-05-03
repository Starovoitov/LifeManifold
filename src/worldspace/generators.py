from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from .spec import WorldSpec


DEFAULT_CELL_TYPES = ["empty", "life", "food"]


def random_walk(value: float, scale: float, low: float = 0.0, high: float = 1.0) -> float:
    """Move a scalar value by Gaussian noise and clamp it to bounds."""
    moved = value + np.random.normal(0.0, scale)
    return float(np.clip(moved, low, high))


class WorldGenerator(ABC):
    @abstractmethod
    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate a list of world specifications."""
        raise NotImplementedError


class RandomWorldGenerator(WorldGenerator):
    def __init__(self, grid_size: int = 50, steps: int = 300):
        """Initialize random world generator defaults."""
        self.grid_size = grid_size
        self.steps = steps

    def _make_world(self, seed: int) -> WorldSpec:
        """Create one random world with bounded parameters."""
        birth = sorted(np.random.choice(np.arange(9), size=np.random.randint(1, 4), replace=False).tolist())
        survival = sorted(np.random.choice(np.arange(9), size=np.random.randint(2, 5), replace=False).tolist())
        return WorldSpec(
            birth=birth,
            survival=survival,
            noise=float(np.random.uniform(0.0, 0.08)),
            resource_regen=float(np.random.uniform(0.0, 0.2)),
            predation=float(np.random.uniform(0.0, 0.5)),
            cell_types=DEFAULT_CELL_TYPES.copy(),
            grid_size=self.grid_size,
            steps=self.steps,
            seed=seed,
        )

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate independent random worlds."""
        return [self._make_world(seed=i) for i in range(n_worlds)]


class RandomWalkWorldGenerator(WorldGenerator):
    def __init__(self, start_world: WorldSpec, scale: float = 0.02):
        """Initialize random-walk generator from a starting world."""
        self.start_world = start_world
        self.scale = scale

    def _mutate_rule_set(self, rule_set: list[int]) -> list[int]:
        """Randomly add or remove one rule value in 0..8."""
        values = set(rule_set)
        if np.random.rand() < 0.5 and len(values) > 1:
            values.discard(int(np.random.choice(list(values))))
        else:
            values.add(int(np.random.randint(0, 9)))
        return sorted(values)

    def _step(self, world: WorldSpec, seed: int) -> WorldSpec:
        """Apply one random-walk mutation step to a world."""
        return replace(
            world,
            birth=self._mutate_rule_set(world.birth),
            survival=self._mutate_rule_set(world.survival),
            noise=random_walk(world.noise, self.scale, 0.0, 0.2),
            resource_regen=random_walk(world.resource_regen, self.scale, 0.0, 0.5),
            predation=random_walk(world.predation, self.scale, 0.0, 1.0),
            seed=seed,
        )

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate a local trajectory through world space."""
        worlds = [self.start_world]
        current = self.start_world
        for seed in range(1, n_worlds):
            current = self._step(current, seed)
            worlds.append(current)
        return worlds


class MarkovWorldGenerator(WorldGenerator, ABC):
    def __init__(self, start_world: WorldSpec):
        """Initialize Markov generator from an initial world."""
        self.start_world = start_world

    @abstractmethod
    def transition(self, world: WorldSpec, state: int, seed: int) -> tuple[WorldSpec, int]:
        """Produce next world and hidden Markov state."""
        raise NotImplementedError

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate worlds by repeatedly applying Markov transitions."""
        worlds = [self.start_world]
        state = 0
        current = self.start_world
        for seed in range(1, n_worlds):
            current, state = self.transition(current, state, seed)
            worlds.append(current)
        return worlds


class TwoStateNoiseMarkovGenerator(MarkovWorldGenerator):
    """State 0 = calm, state 1 = chaotic."""

    def __init__(self, start_world: WorldSpec, p_stay_calm: float = 0.9, p_stay_chaos: float = 0.75):
        """Initialize two-state Markov dynamics for noise level."""
        super().__init__(start_world)
        self.p_stay_calm = p_stay_calm
        self.p_stay_chaos = p_stay_chaos

    def transition(self, world: WorldSpec, state: int, seed: int) -> tuple[WorldSpec, int]:
        """Switch calm/chaos state and adjust world noise accordingly."""
        rnd = np.random.rand()
        if state == 0:
            next_state = 0 if rnd < self.p_stay_calm else 1
        else:
            next_state = 1 if rnd < self.p_stay_chaos else 0
        noise = world.noise * (0.9 if next_state == 0 else 1.2)
        next_world = replace(world, noise=float(np.clip(noise, 0.0, 0.2)), seed=seed)
        return next_world, next_state


class RuleBiasMarkovGenerator(MarkovWorldGenerator):
    """State 0 biases survival, state 1 biases reproduction."""

    def transition(self, world: WorldSpec, state: int, seed: int) -> tuple[WorldSpec, int]:
        """Bias rule sets toward survival or reproduction regimes."""
        next_state = int(np.random.rand() >= 0.6)
        if next_state == 0:
            survival = sorted(set(world.survival + [2, 3]))
            birth = sorted(set(v for v in world.birth if v <= 4))
        else:
            birth = sorted(set(world.birth + [3, 4]))
            survival = sorted(set(v for v in world.survival if v >= 2))
        next_world = replace(world, birth=birth, survival=survival, seed=seed)
        return next_world, next_state


class NeuralWorldGenerator(WorldGenerator):
    """Future placeholder: NN-driven generator."""

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Placeholder for future neural generator implementation."""
        raise NotImplementedError("NeuralWorldGenerator is a placeholder for future work")


class LLMWorldGenerator(WorldGenerator):
    """Future placeholder: LLM-driven generator."""

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Placeholder for future LLM generator implementation."""
        raise NotImplementedError("LLMWorldGenerator is a placeholder for future work")
