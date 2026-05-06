from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np

from .simulator import run_world
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


class GeneticWorldGenerator(WorldGenerator):
    """Basic genetic algorithm over world parameters."""

    def __init__(
        self,
        grid_size: int = 50,
        steps: int = 300,
        population_size: int = 12,
        elite_count: int = 3,
        mutation_scale: float = 0.02,
        diversity_penalty: float = 0.15,
        max_stagnation: int = 3,
        seed: int = 0,
    ):
        """Initialize GA settings for evolutionary search."""
        self.grid_size = grid_size
        self.steps = steps
        self.population_size = max(2, population_size)
        self.elite_count = max(1, min(elite_count, self.population_size))
        self.mutation_scale = mutation_scale
        self.diversity_penalty = max(0.0, diversity_penalty)
        self.max_stagnation = max(1, max_stagnation)
        self.seed = seed

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate best-per-generation worlds with a GA loop."""
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield top world from each generation."""
        if n_worlds <= 0:
            return
        rng = np.random.default_rng(self.seed)
        population = self._initial_population(rng)
        seed_counter = self.population_size + self.seed + 1
        last_best_signature: tuple | None = None
        stagnation = 0
        for _ in range(n_worlds):
            fitness = np.asarray([self._fitness(w) for w in population], dtype=np.float64)
            signatures = [self._genome_signature(w) for w in population]
            duplicate_counts = Counter(signatures)
            adjusted_fitness = np.asarray(
                [
                    fit - self.diversity_penalty * (duplicate_counts[sig] - 1)
                    for fit, sig in zip(fitness, signatures, strict=True)
                ],
                dtype=np.float64,
            )
            order = np.argsort(adjusted_fitness)[::-1]
            best = population[int(order[0])]
            yield best

            best_signature = self._genome_signature(best)
            if best_signature == last_best_signature:
                stagnation += 1
            else:
                stagnation = 0
            last_best_signature = best_signature

            elites = self._unique_elites(population, order)
            population = elites.copy()
            while len(population) < self.population_size:
                parent = self._sample_parent(elites, rng)
                mutation_scale = self.mutation_scale * (1.0 + 0.35 * stagnation)
                child = self._mutate(
                    parent, seed=seed_counter, rng=rng, mutation_scale=mutation_scale
                )
                seed_counter += 1
                if stagnation >= self.max_stagnation:
                    child = self._mutate(
                        child,
                        seed=seed_counter,
                        rng=rng,
                        mutation_scale=mutation_scale * 1.5,
                    )
                    seed_counter += 1
                population.append(child)

    def _initial_population(self, rng: np.random.Generator) -> list[WorldSpec]:
        base = RandomWorldGenerator(grid_size=self.grid_size, steps=self.steps)
        worlds = base.generate(self.population_size)
        seeded: list[WorldSpec] = []
        for i, world in enumerate(worlds):
            seeded.append(replace(world, seed=self.seed + int(rng.integers(1, 1_000_000)) + i))
        return seeded

    def _fitness(self, world: WorldSpec) -> float:
        return float(run_world(world).metrics.interestingness)

    def _sample_parent(
        self,
        elites: list[WorldSpec],
        rng: np.random.Generator,
    ) -> WorldSpec:
        elite_fitness = np.asarray([self._fitness(w) for w in elites], dtype=np.float64)
        min_fit = float(elite_fitness.min(initial=0.0))
        weights = elite_fitness - min_fit + 1e-6
        probs = weights / weights.sum()
        idx = int(rng.choice(np.arange(len(elites)), p=probs))
        return elites[idx]

    def _mutate(
        self,
        world: WorldSpec,
        seed: int,
        rng: np.random.Generator,
        mutation_scale: float | None = None,
    ) -> WorldSpec:
        scale = self.mutation_scale if mutation_scale is None else mutation_scale
        return replace(
            world,
            birth=self._mutate_rule_set(world.birth, rng),
            survival=self._mutate_rule_set(world.survival, rng),
            noise=random_walk(world.noise, scale, rng, 0.0, 0.2),
            resource_regen=random_walk(world.resource_regen, scale, rng, 0.0, 0.5),
            predation=random_walk(world.predation, scale, rng, 0.0, 1.0),
            seed=seed,
        )

    def _mutate_rule_set(
        self, rule_set: list[int], rng: np.random.Generator
    ) -> list[int]:
        values = set(rule_set)
        if rng.random() < 0.5 and len(values) > 1:
            values.discard(int(rng.choice(list(values))))
        else:
            values.add(int(rng.integers(0, 9)))
        return sorted(values)

    def _unique_elites(self, population: list[WorldSpec], order: np.ndarray) -> list[WorldSpec]:
        unique: list[WorldSpec] = []
        seen: set[tuple] = set()
        for idx in order:
            candidate = population[int(idx)]
            sig = self._genome_signature(candidate)
            if sig in seen:
                continue
            unique.append(candidate)
            seen.add(sig)
            if len(unique) >= self.elite_count:
                break
        if not unique:
            unique = [population[int(order[0])]]
        return unique

    def _genome_signature(self, world: WorldSpec) -> tuple:
        return (
            tuple(world.birth),
            tuple(world.survival),
            round(world.noise, 4),
            round(world.resource_regen, 4),
            round(world.predation, 4),
        )


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
