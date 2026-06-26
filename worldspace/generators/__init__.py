from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from collections.abc import Iterator
from dataclasses import replace
import http.client
import json
import os
import ssl
import time
from pathlib import Path
from typing import Any, cast
from urllib import error, request

import numpy as np
import yaml

from worldspace.prompt_files import default_llm_patch_system_content

from ..simulator import SimulationResult, run_world
from ..specs.spec import WorldSpec
from ..specs.world_param_bounds import (
    FLOAT_PARAM_BOUNDS,
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
    clip_scalar,
)
from .llm_config import (
    LLMGeneratorConfig,
    load_llm_config,
    load_llm_generator_yaml as load_llm_generator_yaml,
)
from .llm_patch import LLMPatchAdvisor

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

    @property
    def fallback_count(self) -> int:
        """Number of non-LLM recovery steps (e.g. random-walk) after failed LLM patches."""
        return int(getattr(self, "_fallback_count", 0))

    def _record_fallback(self) -> None:
        """Increment :attr:`fallback_count` (LLM / hybrid generators)."""
        self._fallback_count = self.fallback_count + 1

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
            noise=random_walk(world.noise, self.scale, rng, NOISE_MIN, NOISE_MAX),
            resource_regen=random_walk(
                world.resource_regen,
                self.scale,
                rng,
                RESOURCE_REGEN_MIN,
                RESOURCE_REGEN_MAX,
            ),
            predation=random_walk(
                world.predation, self.scale, rng, PREDATION_MIN, PREDATION_MAX
            ),
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
    """PyGAD-backed genetic algorithm over world parameters."""

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
        spec_path: str | Path | None = None,
    ):
        """Initialize GA settings for evolutionary search."""
        cfg = load_genetic_generator_yaml(spec_path or _DEFAULT_GENETIC_SPEC_PATH)
        pygad_cfg = cfg.get("pygad", {})
        self.grid_size = grid_size
        self.steps = steps
        self.population_size = max(2, int(population_size))
        self.elite_count = max(
            1,
            min(int(elite_count), self.population_size),
        )
        self.mutation_scale = float(mutation_scale)
        self.diversity_penalty = max(
            0.0,
            float(diversity_penalty),
        )
        self.max_stagnation = max(
            1,
            int(max_stagnation),
        )
        self.seed = seed
        self.parent_selection_type = str(
            pygad_cfg.get("parent_selection_type", "tournament")
        )
        self.k_tournament = int(pygad_cfg.get("k_tournament", 4))
        self.crossover_type = str(pygad_cfg.get("crossover_type", "uniform"))
        self.mutation_type = str(pygad_cfg.get("mutation_type", "adaptive"))
        self.suppress_warnings = bool(pygad_cfg.get("suppress_warnings", True))
        self.num_parents_mating = int(
            pygad_cfg.get(
                "num_parents_mating",
                max(2, min(self.population_size, self.elite_count * 2)),
            )
        )
        self.keep_elitism = self.elite_count
        self.mutation_probability = float(
            pygad_cfg.get(
                "mutation_probability",
                float(np.clip(0.12 + self.diversity_penalty * 0.2, 0.05, 0.6)),
            )
        )

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Generate best-per-generation worlds with a GA loop."""
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield top world from each generation from a PyGAD run."""
        if n_worlds <= 0:
            return
        import pygad

        gene_space: list = ([[0, 1] for _ in range(18)]) + [
            {"low": low, "high": high} for low, high in FLOAT_PARAM_BOUNDS
        ]
        gene_type: list[type] = ([int] * 18) + [float, float, float]

        initial_worlds = RandomWorldGenerator(
            grid_size=self.grid_size, steps=self.steps
        ).generate(self.population_size)
        initial_population = np.asarray(
            [self._encode_world(world) for world in initial_worlds], dtype=np.float64
        )
        fitness_cache: dict[tuple, float] = {}
        best_worlds: list[WorldSpec] = []

        def fitness_func(ga_instance: Any, solution: np.ndarray, __: int) -> float:
            generation_idx = max(
                0, int(getattr(ga_instance, "generations_completed", 0))
            )
            key = (
                generation_idx,
                tuple(np.round(solution.astype(np.float64), 6).tolist()),
            )
            cached = fitness_cache.get(key)
            if cached is not None:
                return cached
            world = self._decode_solution(
                solution, seed=self._solution_seed(solution, generation_idx)
            )
            fit = float(run_world(world).metrics.mo_eoc_indicator)
            fitness_cache[key] = fit
            return fit

        def on_generation(ga_instance: Any) -> None:
            sol, _, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness
            )
            gen_idx = len(best_worlds)
            best_worlds.append(
                self._decode_solution(sol, seed=self._solution_seed(sol, gen_idx))
            )

        ga = pygad.GA(
            num_generations=n_worlds,
            num_parents_mating=max(
                2, min(self.population_size, self.num_parents_mating)
            ),
            fitness_func=fitness_func,
            initial_population=initial_population,
            parent_selection_type=self.parent_selection_type,
            K_tournament=max(2, min(self.k_tournament, self.population_size)),
            keep_elitism=max(1, min(self.keep_elitism, self.population_size)),
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_probability=[
                float(np.clip(self.mutation_probability * 1.5, 0.001, 0.999)),
                float(np.clip(self.mutation_probability, 0.001, 0.999)),
            ],
            random_mutation_min_val=-abs(self.mutation_scale),
            random_mutation_max_val=abs(self.mutation_scale),
            gene_space=gene_space,
            gene_type=cast(Any, gene_type),
            random_seed=self.seed,
            save_best_solutions=False,
            on_generation=on_generation,
            suppress_warnings=self.suppress_warnings,
        )
        ga.run()
        yield from best_worlds[:n_worlds]

    def _encode_world(self, world: WorldSpec) -> np.ndarray:
        birth_mask = [1 if i in set(world.birth) else 0 for i in range(9)]
        survival_mask = [1 if i in set(world.survival) else 0 for i in range(9)]
        tail = [float(world.noise), float(world.resource_regen), float(world.predation)]
        return np.asarray(birth_mask + survival_mask + tail, dtype=np.float64)

    def _decode_solution(self, solution: np.ndarray, seed: int) -> WorldSpec:
        vals = np.asarray(solution, dtype=np.float64)
        birth_mask = np.rint(np.clip(vals[:9], 0.0, 1.0)).astype(np.int8)
        survival_mask = np.rint(np.clip(vals[9:18], 0.0, 1.0)).astype(np.int8)
        birth = [i for i in range(9) if int(birth_mask[i]) == 1]
        survival = [i for i in range(9) if int(survival_mask[i]) == 1]
        if not birth:
            birth = [int(np.argmax(vals[:9]))]
        if not survival:
            survival = [int(np.argmax(vals[9:18]))]
        return WorldSpec(
            birth=sorted(set(birth)),
            survival=sorted(set(survival)),
            noise=clip_scalar(vals[18], NOISE_MIN, NOISE_MAX),
            resource_regen=clip_scalar(
                vals[19], RESOURCE_REGEN_MIN, RESOURCE_REGEN_MAX
            ),
            predation=clip_scalar(vals[20], PREDATION_MIN, PREDATION_MAX),
            cell_types=DEFAULT_CELL_TYPES.copy(),
            grid_size=self.grid_size,
            steps=self.steps,
            seed=seed,
        )

    def _solution_seed(self, solution: np.ndarray, generation: int) -> int:
        vals = np.asarray(solution, dtype=np.float64)
        scaled = np.rint(np.clip(vals, -1e6, 1e6) * 1000.0).astype(np.int64)
        weights = np.arange(1, scaled.size + 1, dtype=np.int64)
        sig = int(np.abs(np.dot(scaled, weights)) % 1_000_000_000)
        return int(self.seed + generation * 100_003 + sig % 100_003)


def load_genetic_generator_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate genetic generator YAML."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"Genetic generator YAML not found: {src.resolve()}. "
            "Pass --generator-spec in CLI or place default genetic_world_generator.yaml in worldspace/specs/."
        )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    if raw.get("version") != 1:
        raise ValueError(f"{src}: expected version: 1")
    if "genetic" not in raw or "pygad" not in raw:
        raise ValueError(f"{src}: expected top-level keys 'genetic' and 'pygad'")
    return raw


class LLMWorldGenerator(WorldGenerator):
    """LLM-driven iterative local search (single lineage, scalar fitness)."""

    def __init__(
        self,
        grid_size: int = 50,
        steps: int = 300,
        seed: int = 0,
        spec_path: str | Path | None = None,
        *,
        config: LLMGeneratorConfig | None = None,
    ):
        self.config = config or load_llm_config(spec_path or _DEFAULT_LLM_SPEC_PATH)
        if self.config.global_search:
            raise ValueError(
                "global_search is enabled in the LLM spec; use LLMGlobalSearchWorldGenerator "
                "or make_llm_world_generator() instead of LLMWorldGenerator."
            )
        self.grid_size = grid_size
        self.steps = steps
        self.seed = seed
        self._advisor = LLMPatchAdvisor.from_config(
            self.config,
            call_llm_text=call_llm,
            call_llm_vision=call_llm_vision,
        )

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        if n_worlds <= 0:
            return
        current = self._initial_world()
        yield current
        for generation in range(1, n_worlds):
            result = run_world(current)
            score = float(result.metrics.mo_eoc_indicator)
            prompt = self._advisor.build_local_prompt(current, score)
            response = self._advisor.request_patch(prompt)
            suggested = _extract_world_patch_from_text(response)
            if suggested is None:
                current = self._fallback_step(current, generation)
            else:
                current = self._apply_world_patch(current, suggested, generation)
            yield current

    def _initial_world(self) -> WorldSpec:
        if self.config.initial_generator == "random_walk":
            base = RandomWorldGenerator(
                grid_size=self.grid_size,
                steps=self.steps,
            ).generate(1)[0]
            return RandomWalkWorldGenerator(
                start_world=base, scale=self.config.fallback_scale
            ).generate(1)[0]
        return RandomWorldGenerator(
            grid_size=self.grid_size, steps=self.steps
        ).generate(1)[0]

    def _fallback_step(self, world: WorldSpec, generation: int) -> WorldSpec:
        self._record_fallback()
        walker = RandomWalkWorldGenerator(
            start_world=world, scale=self.config.fallback_scale
        )
        nxt = walker.generate(2)[-1]
        return replace(nxt, seed=self.seed + generation)

    def _apply_world_patch(
        self, world: WorldSpec, patch: dict[str, Any], generation: int
    ) -> WorldSpec:
        return _apply_world_patch(world, patch, seed=self.seed + generation)


class LLMGlobalSearchWorldGenerator(LLMWorldGenerator):
    """LLM iterative search with vision/text simulation description before each patch."""

    def __init__(
        self,
        grid_size: int = 50,
        steps: int = 300,
        seed: int = 0,
        spec_path: str | Path | None = None,
        *,
        config: LLMGeneratorConfig | None = None,
    ):
        loaded = config or load_llm_config(spec_path or _DEFAULT_LLM_SPEC_PATH)
        if not loaded.global_search:
            raise ValueError(
                "LLMGlobalSearchWorldGenerator requires global_search: true in the LLM spec."
            )
        self.config = loaded
        self.grid_size = grid_size
        self.steps = steps
        self.seed = seed
        self._advisor = LLMPatchAdvisor.from_config(
            self.config,
            call_llm_text=call_llm,
            call_llm_vision=call_llm_vision,
        )

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        if n_worlds <= 0:
            return
        current = self._initial_world()
        yield current
        for generation in range(1, n_worlds):
            result = run_world(current)
            description = self._advisor.describe(result)
            prompt = self._advisor.build_global_prompt(current, result, description)
            response = self._advisor.request_patch(prompt)
            suggested = _extract_world_patch_from_text(response)
            if suggested is None:
                current = self._fallback_step(current, generation)
            else:
                current = self._apply_world_patch(current, suggested, generation)
            yield current


def make_llm_world_generator(
    grid_size: int = 50,
    steps: int = 300,
    seed: int = 0,
    spec_path: str | Path | None = None,
) -> WorldGenerator:
    """Instantiate local-search or global-search LLM generator per YAML ``global_search``."""
    config = load_llm_config(spec_path or _DEFAULT_LLM_SPEC_PATH)
    if config.global_search:
        return LLMGlobalSearchWorldGenerator(
            grid_size=grid_size,
            steps=steps,
            seed=seed,
            config=config,
        )
    return LLMWorldGenerator(
        grid_size=grid_size,
        steps=steps,
        seed=seed,
        config=config,
    )


class HybridGALlmWorldGenerator(WorldGenerator):
    """Population-based hybrid: GA random mutation + LLM-guided mutation."""

    def __init__(
        self,
        grid_size: int = 50,
        steps: int = 300,
        seed: int = 0,
        spec_path: str | Path | None = None,
    ):
        cfg = load_hybrid_generator_yaml(spec_path or _DEFAULT_HYBRID_SPEC_PATH)
        evo = cfg["evolution"]
        llm_cfg = cfg["llm"]
        self.grid_size = grid_size
        self.steps = steps
        self.seed = seed
        self.population_size = int(evo["population_size"])
        self.select_top_k = int(evo["select_top_k"])
        self.select_random_k = int(evo["select_random_k"])
        self.elite_keep = int(evo["elite_keep"])
        self.random_mutations = int(evo["random_mutations"])
        self.llm_mutations = int(evo["llm_mutations"])
        self.mutation_scale = float(evo.get("mutation_scale", 0.02))
        self.llm_top_fraction = float(evo.get("llm_top_fraction", 0.2))

        llm_config = LLMGeneratorConfig.from_llm_dict(llm_cfg)
        self._llm_advisor = LLMPatchAdvisor.from_config(
            llm_config,
            call_llm_text=call_llm,
            call_llm_vision=call_llm_vision,
        )

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        return list(self.iter_worlds(n_worlds))

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        if n_worlds <= 0:
            return
        rng = np.random.default_rng(self.seed)
        base_gen = RandomWorldGenerator(self.grid_size, self.steps)
        population = [
            base_gen._make_world(seed=self.seed + i)
            for i in range(self.population_size)
        ]
        for generation in range(n_worlds):
            scored = self._score_population(population)
            best_world = replace(scored[0][0], seed=self.seed + generation)
            yield best_world

            selected = self._select_with_diversity(scored, rng)
            llm_pool_size = max(1, int(np.ceil(len(selected) * self.llm_top_fraction)))
            llm_pool = selected[:llm_pool_size]

            next_population: list[WorldSpec] = [
                replace(w, seed=self.seed + generation * 1000 + i)
                for i, (w, _) in enumerate(scored[: self.elite_keep])
            ]
            seed_counter = self.seed + generation * 10_000

            for _ in range(self.random_mutations):
                parent = selected[int(rng.integers(0, len(selected)))][0]
                next_population.append(
                    self._random_mutate(parent, rng, seed=seed_counter)
                )
                seed_counter += 1

            for _ in range(self.llm_mutations):
                parent, parent_result = llm_pool[int(rng.integers(0, len(llm_pool)))]
                child = self._llm_mutate(
                    parent, parent_result, generation, seed_counter, rng
                )
                next_population.append(child)
                seed_counter += 1

            while len(next_population) < self.population_size:
                parent = selected[int(rng.integers(0, len(selected)))][0]
                next_population.append(
                    self._random_mutate(parent, rng, seed=seed_counter)
                )
                seed_counter += 1
            population = next_population[: self.population_size]

    def _score_population(
        self, population: list[WorldSpec]
    ) -> list[tuple[WorldSpec, SimulationResult]]:
        scored: list[tuple[WorldSpec, SimulationResult]] = []
        for world in population:
            scored.append((world, run_world(world)))
        scored.sort(key=lambda x: float(x[1].metrics.mo_eoc_indicator), reverse=True)
        return scored

    def _select_with_diversity(
        self,
        scored: list[tuple[WorldSpec, SimulationResult]],
        rng: np.random.Generator,
    ) -> list[tuple[WorldSpec, SimulationResult]]:
        top = scored[: min(self.select_top_k, len(scored))]
        rest = scored[len(top) :]
        if not rest:
            return top
        k_rand = min(self.select_random_k, len(rest))
        rand_idx = rng.choice(np.arange(len(rest)), size=k_rand, replace=False).tolist()
        sampled = [rest[int(i)] for i in rand_idx]
        return top + sampled

    def _random_mutate(
        self, world: WorldSpec, rng: np.random.Generator, seed: int
    ) -> WorldSpec:
        return replace(
            world,
            birth=_mutate_rule_set(world.birth, rng),
            survival=_mutate_rule_set(world.survival, rng),
            noise=random_walk(
                world.noise, self.mutation_scale, rng, NOISE_MIN, NOISE_MAX
            ),
            resource_regen=random_walk(
                world.resource_regen,
                self.mutation_scale,
                rng,
                RESOURCE_REGEN_MIN,
                RESOURCE_REGEN_MAX,
            ),
            predation=random_walk(
                world.predation, self.mutation_scale, rng, PREDATION_MIN, PREDATION_MAX
            ),
            seed=seed,
        )

    def _llm_mutate(
        self,
        parent: WorldSpec,
        parent_result: SimulationResult,
        generation: int,
        seed: int,
        rng: np.random.Generator,
    ) -> WorldSpec:
        description = ""
        if self._llm_advisor.config.global_search:
            description = self._llm_advisor.describe(parent_result)
        prompt = self._llm_advisor.build_hybrid_prompt(
            parent, parent_result, description=description
        )
        try:
            response = self._llm_advisor.request_patch(prompt)
            patch = _extract_world_patch_from_text(response)
        except RuntimeError:
            patch = None
        if patch is None:
            self._record_fallback()
            return self._random_mutate(parent, rng, seed=seed)
        return _apply_world_patch(parent, patch, seed=seed)


def call_llm(
    *,
    mode: str,
    provider_name: str,
    providers: dict[str, Any],
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 350,
    system_content: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint and return message content."""
    effective_system = (
        default_llm_patch_system_content() if system_content is None else system_content
    )
    return call_llm_messages(
        mode=mode,
        provider_name=provider_name,
        providers=providers,
        messages=[
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def call_llm_vision(
    *,
    mode: str,
    provider_name: str,
    providers: dict[str, Any],
    system_content: str,
    user_text: str,
    image_png_bytes: bytes,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> str:
    """Multimodal chat completion: text prompt plus one PNG frame."""
    b64 = base64.standard_b64encode(image_png_bytes).decode("ascii")
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
    return call_llm_messages(
        mode=mode,
        provider_name=provider_name,
        providers=providers,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


_LLM_HTTP_TIMEOUT_SECONDS = 45
_LLM_HTTP_MAX_ATTEMPTS = 3
_LLM_HTTP_RETRY_BACKOFF_SECONDS = 2.0


def _retryable_llm_http_status(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


def _fetch_llm_response_body(req: request.Request, *, api_base: str) -> str:
    """POST once with up to three attempts on transient network/API failures."""
    last_error: BaseException | None = None
    for attempt in range(1, _LLM_HTTP_MAX_ATTEMPTS + 1):
        try:
            with request.urlopen(req, timeout=_LLM_HTTP_TIMEOUT_SECONDS) as resp:
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            if (
                _retryable_llm_http_status(exc.code)
                and attempt < _LLM_HTTP_MAX_ATTEMPTS
            ):
                last_error = exc
                time.sleep(_LLM_HTTP_RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(f"LLM HTTP error {exc.code}: {exc.reason}") from exc
        except (
            error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
        ) as exc:
            if attempt < _LLM_HTTP_MAX_ATTEMPTS:
                last_error = exc
                time.sleep(_LLM_HTTP_RETRY_BACKOFF_SECONDS * attempt)
                continue
            if isinstance(exc, error.URLError):
                r = exc.reason
                hint = _llm_url_error_hint(r, api_base)
                raise RuntimeError(f"LLM request failed: {r}.{hint}") from exc
            raise RuntimeError(f"LLM request failed: {exc}") from exc
    msg = f"LLM request failed after {_LLM_HTTP_MAX_ATTEMPTS} attempts"
    if last_error is not None:
        raise RuntimeError(msg) from last_error
    raise RuntimeError(msg)


def call_llm_messages(
    *,
    mode: str,
    provider_name: str,
    providers: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 350,
) -> str:
    """POST chat completions with a pre-built ``messages`` list."""
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        raise ValueError(f"Unknown provider in llm config: {provider_name!r}")
    api_base = str(provider.get("api_base") or "").strip()
    model = str(provider.get("model") or "").strip()
    if not api_base or not model:
        raise ValueError(f"Provider {provider_name!r} must define api_base and model")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(api_base, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    if mode == "remote":
        key_env = str(provider.get("api_key_env") or "").strip()
        if not key_env:
            raise ValueError(
                f"Remote provider {provider_name!r} requires api_key_env in config"
            )
        token = os.getenv(key_env, "").strip()
        if not token:
            raise RuntimeError(
                f"Environment variable {key_env!r} is required for remote provider {provider_name!r}"
            )
        req.add_header("Authorization", f"Bearer {token}")
    elif mode != "local":
        raise ValueError("llm.mode must be either 'local' or 'remote'")

    raw = _fetch_llm_response_body(req, api_base=api_base)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response is not valid JSON") from exc
    choices = data.get("choices")
    if not choices:
        raise RuntimeError("LLM response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response missing message.content")
    return content


def load_hybrid_generator_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate hybrid GA+LLM generator YAML."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"Hybrid generator YAML not found: {src.resolve()}. "
            "Pass --generator-spec in CLI or place default hybrid_world_generator.yaml in worldspace/specs/."
        )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    if raw.get("version") != 1:
        raise ValueError(f"{src}: expected version: 1")
    if "evolution" not in raw or "llm" not in raw:
        raise ValueError(f"{src}: expected top-level keys 'evolution' and 'llm'")
    evo = raw["evolution"]
    llm = raw["llm"]
    if not isinstance(evo, dict):
        raise ValueError(f"{src}: evolution must be a mapping")
    if not isinstance(llm, dict):
        raise ValueError(f"{src}: llm must be a mapping")
    for key in (
        "population_size",
        "select_top_k",
        "select_random_k",
        "elite_keep",
        "random_mutations",
        "llm_mutations",
    ):
        if key not in evo:
            raise ValueError(f"{src}: evolution.{key} is required")
    for key in ("mode", "active_provider", "providers"):
        if key not in llm:
            raise ValueError(f"{src}: llm.{key} is required")
    return raw


def __getattr__(name: str) -> Any:
    """Lazy import so PyTorch is only pulled in when accessing ``NeuralWorldGenerator``."""
    if name == "NeuralWorldGenerator":
        from .neural_world import NeuralWorldGenerator as _NWG

        return _NWG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- module-private defaults and helpers ---

_DEFAULT_GENETIC_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "specs" / "genetic_world_generator.yaml"
)
_DEFAULT_LLM_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "specs" / "llm_world_generator.yaml"
)
_DEFAULT_HYBRID_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "specs" / "hybrid_world_generator.yaml"
)


def _llm_url_error_hint(reason: object, api_base: str) -> str:
    text = str(reason)
    if "Connection refused" in text or "Errno 111" in text:
        return (
            f" Hint: no server accepted TCP to {api_base!r} (connection refused). "
            "If this is a local OpenAI-compatible endpoint (Ollama/LM Studio), start it "
            "or change ``api_base`` / use ``--generator-spec`` with a reachable URL."
        )
    if isinstance(reason, ssl.SSLError) or "SSL" in text or "UNEXPECTED_EOF" in text:
        return (
            " TLS hint: urllib uses system trust and env proxies; set HTTPS_PROXY/HTTP_PROXY "
            "if required, add your intercept CA to SSL_CERT_FILE (urllib does not read "
            "REQUESTS_CA_BUNDLE), or try another network/VPN. DashScope intl endpoint can be "
            "blocked or reset by some paths."
        )
    return ""


def _extract_world_patch_from_text(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output and return it as dict."""
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        chunks = stripped.split("```")
        candidates.extend(chunk.strip() for chunk in chunks if chunk.strip())
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_rule_list(value: Any, fallback: list[int]) -> list[int]:
    if not isinstance(value, list):
        return list(fallback)
    vals: set[int] = set()
    for item in value:
        try:
            v = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= v <= 8:
            vals.add(v)
    if not vals:
        return list(fallback)
    return sorted(vals)


def _clip_float(value: Any, fallback: float, low: float, high: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return float(np.clip(f, low, high))


def _mutate_rule_set(rule_set: list[int], rng: np.random.Generator) -> list[int]:
    values = set(rule_set)
    if rng.random() < 0.5 and len(values) > 1:
        values.discard(int(rng.choice(list(values))))
    else:
        values.add(int(rng.integers(0, 9)))
    return sorted(values)


def _apply_world_patch(world: WorldSpec, patch: dict[str, Any], seed: int) -> WorldSpec:
    birth = _normalize_rule_list(patch.get("birth"), world.birth)
    survival = _normalize_rule_list(patch.get("survival"), world.survival)
    noise = _clip_float(patch.get("noise"), world.noise, NOISE_MIN, NOISE_MAX)
    resource_regen = _clip_float(
        patch.get("resource_regen"),
        world.resource_regen,
        RESOURCE_REGEN_MIN,
        RESOURCE_REGEN_MAX,
    )
    predation = _clip_float(
        patch.get("predation"), world.predation, PREDATION_MIN, PREDATION_MAX
    )
    return replace(
        world,
        birth=birth,
        survival=survival,
        noise=noise,
        resource_regen=resource_regen,
        predation=predation,
        seed=seed,
    )
