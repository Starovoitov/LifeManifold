"""Capability declarations for current native domain runners."""

from __future__ import annotations

from worldspace.attribution.manifest import AdapterCapabilities

ADAPTER_VERSION = "0.1"

_BUDGET_AXES = (
    "proposal",
    "valid_proposal",
    "evaluation",
    "llm_call_attempted",
    "llm_call_completed",
    "prompt_token",
    "completion_token",
    "token",
    "evaluator_wall_time",
    "llm_latency",
    "wall_time",
    "monetary",
)


def ca_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="ca-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="ca",
        initialization_kinds=("empty", "loaded_archive", "generated_floor"),
        selectors=(
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ),
        generators=("random", "genetic", "llm"),
        prompt_channels=(
            "not_applicable",
            "constant",
            "live",
            "shuffled",
            "ablated",
        ),
        repair_fallback_kinds=(
            "identity",
            "random_walk_fallback",
            "child_rewrite",
        ),
        gate_modes=("off", "shadow", "filter"),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static",),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid", "cvt"),
        supports_full_proposal_log=True,
        supports_warm_start=True,
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=1.0,
        empty_cell_fitness=0.0,
    )


def maze_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="maze-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="maze",
        initialization_kinds=("empty",),
        selectors=(
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ),
        generators=("random", "genetic", "llm"),
        prompt_channels=("not_applicable", "constant", "live"),
        repair_fallback_kinds=(
            "identity",
            "solvability_repair",
            "genetic_fallback",
        ),
        gate_modes=("off", "shadow", "filter"),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static",),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid",),
        supports_full_proposal_log=True,
        supports_warm_start=False,
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=1.0,
        empty_cell_fitness=0.0,
    )


def dungeon_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="dungeon-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="dungeon",
        initialization_kinds=("empty",),
        selectors=("uniform_frontier",),
        generators=("random", "genetic", "llm"),
        prompt_channels=("not_applicable", "constant", "live"),
        repair_fallback_kinds=(
            "identity",
            "solvability_repair",
            "genetic_fallback",
        ),
        gate_modes=("off", "shadow", "filter"),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static",),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid",),
        supports_full_proposal_log=False,
        supports_warm_start=False,
        stochastic_evaluation=True,
        native_fitness_min=0.0,
        native_fitness_max=1.0,
        empty_cell_fitness=0.0,
    )


def sphere_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_id="sphere-rq1-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="sphere",
        initialization_kinds=("empty",),
        selectors=(
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ),
        generators=("random", "genetic", "llm"),
        prompt_channels=("not_applicable", "constant", "live"),
        repair_fallback_kinds=("identity", "clip_and_genetic_fallback"),
        gate_modes=("off",),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static",),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid",),
        supports_full_proposal_log=False,
        supports_warm_start=False,
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=100.0,
        empty_cell_fitness=0.0,
    )


def nas201_capabilities() -> AdapterCapabilities:
    """NAS-Bench-201 public task (lookup; identity repair)."""
    return AdapterCapabilities(
        adapter_id="nas201-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="nas201",
        initialization_kinds=("generated_floor", "loaded_archive"),
        selectors=(
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ),
        generators=("random", "genetic", "llm"),
        prompt_channels=("not_applicable", "constant", "live"),
        repair_fallback_kinds=("identity", "genetic_fallback"),
        gate_modes=("off",),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static", "state_aware_median"),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid",),
        supports_full_proposal_log=True,
        supports_warm_start=True,
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=100.0,
        empty_cell_fitness=0.0,
    )


def pcg_sokoban_capabilities() -> AdapterCapabilities:
    """PCG sokoban-v0 public task (matched structural_counts)."""
    return AdapterCapabilities(
        adapter_id="pcg-sokoban-native",
        adapter_version=ADAPTER_VERSION,
        domain_id="pcg_sokoban",
        initialization_kinds=("generated_floor", "loaded_archive"),
        selectors=(
            "min_fitness_frontier",
            "uniform_frontier",
            "max_fitness_frontier",
        ),
        generators=("random", "genetic", "llm"),
        prompt_channels=("not_applicable", "constant", "live"),
        repair_fallback_kinds=(
            "identity",
            "structural_counts",
            "genetic_fallback",
        ),
        gate_modes=("off",),
        replacement_kinds=("strict_single_elite",),
        allocation_kinds=("static", "state_aware_median"),
        budget_axes=_BUDGET_AXES,
        archive_types=("grid",),
        supports_full_proposal_log=True,
        supports_warm_start=True,
        stochastic_evaluation=False,
        native_fitness_min=0.0,
        native_fitness_max=1.0,
        empty_cell_fitness=0.0,
    )


def current_domain_capabilities() -> dict[str, AdapterCapabilities]:
    """Return all current declarations keyed by domain ID."""
    declarations = (
        ca_capabilities(),
        maze_capabilities(),
        dungeon_capabilities(),
        sphere_capabilities(),
        nas201_capabilities(),
        pcg_sokoban_capabilities(),
    )
    return {item.domain_id: item for item in declarations}
