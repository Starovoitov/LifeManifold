"""Finite-difference fitness sensitivity for LLM direction-of-improvement hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.specs.spec import WorldSpec
from worldspace.surrogate.feature_extractor import (
    FEATURE_NAMES_V20,
    extract,
)
from worldspace.surrogate.types import SurrogatePrediction
from worldspace.surrogate.utils import compute_fitness_from_prediction

if TYPE_CHECKING:
    from worldspace.surrogate.model import SurrogateModel

DEFAULT_DIRECTION_EPSILON = 0.05
DEFAULT_TOP_K = 5
DEFAULT_MIN_ABS_GRADIENT = 1e-5
ACTIONABLE_FEATURE_COUNT = len(FEATURE_NAMES_V20)

DIRECTION_HINT_EMPTY = "Surrogate direction hints: unavailable (stub surrogate)."
DIRECTION_HINT_FLAT = (
    "Surrogate direction hints: flat (no strong local sensitivity at this parent)."
)

__all__ = [
    "ACTIONABLE_FEATURE_COUNT",
    "DIRECTION_HINT_EMPTY",
    "DIRECTION_HINT_FLAT",
    "DEFAULT_DIRECTION_EPSILON",
    "DEFAULT_MIN_ABS_GRADIENT",
    "DEFAULT_TOP_K",
    "compute_composed_fitness_gradient",
    "direction_prompt_fields",
    "fitness_at_feature_vector",
    "format_direction_hint_block",
    "suggestion_for_feature",
]


def fitness_at_feature_vector(
    model: SurrogateModel,
    features: np.ndarray,
    *,
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
) -> float:
    """Return surrogate fitness at one feature row (unclipped for sensitivity)."""
    vector = np.asarray(features, dtype=float).reshape(-1)
    direct = model.predict_fitness(vector)
    if direct is not None:
        return float(direct)
    components = model.predict_components(vector)
    prediction = SurrogatePrediction(
        components=components,
        measures={
            "stability": float(components["stability"]),
            "diversity": float(components["diversity"]),
        },
        fitness=0.0,
        uncertainty=0.0,
    )
    return float(
        compute_fitness_from_prediction(
            prediction,
            use_soft_extinction=use_soft_extinction,
            extinction_gate_threshold=extinction_gate_threshold,
        )
    )


def _is_rule_feature(feature_name: str) -> bool:
    return feature_name.startswith("birth_") or feature_name.startswith("survival_")


def compute_composed_fitness_gradient(
    model: SurrogateModel,
    features: np.ndarray,
    *,
    epsilon: float = DEFAULT_DIRECTION_EPSILON,
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
    actionable_dims: int = ACTIONABLE_FEATURE_COUNT,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Finite-difference fitness sensitivity in genome-aligned feature space (21-D)."""
    base = np.asarray(features, dtype=float).reshape(-1)
    limit = min(int(actionable_dims), int(base.shape[0]), len(FEATURE_NAMES_V20))
    names = FEATURE_NAMES_V20[:limit]
    gradient = np.zeros(limit, dtype=float)
    for index in range(limit):
        feature_name = names[index]
        if _is_rule_feature(feature_name):
            enabled = base.copy()
            disabled = base.copy()
            enabled[index] = 1.0
            disabled[index] = 0.0
            fitness_enabled = fitness_at_feature_vector(
                model,
                enabled,
                use_soft_extinction=use_soft_extinction,
                extinction_gate_threshold=extinction_gate_threshold,
            )
            fitness_disabled = fitness_at_feature_vector(
                model,
                disabled,
                use_soft_extinction=use_soft_extinction,
                extinction_gate_threshold=extinction_gate_threshold,
            )
            gradient[index] = float(fitness_enabled - fitness_disabled)
            continue
        step = float(epsilon)
        if feature_name in {"noise", "resource_regen", "predation"}:
            step = min(step, 0.05)
        plus = base.copy()
        minus = base.copy()
        plus[index] = float(np.clip(base[index] + step, 0.0, 1.0))
        minus[index] = float(np.clip(base[index] - step, 0.0, 1.0))
        delta = plus[index] - minus[index]
        if delta <= 0.0:
            gradient[index] = 0.0
            continue
        fitness_plus = fitness_at_feature_vector(
            model,
            plus,
            use_soft_extinction=use_soft_extinction,
            extinction_gate_threshold=extinction_gate_threshold,
        )
        fitness_minus = fitness_at_feature_vector(
            model,
            minus,
            use_soft_extinction=use_soft_extinction,
            extinction_gate_threshold=extinction_gate_threshold,
        )
        gradient[index] = float((fitness_plus - fitness_minus) / delta)
    return gradient, names


def suggestion_for_feature(feature_name: str, gradient: float) -> str:
    """Map one feature partial to an English edit hint for the LLM."""
    magnitude = abs(float(gradient))
    strength = "strongly" if magnitude >= 0.05 else "slightly"
    if feature_name.startswith("birth_"):
        rule_index = int(feature_name.split("_", maxsplit=1)[1])
        if gradient > 0.0:
            return (
                f"{strength} favor enabling birth rule index {rule_index} "
                f"(∂fit/∂birth_{rule_index} ≈ {gradient:+.3f})"
            )
        return (
            f"{strength} favor disabling birth rule index {rule_index} "
            f"(∂fit/∂birth_{rule_index} ≈ {gradient:+.3f})"
        )
    if feature_name.startswith("survival_"):
        rule_index = int(feature_name.split("_", maxsplit=1)[1])
        if gradient > 0.0:
            return (
                f"{strength} favor enabling survival rule index {rule_index} "
                f"(∂fit/∂survival_{rule_index} ≈ {gradient:+.3f})"
            )
        return (
            f"{strength} favor disabling survival rule index {rule_index} "
            f"(∂fit/∂survival_{rule_index} ≈ {gradient:+.3f})"
        )
    if gradient > 0.0:
        return (
            f"{strength} increase `{feature_name}` "
            f"(∂fit/∂{feature_name} ≈ {gradient:+.3f})"
        )
    return (
        f"{strength} decrease `{feature_name}` "
        f"(∂fit/∂{feature_name} ≈ {gradient:+.3f})"
    )


def format_direction_hint_block(
    gradient: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_abs_gradient: float = DEFAULT_MIN_ABS_GRADIENT,
) -> str:
    """Render the direction-of-improvement block for the user prompt."""
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if grad.size == 0:
        return DIRECTION_HINT_FLAT
    order = np.argsort(-np.abs(grad))
    lines = [
        "Surrogate local sensitivity (direction-of-improvement; genome-aligned features):"
    ]
    shown = 0
    for index in order[: max(int(top_k), 1)]:
        if index >= len(feature_names):
            continue
        value = float(grad[index])
        if abs(value) < float(min_abs_gradient):
            continue
        lines.append(f"  - {suggestion_for_feature(feature_names[index], value)}")
        shown += 1
    if shown == 0:
        return DIRECTION_HINT_FLAT
    lines.append(
        "Prefer 1–3 edits aligned with these directions; keep mutations small when "
        "surrogate uncertainty is high."
    )
    return "\n".join(lines)


def direction_prompt_fields(
    world_spec: WorldSpec,
    model: SurrogateModel,
    *,
    epsilon: float = DEFAULT_DIRECTION_EPSILON,
    top_k: int = DEFAULT_TOP_K,
    min_abs_gradient: float = DEFAULT_MIN_ABS_GRADIENT,
    use_soft_extinction: bool = False,
    extinction_gate_threshold: float = 0.5,
) -> dict[str, str]:
    """Build ``direction_hint_block`` kwargs for ``build_user_prompt``."""
    apply_canonical_seed(world_spec)
    features = extract(world_spec)
    if features.shape[0] < len(FEATURE_NAMES_V20):
        msg = (
            f"feature vector dim {features.shape[0]} < actionable "
            f"{len(FEATURE_NAMES_V20)}"
        )
        raise ValueError(msg)
    gradient, names = compute_composed_fitness_gradient(
        model,
        features,
        epsilon=epsilon,
        use_soft_extinction=use_soft_extinction,
        extinction_gate_threshold=extinction_gate_threshold,
    )
    return {
        "direction_hint_block": format_direction_hint_block(
            gradient,
            names,
            top_k=top_k,
            min_abs_gradient=min_abs_gradient,
        )
    }
