"""LLM prompt construction and patch extraction (local vs global-search)."""

from __future__ import annotations

import json
from typing import Any

from ..simulator import SimulationResult
from ..specs.spec import WorldSpec
from ..specs.world_param_bounds import (
    NOISE_MAX,
    NOISE_MIN,
    PREDATION_MAX,
    PREDATION_MIN,
    RESOURCE_REGEN_MAX,
    RESOURCE_REGEN_MIN,
)
from .llm_config import LLMGeneratorConfig, LlmTextCaller, LlmVisionCaller
from .llm_descriptive import describe_simulation


class LLMPatchAdvisor:
    """Build prompts and call LLMs for world-parameter patches."""

    def __init__(
        self,
        config: LLMGeneratorConfig,
        *,
        call_llm_text: LlmTextCaller,
        call_llm_vision: LlmVisionCaller,
    ):
        self.config = config
        self._call_llm_text = call_llm_text
        self._call_llm_vision = call_llm_vision

    @classmethod
    def from_config(
        cls,
        config: LLMGeneratorConfig,
        *,
        call_llm_text: LlmTextCaller,
        call_llm_vision: LlmVisionCaller,
    ) -> LLMPatchAdvisor:
        return cls(config, call_llm_text=call_llm_text, call_llm_vision=call_llm_vision)

    def describe(self, result: SimulationResult) -> str:
        if not self.config.global_search:
            return ""
        return describe_simulation(
            result,
            self.config,
            call_llm_vision=self._call_llm_vision,
        )

    def build_local_prompt(self, world: WorldSpec, score: float) -> str:
        payload = {
            "current_world": world.to_json_dict(),
            "current_mo_eoc_indicator": score,
            "goal": (
                "Increase the Multi-Objective + Edge-of-Chaos indicator (mo_eoc_indicator) "
                "while staying within valid bounds."
            ),
            "constraints": _CONSTRAINTS,
            "output_format": _OUTPUT_FORMAT,
            "instruction": "Return JSON only.",
        }
        return json.dumps(payload, ensure_ascii=True)

    def build_global_prompt(
        self,
        world: WorldSpec,
        result: SimulationResult,
        description: str,
    ) -> str:
        m = result.metrics
        payload = {
            "current_world": world.to_json_dict(),
            "simulation_description": description,
            "metrics": {
                "density_mean": float(m.density_mean),
                "entropy": float(m.entropy),
                "stability": float(m.stability),
                "average_lifespan": float(m.average_lifespan),
                "diversity": float(m.diversity),
                "oscillation_score": float(m.oscillation_score),
                "topology_interface_index": float(m.topology_interface_index),
                "topology_window_heterogeneity": float(m.topology_window_heterogeneity),
                "compressibility_score": float(m.compressibility_score),
                "ecology_state_entropy_norm": float(m.ecology_state_entropy_norm),
                "ecology_resource_adjacency": float(m.ecology_resource_adjacency),
                "mo_eoc_indicator": float(m.mo_eoc_indicator),
            },
            "goal": (
                "Increase mo_eoc_indicator using the simulation_description and metrics. "
                "Ground reasoning in observed patterns; do not invent structures not supported "
                "by the description."
            ),
            "constraints": _CONSTRAINTS,
            "rules": [
                "change at most 2 parameters",
                "keep system stable (avoid extinction or explosion)",
                "aim for balance between order and chaos",
            ],
            "output_format": _OUTPUT_FORMAT,
            "instruction": "Return JSON only.",
        }
        return json.dumps(payload, ensure_ascii=True)

    def build_hybrid_prompt(
        self,
        world: WorldSpec,
        result: SimulationResult,
        *,
        description: str = "",
    ) -> str:
        if self.config.global_search:
            return self.build_global_prompt(world, result, description)
        return self._build_hybrid_local_prompt(world, result.metrics)

    def request_patch(self, prompt: str) -> str:
        return self._call_llm_text(
            mode=self.config.mode,
            provider_name=self.config.active_provider,
            providers=self.config.providers,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _build_hybrid_local_prompt(self, world: WorldSpec, metrics: Any) -> str:
        return (
            "You are improving a cellular automaton world.\n\n"
            "Goal: increase the Multi-Objective + Edge-of-Chaos indicator (mo_eoc_indicator).\n\n"
            f"Current world:\n{json.dumps(world.to_json_dict(), ensure_ascii=True)}\n\n"
            "Metrics:\n"
            f"density: {float(metrics.density_mean):.6f}\n"
            f"entropy: {float(metrics.entropy):.6f}\n"
            f"stability: {float(metrics.stability):.6f}\n"
            f"survival (avg lifespan): {float(metrics.average_lifespan):.6f}\n"
            f"diversity: {float(metrics.diversity):.6f}\n"
            f"oscillation_score: {float(metrics.oscillation_score):.6f}\n"
            f"topology_interface_index: {float(metrics.topology_interface_index):.6f}\n"
            f"topology_window_heterogeneity: {float(metrics.topology_window_heterogeneity):.6f}\n"
            f"compressibility_score: {float(metrics.compressibility_score):.6f}\n"
            f"ecology_state_entropy_norm: {float(metrics.ecology_state_entropy_norm):.6f}\n"
            f"ecology_resource_adjacency: {float(metrics.ecology_resource_adjacency):.6f}\n"
            f"mo_eoc_indicator: {float(metrics.mo_eoc_indicator):.6f}\n\n"
            "Suggest a slightly improved version.\n\n"
            "Rules:\n"
            "- change at most 2 parameters\n"
            "- keep system stable (avoid extinction or explosion)\n"
            "- aim for balance between order and chaos\n\n"
            "Output ONLY JSON."
        )


_CONSTRAINTS = {
    "birth": "unique integers in [0,8], at least 1 item",
    "survival": "unique integers in [0,8], at least 1 item",
    "noise": f"[{NOISE_MIN},{NOISE_MAX}]",
    "resource_regen": f"[{RESOURCE_REGEN_MIN},{RESOURCE_REGEN_MAX}]",
    "predation": f"[{PREDATION_MIN},{PREDATION_MAX}]",
}

_OUTPUT_FORMAT = {
    "birth": [3],
    "survival": [2, 3],
    "noise": 0.05,
    "resource_regen": 0.1,
    "predation": 0.2,
    "reasoning": "short explanation",
}
