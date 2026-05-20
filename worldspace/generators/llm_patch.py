"""LLM prompt construction and patch extraction (local vs global-search)."""

from __future__ import annotations

import json
from typing import Any

from worldspace.prompt_files import read_prompt

from ..simulator import SimulationResult
from ..specs.spec import WorldSpec
from ..specs.world_spec_constraints import WORLD_SPEC_CONSTRAINTS
from .llm_config import LLMGeneratorConfig, LlmTextCaller, LlmVisionCaller
from .llm_descriptive import describe_simulation

_OUTPUT_FORMAT = json.loads(read_prompt("llm_patch_output_format.json"))


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
            "goal": read_prompt("llm_patch_local_goal.txt").strip(),
            "constraints": WORLD_SPEC_CONSTRAINTS,
            "output_format": _OUTPUT_FORMAT,
            "instruction": read_prompt("llm_patch_instruction.txt").strip(),
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
            "goal": read_prompt("llm_patch_global_goal.txt").strip(),
            "constraints": WORLD_SPEC_CONSTRAINTS,
            "rules": _load_global_rules(),
            "output_format": _OUTPUT_FORMAT,
            "instruction": read_prompt("llm_patch_instruction.txt").strip(),
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
        template = read_prompt("llm_hybrid_local_user.txt")
        return template.format(
            current_world_json=json.dumps(world.to_json_dict(), ensure_ascii=True),
            metrics_block=_format_metrics_block(metrics),
        )


def _format_metrics_block(metrics: Any) -> str:
    return "\n".join(
        [
            f"density: {float(metrics.density_mean):.6f}",
            f"entropy: {float(metrics.entropy):.6f}",
            f"stability: {float(metrics.stability):.6f}",
            f"survival (avg lifespan): {float(metrics.average_lifespan):.6f}",
            f"diversity: {float(metrics.diversity):.6f}",
            f"oscillation_score: {float(metrics.oscillation_score):.6f}",
            f"topology_interface_index: {float(metrics.topology_interface_index):.6f}",
            f"topology_window_heterogeneity: {float(metrics.topology_window_heterogeneity):.6f}",
            f"compressibility_score: {float(metrics.compressibility_score):.6f}",
            f"ecology_state_entropy_norm: {float(metrics.ecology_state_entropy_norm):.6f}",
            f"ecology_resource_adjacency: {float(metrics.ecology_resource_adjacency):.6f}",
            f"mo_eoc_indicator: {float(metrics.mo_eoc_indicator):.6f}",
        ]
    )


def _load_global_rules() -> list[str]:
    lines = [
        line.strip()
        for line in read_prompt("llm_patch_global_rules.txt").splitlines()
        if line.strip()
    ]
    return lines
