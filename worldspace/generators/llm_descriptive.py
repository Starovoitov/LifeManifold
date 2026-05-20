"""Simulation visuals and vision/text descriptions for global-search LLM mode."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..simulator import SimulationResult
from worldspace.prompt_files import read_prompt
from .llm_config import LLMGeneratorConfig, LlmVisionCaller


def render_simulation_png(
    result: SimulationResult,
    *,
    max_side: int = 128,
) -> bytes:
    """Render life+food RGB grid as PNG bytes (downsampled for vision APIs)."""
    life = result.final_life
    food = result.final_food
    if life is None or food is None:
        raise ValueError("render_simulation_png requires final_life and final_food")
    life_s = _downsample_grid(life, max_side=max_side)
    food_s = _downsample_grid(food, max_side=max_side)
    rgb = _life_food_rgb(life_s, food_s)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=72)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    m = result.metrics
    ax.set_title(
        f"density={m.density_mean:.3f} osc={m.oscillation_score:.3f} "
        f"mo_eoc={m.mo_eoc_indicator:.3f}",
        fontsize=8,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def deterministic_text_summary(result: SimulationResult) -> str:
    """Compact non-LLM summary from final grids and metrics."""
    life = result.final_life
    food = result.final_food
    if life is None or food is None:
        return "No final grid available."
    m = result.metrics
    live_frac = float((life > 0.5).mean())
    food_frac = float((food > 0.5).mean())
    both_frac = float(((life > 0.5) & (food > 0.5)).mean())
    tags: list[str] = []
    if live_frac < 0.02:
        tags.append("near-extinction")
    elif live_frac > 0.55:
        tags.append("dense live field")
    else:
        tags.append("moderate live density")
    if m.stability > 0.75:
        tags.append("high stability")
    if m.oscillation_score > 0.35:
        tags.append("notable temporal oscillation")
    if m.topology_interface_index > 0.2:
        tags.append("strong live/empty interfaces")
    if m.topology_window_heterogeneity > 0.4:
        tags.append("heterogeneous 2×2 windows")
    if both_frac > 0.05:
        tags.append("life–food co-occurrence patches")
    tag_str = ", ".join(tags) if tags else "mixed dynamics"
    return (
        f"Final state: live occupancy {live_frac:.3f}, food {food_frac:.3f}, "
        f"co-located {both_frac:.3f}. Metrics tags: {tag_str}. "
        f"mo_eoc={m.mo_eoc_indicator:.4f}, entropy={m.entropy:.4f}, "
        f"diversity={m.diversity:.4f}."
    )


def describe_simulation_vision(
    config: LLMGeneratorConfig,
    png_bytes: bytes,
    *,
    call_llm_vision: LlmVisionCaller,
) -> str:
    """Ask a vision-capable chat model to caption the simulation frame."""
    return call_llm_vision(
        mode=config.mode,
        provider_name=config.vision_provider,
        providers=config.providers,
        system_content=read_prompt("llm_vision_system.txt"),
        user_text=read_prompt("llm_vision_user.txt"),
        image_png_bytes=png_bytes,
        temperature=config.vision_temperature,
        max_tokens=config.vision_max_tokens,
    ).strip()


def describe_simulation(
    result: SimulationResult,
    config: LLMGeneratorConfig,
    *,
    call_llm_vision: LlmVisionCaller,
) -> str:
    """Vision caption with deterministic fallback on failure."""
    fallback = deterministic_text_summary(result)
    try:
        png = render_simulation_png(result, max_side=config.descriptive_max_side)
        caption = describe_simulation_vision(
            config, png, call_llm_vision=call_llm_vision
        )
        if caption:
            return caption
    except RuntimeError:
        pass
    except ValueError:
        pass
    return fallback


def _downsample_grid(grid: np.ndarray, *, max_side: int) -> np.ndarray:
    h, w = grid.shape
    if max(h, w) <= max_side:
        return grid
    step = int(np.ceil(max(h, w) / max_side))
    return grid[::step, ::step]


def _life_food_rgb(life: np.ndarray, food: np.ndarray) -> np.ndarray:
    life = life.astype(np.float32)
    food = food.astype(np.float32)
    h, w = life.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    empty = (life < 0.5) & (food < 0.5)
    l_only = (life >= 0.5) & (food < 0.5)
    f_only = (life < 0.5) & (food >= 0.5)
    both = (life >= 0.5) & (food >= 0.5)
    rgb[empty] = np.array([0.10, 0.11, 0.15], dtype=np.float32)
    rgb[l_only] = np.array([0.29, 0.61, 0.56], dtype=np.float32)
    rgb[f_only] = np.array([0.79, 0.63, 0.15], dtype=np.float32)
    rgb[both] = np.array([0.43, 0.48, 0.31], dtype=np.float32)
    return rgb
