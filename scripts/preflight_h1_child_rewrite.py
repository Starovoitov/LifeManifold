#!/usr/bin/env python3
"""Preflight for H1 child-rewrite Path A.

GO if checkpoint loads, rewrite prompt formats, and below_parent_true trigger works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldspace.illuminators.archive import GridArchive  # noqa: E402
from worldspace.illuminators.emitters.llm_emitter import (  # noqa: E402
    build_rewrite_user_prompt,
    should_rewrite_child,
)
from worldspace.illuminators.scheduler import (  # noqa: E402
    ChildRewriteConfig,
    TargetCell,
    load_scheduler,
)
from worldspace.specs.spec import WorldSpec  # noqa: E402
from worldspace.surrogate.model import TARGET_KEYS  # noqa: E402
from worldspace.surrogate.types import SurrogatePrediction  # noqa: E402

YAML = ROOT / "worldspace/specs/map_elites_scheduler_nightly_llm_hints_rewrite.yaml"
CKPT = ROOT / "artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"


def main() -> int:
    errors: list[str] = []
    if not YAML.is_file():
        errors.append(f"missing YAML: {YAML}")
    if not CKPT.is_file():
        errors.append(f"missing checkpoint: {CKPT}")

    cfg = None
    if YAML.is_file():
        cfg = load_scheduler(YAML)
        if not cfg.llm_child_rewrite.enabled:
            errors.append("llm.child_rewrite.enabled is false")
        if cfg.llm_child_rewrite.trigger != "below_parent_true":
            errors.append(f"unexpected trigger: {cfg.llm_child_rewrite.trigger!r}")

    components = {key: 0.40 for key in TARGET_KEYS}
    pred = SurrogatePrediction(
        fitness=0.40,
        uncertainty=0.16,
        measures={"stability": 0.40, "diversity": 0.40},
        components=components,
    )
    rewrite_cfg = ChildRewriteConfig(enabled=True, trigger="below_parent_true")
    if not should_rewrite_child(
        rewrite_cfg, child_pred_fitness=0.40, parent_true_fitness=0.50
    ):
        errors.append("below_parent_true should fire when child_pred < parent_true")
    if should_rewrite_child(
        rewrite_cfg, child_pred_fitness=0.55, parent_true_fitness=0.50
    ):
        errors.append(
            "below_parent_true should NOT fire when child_pred >= parent_true"
        )

    draft = WorldSpec(
        birth=[1, 3],
        survival=[2, 3],
        noise=0.02,
        resource_regen=0.05,
        predation=0.1,
        cell_types=["life", "food"],
        grid_size=8,
        steps=200,
        seed=0,
    )
    try:
        prompt = build_rewrite_user_prompt(
            target=TargetCell(
                cell_id=0,
                target_stability=0.5,
                target_diversity=0.5,
                bin_ij=(0, 0),
            ),
            archive=GridArchive(resolution=10),
            rng=np.random.default_rng(0),
            draft_spec=draft,
            child_prediction=pred,
            parent_true_fitness=0.50,
            user_prompt_path=str(
                ROOT / "prompts/map_elites_llm_emitter_user_rewrite.txt"
            ),
        )
        if "Draft WorldSpec" not in prompt or "0.400" not in prompt:
            errors.append("rewrite prompt missing expected fields")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rewrite prompt format failed: {exc}")

    if errors:
        print("PREFLIGHT NO-GO")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PREFLIGHT GO")
    print(f"  yaml={YAML}")
    print(f"  checkpoint={CKPT}")
    if cfg is not None:
        print(f"  trigger={cfg.llm_child_rewrite.trigger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
