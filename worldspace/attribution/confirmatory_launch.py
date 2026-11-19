"""Execute a frozen confirmatory job plan. Dry-run unless an ack is passed.

The first LLM run writes ``confirmatory_clock.json``. That is the launcher's
record of the protocol's 14-day cap. This module does not edit the registered
protocol file and does not mark a plan as launched.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldspace.attribution.confirmatory_studies import (
    allocation_study,
    build_confirmatory_job_plan,
    generator_study,
    selector_channel_study,
)
from worldspace.attribution.design import JobPlan
from worldspace.attribution.job_builder import write_job_plan
from worldspace.attribution.manifest import RunManifest
from worldspace.attribution.public_loop import (
    SUMMARY_FILENAME,
    PublicRunConfig,
)

EXECUTE_ACK = "ca-phase3-v0-confirmatory"
CLOCK_FILENAME = "confirmatory_clock.json"
REGISTERED_RESPONSE_PIN = "gpt-4o-mini-2024-07-18"
PARTIAL_ARTIFACTS = frozenset(
    {
        SUMMARY_FILENAME,
        "archive_trace.jsonl",
        "nas201_archive.jsonl",
        "pcg_archive.jsonl",
        "llm_call_log.jsonl",
    }
)

STUDY_BUILDERS = {
    "selector-channel": selector_channel_study,
    "generator": generator_study,
    "allocation": allocation_study,
}

RunExecutor = Callable[[RunManifest], None]


class ConfirmatoryLaunchError(RuntimeError):
    """The plan must not start."""


def build_confirmatory_plan(
    domain: str,
    study_kind: str,
    *,
    output_root: str,
) -> JobPlan:
    """Build one unlaunched plan for a registered confirmatory study."""
    if domain not in {"nas201", "pcg_sokoban"}:
        raise ConfirmatoryLaunchError(f"unsupported domain {domain!r}")
    try:
        builder = STUDY_BUILDERS[study_kind]
    except KeyError as exc:
        raise ConfirmatoryLaunchError(f"unsupported study {study_kind!r}") from exc
    study = builder(domain)  # type: ignore[arg-type]
    plan = build_confirmatory_job_plan(study, output_root=output_root)
    if plan.launched:
        raise ConfirmatoryLaunchError("refusing a launched plan")
    return plan


def ordered_runs(plan: JobPlan) -> list[RunManifest]:
    """Seed-major arm interleave already stored on the plan."""
    by_id = {run.run_id: run for run in plan.runs}
    ordered: list[RunManifest] = []
    for block_id in sorted(plan.execution_order):
        for run_id in plan.execution_order[block_id]:
            try:
                ordered.append(by_id[run_id])
            except KeyError as exc:
                raise ConfirmatoryLaunchError(
                    f"execution order names unknown run {run_id!r}"
                ) from exc
    if len(ordered) != len(plan.runs):
        raise ConfirmatoryLaunchError("execution order does not cover every run")
    return ordered


def public_config_from_run(run: RunManifest) -> PublicRunConfig:
    """Map one frozen manifest onto the native runner knobs."""
    treatment = run.treatment
    generator = treatment.generator.kind
    if generator not in {"random", "genetic", "llm"}:
        raise ConfirmatoryLaunchError(f"unsupported generator {generator!r}")
    selector = treatment.selector.kind
    if selector not in {
        "uniform_frontier",
        "min_fitness_frontier",
        "max_fitness_frontier",
    }:
        raise ConfirmatoryLaunchError(f"unsupported selector {selector!r}")
    allocation = treatment.allocation.kind
    if allocation not in {"static", "state_aware_median"}:
        raise ConfirmatoryLaunchError(f"unsupported allocation {allocation!r}")
    channel = treatment.prompt_channel.kind
    if generator != "llm":
        channel = "not_applicable"
    elif channel not in {"constant", "live"}:
        raise ConfirmatoryLaunchError(f"unsupported prompt channel {channel!r}")
    floor = int(treatment.initialization.parameters["floor_random"])
    slots = treatment.budget.caps.proposal_slots
    if slots is None:
        raise ConfirmatoryLaunchError("proposal_slots cap is required")
    horizon = int(slots) - floor
    if floor != 20 or horizon != 200:
        raise ConfirmatoryLaunchError(
            f"registered budget is floor 20 and horizon 200, got {floor}+{horizon}"
        )
    pinned = run.model.parameters.get("pinned_response")
    if generator == "llm":
        if pinned != REGISTERED_RESPONSE_PIN:
            raise ConfirmatoryLaunchError(
                f"LLM arm {run.arm_id!r} is not pinned to {REGISTERED_RESPONSE_PIN}"
            )
        expected: str | None = str(pinned)
    else:
        expected = None
    return PublicRunConfig(
        seed=run.seed,
        generator=generator,  # type: ignore[arg-type]
        selector=selector,  # type: ignore[arg-type]
        allocation=allocation,  # type: ignore[arg-type]
        prompt_channel=channel,  # type: ignore[arg-type]
        repair_kind=treatment.repair_fallback.kind,
        floor_random=floor,
        search_horizon=horizon,
        llm_call_cap=200,
        expected_response_model=expected,
    )


def run_disposition(run: RunManifest) -> str:
    """``complete``, ``incomplete``, or ``pending``. A manifest file alone is pending."""
    run_dir = Path(run.output_paths["run_dir"])
    summary_path = run_dir / SUMMARY_FILENAME
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("completed") is True:
            return "complete"
        return "incomplete"
    if not run_dir.exists():
        return "pending"
    names = {path.name for path in run_dir.iterdir() if path.is_file()}
    if names & (PARTIAL_ARTIFACTS - {SUMMARY_FILENAME}):
        return "incomplete"
    return "pending"


def launch_plan(
    plan: JobPlan,
    *,
    execute: bool,
    ack: str | None,
    output_root: str,
    executor: RunExecutor | None = None,
) -> list[dict[str, Any]]:
    """List the interleaved queue, or run pending manifests after the ack."""
    ordered = ordered_runs(plan)
    if not execute:
        return [_describe(run, "dry-run") for run in ordered]
    if ack != EXECUTE_ACK:
        raise ConfirmatoryLaunchError(
            "refusing to start confirmatory runs without "
            f"--execute --ack {EXECUTE_ACK}"
        )
    incomplete = [run.run_id for run in ordered if run_disposition(run) == "incomplete"]
    if incomplete:
        raise ConfirmatoryLaunchError(
            "incomplete confirmatory run(s) must be resolved before continuing: "
            + ", ".join(incomplete)
        )
    write_job_plan(plan, study_root=Path(output_root) / plan.study_id)
    rows: list[dict[str, Any]] = []
    for run in ordered:
        if run_disposition(run) == "complete":
            rows.append(_describe(run, "skipped-complete"))
            continue
        if run.treatment.generator.kind == "llm":
            _ensure_clock(plan, run, output_root=output_root)
        if executor is None:
            raise ConfirmatoryLaunchError("no run executor was provided")
        executor(run)
        rows.append(_describe(run, "executed"))
    return rows


def execute_native_run(run: RunManifest, *, repo_root: Path) -> None:
    """Run one manifest with the native NAS or Sokoban loop."""
    config = public_config_from_run(run)
    output_dir = Path(run.output_paths["run_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if run.domain_id == "nas201":
        _execute_nas(run, config, output_dir, repo_root=repo_root)
        return
    if run.domain_id == "pcg_sokoban":
        _execute_pcg(run, config, output_dir, repo_root=repo_root)
        return
    raise ConfirmatoryLaunchError(f"unsupported domain {run.domain_id!r}")


def _execute_nas(
    run: RunManifest,
    config: PublicRunConfig,
    output_dir: Path,
    *,
    repo_root: Path,
) -> None:
    from worldspace.nas201.descriptors import load_frozen_bin_edges
    from worldspace.nas201.llm_emitter import Nas201LlmEmitter, bind_nas201_llm
    from worldspace.nas201.runner import run_nas201_qd
    from worldspace.nas201.table import CompactNas201Table

    nas_root = repo_root / "artifacts" / "controlled_attribution" / "nas201"
    table = CompactNas201Table.from_jsonl(
        nas_root / "nas201_compact_cifar10_valid.jsonl",
        nas_root / "nas201_compact_cifar10_valid.meta.json",
    )
    edges = load_frozen_bin_edges(nas_root / "nas201_bin_edges.json")
    llm_emit = None
    if config.generator == "llm":
        emitter = Nas201LlmEmitter(channel=config.prompt_channel)  # type: ignore[arg-type]
        _pin_request_model(emitter, REGISTERED_RESPONSE_PIN)
        llm_emit = bind_nas201_llm(emitter)
    run_nas201_qd(table, edges, config, output_dir=output_dir, llm_emit=llm_emit)


def _execute_pcg(
    run: RunManifest,
    config: PublicRunConfig,
    output_dir: Path,
    *,
    repo_root: Path,
) -> None:
    from worldspace.pcg.descriptors import load_frozen_bin_edges
    from worldspace.pcg.env import BenchmarkPcgEnv
    from worldspace.pcg.llm_emitter import PcgSokobanLlmEmitter, bind_pcg_llm
    from worldspace.pcg.runner import run_pcg_sokoban_qd
    from worldspace.pcg.spec import SOKOBAN_V0

    edges = load_frozen_bin_edges(
        repo_root
        / "artifacts"
        / "controlled_attribution"
        / "pcg"
        / "sokoban_v0_bin_edges_structural_counts.json"
    )
    env = BenchmarkPcgEnv(SOKOBAN_V0.problem_name, seed=run.seed)
    llm_emit = None
    if config.generator == "llm":
        emitter = PcgSokobanLlmEmitter(channel=config.prompt_channel)  # type: ignore[arg-type]
        _pin_request_model(emitter, REGISTERED_RESPONSE_PIN)
        llm_emit = bind_pcg_llm(emitter)
    run_pcg_sokoban_qd(env, edges, config, output_dir=output_dir, llm_emit=llm_emit)


def _pin_request_model(emitter: Any, model: str) -> None:
    name = emitter.config.active_provider
    current = emitter.config.providers[name]
    emitter.config.providers[name] = {**current, "model": model}


def _ensure_clock(plan: JobPlan, run: RunManifest, *, output_root: str) -> None:
    path = Path(output_root) / CLOCK_FILENAME
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": "ca-phase3-v0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": plan.study_id,
        "run_id": run.run_id,
        "note": (
            "14 calendar days from the first confirmatory API call. "
            "Written immediately before that run."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _describe(run: RunManifest, action: str) -> dict[str, Any]:
    return {
        "action": action,
        "run_id": run.run_id,
        "arm_id": run.arm_id,
        "seed": run.seed,
        "domain_id": run.domain_id,
        "generator": run.treatment.generator.kind,
        "output_dir": run.output_paths["run_dir"],
    }


def describe_rows(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)
