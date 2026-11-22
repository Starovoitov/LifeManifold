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
    occupancy_study,
    robustness_study,
    selector_channel_study,
)
from worldspace.attribution.design import JobPlan
from worldspace.attribution.job_builder import write_job_plan
from worldspace.attribution.manifest import ComponentSpec, RunManifest
from worldspace.attribution.public_loop import (
    SUMMARY_FILENAME,
    PublicRunConfig,
)

EXECUTE_ACK = "ca-phase3-v0-confirmatory"
ROBUSTNESS_ACK = "ca-phase3-v0-robustness"
BLOCK_D_ACK = "ca-phase3-v0-block-d"
ROBUSTNESS_STUDY_ID = "robustness-nas201-selector-channel"
BLOCK_D_STUDY_ID = "sensitivity-nas201-occupancy"
BLOCK_D_GENETIC_FLOOR = 50
ROBUSTNESS_REQUEST_MODEL = "gpt-4o"
# Snapshots named by the stamped 2026-09-02 gpt-4o page. The request stays the alias.
ACCEPTED_GPT4O_SNAPSHOTS = frozenset(
    {
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
    }
)
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
    if run.evidence_tier == "sensitivity":
        raise ConfirmatoryLaunchError("Block D runs use the sensitivity launcher")
    treatment = run.treatment
    if int(treatment.initialization.parameters.get("floor_genetic") or 0) != 0:
        raise ConfirmatoryLaunchError("primary runs use the 20-random floor")
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
        if run.model.kind != "gpt-4o-mini" or pinned != REGISTERED_RESPONSE_PIN:
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
    if plan.evidence_tier != "confirmatory":
        raise ConfirmatoryLaunchError("this launcher only starts confirmatory plans")
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


def build_robustness_plan(*, output_root: str) -> JobPlan:
    """Build the unlaunched NAS Block A plan that requests gpt-4o."""
    study = robustness_study()
    plan = build_confirmatory_job_plan(study, output_root=output_root)
    if plan.launched or plan.study_id != ROBUSTNESS_STUDY_ID:
        raise ConfirmatoryLaunchError("refusing a launched or mis-identified robustness plan")
    if plan.evidence_tier != "robustness":
        raise ConfirmatoryLaunchError("robustness plan must stay outside the Holm tier")
    return plan


def public_config_from_robustness_run(run: RunManifest) -> PublicRunConfig:
    """Same budget as Block A, but the request alias is gpt-4o and is not pinned."""
    if run.evidence_tier != "robustness" or run.domain_id != "nas201":
        raise ConfirmatoryLaunchError("robustness runs are NAS Block A only")
    if run.model.kind != ROBUSTNESS_REQUEST_MODEL or run.model.parameters.get("pinned_response"):
        raise ConfirmatoryLaunchError(
            f"robustness arm {run.arm_id!r} must request {ROBUSTNESS_REQUEST_MODEL} without a snapshot pin"
        )
    disguised = run.model_copy(
        update={
            "model": ComponentSpec.model_validate(
                {
                    "kind": "gpt-4o-mini",
                    "version": run.model.version,
                    "parameters": {"pinned_response": REGISTERED_RESPONSE_PIN},
                    "content_hashes": {},
                }
            )
        }
    )
    config = public_config_from_run(disguised)
    return PublicRunConfig(
        seed=config.seed,
        generator=config.generator,
        selector=config.selector,
        allocation=config.allocation,
        prompt_channel=config.prompt_channel,
        repair_kind=config.repair_kind,
        floor_random=config.floor_random,
        search_horizon=config.search_horizon,
        llm_call_cap=config.llm_call_cap,
        expected_response_model=None,
    )


def launch_robustness_plan(
    plan: JobPlan,
    *,
    execute: bool,
    ack: str | None,
    output_root: str,
    executor: RunExecutor | None = None,
) -> list[dict[str, Any]]:
    """List the robustness queue, or run it after its own ack.

    A response snapshot outside the stamped gpt-4o list stops the queue.
    The primary confirmatory ack cannot start this plan.
    """
    if plan.study_id != ROBUSTNESS_STUDY_ID:
        raise ConfirmatoryLaunchError("this launcher only starts the NAS gpt-4o robustness plan")
    ordered = ordered_runs(plan)
    if not execute:
        return [_describe(run, "dry-run") for run in ordered]
    if ack != ROBUSTNESS_ACK:
        raise ConfirmatoryLaunchError(
            "refusing to start robustness runs without "
            f"--execute --ack {ROBUSTNESS_ACK}"
        )
    incomplete = [run.run_id for run in ordered if run_disposition(run) == "incomplete"]
    if incomplete:
        raise ConfirmatoryLaunchError(
            "incomplete robustness run(s) must be resolved before continuing: "
            + ", ".join(incomplete)
        )
    write_job_plan(plan, study_root=Path(output_root) / plan.study_id)
    observed: str | None = None
    rows: list[dict[str, Any]] = []
    for run in ordered:
        if run_disposition(run) == "complete":
            observed = _require_gpt4o_snapshot(run, observed)
            rows.append(_describe(run, "skipped-complete"))
            continue
        if run.treatment.generator.kind == "llm":
            _ensure_clock(plan, run, output_root=output_root)
        if executor is None:
            raise ConfirmatoryLaunchError("no run executor was provided")
        executor(run)
        observed = _require_gpt4o_snapshot(run, observed)
        rows.append(_describe(run, "executed"))
    return rows


def build_block_d_plan(*, output_root: str) -> JobPlan:
    """Build the unlaunched NAS occupancy plan. Both arms request the mini pin."""
    study = occupancy_study()
    plan = build_confirmatory_job_plan(study, output_root=output_root)
    if plan.launched or plan.study_id != BLOCK_D_STUDY_ID:
        raise ConfirmatoryLaunchError("refusing a launched or mis-identified Block D plan")
    if plan.evidence_tier != "sensitivity":
        raise ConfirmatoryLaunchError("Block D must stay outside the Holm tier")
    return plan


def public_config_from_block_d_run(run: RunManifest) -> PublicRunConfig:
    """A-const-uniform search after either the 20-random floor or the 50-genetic floor."""
    if run.evidence_tier != "sensitivity" or run.domain_id != "nas201":
        raise ConfirmatoryLaunchError("Block D runs are the NAS occupancy plan only")
    if run.study_id != BLOCK_D_STUDY_ID:
        raise ConfirmatoryLaunchError("Block D config refuses a different study")
    treatment = run.treatment
    if treatment.generator.kind != "llm":
        raise ConfirmatoryLaunchError(f"Block D arm {run.arm_id!r} is not the LLM treatment")
    if treatment.selector.kind != "uniform_frontier":
        raise ConfirmatoryLaunchError(f"Block D arm {run.arm_id!r} is not uniform_frontier")
    if treatment.allocation.kind != "static":
        raise ConfirmatoryLaunchError(f"Block D arm {run.arm_id!r} is not static")
    if treatment.prompt_channel.kind != "constant":
        raise ConfirmatoryLaunchError(f"Block D arm {run.arm_id!r} is not the constant channel")
    if treatment.repair_fallback.kind != "identity":
        raise ConfirmatoryLaunchError(f"Block D arm {run.arm_id!r} is not identity repair")
    pinned = run.model.parameters.get("pinned_response")
    if run.model.kind != "gpt-4o-mini" or pinned != REGISTERED_RESPONSE_PIN:
        raise ConfirmatoryLaunchError(
            f"Block D arm {run.arm_id!r} is not pinned to {REGISTERED_RESPONSE_PIN}"
        )
    floor_random = int(treatment.initialization.parameters.get("floor_random") or 0)
    floor_genetic = int(treatment.initialization.parameters.get("floor_genetic") or 0)
    if run.arm_id == "D-floor-random" and (floor_random, floor_genetic) != (20, 0):
        raise ConfirmatoryLaunchError("D-floor-random must be the primary 20-random floor")
    if run.arm_id == "D-floor-genetic" and (floor_random, floor_genetic) != (0, BLOCK_D_GENETIC_FLOOR):
        raise ConfirmatoryLaunchError("D-floor-genetic must be 50 one-step genetic proposals")
    if run.arm_id not in {"D-floor-random", "D-floor-genetic"}:
        raise ConfirmatoryLaunchError(f"unknown Block D arm {run.arm_id!r}")
    slots = treatment.budget.caps.proposal_slots
    if slots != floor_random + floor_genetic + 200:
        raise ConfirmatoryLaunchError(
            f"Block D arm {run.arm_id!r} proposal cap {slots} is not floor plus horizon 200"
        )
    return PublicRunConfig(
        seed=run.seed,
        generator="llm",
        selector="uniform_frontier",
        allocation="static",
        prompt_channel="constant",
        repair_kind="identity",
        floor_random=floor_random,
        floor_genetic=floor_genetic,
        search_horizon=200,
        llm_call_cap=200,
        expected_response_model=REGISTERED_RESPONSE_PIN,
    )


def launch_block_d_plan(
    plan: JobPlan,
    *,
    execute: bool,
    ack: str | None,
    output_root: str,
    executor: RunExecutor | None = None,
) -> list[dict[str, Any]]:
    """List the Block D queue, or run it after its own ack.

    The primary and robustness acks cannot start this plan. A response snapshot
    other than the registered mini pin stops the queue. The shared clock file
    is not rewritten.
    """
    if plan.study_id != BLOCK_D_STUDY_ID:
        raise ConfirmatoryLaunchError("this launcher only starts the NAS Block D plan")
    ordered = ordered_runs(plan)
    if not execute:
        return [_describe(run, "dry-run") for run in ordered]
    if ack != BLOCK_D_ACK:
        raise ConfirmatoryLaunchError(
            "refusing to start Block D runs without "
            f"--execute --ack {BLOCK_D_ACK}"
        )
    incomplete = [run.run_id for run in ordered if run_disposition(run) == "incomplete"]
    if incomplete:
        raise ConfirmatoryLaunchError(
            "incomplete Block D run(s) must be resolved before continuing: "
            + ", ".join(incomplete)
        )
    write_job_plan(plan, study_root=Path(output_root) / plan.study_id)
    rows: list[dict[str, Any]] = []
    for run in ordered:
        if run_disposition(run) == "complete":
            _require_registered_pin(run)
            rows.append(_describe(run, "skipped-complete"))
            continue
        if run.treatment.generator.kind == "llm":
            _ensure_clock(plan, run, output_root=output_root)
        if executor is None:
            raise ConfirmatoryLaunchError("no run executor was provided")
        executor(run)
        _require_registered_pin(run)
        rows.append(_describe(run, "executed"))
    return rows


def execute_block_d_run(run: RunManifest, *, repo_root: Path) -> None:
    """Run one Block D manifest on the pinned mini snapshot."""
    config = public_config_from_block_d_run(run)
    output_dir = Path(run.output_paths["run_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    _execute_nas(
        run,
        config,
        output_dir,
        repo_root=repo_root,
        request_model=REGISTERED_RESPONSE_PIN,
    )


def _require_registered_pin(run: RunManifest) -> None:
    summary_path = Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    response_model = payload.get("response_model")
    if response_model != REGISTERED_RESPONSE_PIN:
        raise ConfirmatoryLaunchError(
            f"run {run.run_id} returned response_model {response_model!r}; "
            "stopping the Block D queue for a dated amendment"
        )


def execute_robustness_run(run: RunManifest, *, repo_root: Path) -> None:
    """Run one robustness manifest. The request model is the gpt-4o alias."""
    config = public_config_from_robustness_run(run)
    output_dir = Path(run.output_paths["run_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    _execute_nas(
        run,
        config,
        output_dir,
        repo_root=repo_root,
        request_model=ROBUSTNESS_REQUEST_MODEL,
    )


def _require_gpt4o_snapshot(run: RunManifest, observed: str | None) -> str:
    summary_path = Path(run.output_paths["run_dir"]) / SUMMARY_FILENAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    response_model = payload.get("response_model")
    if response_model not in ACCEPTED_GPT4O_SNAPSHOTS:
        raise ConfirmatoryLaunchError(
            f"run {run.run_id} returned response_model {response_model!r}; "
            "stopping the robustness queue for a dated amendment"
        )
    if observed is not None and response_model != observed:
        raise ConfirmatoryLaunchError(
            f"run {run.run_id} returned {response_model}, after the queue had {observed}; "
            "stopping the robustness queue"
        )
    return str(response_model)


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
    request_model: str = REGISTERED_RESPONSE_PIN,
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
        _pin_request_model(emitter, request_model)
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
