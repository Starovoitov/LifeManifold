#!/usr/bin/env python3
"""Pre-stamp engineering verification for Phase 3 public-task adapters.

Design data only. Does not launch confirmatory LLM. Does not stamp/tag.
Writes a JSON report under artifacts/controlled_attribution/verification/.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from worldspace.attribution.adapters import (
    Nas201NormalizationAdapter,
    NativeRunInputs,
    PcgSokobanNormalizationAdapter,
)
from worldspace.attribution.capabilities import current_domain_capabilities
from worldspace.attribution.confirmatory_admission import (
    refuse_contaminated_artifact_paths,
    refuse_mixed_evidence_tier,
)
from worldspace.attribution.confirmatory_studies import (
    allocation_study,
    build_confirmatory_job_plan,
    generator_study,
    selector_channel_study,
)
from worldspace.attribution.hashing import canonical_sha256
from worldspace.attribution.job_builder import write_job_plan
from worldspace.attribution.manifest import ArmManifest, arm_treatment_hash, freeze_run_manifest
from worldspace.attribution.public_loop import PublicRunConfig, should_use_llm
from worldspace.attribution.validation import AttributionAdmissionError
from worldspace.nas201.descriptors import load_frozen_bin_edges
from worldspace.nas201.live_prompt_scan import assert_live_prompt_templates as assert_nas_live
from worldspace.nas201.prompt_scan import assert_prompt_templates as assert_nas_constant
from worldspace.nas201.runner import run_nas201_qd
from worldspace.nas201.table import CompactNas201Table
from worldspace.pcg.descriptors import load_frozen_bin_edges as load_pcg_edges
from worldspace.pcg.env import BenchmarkPcgEnv
from worldspace.pcg.live_prompt_scan import assert_live_prompt_templates as assert_pcg_live
from worldspace.pcg.prompt_scan import assert_prompt_templates as assert_pcg_constant
from worldspace.pcg.runner import run_pcg_sokoban_qd
from worldspace.pcg.spec import SOKOBAN_V0

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "artifacts/controlled_attribution/verification"
REPORT_DIR = OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

HASH_A = "a" * 64
HASH_B = "b" * 64


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: dict = field(default_factory=dict)
    error: str | None = None


def _component(kind: str, **parameters: object) -> dict:
    return {
        "kind": kind,
        "version": "1",
        "parameters": parameters,
        "content_hashes": {},
    }


def _budget() -> dict:
    return {
        **_component("matched_exact"),
        "caps": {
            "proposal_slots": 40,
            "valid_proposals": None,
            "evaluator_calls": 40,
            "llm_calls_attempted": 0,
            "llm_calls_completed": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "evaluator_wall_seconds": None,
            "llm_latency_seconds": None,
            "wall_seconds": None,
            "monetary_cost": None,
        },
        "indexing_axes": ["proposal", "evaluation"],
        "stopping_precedence": ["proposal"],
    }


def _manifest(
    domain: str,
    *,
    seed: int,
    selector: str,
    generator: str,
    repair: str,
    allocation: str = "static",
    prompt_channel: str = "not_applicable",
) -> object:
    caps = current_domain_capabilities()[domain]
    arm = ArmManifest.model_validate(
        {
            "arm_id": f"{generator}-{selector}",
            "label": "verify",
            "role": "baseline",
            "treatment": {
                "initialization": _component(
                    "generated_floor",
                    floor_random=10,
                    archive_capacity=400 if domain == "nas201" else 100,
                ),
                "selector": _component(selector),
                "generator": _component(generator),
                "prompt_channel": _component(prompt_channel),
                "repair_fallback": _component(repair),
                "gate": _component("off"),
                "replacement": _component(
                    "strict_single_elite", archive_type="grid", strict=True
                ),
                "allocation": _component(allocation),
                "budget": _budget(),
            },
            "representation": _component(f"{domain}_genotype"),
            "model": _component("not_applicable"),
            "evaluator": _component(f"{domain}_evaluator"),
            "reference_arm_id": None,
            "expected_differences": [],
        }
    )
    return freeze_run_manifest(
        {
            "run_id": f"verify-{domain}-{seed}",
            "study_id": f"verify-{domain}",
            "arm_id": arm.arm_id,
            "pair_id": f"verify-{domain}-{seed}",
            "block_id": "verify",
            "evidence_tier": "feasibility",
            "protocol_id": "p3-verify",
            "protocol_hash": HASH_A,
            "domain_id": domain,
            "domain_version": "verify-1",
            "adapter_id": caps.adapter_id,
            "adapter_version": caps.adapter_version,
            "seed": seed,
            "domain_instance_id": f"{domain}-default",
            "initial_archive_id": "generated-floor",
            "initial_archive_hash": canonical_sha256([]),
            "treatment": arm.treatment,
            "representation": arm.representation,
            "model": arm.model,
            "evaluator": arm.evaluator,
            "treatment_hash": arm_treatment_hash(arm),
            "study_manifest_hash": HASH_A,
            "currency": "USD",
            "price_table_id": "fixture-free",
            "price_table_hash": HASH_B,
            "dependency_hashes": {"lock": HASH_B},
            "output_paths": {"run_dir": f"output/{domain}/{seed}"},
            "expected_artifacts": ["nightly_run_summary.json", "archive_trace.jsonl"],
        }
    )


def check_scanners() -> CheckResult:
    try:
        nas_c = assert_nas_constant()
        nas_l = assert_nas_live()
        pcg_c = assert_pcg_constant()
        pcg_l = assert_pcg_live()
        return CheckResult(
            "prompt_scanners",
            True,
            {
                "nas_constant_system_chars": len(nas_c["system"]),
                "nas_live_user_chars": len(nas_l["user"]),
                "pcg_constant_system_chars": len(pcg_c["system"]),
                "pcg_live_user_chars": len(pcg_l["user"]),
            },
        )
    except Exception as exc:  # noqa: BLE001 — collect into report
        return CheckResult("prompt_scanners", False, error=f"{type(exc).__name__}: {exc}")


def check_capabilities() -> CheckResult:
    caps = current_domain_capabilities()
    need = {"nas201", "pcg_sokoban"}
    missing = need - set(caps)
    ok = not missing
    detail = {
        "domains": sorted(caps),
        "nas201_allocation": list(caps["nas201"].allocation_kinds) if "nas201" in caps else None,
        "pcg_repair": list(caps["pcg_sokoban"].repair_fallback_kinds)
        if "pcg_sokoban" in caps
        else None,
    }
    return CheckResult("capabilities", ok, detail, error=None if ok else f"missing {missing}")


def check_allocation_rule() -> CheckResult:
    low = should_use_llm(
        allocation="state_aware_median",
        archive_fitnesses=[1.0, 2.0, 3.0],
        target_empty=False,
        target_fitness=1.5,
        completed_llm_calls=0,
        llm_call_cap=10,
    )
    high = should_use_llm(
        allocation="state_aware_median",
        archive_fitnesses=[1.0, 2.0, 3.0],
        target_empty=False,
        target_fitness=2.5,
        completed_llm_calls=0,
        llm_call_cap=10,
    )
    ok = low is True and high is False
    return CheckResult(
        "state_aware_median_rule",
        ok,
        {"leq_median_uses_llm": low, "above_median_skips": high},
        error=None if ok else "median rule mismatch",
    )


def check_nas_loop() -> CheckResult:
    compact = ROOT / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.jsonl"
    meta = ROOT / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.meta.json"
    edges_path = ROOT / "artifacts/controlled_attribution/nas201/nas201_bin_edges.json"
    try:
        table = CompactNas201Table.from_jsonl(compact, meta)
        edges = load_frozen_bin_edges(edges_path)
        run_dir = REPORT_DIR / "nas201_genetic_seed42"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        config = PublicRunConfig(
            seed=42,
            generator="genetic",
            selector="uniform_frontier",
            repair_kind="identity",
            floor_random=10,
            search_horizon=30,
        )
        result = run_nas201_qd(table, edges, config, output_dir=run_dir)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}
        manifest = _manifest(
            "nas201",
            seed=42,
            selector="uniform_frontier",
            generator="genetic",
            repair="identity",
        )
        bundle = Nas201NormalizationAdapter().normalize(manifest, NativeRunInputs(run_dir))
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}
        ok = before == after and bundle.summary.final_archive.occupied_cells == result.filled_cells
        return CheckResult(
            "nas201_genetic_normalize",
            ok,
            {
                "filled_cells": result.filled_cells,
                "coverage": result.coverage,
                "qd_score": result.qd_score,
                "anytime_auc": result.anytime_auc,
                "proposals": result.summary["proposals"],
                "read_only": before == after,
                "event_completeness": bundle.summary.event_completeness,
                "run_dir": str(run_dir.relative_to(ROOT)),
            },
            error=None if ok else "normalize mismatch or tree mutated",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "nas201_genetic_normalize",
            False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def check_nas_random() -> CheckResult:
    compact = ROOT / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.jsonl"
    meta = ROOT / "artifacts/controlled_attribution/nas201/nas201_compact_cifar10_valid.meta.json"
    edges_path = ROOT / "artifacts/controlled_attribution/nas201/nas201_bin_edges.json"
    try:
        table = CompactNas201Table.from_jsonl(compact, meta)
        edges = load_frozen_bin_edges(edges_path)
        run_dir = REPORT_DIR / "nas201_random_seed43"
        config = PublicRunConfig(
            seed=43,
            generator="random",
            selector="min_fitness_frontier",
            floor_random=8,
            search_horizon=20,
        )
        result = run_nas201_qd(table, edges, config, output_dir=run_dir)
        manifest = _manifest(
            "nas201",
            seed=43,
            selector="min_fitness_frontier",
            generator="random",
            repair="identity",
        )
        bundle = Nas201NormalizationAdapter().normalize(manifest, NativeRunInputs(run_dir))
        return CheckResult(
            "nas201_random_normalize",
            True,
            {
                "filled_cells": result.filled_cells,
                "qd_score": result.qd_score,
                "occupied": bundle.summary.final_archive.occupied_cells,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "nas201_random_normalize",
            False,
            error=f"{type(exc).__name__}: {exc}",
        )


def check_pcg_loop() -> CheckResult:
    edges_path = (
        ROOT
        / "artifacts/controlled_attribution/pcg/sokoban_v0_bin_edges_structural_counts.json"
    )
    try:
        edges = load_pcg_edges(edges_path)
        if edges.problem_name != SOKOBAN_V0.problem_name:
            raise ValueError(f"unexpected edges problem {edges.problem_name}")
        env = BenchmarkPcgEnv(SOKOBAN_V0.problem_name, seed=51)
        run_dir = REPORT_DIR / "pcg_sokoban_genetic_seed51"
        config = PublicRunConfig(
            seed=51,
            generator="genetic",
            selector="uniform_frontier",
            repair_kind="structural_counts",
            floor_random=8,
            search_horizon=16,
        )
        result = run_pcg_sokoban_qd(env, edges, config, output_dir=run_dir)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}
        manifest = _manifest(
            "pcg_sokoban",
            seed=51,
            selector="uniform_frontier",
            generator="genetic",
            repair="structural_counts",
        )
        bundle = PcgSokobanNormalizationAdapter().normalize(
            manifest, NativeRunInputs(run_dir)
        )
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.iterdir()}
        ok = before == after
        return CheckResult(
            "pcg_sokoban_genetic_structural_counts_normalize",
            ok,
            {
                "filled_cells": result.filled_cells,
                "coverage": result.coverage,
                "qd_score": result.qd_score,
                "anytime_auc": result.anytime_auc,
                "playable": result.summary.get("playable"),
                "repair": result.summary.get("repair"),
                "read_only": before == after,
                "occupied": bundle.summary.final_archive.occupied_cells,
                "run_dir": str(run_dir.relative_to(ROOT)),
            },
            error=None if ok else "tree mutated under normalize",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "pcg_sokoban_genetic_structural_counts_normalize",
            False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def check_job_builder() -> CheckResult:
    try:
        out_root = REPORT_DIR / "jobs"
        counts: dict[str, int] = {}
        for domain in ("nas201", "pcg_sokoban"):
            for builder, name in (
                (selector_channel_study, "selector-channel"),
                (generator_study, "generator"),
                (allocation_study, "allocation"),
            ):
                study = builder(domain)  # type: ignore[arg-type]
                plan = build_confirmatory_job_plan(
                    study, output_root=str(out_root / study.study_id)
                )
                if plan.launched:
                    raise RuntimeError(f"{study.study_id} marked launched")
                write_job_plan(plan, study_root=out_root / study.study_id)
                counts[f"{domain}:{name}"] = len(plan.runs)
        return CheckResult("job_builder_emit", True, {"runs_by_study": counts, "launched": False})
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "job_builder_emit",
            False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def check_admission() -> CheckResult:
    refused = []
    try:
        refuse_contaminated_artifact_paths(
            ["artifacts/controlled_attribution/nas201/smoke/feasibility.jsonl"]
        )
        return CheckResult("admission_feasibility", False, error="did not refuse feasibility")
    except AttributionAdmissionError:
        refused.append("feasibility_jsonl")
    try:
        refuse_contaminated_artifact_paths(
            ["artifacts/nightly/q1/nightly_run_summary.json"]
        )
        return CheckResult("admission_q1", False, error="did not refuse q1 nightly")
    except AttributionAdmissionError:
        refused.append("q1_nightly")
    try:
        refuse_mixed_evidence_tier(["confirmatory", "feasibility"])
        return CheckResult("admission_mixed_tier", False, error="did not refuse mixed tier")
    except AttributionAdmissionError:
        refused.append("mixed_tier")
    return CheckResult("admission_guards", True, {"refused": refused})


def check_q1_scheduler_unchanged() -> CheckResult:
    # Spot-check: default mini scheduler path still present and hashed.
    try:
        from worldspace.illuminators.scheduler import DEFAULT_MINI_SCHEDULER_PATH

        path = Path(DEFAULT_MINI_SCHEDULER_PATH)
        if not path.is_file():
            return CheckResult("q1_scheduler_present", False, error=f"missing {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return CheckResult(
            "q1_scheduler_present",
            True,
            {"path": str(path), "sha256": digest, "note": "presence+hash only; not rewritten"},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("q1_scheduler_present", False, error=str(exc))


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checks = [
        check_capabilities(),
        check_scanners(),
        check_allocation_rule(),
        check_nas_loop(),
        check_nas_random(),
        check_pcg_loop(),
        check_job_builder(),
        check_admission(),
        check_q1_scheduler_unchanged(),
    ]
    report = {
        "evidence_tier": "feasibility",
        "role": "pre_stamp_engineering_verification",
        "llm_confirmatory": False,
        "stamp": False,
        "utc": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(REPORT_DIR.relative_to(ROOT)),
        "checks": [asdict(item) for item in checks],
        "passed": sum(1 for item in checks if item.ok),
        "failed": sum(1 for item in checks if not item.ok),
        "all_ok": all(item.ok for item in checks),
    }
    out = REPORT_DIR / "verification_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = REPORT_DIR / "VERIFICATION.md"
    lines = [
        "# Phase 3 pre-stamp verification",
        "",
        f"- UTC: `{report['utc']}`",
        f"- all_ok: **{report['all_ok']}** ({report['passed']}/{len(checks)})",
        f"- confirmatory LLM: no",
        f"- stamp/tag: no",
        "",
        "| Check | OK | Notes |",
        "|---|---|---|",
    ]
    for item in checks:
        note = item.error or json.dumps(item.detail, ensure_ascii=True)[:120]
        lines.append(f"| `{item.name}` | {item.ok} | {note} |")
    lines.append("")
    lines.append("Design data only. Do not pool into Phase 4. Do not move Holm/n/H from these numbers.")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "report": str(out)}, sort_keys=True))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
