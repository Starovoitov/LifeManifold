#!/usr/bin/env python3
"""Plan, preflight, and explicitly execute the clean RQ1 factorial.

The default command is read-only ``preflight``. Live execution requires both
``--execute`` and the exact protocol acknowledgement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any
import uuid

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.generators import call_llm  # noqa: E402
from worldspace.generators.llm_call_log import configure_llm_call_log  # noqa: E402
from worldspace.generators.llm_config import load_llm_config  # noqa: E402
from worldspace.illuminators.archive import (  # noqa: E402
    archive_record_to_elite,
    count_archive_jsonl_lines,
    load_and_collapse_jsonl,
)

PROTOCOL_ACK = "RQ1-CF-2026-08-13"
PLAN_SEED = 20260813
TARGET_FILLED = 971
ARCHIVE_COUNT = 5
CONTINUATIONS_PER_ARCHIVE = 2
SOURCE_SEEDS = tuple(range(41001, 41001 + ARCHIVE_COUNT))
EXPERIMENT_ROOT = _ROOT / "artifacts" / "experiments" / "q1-rq1-contemporaneous"
FIXED_LLM_SPEC = (
    _ROOT / "worldspace" / "specs" / "llm_world_generator_rq1_fixed_openai.yaml"
)
BASELINE_SCHEDULER = (
    _ROOT / "worldspace" / "specs" / "map_elites_scheduler_nightly.yaml"
)
CHECKPOINT = (
    _ROOT / "artifacts" / "surrogate" / "checkpoints" / "nightly_v3_mc_d005.pkl"
)
RUN_SCRIPT = _ROOT / "scripts" / "run_github_llm_map_elites.py"

ARM_SCHEDULERS = {
    "stub_minfit": "map_elites_scheduler_nightly_llm_stub.yaml",
    "live_minfit": "map_elites_scheduler_nightly_llm_hints_minfit.yaml",
    "stub_uniform": "map_elites_scheduler_nightly_llm_stub_uniform.yaml",
    "live_uniform": "map_elites_scheduler_nightly_llm.yaml",
    "shuffled_uniform": "map_elites_scheduler_nightly_llm_hints_placebo.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def _git_state() -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(_git(["status", "--porcelain"])),
    }


def _api_key_available() -> bool:
    if os.getenv("OPENAI_API_KEY", "").strip():
        return True
    dotenv = _ROOT / ".env"
    if not dotenv.is_file():
        return False
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY" and value.strip().strip("\"'"):
            return True
    return False


def _scheduler_path(arm: str) -> Path:
    return _ROOT / "worldspace" / "specs" / ARM_SCHEDULERS[arm]


def floor_path(archive_index: int) -> Path:
    return EXPERIMENT_ROOT / "floors" / f"archive_{archive_index:02d}.jsonl"


def floor_source_dir(archive_index: int) -> Path:
    return EXPERIMENT_ROOT / "floor_sources" / f"archive_{archive_index:02d}"


def continuation_seed(archive_index: int, continuation_index: int) -> int:
    return 42000 + archive_index * 100 + continuation_index


def arm_order(archive_index: int, continuation_index: int) -> list[str]:
    arms = list(ARM_SCHEDULERS)
    rng = random.Random(PLAN_SEED + archive_index * 100 + continuation_index)
    rng.shuffle(arms)
    return arms


def run_dir(archive_index: int, continuation_index: int, arm: str) -> Path:
    return (
        EXPERIMENT_ROOT
        / "blocks"
        / f"archive_{archive_index:02d}"
        / f"continuation_{continuation_index:02d}"
        / arm
    )


def build_plan() -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for archive_index in range(ARCHIVE_COUNT):
        for continuation_index in range(CONTINUATIONS_PER_ARCHIVE):
            blocks.append(
                {
                    "archive_index": archive_index,
                    "continuation_index": continuation_index,
                    "continuation_seed": continuation_seed(
                        archive_index, continuation_index
                    ),
                    "floor": str(floor_path(archive_index).relative_to(_ROOT)),
                    "arm_order": arm_order(archive_index, continuation_index),
                }
            )
    return {
        "protocol_ack": PROTOCOL_ACK,
        "plan_seed": PLAN_SEED,
        "target_filled_cells": TARGET_FILLED,
        "archive_count": ARCHIVE_COUNT,
        "continuations_per_archive": CONTINUATIONS_PER_ARCHIVE,
        "source_seeds": list(SOURCE_SEEDS),
        "fixed_llm_spec": str(FIXED_LLM_SPEC.relative_to(_ROOT)),
        "arms": {
            arm: str(_scheduler_path(arm).relative_to(_ROOT)) for arm in ARM_SCHEDULERS
        },
        "blocks": blocks,
    }


def write_plan(path: Path) -> None:
    payload = {
        **build_plan(),
        "plan_created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "file_hashes": {
            str(path_.relative_to(_ROOT)): _sha256(path_)
            for path_ in [
                FIXED_LLM_SPEC,
                BASELINE_SCHEDULER,
                *(_scheduler_path(arm) for arm in ARM_SCHEDULERS),
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_floor(path: Path) -> dict[str, Any]:
    archive = load_and_collapse_jsonl(path, resolution=50)
    raw = count_archive_jsonl_lines(path, resolution=50)
    filled = archive.filled_count()
    return {
        "path": str(path),
        "exists": True,
        "raw_records": raw,
        "filled_cells": filled,
        "coverage": filled / 2500.0,
        "sha256": _sha256(path),
        "valid": filled == TARGET_FILLED,
    }


def preflight() -> tuple[dict[str, Any], bool]:
    required = [
        FIXED_LLM_SPEC,
        BASELINE_SCHEDULER,
        CHECKPOINT,
        RUN_SCRIPT,
        *(_scheduler_path(arm) for arm in ARM_SCHEDULERS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    floors: list[dict[str, Any]] = []
    for archive_index in range(ARCHIVE_COUNT):
        path = floor_path(archive_index)
        if path.is_file():
            floors.append(_validate_floor(path))
        else:
            floors.append(
                {
                    "path": str(path),
                    "exists": False,
                    "valid": False,
                    "source_seed": SOURCE_SEEDS[archive_index],
                }
            )
    git = _git_state()
    probe_path = EXPERIMENT_ROOT / "preflight" / "model_probe.json"
    probe: dict[str, Any] | None = None
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe_valid = bool(
        probe
        and probe.get("ok")
        and probe.get("requested_model") == "gpt-4o-mini-2024-07-18"
        and probe.get("llm_spec_sha256") == _sha256(FIXED_LLM_SPEC)
    )
    result = {
        "protocol_ack": PROTOCOL_ACK,
        "git": git,
        "required_files_missing": missing,
        "openai_api_key_available_via_env_or_dotenv": _api_key_available(),
        "floors": floors,
        "dated_model_probe": {
            "path": str(probe_path),
            "performed": probe is not None,
            "valid_for_current_spec": probe_valid,
            "response_model": probe.get("response_model") if probe else None,
            "system_fingerprint": (probe.get("system_fingerprint") if probe else None),
        },
        "ready_for_live_execution": (
            not missing
            and not git["dirty"]
            and _api_key_available()
            and all(item["valid"] for item in floors)
            and probe_valid
        ),
    }
    return result, bool(result["ready_for_live_execution"])


def materialize_floor(source: Path, destination: Path) -> dict[str, Any]:
    """Write the shortest valid archive prefix that reaches 971 occupied cells."""
    if not source.is_file():
        raise FileNotFoundError(source)
    occupied: set[tuple[int, int]] = set()
    kept: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            elite = archive_record_to_elite(record, resolution=50)
            kept.append(json.dumps(record, ensure_ascii=True, sort_keys=True))
            occupied.add(elite.bin)
            if len(occupied) == TARGET_FILLED:
                break
    if len(occupied) != TARGET_FILLED:
        raise RuntimeError(
            f"{source} reaches only {len(occupied)} occupied cells; "
            f"need {TARGET_FILLED}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    validation = _validate_floor(destination)
    manifest = {
        **validation,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "prefix_records": len(kept),
        "construction": "shortest JSONL prefix reaching 971 unique bins",
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_execution(args: argparse.Namespace) -> None:
    if not args.execute or args.ack != PROTOCOL_ACK:
        raise SystemExit(
            "live execution refused: pass --execute " f"--ack {PROTOCOL_ACK}"
        )
    state = _git_state()
    if state["dirty"]:
        raise SystemExit("live execution refused: git worktree is dirty")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=_ROOT, env=env, check=True)


def run_floor_source(args: argparse.Namespace) -> None:
    _require_execution(args)
    archive_index = args.archive_index
    out = floor_source_dir(archive_index)
    command = [
        "uv",
        "run",
        "python",
        str(RUN_SCRIPT),
        "--scheduler",
        str(BASELINE_SCHEDULER),
        "--output-dir",
        str(out),
        "--seed",
        str(SOURCE_SEEDS[archive_index]),
        "--iterations",
        "1300",
        "--no-resume-nightly",
        "--llm-spec",
        str(FIXED_LLM_SPEC),
        "--surrogate-buffer",
        str(out / "surrogate_buffer.jsonl"),
    ]
    _run(command)
    materialize_floor(
        out / "map_elites_archive.jsonl",
        floor_path(archive_index),
    )


def probe_model(args: argparse.Namespace) -> None:
    """Make one explicitly authorized, archived request to the dated model."""
    _require_execution(args)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not exported; source .env before probing")
    config = load_llm_config(FIXED_LLM_SPEC)
    provider = config.providers[config.active_provider]
    preflight_dir = EXPERIMENT_ROOT / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    call_log = preflight_dir / "model_probe_llm_call_log.jsonl"
    if call_log.exists():
        raise SystemExit(
            f"probe log already exists: {call_log}; do not overwrite audit history"
        )
    call_id = f"model-probe-{uuid.uuid4().hex}"
    configure_llm_call_log(call_log)
    try:
        response = call_llm(
            mode=config.mode,
            provider_name=config.active_provider,
            providers=config.providers,
            prompt='Return exactly {"probe":"ok"} and no other text.',
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=32,
            system_content="You are a deterministic JSON probe.",
            audit_context={
                "llm_call_id": call_id,
                "purpose": "rq1_fixed_model_preflight",
            },
        )
    finally:
        configure_llm_call_log(None)
    rows = [
        json.loads(line)
        for line in call_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one model-probe log row, found {len(rows)}")
    row = rows[0]
    payload = {
        "ok": bool(row.get("ok")) and bool(response.strip()),
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": provider["model"],
        "response_model": row.get("response_model"),
        "system_fingerprint": row.get("system_fingerprint"),
        "response_id": row.get("response_id"),
        "llm_call_id": call_id,
        "llm_spec_sha256": _sha256(FIXED_LLM_SPEC),
        "git": _git_state(),
    }
    (preflight_dir / "model_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_manifest(
    *,
    archive_index: int,
    continuation_index: int,
    arm: str,
    order: list[str],
) -> dict[str, Any]:
    floor = floor_path(archive_index)
    scheduler = _scheduler_path(arm)
    return {
        "protocol_ack": PROTOCOL_ACK,
        "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "archive_index": archive_index,
        "continuation_index": continuation_index,
        "continuation_seed": continuation_seed(archive_index, continuation_index),
        "arm": arm,
        "arm_order": order,
        "arm_order_position": order.index(arm),
        "floor": str(floor.resolve()),
        "floor_sha256": _sha256(floor),
        "floor_filled_cells": TARGET_FILLED,
        "scheduler": str(scheduler.resolve()),
        "scheduler_sha256": _sha256(scheduler),
        "llm_spec": str(FIXED_LLM_SPEC.resolve()),
        "llm_spec_sha256": _sha256(FIXED_LLM_SPEC),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "llm_cache": "none",
        "llm_transport": {
            "timeout_seconds": 45,
            "max_attempts": 3,
            "retry_backoff_seconds": [2.0, 4.0],
        },
    }


def run_block(args: argparse.Namespace) -> None:
    _require_execution(args)
    archive_index = args.archive_index
    continuation_index = args.continuation_index
    floor = floor_path(archive_index)
    floor_validation = _validate_floor(floor)
    if not floor_validation["valid"]:
        raise SystemExit(f"invalid floor: {floor_validation}")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is missing")
    order = arm_order(archive_index, continuation_index)
    env = dict(os.environ)
    env.update(
        {
            "LIFEMANIFOLD_LLM_CALL_LOG": "1",
            "LIFEMANIFOLD_PROPOSAL_LOG": "1",
            "LIFEMANIFOLD_PROPOSAL_LOG_ALL_EMITTERS": "1",
            "LIFEMANIFOLD_LOG_ITERATION_TIMING": "1",
            "LIFEMANIFOLD_LLM_PARALLEL_WORKERS": str(args.llm_workers),
        }
    )
    for arm in order:
        out = run_dir(archive_index, continuation_index, arm)
        summary = out / "nightly_run_summary.json"
        if summary.is_file():
            print(f"skip completed {out}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        manifest = _run_manifest(
            archive_index=archive_index,
            continuation_index=continuation_index,
            arm=arm,
            order=order,
        )
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            "uv",
            "run",
            "python",
            str(RUN_SCRIPT),
            "--scheduler",
            str(_scheduler_path(arm)),
            "--output-dir",
            str(out),
            "--seed",
            str(continuation_seed(archive_index, continuation_index)),
            "--iterations",
            "650",
            "--load-archive",
            str(floor),
            "--llm-spec",
            str(FIXED_LLM_SPEC),
            "--surrogate-buffer",
            str(out / "surrogate_buffer.jsonl"),
            "--require-surrogate-quality-gate",
        ]
        _run(command, env=env)


def _index(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("index must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--write", type=Path)

    sub.add_parser("preflight")

    floor_parser = sub.add_parser("materialize-floor")
    floor_parser.add_argument("--source", type=Path, required=True)
    floor_parser.add_argument("--archive-index", type=_index, required=True)

    source_parser = sub.add_parser("run-floor-source")
    source_parser.add_argument("--archive-index", type=_index, required=True)
    source_parser.add_argument("--execute", action="store_true")
    source_parser.add_argument("--ack", default="")

    probe_parser = sub.add_parser("probe-model")
    probe_parser.add_argument("--execute", action="store_true")
    probe_parser.add_argument("--ack", default="")

    block_parser = sub.add_parser("run-block")
    block_parser.add_argument("--archive-index", type=_index, required=True)
    block_parser.add_argument("--continuation-index", type=_index, required=True)
    block_parser.add_argument("--llm-workers", type=int, default=4)
    block_parser.add_argument("--execute", action="store_true")
    block_parser.add_argument("--ack", default="")

    args = parser.parse_args(argv)
    if getattr(args, "archive_index", 0) >= ARCHIVE_COUNT:
        parser.error(f"archive-index must be 0..{ARCHIVE_COUNT - 1}")
    if getattr(args, "continuation_index", 0) >= CONTINUATIONS_PER_ARCHIVE:
        parser.error(
            "continuation-index must be " f"0..{CONTINUATIONS_PER_ARCHIVE - 1}"
        )

    if args.command == "plan":
        if args.write:
            write_plan(args.write)
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result, ready = preflight()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if ready else 2
    if args.command == "materialize-floor":
        result = materialize_floor(
            args.source,
            floor_path(args.archive_index),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-floor-source":
        run_floor_source(args)
        return 0
    if args.command == "probe-model":
        probe_model(args)
        return 0
    if args.command == "run-block":
        run_block(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
