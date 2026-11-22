"""Build the Phase 4 public release tree. Does not copy raw prompts or responses."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldspace.attribution.release_privacy import sanitize_llm_call_record

PUBLIC_RUN_FILES = (
    "run_manifest.json",
    "nightly_run_summary.json",
    "archive_trace.jsonl",
    "nas201_archive.jsonl",
    "pcg_archive.jsonl",
)
STUDY_FILES = (
    "job_plan.json",
    "design_matrix.json",
)
PROMPT_FILES = (
    "prompts/nas201_llm_emitter_system.txt",
    "prompts/nas201_llm_emitter_user.txt",
    "prompts/nas201_llm_emitter_live_user.txt",
    "prompts/pcg_sokoban_llm_emitter_system.txt",
    "prompts/pcg_sokoban_llm_emitter_user.txt",
    "prompts/pcg_sokoban_llm_emitter_live_user.txt",
)
FORBIDDEN = ("response_content", "sk-", "Bearer ", "api_key")


def main() -> None:
    source = _ROOT / "artifacts" / "phase4"
    dest = source / "release"
    counts = build_release(source, dest, repo_root=_ROOT)
    print(json.dumps(counts, sort_keys=True))


def build_release(source: Path, dest: Path, *, repo_root: Path) -> dict[str, int]:
    """Write a public copy of Phase 4 under ``dest``. Refuses private payloads."""
    if dest.exists():
        raise FileExistsError(f"release directory already exists: {dest}")
    dest.mkdir(parents=True)
    counts = {"runs": 0, "public_files": 0, "sanitized_calls": 0, "prompts": 0}
    summaries = [
        path
        for path in source.rglob("nightly_run_summary.json")
        if not _is_inside(path, dest)
    ]
    for summary_path in summaries:
        run_dir = summary_path.parent
        target = dest / "runs" / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        template = summary.get("prompt_template_sha256")
        if not isinstance(template, str):
            template = None
        for name in PUBLIC_RUN_FILES:
            src = run_dir / name
            if src.is_file():
                _copy_public(src, target / name)
                counts["public_files"] += 1
        log_path = run_dir / "llm_call_log.jsonl"
        if log_path.is_file():
            written = _write_sanitized(log_path, target / "sanitized_llm_calls.jsonl", template)
            counts["sanitized_calls"] += written
        counts["runs"] += 1
    study_paths = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and not _is_inside(path, dest)
        and (
            path.name in STUDY_FILES
            or path.name.endswith("_inspection.json")
            or path.name.endswith("_freeze.json")
        )
    ]
    for path in study_paths:
        target = dest / "studies" / path.parent.name / path.name
        _copy_public(path, target)
        counts["public_files"] += 1
    for name in (
        "PHASE4_REPORTING.md",
        "section8_reporting_bundle.json",
    ):
        src = source / name
        if src.is_file():
            _copy_public(src, dest / name)
            counts["public_files"] += 1
    prompt_rows = []
    for relative in PROMPT_FILES:
        src = repo_root / relative
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        target = dest / relative
        _copy_public(src, target)
        prompt_rows.append({"path": relative, "sha256": digest})
        counts["prompts"] += 1
    (dest / "prompt_sha256.json").write_text(
        json.dumps(prompt_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    note = {
        "protocol_section": "11",
        "public": [
            "run manifests",
            "nightly summaries",
            "archive genotypes and PCG grids",
            "proposal traces",
            "sanitized call metadata",
            "prompt template text",
        ],
        "private_omitted": [
            "llm_call_log.jsonl",
            "rendered messages",
            "response text",
            "api keys",
        ],
        "counts": counts,
    }
    (dest / "RELEASE_BOUNDARY.json").write_text(
        json.dumps(note, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_sanitized(source: Path, dest: Path, template: str | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open(encoding="utf-8") as handle, dest.open("w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"call log row is not an object: {source}")
            sanitized = sanitize_llm_call_record(raw, prompt_template_sha256=template)
            text = json.dumps(sanitized, ensure_ascii=True, sort_keys=True)
            _refuse_private(text, source)
            out.write(text + "\n")
            written += 1
    return written


def _copy_public(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle, dest.open("w", encoding="utf-8") as out:
            for line in handle:
                _refuse_private(line, source)
                out.write(line)
        return
    text = source.read_text(encoding="utf-8")
    _refuse_private(text, source)
    dest.write_text(text, encoding="utf-8")


def _refuse_private(text: str, source: Path) -> None:
    for marker in FORBIDDEN:
        if marker in text:
            raise ValueError(f"{source} contains private marker {marker!r}")


if __name__ == "__main__":
    main()
