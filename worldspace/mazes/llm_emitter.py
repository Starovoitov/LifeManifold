"""Strict JSON LLM emitter for the maze domain."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Lock

import numpy as np
from pydantic import ValidationError

from worldspace.generators import llm_retry_backoff_seconds
from worldspace.generators.llm_config import (
    LlmTextCaller,
    load_llm_config,
)
from worldspace.mazes.archive import MazeArchive
from worldspace.mazes.emitters import (
    MazeEmitterResult,
    MazeTarget,
    emit_genetic,
)
from worldspace.mazes.evaluation import shortest_path_length, shortest_path_positions
from worldspace.mazes.spec import FLOOR, WALL, MazeSpec
from worldspace.mazes.surrogate import MazePrediction

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = _ROOT / "prompts/maze_llm_emitter_system.txt"
DEFAULT_USER_PROMPT = _ROOT / "prompts/maze_llm_emitter_user.txt"
DEFAULT_LLM_SPEC = _ROOT / "worldspace/specs/llm_world_generator_qwen.yaml"


@dataclass
class MazeLlmAudit:
    attempts: int = 0
    api_calls: int = 0
    retries: int = 0
    parse_successes: int = 0
    fallbacks: int = 0
    repaired_outputs: int = 0
    repair_collapses: int = 0
    zero_distance: int = 0
    total_tile_distance: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    invalid_response_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "api_calls": self.api_calls,
            "retries": self.retries,
            "parse_successes": self.parse_successes,
            "parse_success_rate": (
                self.parse_successes / self.attempts if self.attempts else 0.0
            ),
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallbacks / self.attempts if self.attempts else 0.0,
            "zero_distance": self.zero_distance,
            "repaired_outputs": self.repaired_outputs,
            "repair_rate": (
                self.repaired_outputs / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
            "repair_collapses": self.repair_collapses,
            "repair_collapse_rate": (
                self.repair_collapses / self.attempts if self.attempts else 0.0
            ),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "invalid_response_reasons": dict(
                sorted(self.invalid_response_reasons.items())
            ),
            "mean_tile_distance": (
                self.total_tile_distance / self.parse_successes
                if self.parse_successes
                else 0.0
            ),
        }


class MazeLlmEmitter:
    def __init__(
        self,
        *,
        prompt_mode: str,
        llm_spec_path: Path = DEFAULT_LLM_SPEC,
        call_llm_text: LlmTextCaller | None = None,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        max_retries: int = 2,
    ) -> None:
        if prompt_mode not in ("stub", "hints"):
            raise ValueError("prompt_mode must be stub or hints")
        self.prompt_mode = prompt_mode
        self.config = load_llm_config(llm_spec_path)
        self.call_llm_text = call_llm_text
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")
        self.user_prompt = user_prompt_path.read_text(encoding="utf-8")
        self.max_retries = max(0, max_retries)
        self.audit = MazeLlmAudit()
        self._audit_lock = Lock()
        self.prompt_version = _prompt_hash(self.system_prompt, self.user_prompt)

    def emit(
        self,
        *,
        target: MazeTarget,
        archive: MazeArchive,
        rng: np.random.Generator,
        prediction: MazePrediction | None,
    ) -> MazeEmitterResult:
        if target.parent is None:
            return emit_genetic(target, archive, rng)
        parent = target.parent.spec
        effective = (
            prediction
            if self.prompt_mode == "hints" and prediction is not None
            else MazePrediction({}, {"path_length": 0.5, "branching": 0.5}, 0.5, 1.0)
        )
        prompt = self.user_prompt.format(
            target_path=target.center[0],
            target_branching=target.center[1],
            surrogate_fitness=effective.fitness,
            surrogate_uncertainty=effective.uncertainty,
            surrogate_path=effective.measures.get("path_length", 0.5),
            surrogate_branching=effective.measures.get("branching", 0.5),
            parent_json=parent.canonical_json(),
        )
        with self._audit_lock:
            self.audit.attempts += 1
        child: MazeSpec | None = None
        distance = 0
        repaired = False
        last_reason = "unknown"
        for request_index in range(self.max_retries + 1):
            if request_index:
                with self._audit_lock:
                    self.audit.retries += 1
                time.sleep(llm_retry_backoff_seconds(request_index))
            try:
                with self._audit_lock:
                    self.audit.api_calls += 1
                request_prompt = (
                    prompt
                    if request_index == 0
                    else prompt + "\n\n" + _retry_guidance(parent)
                )
                response = self._request(request_prompt)
                proposed = parse_maze_response(response, parent=parent)
                distance = tile_distance(parent, proposed)
                if distance == 0:
                    raise ValueError("unchanged")
                coerced, was_repaired = coerce_solvable_mutation(parent, proposed)
                if coerced is None:
                    raise ValueError("unsolvable")
                proposed = coerced
                distance = tile_distance(parent, proposed)
                repaired = was_repaired
                child = proposed
                break
            except (ValueError, ValidationError, json.JSONDecodeError) as error:
                last_reason = _failure_reason(error)
                _record_invalid_response(self, last_reason)
            except RuntimeError as error:
                if "LLM request failed" not in str(error):
                    raise
                last_reason = "network"
                _record_invalid_response(self, last_reason)
        if child is None:
            with self._audit_lock:
                self.audit.fallbacks += 1
                self.audit.failure_reasons[last_reason] = (
                    self.audit.failure_reasons.get(last_reason, 0) + 1
                )
                if last_reason == "unsolvable":
                    self.audit.repair_collapses += 1
            fallback = emit_genetic(target, archive, rng)
            return MazeEmitterResult(
                spec=fallback.spec,
                parent_id=fallback.parent_id,
                emitter_type="llm_fallback_genetic",
            )
        with self._audit_lock:
            self.audit.parse_successes += 1
            self.audit.repaired_outputs += repaired
            self.audit.total_tile_distance += distance
            self.audit.zero_distance += distance == 0
        return MazeEmitterResult(
            spec=child,
            parent_id=target.parent.candidate_id,
            emitter_type="llm",
        )

    def emit_batch(
        self,
        jobs: list[
            tuple[
                MazeTarget,
                MazeArchive,
                np.random.Generator,
                MazePrediction | None,
            ]
        ],
        *,
        max_workers: int = 4,
    ) -> list[MazeEmitterResult]:
        """Issue independent LLM slots concurrently and preserve slot order."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda job: self.emit(
                        target=job[0],
                        archive=job[1],
                        rng=job[2],
                        prediction=job[3],
                    ),
                    jobs,
                )
            )

    def _request(self, prompt: str) -> str:
        if self.call_llm_text is None:
            from worldspace.generators import call_llm

            caller: LlmTextCaller = call_llm
        else:
            caller = self.call_llm_text
        return caller(
            mode=self.config.mode,
            provider_name=self.config.active_provider,
            providers=self.config.providers,
            prompt=prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max(self.config.max_tokens, 500),
            system_content=self.system_prompt,
        )


def parse_maze_response(
    response: str,
    *,
    parent: MazeSpec | None = None,
) -> MazeSpec:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", response, 0)
    payload = json.loads(response[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    if set(payload) == {"rows"}:
        return MazeSpec.model_validate(payload)
    if set(payload) != {"edits"} or parent is None:
        raise ValueError("response must contain exactly edits")
    edits = payload["edits"]
    if not isinstance(edits, list) or not 4 <= len(edits) <= 20:
        raise ValueError("edits must contain 4–20 items")
    grid = [list(row) for row in parent.rows]
    seen: set[tuple[int, int]] = set()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"row", "col", "tile"}:
            raise ValueError("each edit requires row, col, and tile")
        row, column = int(edit["row"]), int(edit["col"])
        tile = str(edit["tile"])
        if not (1 <= row <= 14 and 1 <= column <= 14):
            raise ValueError("edit coordinates must be interior")
        if (row, column) in seen:
            raise ValueError("edit coordinates must be unique")
        if parent.rows[row][column] in ("S", "G"):
            raise ValueError("cannot edit start or goal tiles")
        if tile not in (WALL, FLOOR):
            raise ValueError("edit tile must be # or .")
        seen.add((row, column))
        grid[row][column] = tile
    return MazeSpec(rows=tuple("".join(row) for row in grid))


def tile_distance(first: MazeSpec, second: MazeSpec) -> int:
    return sum(
        left != right
        for first_row, second_row in zip(first.rows, second.rows, strict=True)
        for left, right in zip(first_row, second_row, strict=True)
    )


def repair_solvable_mutation(
    parent: MazeSpec,
    proposed: MazeSpec,
) -> MazeSpec | None:
    """Revert the smallest deterministic edit suffix until a path is restored."""
    changed = _changed_positions(parent, proposed)
    wall_first = sorted(
        changed,
        key=lambda position: proposed.rows[position[0]][position[1]] != WALL,
    )
    grid = [list(row) for row in proposed.rows]
    for row, column in wall_first:
        grid[row][column] = parent.rows[row][column]
        candidate = MazeSpec(rows=tuple("".join(item) for item in grid))
        if (
            tile_distance(parent, candidate) > 0
            and shortest_path_length(candidate) is not None
        ):
            return candidate
    return None


def coerce_solvable_mutation(
    parent: MazeSpec,
    proposed: MazeSpec,
) -> tuple[MazeSpec | None, bool]:
    """Return a solvable nonzero-distance child, repairing if needed."""
    if (
        tile_distance(parent, proposed) > 0
        and shortest_path_length(proposed) is not None
    ):
        return proposed, False
    filtered = filter_edits_preserving_solvability(parent, proposed)
    if filtered is not None:
        return filtered, True
    repaired = repair_solvable_mutation(parent, proposed)
    if repaired is not None:
        return repaired, True
    return None, False


def filter_edits_preserving_solvability(
    parent: MazeSpec,
    proposed: MazeSpec,
) -> MazeSpec | None:
    """Apply parent→proposed edits one at a time, keeping solvable prefixes."""
    changed = _changed_positions(parent, proposed)
    if not changed:
        return None
    apply_first = sorted(
        changed,
        key=lambda position: proposed.rows[position[0]][position[1]] == FLOOR,
        reverse=True,
    )
    grid = [list(row) for row in parent.rows]
    applied = False
    for row, column in apply_first:
        previous = grid[row][column]
        grid[row][column] = proposed.rows[row][column]
        candidate = MazeSpec(rows=tuple("".join(item) for item in grid))
        if (
            tile_distance(parent, candidate) > 0
            and shortest_path_length(candidate) is not None
        ):
            applied = True
            continue
        grid[row][column] = previous
    if not applied:
        return None
    return MazeSpec(rows=tuple("".join(item) for item in grid))


def _changed_positions(parent: MazeSpec, proposed: MazeSpec) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(1, parent.SIZE - 1)
        for column in range(1, parent.SIZE - 1)
        if parent.rows[row][column] != proposed.rows[row][column]
    ]


def _record_invalid_response(emitter: MazeLlmEmitter, reason: str) -> None:
    with emitter._audit_lock:
        emitter.audit.invalid_response_reasons[reason] = (
            emitter.audit.invalid_response_reasons.get(reason, 0) + 1
        )


def _failure_reason(error: Exception) -> str:
    message = str(error).lower()
    if "unsolvable" in message:
        return "unsolvable"
    if "unchanged" in message:
        return "unchanged"
    if isinstance(error, json.JSONDecodeError):
        return "json_decode"
    if isinstance(error, ValidationError):
        return "schema"
    return "payload"


def _retry_guidance(parent: MazeSpec) -> str:
    suggestions: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    path = shortest_path_positions(parent)
    if path is not None:
        for row, column in path:
            for neighbor in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if not (1 <= neighbor[0] <= 14 and 1 <= neighbor[1] <= 14):
                    continue
                if neighbor in seen:
                    continue
                current = parent.rows[neighbor[0]][neighbor[1]]
                if current not in (WALL, FLOOR):
                    continue
                if current == FLOOR:
                    continue
                seen.add(neighbor)
                suggestions.append(
                    {
                        "row": neighbor[0],
                        "col": neighbor[1],
                        "current": current,
                        "required_new_tile": FLOOR,
                    }
                )
                if len(suggestions) >= 8:
                    break
            if len(suggestions) >= 8:
                break
    if len(suggestions) < 4:
        for row in range(1, parent.SIZE - 1):
            for column in range(1, parent.SIZE - 1):
                if (row, column) in seen:
                    continue
                current = parent.rows[row][column]
                if current not in (WALL, FLOOR):
                    continue
                new_tile = FLOOR if current == WALL else WALL
                seen.add((row, column))
                suggestions.append(
                    {
                        "row": row,
                        "col": column,
                        "current": current,
                        "required_new_tile": new_tile,
                    }
                )
                if len(suggestions) >= 8:
                    break
            if len(suggestions) >= 8:
                break
    return (
        "Your previous response was invalid or made no effective change. "
        "Return exactly four unique edits selected from this safe list. For each, "
        "copy row and col and put required_new_tile in the JSON field named tile "
        "(never return current): " + json.dumps(suggestions, separators=(",", ":"))
    )


def parse_maze_responses_ordered(
    responses: list[str],
    parents: list[MazeSpec],
    *,
    max_workers: int = 4,
) -> list[MazeSpec]:
    """Parse independent responses in parallel while preserving slot order."""
    if len(responses) != len(parents):
        raise ValueError("responses and parents must have equal length")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda pair: parse_maze_response(pair[0], parent=pair[1]),
                zip(responses, parents, strict=True),
            )
        )


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()[:16]
