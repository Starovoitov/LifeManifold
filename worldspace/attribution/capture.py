"""Opt-in prospective capture of normalized proposal-slot events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldspace.attribution.manifest import RunManifest
from worldspace.attribution.records import ArchiveState, ProposalEvent

PROSPECTIVE_EVENT_FILENAME = "attribution_events.jsonl"


@dataclass
class ProspectiveEventCapture:
    """Validate and append complete normalized slot events for one run."""

    manifest: RunManifest
    path: Path | None = None
    _events: list[ProposalEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.path is not None and self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(
                f"prospective event log already exists and is non-empty: {self.path}"
            )

    @property
    def events(self) -> tuple[ProposalEvent, ...]:
        return tuple(self._events)

    def append_slot(
        self,
        *,
        iteration: int,
        slot: int,
        configured_operator: str,
        realized_operator: str | None,
        target_cell_id: str | None,
        parent_id: str | None,
        parent_genotype_hash: str | None,
        candidate_id: str | None,
        candidate_genotype_hash: str | None,
        before: ArchiveState,
        generation: dict[str, Any],
        gate: dict[str, Any],
        evaluation: dict[str, Any],
        resources: dict[str, Any],
        after: ArchiveState,
    ) -> ProposalEvent:
        """Append exactly one slot in deterministic proposal order."""
        event = ProposalEvent.model_validate(
            {
                "run_id": self.manifest.run_id,
                "study_id": self.manifest.study_id,
                "arm_id": self.manifest.arm_id,
                "pair_id": self.manifest.pair_id,
                "proposal_index": len(self._events) + 1,
                "iteration": iteration,
                "slot": slot,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configured_operator": configured_operator,
                "realized_operator": realized_operator,
                "target_cell_id": target_cell_id,
                "parent_id": parent_id,
                "parent_genotype_hash": parent_genotype_hash,
                "candidate_id": candidate_id,
                "candidate_genotype_hash": candidate_genotype_hash,
                "before": before,
                "generation": generation,
                "gate": gate,
                "evaluation": evaluation,
                "resources": resources,
                "after": after,
            }
        )
        if self._events and event.before != self._events[-1].after:
            raise ValueError(
                "prospective event before-state does not match previous after-state"
            )
        self._events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        return event


def read_prospective_events(
    path: Path,
    *,
    manifest: RunManifest,
) -> tuple[ProposalEvent, ...]:
    """Load a complete event sidecar and verify run identity and order."""
    events: list[ProposalEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = ProposalEvent.model_validate_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prospective event at {path}:{line_number}: {exc}"
                ) from exc
            expected_index = len(events) + 1
            if event.proposal_index != expected_index:
                raise ValueError(
                    f"prospective event index {event.proposal_index} "
                    f"does not match expected {expected_index}"
                )
            identities = (
                ("run_id", event.run_id, manifest.run_id),
                ("study_id", event.study_id, manifest.study_id),
                ("arm_id", event.arm_id, manifest.arm_id),
                ("pair_id", event.pair_id, manifest.pair_id),
            )
            for field_name, actual, expected in identities:
                if actual != expected:
                    raise ValueError(
                        f"prospective event {field_name}={actual!r} "
                        f"does not match manifest {expected!r}"
                    )
            if events and event.before != events[-1].after:
                raise ValueError(
                    "prospective event before-state does not match previous after-state"
                )
            events.append(event)
    return tuple(events)


def archive_state_from_archive(archive: Any) -> ArchiveState:
    """Snapshot common archive metrics from a native archive protocol."""
    fitnesses = [
        float(elite.fitness)
        for cell_id in range(archive.n_cells)
        if (elite := archive.get_cell(cell_id)) is not None
    ]
    occupied = len(fitnesses)
    raw_qd = sum(fitnesses)
    return ArchiveState(
        occupied_cells=occupied,
        capacity=int(archive.n_cells),
        coverage=occupied / archive.n_cells,
        raw_qd_score=raw_qd,
        normalized_qd_score=raw_qd / archive.n_cells,
        maximum_elite_quality=max(fitnesses) if fitnesses else None,
        occupied_mean_quality=raw_qd / occupied if occupied else None,
    )
