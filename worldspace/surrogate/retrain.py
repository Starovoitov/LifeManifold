"""In-run nested surrogate retrain at MAP-Elites iteration boundaries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from worldspace.illuminators.scheduler import (
    SchedulerConfig,
    surrogate_config_from_scheduler,
)
from worldspace.surrogate.buffer import count_buffer_rows
from worldspace.surrogate.checkpoint_io import CHECKPOINT_LOAD_ERRORS
from worldspace.surrogate.checkpoint_paths import resolve_runtime_checkpoint_path
from worldspace.surrogate.training_runtime import TrainResult, train_from_buffer
from worldspace.surrogate.types import SurrogateProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "RetrainOutcome",
    "RetrainState",
    "is_retrain_iteration",
    "maybe_retrain_after_iteration",
]


@dataclass
class RetrainState:
    """Mutable counters for nested retrain within one illuminator run."""

    buffer_row_count_at_last_retrain: int = 0


@dataclass(frozen=True)
class RetrainOutcome:
    """Result of one retrain hook invocation."""

    status: str
    iteration_index: int
    new_buffer_rows: int = 0
    train_result: TrainResult | None = None


def is_retrain_iteration(iteration_index: int, every_iterations: int) -> bool:
    """Return whether ``iteration_index`` (1-based) triggers a retrain check."""
    if every_iterations < 1:
        msg = f"every_iterations must be >= 1, got {every_iterations}"
        raise ValueError(msg)
    return iteration_index > 0 and iteration_index % every_iterations == 0


def maybe_retrain_after_iteration(
    config: SchedulerConfig,
    *,
    iteration_index: int,
    state: RetrainState,
    surrogate: SurrogateProtocol,
) -> RetrainOutcome:
    """Run nested retrain after one iteration when configured and guards pass."""
    retrain = config.retrain
    if not retrain.enabled:
        return RetrainOutcome(status="disabled", iteration_index=iteration_index)

    if not is_retrain_iteration(iteration_index, retrain.every_iterations):
        return RetrainOutcome(status="not_scheduled", iteration_index=iteration_index)

    buffer_path = Path(config.surrogate_buffer_path).expanduser()
    current_rows = count_buffer_rows(buffer_path)
    new_rows = current_rows - state.buffer_row_count_at_last_retrain
    if new_rows < retrain.min_new_buffer_rows:
        logger.debug(
            "retrain_skipped_insufficient_buffer_rows iteration=%s new_rows=%s "
            "required=%s",
            iteration_index,
            new_rows,
            retrain.min_new_buffer_rows,
        )
        return RetrainOutcome(
            status="skipped_insufficient_buffer_rows",
            iteration_index=iteration_index,
            new_buffer_rows=new_rows,
        )

    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path is None:
        logger.warning(
            "retrain_skipped_no_checkpoint_path iteration=%s",
            iteration_index,
        )
        return RetrainOutcome(
            status="skipped_no_checkpoint",
            iteration_index=iteration_index,
            new_buffer_rows=new_rows,
        )

    from worldspace.surrogate.surrogate import SurrogateFacade

    if not isinstance(surrogate, SurrogateFacade):
        logger.warning(
            "retrain_skipped_not_facade iteration=%s surrogate_type=%s",
            iteration_index,
            type(surrogate).__name__,
        )
        return RetrainOutcome(
            status="skipped_not_facade",
            iteration_index=iteration_index,
            new_buffer_rows=new_rows,
        )

    surrogate_config = surrogate_config_from_scheduler(config)
    logger.info(
        "retrain_start iteration=%s buffer_path=%s new_rows=%s checkpoint=%s",
        iteration_index,
        buffer_path,
        new_rows,
        checkpoint_path,
    )
    started = time.perf_counter()
    train_result = train_from_buffer(
        buffer_path=buffer_path,
        checkpoint_path=checkpoint_path,
        model_type=surrogate_config.model_type,
        require_quality_gate=True,
    )
    elapsed = time.perf_counter() - started

    if not train_result.success:
        logger.warning(
            "retrain_end status=failed iteration=%s elapsed_s=%.2f error=%s",
            iteration_index,
            elapsed,
            train_result.error_message,
        )
        return RetrainOutcome(
            status="train_failed",
            iteration_index=iteration_index,
            new_buffer_rows=new_rows,
            train_result=train_result,
        )

    try:
        surrogate.reload(checkpoint_path)
    except CHECKPOINT_LOAD_ERRORS as exc:
        logger.warning(
            "retrain_end status=reload_failed iteration=%s elapsed_s=%.2f error=%s",
            iteration_index,
            elapsed,
            exc,
        )
        return RetrainOutcome(
            status="reload_failed",
            iteration_index=iteration_index,
            new_buffer_rows=new_rows,
            train_result=train_result,
        )

    state.buffer_row_count_at_last_retrain = current_rows
    logger.info(
        "retrain_end status=success iteration=%s elapsed_s=%.2f "
        "samples=%s quality_passed=%s",
        iteration_index,
        elapsed,
        train_result.sample_count,
        train_result.quality_passed,
    )
    return RetrainOutcome(
        status="success",
        iteration_index=iteration_index,
        new_buffer_rows=new_rows,
        train_result=train_result,
    )


def _resolve_checkpoint_path(config: SchedulerConfig) -> Path | None:
    return resolve_runtime_checkpoint_path(config.surrogate_checkpoint)
