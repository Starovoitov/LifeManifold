"""Load surrogate checkpoints and predict from world spec dicts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from dashboard.utils.bootstrap import ensure_repo_on_path
from dashboard.components.artifact_selectors import (
    format_repo_relative_path_with_symlink,
)
from dashboard.utils.config import (
    CHECKPOINT_STUB_VALUE,
    UNSET,
    UnsetType,
    active_archive_path,
    archive_adjacent_surrogate_checkpoint,
    checkpoint_session_key,
    list_surrogate_checkpoint_candidates,
    load_config,
    resolve_surrogate_checkpoint_path,
)
from dashboard.utils.surrogate_checkpoint import is_surrogate_model_checkpoint
from dashboard.utils.data_processing import world_spec_from_dict

ensure_repo_on_path()

from worldspace.illuminators.evaluation import apply_canonical_seed
from worldspace.surrogate import get_surrogate
from worldspace.surrogate.feature_extractor import FEATURE_NAMES
from worldspace.surrogate.model import SurrogateModel
from worldspace.surrogate.surrogate import StubSurrogate
from worldspace.surrogate.types import SurrogateConfig, SurrogateProtocol

__all__ = [
    "SurrogateStatus",
    "feature_importance_from_model",
    "load_surrogate",
    "predict_world_spec_dict",
    "render_surrogate_checkpoint_selector",
    "render_surrogate_status_banner",
    "resolve_checkpoint_path",
    "surrogate_model_from_handle",
    "surrogate_status",
]


@dataclass(frozen=True)
class SurrogateStatus:
    """Checkpoint load state for dashboard UI."""

    available: bool
    is_stub: bool
    checkpoint_path: Path | None
    message: str


def resolve_checkpoint_path(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> Path | None:
    """Resolve checkpoint: explicit selection, session override, then archive-local."""
    if checkpoint_path is not UNSET:
        if checkpoint_path is None:
            return None
        if not isinstance(checkpoint_path, Path):
            msg = (
                "checkpoint_path must be a pathlib.Path, None, or omitted; "
                f"got {type(checkpoint_path).__name__}"
            )
            raise TypeError(msg)
        return (
            checkpoint_path if is_surrogate_model_checkpoint(checkpoint_path) else None
        )
    config = cfg if cfg is not None else load_config()
    if archive_path is None:
        archive_path = active_archive_path(config)
    return resolve_surrogate_checkpoint_path(config, archive_path=archive_path)


def render_surrogate_checkpoint_selector(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
) -> Path | None:
    """Sidebar picker: archive-local checkpoint first, otherwise manual choice."""
    config = cfg if cfg is not None else load_config()
    if archive_path is None:
        archive_path = active_archive_path(config)
    if archive_path is None or not archive_path.is_file():
        return None

    session_key = checkpoint_session_key(archive_path)
    local_auto = archive_adjacent_surrogate_checkpoint(archive_path)
    candidates = list_surrogate_checkpoint_candidates(config, archive_path=archive_path)

    option_values: list[str] = [CHECKPOINT_STUB_VALUE]
    seen_targets: set[str] = set()
    for candidate in candidates:
        target_key = str(candidate.resolve())
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        option_values.append(str(candidate))

    def _stored_path_matches_options(stored: object) -> bool:
        if stored == CHECKPOINT_STUB_VALUE:
            return True
        if not isinstance(stored, str) or not stored.strip():
            return False
        stored_path = Path(stored)
        return any(
            stored == option or stored_path.resolve() == Path(option).resolve()
            for option in option_values
            if option != CHECKPOINT_STUB_VALUE
        )

    def _canonical_option_for_stored(stored: object) -> str:
        if stored == CHECKPOINT_STUB_VALUE:
            return CHECKPOINT_STUB_VALUE
        if not isinstance(stored, str) or not stored.strip():
            return CHECKPOINT_STUB_VALUE
        stored_path = Path(stored)
        for option in option_values:
            if option == CHECKPOINT_STUB_VALUE:
                continue
            if stored == option or stored_path.resolve() == Path(option).resolve():
                return option
        return str(local_auto) if local_auto is not None else CHECKPOINT_STUB_VALUE

    if session_key not in st.session_state:
        st.session_state[session_key] = (
            str(local_auto) if local_auto is not None else CHECKPOINT_STUB_VALUE
        )
    elif st.session_state[
        session_key
    ] != CHECKPOINT_STUB_VALUE and not _stored_path_matches_options(
        st.session_state[session_key]
    ):
        st.session_state[session_key] = (
            str(local_auto) if local_auto is not None else CHECKPOINT_STUB_VALUE
        )
    else:
        st.session_state[session_key] = _canonical_option_for_stored(
            st.session_state[session_key]
        )

    def _label(value: str) -> str:
        if value == CHECKPOINT_STUB_VALUE:
            return "Stub (no checkpoint)"
        return format_repo_relative_path_with_symlink(Path(value))

    selected_value = st.sidebar.selectbox(
        "Surrogate checkpoint",
        option_values,
        format_func=_label,
        key=session_key,
    )
    if selected_value == CHECKPOINT_STUB_VALUE:
        return None
    selected_path = Path(selected_value)
    return selected_path if is_surrogate_model_checkpoint(selected_path) else None


def load_surrogate(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> SurrogateProtocol:
    """Load cached surrogate instance (checkpoint or stub)."""
    config = cfg if cfg is not None else load_config()
    surrogate_cfg = _surrogate_config_from_dashboard(
        config,
        archive_path=archive_path,
        checkpoint_path=checkpoint_path,
    )
    return _cached_load_surrogate(
        str(surrogate_cfg.checkpoint or ""),
        surrogate_cfg.enabled,
        surrogate_cfg.model_type,
        surrogate_cfg.stub_mean,
        surrogate_cfg.stub_uncertainty,
    )


def surrogate_status(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> SurrogateStatus:
    """Describe whether a real checkpoint-backed surrogate is active."""
    config = cfg if cfg is not None else load_config()
    checkpoint = resolve_checkpoint_path(
        config,
        archive_path=archive_path,
        checkpoint_path=checkpoint_path,
    )
    surrogate = load_surrogate(
        config,
        archive_path=archive_path,
        checkpoint_path=checkpoint_path,
    )
    is_stub = isinstance(surrogate, StubSurrogate)
    if is_stub:
        if checkpoint is None:
            message = "No surrogate checkpoint found; using stub predictions."
        else:
            message = "Checkpoint could not be loaded; using stub predictions."
        return SurrogateStatus(
            available=False,
            is_stub=True,
            checkpoint_path=checkpoint,
            message=message,
        )
    return SurrogateStatus(
        available=True,
        is_stub=False,
        checkpoint_path=checkpoint,
        message=f"Loaded surrogate from {checkpoint}",
    )


def predict_world_spec_dict(
    world_spec: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> dict[str, float] | None:
    """Predict fitness and uncertainty for one JSON-like world spec.

    Returns None when the spec cannot be parsed or prediction fails, so batch
    callers (e.g. ``build_prediction_frame``) can skip individual elites.
    """
    try:
        spec = world_spec_from_dict(world_spec)
        apply_canonical_seed(spec)
        prediction = load_surrogate(
            cfg,
            archive_path=archive_path,
            checkpoint_path=checkpoint_path,
        ).predict(spec)
    except (TypeError, ValueError, KeyError, OSError):
        return None
    return {
        "fitness": float(prediction.fitness),
        "uncertainty": float(prediction.uncertainty),
    }


def render_surrogate_status_banner(
    cfg: dict[str, Any] | None = None,
    *,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> SurrogateStatus:
    """Show info/warning banner for surrogate availability."""
    status = surrogate_status(
        cfg,
        archive_path=archive_path,
        checkpoint_path=checkpoint_path,
    )
    if status.is_stub:
        st.info(status.message)
    else:
        st.success(status.message)
    return status


def surrogate_model_from_handle(
    surrogate: SurrogateProtocol,
) -> SurrogateModel | None:
    """Return underlying ``SurrogateModel`` when the handle is a facade."""
    model = getattr(surrogate, "model", None)
    return model if isinstance(model, SurrogateModel) else None


def feature_importance_from_model(model: SurrogateModel) -> dict[str, float] | None:
    """Aggregate LightGBM ``feature_importances_`` when present."""
    if model.model_type != "lightgbm" or not model._uses_lightgbm:
        return None
    if not model._ensemble:
        return None

    total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    count = 0
    for estimators in model._ensemble.values():
        for estimator in estimators:
            raw = getattr(estimator, "feature_importances_", None)
            if raw is None:
                continue
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            if values.size != len(FEATURE_NAMES):
                continue
            total += values
            count += 1
    if count == 0:
        return None
    averaged = total / float(count)
    return {name: float(averaged[index]) for index, name in enumerate(FEATURE_NAMES)}


def _surrogate_config_from_dashboard(
    cfg: dict[str, Any],
    *,
    archive_path: Path | None = None,
    checkpoint_path: Path | None | UnsetType = UNSET,
) -> SurrogateConfig:
    surrogate_section = cfg.get("surrogate")
    block = surrogate_section if isinstance(surrogate_section, dict) else {}
    checkpoint = resolve_checkpoint_path(
        cfg,
        archive_path=archive_path,
        checkpoint_path=checkpoint_path,
    )
    raw_model_type = str(block.get("model_type", "mlp")).strip().lower()
    model_type = raw_model_type if raw_model_type in ("mlp", "lightgbm") else "mlp"
    return SurrogateConfig(
        enabled=bool(block.get("enabled", True)),
        model_type=model_type,  # type: ignore[arg-type]
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        stub_mean=float(block.get("stub_mean", 0.5)),
        stub_uncertainty=float(block.get("stub_uncertainty", 0.85)),
    )


@st.cache_resource(show_spinner=False)
def _cached_load_surrogate(
    checkpoint_str: str,
    enabled: bool,
    model_type: str,
    stub_mean: float,
    stub_uncertainty: float,
) -> SurrogateProtocol:
    config = SurrogateConfig(
        enabled=enabled,
        model_type=(
            model_type if model_type in ("mlp", "lightgbm") else "mlp"  # type: ignore[arg-type]
        ),
        checkpoint=checkpoint_str or None,
        stub_mean=stub_mean,
        stub_uncertainty=stub_uncertainty,
    )
    return get_surrogate(config)
