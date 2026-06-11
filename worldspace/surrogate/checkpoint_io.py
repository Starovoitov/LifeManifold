"""Load and save surrogate model checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

from worldspace.surrogate.model import SurrogateModel

CHECKPOINT_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    TypeError,
    ValueError,
    EOFError,
    AttributeError,
    ImportError,
    pickle.UnpicklingError,
)

__all__ = [
    "CHECKPOINT_LOAD_ERRORS",
    "load_surrogate_checkpoint",
    "save_surrogate_checkpoint",
]


def load_surrogate_checkpoint(path: Path) -> SurrogateModel:
    """Load a pickled ``SurrogateModel`` from disk."""
    with path.expanduser().open("rb") as fh:
        loaded = pickle.load(fh)
    if not isinstance(loaded, SurrogateModel):
        msg = f"Checkpoint must contain SurrogateModel, got {type(loaded)!r}"
        raise TypeError(msg)
    loaded.ensure_legacy_checkpoint_fields()
    return loaded


def save_surrogate_checkpoint(model: SurrogateModel, path: Path) -> None:
    """Persist a trained surrogate model."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
