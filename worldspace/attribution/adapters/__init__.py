"""Read-only native artifact normalizers."""

from worldspace.attribution.adapters.base import (
    NativeRunInputs,
    NormalizationError,
    NormalizationIssue,
    NormalizedRunBundle,
    ReadOnlyNormalizationAdapter,
)
from worldspace.attribution.adapters.ca import CaNormalizationAdapter
from worldspace.attribution.adapters.maze import MazeNormalizationAdapter

__all__ = [
    "CaNormalizationAdapter",
    "MazeNormalizationAdapter",
    "NativeRunInputs",
    "NormalizationError",
    "NormalizationIssue",
    "NormalizedRunBundle",
    "ReadOnlyNormalizationAdapter",
]
