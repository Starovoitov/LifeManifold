"""Read-only normalizer for native PCG sokoban-v0 confirmatory runs."""

from __future__ import annotations

from worldspace.attribution.adapters.base import NativeRunInputs, NormalizedRunBundle
from worldspace.attribution.adapters.public import normalize_public_run
from worldspace.attribution.capabilities import pcg_sokoban_capabilities
from worldspace.attribution.manifest import RunManifest
from worldspace.pcg.runner import ARCHIVE_FILENAME


class PcgSokobanNormalizationAdapter:
    """Normalize PCG public-qd artifacts without rewriting the run directory."""

    def capabilities(self):
        return pcg_sokoban_capabilities()

    def normalize(
        self,
        manifest: RunManifest,
        inputs: NativeRunInputs,
    ) -> NormalizedRunBundle:
        return normalize_public_run(
            manifest,
            inputs,
            capabilities=self.capabilities(),
            archive_filename=ARCHIVE_FILENAME,
        )
