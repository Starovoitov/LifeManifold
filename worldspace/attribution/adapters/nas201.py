"""Read-only normalizer for native NAS-Bench-201 confirmatory runs."""

from __future__ import annotations

from worldspace.attribution.adapters.base import NativeRunInputs, NormalizedRunBundle
from worldspace.attribution.adapters.public import normalize_public_run
from worldspace.attribution.capabilities import nas201_capabilities
from worldspace.attribution.manifest import RunManifest
from worldspace.nas201.runner import ARCHIVE_FILENAME


class Nas201NormalizationAdapter:
    """Normalize NAS public-qd artifacts without rewriting the run directory."""

    def capabilities(self):
        return nas201_capabilities()

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
