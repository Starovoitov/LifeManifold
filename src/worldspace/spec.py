from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WorldSpec:
    """JSON-serializable specification of one cellular-automata world."""

    birth: list[int]
    survival: list[int]
    noise: float
    resource_regen: float
    predation: float
    cell_types: list[str]
    neighborhood: str = "moore"
    grid_size: int = 50
    steps: int = 300
    seed: int = 0

    def to_json_dict(self) -> dict:
        """Return the world spec as a plain dictionary."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Persist the world spec to a JSON file."""
        target = Path(path)
        target.write_text(json.dumps(self.to_json_dict(), ensure_ascii=True, indent=2))

    @classmethod
    def from_json_dict(cls, data: dict) -> "WorldSpec":
        """Construct a world spec from a JSON-like dictionary."""
        return cls(**data)

    @classmethod
    def load_json(cls, path: str | Path) -> "WorldSpec":
        """Load a world spec from a JSON file."""
        src = Path(path)
        return cls.from_json_dict(json.loads(src.read_text()))
