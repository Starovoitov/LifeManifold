"""Canonical NAS-Bench-201 cell genotype (6-edge DAG, 5 operations)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

OpName = Literal[
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]

OPERATIONS: tuple[OpName, ...] = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)
OPERATION_SET = frozenset(OPERATIONS)
N_EDGES = 6

# Official NAS-Bench-201 string: node-1 ←0; node-2 ←0,1; node-3 ←0,1,2.
_ARCH_PATTERN = re.compile(
    r"^\|"
    r"(?P<e0>[^~|]+)~0\|"
    r"\+\|"
    r"(?P<e1>[^~|]+)~0\|(?P<e2>[^~|]+)~1\|"
    r"\+\|"
    r"(?P<e3>[^~|]+)~0\|(?P<e4>[^~|]+)~1\|(?P<e5>[^~|]+)~2\|"
    r"$"
)


class Nas201Spec(BaseModel):
    """Immutable 6-tuple of operations; hash is SHA-256 of the official arch string."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ops: tuple[OpName, OpName, OpName, OpName, OpName, OpName]

    @field_validator("ops", mode="before")
    @classmethod
    def _ops_tuple(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("ops must be a list or tuple of six operation names")
        return tuple(value)

    @field_validator("ops")
    @classmethod
    def _six_known_ops(
        cls, value: tuple[OpName, ...]
    ) -> tuple[OpName, OpName, OpName, OpName, OpName, OpName]:
        if len(value) != N_EDGES:
            raise ValueError(f"ops must have length {N_EDGES}")
        unknown = [op for op in value if op not in OPERATION_SET]
        if unknown:
            raise ValueError(f"unknown NAS-Bench-201 operations: {unknown}")
        return value  # type: ignore[return-value]

    @property
    def arch_str(self) -> str:
        ops = self.ops
        return (
            f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|"
            f"+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
        )

    def canonical_json(self) -> str:
        return json.dumps(
            {"ops": list(self.ops)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def genotype_sha256(self) -> str:
        return hashlib.sha256(self.arch_str.encode("utf-8")).hexdigest()

    def candidate_hash(self) -> str:
        return self.genotype_sha256()[:16]

    def to_json_dict(self) -> dict[str, object]:
        return {"ops": list(self.ops)}

    @classmethod
    def from_arch_str(cls, arch_str: str) -> Nas201Spec:
        parsed = try_parse_arch_str(arch_str)
        if parsed is None:
            raise ValueError(f"not a NAS-Bench-201 architecture string: {arch_str!r}")
        return parsed

    @classmethod
    def from_ops_json(cls, payload: object) -> Nas201Spec:
        if not isinstance(payload, dict) or "ops" not in payload:
            raise ValueError("JSON genotype must be an object with an ops array")
        return cls(ops=payload["ops"])


def try_parse_arch_str(arch_str: object) -> Nas201Spec | None:
    """Return a spec or None; never queries the lookup table."""
    if not isinstance(arch_str, str):
        return None
    match = _ARCH_PATTERN.fullmatch(arch_str)
    if match is None:
        return None
    ops = tuple(match.group(name) for name in ("e0", "e1", "e2", "e3", "e4", "e5"))
    if any(op not in OPERATION_SET for op in ops):
        return None
    return Nas201Spec(ops=ops)  # type: ignore[arg-type]


def try_parse_ops_payload(payload: object) -> Nas201Spec | None:
    """Parse LLM-style JSON ``{\"ops\": [...]}`` without contacting the table."""
    if not isinstance(payload, dict):
        return None
    ops = payload.get("ops")
    if not isinstance(ops, (list, tuple)) or len(ops) != N_EDGES:
        return None
    if any(op not in OPERATION_SET for op in ops):
        return None
    return Nas201Spec(ops=tuple(ops))  # type: ignore[arg-type]


def hamming_ops(first: Nas201Spec, second: Nas201Spec) -> int:
    return sum(a != b for a, b in zip(first.ops, second.ops, strict=True))
