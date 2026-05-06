"""Neural latent → ``WorldSpec`` generator driven by YAML (``neural_world_generator.yaml``)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

from . import WorldGenerator
from ..specs.spec import WorldSpec


def load_neural_generator_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate minimal keys for neural world YAML."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"Neural generator YAML not found: {src.resolve()}. "
            "Pass spec_path= to NeuralWorldGenerator, or ensure "
            "neural_world_generator.yaml is available under worldspace/specs/ "
            "(it is tracked in the repo; *.spec in .gitignore has an exception for this file)."
        )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {src}")
    if raw.get("version") != 1:
        raise ValueError(f"{src}: expected version: 1")
    return raw


class NeuralWorldGenerator(WorldGenerator):
    """
    Sample latent vectors → MLP (from YAML ``model`` section) → decode to :class:`WorldSpec`.

    Default YAML path: ``neural_world_generator.yaml`` in worldspace/specs/.
    ``device`` overrides YAML ``torch.device`` (``None`` = follow YAML; use ``"auto"`` in YAML
    for CUDA-if-available else CPU).
    """

    def __init__(
        self,
        spec_path: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> None:
        path = Path(spec_path) if spec_path is not None else _DEFAULT_SPEC_PATH
        self.spec_path = path
        raw = load_neural_generator_yaml(path)

        torch_cfg = raw["torch"]
        model_cfg = raw["model"]

        latent_dim = int(model_cfg["latent_dim"])
        hidden = [int(x) for x in model_cfg["hidden"]]
        activation_name = str(model_cfg["activation"])
        dropout_p = float(model_cfg.get("dropout") or 0.0)

        dev_from_yaml = str(torch_cfg.get("device") or "auto")
        dtype_s = torch_cfg.get("dtype") or "float32"
        self.device = _resolve_torch_device(dev_from_yaml, device)
        self.dtype = torch.float64 if dtype_s == "float64" else torch.float32

        self._policy = _WorldPolicyMLP(
            latent_dim, hidden, activation_name, dropout_p
        ).to(self.device, dtype=self.dtype)

        weights_path = raw.get("weights_path")
        if weights_path:
            wp = Path(weights_path)
            if not wp.is_file():
                base = path.parent.parent.parent
                cand = base / weights_path
                wp = cand if cand.is_file() else wp
            if wp.is_file():
                try:
                    state = torch.load(wp, map_location=self.device, weights_only=True)
                except TypeError:
                    state = torch.load(wp, map_location=self.device)
                self._policy.load_state_dict(state)
        self._latent_dim = latent_dim

        wd = raw["world_defaults"]
        self._world_defaults = {
            "grid_size": int(wd["grid_size"]),
            "steps": int(wd["steps"]),
            "cell_types": list(wd["cell_types"]),
            "neighborhood": str(wd.get("neighborhood", "moore")),
        }
        self._decoder = _hints_from_yaml(raw)
        self._base_seed = int(raw.get("base_seed") or 0)

    def generate(self, n_worlds: int) -> list[WorldSpec]:
        """Draw ``n_worlds`` latents → MLP → list of ``WorldSpec`` (world ``seed`` = row index)."""
        if n_worlds <= 0:
            return []
        with torch.no_grad():
            return [self._sample_world(i) for i in range(n_worlds)]

    def iter_worlds(self, n_worlds: int) -> Iterator[WorldSpec]:
        """Yield worlds one at a time without building the full list early."""
        with torch.no_grad():
            for i in range(max(0, n_worlds)):
                yield self._sample_world(i)

    def _sample_world(self, index: int) -> WorldSpec:
        """Sample one latent (reproducible from ``base_seed`` + ``index``) → ``WorldSpec``."""
        latent_dim = self._latent_dim
        h = self._decoder
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self._base_seed + index))
        z_cpu = torch.randn((1, latent_dim), generator=gen, dtype=self.dtype)
        z = z_cpu.to(device=self.device, dtype=self.dtype)
        y = _forward_to_numpy(self._policy, z, device=self.device, dtype=self.dtype)
        birth_logits = y[:_RULE_DIM]
        surv_logits = y[_RULE_DIM : 2 * _RULE_DIM]
        fnoise, fregen, fpred = y[2 * _RULE_DIM :]
        birth = _decode_rule_indices(
            birth_logits,
            min_count=h.birth_min,
            max_count=h.birth_max,
            threshold=h.birth_threshold,
        )
        survival = _decode_rule_indices(
            surv_logits,
            min_count=h.survival_min,
            max_count=h.survival_max,
            threshold=h.survival_threshold,
        )
        wd = self._world_defaults
        return WorldSpec(
            birth=birth,
            survival=survival,
            noise=_scalar_from_logit(float(fnoise), h.scale_noise),
            resource_regen=_scalar_from_logit(float(fregen), h.scale_regen),
            predation=_scalar_from_logit(float(fpred), h.scale_predation),
            cell_types=list(wd["cell_types"]),
            neighborhood=str(wd["neighborhood"]),
            grid_size=int(wd["grid_size"]),
            steps=int(wd["steps"]),
            seed=index,
        )


# -----------------------------------------------------------------------------
# Implementation details


_RULE_DIM = 9
_FLOAT_HEAD = 3  # noise, resource_regen, predation
_OUT_DIM = 2 * _RULE_DIM + _FLOAT_HEAD

_DEFAULT_SPEC_PATH = Path(__file__).resolve().parent.parent / "specs" / "neural_world_generator.yaml"


def _resolve_torch_device(yaml_device: str, override: str | None) -> torch.device:
    """Pick ``torch.device`` from YAML and optional string override (``None`` = use YAML only)."""
    raw = (override if override is not None else yaml_device).strip().lower()
    if raw in ("", "auto"):
        pick = "cuda" if torch.cuda.is_available() else "cpu"
    elif raw in ("gpu", "cuda"):
        pick = "cuda"
    else:
        pick = raw
    if pick.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Torch device {pick!r} requested but CUDA is not available."
        )
    return torch.device(pick)


@dataclass(frozen=True)
class _DecoderHints:
    birth_min: int
    birth_max: int
    birth_threshold: float
    survival_min: int
    survival_max: int
    survival_threshold: float
    scale_noise: float
    scale_regen: float
    scale_predation: float


def _relu() -> Callable[..., nn.Module]:
    return lambda: nn.ReLU(inplace=False)


def _activation(name: str) -> Callable[..., nn.Module]:
    n = name.lower().strip()
    if n == "relu":
        return _relu()
    if n == "tanh":
        return lambda: nn.Tanh()
    if n == "gelu":
        return lambda: nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class _WorldPolicyMLP(nn.Module):
    """Maps Gaussian latent vector to logits (9 birth + 9 survival + 3 scalars)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: list[int],
        activation_name: str,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or not hidden:
            raise ValueError("latent_dim and non-empty hidden are required")
        act = _activation(activation_name)
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h
        layers.append(nn.Linear(prev, _OUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _decode_rule_indices(
    logits: np.ndarray,
    *,
    min_count: int,
    max_count: int,
    threshold: float,
) -> list[int]:
    """Subset of ``{0..8}`` using sigmoid/threshold; then enforce ``[min_count, max_count]``."""
    if min_count > max_count or min_count < 0 or max_count > _RULE_DIM:
        raise ValueError("Invalid min_count / max_count for rule decoder")
    probs = _sigmoid_np(logits.astype(np.float64))
    order_desc = sorted(range(_RULE_DIM), key=lambda i: probs[i], reverse=True)
    cand = sorted({i for i in range(_RULE_DIM) if probs[i] >= threshold})
    for i in order_desc:
        if len(cand) >= min_count:
            break
        if i not in cand:
            cand.append(i)
    cand.sort()
    while len(cand) > max_count:
        drop = min(cand, key=lambda i: probs[i])
        cand.remove(drop)
    while len(cand) < min_count:
        added = False
        for i in order_desc:
            if i not in cand:
                cand.append(i)
                cand.sort()
                added = True
                break
        if not added:
            break
    return sorted(set(cand))


def _hints_from_yaml(doc: dict[str, Any]) -> _DecoderHints:
    dec = doc["decoder"]
    scales = dec["scales"]
    birth = dec["birth"]
    surv = dec["survival"]
    return _DecoderHints(
        birth_min=int(birth["min_count"]),
        birth_max=int(birth["max_count"]),
        birth_threshold=float(birth["threshold"]),
        survival_min=int(surv["min_count"]),
        survival_max=int(surv["max_count"]),
        survival_threshold=float(surv["threshold"]),
        scale_noise=float(scales["noise"]),
        scale_regen=float(scales["resource_regen"]),
        scale_predation=float(scales["predation"]),
    )


def _scalar_from_logit(logit: float, scale_max: float) -> float:
    s = float(_sigmoid_np(np.array([logit]))[0])
    return float(np.clip(s * scale_max, 0.0, scale_max))


def _forward_to_numpy(
    model: nn.Module,
    z: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(z.to(device=device, dtype=dtype)).squeeze(0)
    return y.detach().cpu().numpy().astype(np.float64)
