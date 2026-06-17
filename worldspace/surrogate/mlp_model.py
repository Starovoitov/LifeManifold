"""PyTorch MLP backend for Strategy A multi-task surrogate regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from worldspace.surrogate.device import DevicePreference, resolve_training_device
from worldspace.surrogate.determinism import (
    DEFAULT_MLP_HIDDEN_DIMS,
    apply_mlp_determinism,
)
from worldspace.surrogate.model import (
    EXPECTED_FEATURE_DIM,
    FITNESS_TARGET_KEY,
    MIN_FITNESS_HEAD_SAMPLES,
    TARGET_KEYS,
)

MLP_OUTPUT_DIM = len(TARGET_KEYS) + 1
COMPONENT_SLICE = slice(0, len(TARGET_KEYS))
FITNESS_OUTPUT_INDEX = len(TARGET_KEYS)

__all__ = [
    "FITNESS_OUTPUT_INDEX",
    "MLP_OUTPUT_DIM",
    "MlpActivation",
    "MlpTrainConfig",
    "MlpUncertaintyMethod",
    "build_strategy_a_mlp",
    "build_strategy_a_mlp_for_state_dict",
    "hidden_dims_from_state_dict",
    "input_dim_from_state_dict",
    "mlp_state_dict_uses_batch_norm",
    "mlp_state_dict_uses_dropout",
    "predict_mlp_state_dict",
    "sample_mlp_member_outputs_mc",
    "train_mlp_member",
]

MlpActivation = Literal["gelu", "relu"]
MlpUncertaintyMethod = Literal["ensemble", "ensemble_mc"]


@dataclass(frozen=True)
class MlpTrainConfig:
    """Hyper-parameters for one deterministic MLP ensemble member."""

    hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS
    learning_rate: float = 1e-3
    max_epochs: int = 200
    patience: int = 15
    batch_size: int = 256
    fitness_loss_weight: float = 1.0
    huber_delta: float = 0.05
    cosine_eta_min_factor: float = 0.01
    dropout_p: float = 0.0
    mc_samples: int = 16
    uncertainty_method: MlpUncertaintyMethod = "ensemble"


def _make_loss_fn(cfg: MlpTrainConfig) -> Any:
    """Huber (Smooth L1) loss for bounded surrogate regression targets."""
    import torch.nn as nn

    return nn.SmoothL1Loss(beta=float(cfg.huber_delta))


def _member_training_loss(
    preds: Any,
    *,
    component_targets: Any,
    fitness_targets: Any | None,
    fitness_mask: Any | None,
    loss_fn: Any,
    fitness_loss_weight: float,
    train_fitness_head: bool,
) -> Any:
    loss = loss_fn(preds[:, COMPONENT_SLICE], component_targets)
    if (
        train_fitness_head
        and fitness_mask is not None
        and fitness_targets is not None
        and bool(fitness_mask.any())
    ):
        fit_preds = preds[fitness_mask, FITNESS_OUTPUT_INDEX]
        fit_targets = fitness_targets[fitness_mask]
        loss = loss + float(fitness_loss_weight) * loss_fn(fit_preds, fit_targets)
    return loss


def _activation_module(activation: MlpActivation) -> Any:
    import torch.nn as nn

    if activation == "gelu":
        return nn.GELU()
    return nn.ReLU()


def build_strategy_a_mlp(
    *,
    input_dim: int = EXPECTED_FEATURE_DIM,
    hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS,
    activation: MlpActivation = "gelu",
    batch_norm: bool = True,
    dropout_p: float = 0.1,
) -> Any:
    """Construct an untrained ``StrategyAMlp`` module."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(_activation_module(activation))
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=float(dropout_p)))
        prev = width
    layers.append(nn.Linear(prev, MLP_OUTPUT_DIM))

    class StrategyAMlp(nn.Module):
        def __init__(self, seq: nn.Sequential) -> None:
            super().__init__()
            self.net = seq

        def forward(self, features: Any) -> Any:
            return self.net(features)

    return StrategyAMlp(nn.Sequential(*layers))


def mlp_state_dict_uses_batch_norm(state_dict: dict[str, Any]) -> bool:
    """Return whether a checkpoint uses BatchNorm (vs legacy ReLU-only blocks)."""
    return any(key.endswith(".running_mean") for key in state_dict)


def mlp_state_dict_uses_dropout(state_dict: dict[str, Any]) -> bool:
    """Return whether a BN+GELU checkpoint includes Dropout blocks."""
    if not mlp_state_dict_uses_batch_norm(state_dict):
        return False
    return "net.3.weight" not in state_dict


def build_strategy_a_mlp_for_state_dict(
    state_dict: dict[str, Any],
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS,
) -> Any:
    """Rebuild the MLP architecture that matches one pickled ``state_dict``."""
    if not mlp_state_dict_uses_batch_norm(state_dict):
        return build_strategy_a_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            batch_norm=False,
            dropout_p=0.0,
        )
    if mlp_state_dict_uses_dropout(state_dict):
        return build_strategy_a_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation="gelu",
            batch_norm=True,
            dropout_p=0.1,
        )
    return build_strategy_a_mlp(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation="gelu",
        batch_norm=True,
        dropout_p=0.0,
    )


def input_dim_from_state_dict(state_dict: dict[str, Any]) -> int:
    """Infer MLP input width from the first linear layer weights."""
    weight = state_dict.get("net.0.weight")
    if weight is None:
        msg = "state_dict missing net.0.weight"
        raise ValueError(msg)
    return int(weight.shape[1])


def hidden_dims_from_state_dict(state_dict: dict[str, Any]) -> tuple[int, ...]:
    """Infer hidden layer widths from 2-D ``net.*.weight`` tensors (excludes output head)."""
    linear_layers: list[tuple[int, int]] = []
    for key, tensor in state_dict.items():
        if not key.startswith("net.") or not key.endswith(".weight"):
            continue
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        layer_index = int(key.split(".")[1])
        linear_layers.append((layer_index, int(shape[0])))
    if not linear_layers:
        msg = "state_dict has no linear net.*.weight layers"
        raise ValueError(msg)
    linear_layers.sort(key=lambda item: item[0])
    out_dims = [out_features for _, out_features in linear_layers]
    if out_dims[-1] != MLP_OUTPUT_DIM:
        msg = (
            f"expected final linear out_features={MLP_OUTPUT_DIM}, "
            f"got {out_dims[-1]}"
        )
        raise ValueError(msg)
    return tuple(out_dims[:-1])


def train_mlp_member(
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    seed: int,
    config: MlpTrainConfig | None = None,
    val_features: np.ndarray | None = None,
    val_targets: dict[str, np.ndarray] | None = None,
    device: DevicePreference = "auto",
) -> dict[str, Any]:
    """Train one MLP member and return a CPU ``state_dict``."""
    import torch

    apply_mlp_determinism()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch_device = torch.device(resolve_training_device(device))

    cfg = config or MlpTrainConfig()
    x_train = np.asarray(feature_matrix, dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[0] < 1:
        msg = f"feature matrix must be (N, D) with N >= 1, got shape={x_train.shape!r}"
        raise ValueError(msg)
    input_dim = int(x_train.shape[1])

    y_components = np.stack(
        [np.asarray(targets[key], dtype=np.float32) for key in TARGET_KEYS],
        axis=1,
    )
    fitness = targets.get(FITNESS_TARGET_KEY)
    fitness_mask = np.ones(x_train.shape[0], dtype=bool)
    y_fitness = np.zeros(x_train.shape[0], dtype=np.float32)
    train_fitness_head = False
    if fitness is not None:
        fitness_arr = np.asarray(fitness, dtype=np.float32).reshape(-1)
        fitness_mask = np.isfinite(fitness_arr)
        y_fitness = np.where(fitness_mask, fitness_arr, 0.0).astype(np.float32)
        train_fitness_head = int(fitness_mask.sum()) >= MIN_FITNESS_HEAD_SAMPLES

    model = build_strategy_a_mlp(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout_p=cfg.dropout_p,
    )
    model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = _make_loss_fn(cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.max_epochs),
        eta_min=float(cfg.learning_rate) * float(cfg.cosine_eta_min_factor),
    )

    x_tensor = torch.from_numpy(x_train).to(torch_device)
    y_comp_tensor = torch.from_numpy(y_components).to(torch_device)
    y_fit_tensor = torch.from_numpy(y_fitness).to(torch_device)
    fit_mask_tensor = torch.from_numpy(fitness_mask).to(torch_device)

    x_val_tensor: torch.Tensor | None = None
    y_comp_val: torch.Tensor | None = None
    y_fit_val: torch.Tensor | None = None
    fit_mask_val: torch.Tensor | None = None
    if val_features is not None and val_targets is not None:
        x_val = np.asarray(val_features, dtype=np.float32)
        y_comp_val_np = np.stack(
            [np.asarray(val_targets[key], dtype=np.float32) for key in TARGET_KEYS],
            axis=1,
        )
        x_val_tensor = torch.from_numpy(x_val).to(torch_device)
        y_comp_val = torch.from_numpy(y_comp_val_np).to(torch_device)
        val_fitness = val_targets.get(FITNESS_TARGET_KEY)
        if val_fitness is not None:
            val_fit_arr = np.asarray(val_fitness, dtype=np.float32).reshape(-1)
            fit_mask_val = torch.from_numpy(np.isfinite(val_fit_arr)).to(torch_device)
            y_fit_val = torch.from_numpy(
                np.where(np.isfinite(val_fit_arr), val_fit_arr, 0.0).astype(np.float32)
            ).to(torch_device)

    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0
    n_rows = int(x_train.shape[0])

    for _epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n_rows)
        for start in range(0, n_rows, cfg.batch_size):
            batch_idx = perm[start : start + cfg.batch_size]
            batch_x = x_tensor[batch_idx]
            batch_y = y_comp_tensor[batch_idx]
            preds = model(batch_x)
            batch_fit_targets = y_fit_tensor[batch_idx]
            batch_fit_mask = fit_mask_tensor[batch_idx]
            loss = _member_training_loss(
                preds,
                component_targets=batch_y,
                fitness_targets=batch_fit_targets,
                fitness_mask=batch_fit_mask,
                loss_fn=loss_fn,
                fitness_loss_weight=cfg.fitness_loss_weight,
                train_fitness_head=train_fitness_head,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if x_val_tensor is None or y_comp_val is None:
            continue

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_tensor)
            val_loss = _member_training_loss(
                val_preds,
                component_targets=y_comp_val,
                fitness_targets=y_fit_val,
                fitness_mask=fit_mask_val,
                loss_fn=loss_fn,
                fitness_loss_weight=cfg.fitness_loss_weight,
                train_fitness_head=train_fitness_head,
            )
            val_scalar = float(val_loss.item())

        if val_scalar < best_val_loss:
            best_val_loss = val_scalar
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    if best_state is not None:
        return best_state
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict_mlp_state_dict(
    state_dict: dict[str, Any],
    features: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS,
    input_dim: int | None = None,
) -> np.ndarray:
    """Run one pickled MLP member; returns shape ``(N, MLP_OUTPUT_DIM)``."""
    import torch

    vector = np.asarray(features, dtype=np.float32)
    if vector.ndim == 1:
        matrix = vector.reshape(1, -1)
    else:
        matrix = vector
    resolved_input_dim = (
        int(input_dim)
        if input_dim is not None
        else input_dim_from_state_dict(state_dict)
    )
    model = build_strategy_a_mlp_for_state_dict(
        state_dict,
        input_dim=resolved_input_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(matrix))
    return preds.cpu().numpy()


def _set_dropout_train_mode(model: Any) -> None:
    """Keep BatchNorm in eval mode while enabling Dropout for MC sampling."""
    import torch.nn as nn

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def sample_mlp_member_outputs_mc(
    state_dict: dict[str, Any],
    features: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS,
    input_dim: int | None = None,
    n_samples: int = 16,
    seed: int = 0,
) -> list[np.ndarray]:
    """Run MC Dropout forward passes; each item is shape ``(MLP_OUTPUT_DIM,)``."""
    import torch

    if n_samples < 1:
        msg = f"n_samples must be >= 1, got {n_samples}"
        raise ValueError(msg)
    vector = np.asarray(features, dtype=np.float32)
    matrix = vector.reshape(1, -1) if vector.ndim == 1 else vector
    resolved_input_dim = (
        int(input_dim)
        if input_dim is not None
        else input_dim_from_state_dict(state_dict)
    )
    model = build_strategy_a_mlp_for_state_dict(
        state_dict,
        input_dim=resolved_input_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(state_dict)
    _set_dropout_train_mode(model)
    outputs: list[np.ndarray] = []
    x_tensor = torch.from_numpy(matrix)
    with torch.no_grad():
        for sample_index in range(int(n_samples)):
            torch.manual_seed(int(seed) + sample_index)
            row = model(x_tensor)[0]
            outputs.append(row.cpu().numpy())
    return outputs


def ensemble_member_seed(base_seed: int, member_index: int) -> int:
    """Stable per-member seed for MLP bootstrap ensemble."""
    return int(base_seed) + int(member_index)
