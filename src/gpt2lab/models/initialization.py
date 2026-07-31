"""GPT-2 style weight initialization.

The scaled initialization for residual output projections is expressed with an
explicit marker rather than by inspecting module names, so a module declares
its own role at construction time.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = [
    "WeightInitializer",
    "is_residual_projection",
    "mark_residual_projection",
]

_RESIDUAL_FLAG = "_gpt2lab_residual_projection"


def mark_residual_projection(module: nn.Module) -> nn.Module:
    """Flag ``module`` as the output projection of a residual branch."""
    setattr(module, _RESIDUAL_FLAG, True)
    return module


def is_residual_projection(module: nn.Module) -> bool:
    return bool(getattr(module, _RESIDUAL_FLAG, False))


class WeightInitializer:
    """Callable suitable for :meth:`torch.nn.Module.apply`.

    ``Linear`` and ``Embedding`` weights are drawn from ``N(0, init_std**2)``.
    Residual output projections use ``init_std / sqrt(2 * n_layer)`` so the
    variance added by the residual stream stays roughly constant with depth.
    ``LayerNorm`` keeps the PyTorch defaults (unit gain, zero bias).
    """

    def __init__(self, init_std: float, n_layer: int) -> None:
        if init_std <= 0:
            raise ValueError(f"init_std must be positive (received {init_std}).")
        if n_layer <= 0:
            raise ValueError(f"n_layer must be positive (received {n_layer}).")

        self._init_std = init_std
        self._residual_std = init_std * (2 * n_layer) ** -0.5

    @property
    def init_std(self) -> float:
        return self._init_std

    @property
    def residual_std(self) -> float:
        return self._residual_std

    def __call__(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self._residual_std if is_residual_projection(module) else self._init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self._init_std)

