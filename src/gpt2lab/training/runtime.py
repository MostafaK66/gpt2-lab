"""Reproducible runtime and device selection."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Iterator

import torch

from gpt2lab.config import RuntimeConfig
from gpt2lab.errors import DeviceUnavailableError


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def select_device(
    requested: str,
    *,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
    mps_available: Callable[[], bool] = _mps_available,
) -> torch.device:
    """Resolve a device and reject explicitly unavailable accelerators."""
    if requested == "auto":
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not cuda_available():
        raise DeviceUnavailableError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available():
        raise DeviceUnavailableError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise DeviceUnavailableError(f"unsupported device {requested!r}")
    return torch.device(requested)


def configure_runtime(config: RuntimeConfig) -> torch.device:
    """Seed PRNGs, configure deterministic execution, and select a device."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    return select_device(config.device)


@contextlib.contextmanager
def autocast_context(device: torch.device, config: RuntimeConfig) -> Iterator[None]:
    """Enable AMP only where PyTorch supports the configured dtype safely."""
    enabled = config.use_mixed_precision and device.type == "cuda"
    dtype = torch.float16 if config.mixed_precision_dtype == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        yield
