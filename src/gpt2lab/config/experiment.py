"""Composite experiment configuration built from section dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .sections import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EvalConfig,
    CheckpointConfig,
    VizConfig,
    SamplingConfig,
)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        """Build from a flat dict (like configs/default.py CONFIG)."""
        section_map: dict[str, type] = {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "eval": EvalConfig,
            "checkpoint": CheckpointConfig,
            "viz": VizConfig,
            "sampling": SamplingConfig,
        }
        # Build a mapping: field_name -> (section_name, field_name)
        field_to_section: dict[str, str] = {}
        for sec_name, sec_cls in section_map.items():
            for f in fields(sec_cls):
                field_to_section[f.name] = sec_name

        buckets: dict[str, dict[str, Any]] = {k: {} for k in section_map}
        for key, val in d.items():
            sec = field_to_section.get(key)
            if sec is not None:
                buckets[sec][key] = val

        return cls(**{name: section_map[name](**buckets[name]) for name in section_map})

