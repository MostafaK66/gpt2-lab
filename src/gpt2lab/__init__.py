"""A small, maintainable GPT-2 training lab."""

from gpt2lab.config import ExperimentConfig
from gpt2lab.models import GPT

__all__ = ["GPT", "ExperimentConfig", "__version__"]

__version__ = "1.0.0"
