"""gpt2-lab: a small, well-separated GPT-2 training codebase."""

from gpt2lab.config import ExperimentConfig
from gpt2lab.experiment import Experiment

__all__ = ["Experiment", "ExperimentConfig", "__version__"]

__version__ = "0.1.0"
