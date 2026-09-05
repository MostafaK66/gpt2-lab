"""Domain-specific exceptions exposed by :mod:`gpt2lab`."""


class GPT2LabError(Exception):
    """Base class for recoverable application errors."""


class ConfigurationError(GPT2LabError, ValueError):
    """A configuration file or value is invalid."""


class CorpusError(GPT2LabError):
    """A corpus could not be downloaded, read, tokenized, or split."""


class DeviceUnavailableError(GPT2LabError):
    """The requested accelerator is not available."""


class CheckpointError(GPT2LabError):
    """A checkpoint could not be saved or loaded safely."""


class TrainingError(GPT2LabError):
    """Training cannot continue safely."""
