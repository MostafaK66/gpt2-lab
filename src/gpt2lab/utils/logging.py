"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Install a stdout handler (and optionally a file handler) on the root logger.

    Safe to call more than once: existing handlers installed by this function
    are replaced rather than duplicated.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # matplotlib is extremely chatty at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

