"""Logging infrastructure for SPARC-Net.

Provides a single configuration entry point so that library modules can simply call
``get_logger(__name__)`` without worrying about handler setup or duplicate handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
    force: bool = False,
) -> None:
    """Configure the root logger exactly once.

    Args:
        level: Logging level for the root logger.
        log_file: Optional path to additionally write logs to. Parent directories
            are created if they do not exist.
        fmt: ``logging`` format string.
        force: Reconfigure even if configuration already happened.

    Raises:
        OSError: If ``log_file`` is given and its parent directory cannot be created.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    root.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


class JsonlLogger:
    """Append-only JSON-lines logger for structured metric records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one JSON object to the file.

        Args:
            record: Any JSON-serialisable mapping.

        Raises:
            TypeError: If ``record`` contains values that cannot be serialised.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), default=str) + "\n")


class CsvLogger:
    """Append-only CSV logger with a fixed header written on first use."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(",".join(self.fields) + "\n", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one row; missing fields are written as empty strings."""
        row = [str(record.get(field, "")) for field in self.fields]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(",".join(row) + "\n")
