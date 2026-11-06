"""Central logging setup."""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"finnie.{name}")
