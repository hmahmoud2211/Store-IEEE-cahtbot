"""Structured logging (structlog) and latency timing helpers.

Logs are JSON by default. Never pass secrets or full document content to the
logger: log ids, counts, hashes and timings instead.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "sentence_transformers", "urllib3", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def text_fingerprint(text: str) -> str:
    """Short, non-reversible fingerprint used to correlate queries in logs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Record the elapsed wall time of the block in milliseconds under `key`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - start) * 1000, 2)
