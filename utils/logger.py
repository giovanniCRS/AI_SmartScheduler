"""Structured logging for SmartScheduler.

Every orchestration iteration should log: tokens used, error encountered
(if any), fairness scores, and execution time, both to console and to a
log file (see config.LOG_FILE).
"""
import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional

import config


def setup_logger(name: str = "smartscheduler") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (idempotent)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # sandbox might be read-only for cwd; console logging still works
        pass

    return logger


@contextmanager
def timed_step(logger: logging.Logger, step_name: str):
    """Context manager that logs how long a graph node took to run."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[{step_name}] completed in {elapsed:.2f}s")


def log_iteration(
    logger: logging.Logger,
    iteration: int,
    tokens_used: Optional[int] = None,
    error: Optional[str] = None,
    fairness_scores: Optional[dict] = None,
    elapsed_seconds: Optional[float] = None,
) -> None:
    """Emit one structured line summarizing an orchestration iteration."""
    parts = [f"iteration={iteration}"]
    if tokens_used is not None:
        parts.append(f"tokens_used={tokens_used}")
    if elapsed_seconds is not None:
        parts.append(f"elapsed={elapsed_seconds:.2f}s")
    if error:
        parts.append(f"error={error!r}")
    if fairness_scores:
        worst = min(fairness_scores.items(), key=lambda kv: kv[1])
        parts.append(f"least_satisfied=worker_{worst[0]}({worst[1]:.2f})")
    logger.info(" | ".join(parts))
