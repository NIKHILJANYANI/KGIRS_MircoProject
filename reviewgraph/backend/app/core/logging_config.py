"""
Logging configuration for the ReviewGraph backend.

Provides JSON-structured console logging with noise suppression for
third-party libraries (uvicorn, neo4j) unless the effective level is DEBUG.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# JSON formatter
# --------------------------------------------------------------------------- #

class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    # Fields that the standard LogRecord already carries – we handle them
    # explicitly so they don't show up twice in *extra*.
    _STDLIB_ATTRS = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Let the base class format the *message* (handles % substitution,
        # exception text, stack info, etc.)
        message = record.getMessage()

        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": message,
        }

        # Attach exception info when present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Attach any *extra* keys the caller passed in
        extra: Dict[str, Any] = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._STDLIB_ATTRS and not k.startswith("_")
        }
        if extra:
            payload["extra"] = extra

        return json.dumps(payload, default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Public setup function
# --------------------------------------------------------------------------- #

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure application-wide logging.

    Parameters
    ----------
    log_level:
        One of DEBUG / INFO / WARNING / ERROR / CRITICAL (case-insensitive).

    Returns
    -------
    logging.Logger
        The root application logger (name ``reviewgraph``).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # ------------------------------------------------------------------ #
    # Console handler with JSON formatter
    # ------------------------------------------------------------------ #
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(JsonFormatter())

    # ------------------------------------------------------------------ #
    # Root logger – set to WARNING so unrelated libraries stay quiet
    # ------------------------------------------------------------------ #
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    # Remove any handlers already attached (e.g. from a previous call)
    root.handlers.clear()
    root.addHandler(handler)

    # ------------------------------------------------------------------ #
    # Application logger
    # ------------------------------------------------------------------ #
    app_logger = logging.getLogger("reviewgraph")
    app_logger.setLevel(numeric_level)
    app_logger.propagate = False
    app_logger.handlers.clear()
    app_logger.addHandler(handler)

    # ------------------------------------------------------------------ #
    # Third-party noise suppression
    # ------------------------------------------------------------------ #
    _noisy_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.protocols",
        "neo4j",
        "neo4j.io",
        "neo4j.bolt",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "faiss",
        "transformers",
        "torch",
    ]

    # When DEBUG is requested we let all loggers through; otherwise cap them
    # at INFO so we still see important messages but not internal chatter.
    third_party_level = logging.DEBUG if numeric_level <= logging.DEBUG else logging.INFO

    for name in _noisy_loggers:
        tpl = logging.getLogger(name)
        tpl.setLevel(third_party_level)
        tpl.propagate = True  # bubble up to root which has our handler

    app_logger.info(
        "Logging initialised",
        extra={"log_level": log_level.upper(), "effective_level": numeric_level},
    )

    return app_logger


# --------------------------------------------------------------------------- #
# Module-level convenience logger (importable by other modules)
# --------------------------------------------------------------------------- #
logger: logging.Logger = logging.getLogger("reviewgraph")
