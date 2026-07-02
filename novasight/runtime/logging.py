from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from novasight.config import RuntimeConfig


def configure_logging(config: RuntimeConfig) -> Path:
    log_dir = Path(config.logging.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "novasight.log"
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    root = logging.getLogger("novasight")
    root.setLevel(level)
    root.handlers.clear()
    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    return log_path
