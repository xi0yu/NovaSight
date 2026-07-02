from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn


class FailFastHandler:
    def __init__(
        self,
        *,
        log_dir: str | Path = "logs",
        exit_fn: Callable[[int], NoReturn] = os._exit,
        on_fatal: Callable[[str, BaseException, Path], None] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.exit_fn = exit_fn
        self.on_fatal = on_fatal

    def run(self, name: str, target: Callable[[], None]) -> None:
        try:
            target()
        except BaseException as exc:
            self.crash(name, exc)

    def crash(self, name: str, exc: BaseException) -> NoReturn:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"crash_{name}.log"
        body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        path.write_text(body, encoding="utf-8")
        if self.on_fatal is not None:
            self.on_fatal(name, exc, path)
        logging.getLogger("novasight").critical("fatal thread crash in %s: %s", name, exc)
        self.exit_fn(1)
