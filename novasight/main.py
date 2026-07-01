from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_runtime_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("novasight")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default="config/novasight.yaml")
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_runtime_config(args.config)
    if args.host is not None:
        cfg.web.host = args.host
    if args.port is not None:
        cfg.web.port = args.port
    app = create_app(
        data_dir=Path(args.data_dir),
        config_path=Path(args.config),
        config=cfg,
    )
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port)
    return 0
