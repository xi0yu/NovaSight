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
    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    app = create_app(data_dir=Path(args.data_dir), config_path=Path(args.config))
    uvicorn.run(app, host=host, port=port)
    return 0
