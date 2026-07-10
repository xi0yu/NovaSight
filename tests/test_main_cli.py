from __future__ import annotations

import importlib

from novasight.config import RuntimeConfig


def test_main_applies_port_override_to_uvicorn(monkeypatch, tmp_path) -> None:
    main_module = importlib.import_module("novasight.main")
    config = RuntimeConfig()
    observed: dict[str, object] = {}

    monkeypatch.setattr(main_module, "load_runtime_config", lambda _path: config)
    monkeypatch.setattr(main_module, "configure_logging", lambda _config: None)
    monkeypatch.setattr(main_module, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda _app, *, host, port: observed.update(host=host, port=port),
    )

    result = main_module.main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "5175",
            "--data-dir",
            str(tmp_path),
            "--config",
            str(tmp_path / "runtime.yaml"),
        ]
    )

    assert result == 0
    assert observed == {"host": "127.0.0.1", "port": 5175}
