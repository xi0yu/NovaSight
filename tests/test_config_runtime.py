from pathlib import Path

import pytest

from novasight.config import RuntimeConfig, load_runtime_config, save_runtime_config


def test_runtime_config_defaults_do_not_contain_model_truth() -> None:
    cfg = RuntimeConfig()

    assert cfg.web.port == 5174
    assert cfg.executor.default == "dry_run"
    assert not hasattr(cfg, "model_path")
    assert not hasattr(cfg, "plugin_settings")


def test_runtime_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    cfg = RuntimeConfig()
    cfg.web.port = 6000
    cfg.source.default = "image:/tmp/frame.jpg"
    cfg.executor.default = "dry_run"

    save_runtime_config(cfg, path)
    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.source.default == "image:/tmp/frame.jpg"
    assert loaded.executor.default == "dry_run"


def test_runtime_config_rejects_falsey_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime config must be a mapping"):
        load_runtime_config(path)
