"""Tests for runtime configuration loading and validation.

Covers the schema-enforced contract: round-trip, missing/empty inputs,
unknown keys, and type errors. Defaults are exercised in one test, and
unknown-key rejection is parametrized.
"""
from pathlib import Path

import pytest

from novasight.config import (
    RuntimeConfig,
    load_runtime_config,
    parse_runtime_config,
    save_runtime_config,
)


def test_runtime_config_defaults_are_stable() -> None:
    cfg = RuntimeConfig()

    assert cfg.web.port == 5174
    assert cfg.executor.default == "dry_run"
    # Old attributes that drove the first prototype must not have leaked back.
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


@pytest.mark.parametrize(
    "yaml_text",
    ["", "null\n", "   \n"],
)
def test_runtime_config_returns_defaults_for_blank_or_empty_inputs(
    tmp_path: Path, yaml_text: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    assert load_runtime_config(path) == RuntimeConfig()


def test_runtime_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_runtime_config(tmp_path / "missing.yaml") == RuntimeConfig()


def test_runtime_config_partial_nested_config_preserves_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web:\n  port: 6000\n", encoding="utf-8")

    loaded = load_runtime_config(path)

    assert loaded.web.port == 6000
    assert loaded.web.host == "0.0.0.0"


def test_save_runtime_config_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config" / "novasight.yaml"

    save_runtime_config(RuntimeConfig(), path)

    assert path.exists()


def test_runtime_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime config must be a mapping"):
        load_runtime_config(path)


def test_runtime_config_rejects_nested_non_mapping_section(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web: false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="web.*must be a mapping"):
        load_runtime_config(path)


@pytest.mark.parametrize("unknown_key", ["model_registry", "weeb"])
def test_runtime_config_rejects_unknown_top_level_key(
    tmp_path: Path, unknown_key: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(f"{unknown_key}: {{}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"unknown config key.*{unknown_key}"):
        load_runtime_config(path)


def test_runtime_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text("web:\n  prt: 5174\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key.*web.prt"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("yaml_text", "key_path"),
    [
        ("web:\n  port: nope\n", "web.port"),
        ("source:\n  target_fps: []\n", "source.target_fps"),
        ("source:\n  default: null\n", "source.default"),
    ],
)
def test_runtime_config_rejects_invalid_leaf_types(
    tmp_path: Path, yaml_text: str, key_path: str
) -> None:
    path = tmp_path / "novasight.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=key_path):
        load_runtime_config(path)


def test_runtime_config_restricts_preview_fps_to_supported_values() -> None:
    for fps in (15, 30, 60):
        cfg = parse_runtime_config({"limits": {"stream_fps": fps}})
        assert cfg.limits.stream_fps == fps

    with pytest.raises(ValueError, match="limits.stream_fps.*15, 30, 60"):
        parse_runtime_config({"limits": {"stream_fps": 120}})
