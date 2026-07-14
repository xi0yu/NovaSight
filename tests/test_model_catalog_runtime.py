from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from novasight.api import routes_models
from novasight.api.app import create_app
from novasight.api.routes_models import PublishRequest
from novasight.config import RuntimeConfig
from novasight.model_registry import ModelRegistry, read_manifest
from novasight.model_registry import scanner
from novasight.model_registry.schema import Deployment


def _request_with_registry(registry: ModelRegistry):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                models=registry,
                inference=SimpleNamespace(),
            )
        )
    )


def test_project_listing_reads_registry_without_scanning_disk(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    registry.create_project("cached", "cached project")

    def fail_scan(_root, **_kwargs):
        raise AssertionError("project listing must not scan model files")

    monkeypatch.setattr(routes_models, "scan_model_artifacts", fail_scan)

    projects = routes_models.list_projects(_request_with_registry(registry))

    assert [project["name"] for project in projects] == ["cached"]


def test_model_catalog_read_does_not_import_or_copy_discovered_engine(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "demo.engine"
    source_path.write_bytes(b"engine")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")

    catalog = routes_models.get_model_catalog(_request_with_registry(registry))

    assert catalog["updated_files"] == 0
    assert catalog["model_count"] == 1
    assert catalog["root"]["children"][0]["relative_path"] == "demo.engine"
    assert registry.list_projects() == []
    assert list(registry.data_dir.rglob("*.engine")) == []


def test_model_scan_is_read_only_and_does_not_import_or_copy_engine(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "demo.engine"
    source_path.write_bytes(b"engine")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")

    result = routes_models.scan_models(
        _request_with_registry(registry),
        force=True,
    )

    assert result["discovered_files"] == 1
    assert result["updated_files"] == 0
    assert result["model_count"] == 1
    assert registry.list_projects() == []
    assert list(registry.data_dir.rglob("*.engine")) == []


def test_catalog_engine_registration_references_original_without_asset_directories(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "demo.engine"
    source_path.write_bytes(b"engine")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")

    result = routes_models.register_catalog_model(
        _request_with_registry(registry),
        routes_models.CatalogRegisterRequest(relative_path="demo.engine"),
    )

    artifact = registry.get_artifact(result["artifact"]["id"])
    assert artifact is not None
    assert Path(artifact.path) == source_path.resolve()
    assert registry.resolve_artifact_path(artifact) == source_path.resolve()
    assert list(registry.data_dir.rglob("*.engine")) == []
    assert not (registry.data_dir / "demo").exists()

    catalog = routes_models.get_model_catalog(_request_with_registry(registry))
    model = catalog["root"]["children"][0]
    assert model["relative_path"] == "demo.engine"
    assert model["artifact_id"] == artifact.id

    repeated = routes_models.register_catalog_model(
        _request_with_registry(registry),
        routes_models.CatalogRegisterRequest(relative_path="demo.engine"),
    )
    assert repeated["created"] is False
    assert repeated["artifact"]["id"] == artifact.id
    assert len(registry.list_projects()) == 1


def test_catalog_engine_registration_rejects_path_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "outside.engine").write_bytes(b"engine")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")

    with pytest.raises(HTTPException) as error:
        routes_models.register_catalog_model(
            _request_with_registry(registry),
            routes_models.CatalogRegisterRequest(relative_path="../outside.engine"),
        )

    assert error.value.status_code == 400
    assert registry.list_projects() == []


def test_legacy_model_import_and_build_routes_are_not_registered(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config=RuntimeConfig())
    paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is not None:
            paths.add(str(path))
        included = getattr(route, "original_router", None)
        for child in getattr(included, "routes", ()):
            child_path = getattr(child, "path", None)
            if child_path is not None:
                paths.add(str(child_path))

    assert not any(path.startswith("/api/v1/models") for path in paths)
    assert "/api/models/artifacts/{artifact_id}/deepstream/prepare" not in paths


def test_model_catalog_deduplicates_same_relative_path_across_roots(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    (source_root / "same.engine").write_bytes(b"source")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")
    (registry.data_dir / "same.engine").write_bytes(b"managed")

    catalog = routes_models.get_model_catalog(_request_with_registry(registry))

    assert catalog["model_count"] == 1
    assert catalog["root"]["children"][0]["relative_path"] == "same.engine"
    assert catalog["root"]["children"][0]["size_bytes"] == len(b"source")


def test_model_catalog_keeps_same_content_external_engine_selectable_by_path(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    (source_root / "same.engine").write_bytes(b"engine")
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "same.engine",
        ["target"],
        "engine-probe-required",
    )
    managed_path = registry.data_dir / project.name / version.version / "same.engine"
    managed_path.write_bytes(b"engine")
    inspection = scanner.inspect_model_artifact(managed_path, force=True)
    artifact = registry.create_artifact(
        version.id,
        "engine",
        managed_path.name,
        inspection.sha256,
        "pending",
    )

    catalog = routes_models.get_model_catalog(_request_with_registry(registry))

    assert catalog["model_count"] == 2
    managed = catalog["root"]["children"][0]["children"][0]["children"][0]
    external = catalog["root"]["children"][1]
    assert managed["relative_path"] == "demo/v1/same.engine"
    assert managed["artifact_id"] == artifact.id
    assert external["relative_path"] == "same.engine"
    assert "artifact_id" not in external


def test_model_catalog_does_not_bind_source_to_missing_registered_file(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "external.engine"
    source_path.write_bytes(b"engine")
    checksum = scanner.inspect_model_artifact(source_path, force=True).sha256
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "data" / "models")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "missing.engine",
        ["target"],
        "engine-probe-required",
    )
    registry.create_artifact(
        version.id,
        "engine",
        "missing.engine",
        checksum,
        "ready",
    )

    catalog = routes_models.get_model_catalog(_request_with_registry(registry))

    model = catalog["root"]["children"][0]
    assert model["relative_path"] == "external.engine"
    assert "artifact_id" not in model
    assert "artifact_status" not in model


def test_artifact_inspection_reuses_hash_until_file_changes(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "demo.engine"
    model_path.write_bytes(b"first")
    hashes: list[Path] = []
    cache = getattr(scanner, "_MODEL_SCAN_CACHE", None)
    if cache is not None:
        cache.clear()

    def fake_sha(path: Path) -> str:
        hashes.append(Path(path))
        return f"sha256:{Path(path).stat().st_size}"

    monkeypatch.setattr(scanner, "sha256_file", fake_sha)

    scanner.inspect_model_artifact(model_path)
    scanner.inspect_model_artifact(model_path)
    model_path.write_bytes(b"second-version")
    scanner.inspect_model_artifact(model_path)

    assert hashes == [model_path, model_path]


def test_artifact_listing_includes_real_file_size(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "demo.engine",
        ["target"],
        "1x3x320x320",
    )
    artifact_path = registry.data_dir / project.name / version.version / "demo.engine"
    artifact_path.write_bytes(b"x" * 1_500_000)
    registry.create_artifact(
        version.id,
        "engine",
        "demo.engine",
        "sha256:test",
        "ready",
    )

    artifacts = routes_models.list_artifacts(_request_with_registry(registry), version.id)

    assert artifacts[0]["size_bytes"] == 1_500_000


def test_deepstream_recommendation_uses_engine_contract_instead_of_ui_shape_guess(
    tmp_path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    project = registry.create_project("custom", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "custom_v8_fp16.engine",
        ["target"],
        "1x3x640x640",
    )
    artifact_path = registry.data_dir / project.name / version.version / "custom_v8_fp16.engine"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        artifact_path.name,
        "pending",
        "pending",
    )
    request = _request_with_registry(registry)
    request.app.state.inference.probe = lambda *_args: {
        "loaded": True,
        # Flat fields may be stale registry/status values. The enumerated
        # TensorRT I/O contract must be authoritative.
        "input_name": "guessed_input",
        "input_shape": "1x3x640x640",
        "input_dtype": "float32",
        "output_name": "guessed_output",
        "output_shape": "1x8400x7",
        "output_dtype": "float32",
        "io_tensors": [
            {
                "name": "input_tensor",
                "shape": [1, 3, 320, 512],
                "dtype": "float16",
                "mode": "input",
            },
            {
                "name": "predictions",
                "shape": [1, 8400, 7],
                "dtype": "float16",
                "mode": "output",
            },
        ],
    }

    result = routes_models.recommend_deepstream_artifact(request, artifact.id)

    recommendation = result["recommendation"]
    assert recommendation["input_name"] == "input_tensor"
    assert recommendation["input_shape"] == [1, 3, 320, 512]
    assert recommendation["input_dtype"] == "float16"
    assert recommendation["output_name"] == "predictions"
    assert recommendation["output_shape"] == [1, 8400, 7]
    assert recommendation["output_dtype"] == "float16"
    assert recommendation["class_count"] == 3
    assert result["io_tensors"] == [
        {
            "name": "input_tensor",
            "shape": [1, 3, 320, 512],
            "dtype": "float16",
            "mode": "input",
        },
        {
            "name": "predictions",
            "shape": [1, 8400, 7],
            "dtype": "float16",
            "mode": "output",
        },
    ]
    assert result["class_names"] == ["class_0", "class_1", "class_2"]
    assert result["output_has_objectness"] is False
    assert result["sources"]["input_contract"] == "tensorrt_engine_probe"
    assert result["warnings"]
    assert not artifact_path.with_name("model.manifest.json").exists()

def test_publish_rejects_non_deepstream_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "demo.engine",
        ["target"],
        "1x3x640x640",
    )
    artifact_path = registry.data_dir / project.name / version.version / "demo.engine"
    artifact_path.write_bytes(b"engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "demo.engine",
        "sha256:pending",
        "pending",
    )

    class Inference:
        def prepare(self, path, classes, input_shape):
            assert path == artifact_path
            assert classes == ["target"]
            assert input_shape == "1x3x640x640"
            return "candidate", {"loaded": True, "input_shape": input_shape}

        def commit(self, candidate, **_kwargs):
            assert candidate == "candidate"

        def status(self):
            return {"loaded": True, "selected": "tensorrt"}

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                models=registry,
                inference=Inference(),
                runtime=SimpleNamespace(pipeline=None),
            )
        )
    )
    monkeypatch.setattr(
        routes_models,
        "_resume_runtime_pipeline_after_model_switch",
        lambda *_args: None,
    )

    with pytest.raises(HTTPException, match="deepstream_nvinfer"):
        routes_models.publish(
            request,
            project.id,
            PublishRequest(artifact_id=artifact.id),
        )

    assert registry.get_artifact(artifact.id).status == "pending"


def test_publish_deepstream_engine_auto_generates_single_runtime_manifest(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "demo.engine",
        ["body", "head"],
        "1x3x256x256",
    )
    artifact_path = registry.data_dir / project.name / version.version / "demo.engine"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        artifact_path.name,
        "sha256:pending",
        "pending",
    )
    config = RuntimeConfig()
    config.inference.backend = "deepstream_nvinfer"

    class Inference:
        def probe(self, *_args):
            return {
                "loaded": True,
                "input_name": "images",
                "input_shape": "1x3x256x256",
                "input_dtype": "float32",
                "output_name": "output0",
                "output_shape": "1x6x1344",
                "output_dtype": "float32",
            }

        def unload(self, _reason):
            return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                models=registry,
                inference=Inference(),
                config=config,
                runtime=SimpleNamespace(pipeline=None, running=False),
            )
        )
    )

    response = routes_models.publish(
        request,
        project.id,
        PublishRequest(artifact_id=artifact.id),
    )

    manifest_path = artifact_path.with_name(f"{artifact_path.name}.manifest.json")
    manifest = read_manifest(manifest_path)
    assert response["report"]["applied"] is True
    assert response["report"]["input_shape"] == "1x3x256x256"
    assert manifest.input.shape == [1, 3, 256, 256]
    assert manifest.output.shape == [1, 6, 1344]
    assert manifest.output.class_names == ["body", "head"]
    assert registry.get_artifact(artifact.id).status == "ready"


def test_publish_prepares_candidate_before_pausing_pipeline(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    deployment = Deployment(
        id=1,
        project_id=3,
        artifact_id=7,
        previous_artifact_id=None,
    )

    class Registry:
        def publish(self, *, project_id: int, artifact_id: int):
            assert project_id == 3
            assert artifact_id == 7
            events.append("publish")
            return deployment

    class Inference:
        def commit(self, candidate, **_kwargs):
            assert candidate == "candidate"
            events.append("commit")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                models=Registry(),
                inference=Inference(),
            )
        )
    )
    monkeypatch.setattr(routes_models, "_require_project", lambda *_args: None)
    monkeypatch.setattr(
        routes_models,
        "_resolve_runnable_artifact",
        lambda *_args, **_kwargs: (tmp_path / "demo.engine", ["target"], "1x3x640x640"),
    )

    def prepare(*_args, **_kwargs):
        events.append("prepare")
        return "candidate", {
            "loaded": True,
            "warmed": True,
            "input_shape": "1x3x640x640",
        }

    def pause(_request):
        events.append("pause")
        return True

    def resume(_request, should_resume):
        assert should_resume is True
        events.append("resume")

    monkeypatch.setattr(routes_models, "_prepare_runnable_artifact", prepare)
    monkeypatch.setattr(routes_models, "_pause_runtime_pipeline_for_model_switch", pause)
    monkeypatch.setattr(routes_models, "_resume_runtime_pipeline_after_model_switch", resume)
    monkeypatch.setattr(
        routes_models,
        "_sync_artifact_version_input_shape",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        routes_models,
        "_inference_status",
        lambda _request: {"loaded": True, "selected": "tensorrt"},
    )
    monkeypatch.setattr(
        routes_models,
        "set_profile_activation",
        lambda *_args, **_kwargs: None,
    )

    response = routes_models.publish(request, 3, PublishRequest(artifact_id=7))

    assert response["deployment"]["artifact_id"] == 7
    assert events == ["prepare", "pause", "publish", "commit", "resume"]
