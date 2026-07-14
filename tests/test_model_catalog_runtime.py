from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from novasight.api import routes_models
from novasight.api.routes_models import DeepStreamPrepareRequest, PublishRequest
from novasight.config import RuntimeConfig
from novasight.model_registry import ModelRegistry
from novasight.model_registry import read_manifest
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

    def fail_sync(_registry, **_kwargs):
        raise AssertionError("project listing must not scan model files")

    monkeypatch.setattr(routes_models, "_sync_models_directory", fail_sync)

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
    assert catalog["root"]["children"][0]["size_bytes"] == len(b"managed")


def test_model_catalog_prefers_registered_path_over_same_content_source(
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

    assert catalog["root"]["children"][0]["relative_path"] == "demo"
    model = catalog["root"]["children"][0]["children"][0]["children"][0]
    assert catalog["model_count"] == 1
    assert model["relative_path"] == "demo/v1/same.engine"
    assert model["artifact_id"] == artifact.id


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


def test_model_directory_sync_skips_unchanged_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    model_path = registry.data_dir / "demo.onnx"
    model_path.write_bytes(b"model")
    calls: list[Path] = []

    cache = getattr(routes_models, "_MODEL_SYNC_CACHE", None)
    if cache is not None:
        cache.clear()

    def record_sync(_registry, _root, path, *, force=False):
        del force
        calls.append(path)
        return True

    monkeypatch.setattr(routes_models, "_sync_model_file", record_sync)

    routes_models._sync_models_directory(registry)
    routes_models._sync_models_directory(registry)

    assert calls == [model_path.resolve()]


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


def test_managed_engine_without_manifest_is_downgraded_to_pending(tmp_path) -> None:
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
    inspection = scanner.inspect_model_artifact(artifact_path, force=True)
    artifact = registry.create_artifact(
        version.id,
        "engine",
        artifact_path.name,
        inspection.sha256,
        "ready",
    )

    assert routes_models._sync_model_file(
        registry,
        Path(registry.data_dir),
        artifact_path,
        force=True,
    )

    assert registry.get_artifact(artifact.id).status == "pending"


def test_prepare_deepstream_engine_writes_confirmed_manifest(tmp_path) -> None:
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
        "pending",
        "pending",
    )

    request = _request_with_registry(registry)
    request.app.state.inference.probe = lambda *_args: {
        "loaded": True,
        "input_name": "images",
        "input_shape": "1x3x256x256",
        "input_dtype": "float32",
        "output_name": "output0",
        "output_shape": "1x6x1344",
        "output_dtype": "float32",
    }

    result = routes_models.prepare_deepstream_artifact(
        request,
        artifact.id,
        DeepStreamPrepareRequest(
            model_id=project.name,
            display_name=project.name,
            input_shape=[1, 3, 256, 256],
            output_shape=[1, 6, 1344],
            class_count=2,
        ),
    )

    manifest = read_manifest(artifact_path.with_name("model.manifest.json"))
    assert result["status"] == "ready"
    assert result["nvinfer_config_owner"] == "runtime"
    assert registry.get_artifact(artifact.id).status == "ready"
    assert manifest.input.shape == [1, 3, 256, 256]
    assert manifest.output.shape == [1, 6, 1344]
    assert manifest.output.class_names == ["body", "head"]
    assert manifest.output.has_objectness is False


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

    prepared = routes_models.prepare_deepstream_artifact(
        request,
        artifact.id,
        DeepStreamPrepareRequest(**recommendation),
    )

    assert prepared["status"] == "ready"
    assert registry.get_version(version.id).classes == ["class_0", "class_1", "class_2"]
    assert read_manifest(artifact_path.with_name("model.manifest.json")).output.shape == [
        1,
        8400,
        7,
    ]


def test_prepare_deepstream_engine_rejects_confirmed_shape_that_differs_from_engine(
    tmp_path,
) -> None:
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
        "pending",
        "pending",
    )
    request = _request_with_registry(registry)
    request.app.state.inference.probe = lambda *_args: {
        "loaded": True,
        "input_name": "images",
        "input_shape": "1x3x256x256",
        "input_dtype": "float32",
        "output_name": "output0",
        "output_shape": "1x6x1344",
        "output_dtype": "float32",
    }

    try:
        routes_models.prepare_deepstream_artifact(
            request,
            artifact.id,
            DeepStreamPrepareRequest(
                model_id=project.name,
                display_name=project.name,
                input_shape=[1, 3, 320, 320],
                output_shape=[1, 6, 2100],
                class_count=2,
            ),
        )
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
        assert "does not match TensorRT engine" in str(getattr(exc, "detail", exc))
    else:
        raise AssertionError("mismatched confirmed contract must be rejected")

    assert not artifact_path.with_name("model.manifest.json").exists()


def test_model_replacement_keeps_deployed_artifact_file_immutable(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "demo.engine"
    source_path.write_bytes(b"first-engine")

    assert routes_models._sync_model_file(registry, source_root, source_path)
    project = registry.list_projects()[0]
    version = registry.list_versions(project.id)[0]
    first_artifact = registry.list_artifacts(version.id)[0]
    registry.update_artifact_status(first_artifact.id, "ready")
    registry.publish(project.id, first_artifact.id)
    first_asset = registry.data_dir / project.name / version.version / first_artifact.path

    source_path.write_bytes(b"replacement-engine")
    assert routes_models._sync_model_file(registry, source_root, source_path)

    versions = registry.list_versions(project.id)
    artifacts = [
        artifact
        for item in versions
        for artifact in registry.list_artifacts(item.id)
    ]
    active = registry.get_active_deployment()
    assert active is not None
    assert active.artifact_id == first_artifact.id
    assert first_asset.read_bytes() == b"first-engine"
    assert len(artifacts) == 2
    assert artifacts[-1].id != first_artifact.id
    assert artifacts[-1].checksum != first_artifact.checksum
    assert len(versions) == 2
    assert versions[-1].input_shape == version.input_shape
    assert versions[-1].classes == version.classes


def test_engine_replacement_inherits_classes_but_requires_a_fresh_shape_probe(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "assets")
    source_root = tmp_path / "models"
    source_root.mkdir()
    source_path = source_root / "demo.engine"
    sidecar_path = source_root / "demo.json"
    source_path.write_bytes(b"first-engine")
    sidecar_path.write_text(
        '{"classes": ["person", "head"], "input_shape": "1x3x320x320"}',
        encoding="utf-8",
    )

    assert routes_models._sync_model_file(registry, source_root, source_path)
    project = registry.list_projects()[0]
    first_version = registry.list_versions(project.id)[0]
    assert first_version.classes == ["person", "head"]
    assert first_version.input_shape == "engine-probe-required"

    sidecar_path.unlink()
    source_path.write_bytes(b"replacement-engine")
    assert routes_models._sync_model_file(registry, source_root, source_path)

    replacement_version = registry.list_versions(project.id)[-1]
    assert replacement_version.id != first_version.id
    assert replacement_version.classes == ["person", "head"]
    assert replacement_version.input_shape == "engine-probe-required"


def test_publish_rejects_pending_engine_without_validated_model_profile(
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

    with pytest.raises(HTTPException, match="ModelProfile"):
        routes_models.publish(
            request,
            project.id,
            PublishRequest(artifact_id=artifact.id),
        )

    assert registry.get_artifact(artifact.id).status == "pending"


def test_publish_deepstream_engine_does_not_guess_missing_model_profile(tmp_path) -> None:
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

    with pytest.raises(HTTPException, match="ModelProfile"):
        routes_models.publish(
            request,
            project.id,
            PublishRequest(artifact_id=artifact.id),
        )

    assert not artifact_path.with_name("model.manifest.json").exists()


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
