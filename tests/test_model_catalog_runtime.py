from pathlib import Path
from types import SimpleNamespace

from novasight.api import routes_models
from novasight.api.routes_models import PublishRequest
from novasight.model_registry import ModelRegistry
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


def test_engine_replacement_without_sidecar_inherits_model_metadata(tmp_path) -> None:
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
    assert first_version.input_shape == "1x3x320x320"

    sidecar_path.unlink()
    source_path.write_bytes(b"replacement-engine")
    assert routes_models._sync_model_file(registry, source_root, source_path)

    replacement_version = registry.list_versions(project.id)[-1]
    assert replacement_version.id != first_version.id
    assert replacement_version.classes == ["person", "head"]
    assert replacement_version.input_shape == "1x3x320x320"


def test_publish_validates_pending_engine_before_marking_it_ready(tmp_path, monkeypatch) -> None:
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

    response = routes_models.publish(request, project.id, PublishRequest(artifact_id=artifact.id))

    assert response["deployment"]["artifact_id"] == artifact.id
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
        return "candidate", {"loaded": True, "input_shape": "1x3x640x640"}

    def pause(_request):
        events.append("pause")
        return True

    def resume(_request, should_resume):
        assert should_resume is True
        events.append("resume")

    monkeypatch.setattr(routes_models, "_prepare_runnable_artifact", prepare)
    monkeypatch.setattr(routes_models, "_pause_runtime_pipeline_for_model_switch", pause)
    monkeypatch.setattr(routes_models, "_resume_runtime_pipeline_after_model_switch", resume)
    monkeypatch.setattr(routes_models, "_sync_artifact_version_input_shape", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        routes_models,
        "_inference_status",
        lambda _request: {"loaded": True, "selected": "tensorrt"},
    )

    response = routes_models.publish(request, 3, PublishRequest(artifact_id=7))

    assert response["deployment"]["artifact_id"] == 7
    assert events == ["prepare", "pause", "publish", "commit", "resume"]
