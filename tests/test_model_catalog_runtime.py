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
