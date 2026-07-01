import sqlite3
from pathlib import Path

import pytest

from novasight.model_registry import (
    ModelRegistry,
    RegistryNotFoundError,
    RegistryValidationError,
)


def _create_version(registry: ModelRegistry, project_id: int, version: str = "v1"):
    return registry.create_version(
        project_id=project_id,
        version=version,
        source_kind="onnx",
        source_path=f"/tmp/{version}.onnx",
        classes=["person"],
        input_shape="1x3x640x640",
    )


def _artifact_path(data_dir: Path, project: str = "demo", version: str = "v1") -> str:
    return "model.engine"


def _deployment_seq(db_path: Path, project_id: int) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT updated_seq FROM deployments WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_model_registry_publish_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "novasight.db"
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=db_path, data_dir=data_dir)

    project = registry.create_project("person-detector", "Person detector")
    v1 = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="onnx",
        source_path="/tmp/person-v1.onnx",
        classes=["person"],
        input_shape="1x3x640x640",
    )
    artifact1 = registry.create_artifact(
        version_id=v1.id,
        kind="onnx",
        path="model.onnx",
        checksum="sha256:v1",
        status="ready",
    )
    registry.publish(project.id, artifact1.id)

    active = registry.get_deployment(project.id)
    assert active is not None
    assert active.artifact_id == artifact1.id

    v2 = registry.create_version(
        project_id=project.id,
        version="v2",
        source_kind="onnx",
        source_path="/tmp/person-v2.onnx",
        classes=["person", "head"],
        input_shape="1x3x960x960",
    )
    artifact2 = registry.create_artifact(
        version_id=v2.id,
        kind="engine",
        path="model.engine",
        checksum="sha256:v2",
        status="ready",
    )
    registry.publish(project.id, artifact2.id)
    registry.rollback(project.id)

    rolled_back = registry.get_deployment(project.id)
    assert rolled_back is not None
    assert rolled_back.artifact_id == artifact1.id


def test_conversion_job_tracks_status_and_log(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="pt",
        source_path="/tmp/demo.pt",
        classes=["target"],
        input_shape="dynamic",
    )

    job = registry.create_conversion_job(
        version_id=version.id,
        target_kind="engine",
        command=["trtexec", "--onnx=demo.onnx", "--saveEngine=demo.engine"],
    )
    registry.finish_conversion_job(job.id, status="failed", log="TensorRT parse error")

    loaded = registry.get_conversion_job(job.id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert "TensorRT parse error" in loaded.log


@pytest.mark.parametrize("name", ["../escape", "/abs", "a/b", "a\\b", "  "])
def test_project_name_rejects_unsafe_path_components(
    tmp_path: Path, name: str
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)

    with pytest.raises(ValueError, match="project name"):
        registry.create_project(name, "")

    assert registry.list_projects() == []
    assert not (tmp_path / "escape").exists()
    assert not (data_dir / "a").exists()
    assert not (data_dir / "a\\b").exists()


@pytest.mark.parametrize("version", ["../escape", "/abs", "a/b", "a\\b", "  "])
def test_version_rejects_unsafe_path_components(
    tmp_path: Path, version: str
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")

    with pytest.raises(ValueError, match="version"):
        _create_version(registry, project.id, version=version)

    assert registry.list_versions(project.id) == []
    assert not (tmp_path / "escape").exists()
    assert not (data_dir / "demo" / "a").exists()
    assert not (data_dir / "demo" / "a\\b").exists()


def test_create_version_creates_version_asset_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("person-detector", "")

    _create_version(registry, project.id, version="v1")

    assert (data_dir / "person-detector" / "v1").is_dir()


def test_publish_rejects_artifacts_that_are_not_ready(tmp_path: Path) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=data_dir,
    )
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    artifact = registry.create_artifact(
        version_id=version.id,
        kind="engine",
        path=_artifact_path(data_dir),
        checksum="sha256:not-ready",
        status="pending",
    )

    with pytest.raises(ValueError, match="ready"):
        registry.publish(project.id, artifact.id)

    assert registry.get_deployment(project.id) is None


def test_json_fields_round_trip_through_new_registry_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "novasight.db"
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=db_path, data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="pt",
        source_path="/tmp/demo.pt",
        classes=["person", "head"],
        input_shape="dynamic",
    )
    job = registry.create_conversion_job(
        version_id=version.id,
        target_kind="engine",
        command=["trtexec", "--onnx=demo.onnx", "--saveEngine=demo.engine"],
    )

    reloaded = ModelRegistry(db_path=db_path, data_dir=data_dir)

    loaded_version = reloaded.list_versions(project.id)[0]
    loaded_job = reloaded.get_conversion_job(job.id)
    assert loaded_version.classes == ["person", "head"]
    assert loaded_job is not None
    assert loaded_job.command == [
        "trtexec",
        "--onnx=demo.onnx",
        "--saveEngine=demo.engine",
    ]


def test_list_methods_return_registry_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=data_dir,
    )
    project = registry.create_project("demo", "Demo project")
    version = _create_version(registry, project.id)
    artifact = registry.create_artifact(
        version_id=version.id,
        kind="onnx",
        path=_artifact_path(data_dir),
        checksum="sha256:model",
        status="ready",
    )
    job = registry.create_conversion_job(
        version_id=version.id,
        target_kind="engine",
        command=["trtexec"],
    )

    assert registry.list_projects() == [project]
    assert registry.list_versions(project.id) == [version]
    assert registry.list_artifacts(version.id) == [artifact]
    assert registry.list_conversion_jobs() == [job]
    assert registry.list_conversion_jobs(version.id) == [job]


def test_list_conversion_jobs_can_scope_by_version(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")
    v1 = _create_version(registry, project.id, version="v1")
    v2 = _create_version(registry, project.id, version="v2")
    job1 = registry.create_conversion_job(v1.id, "engine", ["trtexec", "v1"])
    job2 = registry.create_conversion_job(v2.id, "engine", ["trtexec", "v2"])

    assert registry.list_conversion_jobs() == [job1, job2]
    assert registry.list_conversion_jobs(v1.id) == [job1]
    assert registry.list_conversion_jobs(v2.id) == [job2]


def test_rollback_without_previous_deployment_leaves_current_unchanged(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=data_dir,
    )
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    artifact = registry.create_artifact(
        version_id=version.id,
        kind="onnx",
        path=_artifact_path(data_dir),
        checksum="sha256:model",
        status="ready",
    )
    deployment = registry.publish(project.id, artifact.id)

    rolled_back = registry.rollback(project.id)

    assert rolled_back == deployment
    assert registry.get_deployment(project.id) == deployment


def test_duplicate_publish_preserves_previous_artifact_for_rollback(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    v1 = _create_version(registry, project.id, version="v1")
    v2 = _create_version(registry, project.id, version="v2")
    artifact1 = registry.create_artifact(
        version_id=v1.id,
        kind="engine",
        path=_artifact_path(data_dir, version="v1"),
        checksum="sha256:v1",
        status="ready",
    )
    artifact2 = registry.create_artifact(
        version_id=v2.id,
        kind="engine",
        path=_artifact_path(data_dir, version="v2"),
        checksum="sha256:v2",
        status="ready",
    )

    registry.publish(project.id, artifact1.id)
    deployed_v2 = registry.publish(project.id, artifact2.id)
    duplicate_publish = registry.publish(project.id, artifact2.id)
    rolled_back = registry.rollback(project.id)

    assert duplicate_publish == deployed_v2
    assert rolled_back.artifact_id == artifact1.id


def test_deployment_sequence_tracks_publish_duplicate_publish_and_rollback(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "novasight.db"
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=db_path, data_dir=data_dir)
    project_a = registry.create_project("demo-a", "")
    a_v1 = _create_version(registry, project_a.id, version="v1")
    a_v2 = _create_version(registry, project_a.id, version="v2")
    artifact_a1 = registry.create_artifact(
        version_id=a_v1.id,
        kind="engine",
        path=_artifact_path(data_dir, project="demo-a", version="v1"),
        checksum="sha256:a1",
        status="ready",
    )
    artifact_a2 = registry.create_artifact(
        version_id=a_v2.id,
        kind="engine",
        path=_artifact_path(data_dir, project="demo-a", version="v2"),
        checksum="sha256:a2",
        status="ready",
    )
    project_b = registry.create_project("demo-b", "")
    b_v1 = _create_version(registry, project_b.id, version="v1")
    artifact_b1 = registry.create_artifact(
        version_id=b_v1.id,
        kind="engine",
        path=_artifact_path(data_dir, project="demo-b", version="v1"),
        checksum="sha256:b1",
        status="ready",
    )

    registry.publish(project_a.id, artifact_a1.id)
    first_seq = _deployment_seq(db_path, project_a.id)
    registry.publish(project_b.id, artifact_b1.id)
    second_seq = _deployment_seq(db_path, project_b.id)
    assert registry.get_active_deployment().project_id == project_b.id

    registry.publish(project_a.id, artifact_a1.id)
    duplicate_seq = _deployment_seq(db_path, project_a.id)
    assert registry.get_active_deployment().project_id == project_a.id

    registry.publish(project_a.id, artifact_a2.id)
    update_seq = _deployment_seq(db_path, project_a.id)
    registry.rollback(project_a.id)
    rollback_seq = _deployment_seq(db_path, project_a.id)

    assert [first_seq, second_seq, duplicate_seq, update_seq, rollback_seq] == [
        1,
        2,
        3,
        4,
        5,
    ]
    active = registry.get_active_deployment()
    assert active is not None
    assert active.project_id == project_a.id
    assert active.artifact_id == artifact_a1.id


def test_deployment_sequence_missing_row_self_heals_on_publish_and_rollback(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "novasight.db"
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=db_path, data_dir=data_dir)
    project = registry.create_project("demo", "")
    v1 = _create_version(registry, project.id, version="v1")
    v2 = _create_version(registry, project.id, version="v2")
    artifact1 = registry.create_artifact(
        version_id=v1.id,
        kind="engine",
        path="v1.engine",
        checksum="sha256:v1",
        status="ready",
    )
    artifact2 = registry.create_artifact(
        version_id=v2.id,
        kind="engine",
        path="v2.engine",
        checksum="sha256:v2",
        status="ready",
    )
    registry.publish(project.id, artifact1.id)
    assert _deployment_seq(db_path, project.id) == 1

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM registry_sequence WHERE name = ?", ("deployment",))

    registry.publish(project.id, artifact2.id)
    assert _deployment_seq(db_path, project.id) == 2

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM registry_sequence WHERE name = ?", ("deployment",))

    registry.rollback(project.id)
    assert _deployment_seq(db_path, project.id) == 3
    assert registry.get_deployment(project.id).artifact_id == artifact1.id


def test_publish_rejects_unknown_project_before_artifact_ownership(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    artifact = registry.create_artifact(
        version_id=version.id,
        kind="engine",
        path=_artifact_path(data_dir),
        checksum="sha256:model",
        status="ready",
    )

    with pytest.raises(RegistryNotFoundError, match="unknown project id: 999"):
        registry.publish(999, artifact.id)

    assert registry.get_deployment(project.id) is None


def test_create_artifact_rejects_absolute_path_inside_version_directory(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    inside_path = data_dir / "demo" / "v1" / "model.engine"

    with pytest.raises(RegistryValidationError, match="relative"):
        registry.create_artifact(
            version_id=version.id,
            kind="engine",
            path=str(inside_path),
            checksum="sha256:model",
            status="ready",
        )

    assert registry.list_artifacts(version.id) == []


def test_create_artifact_stores_normalized_contained_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    raw_path = Path("nested") / ".." / "model.engine"

    artifact = registry.create_artifact(
        version_id=version.id,
        kind="engine",
        path=str(raw_path),
        checksum="sha256:model",
        status="ready",
    )

    assert artifact.path == "model.engine"
    assert registry.list_artifacts(version.id)[0].path == "model.engine"


@pytest.mark.parametrize(
    ("raw_path", "expected_parts"),
    [
        ("model.engine", ("model.engine",)),
        ("nested/model.engine", ("nested", "model.engine")),
    ],
)
def test_create_artifact_accepts_asset_relative_paths(
    tmp_path: Path, raw_path: str, expected_parts: tuple[str, ...]
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    expected_path = Path(*expected_parts).as_posix()

    artifact = registry.create_artifact(
        version_id=version.id,
        kind="engine",
        path=raw_path,
        checksum="sha256:model",
        status="ready",
    )

    assert artifact.path == expected_path


@pytest.mark.parametrize("raw_path", ["", "  ", ".", "nested/.."])
def test_create_artifact_rejects_empty_or_directory_paths(
    tmp_path: Path, raw_path: str
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)

    with pytest.raises(RegistryValidationError, match="artifact path"):
        registry.create_artifact(
            version_id=version.id,
            kind="engine",
            path=raw_path,
            checksum="sha256:model",
            status="ready",
        )

    assert registry.list_artifacts(version.id) == []


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path, data_dir: tmp_path / "outside.engine",
        lambda tmp_path, data_dir: Path("../outside.engine"),
        lambda tmp_path, data_dir: data_dir / "demo" / "v1" / ".." / "outside.engine",
    ],
)
def test_create_artifact_rejects_paths_outside_version_directory(
    tmp_path: Path, path_factory
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)

    with pytest.raises(ValueError, match="artifact path"):
        registry.create_artifact(
            version_id=version.id,
            kind="engine",
            path=str(path_factory(tmp_path, data_dir)),
            checksum="sha256:model",
            status="ready",
        )

    assert registry.list_artifacts(version.id) == []


def test_create_version_does_not_commit_row_when_directory_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    original_mkdir = Path.mkdir

    def fail_version_dir(
        self,
        mode=0o777,
        parents=False,
        exist_ok=False,
    ):
        if self == data_dir / "demo" / "v1":
            raise OSError("cannot create version dir")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_version_dir)

    with pytest.raises(OSError, match="cannot create version dir"):
        _create_version(registry, project.id)

    assert registry.list_versions(project.id) == []


def test_create_project_does_not_commit_row_when_directory_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    original_mkdir = Path.mkdir

    def fail_project_dir(
        self,
        mode=0o777,
        parents=False,
        exist_ok=False,
    ):
        if self == data_dir / "demo":
            raise OSError("cannot create project dir")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fail_project_dir)

    with pytest.raises(OSError, match="cannot create project dir"):
        registry.create_project("demo", "")

    assert registry.list_projects() == []


def test_rollback_without_deployment_raises_clear_error(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")

    with pytest.raises(ValueError, match="no deployment"):
        registry.rollback(project.id)


def test_create_version_rejects_invalid_source_kind(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")

    with pytest.raises(ValueError, match="source kind"):
        registry.create_version(
            project_id=project.id,
            version="v1",
            source_kind="bogus",
            source_path="/tmp/model.bogus",
            classes=["person"],
            input_shape="dynamic",
        )

    assert registry.list_versions(project.id) == []


@pytest.mark.parametrize("classes", ["person", ["person", 7], [object()]])
def test_create_version_rejects_invalid_classes(
    tmp_path: Path, classes
) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")

    with pytest.raises(ValueError, match="classes"):
        registry.create_version(
            project_id=project.id,
            version="v1",
            source_kind="onnx",
            source_path="/tmp/model.onnx",
            classes=classes,
            input_shape="dynamic",
        )

    assert registry.list_versions(project.id) == []


@pytest.mark.parametrize(
    ("kind", "status", "match"),
    [
        ("bogus", "ready", "artifact kind"),
        ("engine", "redy", "artifact status"),
    ],
)
def test_create_artifact_rejects_invalid_kind_or_status(
    tmp_path: Path, kind: str, status: str, match: str
) -> None:
    data_dir = tmp_path / "models"
    registry = ModelRegistry(db_path=tmp_path / "novasight.db", data_dir=data_dir)
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)

    with pytest.raises(ValueError, match=match):
        registry.create_artifact(
            version_id=version.id,
            kind=kind,
            path="model.engine",
            checksum="sha256:model",
            status=status,
        )

    assert registry.list_artifacts(version.id) == []


def test_create_conversion_job_rejects_invalid_target_kind(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)

    with pytest.raises(ValueError, match="conversion target kind"):
        registry.create_conversion_job(version.id, "bogus", ["convert"])

    assert registry.list_conversion_jobs() == []


def test_create_conversion_job_rejects_unknown_version_id(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )

    with pytest.raises(ValueError, match="unknown version id: 42"):
        registry.create_conversion_job(42, "engine", ["convert"])

    assert registry.list_conversion_jobs() == []


@pytest.mark.parametrize("command", ["convert", ["convert", 7], [object()]])
def test_create_conversion_job_rejects_invalid_command(
    tmp_path: Path, command
) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)

    with pytest.raises(ValueError, match="command"):
        registry.create_conversion_job(version.id, "engine", command)

    assert registry.list_conversion_jobs() == []


def test_finish_conversion_job_rejects_invalid_status(tmp_path: Path) -> None:
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    job = registry.create_conversion_job(version.id, "engine", ["convert"])

    with pytest.raises(ValueError, match="conversion job status"):
        registry.finish_conversion_job(job.id, status="faild", log="typo")

    assert registry.get_conversion_job(job.id) == job


@pytest.mark.parametrize(
    "classes_json",
    ["not json", '{"class": "person"}', '["ok", 7]'],
)
def test_list_versions_rejects_invalid_classes_json(
    tmp_path: Path, classes_json: str
) -> None:
    db_path = tmp_path / "novasight.db"
    registry = ModelRegistry(db_path=db_path, data_dir=tmp_path / "models")
    project = registry.create_project("demo", "")
    _create_version(registry, project.id)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE model_versions SET classes_json = ?",
            (classes_json,),
        )

    with pytest.raises(ValueError, match="classes_json"):
        registry.list_versions(project.id)


@pytest.mark.parametrize(
    "command_json",
    ["not json", '{"cmd": "convert"}', '["ok", 7]'],
)
def test_conversion_job_reads_reject_invalid_command_json(
    tmp_path: Path, command_json: str
) -> None:
    db_path = tmp_path / "novasight.db"
    registry = ModelRegistry(db_path=db_path, data_dir=tmp_path / "models")
    project = registry.create_project("demo", "")
    version = _create_version(registry, project.id)
    job = registry.create_conversion_job(version.id, "engine", ["convert"])
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE conversion_jobs SET command_json = ? WHERE id = ?",
            (command_json, job.id),
        )

    with pytest.raises(ValueError, match="command_json"):
        registry.get_conversion_job(job.id)

    with pytest.raises(ValueError, match="command_json"):
        registry.list_conversion_jobs()
