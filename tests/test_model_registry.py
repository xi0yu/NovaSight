from pathlib import Path

from novasight.model_registry import ModelRegistry


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
        path=str(data_dir / "person-detector" / "v1" / "model.onnx"),
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
        path=str(data_dir / "person-detector" / "v2" / "model.engine"),
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
