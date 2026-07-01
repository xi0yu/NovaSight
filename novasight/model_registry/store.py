from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .schema import (
    ConversionJob,
    Deployment,
    ModelArtifact,
    ModelProject,
    ModelVersion,
)


class ModelRegistry:
    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES model_projects(id),
                    version TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    classes_json TEXT NOT NULL,
                    input_shape TEXT NOT NULL,
                    UNIQUE(project_id, version)
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL REFERENCES model_versions(id),
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversion_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL REFERENCES model_versions(id),
                    target_kind TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    log TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id),
                    artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id),
                    previous_artifact_id INTEGER REFERENCES model_artifacts(id)
                );
                """
            )

    def create_project(self, name: str, description: str) -> ModelProject:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO model_projects (name, description) VALUES (?, ?)",
                (name, description),
            )
            project = ModelProject(int(cursor.lastrowid), name, description)
        (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        return project

    def create_version(
        self,
        project_id: int,
        version: str,
        source_kind: str,
        source_path: str,
        classes: list[str],
        input_shape: str,
    ) -> ModelVersion:
        with self._connect() as conn:
            project = conn.execute(
                "SELECT name FROM model_projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ValueError(f"unknown project id: {project_id}")
            cursor = conn.execute(
                """
                INSERT INTO model_versions (
                    project_id,
                    version,
                    source_kind,
                    source_path,
                    classes_json,
                    input_shape
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    version,
                    source_kind,
                    source_path,
                    json.dumps(classes),
                    input_shape,
                ),
            )
            model_version = ModelVersion(
                int(cursor.lastrowid),
                project_id,
                version,
                source_kind,
                source_path,
                list(classes),
                input_shape,
            )
            project_name = str(project["name"])
        (self.data_dir / project_name / version).mkdir(parents=True, exist_ok=True)
        return model_version

    def create_artifact(
        self,
        version_id: int,
        kind: str,
        path: str,
        checksum: str,
        status: str,
    ) -> ModelArtifact:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_artifacts (version_id, kind, path, checksum, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (version_id, kind, path, checksum, status),
            )
            return ModelArtifact(
                int(cursor.lastrowid), version_id, kind, path, checksum, status
            )

    def create_conversion_job(
        self, version_id: int, target_kind: str, command: list[str]
    ) -> ConversionJob:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversion_jobs (
                    version_id,
                    target_kind,
                    command_json,
                    status,
                    log
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (version_id, target_kind, json.dumps(command), "pending", ""),
            )
            return ConversionJob(
                int(cursor.lastrowid),
                version_id,
                target_kind,
                list(command),
                "pending",
                "",
            )

    def finish_conversion_job(self, job_id: int, status: str, log: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversion_jobs SET status = ?, log = ? WHERE id = ?",
                (status, log, job_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"unknown conversion job id: {job_id}")

    def get_conversion_job(self, job_id: int) -> ConversionJob | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._conversion_job_from_row(row) if row is not None else None

    def publish(self, project_id: int, artifact_id: int) -> Deployment:
        with self._connect() as conn:
            artifact = conn.execute(
                """
                SELECT
                    model_artifacts.*,
                    model_versions.project_id AS artifact_project_id
                FROM model_artifacts
                JOIN model_versions ON model_versions.id = model_artifacts.version_id
                WHERE model_artifacts.id = ?
                """,
                (artifact_id,),
            ).fetchone()
            if artifact is None:
                raise ValueError(f"unknown artifact id: {artifact_id}")
            if int(artifact["artifact_project_id"]) != project_id:
                raise ValueError("artifact does not belong to project")
            if artifact["status"] != "ready":
                raise ValueError("only ready artifacts can be published")

            deployment = conn.execute(
                "SELECT * FROM deployments WHERE project_id = ?", (project_id,)
            ).fetchone()
            if deployment is None:
                cursor = conn.execute(
                    """
                    INSERT INTO deployments (
                        project_id,
                        artifact_id,
                        previous_artifact_id
                    )
                    VALUES (?, ?, NULL)
                    """,
                    (project_id, artifact_id),
                )
                return Deployment(int(cursor.lastrowid), project_id, artifact_id, None)

            previous_artifact_id = int(deployment["artifact_id"])
            conn.execute(
                """
                UPDATE deployments
                SET artifact_id = ?, previous_artifact_id = ?
                WHERE project_id = ?
                """,
                (artifact_id, previous_artifact_id, project_id),
            )
            return Deployment(
                int(deployment["id"]), project_id, artifact_id, previous_artifact_id
            )

    def rollback(self, project_id: int) -> Deployment:
        with self._connect() as conn:
            deployment = conn.execute(
                "SELECT * FROM deployments WHERE project_id = ?", (project_id,)
            ).fetchone()
            if deployment is None:
                raise ValueError(f"no deployment for project id: {project_id}")
            previous_artifact_id = deployment["previous_artifact_id"]
            if previous_artifact_id is None:
                raise ValueError("no previous artifact to roll back to")

            artifact_id = int(deployment["artifact_id"])
            rollback_artifact_id = int(previous_artifact_id)
            conn.execute(
                """
                UPDATE deployments
                SET artifact_id = ?, previous_artifact_id = ?
                WHERE project_id = ?
                """,
                (rollback_artifact_id, artifact_id, project_id),
            )
            return Deployment(
                int(deployment["id"]), project_id, rollback_artifact_id, artifact_id
            )

    def get_deployment(self, project_id: int) -> Deployment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM deployments WHERE project_id = ?", (project_id,)
            ).fetchone()
        return self._deployment_from_row(row) if row is not None else None

    def list_projects(self) -> list[ModelProject]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM model_projects ORDER BY id").fetchall()
        return [self._project_from_row(row) for row in rows]

    def list_versions(self, project_id: int) -> list[ModelVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE project_id = ? ORDER BY id",
                (project_id,),
            ).fetchall()
        return [self._version_from_row(row) for row in rows]

    def list_artifacts(self, version_id: int) -> list[ModelArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_artifacts WHERE version_id = ? ORDER BY id",
                (version_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def _project_from_row(self, row: sqlite3.Row) -> ModelProject:
        return ModelProject(int(row["id"]), str(row["name"]), str(row["description"]))

    def _version_from_row(self, row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            int(row["id"]),
            int(row["project_id"]),
            str(row["version"]),
            str(row["source_kind"]),
            str(row["source_path"]),
            json.loads(str(row["classes_json"])),
            str(row["input_shape"]),
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ModelArtifact:
        return ModelArtifact(
            int(row["id"]),
            int(row["version_id"]),
            str(row["kind"]),
            str(row["path"]),
            str(row["checksum"]),
            str(row["status"]),
        )

    def _conversion_job_from_row(self, row: sqlite3.Row) -> ConversionJob:
        return ConversionJob(
            int(row["id"]),
            int(row["version_id"]),
            str(row["target_kind"]),
            json.loads(str(row["command_json"])),
            str(row["status"]),
            str(row["log"]),
        )

    def _deployment_from_row(self, row: sqlite3.Row) -> Deployment:
        previous_artifact_id = row["previous_artifact_id"]
        return Deployment(
            int(row["id"]),
            int(row["project_id"]),
            int(row["artifact_id"]),
            int(previous_artifact_id) if previous_artifact_id is not None else None,
        )
