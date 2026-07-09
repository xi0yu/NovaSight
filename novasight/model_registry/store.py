from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import get_args

from .errors import (
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryValidationError,
)
from .schema import (
    ArtifactKind,
    ArtifactStatus,
    ConversionJob,
    ConversionJobStatus,
    ConversionTargetKind,
    Deployment,
    ModelArtifact,
    ModelProject,
    ModelVersion,
    SourceKind,
)


def _validate_path_component(value: str, label: str) -> None:
    if not value.strip():
        raise RegistryValidationError(f"{label} must not be empty")
    if Path(value).is_absolute():
        raise RegistryValidationError(f"{label} must not be an absolute path")
    if "/" in value or "\\" in value:
        raise RegistryValidationError(f"{label} must not contain path separators")
    if value in {".", ".."}:
        raise RegistryValidationError(f"{label} must be a single safe path component")


def _validate_choice(value: str, allowed: tuple[str, ...], label: str) -> None:
    if value not in allowed:
        choices = ", ".join(allowed)
        raise RegistryValidationError(f"{label} must be one of: {choices}")


def _validate_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise RegistryValidationError(f"{label} must be a list of strings")
    return list(value)


def _load_string_list_json(value: str, field_name: str) -> list[str]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(
            f"{field_name} must contain a JSON list of strings"
        ) from exc
    return _validate_string_list(decoded, field_name)


class ModelRegistry:
    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
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
                    source_kind TEXT NOT NULL
                        CHECK (source_kind IN ('pt', 'onnx')),
                    source_path TEXT NOT NULL,
                    classes_json TEXT NOT NULL,
                    input_shape TEXT NOT NULL,
                    UNIQUE(project_id, version)
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL REFERENCES model_versions(id),
                    kind TEXT NOT NULL
                        CHECK (kind IN ('pt', 'onnx', 'engine')),
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('pending', 'running', 'ready', 'failed'))
                );

                CREATE TABLE IF NOT EXISTS conversion_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL REFERENCES model_versions(id),
                    target_kind TEXT NOT NULL
                        CHECK (target_kind IN ('onnx', 'engine')),
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (
                            status IN ('pending', 'running', 'failed', 'succeeded')
                        ),
                    log TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id),
                    artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id),
                    previous_artifact_id INTEGER REFERENCES model_artifacts(id),
                    updated_seq INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS registry_sequence (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                """
            )
            self._ensure_deployment_columns(conn)
            self._ensure_registry_sequence(conn)

    def _ensure_deployment_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(deployments)").fetchall()
        }
        if "updated_seq" not in columns:
            conn.execute(
                "ALTER TABLE deployments ADD COLUMN updated_seq INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute("UPDATE deployments SET updated_seq = id WHERE updated_seq = 0")

    def _ensure_registry_sequence(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT COALESCE(MAX(updated_seq), 0) AS value FROM deployments"
        ).fetchone()
        deployment_value = int(row["value"])
        existing = conn.execute(
            "SELECT value FROM registry_sequence WHERE name = ?",
            ("deployment",),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO registry_sequence (name, value) VALUES (?, ?)",
                ("deployment", deployment_value),
            )
        elif int(existing["value"]) < deployment_value:
            conn.execute(
                "UPDATE registry_sequence SET value = ? WHERE name = ?",
                (deployment_value, "deployment"),
            )

    def _next_deployment_sequence(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(updated_seq), 0) AS value FROM deployments"
        ).fetchone()
        deployment_value = int(row["value"])
        conn.execute(
            """
            INSERT OR IGNORE INTO registry_sequence (name, value)
            VALUES (?, ?)
            """,
            ("deployment", deployment_value),
        )
        conn.execute(
            """
            UPDATE registry_sequence
            SET value = value + 1
            WHERE name = ?
            """,
            ("deployment",),
        )
        row = conn.execute(
            "SELECT value FROM registry_sequence WHERE name = ?",
            ("deployment",),
        ).fetchone()
        if row is None:
            raise RegistryValidationError("deployment sequence is not initialized")
        return int(row["value"])

    def create_project(self, name: str, description: str) -> ModelProject:
        _validate_path_component(name, "project name")
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM model_projects WHERE name = ?", (name,)
            ).fetchone()
            if existing is not None:
                raise RegistryConflictError(f"project already exists: {name}")
            try:
                cursor = conn.execute(
                    "INSERT INTO model_projects (name, description) VALUES (?, ?)",
                    (name, description),
                )
            except sqlite3.IntegrityError as exc:
                raise RegistryConflictError(f"project already exists: {name}") from exc
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
        _validate_path_component(version, "version")
        _validate_choice(source_kind, get_args(SourceKind), "source kind")
        classes = _validate_string_list(classes, "classes")
        with self._connect() as conn:
            project = conn.execute(
                "SELECT name FROM model_projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise RegistryNotFoundError(f"unknown project id: {project_id}")
            existing = conn.execute(
                """
                SELECT 1 FROM model_versions
                WHERE project_id = ? AND version = ?
                """,
                (project_id, version),
            ).fetchone()
            if existing is not None:
                raise RegistryConflictError(f"model version already exists: {version}")
            try:
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
            except sqlite3.IntegrityError as exc:
                raise RegistryConflictError(
                    f"model version already exists: {version}"
                ) from exc
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
            (self.data_dir / project_name / version).mkdir(
                parents=True, exist_ok=True
            )
            return model_version

    def update_version_input_shape(
        self,
        version_id: int,
        input_shape: str,
    ) -> ModelVersion:
        if not input_shape.strip():
            raise RegistryValidationError("input shape must not be empty")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if row is None:
                raise RegistryNotFoundError(f"unknown version id: {version_id}")
            conn.execute(
                "UPDATE model_versions SET input_shape = ? WHERE id = ?",
                (input_shape, version_id),
            )
            updated = conn.execute(
                "SELECT * FROM model_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if updated is None:
                raise RegistryNotFoundError(f"unknown version id: {version_id}")
            return self._version_from_row(updated)

    def create_artifact(
        self,
        version_id: int,
        kind: str,
        path: str,
        checksum: str,
        status: str,
    ) -> ModelArtifact:
        _validate_choice(kind, get_args(ArtifactKind), "artifact kind")
        _validate_choice(status, get_args(ArtifactStatus), "artifact status")
        with self._connect() as conn:
            asset_dir = self._version_asset_dir(conn, version_id)
            normalized_path = self._normalize_artifact_path(path, asset_dir)
            cursor = conn.execute(
                """
                INSERT INTO model_artifacts (version_id, kind, path, checksum, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (version_id, kind, normalized_path, checksum, status),
            )
            return ModelArtifact(
                int(cursor.lastrowid),
                version_id,
                kind,
                normalized_path,
                checksum,
                status,
            )

    def update_artifact_status(
        self,
        artifact_id: int,
        status: str,
        *,
        checksum: str | None = None,
    ) -> ModelArtifact:
        _validate_choice(status, get_args(ArtifactStatus), "artifact status")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
            if row is None:
                raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
            if checksum is None:
                conn.execute(
                    "UPDATE model_artifacts SET status = ? WHERE id = ?",
                    (status, artifact_id),
                )
            else:
                conn.execute(
                    "UPDATE model_artifacts SET checksum = ?, status = ? WHERE id = ?",
                    (checksum, status, artifact_id),
                )
            updated = conn.execute(
                "SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
            if updated is None:
                raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
            return self._artifact_from_row(updated)

    def create_conversion_job(
        self, version_id: int, target_kind: str, command: list[str]
    ) -> ConversionJob:
        _validate_choice(
            target_kind, get_args(ConversionTargetKind), "conversion target kind"
        )
        command = _validate_string_list(command, "command")
        with self._connect() as conn:
            self._preflight_version_id(conn, version_id)
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
        _validate_choice(
            status, get_args(ConversionJobStatus), "conversion job status"
        )
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversion_jobs SET status = ?, log = ? WHERE id = ?",
                (status, log, job_id),
            )
            if cursor.rowcount == 0:
                raise RegistryNotFoundError(f"unknown conversion job id: {job_id}")

    def get_conversion_job(self, job_id: int) -> ConversionJob | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._conversion_job_from_row(row) if row is not None else None

    def list_conversion_jobs(
        self, version_id: int | None = None
    ) -> list[ConversionJob]:
        with self._connect() as conn:
            if version_id is None:
                rows = conn.execute(
                    "SELECT * FROM conversion_jobs ORDER BY id"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM conversion_jobs
                    WHERE version_id = ?
                    ORDER BY id
                    """,
                    (version_id,),
                ).fetchall()
        return [self._conversion_job_from_row(row) for row in rows]

    def publish(self, project_id: int, artifact_id: int) -> Deployment:
        with self._connect() as conn:
            project = conn.execute(
                "SELECT 1 FROM model_projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise RegistryNotFoundError(f"unknown project id: {project_id}")
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
                raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
            if int(artifact["artifact_project_id"]) != project_id:
                raise RegistryValidationError("artifact does not belong to project")
            if artifact["status"] != "ready":
                raise RegistryValidationError("only ready artifacts can be published")

            deployment = conn.execute(
                "SELECT * FROM deployments WHERE project_id = ?", (project_id,)
            ).fetchone()
            if deployment is None:
                updated_seq = self._next_deployment_sequence(conn)
                cursor = conn.execute(
                    """
                    INSERT INTO deployments (
                        project_id,
                        artifact_id,
                        previous_artifact_id,
                        updated_seq
                    )
                    VALUES (?, ?, NULL, ?)
                    """,
                    (project_id, artifact_id, updated_seq),
                )
                return Deployment(int(cursor.lastrowid), project_id, artifact_id, None)

            if int(deployment["artifact_id"]) == artifact_id:
                updated_seq = self._next_deployment_sequence(conn)
                conn.execute(
                    """
                    UPDATE deployments
                    SET updated_seq = ?
                    WHERE project_id = ?
                    """,
                    (updated_seq, project_id),
                )
                updated = conn.execute(
                    "SELECT * FROM deployments WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
                return self._deployment_from_row(updated)

            previous_artifact_id = int(deployment["artifact_id"])
            updated_seq = self._next_deployment_sequence(conn)
            conn.execute(
                """
                UPDATE deployments
                SET artifact_id = ?, previous_artifact_id = ?, updated_seq = ?
                WHERE project_id = ?
                """,
                (artifact_id, previous_artifact_id, updated_seq, project_id),
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
                raise RegistryValidationError(
                    f"no deployment for project id: {project_id}"
                )
            previous_artifact_id = deployment["previous_artifact_id"]
            if previous_artifact_id is None:
                return self._deployment_from_row(deployment)

            artifact_id = int(deployment["artifact_id"])
            rollback_artifact_id = int(previous_artifact_id)
            updated_seq = self._next_deployment_sequence(conn)
            conn.execute(
                """
                UPDATE deployments
                SET artifact_id = ?, previous_artifact_id = ?, updated_seq = ?
                WHERE project_id = ?
                """,
                (rollback_artifact_id, artifact_id, updated_seq, project_id),
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

    def list_deployments(self) -> list[Deployment]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM deployments ORDER BY id").fetchall()
        return [self._deployment_from_row(row) for row in rows]

    def get_active_deployment(self) -> Deployment | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM deployments
                ORDER BY updated_seq DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return self._deployment_from_row(row) if row is not None else None

    def get_project(self, project_id: int) -> ModelProject | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_projects WHERE id = ?", (project_id,)
            ).fetchone()
        return self._project_from_row(row) if row is not None else None

    def list_projects(self) -> list[ModelProject]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM model_projects ORDER BY id").fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_version(self, version_id: int) -> ModelVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE id = ?", (version_id,)
            ).fetchone()
        return self._version_from_row(row) if row is not None else None

    def list_versions(self, project_id: int) -> list[ModelVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE project_id = ? ORDER BY id",
                (project_id,),
            ).fetchall()
        return [self._version_from_row(row) for row in rows]

    def get_artifact(self, artifact_id: int) -> ModelArtifact | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        return self._artifact_from_row(row) if row is not None else None

    def list_artifacts(self, version_id: int) -> list[ModelArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_artifacts WHERE version_id = ? ORDER BY id",
                (version_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def _project_from_row(self, row: sqlite3.Row) -> ModelProject:
        return ModelProject(int(row["id"]), str(row["name"]), str(row["description"]))

    def _version_asset_dir(self, conn: sqlite3.Connection, version_id: int) -> Path:
        row = conn.execute(
            """
            SELECT model_projects.name AS project_name, model_versions.version
            FROM model_versions
            JOIN model_projects ON model_projects.id = model_versions.project_id
            WHERE model_versions.id = ?
            """,
            (version_id,),
        ).fetchone()
        if row is None:
            raise RegistryNotFoundError(f"unknown version id: {version_id}")
        return self.data_dir / str(row["project_name"]) / str(row["version"])

    def _preflight_version_id(
        self, conn: sqlite3.Connection, version_id: int
    ) -> None:
        row = conn.execute(
            "SELECT 1 FROM model_versions WHERE id = ?", (version_id,)
        ).fetchone()
        if row is None:
            raise RegistryNotFoundError(f"unknown version id: {version_id}")

    def _normalize_artifact_path(self, path: str, asset_dir: Path) -> str:
        if not path.strip():
            raise RegistryValidationError("artifact path must not be empty")
        raw_path = Path(path)
        if raw_path.is_absolute():
            raise RegistryValidationError("artifact path must be relative")
        artifact_path = (asset_dir / raw_path).resolve(strict=False)
        asset_dir_path = asset_dir.resolve(strict=False)
        try:
            normalized_path = artifact_path.relative_to(asset_dir_path)
        except ValueError as exc:
            raise RegistryValidationError(
                "artifact path must be inside version asset directory"
            ) from exc
        if normalized_path == Path("."):
            raise RegistryValidationError("artifact path must not resolve to directory")
        return normalized_path.as_posix()

    def _version_from_row(self, row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            int(row["id"]),
            int(row["project_id"]),
            str(row["version"]),
            str(row["source_kind"]),
            str(row["source_path"]),
            _load_string_list_json(str(row["classes_json"]), "classes_json"),
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
            _load_string_list_json(str(row["command_json"]), "command_json"),
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
