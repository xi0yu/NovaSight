from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProject:
    id: int
    name: str
    description: str


@dataclass(frozen=True)
class ModelVersion:
    id: int
    project_id: int
    version: str
    source_kind: str
    source_path: str
    classes: list[str]
    input_shape: str


@dataclass(frozen=True)
class ModelArtifact:
    id: int
    version_id: int
    kind: str
    path: str
    checksum: str
    status: str


@dataclass(frozen=True)
class ConversionJob:
    id: int
    version_id: int
    target_kind: str
    command: list[str]
    status: str
    log: str


@dataclass(frozen=True)
class Deployment:
    id: int
    project_id: int
    artifact_id: int
    previous_artifact_id: int | None
