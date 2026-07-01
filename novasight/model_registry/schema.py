from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SourceKind = Literal["pt", "onnx"]
ArtifactKind = Literal["pt", "onnx", "engine"]
ArtifactStatus = Literal["pending", "running", "ready", "failed"]
ConversionTargetKind = Literal["onnx", "engine"]
ConversionJobStatus = Literal["pending", "running", "failed", "succeeded"]


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
    source_kind: SourceKind
    source_path: str
    classes: list[str]
    input_shape: str


@dataclass(frozen=True)
class ModelArtifact:
    id: int
    version_id: int
    kind: ArtifactKind
    path: str
    checksum: str
    status: ArtifactStatus


@dataclass(frozen=True)
class ConversionJob:
    id: int
    version_id: int
    target_kind: ConversionTargetKind
    command: list[str]
    status: ConversionJobStatus
    log: str


@dataclass(frozen=True)
class Deployment:
    id: int
    project_id: int
    artifact_id: int
    previous_artifact_id: int | None
