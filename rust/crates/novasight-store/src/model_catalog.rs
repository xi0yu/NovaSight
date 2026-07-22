use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct SqliteModelCatalog {
    path: PathBuf,
    model_root: PathBuf,
    catalog_roots: Vec<PathBuf>,
    catalog_cache: Arc<Mutex<Option<(Instant, ModelCatalogResponse)>>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelProject {
    pub id: i64,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: i64,
    pub project_id: i64,
    pub version: String,
    pub source_kind: String,
    pub source_path: String,
    pub classes: Vec<String>,
    pub input_shape: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub id: i64,
    pub version_id: i64,
    pub kind: String,
    pub path: String,
    pub checksum: String,
    pub status: String,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogResponse {
    pub root: ModelCatalogDirectory,
    pub directory_count: usize,
    pub model_count: usize,
    pub discovered_files: usize,
    pub updated_files: usize,
    pub cache_hits: usize,
    pub force: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogDirectory {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    pub relative_path: String,
    pub children: Vec<ModelCatalogNode>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelCatalogNode {
    Directory(ModelCatalogDirectory),
    Model(ModelCatalogModel),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogModel {
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    pub relative_path: String,
    pub kind: String,
    pub size_bytes: u64,
    pub scan_status: String,
    pub scan_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_status: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConversionJob {
    pub id: i64,
    pub version_id: i64,
    pub target_kind: String,
    pub command: Vec<String>,
    pub status: String,
    pub log: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Deployment {
    pub id: i64,
    pub project_id: i64,
    pub artifact_id: i64,
    pub previous_artifact_id: Option<i64>,
    pub updated_seq: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogSnapshot {
    pub projects: Vec<ModelProject>,
    pub versions: Vec<ModelVersion>,
    pub artifacts: Vec<ModelArtifact>,
    pub jobs: Vec<ConversionJob>,
    pub deployments: Vec<Deployment>,
    pub active_deployment: Option<Deployment>,
}

impl SqliteModelCatalog {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ModelCatalogError> {
        let path = path.into();
        let model_root = path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("models");
        Self::open_with_model_root(path, model_root)
    }

    pub fn open_with_model_root(
        path: impl Into<PathBuf>,
        model_root: impl Into<PathBuf>,
    ) -> Result<Self, ModelCatalogError> {
        let catalog = Self {
            path: path.into(),
            model_root: model_root.into(),
            catalog_roots: Vec::new(),
            catalog_cache: Arc::new(Mutex::new(None)),
        };
        let mut catalog = catalog;
        catalog.catalog_roots.push(PathBuf::from("models"));
        if catalog.model_root != Path::new("models") {
            catalog.catalog_roots.push(catalog.model_root.clone());
        }
        let mut connection = catalog.connect()?;
        migrate(&mut connection)?;
        Ok(catalog)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn list_projects(&self) -> Result<Vec<ModelProject>, ModelCatalogError> {
        let connection = self.connect()?;
        collect(
            &connection,
            "SELECT id, name, description FROM model_projects ORDER BY id",
            [],
            |row| {
                Ok(ModelProject {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                })
            },
        )
    }

    pub fn list_versions(&self, project_id: i64) -> Result<Vec<ModelVersion>, ModelCatalogError> {
        let connection = self.connect()?;
        require_project(&connection, project_id)?;
        collect(
            &connection,
            "SELECT id, project_id, version, source_kind, source_path, classes_json, input_shape FROM model_versions WHERE project_id = ?1 ORDER BY id",
            [project_id],
            model_version_from_row,
        )
    }

    pub fn list_artifacts(&self, version_id: i64) -> Result<Vec<ModelArtifact>, ModelCatalogError> {
        let connection = self.connect()?;
        require_version(&connection, version_id)?;
        let (project_name, version_name) = connection
            .query_row(
                "SELECT model_projects.name, model_versions.version FROM model_versions JOIN model_projects ON model_projects.id = model_versions.project_id WHERE model_versions.id = ?1",
                [version_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(ModelCatalogError::Sqlite)?;
        let mut artifacts = collect(
            &connection,
            "SELECT id, version_id, kind, path, checksum, status FROM model_artifacts WHERE version_id = ?1 ORDER BY id",
            [version_id],
            model_artifact_from_row,
        )?;
        for artifact in &mut artifacts {
            artifact.size_bytes = self
                .resolve_artifact_path(&project_name, &version_name, &artifact.path)
                .metadata()
                .ok()
                .map(|metadata| metadata.len());
        }
        Ok(artifacts)
    }

    pub fn list_jobs(
        &self,
        version_id: Option<i64>,
    ) -> Result<Vec<ConversionJob>, ModelCatalogError> {
        let connection = self.connect()?;
        match version_id {
            Some(version_id) => {
                require_version(&connection, version_id)?;
                collect(
                    &connection,
                    "SELECT id, version_id, target_kind, command_json, status, log FROM conversion_jobs WHERE version_id = ?1 ORDER BY id",
                    [version_id],
                    conversion_job_from_row,
                )
            }
            None => collect(
                &connection,
                "SELECT id, version_id, target_kind, command_json, status, log FROM conversion_jobs ORDER BY id",
                [],
                conversion_job_from_row,
            ),
        }
    }

    pub fn snapshot(&self) -> Result<ModelCatalogSnapshot, ModelCatalogError> {
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(ModelCatalogError::Sqlite)?;
        let projects = collect(
            &transaction,
            "SELECT id, name, description FROM model_projects ORDER BY id",
            [],
            |row| {
                Ok(ModelProject {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                })
            },
        )?;
        let versions = collect(
            &transaction,
            "SELECT id, project_id, version, source_kind, source_path, classes_json, input_shape FROM model_versions ORDER BY id",
            [],
            model_version_from_row,
        )?;
        let artifacts = collect(
            &transaction,
            "SELECT id, version_id, kind, path, checksum, status FROM model_artifacts ORDER BY id",
            [],
            model_artifact_from_row,
        )?;
        let jobs = collect(
            &transaction,
            "SELECT id, version_id, target_kind, command_json, status, log FROM conversion_jobs ORDER BY id",
            [],
            conversion_job_from_row,
        )?;
        let deployments = collect(
            &transaction,
            "SELECT id, project_id, artifact_id, previous_artifact_id, updated_seq FROM deployments ORDER BY id",
            [],
            deployment_from_row,
        )?;
        let active_deployment = transaction
            .query_row("SELECT id, project_id, artifact_id, previous_artifact_id, updated_seq FROM deployments ORDER BY updated_seq DESC, id DESC LIMIT 1", [], deployment_from_row)
            .optional()
            .map_err(ModelCatalogError::Sqlite)?;
        transaction.commit().map_err(ModelCatalogError::Sqlite)?;
        Ok(ModelCatalogSnapshot {
            projects,
            versions,
            artifacts,
            jobs,
            deployments,
            active_deployment,
        })
    }

    pub fn catalog(&self, force: bool) -> Result<ModelCatalogResponse, ModelCatalogError> {
        if !force {
            let cache = self
                .catalog_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((created, response)) = cache.as_ref()
                && created.elapsed() < Duration::from_secs(2)
            {
                let mut response = response.clone();
                response.cache_hits += 1;
                response.force = false;
                return Ok(response);
            }
        }

        let registry_rows = self.catalog_registry_rows()?;
        let projects = registry_rows
            .projects
            .iter()
            .map(|project| (project.id, project))
            .collect::<HashMap<_, _>>();
        let versions = registry_rows
            .versions
            .iter()
            .map(|version| (version.id, version))
            .collect::<HashMap<_, _>>();
        let mut registered = HashMap::new();
        for artifact in &registry_rows.artifacts {
            let Some(version) = versions.get(&artifact.version_id) else {
                continue;
            };
            let Some(project) = projects.get(&version.project_id) else {
                continue;
            };
            let path = self.resolve_artifact_path(&project.name, &version.version, &artifact.path);
            registered.insert(
                canonical_or_original(path),
                RegisteredArtifact {
                    project_id: project.id,
                    project_name: project.name.clone(),
                    version_id: version.id,
                    version_name: version.version.clone(),
                    artifact_id: artifact.id,
                    artifact_status: artifact.status.clone(),
                },
            );
        }

        let mut unique_files = HashMap::new();
        for catalog_root in &self.catalog_roots {
            let mut root_files = Vec::new();
            scan_model_files(catalog_root, catalog_root, &mut root_files)?;
            for (relative, absolute, size_bytes) in root_files {
                unique_files
                    .entry(relative.clone())
                    .or_insert((relative, absolute, size_bytes));
            }
        }
        let mut files = unique_files.into_values().collect::<Vec<_>>();
        files.sort_by(|left, right| left.0.cmp(&right.0));
        let mut root = ModelCatalogDirectory {
            node_type: "directory".to_owned(),
            name: "models".to_owned(),
            relative_path: String::new(),
            children: Vec::new(),
        };
        for (relative, absolute, size_bytes) in &files {
            let kind = relative
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("unknown")
                .to_ascii_lowercase();
            let registered_artifact = registered.get(&canonical_or_original(absolute.clone()));
            let (scan_status, scan_reason) = if force {
                inspect_model_file(absolute, *size_bytes)?
            } else if kind == "engine" {
                (
                    "need_confirm".to_owned(),
                    "TensorRT engine suffix accepted; contract validation deferred until load"
                        .to_owned(),
                )
            } else {
                (
                    "unsupported".to_owned(),
                    "current DeepStream mainline only loads TensorRT .engine models".to_owned(),
                )
            };
            let model = ModelCatalogModel {
                node_type: "model".to_owned(),
                name: relative
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                relative_path: relative.to_string_lossy().replace('\\', "/"),
                kind,
                size_bytes: *size_bytes,
                scan_status,
                scan_reason,
                project_id: registered_artifact.map(|entry| entry.project_id),
                project_name: registered_artifact.map(|entry| entry.project_name.clone()),
                version_id: registered_artifact.map(|entry| entry.version_id),
                version_name: registered_artifact.map(|entry| entry.version_name.clone()),
                artifact_id: registered_artifact.map(|entry| entry.artifact_id),
                artifact_status: registered_artifact.map(|entry| entry.artifact_status.clone()),
            };
            insert_model_node(&mut root, relative, model);
        }
        sort_catalog(&mut root);
        let directory_count = count_directories(&root);
        let response = ModelCatalogResponse {
            root,
            directory_count,
            model_count: files.len(),
            discovered_files: files.len(),
            updated_files: 0,
            cache_hits: 0,
            force,
        };
        let mut cached_response = response.clone();
        cached_response.force = false;
        *self
            .catalog_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some((Instant::now(), cached_response));
        Ok(response)
    }

    fn catalog_registry_rows(&self) -> Result<CatalogRegistryRows, ModelCatalogError> {
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(ModelCatalogError::Sqlite)?;
        let projects = collect(
            &transaction,
            "SELECT id, name, description FROM model_projects ORDER BY id",
            [],
            |row| {
                Ok(ModelProject {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                })
            },
        )?;
        let versions = collect(
            &transaction,
            "SELECT id, project_id, version, source_kind, source_path, classes_json, input_shape FROM model_versions ORDER BY id",
            [],
            model_version_from_row,
        )?;
        let artifacts = collect(
            &transaction,
            "SELECT id, version_id, kind, path, checksum, status FROM model_artifacts ORDER BY id",
            [],
            model_artifact_from_row,
        )?;
        transaction.commit().map_err(ModelCatalogError::Sqlite)?;
        Ok(CatalogRegistryRows {
            projects,
            versions,
            artifacts,
        })
    }

    pub fn publish(
        &self,
        project_id: i64,
        artifact_id: i64,
    ) -> Result<Deployment, ModelCatalogError> {
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(ModelCatalogError::Sqlite)?;
        transaction
            .query_row(
                "SELECT 1 FROM model_projects WHERE id = ?1",
                [project_id],
                |_| Ok(()),
            )
            .optional()
            .map_err(ModelCatalogError::Sqlite)?
            .ok_or(ModelCatalogError::ProjectNotFound(project_id))?;
        let artifact = transaction
            .query_row(
                "SELECT model_versions.project_id, model_artifacts.status FROM model_artifacts JOIN model_versions ON model_versions.id = model_artifacts.version_id WHERE model_artifacts.id = ?1",
                [artifact_id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(ModelCatalogError::Sqlite)?
            .ok_or(ModelCatalogError::ArtifactNotFound(artifact_id))?;
        if artifact.0 != project_id {
            return Err(ModelCatalogError::ArtifactProjectMismatch {
                artifact_id,
                project_id,
            });
        }
        if artifact.1 != "ready" {
            return Err(ModelCatalogError::ArtifactNotReady(artifact_id));
        }
        let existing = deployment_for_project(&transaction, project_id)?;
        let sequence = next_deployment_sequence(&transaction)?;
        let deployment = match existing {
            Some(current) if current.artifact_id == artifact_id => {
                transaction
                    .execute(
                        "UPDATE deployments SET updated_seq = ?1 WHERE project_id = ?2",
                        params![sequence, project_id],
                    )
                    .map_err(ModelCatalogError::Sqlite)?;
                Deployment {
                    updated_seq: sequence,
                    ..current
                }
            }
            Some(current) => {
                transaction
                    .execute(
                        "UPDATE deployments SET artifact_id = ?1, previous_artifact_id = ?2, updated_seq = ?3 WHERE project_id = ?4",
                        params![artifact_id, current.artifact_id, sequence, project_id],
                    )
                    .map_err(ModelCatalogError::Sqlite)?;
                Deployment {
                    artifact_id,
                    previous_artifact_id: Some(current.artifact_id),
                    updated_seq: sequence,
                    ..current
                }
            }
            None => {
                transaction
                    .execute(
                        "INSERT INTO deployments(project_id, artifact_id, previous_artifact_id, updated_seq) VALUES (?1, ?2, NULL, ?3)",
                        params![project_id, artifact_id, sequence],
                    )
                    .map_err(ModelCatalogError::Sqlite)?;
                Deployment {
                    id: transaction.last_insert_rowid(),
                    project_id,
                    artifact_id,
                    previous_artifact_id: None,
                    updated_seq: sequence,
                }
            }
        };
        transaction.commit().map_err(ModelCatalogError::Sqlite)?;
        Ok(deployment)
    }

    pub fn rollback(&self, project_id: i64) -> Result<Deployment, ModelCatalogError> {
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(ModelCatalogError::Sqlite)?;
        let current = deployment_for_project(&transaction, project_id)?
            .ok_or(ModelCatalogError::DeploymentNotFound(project_id))?;
        let Some(previous) = current.previous_artifact_id else {
            transaction.commit().map_err(ModelCatalogError::Sqlite)?;
            return Ok(current);
        };
        let sequence = next_deployment_sequence(&transaction)?;
        transaction
            .execute(
                "UPDATE deployments SET artifact_id = ?1, previous_artifact_id = ?2, updated_seq = ?3 WHERE project_id = ?4",
                params![previous, current.artifact_id, sequence, project_id],
            )
            .map_err(ModelCatalogError::Sqlite)?;
        let deployment = Deployment {
            artifact_id: previous,
            previous_artifact_id: Some(current.artifact_id),
            updated_seq: sequence,
            ..current
        };
        transaction.commit().map_err(ModelCatalogError::Sqlite)?;
        Ok(deployment)
    }

    fn connect(&self) -> Result<Connection, ModelCatalogError> {
        if let Some(parent) = self
            .path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|source| {
                ModelCatalogError::CreateDirectory {
                    path: parent.to_owned(),
                    source,
                }
            })?;
        }
        let connection = Connection::open(&self.path).map_err(ModelCatalogError::Sqlite)?;
        connection
            .busy_timeout(Duration::from_secs(30))
            .map_err(ModelCatalogError::Sqlite)?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(ModelCatalogError::Sqlite)?;
        Ok(connection)
    }

    fn resolve_artifact_path(
        &self,
        project_name: &str,
        version_name: &str,
        artifact_path: &str,
    ) -> PathBuf {
        let path = Path::new(artifact_path);
        if path.is_absolute() {
            path.to_owned()
        } else {
            self.model_root
                .join(project_name)
                .join(version_name)
                .join(path)
        }
    }
}

fn require_project(connection: &Connection, project_id: i64) -> Result<(), ModelCatalogError> {
    connection
        .query_row(
            "SELECT 1 FROM model_projects WHERE id = ?1",
            [project_id],
            |_| Ok(()),
        )
        .optional()
        .map_err(ModelCatalogError::Sqlite)?
        .ok_or(ModelCatalogError::ProjectNotFound(project_id))
}

fn require_version(connection: &Connection, version_id: i64) -> Result<(), ModelCatalogError> {
    connection
        .query_row(
            "SELECT 1 FROM model_versions WHERE id = ?1",
            [version_id],
            |_| Ok(()),
        )
        .optional()
        .map_err(ModelCatalogError::Sqlite)?
        .ok_or(ModelCatalogError::VersionNotFound(version_id))
}

fn model_version_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelVersion> {
    let classes_json: String = row.get(5)?;
    let classes = serde_json::from_str(&classes_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(ModelVersion {
        id: row.get(0)?,
        project_id: row.get(1)?,
        version: row.get(2)?,
        source_kind: row.get(3)?,
        source_path: row.get(4)?,
        classes,
        input_shape: row.get(6)?,
    })
}

fn model_artifact_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelArtifact> {
    Ok(ModelArtifact {
        id: row.get(0)?,
        version_id: row.get(1)?,
        kind: row.get(2)?,
        path: row.get(3)?,
        checksum: row.get(4)?,
        status: row.get(5)?,
        size_bytes: None,
    })
}

#[derive(Clone, Debug)]
struct RegisteredArtifact {
    project_id: i64,
    project_name: String,
    version_id: i64,
    version_name: String,
    artifact_id: i64,
    artifact_status: String,
}

struct CatalogRegistryRows {
    projects: Vec<ModelProject>,
    versions: Vec<ModelVersion>,
    artifacts: Vec<ModelArtifact>,
}

fn scan_model_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(PathBuf, PathBuf, u64)>,
) -> Result<(), ModelCatalogError> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && directory == root => {
            return Ok(());
        }
        Err(source) => {
            return Err(ModelCatalogError::ReadModelDirectory {
                path: directory.to_owned(),
                source,
            });
        }
    };
    for entry in entries {
        let entry = entry.map_err(|source| ModelCatalogError::ReadModelDirectory {
            path: directory.to_owned(),
            source,
        })?;
        let file_type =
            entry
                .file_type()
                .map_err(|source| ModelCatalogError::ReadModelDirectory {
                    path: entry.path(),
                    source,
                })?;
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            scan_model_files(root, &path, files)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        if !extension.eq_ignore_ascii_case("engine") && !extension.eq_ignore_ascii_case("onnx") {
            continue;
        }
        let metadata =
            entry
                .metadata()
                .map_err(|source| ModelCatalogError::ReadModelDirectory {
                    path: path.clone(),
                    source,
                })?;
        let relative =
            path.strip_prefix(root)
                .map_err(|_| ModelCatalogError::ModelPathOutsideRoot {
                    path: path.clone(),
                    root: root.to_owned(),
                })?;
        files.push((relative.to_owned(), path, metadata.len()));
    }
    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct CatalogManifest {
    #[serde(default = "default_manifest_schema_version")]
    schema_version: u32,
    model_id: String,
    display_name: String,
    artifact: CatalogManifestArtifact,
    #[serde(default)]
    runtime: CatalogManifestRuntime,
    input: CatalogManifestInput,
    output: CatalogManifestOutput,
    #[serde(default)]
    postprocess: CatalogManifestPostprocess,
    #[serde(default)]
    validated: bool,
    #[serde(default)]
    model_fingerprint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestArtifact {
    engine_path: String,
    sha256: String,
    size_bytes: u64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestRuntime {
    #[serde(default = "default_manifest_backend")]
    backend: String,
    #[serde(default = "default_manifest_precision")]
    precision: String,
    #[serde(default = "default_manifest_batch_size")]
    batch_size: u32,
}

impl Default for CatalogManifestRuntime {
    fn default() -> Self {
        Self {
            backend: default_manifest_backend(),
            precision: default_manifest_precision(),
            batch_size: default_manifest_batch_size(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestInput {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    #[serde(default = "default_manifest_layout")]
    layout: String,
    #[serde(default = "default_manifest_color_format")]
    color_format: String,
    #[serde(default = "default_manifest_scale_factor")]
    scale_factor: f64,
    #[serde(default)]
    maintain_aspect_ratio: bool,
    #[serde(default)]
    symmetric_padding: bool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestTensor {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    #[serde(default = "default_manifest_layout")]
    layout: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestOutput {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    #[serde(default = "default_manifest_layout")]
    layout: String,
    #[serde(default = "default_manifest_output_format")]
    format: String,
    #[serde(default)]
    class_count: u32,
    #[serde(default)]
    class_names: Vec<String>,
    #[serde(default)]
    has_objectness: bool,
    #[serde(default = "default_true")]
    scores_are_sigmoid: bool,
    #[serde(default = "default_manifest_coordinate_mode")]
    coordinate_mode: String,
    #[serde(default)]
    bindings: Vec<CatalogManifestTensor>,
    #[serde(default)]
    strides: Vec<u32>,
    #[serde(default)]
    anchors: Vec<Vec<f64>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogManifestPostprocess {
    #[serde(default = "default_manifest_parser")]
    parser: String,
    #[serde(default = "default_manifest_parser_preset")]
    parser_preset: String,
    #[serde(default = "default_manifest_confidence")]
    confidence_threshold: f64,
    #[serde(default = "default_manifest_nms")]
    nms_iou_threshold: f64,
    #[serde(default = "default_true")]
    class_aware_nms: bool,
    #[serde(default = "default_manifest_max_detections")]
    max_detections: u32,
}

impl Default for CatalogManifestPostprocess {
    fn default() -> Self {
        Self {
            parser: default_manifest_parser(),
            parser_preset: default_manifest_parser_preset(),
            confidence_threshold: default_manifest_confidence(),
            nms_iou_threshold: default_manifest_nms(),
            class_aware_nms: true,
            max_detections: default_manifest_max_detections(),
        }
    }
}

const fn default_manifest_schema_version() -> u32 {
    1
}
fn default_manifest_backend() -> String {
    "custom_tensorrt".to_owned()
}
fn default_manifest_precision() -> String {
    "fp16".to_owned()
}
const fn default_manifest_batch_size() -> u32 {
    1
}
fn default_manifest_layout() -> String {
    "NCHW".to_owned()
}
fn default_manifest_color_format() -> String {
    "RGB".to_owned()
}
const fn default_manifest_scale_factor() -> f64 {
    1.0 / 255.0
}
fn default_manifest_output_format() -> String {
    "yolo_cxcywh_class_scores".to_owned()
}
const fn default_true() -> bool {
    true
}
fn default_manifest_coordinate_mode() -> String {
    "pixel".to_owned()
}
fn default_manifest_parser() -> String {
    "yolo".to_owned()
}
fn default_manifest_parser_preset() -> String {
    "auto".to_owned()
}
const fn default_manifest_confidence() -> f64 {
    0.25
}
const fn default_manifest_nms() -> f64 {
    0.45
}
const fn default_manifest_max_detections() -> u32 {
    300
}

fn inspect_model_file(
    artifact_path: &Path,
    size_bytes: u64,
) -> Result<(String, String), ModelCatalogError> {
    let manifest_path = artifact_path.with_file_name(format!(
        "{}.manifest.json",
        artifact_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    ));
    let manifest_path = if manifest_path.is_file() {
        manifest_path
    } else {
        let legacy = artifact_path.with_file_name("model.manifest.json");
        if legacy.is_file() {
            legacy
        } else {
            return Ok((
                "need_confirm".to_owned(),
                "model manifest is missing".to_owned(),
            ));
        }
    };
    let manifest_file =
        std::fs::File::open(&manifest_path).map_err(|source| ModelCatalogError::ReadModelFile {
            path: manifest_path.clone(),
            source,
        })?;
    let manifest: serde_json::Value = match serde_json::from_reader(manifest_file) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Ok((
                "invalid".to_owned(),
                format!("model manifest is invalid: {error}"),
            ));
        }
    };
    if let Some(profile) = manifest
        .get("model_profile")
        .and_then(|value| value.as_object())
    {
        let status = profile
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if !matches!(status, "VALIDATED" | "ACTIVE") {
            return Ok((
                "need_confirm".to_owned(),
                "model manifest is awaiting configuration or diagnostics".to_owned(),
            ));
        }
    }
    let manifest_has_class_names = manifest
        .get("output")
        .and_then(|value| value.as_object())
        .is_some_and(|output| output.contains_key("class_names"));
    let manifest: CatalogManifest = match serde_json::from_value(manifest) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Ok((
                "invalid".to_owned(),
                format!("model manifest is invalid: {error}"),
            ));
        }
    };
    let artifact_name = artifact_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    if manifest.artifact.engine_path != artifact_name {
        return Ok((
            "invalid".to_owned(),
            "model manifest artifact does not match model file name".to_owned(),
        ));
    }
    if manifest.artifact.size_bytes != size_bytes {
        return Ok((
            "invalid".to_owned(),
            "model manifest artifact size does not match file".to_owned(),
        ));
    }
    if manifest.artifact.sha256.is_empty() {
        return Ok((
            "invalid".to_owned(),
            "model manifest is invalid: artifact.sha256 is missing".to_owned(),
        ));
    }
    let actual_sha = sha256_file(artifact_path)?;
    if manifest.artifact.sha256 != actual_sha {
        return Ok((
            "invalid".to_owned(),
            "model manifest artifact sha256 does not match file".to_owned(),
        ));
    }
    if !manifest.model_fingerprint.is_empty() {
        let expected_fingerprint = &manifest.model_fingerprint;
        let fingerprint = manifest_fingerprint(&manifest, true);
        let legacy_fingerprint =
            (!manifest_has_class_names).then(|| manifest_fingerprint(&manifest, false));
        let fingerprint_matches = fingerprint
            .as_ref()
            .is_ok_and(|value| value == expected_fingerprint);
        let legacy_matches = legacy_fingerprint.as_ref().is_some_and(|fingerprint| {
            fingerprint
                .as_ref()
                .is_ok_and(|value| value == expected_fingerprint)
        });
        if !fingerprint_matches && !legacy_matches {
            return Ok((
                "invalid".to_owned(),
                "model manifest fingerprint does not match manifest content".to_owned(),
            ));
        }
    }
    Ok((
        "ready".to_owned(),
        "model manifest matches artifact".to_owned(),
    ))
}

fn manifest_fingerprint(
    manifest: &CatalogManifest,
    include_class_names: bool,
) -> Result<String, serde_json::Error> {
    let mut output = serde_json::Map::new();
    output.insert("name".to_owned(), serde_json::json!(manifest.output.name));
    output.insert("shape".to_owned(), serde_json::json!(manifest.output.shape));
    output.insert("dtype".to_owned(), serde_json::json!(manifest.output.dtype));
    output.insert(
        "layout".to_owned(),
        serde_json::json!(manifest.output.layout),
    );
    output.insert(
        "format".to_owned(),
        serde_json::json!(manifest.output.format),
    );
    output.insert(
        "class_count".to_owned(),
        serde_json::json!(manifest.output.class_count),
    );
    output.insert(
        "has_objectness".to_owned(),
        serde_json::json!(manifest.output.has_objectness),
    );
    output.insert(
        "coordinate_mode".to_owned(),
        serde_json::json!(manifest.output.coordinate_mode),
    );
    if !manifest.output.bindings.is_empty() {
        output.insert(
            "bindings".to_owned(),
            serde_json::to_value(&manifest.output.bindings)?,
        );
    }
    if !manifest.output.strides.is_empty() {
        output.insert(
            "strides".to_owned(),
            serde_json::json!(manifest.output.strides),
        );
    }
    if !manifest.output.anchors.is_empty() {
        output.insert(
            "anchors".to_owned(),
            serde_json::json!(manifest.output.anchors),
        );
    }
    if include_class_names {
        output.insert(
            "class_names".to_owned(),
            serde_json::json!(manifest.output.class_names),
        );
    }
    let payload = serde_json::json!({
        "artifact_sha256": manifest.artifact.sha256,
        "input": {
            "name": manifest.input.name,
            "shape": manifest.input.shape,
            "dtype": manifest.input.dtype,
            "layout": manifest.input.layout,
        },
        "output": output,
        "parser_schema": "yolo-v1",
    });
    let stable = serde_json::to_string(&payload)?;
    let mut digest = Sha256::new();
    digest.update(python_json_numbers(&stable).as_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn python_json_numbers(json: &str) -> String {
    let bytes = json.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            output.push(byte);
            index += 1;
            continue;
        }
        if !byte.is_ascii_digit() && byte != b'-' {
            output.push(byte);
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < bytes.len()
            && (bytes[index].is_ascii_digit()
                || matches!(bytes[index], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            index += 1;
        }
        output.extend_from_slice(python_number_token(&bytes[start..index]).as_bytes());
    }
    String::from_utf8(output).expect("valid JSON remains UTF-8")
}

fn python_number_token(token: &[u8]) -> String {
    let token = std::str::from_utf8(token).expect("JSON numbers are ASCII");
    if let Some(exponent_at) = token.find(['e', 'E']) {
        let (mantissa, exponent) = token.split_at(exponent_at);
        let exponent = &exponent[1..];
        let (sign, digits) = exponent
            .strip_prefix(['+', '-'])
            .map_or(("", exponent), |digits| (&exponent[..1], digits));
        return format!(
            "{mantissa}e{sign}{}{digits}",
            if digits.len() == 1 { "0" } else { "" }
        );
    }
    let (sign, unsigned) = token
        .strip_prefix('-')
        .map_or(("", token), |value| ("-", value));
    let Some(fraction) = unsigned.strip_prefix("0.") else {
        return token.to_owned();
    };
    let leading_zeros = fraction.bytes().take_while(|byte| *byte == b'0').count();
    if leading_zeros < 4 || leading_zeros == fraction.len() {
        return token.to_owned();
    }
    let significant = &fraction[leading_zeros..];
    let (first, rest) = significant.split_at(1);
    let mantissa = if rest.is_empty() {
        first.to_owned()
    } else {
        format!("{first}.{rest}")
    };
    let exponent = leading_zeros + 1;
    format!(
        "{sign}{mantissa}e-{}{exponent}",
        if exponent < 10 { "0" } else { "" }
    )
}

fn sha256_file(path: &Path) -> Result<String, ModelCatalogError> {
    let mut file =
        std::fs::File::open(path).map_err(|source| ModelCatalogError::ReadModelFile {
            path: path.to_owned(),
            source,
        })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ModelCatalogError::ReadModelFile {
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn insert_model_node(
    directory: &mut ModelCatalogDirectory,
    relative_path: &Path,
    model: ModelCatalogModel,
) {
    let parts = relative_path
        .parent()
        .into_iter()
        .flat_map(Path::components)
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let mut current = directory;
    let mut accumulated = PathBuf::new();
    for part in parts {
        accumulated.push(&part);
        let index = current.children.iter().position(
            |node| matches!(node, ModelCatalogNode::Directory(child) if child.name == part),
        );
        let index = match index {
            Some(index) => index,
            None => {
                current
                    .children
                    .push(ModelCatalogNode::Directory(ModelCatalogDirectory {
                        node_type: "directory".to_owned(),
                        name: part,
                        relative_path: accumulated.to_string_lossy().replace('\\', "/"),
                        children: Vec::new(),
                    }));
                current.children.len() - 1
            }
        };
        let ModelCatalogNode::Directory(child) = &mut current.children[index] else {
            unreachable!("directory lookup only returns directory nodes");
        };
        current = child;
    }
    current.children.push(ModelCatalogNode::Model(model));
}

fn sort_catalog(directory: &mut ModelCatalogDirectory) {
    directory.children.sort_by(|left, right| {
        let left_key = match left {
            ModelCatalogNode::Directory(child) => (0, child.name.to_ascii_lowercase()),
            ModelCatalogNode::Model(child) => (1, child.name.to_ascii_lowercase()),
        };
        let right_key = match right {
            ModelCatalogNode::Directory(child) => (0, child.name.to_ascii_lowercase()),
            ModelCatalogNode::Model(child) => (1, child.name.to_ascii_lowercase()),
        };
        left_key.cmp(&right_key)
    });
    for child in &mut directory.children {
        if let ModelCatalogNode::Directory(child) = child {
            sort_catalog(child);
        }
    }
}

fn count_directories(directory: &ModelCatalogDirectory) -> usize {
    directory
        .children
        .iter()
        .map(|node| match node {
            ModelCatalogNode::Directory(child) => 1 + count_directories(child),
            ModelCatalogNode::Model(_) => 0,
        })
        .sum()
}

fn canonical_or_original(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

fn conversion_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConversionJob> {
    let command_json: String = row.get(3)?;
    let command = serde_json::from_str(&command_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(ConversionJob {
        id: row.get(0)?,
        version_id: row.get(1)?,
        target_kind: row.get(2)?,
        command,
        status: row.get(4)?,
        log: row.get(5)?,
    })
}

fn deployment_for_project(
    transaction: &Transaction<'_>,
    project_id: i64,
) -> Result<Option<Deployment>, ModelCatalogError> {
    transaction
        .query_row(
            "SELECT id, project_id, artifact_id, previous_artifact_id, updated_seq FROM deployments WHERE project_id = ?1",
            [project_id],
            deployment_from_row,
        )
        .optional()
        .map_err(ModelCatalogError::Sqlite)
}

fn next_deployment_sequence(transaction: &Transaction<'_>) -> Result<i64, ModelCatalogError> {
    transaction
        .execute(
            "UPDATE registry_sequence SET value = value + 1 WHERE name = 'deployment'",
            [],
        )
        .map_err(ModelCatalogError::Sqlite)?;
    transaction
        .query_row(
            "SELECT value FROM registry_sequence WHERE name = 'deployment'",
            [],
            |row| row.get(0),
        )
        .map_err(ModelCatalogError::Sqlite)
}

fn deployment_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Deployment> {
    Ok(Deployment {
        id: row.get(0)?,
        project_id: row.get(1)?,
        artifact_id: row.get(2)?,
        previous_artifact_id: row.get(3)?,
        updated_seq: row.get(4)?,
    })
}

fn collect<T, P, F>(
    connection: &Connection,
    sql: &str,
    params: P,
    map: F,
) -> Result<Vec<T>, ModelCatalogError>
where
    P: rusqlite::Params,
    F: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    let mut statement = connection.prepare(sql).map_err(ModelCatalogError::Sqlite)?;
    statement
        .query_map(params, map)
        .map_err(ModelCatalogError::Sqlite)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(ModelCatalogError::Sqlite)
}

fn migrate(connection: &mut Connection) -> Result<(), ModelCatalogError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(ModelCatalogError::Sqlite)?;
    transaction.execute_batch(r#"
        CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS model_projects (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, description TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL REFERENCES model_projects(id), version TEXT NOT NULL, source_kind TEXT NOT NULL CHECK (source_kind IN ('pt','onnx')), source_path TEXT NOT NULL, classes_json TEXT NOT NULL, input_shape TEXT NOT NULL, UNIQUE(project_id, version));
        CREATE TABLE IF NOT EXISTS model_artifacts (id INTEGER PRIMARY KEY AUTOINCREMENT, version_id INTEGER NOT NULL REFERENCES model_versions(id), kind TEXT NOT NULL CHECK (kind IN ('pt','onnx','engine')), path TEXT NOT NULL, checksum TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending','running','ready','failed')));
        CREATE TABLE IF NOT EXISTS conversion_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, version_id INTEGER NOT NULL REFERENCES model_versions(id), target_kind TEXT NOT NULL CHECK (target_kind IN ('onnx','engine')), command_json TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending','running','failed','succeeded')), log TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS deployments (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id), artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id), previous_artifact_id INTEGER REFERENCES model_artifacts(id), updated_seq INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS registry_sequence (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
    "#).map_err(ModelCatalogError::Sqlite)
    ?;
    let has_updated_sequence = {
        let mut statement = transaction
            .prepare("PRAGMA table_info(deployments)")
            .map_err(ModelCatalogError::Sqlite)?;
        statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(ModelCatalogError::Sqlite)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(ModelCatalogError::Sqlite)?
            .iter()
            .any(|column| column == "updated_seq")
    };
    if !has_updated_sequence {
        transaction
            .execute_batch(
                "ALTER TABLE deployments ADD COLUMN updated_seq INTEGER NOT NULL DEFAULT 0; UPDATE deployments SET updated_seq = id WHERE updated_seq = 0;",
            )
            .map_err(ModelCatalogError::Sqlite)?;
    }
    transaction
        .execute_batch(
            r#"
            INSERT OR IGNORE INTO registry_sequence(name, value) SELECT 'deployment', COALESCE(MAX(updated_seq), 0) FROM deployments;
            UPDATE registry_sequence SET value = MAX(value, (SELECT COALESCE(MAX(updated_seq), 0) FROM deployments)) WHERE name = 'deployment';
            INSERT INTO schema_migrations(name, version) VALUES ('model_catalog', 1)
            ON CONFLICT(name) DO UPDATE SET version = MAX(version, excluded.version);
            "#,
        )
        .map_err(ModelCatalogError::Sqlite)?;
    transaction.commit().map_err(ModelCatalogError::Sqlite)
}

#[derive(Debug, Error)]
pub enum ModelCatalogError {
    #[error("failed to create model catalog directory {}: {source}", path.display())]
    CreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read model directory {}: {source}", path.display())]
    ReadModelDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read model file {}: {source}", path.display())]
    ReadModelFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "model path {} escaped configured root {}",
        path.display(),
        root.display()
    )]
    ModelPathOutsideRoot { path: PathBuf, root: PathBuf },
    #[error("model catalog SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("unknown model project id: {0}")]
    ProjectNotFound(i64),
    #[error("unknown model version id: {0}")]
    VersionNotFound(i64),
    #[error("unknown model artifact id: {0}")]
    ArtifactNotFound(i64),
    #[error("model artifact {artifact_id} does not belong to project {project_id}")]
    ArtifactProjectMismatch { artifact_id: i64, project_id: i64 },
    #[error("model artifact {0} is not ready for deployment")]
    ArtifactNotReady(i64),
    #[error("no model deployment exists for project {0}")]
    DeploymentNotFound(i64),
}
