use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use novasight_runtime::{ModelActivationRequest, ModelActivationResult};
use novasight_store::model_catalog::{
    ConversionJob, Deployment, ModelArtifact, ModelCatalogError, ModelCatalogResponse,
    ModelProject, ModelVersion, SqliteModelCatalog,
};
use serde::{Deserialize, Serialize};

use super::{ControlApiError, ControlState};

pub(super) fn routes() -> Router<ControlState> {
    Router::new()
        .route("/api/models/projects", get(model_projects))
        .route("/api/models/catalog", get(get_model_catalog))
        .route(
            "/api/models/projects/{project_id}/versions",
            get(model_versions),
        )
        .route(
            "/api/models/versions/{version_id}/artifacts",
            get(model_artifacts),
        )
        .route("/api/models/jobs", get(model_jobs))
        .route("/api/models/jobs/list", post(model_jobs_for_version))
        .route(
            "/api/models/projects/{project_id}/publish",
            post(publish_model),
        )
        .route(
            "/api/models/projects/{project_id}/rollback",
            post(rollback_model),
        )
}

async fn run<T>(
    state: &ControlState,
    operation: impl FnOnce(SqliteModelCatalog) -> Result<T, ModelCatalogError> + Send + 'static,
) -> Result<T, ControlApiError>
where
    T: Send + 'static,
{
    let catalog = state
        .model_catalog
        .as_ref()
        .ok_or(ControlApiError::ModelCatalogUnavailable)?
        .clone();
    tokio::task::spawn_blocking(move || operation(catalog))
        .await
        .map_err(ControlApiError::ModelCatalogTask)?
        .map_err(ControlApiError::ModelCatalog)
}

async fn model_projects(
    State(state): State<ControlState>,
) -> Result<Json<Vec<ModelProject>>, ControlApiError> {
    Ok(Json(run(&state, |catalog| catalog.list_projects()).await?))
}

#[derive(Default, Deserialize)]
struct ModelCatalogQuery {
    force: bool,
}

async fn get_model_catalog(
    State(state): State<ControlState>,
    Query(query): Query<ModelCatalogQuery>,
) -> Result<Json<ModelCatalogResponse>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| catalog.catalog(query.force)).await?,
    ))
}

async fn model_versions(
    Path(project_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<Vec<ModelVersion>>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| catalog.list_versions(project_id)).await?,
    ))
}

async fn model_artifacts(
    Path(version_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<Vec<ModelArtifact>>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| catalog.list_artifacts(version_id)).await?,
    ))
}

#[derive(Default, Deserialize)]
struct ModelJobsRequest {
    version_id: Option<i64>,
}

async fn model_jobs(
    State(state): State<ControlState>,
    Query(query): Query<ModelJobsRequest>,
) -> Result<Json<Vec<ConversionJob>>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| catalog.list_jobs(query.version_id)).await?,
    ))
}

async fn model_jobs_for_version(
    State(state): State<ControlState>,
    Json(request): Json<ModelJobsRequest>,
) -> Result<Json<Vec<ConversionJob>>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| catalog.list_jobs(request.version_id)).await?,
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishModelRequest {
    artifact_id: i64,
    #[serde(default = "default_parser_preset")]
    parser_preset: String,
}

fn default_parser_preset() -> String {
    "auto".to_owned()
}

#[derive(Serialize)]
struct ModelSwitchResponse {
    deployment: Deployment,
    inference: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    parser_contract: Option<serde_json::Value>,
    preparation: ModelPreparation,
    report: ModelSwitchReport,
}

#[derive(Serialize)]
struct ModelPreparation {
    manifest_action: &'static str,
    reason: &'static str,
    input_shape: String,
    classes: Vec<String>,
}

#[derive(Serialize)]
struct ModelSwitchReport {
    action: &'static str,
    applied: bool,
    rolled_back: bool,
    message: String,
    runtime_error: String,
    artifact_id: i64,
    previous_artifact_id: Option<i64>,
    artifact_path: String,
    backend: &'static str,
    input_shape: String,
    classes: usize,
    sections: Vec<ModelSwitchSection>,
}

#[derive(Serialize)]
struct ModelSwitchSection {
    section: &'static str,
    impact: &'static str,
    status: &'static str,
    message: &'static str,
}

async fn publish_model(
    Path(project_id): Path<i64>,
    State(state): State<ControlState>,
    Json(request): Json<PublishModelRequest>,
) -> Result<Json<ModelSwitchResponse>, ControlApiError> {
    super::ensure_config_effective(&state).await?;
    state
        .runtime
        .activate_model(ModelActivationRequest::Publish {
            project_id,
            artifact_id: request.artifact_id,
            parser_preset: request.parser_preset,
        })
        .await
        .map(success_response)
        .map(Json)
        .map_err(ControlApiError::ModelActivation)
}

async fn rollback_model(
    Path(project_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<ModelSwitchResponse>, ControlApiError> {
    super::ensure_config_effective(&state).await?;
    state
        .runtime
        .activate_model(ModelActivationRequest::Rollback { project_id })
        .await
        .map(success_response)
        .map(Json)
        .map_err(ControlApiError::ModelActivation)
}

fn success_response(result: ModelActivationResult) -> ModelSwitchResponse {
    let loaded = result.restarted;
    let changed = result.changed;
    let input_shape = result
        .contract
        .as_ref()
        .map(|contract| contract.input_shape.clone())
        .unwrap_or_else(|| result.candidate.version.input_shape.clone());
    let classes = result
        .contract
        .as_ref()
        .map(|contract| contract.classes.clone())
        .unwrap_or_else(|| result.candidate.version.classes.clone());
    let parser_contract = result
        .contract
        .as_ref()
        .and_then(|contract| serde_json::to_value(&contract.parser).ok());
    let artifact_path = result.candidate.artifact_path.display().to_string();
    let message = if !result.changed {
        "当前部署没有可回滚的上一版本；保持现有模型".to_owned()
    } else if loaded {
        format!(
            "模型已切换并通过首个 DetectionBatch 就绪门：{}",
            result.candidate.artifact_path.display()
        )
    } else {
        format!(
            "模型已切换并通过 Rust DeepStream 启动前校验：{}",
            result.candidate.artifact_path.display()
        )
    };
    ModelSwitchResponse {
        deployment: result.deployment.clone(),
        inference: serde_json::json!({
            "selected": "deepstream_nvinfer",
            "available": true,
            "loaded": loaded,
            "configured": true,
            "reason": if !changed {
                "deployment unchanged; runtime was not restarted"
            } else if loaded {
                "active deployment produced a valid DetectionBatch"
            } else {
                "active deployment passed Rust DeepStream preflight"
            },
        }),
        parser_contract,
        preparation: ModelPreparation {
            manifest_action: if changed { "reused" } else { "unchanged" },
            reason: if changed {
                "validated existing model manifest and immutable engine identity"
            } else {
                "no previous deployment exists; candidate validation was not required"
            },
            input_shape: input_shape.clone(),
            classes: classes.clone(),
        },
        report: ModelSwitchReport {
            action: result.action,
            applied: changed,
            rolled_back: false,
            message,
            runtime_error: String::new(),
            artifact_id: result.deployment.artifact_id,
            previous_artifact_id: result.deployment.previous_artifact_id,
            artifact_path,
            backend: "deepstream_nvinfer",
            input_shape,
            classes: classes.len(),
            sections: vec![
                ModelSwitchSection {
                    section: "模型部署",
                    impact: "推理入口",
                    status: if changed { "applied" } else { "unchanged" },
                    message: if changed {
                        "SQLite active deployment 已提交"
                    } else {
                        "SQLite active deployment 保持不变"
                    },
                },
                ModelSwitchSection {
                    section: "推理运行态",
                    impact: "采集 -> 推理 -> 控制",
                    status: if changed { "applied" } else { "unchanged" },
                    message: if !changed {
                        "当前运行态保持不变"
                    } else if loaded {
                        "新 epoch 已通过首个 DetectionBatch 就绪门"
                    } else {
                        "候选模型已通过 Rust DeepStream 启动前校验"
                    },
                },
            ],
        },
    }
}
