use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use novasight_runtime::{
    ModelActivationRequest, ModelActivationResult, ModelIngressRequest, ModelIngressResult,
    ModelProbeInputMode, ModelProfileConfigureRequest,
};
use novasight_store::model_catalog::{
    CatalogEngineRegistration, ConversionJob, Deployment, ModelArtifact, ModelArtifactMetadata,
    ModelCatalogError, ModelCatalogResponse, ModelProject, ModelRecommendation, ModelVersion,
    SqliteModelCatalog,
};
use serde::{Deserialize, Serialize};

use super::{ControlApiError, ControlState};

pub(super) fn routes() -> Router<ControlState> {
    Router::new()
        .route("/api/models/projects", get(model_projects))
        .route("/api/models/catalog", get(get_model_catalog))
        .route("/api/models/catalog/register", post(register_catalog_model))
        .route(
            "/api/models/artifacts/{artifact_id}/metadata",
            get(get_artifact_metadata).put(update_artifact_metadata),
        )
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
            "/api/models/artifacts/{artifact_id}/inspect",
            post(inspect_model),
        )
        .route(
            "/api/models/artifacts/{artifact_id}/profile",
            get(model_profile).put(configure_model_profile),
        )
        .route(
            "/api/models/artifacts/{artifact_id}/deepstream/recommendation",
            get(deepstream_recommendation),
        )
        .route(
            "/api/models/artifacts/{artifact_id}/probe",
            post(probe_model),
        )
        .route(
            "/api/models/projects/{project_id}/publish",
            post(publish_model),
        )
        .route(
            "/api/models/projects/{project_id}/rollback",
            post(rollback_model),
        )
}

async fn inspect_model(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<ModelIngressResult>, ControlApiError> {
    state
        .runtime
        .model_ingress(ModelIngressRequest::Inspect { artifact_id })
        .await
        .map(Json)
        .map_err(ControlApiError::ModelIngress)
}

async fn model_profile(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<ModelIngressResult>, ControlApiError> {
    state
        .runtime
        .model_ingress(ModelIngressRequest::GetProfile { artifact_id })
        .await
        .map(Json)
        .map_err(ControlApiError::ModelIngress)
}

async fn configure_model_profile(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
    Json(profile): Json<ModelProfileConfigureRequest>,
) -> Result<Json<ModelIngressResult>, ControlApiError> {
    state
        .runtime
        .model_ingress(ModelIngressRequest::Configure {
            artifact_id,
            profile: Box::new(profile),
        })
        .await
        .map(Json)
        .map_err(ControlApiError::ModelIngress)
}

#[derive(Serialize)]
struct DeepStreamRecommendationResponse {
    artifact_id: i64,
    artifact_path: String,
    recommendation: DeepStreamRecommendation,
    io_tensors: Vec<RecommendedTensor>,
    class_names: Vec<String>,
    output_has_objectness: bool,
    sources: std::collections::BTreeMap<&'static str, &'static str>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct DeepStreamRecommendation {
    model_id: String,
    display_name: String,
    runtime_precision: String,
    input_name: String,
    input_shape: Vec<u64>,
    input_dtype: String,
    input_color_format: String,
    input_scale_factor: f64,
    maintain_aspect_ratio: bool,
    symmetric_padding: bool,
    output_name: String,
    output_shape: Vec<u64>,
    output_dtype: String,
    class_count: u64,
    confidence_threshold: f64,
    nms_iou_threshold: f64,
}

#[derive(Serialize)]
struct RecommendedTensor {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    mode: &'static str,
}

async fn deepstream_recommendation(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<DeepStreamRecommendationResponse>, ControlApiError> {
    let artifact = run(&state, move |catalog| {
        catalog.runtime_artifact_by_id(artifact_id)
    })
    .await?;
    if artifact.artifact.kind != "engine" {
        return Err(ControlApiError::ModelRecommendationInvalid(
            "DeepStream recommendation requires a TensorRT .engine artifact".to_owned(),
        ));
    }
    let profile = state
        .runtime
        .model_ingress(ModelIngressRequest::GetProfile { artifact_id })
        .await
        .map_err(ControlApiError::ModelIngress)?
        .profile;
    recommendation_from_profile(artifact, profile)
        .map(Json)
        .map_err(ControlApiError::ModelRecommendationInvalid)
}

fn recommendation_from_profile(
    artifact: novasight_store::model_catalog::RuntimeModelArtifact,
    profile: serde_json::Value,
) -> Result<DeepStreamRecommendationResponse, String> {
    let object = profile
        .as_object()
        .ok_or_else(|| "model profile must be an object".to_owned())?;
    let input = object_value(object, "input")?;
    let preprocess = object_value(object, "preprocess")?;
    let decoder = object_value(object, "decoder")?;
    let postprocess = object_value(object, "postprocess")?;
    let output = object
        .get("outputs")
        .and_then(serde_json::Value::as_array)
        .and_then(|outputs| outputs.first())
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "model profile has no output tensor".to_owned())?;
    let labels = string_list(object.get("labels"), "labels")?;
    let class_count = integer(decoder, "class_count")?;
    if class_count == 0 || labels.len() != class_count as usize {
        return Err(
            "configure the model class contract before requesting a recommendation".to_owned(),
        );
    }
    let input_name = string(input, "name")?;
    let input_shape = shape(input, "runtime_shape")?;
    let input_dtype = string(input, "dtype")?;
    let output_name = string(output, "name")?;
    let output_shape = shape(output, "shape")?;
    let output_dtype = string(output, "dtype")?;
    let output_has_objectness = boolean(decoder, "has_objectness")?;
    let warnings = object
        .get("inspection")
        .and_then(serde_json::Value::as_object)
        .and_then(|inspection| inspection.get("warnings"))
        .map(|value| string_list(Some(value), "inspection.warnings"))
        .transpose()?
        .unwrap_or_default();
    let io_tensors = vec![
        RecommendedTensor {
            name: input_name.clone(),
            shape: input_shape.clone(),
            dtype: input_dtype.clone(),
            mode: "input",
        },
        RecommendedTensor {
            name: output_name.clone(),
            shape: output_shape.clone(),
            dtype: output_dtype.clone(),
            mode: "output",
        },
    ];
    Ok(DeepStreamRecommendationResponse {
        artifact_id: artifact.artifact.id,
        artifact_path: artifact.artifact.path,
        recommendation: DeepStreamRecommendation {
            model_id: string(object, "model_id")?,
            display_name: string(object, "display_name")?,
            runtime_precision: runtime_precision(&input_dtype).to_owned(),
            input_name,
            input_shape,
            input_dtype,
            input_color_format: string(preprocess, "color_format")?,
            input_scale_factor: number(preprocess, "scale")?,
            maintain_aspect_ratio: string(preprocess, "resize_mode")? == "letterbox",
            symmetric_padding: boolean(preprocess, "symmetric_padding")?,
            output_name,
            output_shape,
            output_dtype,
            class_count,
            confidence_threshold: number(postprocess, "confidence_threshold")?,
            nms_iou_threshold: number(postprocess, "nms_threshold")?,
        },
        io_tensors,
        class_names: labels,
        output_has_objectness,
        sources: std::collections::BTreeMap::from([
            ("input_contract", "model_profile_receipt"),
            ("output_contract", "model_profile_receipt"),
            ("class_contract", "configured_model_profile"),
            ("preprocess_contract", "configured_model_profile"),
        ]),
        warnings,
    })
}

fn object_value<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("model profile field {field} must be an object"))
}

fn string(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<String, String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("model profile field {field} must be a non-empty string"))
}

fn shape(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Vec<u64>, String> {
    let values = object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("model profile field {field} must be an array"))?;
    let shape = values
        .iter()
        .map(serde_json::Value::as_u64)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("model profile field {field} must contain positive integers"))?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!(
            "model profile field {field} must contain positive integers"
        ));
    }
    Ok(shape)
}

fn integer(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<u64, String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("model profile field {field} must be an unsigned integer"))
}

fn number(object: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<f64, String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("model profile field {field} must be a finite number"))
}

fn boolean(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<bool, String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| format!("model profile field {field} must be boolean"))
}

fn string_list(value: Option<&serde_json::Value>, field: &str) -> Result<Vec<String>, String> {
    value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("model profile field {field} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("model profile field {field} must contain strings"))
        })
        .collect()
}

fn runtime_precision(dtype: &str) -> &str {
    match dtype.to_ascii_lowercase().as_str() {
        "float16" | "fp16" | "half" => "fp16",
        "int8" => "int8",
        _ => "fp32",
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeModelRequest {
    #[serde(default = "default_probe_input_mode")]
    input_mode: ModelProbeInputMode,
}

const fn default_probe_input_mode() -> ModelProbeInputMode {
    ModelProbeInputMode::Fixed
}

async fn probe_model(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
    Json(request): Json<ProbeModelRequest>,
) -> Result<Json<ModelIngressResult>, ControlApiError> {
    super::ensure_config_effective(&state).await?;
    state
        .runtime
        .model_ingress(ModelIngressRequest::Probe {
            artifact_id,
            input_mode: request.input_mode,
        })
        .await
        .map(Json)
        .map_err(ControlApiError::ModelIngress)
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
    #[serde(default)]
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogRegisterRequest {
    relative_path: String,
}

async fn register_catalog_model(
    State(state): State<ControlState>,
    Json(request): Json<CatalogRegisterRequest>,
) -> Result<Json<CatalogEngineRegistration>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| {
            catalog.register_catalog_engine(&request.relative_path)
        })
        .await?,
    ))
}

async fn get_artifact_metadata(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<ModelArtifactMetadata>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| {
            catalog.artifact_metadata(artifact_id)
        })
        .await?,
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateArtifactMetadataRequest {
    recommendation: ModelRecommendation,
    #[serde(default)]
    tags: Vec<String>,
}

async fn update_artifact_metadata(
    Path(artifact_id): Path<i64>,
    State(state): State<ControlState>,
    Json(request): Json<UpdateArtifactMetadataRequest>,
) -> Result<Json<ModelArtifactMetadata>, ControlApiError> {
    Ok(Json(
        run(&state, move |catalog| {
            catalog.update_artifact_metadata(artifact_id, request.recommendation, request.tags)
        })
        .await?,
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
    backend: String,
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
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let backend = configured_inference_backend(&state).await;
    let result = state
        .runtime
        .activate_model(ModelActivationRequest::Publish {
            project_id,
            artifact_id: request.artifact_id,
            parser_preset: request.parser_preset,
        })
        .await
        .map_err(ControlApiError::ModelActivation)?;
    Ok(Json(success_response(result, backend)))
}

async fn rollback_model(
    Path(project_id): Path<i64>,
    State(state): State<ControlState>,
) -> Result<Json<ModelSwitchResponse>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let backend = configured_inference_backend(&state).await;
    let result = state
        .runtime
        .activate_model(ModelActivationRequest::Rollback { project_id })
        .await
        .map_err(ControlApiError::ModelActivation)?;
    Ok(Json(success_response(result, backend)))
}

async fn configured_inference_backend(state: &ControlState) -> String {
    let Some(config) = &state.config else {
        return "unconfigured".to_owned();
    };
    config
        .snapshot()
        .await
        .inference
        .as_ref()
        .map(|inference| {
            serde_json::to_value(inference.backend)
                .expect("inference backend enum must serialize")
                .as_str()
                .expect("inference backend enum must serialize as a string")
                .to_owned()
        })
        .unwrap_or_else(|| "unconfigured".to_owned())
}

fn success_response(result: ModelActivationResult, backend: String) -> ModelSwitchResponse {
    let loaded = result.restarted;
    let changed = result.changed;
    let manifest_generated = result.manifest_generated;
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
            "模型已切换并通过 Rust 推理启动前校验：{}",
            result.candidate.artifact_path.display()
        )
    };
    ModelSwitchResponse {
        deployment: result.deployment.clone(),
        inference: serde_json::json!({
            "selected": backend,
            "available": true,
            "loaded": loaded,
            "configured": true,
            "reason": if !changed {
                "deployment unchanged; runtime was not restarted"
            } else if loaded {
                "active deployment produced a valid DetectionBatch"
            } else {
                "active deployment passed Rust perception preflight"
            },
        }),
        parser_contract,
        preparation: ModelPreparation {
            manifest_action: if manifest_generated {
                "generated"
            } else if changed {
                "reused"
            } else {
                "unchanged"
            },
            reason: if manifest_generated {
                "inspected the TensorRT Engine and generated a validated runtime manifest"
            } else if changed {
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
            backend,
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
                        "候选模型已通过 Rust 推理启动前校验"
                    },
                },
            ],
        },
    }
}
