use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use novasight_store::model_catalog::{
    ConversionJob, ModelArtifact, ModelCatalogError, ModelCatalogResponse, ModelProject,
    ModelVersion, SqliteModelCatalog,
};
use serde::Deserialize;

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
