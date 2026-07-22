use std::time::Duration;

use axum::{
    Json, Router,
    body::Body,
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{CloseFrame, Message, WebSocket},
    },
    http::{Method, Request, StatusCode},
    middleware,
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::{Sink, Stream, StreamExt};
use novasight_core::DeviceReceipt;
use novasight_runtime::{
    AppConfig, ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate, DaemonState,
    ModelActivationError, RuntimeError, RuntimeErrorKind, RuntimeHandle, RuntimeSnapshot,
};
use novasight_store::license::{FileLicenseRepository, LicenseError, LicenseStatus};
use novasight_store::model_catalog::{ModelCatalogError, SqliteModelCatalog};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::dto::{
    CompatibilityHealth, CompatibilityRuntimeStart, CompatibilityRuntimeState,
    CompatibilityStatusFrame,
};
use crate::websocket::status::send_while_receiving;

const COMPATIBILITY_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(200);

mod models;

/// Cuttlefish-style control surface: every mutation delegates to the
/// single daemon-owned RuntimeHandle and returns its immutable snapshot.
pub fn build_control_router(runtime: RuntimeHandle) -> Router {
    build_control_router_with_shutdown(runtime, None)
}

/// Build the control surface with a daemon-owned shutdown signal.
/// WebSocket upgrades observe it and leave Axum's graceful drain.
pub fn build_control_router_with_shutdown(
    runtime: RuntimeHandle,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    build_control_router_with_services(runtime, None, shutdown)
}

pub fn build_control_router_with_services(
    runtime: RuntimeHandle,
    config_service: impl Into<Option<ConfigService>>,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    build_control_router_with_capabilities(runtime, config_service, false, shutdown)
}

pub fn build_control_router_with_capabilities(
    runtime: RuntimeHandle,
    config_service: impl Into<Option<ConfigService>>,
    hardware_output_enabled: bool,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    build_control_router_with_control_plane(
        runtime,
        config_service,
        None,
        None,
        hardware_output_enabled,
        shutdown,
    )
}

pub fn build_control_router_with_control_plane(
    runtime: RuntimeHandle,
    config_service: impl Into<Option<ConfigService>>,
    license: impl Into<Option<FileLicenseRepository>>,
    model_catalog: impl Into<Option<SqliteModelCatalog>>,
    hardware_output_enabled: bool,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    let state = ControlState {
        runtime,
        config: config_service.into(),
        license: license.into(),
        model_catalog: model_catalog.into(),
        hardware_output_enabled,
        shutdown: shutdown.into(),
    };
    let license_gate_enabled = state.license.is_some();
    let router = Router::new()
        .route("/healthz", get(health))
        .route(
            "/api/license",
            get(license_status)
                .put(activate_license)
                .delete(clear_license),
        )
        .route("/api/license/activate", post(activate_license))
        .route("/api/runtime/state", get(legacy_status))
        .route("/api/runtime/start", post(legacy_start))
        .route("/api/runtime/stop", post(legacy_stop))
        .route("/ws/status", get(legacy_events))
        .route("/api/v1/status", get(status))
        .route("/api/v1/config", get(config).patch(update_config))
        .route("/api/v1/runtime/start", post(start))
        .route("/api/v1/runtime/stop", post(stop))
        .route("/api/v1/runtime/restart", post(restart))
        .route("/api/v1/runtime/emergency-stop", post(emergency_stop))
        .route("/api/v1/events", get(events))
        .route("/api/config", get(config).post(update_legacy_config))
        .route("/api/executors", get(executors))
        .merge(models::routes())
        .route(
            "/api/executors/kmnet/connect",
            post(device_lifecycle_managed),
        )
        .route(
            "/api/executors/kmnet/disconnect",
            post(device_lifecycle_managed),
        )
        .route(
            "/api/executors/kmnet/diagnostic-move",
            post(diagnostic_device_move),
        )
        .with_state(state.clone());
    let router = if license_gate_enabled {
        router.layer(middleware::from_fn_with_state(state, require_license))
    } else {
        router
    };
    router.layer(super::app::studio_cors_layer())
}

#[derive(Clone)]
struct ControlState {
    runtime: RuntimeHandle,
    config: Option<ConfigService>,
    license: Option<FileLicenseRepository>,
    model_catalog: Option<SqliteModelCatalog>,
    hardware_output_enabled: bool,
    shutdown: Option<watch::Receiver<bool>>,
}

#[derive(Debug, Deserialize)]
struct LicenseActivationRequest {
    key: String,
}

async fn license_status(
    State(state): State<ControlState>,
) -> Result<Json<LicenseStatus>, ControlApiError> {
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    Ok(Json(
        run_license_operation(license.clone(), |repository| repository.status()).await?,
    ))
}

async fn activate_license(
    State(state): State<ControlState>,
    Json(request): Json<LicenseActivationRequest>,
) -> Result<Json<LicenseStatus>, ControlApiError> {
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    Ok(Json(
        run_license_operation(license.clone(), move |repository| {
            repository.activate(&request.key)
        })
        .await?,
    ))
}

async fn clear_license(
    State(state): State<ControlState>,
) -> Result<Json<LicenseStatus>, ControlApiError> {
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    Ok(Json(
        run_license_operation(license.clone(), |repository| repository.clear()).await?,
    ))
}

async fn run_license_operation(
    repository: FileLicenseRepository,
    operation: impl FnOnce(FileLicenseRepository) -> Result<LicenseStatus, LicenseError>
    + Send
    + 'static,
) -> Result<LicenseStatus, ControlApiError> {
    tokio::task::spawn_blocking(move || operation(repository))
        .await
        .map_err(ControlApiError::LicenseTask)?
        .map_err(ControlApiError::License)
}

async fn require_license(
    State(state): State<ControlState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let path = request.uri().path();
    let method = request.method();
    if is_license_open_path(method, path) {
        return next.run(request).await;
    }
    let Some(license) = state.license.as_ref() else {
        return next.run(request).await;
    };
    let status =
        match run_license_operation(license.clone(), |repository| repository.status()).await {
            Ok(status) => status,
            Err(error) => return error.into_response(),
        };
    if !status.configured || !status.valid {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "detail": "license required",
                "license": status,
            })),
        )
            .into_response();
    }
    if let Some(feature) = required_license_feature(method, path)
        && !status.features.iter().any(|candidate| candidate == feature)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "code": "LICENSE_FEATURE_REQUIRED",
                "detail": format!("license feature {feature} is required"),
                "required_feature": feature,
                "license": status,
            })),
        )
            .into_response();
    }
    next.run(request).await
}

fn is_license_open_path(method: &Method, path: &str) -> bool {
    *method == Method::OPTIONS
        || path == "/healthz"
        || (path == "/api/license" && matches!(*method, Method::GET | Method::PUT))
        || (path == "/api/license/activate" && *method == Method::POST)
        || path == "/ws/status"
        || path == "/api/config/schema"
        || (*method == Method::POST
            && matches!(
                path,
                "/api/runtime/stop" | "/api/v1/runtime/stop" | "/api/v1/runtime/emergency-stop"
            ))
        || !path.starts_with("/api/")
}

fn required_license_feature(method: &Method, path: &str) -> Option<&'static str> {
    if path == "/api/config" || path == "/api/v1/config" {
        return Some(if *method == Method::GET {
            "config_read"
        } else {
            "config_write"
        });
    }
    if path.starts_with("/api/runtime/")
        || path.starts_with("/api/v1/runtime/")
        || path == "/api/v1/status"
        || path == "/api/v1/events"
    {
        return Some("runtime");
    }
    if path.starts_with("/api/executors") {
        return Some("hardware_control");
    }
    if path.starts_with("/api/models") {
        return Some("models");
    }
    if path.starts_with("/api/capture") {
        return Some("capture");
    }
    None
}

async fn health(State(state): State<ControlState>) -> Json<CompatibilityHealth> {
    let daemon = state.runtime.snapshot().daemon.state;
    Json(CompatibilityHealth {
        ok: daemon == DaemonState::Ready,
    })
}

async fn legacy_status(State(state): State<ControlState>) -> Json<CompatibilityRuntimeState> {
    let snapshot = state.runtime.snapshot();
    Json(compatibility_state(&state, &snapshot).await)
}

async fn legacy_start(
    State(state): State<ControlState>,
) -> Result<Json<CompatibilityRuntimeStart>, ControlApiError> {
    ensure_config_effective(&state).await?;
    let snapshot = state.runtime.start().await?;
    Ok(Json(CompatibilityRuntimeStart::from(&snapshot)))
}

async fn legacy_stop(
    State(state): State<ControlState>,
) -> Result<Json<CompatibilityRuntimeState>, ControlApiError> {
    let snapshot = state.runtime.stop().await?;
    Ok(Json(compatibility_state(&state, &snapshot).await))
}

async fn compatibility_state(
    state: &ControlState,
    snapshot: &RuntimeSnapshot,
) -> CompatibilityRuntimeState {
    let config = match &state.config {
        Some(service) => Some(service.snapshot().await),
        None => None,
    };
    let effective_revision = state.config.as_ref().map(ConfigService::effective_revision);
    CompatibilityRuntimeState::new(
        snapshot,
        config.as_ref(),
        effective_revision,
        state.hardware_output_enabled,
    )
}

async fn executors(State(state): State<ControlState>) -> Json<serde_json::Value> {
    let snapshot = state.runtime.snapshot();
    let compatibility = compatibility_state(&state, &snapshot).await;
    Json(
        serde_json::to_value(compatibility.executor)
            .expect("compatibility executor DTO must serialize"),
    )
}

async fn device_lifecycle_managed() -> ControlApiError {
    ControlApiError::DeviceLifecycleManaged
}

#[derive(Debug, Deserialize)]
struct DiagnosticMoveRequest {
    dx: i32,
    dy: i32,
    #[serde(default = "one_u32")]
    repeat: u32,
    #[serde(default)]
    interval_ms: u64,
    #[serde(default)]
    move_kind: Option<String>,
}

const fn one_u32() -> u32 {
    1
}

#[derive(Debug, Serialize)]
struct DiagnosticMoveResponse {
    sent: bool,
    queued: bool,
    steps_sent: u32,
    message: &'static str,
    receipt: DeviceReceipt,
    status: DiagnosticDeviceStatus,
    metadata: DiagnosticMetadata,
}

#[derive(Debug, Serialize)]
struct DiagnosticDeviceStatus {
    available: bool,
    connected: bool,
    connection_state: &'static str,
    move_count: u64,
    last_dx: i32,
    last_dy: i32,
    managed_by_runtime: bool,
}

#[derive(Debug, Serialize)]
struct DiagnosticMetadata {
    api_name: &'static str,
}

async fn diagnostic_device_move(
    State(state): State<ControlState>,
    Json(request): Json<DiagnosticMoveRequest>,
) -> Result<Json<DiagnosticMoveResponse>, ControlApiError> {
    if !state.hardware_output_enabled {
        return Err(ControlApiError::HardwareOutputDisabled);
    }
    ensure_config_effective(&state).await?;
    let config = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?
        .snapshot()
        .await;
    if config.device.is_none() {
        return Err(ControlApiError::DeviceNotConfigured);
    }
    let move_kind = request.move_kind.as_deref().unwrap_or("raw");
    if request.repeat != 1 || request.interval_ms != 0 || move_kind != "raw" {
        return Err(ControlApiError::UnsupportedDiagnostic(
            "only one immediate raw move is supported; repeat must be 1 and interval_ms must be 0"
                .to_owned(),
        ));
    }
    let receipt = state
        .runtime
        .diagnose_device_move(request.dx, request.dy)
        .await?;
    Ok(Json(DiagnosticMoveResponse {
        sent: true,
        queued: false,
        steps_sent: 1,
        message: "diagnostic move accepted by the daemon-owned device adapter",
        status: DiagnosticDeviceStatus {
            available: true,
            connected: true,
            connection_state: "connected",
            move_count: receipt.attempt,
            last_dx: receipt.delta_x_counts,
            last_dy: receipt.delta_y_counts,
            managed_by_runtime: true,
        },
        metadata: DiagnosticMetadata {
            api_name: "rust_pointer_device_send",
        },
        receipt,
    }))
}

async fn ensure_config_effective(state: &ControlState) -> Result<(), ControlApiError> {
    if let Some(service) = &state.config {
        service.ensure_effective().await?;
    }
    Ok(())
}

async fn config(State(state): State<ControlState>) -> Result<Json<AppConfig>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    Ok(Json(service.snapshot().await))
}

async fn update_config(
    State(state): State<ControlState>,
    Json(update): Json<ConfigFieldUpdate>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    Ok(Json(service.update_field(update).await?))
}

async fn update_legacy_config(
    State(state): State<ControlState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    let is_field_update = payload.get("section").is_some()
        || payload.get("key").is_some()
        || payload.get("value").is_some();
    let update = if is_field_update {
        let update =
            serde_json::from_value(payload).map_err(ControlApiError::InvalidFieldUpdate)?;
        service.update_field(update).await?
    } else {
        service.replace(payload).await?
    };
    Ok(Json(update))
}

async fn status(State(state): State<ControlState>) -> Json<RuntimeSnapshot> {
    Json(state.runtime.snapshot())
}

async fn start(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    ensure_config_effective(&state).await?;
    Ok(Json(state.runtime.start().await?))
}

async fn stop(State(state): State<ControlState>) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.stop().await?))
}

async fn restart(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    ensure_config_effective(&state).await?;
    Ok(Json(state.runtime.restart().await?))
}

async fn emergency_stop(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.emergency_stop().await?))
}

async fn events(websocket: WebSocketUpgrade, State(state): State<ControlState>) -> Response {
    websocket.on_upgrade(move |socket| stream_events(socket, state.runtime, state.shutdown))
}

#[derive(Debug, Default, Deserialize)]
struct CompatibilityStatusQuery {
    topic: Option<String>,
}

async fn legacy_events(
    websocket: WebSocketUpgrade,
    Query(query): Query<CompatibilityStatusQuery>,
    State(state): State<ControlState>,
) -> Response {
    if let Some(license) = state.license.as_ref() {
        match run_license_operation(license.clone(), |repository| repository.status()).await {
            Ok(status) if status.configured && status.valid => {}
            Ok(_) => {
                return websocket.on_upgrade(close_unlicensed_websocket);
            }
            Err(error) => return error.into_response(),
        }
    }
    websocket.on_upgrade(move |socket| stream_legacy_events(socket, state, query))
}

async fn close_unlicensed_websocket(mut socket: WebSocket) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: 4401,
            reason: "license required".into(),
        })))
        .await;
}

async fn stream_legacy_events(
    socket: WebSocket,
    state: ControlState,
    query: CompatibilityStatusQuery,
) {
    let mut snapshots = state.runtime.subscribe();
    let mut shutdown = state.shutdown.clone();
    let topic = normalize_compatibility_topic(query.topic.as_deref());
    let mut heartbeat = tokio::time::interval(COMPATIBILITY_HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    heartbeat.tick().await;
    let (mut outbound, mut inbound) = socket.split();
    loop {
        let current = snapshots.borrow_and_update().clone();
        let frame = CompatibilityStatusFrame {
            kind: "runtime_snapshot",
            topic: topic.clone(),
            full: true,
            state: compatibility_state(&state, current.as_ref()).await,
        };
        let Ok(payload) = serde_json::to_string(&frame) else {
            return;
        };
        match send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text(payload.into()),
            &mut shutdown,
        )
        .await
        {
            SendOutcome::Sent => {}
            SendOutcome::Closed | SendOutcome::Shutdown => return,
        }

        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
            }
            _ = heartbeat.tick() => {}
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    Some(Ok(_)) => {}
                }
            }
            () = shutdown_requested(&mut shutdown) => return,
        }
    }
}

fn normalize_compatibility_topic(topic: Option<&str>) -> String {
    let topic = topic.unwrap_or_default().trim().to_ascii_lowercase();
    match topic.as_str() {
        "summary" | "capture" | "infer" | "control" | "latency" => topic,
        _ => "full".to_owned(),
    }
}

#[derive(Serialize)]
struct RuntimeEvent<'a> {
    kind: &'static str,
    snapshot: &'a RuntimeSnapshot,
}

async fn stream_events(
    socket: WebSocket,
    runtime: RuntimeHandle,
    mut shutdown: Option<watch::Receiver<bool>>,
) {
    let mut snapshots = runtime.subscribe();
    let (mut outbound, mut inbound) = socket.split();
    loop {
        let current = snapshots.borrow_and_update().clone();
        let event = RuntimeEvent {
            kind: "runtime_snapshot",
            snapshot: current.as_ref(),
        };
        let Ok(payload) = serde_json::to_string(&event) else {
            return;
        };
        match send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text(payload.into()),
            &mut shutdown,
        )
        .await
        {
            SendOutcome::Sent => {}
            SendOutcome::Closed | SendOutcome::Shutdown => return,
        }

        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
            }
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    Some(Ok(_)) => {}
                }
            }
            () = shutdown_requested(&mut shutdown) => return,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SendOutcome {
    Sent,
    Closed,
    Shutdown,
}

async fn send_or_shutdown<S, R, E>(
    outbound: &mut S,
    inbound: &mut R,
    message: Message,
    shutdown: &mut Option<watch::Receiver<bool>>,
) -> SendOutcome
where
    S: Sink<Message> + Unpin,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    tokio::select! {
        sent = send_while_receiving(outbound, inbound, message) => {
            if sent {
                SendOutcome::Sent
            } else {
                SendOutcome::Closed
            }
        }
        () = shutdown_requested(shutdown) => SendOutcome::Shutdown,
    }
}

async fn shutdown_requested(shutdown: &mut Option<watch::Receiver<bool>>) {
    let Some(receiver) = shutdown else {
        std::future::pending::<()>().await;
        return;
    };
    if *receiver.borrow() {
        return;
    }
    let _ = receiver.changed().await;
}

enum ControlApiError {
    Runtime(RuntimeError),
    Config(ConfigServiceError),
    ConfigUnavailable,
    InvalidFieldUpdate(serde_json::Error),
    HardwareOutputDisabled,
    DeviceNotConfigured,
    DeviceLifecycleManaged,
    UnsupportedDiagnostic(String),
    License(LicenseError),
    LicenseTask(tokio::task::JoinError),
    LicenseUnavailable,
    ModelCatalog(ModelCatalogError),
    ModelCatalogTask(tokio::task::JoinError),
    ModelCatalogUnavailable,
    ModelActivation(ModelActivationError),
}

impl From<LicenseError> for ControlApiError {
    fn from(error: LicenseError) -> Self {
        Self::License(error)
    }
}

impl From<RuntimeError> for ControlApiError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<ConfigServiceError> for ControlApiError {
    fn from(error: ConfigServiceError) -> Self {
        Self::Config(error)
    }
}

#[derive(Serialize)]
struct ControlErrorBody {
    code: &'static str,
    message: String,
    detail: String,
}

impl IntoResponse for ControlApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::Runtime(error) => {
                let status = match error.kind {
                    RuntimeErrorKind::InvalidPipelineState => StatusCode::CONFLICT,
                    RuntimeErrorKind::SupervisorUnavailable
                    | RuntimeErrorKind::SupervisorClosed
                    | RuntimeErrorKind::SupervisorReplyLost
                    | RuntimeErrorKind::PipelineUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                    RuntimeErrorKind::DeviceUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                    RuntimeErrorKind::InvalidDeviceCommand => StatusCode::BAD_REQUEST,
                    RuntimeErrorKind::PipelineRejected
                    | RuntimeErrorKind::RuntimeEpochExhausted
                    | RuntimeErrorKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, error.kind.code(), error.message)
            }
            Self::Config(error) => {
                let status = match error.code() {
                    "CONFIG_REVISION_CONFLICT" => StatusCode::CONFLICT,
                    "CONFIG_RESTART_REQUIRED" => StatusCode::CONFLICT,
                    "CONFIG_PARSE_ERROR"
                    | "CONFIG_VALIDATION_ERROR"
                    | "CONFIG_RESERVED_LEGACY_KEY"
                    | "CONFIG_INVALID_FIELD_TARGET"
                    | "CONFIG_FIELD_VALUE_INVALID"
                    | "CONFIG_REPLACEMENT_INVALID"
                    | "CONFIG_REPLACEMENT_REVISION_REQUIRED" => StatusCode::BAD_REQUEST,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                let code = error.code();
                (status, code, error.to_string())
            }
            Self::ConfigUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "CONFIG_SERVICE_UNAVAILABLE",
                "configuration service is not attached to this control surface".to_owned(),
            ),
            Self::InvalidFieldUpdate(error) => (
                StatusCode::BAD_REQUEST,
                "CONFIG_FIELD_UPDATE_INVALID",
                format!("expected a field update with section, key, and value: {error}"),
            ),
            Self::HardwareOutputDisabled => (
                StatusCode::SERVICE_UNAVAILABLE,
                "HARDWARE_OUTPUT_DISABLED",
                "hardware diagnostics are unavailable in dry-run mode".to_owned(),
            ),
            Self::DeviceNotConfigured => (
                StatusCode::SERVICE_UNAVAILABLE,
                "DEVICE_NOT_CONFIGURED",
                "no production pointer device is configured".to_owned(),
            ),
            Self::DeviceLifecycleManaged => (
                StatusCode::CONFLICT,
                "DEVICE_LIFECYCLE_MANAGED_BY_RUNTIME",
                "the Rust daemon owns device connection lifetime; use runtime start/stop"
                    .to_owned(),
            ),
            Self::UnsupportedDiagnostic(message) => (
                StatusCode::BAD_REQUEST,
                "DEVICE_DIAGNOSTIC_UNSUPPORTED",
                message,
            ),
            Self::License(error) => {
                let status = if error.is_client_error() {
                    StatusCode::BAD_REQUEST
                } else if error.is_configuration_error() {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                (status, error.code(), error.to_string())
            }
            Self::LicenseTask(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "LICENSE_TASK_FAILED",
                format!("license storage task failed: {error}"),
            ),
            Self::LicenseUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "LICENSE_SERVICE_UNAVAILABLE",
                "license service is not configured".to_owned(),
            ),
            Self::ModelCatalog(error) => match error {
                ModelCatalogError::ProjectNotFound(_)
                | ModelCatalogError::VersionNotFound(_)
                | ModelCatalogError::ArtifactNotFound(_)
                | ModelCatalogError::DeploymentNotFound(_) => (
                    StatusCode::NOT_FOUND,
                    "MODEL_CATALOG_NOT_FOUND",
                    error.to_string(),
                ),
                ModelCatalogError::ArtifactProjectMismatch { .. }
                | ModelCatalogError::ArtifactNotReady(_)
                | ModelCatalogError::RollbackUnavailable(_) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "MODEL_DEPLOYMENT_INVALID",
                    error.to_string(),
                ),
                ModelCatalogError::DeploymentChangedDuringActivation { .. } => (
                    StatusCode::CONFLICT,
                    "MODEL_DEPLOYMENT_CONFLICT",
                    error.to_string(),
                ),
                _ => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "MODEL_CATALOG_FAILED",
                    error.to_string(),
                ),
            },
            Self::ModelCatalogTask(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MODEL_CATALOG_TASK_FAILED",
                format!("model catalog task failed: {error}"),
            ),
            Self::ModelCatalogUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "MODEL_CATALOG_UNAVAILABLE",
                "model catalog is not configured".to_owned(),
            ),
            Self::ModelActivation(error) => match error {
                ModelActivationError::Catalog(error) => match error {
                    ModelCatalogError::ProjectNotFound(_)
                    | ModelCatalogError::VersionNotFound(_)
                    | ModelCatalogError::ArtifactNotFound(_)
                    | ModelCatalogError::DeploymentNotFound(_) => (
                        StatusCode::NOT_FOUND,
                        "MODEL_CATALOG_NOT_FOUND",
                        error.to_string(),
                    ),
                    ModelCatalogError::ArtifactProjectMismatch { .. }
                    | ModelCatalogError::ArtifactNotReady(_)
                    | ModelCatalogError::RollbackUnavailable(_) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "MODEL_DEPLOYMENT_INVALID",
                        error.to_string(),
                    ),
                    ModelCatalogError::DeploymentChangedDuringActivation { .. } => (
                        StatusCode::CONFLICT,
                        "MODEL_DEPLOYMENT_CONFLICT",
                        error.to_string(),
                    ),
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "MODEL_CATALOG_FAILED",
                        error.to_string(),
                    ),
                },
                ModelActivationError::Runtime(error) => {
                    let status = match error.kind {
                        RuntimeErrorKind::InvalidPipelineState => StatusCode::CONFLICT,
                        RuntimeErrorKind::SupervisorUnavailable
                        | RuntimeErrorKind::SupervisorClosed
                        | RuntimeErrorKind::SupervisorReplyLost
                        | RuntimeErrorKind::PipelineUnavailable
                        | RuntimeErrorKind::DeviceUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                        RuntimeErrorKind::InvalidDeviceCommand => StatusCode::BAD_REQUEST,
                        RuntimeErrorKind::PipelineRejected
                        | RuntimeErrorKind::RuntimeEpochExhausted
                        | RuntimeErrorKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
                    };
                    (status, error.kind.code(), error.message)
                }
                ModelActivationError::Failed { message, .. } => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "MODEL_ACTIVATION_FAILED",
                    message,
                ),
                ModelActivationError::Unavailable => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "MODEL_CATALOG_UNAVAILABLE",
                    "model catalog is not configured in the runtime supervisor".to_owned(),
                ),
                ModelActivationError::PerceptionUnavailable => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "MODEL_PREFLIGHT_UNAVAILABLE",
                    "model activation requires a configured perception preflight adapter"
                        .to_owned(),
                ),
            },
        };
        let body = ControlErrorBody {
            code,
            detail: message.clone(),
            message,
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll},
        time::Duration,
    };

    use futures_util::{Sink, task::noop_waker};

    use super::*;

    struct PendingSink;

    impl Sink<Message> for PendingSink {
        type Error = ();

        fn poll_ready(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn start_send(self: Pin<&mut Self>, _item: Message) -> Result<(), Self::Error> {
            unreachable!("a permanently backpressured sink is never ready")
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
    }

    #[tokio::test]
    async fn daemon_shutdown_interrupts_a_backpressured_websocket_send() {
        let mut outbound = PendingSink;
        let mut inbound = futures_util::stream::pending::<Result<Message, ()>>();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut shutdown = Some(shutdown_rx);
        let operation = send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text("snapshot".into()),
            &mut shutdown,
        );
        tokio::pin!(operation);
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(operation.as_mut().poll(&mut context).is_pending());

        shutdown_tx.send_replace(true);
        let outcome = tokio::time::timeout(Duration::from_millis(100), operation)
            .await
            .expect("daemon shutdown must not wait for a backpressured peer");

        assert_eq!(outcome, SendOutcome::Shutdown);
    }
}
