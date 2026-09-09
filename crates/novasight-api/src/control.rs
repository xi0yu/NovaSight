use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::{
    Extension, Json, Router,
    body::{Body, Bytes},
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{CloseFrame, Message, WebSocket},
    },
    http::{HeaderMap, Method, Request, StatusCode, header::SET_COOKIE},
    middleware,
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::{Sink, Stream, StreamExt};
use novasight_core::{
    CaptureCapabilities, CaptureCapabilityProbe, CaptureProbeError, CaptureSelectionError,
    CaptureSelectionPreference, DeviceReceipt, select_capture_profile_for_formats,
};
use novasight_runtime::{
    AppConfig, ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate, DaemonState,
    ModelActivationError, ModelIngressError, PipelineState, RuntimeError, RuntimeErrorKind,
    RuntimeHandle, RuntimeSnapshot,
};
use novasight_store::license::{FileLicenseRepository, LicenseDenial, LicenseError, LicenseStatus};
use novasight_store::model_catalog::{ModelCatalogError, SqliteModelCatalog};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, watch};
use tracing::Instrument;

use crate::dto::{
    ConfigSchemaResponse, RuntimeHealth, RuntimeStatusState, serialize_runtime_status_frame,
};
use crate::license_session::LicenseSession;
use crate::websocket::status::send_while_receiving;

const STATUS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
static NEXT_HTTP_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug)]
struct TrustedLocalControl;

mod models;

/// Cuttlefish-style control surface: every mutation delegates to the
/// single daemon-owned RuntimeHandle and returns its immutable snapshot.
pub fn build_control_router(runtime: RuntimeHandle) -> Router {
    build_control_router_with_shutdown(runtime, None)
}

/// Mark a router served only through the daemon-owned Unix socket. The socket
/// permissions are its session boundary, while license features remain
/// enforced by the same middleware as HTTP.
pub fn with_trusted_local_control(router: Router) -> Router {
    router.layer(Extension(TrustedLocalControl))
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
    build_control_router_with_platform_queries(
        runtime,
        config_service,
        license,
        model_catalog,
        None,
        hardware_output_enabled,
        shutdown,
    )
}

pub fn build_control_router_with_platform_queries(
    runtime: RuntimeHandle,
    config_service: impl Into<Option<ConfigService>>,
    license: impl Into<Option<FileLicenseRepository>>,
    model_catalog: impl Into<Option<SqliteModelCatalog>>,
    capture_probe: impl Into<Option<Arc<dyn CaptureCapabilityProbe>>>,
    hardware_output_enabled: bool,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    let state = ControlState {
        daemon_instance_id: generate_daemon_instance_id(),
        runtime,
        config: config_service.into(),
        license: license.into(),
        model_catalog: model_catalog.into(),
        capture_probe: capture_probe.into(),
        hardware_output_enabled,
        shutdown: shutdown.into(),
        lifecycle_lock: Arc::new(Mutex::new(())),
        license_session: LicenseSession::new(),
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
        .route("/api/runtime/state", get(runtime_status))
        .route("/api/runtime/start", post(runtime_start))
        .route("/api/runtime/stop", post(runtime_stop))
        .route("/api/runtime/emergency-stop", post(runtime_emergency_stop))
        .route("/ws/status", get(runtime_events))
        .route("/api/v1/status", get(status))
        .route("/api/v1/config", get(config).patch(update_config))
        .route("/api/v1/config/commands", post(apply_config_command))
        .route("/api/v1/runtime/start", post(start))
        .route("/api/v1/runtime/stop", post(stop))
        .route("/api/v1/runtime/restart", post(restart))
        .route("/api/v1/runtime/emergency-stop", post(emergency_stop))
        .route("/api/v1/daemon/shutdown", post(shutdown_daemon))
        .route("/api/v1/events", get(events))
        .route("/api/config", get(config).post(update_config_document))
        .route("/api/config/schema", get(config_schema))
        .route("/api/capture/state", get(capture_state))
        .route(
            "/api/capture/preview",
            get(preview_status).post(set_preview),
        )
        .route("/api/capture/stream.mjpg", get(preview_stream))
        .route("/api/crosshair", get(crosshair_status))
        .route("/api/crosshair/learn", post(learn_crosshair))
        .route(
            "/api/crosshair/template",
            axum::routing::delete(clear_crosshair),
        )
        .route(
            "/api/crosshair/template.png",
            get(crosshair_template_preview),
        )
        .route(
            "/api/capture/capabilities",
            get(capture_capabilities).post(post_capture_capabilities),
        )
        .route("/api/capture/select", post(select_capture))
        .route("/api/capture/stop", post(stop_capture))
        .route("/api/executors", get(executors))
        .route("/api/executors/kmnet/buttons", get(device_buttons))
        .merge(models::routes())
        .route("/api/executors/kmnet/connect", post(connect_device))
        .route("/api/executors/kmnet/disconnect", post(disconnect_device))
        .route(
            "/api/executors/kmnet/diagnostic-move",
            post(diagnostic_device_move),
        )
        .with_state(state.clone());
    let router = if license_gate_enabled {
        router.layer(middleware::from_fn_with_state(
            state.clone(),
            require_license,
        ))
    } else {
        router
    };
    router
        .layer(super::app::studio_cors_layer())
        .layer(middleware::from_fn(log_http_request))
}

async fn log_http_request(request: Request<Body>, next: Next) -> Response {
    let request_id = NEXT_HTTP_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let span = tracing::info_span!(
        "http_request",
        service = "novasightd",
        version = env!("CARGO_PKG_VERSION"),
        request_id,
        method = %method,
        path = %path,
    );
    async move {
        let started = Instant::now();
        let response = next.run(request).await;
        let status = response.status();
        let latency_ms = started.elapsed().as_millis() as u64;
        tracing::info!(
            http_status = status.as_u16(),
            latency_ms,
            "api request completed"
        );
        response
    }
    .instrument(span)
    .await
}

#[derive(Clone)]
struct ControlState {
    daemon_instance_id: Arc<str>,
    runtime: RuntimeHandle,
    config: Option<ConfigService>,
    license: Option<FileLicenseRepository>,
    model_catalog: Option<SqliteModelCatalog>,
    capture_probe: Option<Arc<dyn CaptureCapabilityProbe>>,
    hardware_output_enabled: bool,
    shutdown: Option<watch::Receiver<bool>>,
    lifecycle_lock: Arc<Mutex<()>>,
    license_session: LicenseSession,
}

fn generate_daemon_instance_id() -> Arc<str> {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
        .into()
}

#[derive(Debug, Deserialize)]
struct LicenseActivationRequest {
    key: String,
}

async fn license_status(State(state): State<ControlState>) -> Result<Response, ControlApiError> {
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    let status = run_license_operation(license.clone(), |repository| repository.status()).await?;
    license_response(&state, &status, status.clone())
}

async fn activate_license(
    State(state): State<ControlState>,
    Json(request): Json<LicenseActivationRequest>,
) -> Result<Response, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    let status = run_license_operation(license.clone(), move |repository| {
        repository.activate(&request.key)
    })
    .await?;
    tracing::info!(
        valid = status.valid,
        tier = %status.tier,
        credential_format = %status.credential_format,
        license_id = %status.license_id,
        fingerprint = %status.fingerprint,
        "license activation completed"
    );
    stop_if_license_disallows_runtime(&state, &status).await?;
    license_response(&state, &status, status.clone())
}

async fn clear_license(State(state): State<ControlState>) -> Result<Response, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let license = state
        .license
        .as_ref()
        .ok_or(ControlApiError::LicenseUnavailable)?;
    ensure_runtime_safe_locked(&state).await?;
    let status = run_license_operation(license.clone(), |repository| repository.clear()).await?;
    tracing::info!("license cleared");
    let mut response = Json(status).into_response();
    response
        .headers_mut()
        .append(SET_COOKIE, LicenseSession::clear_cookie());
    Ok(response)
}

fn license_response<T: Serialize>(
    state: &ControlState,
    status: &LicenseStatus,
    payload: T,
) -> Result<Response, ControlApiError> {
    let cookie = state
        .license_session
        .issue_cookie(status)
        .map_err(ControlApiError::LicenseSession)?;
    let mut response = Json(payload).into_response();
    response.headers_mut().append(SET_COOKIE, cookie);
    Ok(response)
}

async fn stop_if_license_disallows_runtime(
    state: &ControlState,
    status: &LicenseStatus,
) -> Result<(), ControlApiError> {
    if status
        .authorize_runtime(hardware_output_requested(state).await)
        .is_err()
        && matches!(
            state.runtime.snapshot().pipeline.state,
            PipelineState::Starting | PipelineState::Running | PipelineState::Standby
        )
    {
        emergency_stop_locked(state).await?;
    }
    Ok(())
}

fn runtime_snapshot_confirms_safe(snapshot: &RuntimeSnapshot) -> bool {
    snapshot.pipeline.state == PipelineState::Stopped
        && !snapshot.pipeline_metrics.device_connected
        && !snapshot.pipeline_metrics.control.emit_allowed
}

async fn emergency_stop_locked(state: &ControlState) -> Result<RuntimeSnapshot, ControlApiError> {
    let snapshot = state.runtime.emergency_stop().await?;
    if !runtime_snapshot_confirms_safe(&snapshot) {
        return Err(ControlApiError::Runtime(
            RuntimeError::invalid_pipeline_state(
                "emergency stop completed without a confirmed stopped, disconnected, output-blocked snapshot",
            ),
        ));
    }
    Ok(snapshot)
}

async fn ensure_runtime_safe_locked(
    state: &ControlState,
) -> Result<RuntimeSnapshot, ControlApiError> {
    let current = state.runtime.snapshot();
    if runtime_snapshot_confirms_safe(&current) {
        return Ok(current);
    }
    emergency_stop_locked(state).await
}

async fn emergency_stop_with_lifecycle_barrier(
    state: &ControlState,
) -> Result<RuntimeSnapshot, ControlApiError> {
    let immediate_result = state.runtime.emergency_stop().await;
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let snapshot = emergency_stop_locked(state).await?;
    if let Err(error) = immediate_result {
        tracing::warn!(
            error = %error,
            "initial emergency stop attempt failed; lifecycle-barrier stop confirmed safe"
        );
    }
    Ok(snapshot)
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
        tracing::warn!(
            error_code = "LICENSE_REQUIRED",
            configured = status.configured,
            valid = status.valid,
            reason = %status.message,
            "license authorization rejected"
        );
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "code": "LICENSE_REQUIRED",
                "detail": "license required",
                "message": "a valid license is required",
                "license": status,
            })),
        )
            .into_response();
    }
    let trusted_local_control = request.extensions().get::<TrustedLocalControl>().is_some();
    let refresh_session = if trusted_local_control {
        false
    } else {
        match state.license_session.verify(request.headers(), &status) {
            Ok(session) => session.refresh,
            Err(reason) => {
                tracing::warn!(
                    error_code = "LICENSE_SESSION_REQUIRED",
                    reason,
                    "license session rejected"
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({
                        "code": "LICENSE_SESSION_REQUIRED",
                        "detail": "license session required",
                        "message": "refresh the license status to establish a browser session",
                        "license": status,
                    })),
                )
                    .into_response();
            }
        }
    };
    if let Some(feature) = required_license_feature(method, path)
        && !status.features.iter().any(|candidate| candidate == feature)
    {
        tracing::warn!(
            error_code = "LICENSE_FEATURE_REQUIRED",
            required_feature = feature,
            "license authorization rejected"
        );
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "code": "LICENSE_FEATURE_REQUIRED",
                "detail": format!("license feature {feature} is required"),
                "message": format!("license feature {feature} is required"),
                "required_feature": feature,
                "license": status,
            })),
        )
            .into_response();
    }
    if is_runtime_start(method, path)
        && hardware_output_requested(&state).await
        && !status
            .features
            .iter()
            .any(|candidate| candidate == "hardware_control")
    {
        tracing::warn!(
            error_code = "LICENSE_FEATURE_REQUIRED",
            required_feature = "hardware_control",
            "license authorization rejected"
        );
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "code": "LICENSE_FEATURE_REQUIRED",
                "detail": "license feature hardware_control is required",
                "message": "license feature hardware_control is required",
                "required_feature": "hardware_control",
                "license": status,
            })),
        )
            .into_response();
    }
    let mut response = next.run(request).await;
    if refresh_session {
        match state.license_session.issue_cookie(&status) {
            Ok(cookie) => {
                response.headers_mut().append(SET_COOKIE, cookie);
            }
            Err(error) => {
                tracing::error!(error_code = "LICENSE_SESSION_FAILED", %error);
                return ControlApiError::LicenseSession(error).into_response();
            }
        }
    }
    response
}

fn is_runtime_start(method: &Method, path: &str) -> bool {
    *method == Method::POST
        && matches!(
            path,
            "/api/runtime/start" | "/api/v1/runtime/start" | "/api/v1/runtime/restart"
        )
}

fn is_license_open_path(method: &Method, path: &str) -> bool {
    *method == Method::OPTIONS
        || path == "/healthz"
        || (path == "/api/license" && matches!(*method, Method::GET | Method::PUT | Method::DELETE))
        || (path == "/api/license/activate" && *method == Method::POST)
        || path == "/ws/status"
        || path == "/api/config/schema"
        || (*method == Method::POST
            && matches!(
                path,
                "/api/runtime/stop"
                    | "/api/runtime/emergency-stop"
                    | "/api/v1/runtime/stop"
                    | "/api/v1/runtime/emergency-stop"
                    | "/api/v1/daemon/shutdown"
            ))
        || !path.starts_with("/api/")
}

fn required_license_feature(method: &Method, path: &str) -> Option<&'static str> {
    if path == "/api/v1/config/commands" {
        return Some("config_write");
    }
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

async fn health(State(state): State<ControlState>) -> Json<RuntimeHealth> {
    let daemon = state.runtime.snapshot().daemon.state;
    Json(RuntimeHealth {
        ok: daemon == DaemonState::Ready && state.runtime.is_supervisor_alive(),
    })
}

async fn runtime_status(State(state): State<ControlState>) -> Json<RuntimeStatusState> {
    let snapshot = state.runtime.snapshot();
    Json(runtime_state(&state, &snapshot).await)
}

async fn runtime_start(State(state): State<ControlState>) -> Result<StatusCode, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    prepare_config_for_start(&state).await?;
    state.runtime.start().await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn runtime_stop(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeStatusState>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let stopped = state.runtime.stop().await?;
    let snapshot = if runtime_snapshot_confirms_safe(&stopped) {
        stopped
    } else {
        // Ordinary Stop has the same externally visible safety postcondition as
        // E-stop. If the graceful path does not prove it, fail closed now.
        emergency_stop_locked(&state).await?
    };
    Ok(Json(runtime_state(&state, &snapshot).await))
}

async fn runtime_emergency_stop(
    State(state): State<ControlState>,
) -> Result<StatusCode, ControlApiError> {
    emergency_stop_with_lifecycle_barrier(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn runtime_state(state: &ControlState, snapshot: &RuntimeSnapshot) -> RuntimeStatusState {
    runtime_state_for_topic(state, snapshot, "full").await
}

async fn runtime_state_for_topic(
    state: &ControlState,
    snapshot: &RuntimeSnapshot,
    topic: &'static str,
) -> RuntimeStatusState {
    let config = match &state.config {
        Some(service) => Some(service.snapshot().await),
        None => None,
    };
    let effective_config = state
        .config
        .as_ref()
        .map(ConfigService::blocking_effective_snapshot);
    let effective_revision = state.config.as_ref().map(ConfigService::effective_revision);
    RuntimeStatusState::for_topic(
        &state.daemon_instance_id,
        snapshot,
        config.as_ref(),
        effective_config.as_ref(),
        effective_revision,
        state.hardware_output_enabled,
        state.runtime.preview_snapshot().as_ref(),
        state.runtime.crosshair_snapshot().as_ref(),
        topic,
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreviewRequest {
    enabled: bool,
}

async fn preview_status(
    State(state): State<ControlState>,
) -> Result<Json<novasight_runtime::PreviewSnapshot>, ControlApiError> {
    state
        .runtime
        .preview_snapshot()
        .map(Json)
        .ok_or_else(|| ControlApiError::Runtime(RuntimeError::pipeline_unavailable()))
}

async fn set_preview(
    State(state): State<ControlState>,
    Json(request): Json<PreviewRequest>,
) -> Result<Json<novasight_runtime::PreviewSnapshot>, ControlApiError> {
    state
        .runtime
        .set_preview_active(request.enabled)
        .await
        .map(Json)
        .map_err(Into::into)
}

#[derive(Debug, Default, Deserialize)]
struct PreviewStreamQuery {
    fps: Option<u32>,
}

async fn preview_stream(
    State(state): State<ControlState>,
    Query(query): Query<PreviewStreamQuery>,
) -> Result<Response, ControlApiError> {
    let configured_fps = match state.config.as_ref() {
        Some(config) => config.snapshot().await.limits.stream_fps,
        None => 30,
    };
    let configured_fps = configured_fps.max(1);
    let fps = query.fps.unwrap_or(configured_fps).clamp(1, configured_fps);
    let interval = Duration::from_secs_f64(1.0 / f64::from(fps));
    let subscription = state.runtime.subscribe_preview()?;
    let stream = futures_util::stream::unfold(subscription, move |mut subscription| async move {
        tokio::time::sleep(interval).await;
        let frame = subscription.next().await?;
        let mut part = Vec::with_capacity(frame.jpeg.len() + 96);
        part.extend_from_slice(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ");
        part.extend_from_slice(frame.jpeg.len().to_string().as_bytes());
        part.extend_from_slice(b"\r\n\r\n");
        part.extend_from_slice(&frame.jpeg);
        part.extend_from_slice(b"\r\n");
        Some((
            Ok::<Bytes, std::convert::Infallible>(Bytes::from(part)),
            subscription,
        ))
    });
    Ok((
        [
            (
                axum::http::header::CONTENT_TYPE,
                "multipart/x-mixed-replace; boundary=frame",
            ),
            (axum::http::header::CACHE_CONTROL, "no-store, no-cache"),
        ],
        Body::from_stream(stream),
    )
        .into_response())
}

async fn crosshair_status(
    State(state): State<ControlState>,
) -> Result<Json<novasight_runtime::CrosshairSnapshot>, ControlApiError> {
    state
        .runtime
        .crosshair_snapshot()
        .map(Json)
        .ok_or_else(|| ControlApiError::Runtime(RuntimeError::pipeline_unavailable()))
}

async fn learn_crosshair(
    State(state): State<ControlState>,
) -> Result<Json<serde_json::Value>, ControlApiError> {
    let template = state.runtime.learn_crosshair()?;
    let status = state
        .runtime
        .crosshair_snapshot()
        .ok_or_else(|| ControlApiError::Runtime(RuntimeError::pipeline_unavailable()))?;
    Ok(Json(serde_json::json!({
        "learned": true,
        "template": template,
        "status": status,
    })))
}

async fn clear_crosshair(
    State(state): State<ControlState>,
) -> Result<Json<novasight_runtime::CrosshairSnapshot>, ControlApiError> {
    state
        .runtime
        .clear_crosshair()
        .map(Json)
        .map_err(Into::into)
}

async fn crosshair_template_preview(
    State(state): State<ControlState>,
) -> Result<Response, ControlApiError> {
    let payload = state.runtime.crosshair_template_preview()?.ok_or_else(|| {
        ControlApiError::Runtime(RuntimeError::invalid_pipeline_state(
            "crosshair template unavailable",
        ))
    })?;
    Ok((
        [
            (axum::http::header::CONTENT_TYPE, "image/png"),
            (axum::http::header::CACHE_CONTROL, "no-store"),
        ],
        payload,
    )
        .into_response())
}

async fn executors(State(state): State<ControlState>) -> Json<serde_json::Value> {
    let snapshot = state.runtime.snapshot();
    let status = runtime_state(&state, &snapshot).await;
    Json(serde_json::to_value(status.executor).expect("runtime executor DTO must serialize"))
}

async fn connect_device(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    if !state.hardware_output_enabled {
        return Err(ControlApiError::HardwareOutputDisabled);
    }
    ensure_config_effective(&state).await?;
    Ok(Json(state.runtime.connect_device().await?))
}

async fn disconnect_device(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    if !state.hardware_output_enabled {
        return Err(ControlApiError::HardwareOutputDisabled);
    }
    Ok(Json(state.runtime.disconnect_device().await?))
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

#[derive(Debug, Serialize)]
struct DeviceButtonsResponse {
    available: bool,
    left: bool,
    right: bool,
    managed_by_runtime: bool,
}

async fn device_buttons(State(state): State<ControlState>) -> Json<DeviceButtonsResponse> {
    let metrics = state.runtime.snapshot().pipeline_metrics;
    Json(DeviceButtonsResponse {
        available: metrics.buttons_available,
        left: metrics.button_left,
        right: metrics.button_right,
        managed_by_runtime: true,
    })
}

async fn diagnostic_device_move(
    State(state): State<ControlState>,
    Json(request): Json<DiagnosticMoveRequest>,
) -> Result<Json<DiagnosticMoveResponse>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
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

async fn prepare_config_for_start(state: &ControlState) -> Result<(), ControlApiError> {
    if let Some(service) = &state.config {
        service.prepare_runtime_start(&state.runtime).await?;
    }
    Ok(())
}

fn hot_pipeline_config_update(update: &ConfigFieldUpdate) -> bool {
    update.section == "pipeline"
}

fn runtime_reconfigurable_config_update(update: &ConfigFieldUpdate) -> bool {
    matches!(
        update.section.as_str(),
        "replay" | "consumers" | "limits" | "crosshair" | "capture" | "inference" | "hardware"
    )
}

async fn apply_pipeline_config_update(
    state: &ControlState,
    service: &ConfigService,
    update: ConfigFieldUpdate,
) -> Result<ConfigUpdate, ControlApiError> {
    service
        .update_pipeline(&state.runtime, update)
        .await
        .map_err(ControlApiError::Config)
}

async fn apply_config_field_update(
    state: &ControlState,
    service: &ConfigService,
    update: ConfigFieldUpdate,
) -> Result<ConfigUpdate, ControlApiError> {
    let hot_output_gate = update.section == "control" && update.key == "output_enabled";
    let hot_trigger_mode = update.section == "control" && update.key == "trigger_mode";
    let hot_recoil = update.section == "control" && update.key == "recoil";
    let hot_pipeline = hot_pipeline_config_update(&update);
    if hot_output_gate && update.value.as_bool() == Some(true) {
        ensure_hardware_control_license(state).await?;
    }
    let result = if hot_output_gate {
        service.update_output_gate(&state.runtime, update).await?
    } else if hot_trigger_mode {
        service.update_trigger_mode(&state.runtime, update).await?
    } else if hot_recoil {
        service.update_recoil(&state.runtime, update).await?
    } else if hot_pipeline {
        apply_pipeline_config_update(state, service, update).await?
    } else if runtime_reconfigurable_config_update(&update) {
        service.update_runtime_field(&state.runtime, update).await?
    } else {
        service.update_field(update).await?
    };
    Ok(result)
}

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum ConfigCommandRequest {
    SetOutputGate {
        enabled: bool,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
    SetTriggerMode {
        mode: String,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
}

impl ConfigCommandRequest {
    async fn apply(
        self,
        state: &ControlState,
        service: &ConfigService,
    ) -> Result<ConfigUpdate, ControlApiError> {
        match self {
            Self::SetOutputGate {
                enabled,
                expected_revision,
            } => {
                if enabled {
                    ensure_hardware_control_license(state).await?;
                }
                Ok(service
                    .update_output_gate(
                        &state.runtime,
                        ConfigFieldUpdate {
                            section: "control".to_owned(),
                            key: "output_enabled".to_owned(),
                            value: serde_json::Value::Bool(enabled),
                            expected_revision,
                        },
                    )
                    .await?)
            }
            Self::SetTriggerMode {
                mode,
                expected_revision,
            } => Ok(service
                .update_trigger_mode(
                    &state.runtime,
                    ConfigFieldUpdate {
                        section: "control".to_owned(),
                        key: "trigger_mode".to_owned(),
                        value: serde_json::Value::String(mode),
                        expected_revision,
                    },
                )
                .await?),
        }
    }
}

async fn ensure_runtime_license(state: &ControlState) -> Result<(), ControlApiError> {
    let Some(repository) = state.license.as_ref() else {
        return Ok(());
    };
    let status =
        run_license_operation(repository.clone(), |repository| repository.status()).await?;
    status
        .authorize_runtime(hardware_output_requested(state).await)
        .map_err(control_error_from_license_denial)
}

async fn ensure_hardware_control_license(state: &ControlState) -> Result<(), ControlApiError> {
    if !state.hardware_output_enabled {
        return Ok(());
    }
    let Some(repository) = state.license.as_ref() else {
        return Ok(());
    };
    let status =
        run_license_operation(repository.clone(), |repository| repository.status()).await?;
    status
        .authorize_feature("hardware_control")
        .map_err(control_error_from_license_denial)
}

fn control_error_from_license_denial(denial: LicenseDenial) -> ControlApiError {
    match denial {
        LicenseDenial::Required => ControlApiError::LicenseRequired,
        LicenseDenial::FeatureRequired(feature) => ControlApiError::LicenseFeatureRequired(feature),
    }
}

async fn hardware_output_requested(state: &ControlState) -> bool {
    if !state.hardware_output_enabled {
        return false;
    }
    if state.runtime.snapshot().pipeline_metrics.output_gate_open {
        return true;
    }
    match &state.config {
        Some(service) => service.snapshot().await.control.output_enabled,
        None => false,
    }
}

async fn config(State(state): State<ControlState>) -> Result<Json<AppConfig>, ControlApiError> {
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    Ok(Json(service.snapshot().await))
}

async fn config_schema(
    State(state): State<ControlState>,
) -> Result<Json<ConfigSchemaResponse>, ControlApiError> {
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let config = service.snapshot().await;
    Ok(Json(ConfigSchemaResponse::new(&config)))
}

async fn capture_state(State(state): State<ControlState>) -> Json<serde_json::Value> {
    let snapshot = state.runtime.snapshot();
    let status = runtime_state(&state, &snapshot).await;
    Json(serde_json::to_value(status.capture).expect("runtime capture DTO must serialize"))
}

#[derive(Debug, Default, Deserialize)]
struct CaptureCapabilitiesQuery {
    device: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCapabilitiesRequest {
    device: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSelectRequest {
    device: String,
    #[serde(default)]
    preference: CaptureSelectionPreference,
    pixel_format: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
}

async fn capture_capabilities(
    State(state): State<ControlState>,
    Query(query): Query<CaptureCapabilitiesQuery>,
) -> Result<Json<CaptureCapabilities>, ControlApiError> {
    probe_capture_capabilities(state, query.device)
        .await
        .map(Json)
}

async fn post_capture_capabilities(
    State(state): State<ControlState>,
    Json(request): Json<CaptureCapabilitiesRequest>,
) -> Result<Json<CaptureCapabilities>, ControlApiError> {
    probe_capture_capabilities(state, Some(request.device))
        .await
        .map(Json)
}

async fn select_capture(
    State(state): State<ControlState>,
    Json(request): Json<CaptureSelectRequest>,
) -> Result<Json<serde_json::Value>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    if state.runtime.snapshot().pipeline.state != PipelineState::Stopped {
        return Err(ControlApiError::CaptureSelectionRequiresStoppedRuntime);
    }
    let capabilities = probe_capture_capabilities(state.clone(), Some(request.device)).await?;
    let manual = match (
        request.pixel_format.as_deref(),
        request.width,
        request.height,
        request.fps,
    ) {
        (Some(pixel_format), Some(width), Some(height), Some(fps)) => {
            Some((pixel_format, width, height, fps))
        }
        _ => None,
    };
    let selected = select_capture_profile_for_formats(
        &capabilities,
        request.preference,
        manual,
        &["MJPG", "NV12", "YUYV"],
    )
    .map_err(ControlApiError::CaptureSelection)?;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    // Studio deliberately confirms the capture profile before starting the
    // mainline. Load any saved epoch-scoped settings first so this preflight
    // step cannot dead-end on a harmless desired/effective revision split.
    service.prepare_runtime_start(&state.runtime).await?;
    let persisted = service.apply_capture_profile(&selected).await?;
    service
        .install_persisted_runtime_config(&state.runtime, persisted.config)
        .await?;

    let snapshot = state.runtime.snapshot();
    let status = runtime_state(&state, &snapshot).await;
    Ok(Json(
        serde_json::to_value(status.capture).expect("runtime capture DTO must serialize"),
    ))
}

async fn stop_capture(
    State(state): State<ControlState>,
) -> Result<Json<serde_json::Value>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let snapshot = state.runtime.stop().await?;
    let status = runtime_state(&state, &snapshot).await;
    Ok(Json(
        serde_json::to_value(status.capture).expect("runtime capture DTO must serialize"),
    ))
}

async fn probe_capture_capabilities(
    state: ControlState,
    requested_device: Option<String>,
) -> Result<CaptureCapabilities, ControlApiError> {
    let device = match requested_device {
        Some(device) => device,
        None => match &state.config {
            Some(config) => config
                .snapshot()
                .await
                .capture
                .map(|capture| capture.device.to_string_lossy().into_owned())
                .unwrap_or_else(|| "/dev/video0".to_owned()),
            None => "/dev/video0".to_owned(),
        },
    };
    let probe = state
        .capture_probe
        .ok_or(ControlApiError::CaptureProbeUnavailable)?;
    tokio::task::spawn_blocking(move || probe.probe(&device))
        .await
        .map_err(ControlApiError::CaptureProbeTask)?
        .map_err(ControlApiError::CaptureProbe)
}

async fn update_config(
    State(state): State<ControlState>,
    Json(update): Json<ConfigFieldUpdate>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let result = apply_config_field_update(&state, service, update).await?;
    Ok(Json(result))
}

async fn apply_config_command(
    State(state): State<ControlState>,
    Json(command): Json<ConfigCommandRequest>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let result = command.apply(&state, service).await?;
    Ok(Json(result))
}

async fn update_config_document(
    State(state): State<ControlState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let is_field_update = payload.get("section").is_some()
        || payload.get("key").is_some()
        || payload.get("value").is_some();
    let update = if is_field_update {
        let field_update: ConfigFieldUpdate =
            serde_json::from_value(payload).map_err(ControlApiError::InvalidFieldUpdate)?;
        apply_config_field_update(&state, service, field_update).await?
    } else {
        service.replace_runtime(&state.runtime, payload).await?
    };
    Ok(Json(update))
}

async fn status(State(state): State<ControlState>) -> Json<RuntimeSnapshot> {
    Json(state.runtime.snapshot())
}

async fn start(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    prepare_config_for_start(&state).await?;
    Ok(Json(state.runtime.start().await?))
}

async fn stop(State(state): State<ControlState>) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    Ok(Json(state.runtime.stop().await?))
}

async fn restart(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    state.runtime.stop().await?;
    prepare_config_for_start(&state).await?;
    Ok(Json(state.runtime.start().await?))
}

async fn emergency_stop(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(emergency_stop_with_lifecycle_barrier(&state).await?))
}

async fn shutdown_daemon(
    State(state): State<ControlState>,
    trusted_local_control: Option<Extension<TrustedLocalControl>>,
) -> Result<Json<serde_json::Value>, ControlApiError> {
    if trusted_local_control.is_none() {
        return Err(ControlApiError::LocalControlRequired);
    }
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    state.runtime.shutdown_daemon().await?;
    Ok(Json(serde_json::json!({
        "shutdown": true,
        "message": "NovaSight daemon shutdown requested",
    })))
}

async fn events(websocket: WebSocketUpgrade, State(state): State<ControlState>) -> Response {
    websocket.on_upgrade(move |socket| stream_events(socket, state.runtime, state.shutdown))
}

#[derive(Debug, Default, Deserialize)]
struct RuntimeStatusQuery {
    topic: Option<String>,
}

#[derive(Serialize)]
struct RuntimeStatusHeartbeat {
    kind: &'static str,
    daemon_instance_id: String,
    snapshot_sequence: u64,
}

async fn runtime_events(
    websocket: WebSocketUpgrade,
    Query(query): Query<RuntimeStatusQuery>,
    State(state): State<ControlState>,
    trusted_local_control: Option<Extension<TrustedLocalControl>>,
    headers: HeaderMap,
) -> Response {
    if let Some(license) = state.license.as_ref() {
        match run_license_operation(license.clone(), |repository| repository.status()).await {
            Ok(status) if status.configured && status.valid => {
                if trusted_local_control.is_none()
                    && state.license_session.verify(&headers, &status).is_err()
                {
                    return websocket.on_upgrade(close_unlicensed_websocket);
                }
            }
            Ok(_) => {
                return websocket.on_upgrade(close_unlicensed_websocket);
            }
            Err(error) => return error.into_response(),
        }
    }
    websocket.on_upgrade(move |socket| stream_runtime_events(socket, state, query))
}

async fn close_unlicensed_websocket(mut socket: WebSocket) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: 4401,
            reason: "license required".into(),
        })))
        .await;
}

async fn stream_runtime_events(socket: WebSocket, state: ControlState, query: RuntimeStatusQuery) {
    let mut snapshots = state.runtime.subscribe();
    let mut shutdown = state.shutdown.clone();
    let topic = normalize_runtime_topic(query.topic.as_deref());
    let mut heartbeat = tokio::time::interval(STATUS_HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    heartbeat.tick().await;
    let (mut outbound, mut inbound) = socket.split();
    let mut snapshot_changed = true;
    let mut first_frame = true;
    let mut last_semantic = None;
    loop {
        if snapshot_changed {
            let current = snapshots.borrow_and_update().clone();
            let status = runtime_state_for_topic(&state, current.as_ref(), topic).await;
            let semantic = (
                status.semantic.phase,
                status.semantic.perception_phase,
                status.semantic.epoch,
            );
            let send_full = first_frame || last_semantic.is_some_and(|last| last != semantic);
            let Ok(payload) = serialize_runtime_status_frame(topic, send_full, &status) else {
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
                SendOutcome::Sent => {
                    first_frame = false;
                    last_semantic = Some(semantic);
                    snapshot_changed = false;
                }
                SendOutcome::Closed | SendOutcome::Shutdown => return,
            }
        }

        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
                snapshot_changed = true;
            }
            _ = heartbeat.tick() => {
                let snapshot_sequence = snapshots.borrow().sequence;
                let Ok(payload) = serde_json::to_string(&RuntimeStatusHeartbeat {
                    kind: "runtime_heartbeat",
                    daemon_instance_id: state.daemon_instance_id.to_string(),
                    snapshot_sequence,
                }) else {
                    return;
                };
                match send_or_shutdown(
                    &mut outbound,
                    &mut inbound,
                    Message::Text(payload.into()),
                    &mut shutdown,
                ).await {
                    SendOutcome::Sent => {}
                    SendOutcome::Closed | SendOutcome::Shutdown => return,
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

fn normalize_runtime_topic(topic: Option<&str>) -> &'static str {
    let topic = topic.unwrap_or_default().trim().to_ascii_lowercase();
    match topic.as_str() {
        "summary" => "summary",
        "capture" => "capture",
        "infer" => "infer",
        "control" => "control",
        "latency" => "latency",
        _ => "full",
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
    UnsupportedDiagnostic(String),
    License(LicenseError),
    LicenseTask(tokio::task::JoinError),
    LicenseSession(String),
    LicenseUnavailable,
    LicenseRequired,
    LicenseFeatureRequired(&'static str),
    ModelCatalog(ModelCatalogError),
    ModelCatalogTask(tokio::task::JoinError),
    ModelCatalogUnavailable,
    ModelActivation(ModelActivationError),
    ModelIngress(ModelIngressError),
    ModelRecommendationInvalid(String),
    CaptureProbe(CaptureProbeError),
    CaptureProbeTask(tokio::task::JoinError),
    CaptureProbeUnavailable,
    CaptureSelection(CaptureSelectionError),
    CaptureSelectionRequiresStoppedRuntime,
    LocalControlRequired,
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
        match error {
            ConfigServiceError::Runtime(error) => Self::Runtime(error),
            error => Self::Config(error),
        }
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
                    RuntimeErrorKind::InvalidPipelineState
                    | RuntimeErrorKind::ModelUnavailable
                    | RuntimeErrorKind::OutputGateClosed
                    | RuntimeErrorKind::DeviceUncommissioned => StatusCode::CONFLICT,
                    RuntimeErrorKind::SupervisorUnavailable
                    | RuntimeErrorKind::SupervisorClosed
                    | RuntimeErrorKind::SupervisorReplyLost
                    | RuntimeErrorKind::SupervisorBusy
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
                    "CONFIG_RUNTIME_DIVERGED" => StatusCode::CONFLICT,
                    "CONFIG_BUSY"
                    | "CONFIG_OUTPUT_GATE_DISABLED_NOT_PERSISTED"
                    | "CONFIG_RUNTIME_APPLY_FAILED" => {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                    "CONFIG_PARSE_ERROR"
                    | "CONFIG_VALIDATION_ERROR"
                    | "CONFIG_UNSUPPORTED_CONFIG_KEY"
                    | "CONFIG_INVALID_FIELD_TARGET"
                    | "CONFIG_FIELD_VALUE_INVALID"
                    | "CONFIG_HOT_UPDATE_TRANSACTION_REQUIRED"
                    | "CONFIG_REPLACEMENT_INVALID"
                    | "CONFIG_REPLACEMENT_REVISION_REQUIRED"
                    | "CAPTURE_NOT_CONFIGURED"
                    | "CAPTURE_PROFILE_INVALID" => StatusCode::BAD_REQUEST,
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
                "physical kmNet is unavailable because this control surface has no production hardware output adapter"
                    .to_owned(),
            ),
            Self::DeviceNotConfigured => (
                StatusCode::SERVICE_UNAVAILABLE,
                "DEVICE_NOT_CONFIGURED",
                "no production pointer device is configured".to_owned(),
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
            Self::LicenseSession(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "LICENSE_SESSION_FAILED",
                error,
            ),
            Self::LicenseUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "LICENSE_SERVICE_UNAVAILABLE",
                "license service is not configured".to_owned(),
            ),
            Self::LicenseRequired => (
                StatusCode::UNAUTHORIZED,
                "LICENSE_REQUIRED",
                "a valid license is required".to_owned(),
            ),
            Self::LicenseFeatureRequired(feature) => (
                StatusCode::FORBIDDEN,
                "LICENSE_FEATURE_REQUIRED",
                format!("license feature {feature} is required"),
            ),
            Self::ModelCatalog(error) => match error {
                ModelCatalogError::ProjectNotFound(_)
                | ModelCatalogError::VersionNotFound(_)
                | ModelCatalogError::ArtifactNotFound(_)
                | ModelCatalogError::DeploymentNotFound(_)
                | ModelCatalogError::CatalogModelNotFound(_) => (
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
                ModelCatalogError::InvalidCatalogModelPath(_) => (
                    StatusCode::BAD_REQUEST,
                    "MODEL_CATALOG_PATH_INVALID",
                    error.to_string(),
                ),
                ModelCatalogError::InvalidArtifactTags(_) => (
                    StatusCode::BAD_REQUEST,
                    "MODEL_ARTIFACT_TAGS_INVALID",
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
                    | ModelCatalogError::IngressManifestInvalid { .. }
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
                        RuntimeErrorKind::InvalidPipelineState
                        | RuntimeErrorKind::ModelUnavailable
                        | RuntimeErrorKind::OutputGateClosed
                        | RuntimeErrorKind::DeviceUncommissioned => StatusCode::CONFLICT,
                        RuntimeErrorKind::SupervisorUnavailable
                        | RuntimeErrorKind::SupervisorClosed
                        | RuntimeErrorKind::SupervisorReplyLost
                        | RuntimeErrorKind::SupervisorBusy
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
            Self::ModelIngress(error) => match error {
                ModelIngressError::InvalidRequest(message) => {
                    (StatusCode::BAD_REQUEST, "MODEL_PROFILE_INVALID", message)
                }
                ModelIngressError::ProfileNotFound(_)
                | ModelIngressError::Catalog(ModelCatalogError::ProjectNotFound(_))
                | ModelIngressError::Catalog(ModelCatalogError::VersionNotFound(_))
                | ModelIngressError::Catalog(ModelCatalogError::ArtifactNotFound(_)) => (
                    StatusCode::NOT_FOUND,
                    "MODEL_PROFILE_NOT_FOUND",
                    error.to_string(),
                ),
                ModelIngressError::Catalog(ModelCatalogError::ArtifactCurrentlyActive(_)) => (
                    StatusCode::CONFLICT,
                    "MODEL_ARTIFACT_ACTIVE",
                    error.to_string(),
                ),
                ModelIngressError::LatestFrameUnavailable => (
                    StatusCode::CONFLICT,
                    "MODEL_LATEST_FRAME_UNAVAILABLE",
                    error.to_string(),
                ),
                ModelIngressError::Cancelled => (
                    StatusCode::CONFLICT,
                    "MODEL_INGRESS_CANCELLED",
                    error.to_string(),
                ),
                ModelIngressError::Unavailable => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "MODEL_INGRESS_UNAVAILABLE",
                    error.to_string(),
                ),
                ModelIngressError::Runtime(error) => {
                    let status = match error.kind {
                        RuntimeErrorKind::InvalidPipelineState
                        | RuntimeErrorKind::ModelUnavailable
                        | RuntimeErrorKind::OutputGateClosed
                        | RuntimeErrorKind::DeviceUncommissioned => StatusCode::CONFLICT,
                        RuntimeErrorKind::SupervisorUnavailable
                        | RuntimeErrorKind::SupervisorClosed
                        | RuntimeErrorKind::SupervisorReplyLost
                        | RuntimeErrorKind::SupervisorBusy
                        | RuntimeErrorKind::PipelineUnavailable
                        | RuntimeErrorKind::DeviceUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                        RuntimeErrorKind::InvalidDeviceCommand => StatusCode::BAD_REQUEST,
                        RuntimeErrorKind::PipelineRejected
                        | RuntimeErrorKind::RuntimeEpochExhausted
                        | RuntimeErrorKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
                    };
                    (status, error.kind.code(), error.message)
                }
                ModelIngressError::Catalog(error) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "MODEL_INGRESS_CATALOG_REJECTED",
                    error.to_string(),
                ),
                ModelIngressError::ReadProfile(_)
                | ModelIngressError::OutputLimitExceeded(_)
                | ModelIngressError::DecodeResponse(_)
                | ModelIngressError::Protocol(_)
                | ModelIngressError::Failed(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "MODEL_INGRESS_INTERNAL",
                    error.to_string(),
                ),
            },
            Self::ModelRecommendationInvalid(message) => (
                StatusCode::CONFLICT,
                "MODEL_RECOMMENDATION_INVALID",
                message,
            ),
            Self::CaptureProbe(error) => {
                let status = match error.code() {
                    "CAPTURE_DEVICE_INVALID" => StatusCode::BAD_REQUEST,
                    "CAPTURE_PROBE_UNSUPPORTED_PLATFORM" => StatusCode::SERVICE_UNAVAILABLE,
                    "CAPTURE_CAPABILITY_IOCTL_FAILED" => StatusCode::BAD_GATEWAY,
                    "CAPTURE_CAPABILITY_LIMIT_EXCEEDED" => StatusCode::BAD_GATEWAY,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, error.code(), error.to_string())
            }
            Self::CaptureProbeTask(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "CAPTURE_PROBE_TASK_FAILED",
                format!("capture capability task failed: {error}"),
            ),
            Self::CaptureProbeUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "CAPTURE_PROBE_UNAVAILABLE",
                "no platform capture capability probe is attached".to_owned(),
            ),
            Self::CaptureSelection(error) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "CAPTURE_PROFILE_UNSUPPORTED",
                error.to_string(),
            ),
            Self::CaptureSelectionRequiresStoppedRuntime => (
                StatusCode::CONFLICT,
                "CAPTURE_SELECTION_REQUIRES_STOPPED_RUNTIME",
                "stop the runtime before changing its concrete capture profile".to_owned(),
            ),
            Self::LocalControlRequired => (
                StatusCode::NOT_FOUND,
                "LOCAL_CONTROL_REQUIRED",
                "daemon shutdown is available only through the local control socket".to_owned(),
            ),
        };
        let body = ControlErrorBody {
            code,
            detail: message.clone(),
            message,
        };
        if status.is_server_error() {
            tracing::error!(
                http_status = status.as_u16(),
                error_code = code,
                error = %body.message,
                "api request failed"
            );
        } else {
            tracing::warn!(
                http_status = status.as_u16(),
                error_code = code,
                error = %body.message,
                "api request rejected"
            );
        }
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::runtime_snapshot_confirms_safe;
    use novasight_runtime::{PipelineState, RuntimeSnapshot};

    #[test]
    fn safe_snapshot_requires_stopped_disconnected_and_output_blocked() {
        let safe = RuntimeSnapshot::default();
        assert!(runtime_snapshot_confirms_safe(&safe));

        let mut running = safe.clone();
        running.pipeline.state = PipelineState::Running;
        assert!(!runtime_snapshot_confirms_safe(&running));

        let mut connected = safe.clone();
        connected.pipeline_metrics.device_connected = true;
        assert!(!runtime_snapshot_confirms_safe(&connected));

        let mut emitting = safe;
        emitting.pipeline_metrics.control.emit_allowed = true;
        assert!(!runtime_snapshot_confirms_safe(&emitting));
    }
}
