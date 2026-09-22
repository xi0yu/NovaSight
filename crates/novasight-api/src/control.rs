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
    http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header::SET_COOKIE},
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
const PHYSICAL_OUTPUT_ACK_HEADER: &str = "x-novasight-physical-output-ack";
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
        .layer(middleware::from_fn_with_state(state, log_http_request))
}

async fn log_http_request(
    State(state): State<ControlState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(str::to_owned)
        .unwrap_or_else(|| {
            format!(
                "{}-{:016x}",
                state.daemon_instance_id,
                NEXT_HTTP_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
            )
        });
    let entry_point = if request.extensions().get::<TrustedLocalControl>().is_some() {
        "local_socket"
    } else {
        "http"
    };
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let span = tracing::info_span!(
        "http_request",
        service = "novasightd",
        version = env!("CARGO_PKG_VERSION"),
        request_id = %request_id,
        entry_point,
        method = %method,
        path = %path,
    );
    async move {
        let started = Instant::now();
        let mut response = next.run(request).await;
        let status = response.status();
        let latency_ms = started.elapsed().as_millis() as u64;
        tracing::info!(
            event = "http_request_completed",
            http_status = status.as_u16(),
            latency_ms,
            "api request completed"
        );
        response.headers_mut().insert(
            "x-request-id",
            HeaderValue::from_str(&request_id).expect("request ID is validated ASCII"),
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
                    // Risk-reducing action: never require a product license
                    // to disconnect. The Web boundary still enforces auth/CSRF.
                    | "/api/executors/kmnet/disconnect"
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

async fn runtime_start(
    State(state): State<ControlState>,
    headers: HeaderMap,
) -> Result<StatusCode, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    require_physical_output_ack(&state, &headers).await?;
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
    headers: HeaderMap,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    if !state.hardware_output_enabled {
        return Err(ControlApiError::HardwareOutputDisabled);
    }
    ensure_config_effective(&state).await?;
    require_physical_output_ack(&state, &headers).await?;
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
    headers: &HeaderMap,
    update: ConfigFieldUpdate,
) -> Result<ConfigUpdate, ControlApiError> {
    let hot_output_gate = update.section == "control" && update.key == "output_enabled";
    let hot_trigger_mode = update.section == "control" && update.key == "trigger_mode";
    let hot_recoil = update.section == "control" && update.key == "recoil";
    let hot_pipeline = hot_pipeline_config_update(&update);
    if hot_output_gate && update.value.as_bool() == Some(true) {
        ensure_hardware_control_license(state).await?;
        if state.hardware_output_enabled {
            require_physical_output_ack_header(headers)?;
        }
    }
    if (hot_trigger_mode && update.value.as_str() == Some("always")) || hot_recoil || hot_pipeline {
        require_physical_output_ack(state, headers).await?;
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
        require_epoch_reload_ack(state, headers).await?;
        service.update_runtime_field(&state.runtime, update).await?
    } else {
        service.update_field(update).await?
    };
    Ok(result)
}

async fn require_epoch_reload_ack(
    state: &ControlState,
    headers: &HeaderMap,
) -> Result<(), ControlApiError> {
    if matches!(
        state.runtime.snapshot().pipeline.state,
        PipelineState::Running | PipelineState::Standby
    ) {
        require_physical_output_ack(state, headers).await?;
    }
    Ok(())
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
        headers: &HeaderMap,
    ) -> Result<ConfigUpdate, ControlApiError> {
        match self {
            Self::SetOutputGate {
                enabled,
                expected_revision,
            } => {
                if enabled {
                    ensure_hardware_control_license(state).await?;
                    if state.hardware_output_enabled {
                        require_physical_output_ack_header(headers)?;
                    }
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
            } => {
                if mode == "always" {
                    require_physical_output_ack(state, headers).await?;
                }
                Ok(service
                    .update_trigger_mode(
                        &state.runtime,
                        ConfigFieldUpdate {
                            section: "control".to_owned(),
                            key: "trigger_mode".to_owned(),
                            value: serde_json::Value::String(mode),
                            expected_revision,
                        },
                    )
                    .await?)
            }
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
        .authorize_feature("runtime")
        .map_err(control_error_from_license_denial)?;
    if status.authorize_feature("hardware_control").is_err()
        && hardware_output_requested(state).await
    {
        // All start/restart callers hold lifecycle_lock. Close and persist the
        // old output preference before starting computation; never grant output
        // implicitly, and fail closed if the configuration transaction fails.
        let service = state
            .config
            .as_ref()
            .ok_or(ControlApiError::ConfigUnavailable)?;
        service
            .update_output_gate(
                &state.runtime,
                ConfigFieldUpdate {
                    section: "control".to_owned(),
                    key: "output_enabled".to_owned(),
                    value: serde_json::Value::Bool(false),
                    expected_revision: None,
                },
            )
            .await?;
        tracing::warn!(
            code = "HARDWARE_OUTPUT_DISABLED_FOR_RUNTIME",
            "当前授权不含硬件控制，已关闭并保存物理输出开关；允许启动识别主链"
        );
    }
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

async fn require_physical_output_ack(
    state: &ControlState,
    headers: &HeaderMap,
) -> Result<(), ControlApiError> {
    if state.hardware_output_enabled
        && (state.config.is_none() || hardware_output_requested(state).await)
    {
        require_physical_output_ack_header(headers)?;
    }
    Ok(())
}

fn require_physical_output_ack_header(headers: &HeaderMap) -> Result<(), ControlApiError> {
    if headers
        .get(PHYSICAL_OUTPUT_ACK_HEADER)
        .and_then(|value| value.to_str().ok())
        != Some("confirmed")
    {
        return Err(ControlApiError::PhysicalOutputAcknowledgementRequired);
    }
    Ok(())
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
    headers: HeaderMap,
    Json(update): Json<ConfigFieldUpdate>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let result = apply_config_field_update(&state, service, &headers, update).await?;
    Ok(Json(result))
}

async fn apply_config_command(
    State(state): State<ControlState>,
    headers: HeaderMap,
    Json(command): Json<ConfigCommandRequest>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    let service = state
        .config
        .as_ref()
        .ok_or(ControlApiError::ConfigUnavailable)?;
    let result = command.apply(&state, service, &headers).await?;
    Ok(Json(result))
}

async fn update_config_document(
    State(state): State<ControlState>,
    headers: HeaderMap,
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
        apply_config_field_update(&state, service, &headers, field_update).await?
    } else {
        require_epoch_reload_ack(&state, &headers).await?;
        service.replace_runtime(&state.runtime, payload).await?
    };
    Ok(Json(update))
}

async fn status(State(state): State<ControlState>) -> Json<RuntimeSnapshot> {
    Json(state.runtime.snapshot())
}

async fn start(
    State(state): State<ControlState>,
    headers: HeaderMap,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    require_physical_output_ack(&state, &headers).await?;
    prepare_config_for_start(&state).await?;
    Ok(Json(state.runtime.start().await?))
}

async fn stop(State(state): State<ControlState>) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    Ok(Json(state.runtime.stop().await?))
}

async fn restart(
    State(state): State<ControlState>,
    headers: HeaderMap,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    let _lifecycle_guard = state.lifecycle_lock.lock().await;
    ensure_runtime_license(&state).await?;
    require_physical_output_ack(&state, &headers).await?;
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
    PhysicalOutputAcknowledgementRequired,
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
            Self::PhysicalOutputAcknowledgementRequired => (
                StatusCode::PRECONDITION_REQUIRED,
                "PHYSICAL_OUTPUT_ACK_REQUIRED",
                "物理输出已开启；本次启动或连接必须由操作者明确确认。取消操作或先关闭物理输出。".to_owned(),
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
                ModelCatalogError::InvalidCatalogModelPath(_)
                | ModelCatalogError::InvalidCatalogDirectoryPath(_) => (
                    StatusCode::BAD_REQUEST,
                    "MODEL_CATALOG_PATH_INVALID",
                    error.to_string(),
                ),
                ModelCatalogError::CatalogDirectoryExists(_) => (
                    StatusCode::CONFLICT,
                    "MODEL_DIRECTORY_EXISTS",
                    error.to_string(),
                ),
                ModelCatalogError::CatalogModelDestinationExists(_)
                | ModelCatalogError::CatalogModelAmbiguousSource(_) => (
                    StatusCode::CONFLICT,
                    "MODEL_FILE_EXISTS",
                    error.to_string(),
                ),
                ModelCatalogError::CatalogModelRegistered(_)
                | ModelCatalogError::CatalogModelHasManifest(_) => (
                    StatusCode::CONFLICT,
                    "MODEL_FILE_IN_USE",
                    error.to_string(),
                ),
                ModelCatalogError::CatalogDirectoryParentInvalid(_) => (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "MODEL_DIRECTORY_PARENT_INVALID",
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

    #[tokio::test]
    async fn catalog_folder_http_create_readback_and_conflict() {
        use super::*;
        use novasight_runtime::RuntimeSupervisor;
        use tower::ServiceExt;

        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "novasight-catalog-http-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let catalog =
            SqliteModelCatalog::open_with_model_root(root.join("registry.db"), root.join("models"))
                .unwrap();
        let (_supervisor, runtime) = RuntimeSupervisor::spawn_recording();
        let app = build_control_router_with_control_plane(
            runtime,
            None,
            None,
            Some(catalog),
            false,
            None,
        );
        let create = || {
            Request::builder()
                .method("POST")
                .uri("/api/models/catalog/folders")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"relative_path":"Arena"}"#))
                .unwrap()
        };
        let response = app.clone().oneshot(create()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let readback = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/models/catalog?force=true")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(readback.status(), StatusCode::OK);
        let body = axum::body::to_bytes(readback.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["root"]["children"][0]["relative_path"], "Arena");
        assert_eq!(
            app.oneshot(create()).await.unwrap().status(),
            StatusCode::CONFLICT
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn catalog_engine_move_http_reads_back_without_replacing_a_file() {
        use super::*;
        use novasight_runtime::RuntimeSupervisor;
        use tower::ServiceExt;

        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "novasight-move-http-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let catalog =
            SqliteModelCatalog::open_with_model_root(root.join("registry.db"), root.join("models"))
                .unwrap();
        catalog.create_catalog_directory("Arena").unwrap();
        std::fs::write(root.join("models/model.engine"), b"engine").unwrap();
        let (_supervisor, runtime) = RuntimeSupervisor::spawn_recording();
        let app = build_control_router_with_control_plane(
            runtime,
            None,
            None,
            Some(catalog),
            false,
            None,
        );
        let move_request = || {
            Request::builder()
                .method("POST")
                .uri("/api/models/catalog/move")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"from_path":"model.engine","to_path":"Arena/renamed.engine"}"#,
                ))
                .unwrap()
        };
        assert_eq!(
            app.clone().oneshot(move_request()).await.unwrap().status(),
            StatusCode::OK
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/models/catalog")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["root"]["children"][0]["children"][0]["relative_path"],
            "Arena/renamed.engine"
        );
        assert_eq!(
            app.oneshot(move_request()).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn request_id_is_shared_by_success_and_rejection_and_exposed_to_studio() {
        use super::*;
        use novasight_runtime::RuntimeSupervisor;
        use tower::ServiceExt;

        let (_supervisor, runtime) = RuntimeSupervisor::spawn_recording();
        let app = build_control_router(runtime.clone());
        let request_id = "0123456789abcdef0123456789abcdef";
        let success = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header("origin", "http://localhost:7351")
                    .header("x-request-id", request_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(success.status().is_success());
        assert_eq!(success.headers()["x-request-id"], request_id);
        assert_eq!(
            success.headers()["access-control-expose-headers"],
            "x-request-id"
        );

        let failure = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/config")
                    .header("x-request-id", request_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!failure.status().is_success());
        assert_eq!(failure.headers()["x-request-id"], request_id);

        let rejected_id = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header("x-request-id", "untrusted-value")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let generated = rejected_id.headers()["x-request-id"].to_str().unwrap();
        assert_eq!(generated.len(), 49);
        assert!(!generated.contains("untrusted-value"));
        runtime.shutdown_daemon().await.unwrap();
    }

    #[tokio::test]
    async fn physical_output_start_and_reconnect_require_explicit_request_acknowledgement() {
        use super::*;
        use novasight_runtime::{RuntimeDependencies, RuntimeSupervisor};
        use novasight_store::config::YamlConfigRepository;
        use tower::ServiceExt;

        // RecordingPointerDevice is commissioned but never moves hardware.
        let (_supervisor, runtime) =
            RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
        let app =
            build_control_router_with_control_plane(runtime.clone(), None, None, None, true, None);
        for endpoint in ["/api/runtime/start", "/api/v1/runtime/start"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(endpoint)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::PRECONDITION_REQUIRED,
                "{endpoint}"
            );
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            assert!(String::from_utf8_lossy(&body).contains("PHYSICAL_OUTPUT_ACK_REQUIRED"));
            assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);
        }

        let acknowledged = |endpoint| {
            Request::builder()
                .method("POST")
                .uri(endpoint)
                .header(PHYSICAL_OUTPUT_ACK_HEADER, "confirmed")
                .body(Body::empty())
                .unwrap()
        };
        assert_eq!(
            app.clone()
                .oneshot(acknowledged("/api/runtime/start"))
                .await
                .unwrap()
                .status(),
            StatusCode::NO_CONTENT
        );
        let before_restart = runtime.snapshot().pipeline.state;
        assert!(matches!(
            before_restart,
            PipelineState::Running | PipelineState::Standby
        ));
        for (endpoint, body) in [
            (
                "/api/models/projects/1/publish",
                r#"{"artifact_id":1,"parser_preset":"auto"}"#,
            ),
            ("/api/models/projects/1/rollback", "{}"),
            ("/api/models/artifacts/1/probe", r#"{"input_mode":"fixed"}"#),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(endpoint)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::PRECONDITION_REQUIRED,
                "{endpoint}"
            );
            assert_eq!(runtime.snapshot().pipeline.state, before_restart);
        }
        let restart = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/runtime/restart")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(restart.status(), StatusCode::PRECONDITION_REQUIRED);
        assert_eq!(runtime.snapshot().pipeline.state, before_restart);
        runtime.disconnect_device().await.unwrap();
        assert!(!runtime.snapshot().pipeline_metrics.device_connected);
        let connect = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/executors/kmnet/connect")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(connect.status(), StatusCode::PRECONDITION_REQUIRED);
        assert!(!runtime.snapshot().pipeline_metrics.device_connected);
        assert!(
            app.clone()
                .oneshot(acknowledged("/api/executors/kmnet/connect"))
                .await
                .unwrap()
                .status()
                .is_success()
        );
        assert!(runtime.snapshot().pipeline_metrics.device_connected);

        // Safety-decreasing actions remain available without acknowledgement.
        assert!(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/executors/kmnet/disconnect")
                        .body(Body::empty())
                        .unwrap()
                )
                .await
                .unwrap()
                .status()
                .is_success()
        );
        assert!(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runtime/stop")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status()
            .is_success()
        );
        runtime.shutdown_daemon().await.unwrap();

        // Persisted output intent also requires an acknowledgement when the
        // current process has not opened its live output gate yet.
        let root =
            std::env::temp_dir().join(format!("ns-output-ack-{}", generate_daemon_instance_id()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("config.yaml");
        std::fs::write(&path, "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n").unwrap();
        let service = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
        let (_supervisor, runtime) = RuntimeSupervisor::spawn_recording();
        let app = build_control_router_with_capabilities(runtime.clone(), service, true, None);
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runtime/start")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PRECONDITION_REQUIRED);
        assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);
        assert_eq!(
            app.clone()
                .oneshot(acknowledged("/api/runtime/start"))
                .await
                .unwrap()
                .status(),
            StatusCode::NO_CONTENT
        );
        assert!(matches!(
            runtime.snapshot().pipeline.state,
            PipelineState::Running | PipelineState::Standby
        ));
        let revision_before_reload = YamlConfigRepository::load(&path).unwrap().revision;
        for (endpoint, method, body) in [
            (
                "/api/config",
                "POST",
                r#"{"section":"capture","key":"roi_left","value":0}"#,
            ),
            (
                "/api/v1/config",
                "PATCH",
                r#"{"section":"capture","key":"roi_left","value":0}"#,
            ),
            ("/api/config", "POST", r#"{"revision":0}"#),
            (
                "/api/config",
                "POST",
                r#"{"section":"pipeline","key":"target_lost_grace_ms","value":99}"#,
            ),
            (
                "/api/v1/config/commands",
                "POST",
                r#"{"command":"set_trigger_mode","mode":"always"}"#,
            ),
            (
                "/api/v1/config/commands",
                "POST",
                r#"{"command":"set_output_gate","enabled":true}"#,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(endpoint)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::PRECONDITION_REQUIRED,
                "{endpoint}"
            );
            assert_eq!(
                YamlConfigRepository::load(&path).unwrap().revision,
                revision_before_reload
            );
        }
        let close_output = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/config/commands")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"command":"set_output_gate","enabled":false}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(close_output.status(), StatusCode::OK);
        assert!(
            !YamlConfigRepository::load(&path)
                .unwrap()
                .control
                .output_enabled
        );
        runtime.stop().await.unwrap();
        runtime.shutdown_daemon().await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn runtime_only_start_closes_stale_output_and_keeps_hardware_gated() {
        use super::*;
        use novasight_runtime::RuntimeSupervisor;
        use novasight_store::{config::YamlConfigRepository, license::LicensePolicy};
        use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};
        use tower::ServiceExt;

        let root = std::env::temp_dir().join(format!(
            "ns-license-start-{}",
            generate_daemon_instance_id()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("config.yaml");
        std::fs::write(&path, "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n").unwrap();
        let config = YamlConfigRepository::load(&path).unwrap();
        let service = ConfigService::new(&path, config);
        let (_supervisor, runtime) = RuntimeSupervisor::spawn_recording();
        let temporary_code = "temporary-license-code-for-runtime-test";
        // Temporary authorization now includes hardware; retain a real signed,
        // runtime-only credential to exercise the restricted-license boundary.
        let private = rsa::RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048).unwrap();
        let public = private
            .to_public_key()
            .to_public_key_pem(LineEnding::LF)
            .unwrap();
        let pem = private.to_pkcs8_pem(LineEnding::LF).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let credential = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256),
            &serde_json::json!({"iss":"novasight-license", "aud":"novasightd",
                "sub":"runtime-only-test", "jti":"runtime-only-test", "tier":"test",
                "features":["runtime", "config_read", "config_write"], "iat":now, "exp":now + 3600}),
            &jsonwebtoken::EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap(),
        ).unwrap();
        let key = credential.as_str();
        let license = FileLicenseRepository::new(
            root.join("license.json"),
            LicensePolicy::new(Some(public))
                .with_temporary_access_code(temporary_code)
                .unwrap(),
        );
        license.activate(key).unwrap();
        let app = build_control_router_with_control_plane(
            runtime.clone(),
            Some(service.clone()),
            Some(license.clone()),
            None,
            true,
            None,
        );
        for endpoint in [
            "/api/runtime/start",
            "/api/v1/runtime/start",
            "/api/v1/runtime/restart",
        ] {
            service
                .update_output_gate(
                    &runtime,
                    ConfigFieldUpdate {
                        section: "control".into(),
                        key: "output_enabled".into(),
                        value: serde_json::Value::Bool(true),
                        expected_revision: None,
                    },
                )
                .await
                .unwrap();
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(endpoint)
                        .extension(TrustedLocalControl)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(
                response.status().is_success(),
                "{endpoint}: {}",
                response.status()
            );
            assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
            assert!(!service.snapshot().await.control.output_enabled);
            assert!(
                !YamlConfigRepository::load(&path)
                    .unwrap()
                    .control
                    .output_enabled
            );
            runtime.stop().await.unwrap();
        }
        // A previously connected device must remain stoppable without the
        // hardware feature, and even after the product license is cleared.
        // This supervisor owns a recording device, never a physical kmNet.
        for valid_license in [true, false] {
            runtime.start().await.unwrap();
            let connected = runtime.connect_device().await.unwrap();
            assert!(connected.pipeline_metrics.device_connected);
            if !valid_license {
                license.clear().unwrap();
            }
            for _ in 0..2 {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/api/executors/kmnet/disconnect")
                            .extension(TrustedLocalControl)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let status = response.status();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                assert!(
                    status.is_success(),
                    "disconnect: {status}: {}",
                    String::from_utf8_lossy(&body)
                );
                let snapshot = runtime.snapshot();
                assert!(!snapshot.pipeline_metrics.device_connected);
                assert!(!snapshot.pipeline_metrics.output_gate_open);
                assert_eq!(snapshot.pipeline.state, connected.pipeline.state);
            }
            license.activate(key).unwrap();
            runtime.stop().await.unwrap();
        }
        for endpoint in [
            "/api/executors/kmnet/connect",
            "/api/executors/kmnet/diagnostic-move",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(endpoint)
                        .extension(TrustedLocalControl)
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"dx":1,"dy":1}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/api/v1/config")
                    .extension(TrustedLocalControl)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"section":"control","key":"output_enabled","value":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(!service.snapshot().await.control.output_enabled);

        // The same activation entry must allow temporary authorization to open
        // the output gate. This test only uses the recording device above.
        license.clear().unwrap();
        license.activate(temporary_code).unwrap();
        assert!(!runtime.snapshot().pipeline_metrics.device_connected);
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
        runtime.start().await.unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/executors/kmnet/connect")
                    .extension(TrustedLocalControl)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            response.status().is_success(),
            "temporary connect: {}",
            response.status()
        );
        let unacknowledged_output = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/config/commands")
                    .extension(TrustedLocalControl)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"command":"set_output_gate","enabled":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            unacknowledged_output.status(),
            StatusCode::PRECONDITION_REQUIRED
        );
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
        for enabled in [true, false] {
            let mut request = Request::builder()
                .method("POST")
                .uri("/api/v1/config/commands")
                .extension(TrustedLocalControl)
                .header("content-type", "application/json");
            if enabled {
                request = request.header(PHYSICAL_OUTPUT_ACK_HEADER, "confirmed");
            }
            let response = app
                .clone()
                .oneshot(
                    request
                        .body(Body::from(
                            serde_json::json!({"command":"set_output_gate", "enabled":enabled})
                                .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(
                response.status().is_success(),
                "temporary output: {}",
                response.status()
            );
            assert_eq!(
                runtime.snapshot().pipeline_metrics.output_gate_open,
                enabled
            );
            assert_eq!(service.snapshot().await.control.output_enabled, enabled);
        }
        runtime.stop().await.unwrap();
        license.activate(key).unwrap();

        // A failed safe-output persistence must not allow the start to continue.
        service
            .update_output_gate(
                &runtime,
                ConfigFieldUpdate {
                    section: "control".into(),
                    key: "output_enabled".into(),
                    value: serde_json::Value::Bool(true),
                    expected_revision: None,
                },
            )
            .await
            .unwrap();
        std::fs::rename(&path, root.join("config.saved.yaml")).unwrap();
        std::fs::create_dir(&path).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runtime/start")
                    .extension(TrustedLocalControl)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!response.status().is_success());
        assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);
        runtime.shutdown_daemon().await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

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
