#![forbid(unsafe_code)]

mod auth;
mod proxy;

use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Component, Path, PathBuf};
use std::process::{self, ExitCode};
use std::sync::Arc;

use auth::{AuthError, AuthService, SESSION_SECONDS, SessionStatus};
use axum::body::Body;
use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use novasight_config::{YamlConfigRepository, studio_endpoint_contract};
use proxy::DaemonProxy;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

const DEFAULT_CONFIG_PATH: &str = "data/novasight.yaml";
const DEFAULT_WEB_ROOT: &str = "web";
const DEFAULT_READY_FILE: &str = "run/ready.json";
const ACCESS_CODE_ENV: &str = "NOVASIGHT_WEB_ACCESS_CODE";
const SECURE_COOKIE_ENV: &str = "NOVASIGHT_WEB_SECURE_COOKIE";
const ALLOWED_HOSTS_ENV: &str = "NOVASIGHT_WEB_ALLOWED_HOSTS";
const WEB_ROOT_ENV: &str = "NOVASIGHT_WEB_ROOT";

#[derive(Parser, Debug)]
#[command(about = "NovaSight authenticated Web/API gateway")]
struct Args {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Bind the private API endpoint used behind Vite HMR and do not serve assets.
    #[arg(long)]
    frontend_dev: bool,

    /// Validate authentication, routing, and static assets without binding a port.
    #[arg(long)]
    check: bool,
}

#[derive(Clone)]
struct AppState {
    auth: AuthService,
    daemon: DaemonProxy,
    web_root: Option<Arc<PathBuf>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoginRequest {
    access_code: String,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    code: &'static str,
    message: String,
}

#[derive(Serialize)]
struct ReadyDocument {
    schema_version: u32,
    pid: u32,
    service: &'static str,
    address: String,
    url: String,
    daemon_socket: String,
    authentication: &'static str,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "novasight=info".into()),
        )
        .try_init();
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), WebServerError> {
    let config_path = args
        .config
        .as_deref()
        .unwrap_or(Path::new(DEFAULT_CONFIG_PATH));
    let config =
        YamlConfigRepository::load(config_path).map_err(|source| WebServerError::ConfigLoad {
            path: config_path.to_owned(),
            detail: source.to_string(),
        })?;
    let access_code = std::env::var(ACCESS_CODE_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or(WebServerError::AccessCodeMissing)?;
    let secure_cookie = boolean_environment(SECURE_COOKIE_ENV)?;
    let allowed_hosts = std::env::var(ALLOWED_HOSTS_ENV)
        .unwrap_or_default()
        .split(',')
        .filter(|host| !host.trim().is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let auth = AuthService::new(&access_code, secure_cookie)
        .map_err(WebServerError::AccessCodeInvalid)?
        .with_allowed_hosts(&allowed_hosts)
        .map_err(WebServerError::AllowedHostsInvalid)?;
    let web_root = if args.frontend_dev {
        None
    } else {
        let root = std::env::var_os(WEB_ROOT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WEB_ROOT));
        if !root.join("index.html").is_file() {
            return Err(WebServerError::WebRootInvalid(root));
        }
        Some(Arc::new(root))
    };
    let endpoint = if args.frontend_dev {
        studio_endpoint_contract().frontend_development_api.clone()
    } else {
        novasight_config::NetworkEndpoint {
            host: config.server.host.clone(),
            port: config.server.port,
        }
    };
    let state = AppState {
        auth,
        daemon: DaemonProxy::new(config.server.control_socket.clone()),
        web_root,
    };

    if args.check {
        println!(
            "PASS service=novasight-web authentication=access_code session=server_side authorization=operator csrf=required daemon_transport=unix static_assets={} bind={}:{}",
            if state.web_root.is_some() {
                "ready"
            } else {
                "vite"
            },
            endpoint.host,
            endpoint.port
        );
        return Ok(());
    }

    let listener = TcpListener::bind((endpoint.host.as_str(), endpoint.port))
        .await
        .map_err(|source| WebServerError::Bind {
            host: endpoint.host.clone(),
            port: endpoint.port,
            source,
        })?;
    let address = listener
        .local_addr()
        .map_err(WebServerError::LocalAddress)?;
    let ready_file = PathBuf::from(DEFAULT_READY_FILE);
    let _ready_guard = ReadyFileGuard(ready_file.clone());
    write_ready_file(&ready_file, address, state.daemon.socket())?;
    eprintln!(
        "novasight-web ready address={address} daemon_socket={}",
        state.daemon.socket().display()
    );
    axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(WebServerError::Serve)
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route(
            "/api/auth/session",
            get(get_session).post(login).delete(logout),
        )
        .fallback(dispatch)
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}

async fn health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = state.auth.validate_request_site(&headers) {
        return auth_error_response(error);
    }
    let ok = state.daemon.healthy().await;
    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    no_store((status, Json(HealthResponse { ok })).into_response())
}

async fn get_session(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = state.auth.validate_request_site(&headers) {
        return auth_error_response(error);
    }
    no_store(Json(state.auth.status(&headers)).into_response())
}

async fn login(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(request): Json<LoginRequest>,
) -> Response {
    if let Err(error) = state.auth.validate_request_site(&headers) {
        return auth_error_response(error);
    }
    match state.auth.login(&request.access_code, peer.ip()) {
        Ok(issued) => {
            let mut response = Json(issued.status).into_response();
            response
                .headers_mut()
                .insert(header::SET_COOKIE, issued.cookie);
            no_store(response)
        }
        Err(error) => auth_error_response(error),
    }
}

async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(error) = state.auth.validate_request_site(&headers) {
        return auth_error_response(error);
    }
    match state.auth.logout(&headers) {
        Ok(cookie) => {
            let mut response = Json(SessionStatus {
                authenticated: false,
                principal: None,
                role: None,
                permissions: Vec::new(),
                csrf_token: None,
                expires_at: None,
                session_lifetime_seconds: SESSION_SECONDS,
            })
            .into_response();
            response.headers_mut().insert(header::SET_COOKIE, cookie);
            no_store(response)
        }
        Err(error) => auth_error_response(error),
    }
}

async fn dispatch(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request<Body>,
) -> Response {
    if let Err(error) = state.auth.validate_request_site(request.headers()) {
        return auth_error_response(error);
    }
    let path = request.uri().path();
    if is_daemon_path(path) {
        if request.method() == Method::OPTIONS {
            return StatusCode::NO_CONTENT.into_response();
        }
        let session = match state
            .auth
            .authorize_proxy(request.method(), path, request.headers())
        {
            Ok(session) => session,
            Err(error) => return auth_error_response(error),
        };
        if is_license_activation(request.method(), path)
            && let Err(error) = state.auth.authorize_license_activation(peer.ip())
        {
            return auth_error_response(error);
        }
        let websocket_session = path.starts_with("/ws/").then_some(session);
        return match state.daemon.forward(request, websocket_session).await {
            Ok(response) => no_store(response),
            Err(error) => {
                tracing::warn!(%error, "daemon IPC request failed");
                api_error(
                    StatusCode::BAD_GATEWAY,
                    "DAEMON_UNAVAILABLE",
                    "NovaSight daemon 本地控制接口不可用",
                )
            }
        };
    }
    serve_web_ui(&state, request.uri()).await
}

fn is_license_activation(method: &Method, path: &str) -> bool {
    (*method == Method::POST && path == "/api/license/activate")
        || (*method == Method::PUT && path == "/api/license")
}

fn is_daemon_path(path: &str) -> bool {
    matches!(path, "/api" | "/ws") || path.starts_with("/api/") || path.starts_with("/ws/")
}

async fn serve_web_ui(state: &AppState, uri: &Uri) -> Response {
    let Some(root) = &state.web_root else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(asset_path) = web_asset_path(root, uri.path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let content_type = content_type_for_path(&asset_path);
    match tokio::fs::read(&asset_path).await {
        Ok(bytes) => web_asset_response(content_type, bytes),
        Err(_) if asset_path != root.join("index.html") => {
            match tokio::fs::read(root.join("index.html")).await {
                Ok(bytes) => web_asset_response("text/html; charset=utf-8", bytes),
                Err(_) => StatusCode::NOT_FOUND.into_response(),
            }
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

fn web_asset_path(root: &Path, request_path: &str) -> Option<PathBuf> {
    let trimmed = request_path.trim_start_matches('/');
    let relative = if trimmed.is_empty() {
        Path::new("index.html")
    } else {
        Path::new(trimmed)
    };
    let mut sanitized = PathBuf::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => sanitized.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    let candidate = root.join(sanitized);
    Some(if candidate.is_file() {
        candidate
    } else {
        root.join("index.html")
    })
}

fn content_type_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("ico") => "image/x-icon",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("wasm") => "application/wasm",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

fn web_asset_response(content_type: &'static str, bytes: Vec<u8>) -> Response {
    let mut response = ([(header::CONTENT_TYPE, content_type)], bytes).into_response();
    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ws: wss:",
        ),
    );
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
    if content_type.starts_with("text/html") {
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

fn auth_error_response(error: AuthError) -> Response {
    match error {
        AuthError::Rejected => api_error(
            StatusCode::UNAUTHORIZED,
            "ACCESS_CODE_REJECTED",
            "接入码无效",
        ),
        AuthError::RateLimited => {
            let mut response = api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "AUTH_RATE_LIMITED",
                "失败次数过多，请稍后再试",
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("60"));
            response
        }
        AuthError::LicenseActivationRateLimited => {
            let mut response = api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "LICENSE_ACTIVATION_RATE_LIMITED",
                "授权验证次数过多，请稍后再试",
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("60"));
            response
        }
        AuthError::Required => api_error(
            StatusCode::UNAUTHORIZED,
            "AUTHENTICATION_REQUIRED",
            "需要有效的 Web 会话",
        ),
        AuthError::AuthorizationDenied => api_error(
            StatusCode::FORBIDDEN,
            "AUTHORIZATION_DENIED",
            "当前 Web 身份无权执行本地专用操作",
        ),
        AuthError::CsrfRejected => api_error(
            StatusCode::FORBIDDEN,
            "CSRF_REJECTED",
            "CSRF 会话令牌缺失或已失效",
        ),
        AuthError::HostRejected(message) => api_error(
            StatusCode::BAD_REQUEST,
            "HOST_REJECTED",
            &format!("请求 Host 未获允许：{message}"),
        ),
        AuthError::OriginRejected => api_error(
            StatusCode::BAD_REQUEST,
            "ORIGIN_REJECTED",
            "请求 Origin 与 Host 不一致",
        ),
        AuthError::Clock | AuthError::Cookie(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "AUTH_INTERNAL_ERROR",
            "Web 会话服务暂时不可用",
        ),
    }
}

fn api_error(status: StatusCode, code: &'static str, message: &str) -> Response {
    no_store(
        (
            status,
            Json(ErrorResponse {
                code,
                message: message.to_owned(),
            }),
        )
            .into_response(),
    )
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn boolean_environment(name: &'static str) -> Result<bool, WebServerError> {
    match std::env::var(name) {
        Ok(value) if matches!(value.as_str(), "1" | "true" | "yes") => Ok(true),
        Ok(value) if matches!(value.as_str(), "0" | "false" | "no" | "") => Ok(false),
        Ok(value) => Err(WebServerError::BooleanEnvironmentInvalid { name, value }),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err(WebServerError::BooleanEnvironmentInvalid {
            name,
            value: "non-Unicode value".to_owned(),
        }),
    }
}

fn write_ready_file(
    path: &Path,
    address: SocketAddr,
    daemon_socket: &Path,
) -> Result<(), WebServerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| WebServerError::ReadyFile {
            path: parent.to_owned(),
            source,
        })?;
    }
    let public_address = if address.ip().is_unspecified() {
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), address.port())
    } else {
        address
    };
    let payload = serde_json::to_vec_pretty(&ReadyDocument {
        schema_version: 2,
        pid: process::id(),
        service: "novasight-web",
        address: address.to_string(),
        url: format!("http://{public_address}/"),
        daemon_socket: daemon_socket.display().to_string(),
        authentication: "server-session-csrf",
    })
    .map_err(WebServerError::ReadySerialize)?;
    let temporary = path.with_extension(format!("tmp.{}", process::id()));
    fs::write(&temporary, payload).map_err(|source| WebServerError::ReadyFile {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| WebServerError::ReadyFile {
        path: path.to_owned(),
        source,
    })
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

struct ReadyFileGuard(PathBuf);

impl Drop for ReadyFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Debug, Error)]
enum WebServerError {
    #[error("failed to load config {}: {detail}", path.display())]
    ConfigLoad { path: PathBuf, detail: String },
    #[error("{ACCESS_CODE_ENV} is required")]
    AccessCodeMissing,
    #[error("configured Web access code is invalid: {0}")]
    AccessCodeInvalid(String),
    #[error("configured Web allowed hosts are invalid: {0}")]
    AllowedHostsInvalid(String),
    #[error("{name} must be one of 1, true, yes, 0, false, or no; got {value:?}")]
    BooleanEnvironmentInvalid { name: &'static str, value: String },
    #[error("Web root is missing index.html: {}", .0.display())]
    WebRootInvalid(PathBuf),
    #[error("failed to bind Web/API server at {host}:{port}: {source}")]
    Bind {
        host: String,
        port: u16,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to inspect bound Web/API address: {0}")]
    LocalAddress(std::io::Error),
    #[error("Web/API server failed: {0}")]
    Serve(std::io::Error),
    #[error("failed to serialize Web/API ready document: {0}")]
    ReadySerialize(serde_json::Error),
    #[error("failed to publish Web/API ready state at {}: {source}", path.display())]
    ReadyFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl WebServerError {
    const fn code(&self) -> &'static str {
        match self {
            Self::ConfigLoad { .. } => "WEB_CONFIG_LOAD_FAILED",
            Self::AccessCodeMissing => "WEB_ACCESS_CODE_MISSING",
            Self::AccessCodeInvalid(_) => "WEB_ACCESS_CODE_INVALID",
            Self::AllowedHostsInvalid(_) => "WEB_ALLOWED_HOSTS_INVALID",
            Self::BooleanEnvironmentInvalid { .. } => "WEB_BOOLEAN_ENV_INVALID",
            Self::WebRootInvalid(_) => "WEB_ROOT_INVALID",
            Self::Bind { .. } => "WEB_BIND_FAILED",
            Self::LocalAddress(_) => "WEB_LOCAL_ADDRESS_FAILED",
            Self::Serve(_) => "WEB_SERVER_FAILED",
            Self::ReadySerialize(_) => "WEB_READY_SERIALIZE_FAILED",
            Self::ReadyFile { .. } => "WEB_READY_FILE_FAILED",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use axum::Router;
    use axum::body::Body;
    use axum::extract::{ConnectInfo, WebSocketUpgrade};
    use axum::http::{HeaderValue, Request, StatusCode, header};
    use axum::routing::get;
    use futures_util::StreamExt;
    use http_body_util::BodyExt;
    use tokio::net::{TcpListener, UnixListener};
    use tokio::time::timeout;
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tower::ServiceExt;

    use super::{AppState, AuthService, DaemonProxy, auth::CSRF_HEADER, router};

    const ACCESS_CODE: &str = "8f0ed8de1c54cb8c34a4369c02fd0c6883b2f3e2b7b3321c4b4cb08c1e0d7f50";

    #[tokio::test]
    async fn login_issues_server_session_and_csrf_contract() {
        let app = router(AppState {
            auth: AuthService::new(ACCESS_CODE, false).unwrap(),
            daemon: DaemonProxy::new("/tmp/novasight-web-test-missing.sock"),
            web_root: None,
        });
        let request = Request::builder()
            .method("POST")
            .uri("/api/auth/session")
            .header(header::HOST, "192.168.10.20:7351")
            .header(header::CONTENT_TYPE, "application/json")
            .extension(ConnectInfo(SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 10, 21)),
                50000,
            )))
            .body(Body::from(format!(r#"{{"access_code":"{ACCESS_CODE}"}}"#)))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let cookie = response
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(cookie.starts_with("novasight_web_session="));
        assert!(cookie.contains("; HttpOnly"));
        assert!(cookie.contains("; SameSite=Strict"));
        assert!(cookie.contains("; Path=/"));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["authenticated"], true);
        assert_eq!(body["role"], "operator");
        assert_eq!(body["permissions"].as_array().map(Vec::len), Some(6));
        assert!(
            body["csrf_token"]
                .as_str()
                .is_some_and(|value| value.len() >= 32)
        );
    }

    #[tokio::test]
    async fn authenticated_mutation_crosses_the_csrf_gate() {
        let app = router(AppState {
            auth: AuthService::new(ACCESS_CODE, false).unwrap(),
            daemon: DaemonProxy::new("/tmp/novasight-web-test-missing.sock"),
            web_root: None,
        });
        let login = Request::builder()
            .method("POST")
            .uri("/api/auth/session")
            .header(header::HOST, "192.168.10.20:7351")
            .header(header::CONTENT_TYPE, "application/json")
            .extension(ConnectInfo(SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 10, 21)),
                50000,
            )))
            .body(Body::from(format!(r#"{{"access_code":"{ACCESS_CODE}"}}"#)))
            .unwrap();
        let response = app.clone().oneshot(login).await.unwrap();
        let cookie = response
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_owned();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let csrf = body["csrf_token"].as_str().unwrap();

        let mutation = Request::builder()
            .method("POST")
            .uri("/api/license/activate")
            .header(header::HOST, "192.168.10.20:7351")
            .header(header::COOKIE, cookie)
            .header(CSRF_HEADER, csrf)
            .header(header::CONTENT_TYPE, "application/json")
            .extension(ConnectInfo(SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 10, 21)),
                50001,
            )))
            .body(Body::from(r#"{"key":"not-forwarded"}"#))
            .unwrap();
        let response = app.oneshot(mutation).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["code"], "DAEMON_UNAVAILABLE");
    }

    #[tokio::test]
    async fn logout_closes_an_established_status_websocket() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let socket =
            std::path::PathBuf::from(format!("/tmp/ns-ws-{}-{unique}.sock", std::process::id()));
        let upstream_listener = UnixListener::bind(&socket).unwrap();
        let upstream = Router::new().route(
            "/ws/status",
            get(|upgrade: WebSocketUpgrade| async move {
                upgrade
                    .on_upgrade(|mut socket| async move { while socket.recv().await.is_some() {} })
            }),
        );
        let upstream_task = tokio::spawn(async move {
            axum::serve(upstream_listener, upstream).await.unwrap();
        });

        let app = router(AppState {
            auth: AuthService::new(ACCESS_CODE, false).unwrap(),
            daemon: DaemonProxy::new(&socket),
            web_root: None,
        });
        let login = Request::builder()
            .method("POST")
            .uri("/api/auth/session")
            .header(header::HOST, "127.0.0.1:7351")
            .header(header::CONTENT_TYPE, "application/json")
            .extension(ConnectInfo(SocketAddr::new(
                IpAddr::V4(Ipv4Addr::LOCALHOST),
                50000,
            )))
            .body(Body::from(format!(r#"{{"access_code":"{ACCESS_CODE}"}}"#)))
            .unwrap();
        let response = app.clone().oneshot(login).await.unwrap();
        let cookie = response
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_owned();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let csrf = body["csrf_token"].as_str().unwrap().to_owned();

        let web_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let web_address = web_listener.local_addr().unwrap();
        let web_app = app.clone();
        let web_task = tokio::spawn(async move {
            axum::serve(
                web_listener,
                web_app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });

        let mut websocket_request = format!("ws://{web_address}/ws/status")
            .into_client_request()
            .unwrap();
        websocket_request
            .headers_mut()
            .insert(header::COOKIE, HeaderValue::from_str(&cookie).unwrap());
        let (mut websocket, _) = connect_async(websocket_request).await.unwrap();

        let logout = Request::builder()
            .method("DELETE")
            .uri("/api/auth/session")
            .header(header::HOST, "127.0.0.1:7351")
            .header(header::COOKIE, &cookie)
            .header(CSRF_HEADER, csrf)
            .extension(ConnectInfo(SocketAddr::new(
                IpAddr::V4(Ipv4Addr::LOCALHOST),
                50001,
            )))
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(logout).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let message = timeout(Duration::from_secs(5), websocket.next())
            .await
            .expect("logout must close an established WebSocket immediately")
            .expect("WebSocket must return a close frame")
            .expect("WebSocket close frame must be valid");
        match message {
            Message::Close(Some(frame)) => assert_eq!(u16::from(frame.code), 4403),
            other => panic!("expected session-revoked close frame, got {other:?}"),
        }

        web_task.abort();
        upstream_task.abort();
        let _ = std::fs::remove_file(socket);
    }
}
