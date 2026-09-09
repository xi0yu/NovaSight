use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, HeaderValue, Method, Uri, header};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::{RngCore, rngs::OsRng};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::watch;

pub const SESSION_COOKIE: &str = "novasight_web_session";
pub const CSRF_HEADER: &str = "x-novasight-csrf";
pub const SESSION_SECONDS: u64 = 60 * 60;
const OPERATOR_PERMISSIONS: [&str; 6] = [
    "studio:read",
    "runtime:operate",
    "configuration:write",
    "models:manage",
    "license:manage",
    "hardware:operate",
];
const MIN_ACCESS_CODE_BYTES: usize = 32;
const MAX_ACCESS_CODE_BYTES: usize = 1024;
const MAX_SESSIONS: usize = 64;
const LOGIN_WINDOW_SECONDS: u64 = 60;
const LOGIN_FAILURE_LIMIT: u32 = 5;
const LICENSE_ACTIVATION_WINDOW_SECONDS: u64 = 60;
const LICENSE_ACTIVATION_LIMIT: u32 = 6;

#[derive(Clone)]
pub struct AuthService {
    access_digest: Arc<[u8; 32]>,
    secure_cookie: bool,
    allowed_hosts: Arc<[String]>,
    state: Arc<Mutex<AuthState>>,
}

#[derive(Default)]
struct AuthState {
    sessions: HashMap<[u8; 32], SessionRecord>,
    login_attempts: HashMap<IpAddr, LoginAttempt>,
    license_activation_attempts: HashMap<IpAddr, RequestWindow>,
}

struct SessionRecord {
    csrf_token: String,
    expires_at: u64,
    revocation: watch::Sender<bool>,
}

#[derive(Clone, Copy)]
struct LoginAttempt {
    window_started_at: u64,
    failures: u32,
    blocked_until: u64,
}

#[derive(Clone, Copy)]
struct RequestWindow {
    window_started_at: u64,
    attempts: u32,
}

pub struct IssuedSession {
    pub cookie: HeaderValue,
    pub status: SessionStatus,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionStatus {
    pub authenticated: bool,
    pub principal: Option<&'static str>,
    pub role: Option<&'static str>,
    pub permissions: Vec<&'static str>,
    pub csrf_token: Option<String>,
    pub expires_at: Option<u64>,
    pub session_lifetime_seconds: u64,
}

pub struct AuthenticatedSession {
    pub csrf_token: String,
    pub expires_at: u64,
    pub revocation: watch::Receiver<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("access code is invalid")]
    Rejected,
    #[error("too many failed access attempts; retry later")]
    RateLimited,
    #[error("too many license activation attempts; retry later")]
    LicenseActivationRateLimited,
    #[error("authentication is required")]
    Required,
    #[error("this operation is restricted to the local daemon client")]
    AuthorizationDenied,
    #[error("CSRF token is missing or invalid")]
    CsrfRejected,
    #[error("request host is not allowed: {0}")]
    HostRejected(String),
    #[error("request origin does not match its host")]
    OriginRejected,
    #[error("system clock is before the Unix epoch")]
    Clock,
    #[error("failed to construct a session cookie: {0}")]
    Cookie(String),
}

impl AuthService {
    pub fn new(access_code: &str, secure_cookie: bool) -> Result<Self, String> {
        if access_code != access_code.trim() {
            return Err("web access code must not have edge whitespace".to_owned());
        }
        if access_code.len() < MIN_ACCESS_CODE_BYTES {
            return Err(format!(
                "web access code must contain at least {MIN_ACCESS_CODE_BYTES} bytes"
            ));
        }
        if access_code.len() > MAX_ACCESS_CODE_BYTES {
            return Err(format!(
                "web access code must not exceed {MAX_ACCESS_CODE_BYTES} bytes"
            ));
        }
        Ok(Self {
            access_digest: Arc::new(Sha256::digest(access_code.as_bytes()).into()),
            secure_cookie,
            allowed_hosts: Arc::from([]),
            state: Arc::new(Mutex::new(AuthState::default())),
        })
    }

    pub fn with_allowed_hosts(mut self, hosts: &[String]) -> Result<Self, String> {
        let mut allowed_hosts = Vec::with_capacity(hosts.len());
        for host in hosts {
            let host = host.trim().to_ascii_lowercase();
            if host.is_empty()
                || host.contains('/')
                || host.contains(':')
                || !host
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
            {
                return Err(format!("invalid web allowed host {host:?}"));
            }
            if !allowed_hosts.contains(&host) {
                allowed_hosts.push(host);
            }
        }
        self.allowed_hosts = Arc::from(allowed_hosts);
        Ok(self)
    }

    pub fn validate_request_site(&self, headers: &HeaderMap) -> Result<(), AuthError> {
        let authority = headers
            .get(header::HOST)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| AuthError::HostRejected("missing or invalid Host".to_owned()))?
            .parse::<axum::http::uri::Authority>()
            .map_err(|_| AuthError::HostRejected("invalid Host".to_owned()))?;
        let request_host = normalized_host(authority.host());
        if request_host != "localhost"
            && request_host.parse::<IpAddr>().is_err()
            && !self.allowed_hosts.iter().any(|host| host == &request_host)
        {
            return Err(AuthError::HostRejected(request_host));
        }

        let Some(origin) = headers.get(header::ORIGIN) else {
            return Ok(());
        };
        let origin = origin.to_str().map_err(|_| AuthError::OriginRejected)?;
        let origin = origin
            .parse::<Uri>()
            .map_err(|_| AuthError::OriginRejected)?;
        if !matches!(origin.scheme_str(), Some("http" | "https")) {
            return Err(AuthError::OriginRejected);
        }
        let origin_host = origin
            .authority()
            .map(|authority| normalized_host(authority.host()))
            .ok_or(AuthError::OriginRejected)?;
        if origin_host != request_host
            && !(is_loopback_host(&origin_host) && is_loopback_host(&request_host))
        {
            return Err(AuthError::OriginRejected);
        }
        Ok(())
    }

    pub fn login(&self, access_code: &str, peer: IpAddr) -> Result<IssuedSession, AuthError> {
        let now = unix_seconds()?;
        let mut state = self.state.lock().expect("auth state mutex poisoned");
        prune(&mut state, now);
        if state
            .login_attempts
            .get(&peer)
            .is_some_and(|attempt| attempt.blocked_until > now)
        {
            return Err(AuthError::RateLimited);
        }
        let candidate: [u8; 32] = Sha256::digest(access_code.as_bytes()).into();
        if access_code.len() > MAX_ACCESS_CODE_BYTES
            || !constant_time_equal(self.access_digest.as_ref(), &candidate)
        {
            record_failure(&mut state, peer, now);
            return Err(AuthError::Rejected);
        }
        state.login_attempts.remove(&peer);
        while state.sessions.len() >= MAX_SESSIONS {
            let Some(oldest) = state
                .sessions
                .iter()
                .min_by_key(|(_, session)| session.expires_at)
                .map(|(digest, _)| *digest)
            else {
                break;
            };
            if let Some(session) = state.sessions.remove(&oldest) {
                session.revocation.send_replace(true);
            }
        }

        let session_token = random_token();
        let csrf_token = random_token();
        let expires_at = now + SESSION_SECONDS;
        let (revocation, _) = watch::channel(false);
        state.sessions.insert(
            token_digest(&session_token),
            SessionRecord {
                csrf_token: csrf_token.clone(),
                expires_at,
                revocation,
            },
        );
        let secure = if self.secure_cookie { "; Secure" } else { "" };
        let cookie = HeaderValue::from_str(&format!(
            "{SESSION_COOKIE}={session_token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}{secure}"
        ))
        .map_err(|error| AuthError::Cookie(error.to_string()))?;
        Ok(IssuedSession {
            cookie,
            status: authenticated_status(csrf_token, expires_at),
        })
    }

    pub fn session(&self, headers: &HeaderMap) -> Result<AuthenticatedSession, AuthError> {
        let token = cookie_value(headers, SESSION_COOKIE).ok_or(AuthError::Required)?;
        let now = unix_seconds()?;
        let mut state = self.state.lock().expect("auth state mutex poisoned");
        prune(&mut state, now);
        let session = state
            .sessions
            .get(&token_digest(&token))
            .ok_or(AuthError::Required)?;
        Ok(AuthenticatedSession {
            csrf_token: session.csrf_token.clone(),
            expires_at: session.expires_at,
            revocation: session.revocation.subscribe(),
        })
    }

    pub fn status(&self, headers: &HeaderMap) -> SessionStatus {
        let Ok(token) = cookie_value(headers, SESSION_COOKIE).ok_or(()) else {
            return anonymous_status();
        };
        let Ok(now) = unix_seconds() else {
            return anonymous_status();
        };
        let mut state = self.state.lock().expect("auth state mutex poisoned");
        prune(&mut state, now);
        state
            .sessions
            .get(&token_digest(&token))
            .map(|session| authenticated_status(session.csrf_token.clone(), session.expires_at))
            .unwrap_or_else(anonymous_status)
    }

    pub fn authorize_proxy(
        &self,
        method: &Method,
        path: &str,
        headers: &HeaderMap,
    ) -> Result<AuthenticatedSession, AuthError> {
        let session = self.session(headers)?;
        if path == "/api/v1/daemon/shutdown" {
            return Err(AuthError::AuthorizationDenied);
        }
        let required_permission = required_permission(method, path)
            .filter(|permission| OPERATOR_PERMISSIONS.contains(permission))
            .ok_or(AuthError::AuthorizationDenied)?;
        debug_assert!(OPERATOR_PERMISSIONS.contains(&required_permission));
        if !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS) {
            let supplied = headers
                .get(CSRF_HEADER)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default();
            if !constant_time_text_equal(&session.csrf_token, supplied) {
                return Err(AuthError::CsrfRejected);
            }
        }
        Ok(session)
    }

    pub fn authorize_license_activation(&self, peer: IpAddr) -> Result<(), AuthError> {
        let now = unix_seconds()?;
        let mut state = self.state.lock().expect("auth state mutex poisoned");
        prune(&mut state, now);
        let window = state
            .license_activation_attempts
            .entry(peer)
            .or_insert(RequestWindow {
                window_started_at: now,
                attempts: 0,
            });
        if now.saturating_sub(window.window_started_at) >= LICENSE_ACTIVATION_WINDOW_SECONDS {
            *window = RequestWindow {
                window_started_at: now,
                attempts: 0,
            };
        }
        if window.attempts >= LICENSE_ACTIVATION_LIMIT {
            return Err(AuthError::LicenseActivationRateLimited);
        }
        window.attempts += 1;
        Ok(())
    }

    pub fn logout(&self, headers: &HeaderMap) -> Result<HeaderValue, AuthError> {
        let session = self.session(headers)?;
        let supplied = headers
            .get(CSRF_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !constant_time_text_equal(&session.csrf_token, supplied) {
            return Err(AuthError::CsrfRejected);
        }
        if let Some(token) = cookie_value(headers, SESSION_COOKIE)
            && let Some(session) = self
                .state
                .lock()
                .expect("auth state mutex poisoned")
                .sessions
                .remove(&token_digest(&token))
        {
            session.revocation.send_replace(true);
        }
        let secure = if self.secure_cookie { "; Secure" } else { "" };
        HeaderValue::from_str(&format!(
            "{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}"
        ))
        .map_err(|error| AuthError::Cookie(error.to_string()))
    }
}

fn authenticated_status(csrf_token: String, expires_at: u64) -> SessionStatus {
    SessionStatus {
        authenticated: true,
        principal: Some("local-operator"),
        role: Some("operator"),
        permissions: OPERATOR_PERMISSIONS.to_vec(),
        csrf_token: Some(csrf_token),
        expires_at: Some(expires_at),
        session_lifetime_seconds: SESSION_SECONDS,
    }
}

fn required_permission(method: &Method, path: &str) -> Option<&'static str> {
    if matches!(*method, Method::GET | Method::HEAD)
        && [
            "/api/license",
            "/api/runtime",
            "/api/config",
            "/api/capture",
            "/api/crosshair",
            "/api/executors",
            "/api/models",
            "/api/v1/status",
            "/api/v1/config",
            "/api/v1/events",
            "/api/v1/runtime",
            "/ws/status",
        ]
        .iter()
        .any(|prefix| path_is(prefix, path))
    {
        return Some("studio:read");
    }
    for (prefix, permission) in [
        ("/api/runtime", "runtime:operate"),
        ("/api/v1/runtime", "runtime:operate"),
        ("/api/config", "configuration:write"),
        ("/api/v1/config", "configuration:write"),
        ("/api/capture", "configuration:write"),
        ("/api/crosshair", "configuration:write"),
        ("/api/models", "models:manage"),
        ("/api/license", "license:manage"),
        ("/api/executors", "hardware:operate"),
    ] {
        if path_is(prefix, path) {
            return Some(permission);
        }
    }
    None
}

fn path_is(prefix: &str, path: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn anonymous_status() -> SessionStatus {
    SessionStatus {
        authenticated: false,
        principal: None,
        role: None,
        permissions: Vec::new(),
        csrf_token: None,
        expires_at: None,
        session_lifetime_seconds: SESSION_SECONDS,
    }
}

fn prune(state: &mut AuthState, now: u64) {
    state.sessions.retain(|_, session| {
        let active = session.expires_at > now;
        if !active {
            session.revocation.send_replace(true);
        }
        active
    });
    state.login_attempts.retain(|_, attempt| {
        attempt.blocked_until > now
            || now.saturating_sub(attempt.window_started_at) < LOGIN_WINDOW_SECONDS
    });
    state.license_activation_attempts.retain(|_, window| {
        now.saturating_sub(window.window_started_at) < LICENSE_ACTIVATION_WINDOW_SECONDS
    });
}

fn record_failure(state: &mut AuthState, peer: IpAddr, now: u64) {
    let attempt = state.login_attempts.entry(peer).or_insert(LoginAttempt {
        window_started_at: now,
        failures: 0,
        blocked_until: 0,
    });
    if now.saturating_sub(attempt.window_started_at) >= LOGIN_WINDOW_SECONDS {
        *attempt = LoginAttempt {
            window_started_at: now,
            failures: 0,
            blocked_until: 0,
        };
    }
    attempt.failures += 1;
    if attempt.failures >= LOGIN_FAILURE_LIMIT {
        attempt.blocked_until = now + LOGIN_WINDOW_SECONDS;
    }
}

fn random_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn token_digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

fn constant_time_equal(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn constant_time_text_equal(left: &str, right: &str) -> bool {
    let left_digest: [u8; 32] = Sha256::digest(left.as_bytes()).into();
    let right_digest: [u8; 32] = Sha256::digest(right.as_bytes()).into();
    constant_time_equal(&left_digest, &right_digest) && left.len() == right.len()
}

fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get_all(header::COOKIE).iter().find_map(|header| {
        header.to_str().ok()?.split(';').find_map(|cookie| {
            let (cookie_name, value) = cookie.trim().split_once('=')?;
            (cookie_name == name && !value.is_empty()).then(|| value.to_owned())
        })
    })
}

fn normalized_host(host: &str) -> String {
    host.trim_matches(['[', ']']).to_ascii_lowercase()
}

fn is_loopback_host(host: &str) -> bool {
    host == "localhost"
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn unix_seconds() -> Result<u64, AuthError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| AuthError::Clock)
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use axum::http::{HeaderMap, HeaderValue, Method, header};

    use super::{AuthError, AuthService, CSRF_HEADER};

    const ACCESS_CODE: &str = "8f0ed8de1c54cb8c34a4369c02fd0c6883b2f3e2b7b3321c4b4cb08c1e0d7f50";

    #[test]
    fn session_requires_its_csrf_token_for_mutations() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let issued = auth
            .login(ACCESS_CODE, IpAddr::V4(Ipv4Addr::LOCALHOST))
            .unwrap();
        let cookie = issued.cookie.to_str().unwrap().split(';').next().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_str(cookie).unwrap());
        assert!(matches!(
            auth.authorize_proxy(&Method::POST, "/api/runtime/start", &headers),
            Err(AuthError::CsrfRejected)
        ));
        headers.insert(
            CSRF_HEADER,
            HeaderValue::from_str(issued.status.csrf_token.as_deref().unwrap()).unwrap(),
        );
        auth.authorize_proxy(&Method::POST, "/api/runtime/start", &headers)
            .unwrap();
    }

    #[test]
    fn local_daemon_shutdown_is_never_web_authorized() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let issued = auth
            .login(ACCESS_CODE, IpAddr::V4(Ipv4Addr::LOCALHOST))
            .unwrap();
        let cookie = issued.cookie.to_str().unwrap().split(';').next().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_str(cookie).unwrap());
        headers.insert(
            CSRF_HEADER,
            HeaderValue::from_str(issued.status.csrf_token.as_deref().unwrap()).unwrap(),
        );
        assert!(
            auth.authorize_proxy(&Method::POST, "/api/v1/daemon/shutdown", &headers)
                .is_err()
        );
    }

    #[test]
    fn unknown_daemon_routes_are_denied_until_policy_names_them() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let issued = auth
            .login(ACCESS_CODE, IpAddr::V4(Ipv4Addr::LOCALHOST))
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(issued.cookie.to_str().unwrap()).unwrap(),
        );

        assert!(
            auth.authorize_proxy(&Method::GET, "/api/future-sensitive-route", &headers)
                .is_err()
        );
    }

    #[test]
    fn dns_hosts_require_an_exact_allowlist_and_matching_origin() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::HOST,
            HeaderValue::from_static("studio.example:7351"),
        );
        assert!(matches!(
            auth.validate_request_site(&headers),
            Err(AuthError::HostRejected(_))
        ));

        let auth = auth
            .with_allowed_hosts(&["studio.example".to_owned()])
            .unwrap();
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://other.example"),
        );
        assert!(matches!(
            auth.validate_request_site(&headers),
            Err(AuthError::OriginRejected)
        ));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://studio.example"),
        );
        auth.validate_request_site(&headers).unwrap();
    }

    #[test]
    fn repeated_login_failures_are_rate_limited_per_peer() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let peer = IpAddr::V4(Ipv4Addr::new(192, 168, 10, 42));
        for _ in 0..5 {
            assert!(matches!(
                auth.login("wrong", peer),
                Err(AuthError::Rejected)
            ));
        }
        assert!(matches!(
            auth.login(ACCESS_CODE, peer),
            Err(AuthError::RateLimited)
        ));
    }

    #[test]
    fn license_activation_attempts_are_rate_limited_per_peer() {
        let auth = AuthService::new(ACCESS_CODE, false).unwrap();
        let peer = IpAddr::V4(Ipv4Addr::new(192, 168, 10, 42));
        for _ in 0..6 {
            auth.authorize_license_activation(peer).unwrap();
        }
        assert!(matches!(
            auth.authorize_license_activation(peer),
            Err(AuthError::LicenseActivationRateLimited)
        ));
        auth.authorize_license_activation(IpAddr::V4(Ipv4Addr::new(192, 168, 10, 43)))
            .unwrap();
    }
}
