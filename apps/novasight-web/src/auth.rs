use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, HeaderValue, Method, Uri, header};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::watch;

pub const SESSION_COOKIE: &str = "novasight_web_session";
pub const REFRESH_COOKIE: &str = "novasight_web_refresh";
const ROTATION_GRACE_SECONDS: u64 = 10;
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
const MAX_SESSIONS: usize = 64;
const LICENSE_ACTIVATION_WINDOW_SECONDS: u64 = 60;
const LICENSE_ACTIVATION_LIMIT: u32 = 6;

#[derive(Clone)]
pub struct AuthService {
    secure_cookie: bool,
    allowed_hosts: Arc<[String]>,
    state: Arc<Mutex<AuthState>>,
    key: Arc<[u8; 32]>,
    store: Option<Arc<PathBuf>>,
    _lock: Option<Arc<File>>,
}

#[derive(Default)]
struct AuthState {
    families: HashMap<String, SessionRecord>,
    license_activation_attempts: HashMap<IpAddr, RequestWindow>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SessionRecord {
    csrf_token: String,
    binding: LicenseBinding,
    generation: u64,
    issued_at: u64,
    #[serde(default)]
    rotated_at: u64,
    #[serde(default)]
    confirmed_generation: u64,
    #[serde(default)]
    confirmed_at: u64,
    revoked: bool,
    #[serde(skip, default = "revocation_signal")]
    revocation: watch::Sender<bool>,
}

fn revocation_signal() -> watch::Sender<bool> {
    watch::channel(false).0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LicenseBinding {
    pub license_id: String,
    pub fingerprint: String,
    pub created_at: Option<f64>,
    pub expires_at: f64,
}

#[derive(Serialize, Deserialize)]
struct AuthDocument {
    schema_version: u32,
    key: [u8; 32],
    families: HashMap<String, SessionRecord>,
}

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String,
    kind: String,
    generation: u64,
    iat: u64,
    exp: u64,
}

#[derive(Clone, Copy)]
struct RequestWindow {
    window_started_at: u64,
    attempts: u32,
}

pub struct IssuedSession {
    pub cookie: HeaderValue,
    pub refresh_cookie: HeaderValue,
    pub status: SessionStatus,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionStatus {
    pub authenticated: bool,
    pub auth_state: &'static str,
    pub server_time: u64,
    pub license_expires_at: Option<f64>,
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
    #[error("too many license activation attempts; retry later")]
    LicenseActivationRateLimited,
    #[error("authentication is required")]
    Required,
    #[error("authorization expired")]
    Expired,
    #[error("authorization revoked")]
    Revoked,
    #[error("session storage unavailable: {0}")]
    Store(String),
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
    pub fn new(secure_cookie: bool) -> Self {
        let mut key = [0; 32];
        OsRng.fill_bytes(&mut key);
        Self {
            secure_cookie,
            allowed_hosts: Arc::from([]),
            state: Arc::new(Mutex::new(AuthState::default())),
            key: Arc::new(key),
            store: None,
            _lock: None,
        }
    }

    pub fn open(path: &Path, secure_cookie: bool) -> Result<Self, AuthError> {
        let parent = path.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(store_error)?;
        let lock_path = path.with_extension("lock");
        let existing_store = lock_path.exists();
        let lock = private_file(&lock_path, false)?;
        fs2::FileExt::try_lock_exclusive(&lock).map_err(store_error)?;
        let mut auth = Self::new(secure_cookie);
        match fs::read(path) {
            Ok(bytes) => {
                let document: AuthDocument = serde_json::from_slice(&bytes).map_err(store_error)?;
                if document.schema_version != 1 {
                    return Err(AuthError::Store("unsupported schema".into()));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if fs::symlink_metadata(path)
                        .map_err(store_error)?
                        .permissions()
                        .mode()
                        & 0o077
                        != 0
                    {
                        return Err(AuthError::Store(
                            "session file must be private (0600)".into(),
                        ));
                    }
                }
                auth.key = Arc::new(document.key);
                auth.state.lock().unwrap().families = document.families;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !existing_store => {}
            Err(error) => return Err(store_error(error)),
        }
        auth.store = Some(Arc::new(path.to_owned()));
        auth._lock = Some(Arc::new(lock));
        auth.persist(&auth.state.lock().unwrap().families)?;
        Ok(auth)
    }

    fn persist(&self, families: &HashMap<String, SessionRecord>) -> Result<(), AuthError> {
        let Some(path) = &self.store else {
            return Ok(());
        };
        let bytes = serde_json::to_vec(&AuthDocument {
            schema_version: 1,
            key: *self.key,
            families: families.clone(),
        })
        .map_err(store_error)?;
        let temporary = path.with_extension(format!("{}.tmp", random_token()));
        let result = (|| {
            let mut file = private_file(&temporary, true)?;
            file.write_all(&bytes).map_err(store_error)?;
            file.sync_all().map_err(store_error)?;
            fs::rename(&temporary, path.as_ref()).map_err(store_error)?;
            File::open(path.parent().unwrap_or(Path::new(".")))
                .and_then(|file| file.sync_all())
                .map_err(store_error)
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
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

    /// Called only after the daemon has validated the submitted license.
    pub fn issue_session(
        &self,
        binding: LicenseBinding,
        now: u64,
    ) -> Result<IssuedSession, AuthError> {
        if binding.expires_at <= now as f64 {
            return Err(AuthError::Expired);
        }
        let mut state = self.state.lock().expect("auth state mutex poisoned");
        let mut families = state.families.clone();
        families.retain(|_, family| family.binding.expires_at > now as f64);
        if families.len() >= MAX_SESSIONS {
            return Err(AuthError::Store("session capacity reached".into()));
        }
        let id = random_token();
        let record = SessionRecord {
            csrf_token: random_token(),
            binding,
            generation: 0,
            issued_at: now,
            rotated_at: now,
            confirmed_generation: 0,
            confirmed_at: now,
            revoked: false,
            revocation: revocation_signal(),
        };
        let issued = self.encode_session(&id, &record, now)?;
        families.insert(id, record);
        self.persist(&families)?;
        state.families = families;
        Ok(issued)
    }

    fn decode_token(&self, token: &str, kind: &str) -> Result<Claims, AuthError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = false;
        let claims = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.key.as_ref()),
            &validation,
        )
        .map_err(|_| AuthError::Required)?
        .claims;
        if claims.kind != kind {
            return Err(AuthError::Required);
        }
        Ok(claims)
    }

    fn encode_session(
        &self,
        id: &str,
        record: &SessionRecord,
        now: u64,
    ) -> Result<IssuedSession, AuthError> {
        let access_deadline = ((record.issued_at + SESSION_SECONDS) as f64)
            .min(record.binding.expires_at)
            .ceil() as u64;
        let token = |kind: &str, exp: u64| {
            encode(
                &Header::new(Algorithm::HS256),
                &Claims {
                    sub: id.to_owned(),
                    kind: kind.to_owned(),
                    generation: record.generation,
                    iat: if kind == "access" {
                        record.issued_at
                    } else {
                        record.rotated_at
                    },
                    exp,
                },
                &EncodingKey::from_secret(self.key.as_ref()),
            )
            .map_err(store_error)
        };
        Ok(IssuedSession {
            cookie: self.cookie(
                SESSION_COOKIE,
                &token("access", access_deadline)?,
                "/",
                access_deadline.saturating_sub(now),
            )?,
            refresh_cookie: self.cookie(
                REFRESH_COOKIE,
                &token("refresh", record.binding.expires_at.ceil() as u64)?,
                "/api/auth",
                (record.binding.expires_at.ceil() as u64).saturating_sub(now),
            )?,
            status: authenticated_status(
                record.csrf_token.clone(),
                access_deadline,
                record.binding.expires_at,
                now,
            ),
        })
    }

    fn cookie(
        &self,
        name: &str,
        token: &str,
        path: &str,
        max_age: u64,
    ) -> Result<HeaderValue, AuthError> {
        let secure = if self.secure_cookie { "; Secure" } else { "" };
        HeaderValue::from_str(&format!(
            "{name}={token}; HttpOnly; SameSite=Strict; Path={path}; Max-Age={max_age}{secure}"
        ))
        .map_err(|error| AuthError::Cookie(error.to_string()))
    }

    pub fn access_binding(&self, headers: &HeaderMap) -> Result<LicenseBinding, AuthError> {
        let token = cookie_value(headers, SESSION_COOKIE).ok_or(AuthError::Required)?;
        let claims = self.decode_token(&token, "access")?;
        self.state
            .lock()
            .unwrap()
            .families
            .get(&claims.sub)
            .map(|family| family.binding.clone())
            .ok_or(AuthError::Required)
    }

    pub fn rebind(
        &self,
        headers: &HeaderMap,
        binding: LicenseBinding,
    ) -> Result<IssuedSession, AuthError> {
        let token = cookie_value(headers, SESSION_COOKIE).ok_or(AuthError::Required)?;
        let claims = self.decode_token(&token, "access")?;
        let now = unix_seconds()?;
        if binding.expires_at <= now as f64 {
            return Err(AuthError::Expired);
        }
        let mut state = self.state.lock().unwrap();
        let mut families = state.families.clone();
        let family = families.get_mut(&claims.sub).ok_or(AuthError::Required)?;
        if family.revoked {
            return Err(AuthError::Revoked);
        }
        family.binding = binding;
        family.generation += 1;
        family.issued_at = now;
        family.rotated_at = now;
        let issued = self.encode_session(&claims.sub, family, now)?;
        self.persist(&families)?;
        state.families = families;
        Ok(issued)
    }

    pub fn revoke(&self, headers: &HeaderMap) -> Result<(), AuthError> {
        let token = cookie_value(headers, REFRESH_COOKIE)
            .or_else(|| cookie_value(headers, SESSION_COOKIE))
            .ok_or(AuthError::Required)?;
        let claims = self.decode_token(
            &token,
            if cookie_value(headers, REFRESH_COOKIE).is_some() {
                "refresh"
            } else {
                "access"
            },
        )?;
        let mut state = self.state.lock().unwrap();
        let mut families = state.families.clone();
        let family = families.get_mut(&claims.sub).ok_or(AuthError::Required)?;
        family.revoked = true;
        let signal = family.revocation.clone();
        self.persist(&families)?;
        state.families = families;
        signal.send_replace(true);
        Ok(())
    }

    pub fn refresh_binding(&self, headers: &HeaderMap) -> Result<LicenseBinding, AuthError> {
        let token = cookie_value(headers, REFRESH_COOKIE).ok_or(AuthError::Required)?;
        let claims = self.decode_token(&token, "refresh")?;
        let state = self.state.lock().unwrap();
        let family = state.families.get(&claims.sub).ok_or(AuthError::Required)?;
        if family.revoked {
            return Err(AuthError::Revoked);
        }
        if family.binding.expires_at <= unix_timestamp()?
            || claims.exp != family.binding.expires_at.ceil() as u64
        {
            return Err(AuthError::Expired);
        }
        Ok(family.binding.clone())
    }

    pub fn refresh(
        &self,
        headers: &HeaderMap,
        binding: &LicenseBinding,
    ) -> Result<IssuedSession, AuthError> {
        let now = unix_seconds()?;
        let token = cookie_value(headers, REFRESH_COOKIE).ok_or(AuthError::Required)?;
        let claims = self.decode_token(&token, "refresh")?;
        let mut state = self.state.lock().unwrap();
        let mut families = state.families.clone();
        let family = families.get_mut(&claims.sub).ok_or(AuthError::Required)?;
        if family.revoked {
            return Err(AuthError::Revoked);
        }
        if family.binding.expires_at <= unix_timestamp()? {
            return Err(AuthError::Expired);
        }
        if &family.binding != binding || claims.exp != family.binding.expires_at.ceil() as u64 {
            family.revoked = true;
        } else if claims.generation == family.generation {
            if family.generation > family.confirmed_generation {
                family.confirmed_generation = family.generation;
                family.confirmed_at = now;
            }
            if family.generation == 0
                || now.saturating_sub(family.confirmed_at) > ROTATION_GRACE_SECONDS
            {
                family.generation += 1;
                family.rotated_at = now;
                family.issued_at = now;
            } else if family.issued_at + SESSION_SECONDS <= now + 60 {
                family.issued_at = now;
            }
        } else if claims.generation.checked_add(1) == Some(family.generation)
            && (family.confirmed_generation < family.generation
                || now.saturating_sub(family.confirmed_at) <= ROTATION_GRACE_SECONDS)
        {
            if family.issued_at + SESSION_SECONDS <= now + 60 {
                family.issued_at = now;
            }
        } else {
            family.revoked = true;
        }
        let revoked = family.revoked;
        let issued = if revoked {
            None
        } else {
            Some(self.encode_session(&claims.sub, family, now)?)
        };
        let signal = family.revocation.clone();
        self.persist(&families)?;
        state.families = families;
        if revoked {
            signal.send_replace(true);
            return Err(AuthError::Revoked);
        }
        Ok(issued.expect("non-revoked family"))
    }

    pub fn session(&self, headers: &HeaderMap) -> Result<AuthenticatedSession, AuthError> {
        let token = cookie_value(headers, SESSION_COOKIE).ok_or(AuthError::Required)?;
        let claims = self.decode_token(&token, "access")?;
        let now = unix_seconds()?;
        let mut state = self.state.lock().unwrap();
        let family = state.families.get(&claims.sub).ok_or(AuthError::Required)?;
        if family.revoked {
            return Err(AuthError::Revoked);
        }
        if family.binding.expires_at <= unix_timestamp()? {
            return Err(AuthError::Expired);
        }
        if claims.exp <= now
            || claims.exp > family.binding.expires_at.ceil() as u64
            || claims.generation > family.generation
        {
            return Err(AuthError::Required);
        }
        if claims.generation == family.generation && family.confirmed_generation < family.generation
        {
            let mut families = state.families.clone();
            let current = families.get_mut(&claims.sub).unwrap();
            current.confirmed_generation = current.generation;
            current.confirmed_at = now;
            self.persist(&families)?;
            state.families = families;
        }
        let family = state.families.get(&claims.sub).unwrap();
        Ok(AuthenticatedSession {
            csrf_token: family.csrf_token.clone(),
            // Established channels belong to the revocable family, not a short access token.
            expires_at: family.binding.expires_at.ceil() as u64,
            revocation: family.revocation.subscribe(),
        })
    }

    pub fn status(&self, headers: &HeaderMap) -> SessionStatus {
        let now = unix_seconds().unwrap_or_default();
        if let Ok(session) = self.session(headers) {
            let token = cookie_value(headers, SESSION_COOKIE).unwrap();
            let claims = self.decode_token(&token, "access").unwrap();
            if let Ok(binding) = self.access_binding(headers) {
                return authenticated_status(
                    session.csrf_token,
                    claims.exp,
                    binding.expires_at,
                    now,
                );
            }
        }
        // Only refresh can decide whether a missing access token is recoverable.
        anonymous_status(
            if cookie_value(headers, REFRESH_COOKIE).is_some() {
                "checking"
            } else {
                "required"
            },
            now,
        )
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

    pub fn logout(&self, headers: &HeaderMap) -> Result<[HeaderValue; 2], AuthError> {
        // Origin validation plus CSRF where an access token exists; an expired
        // access token can still be logged out using its refresh family.
        let token = cookie_value(headers, REFRESH_COOKIE)
            .or_else(|| cookie_value(headers, SESSION_COOKIE))
            .ok_or(AuthError::Required)?;
        let claims = self.decode_token(
            &token,
            if cookie_value(headers, REFRESH_COOKIE).is_some() {
                "refresh"
            } else {
                "access"
            },
        )?;
        let mut state = self.state.lock().unwrap();
        let mut families = state.families.clone();
        let family = families.get_mut(&claims.sub).ok_or(AuthError::Required)?;
        if cookie_value(headers, SESSION_COOKIE).is_some() {
            let supplied = headers
                .get(CSRF_HEADER)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default();
            if !constant_time_text_equal(&family.csrf_token, supplied) {
                return Err(AuthError::CsrfRejected);
            }
        }
        family.revoked = true;
        let signal = family.revocation.clone();
        self.persist(&families)?;
        state.families = families;
        signal.send_replace(true);
        Ok([
            self.cookie(SESSION_COOKIE, "", "/", 0)?,
            self.cookie(REFRESH_COOKIE, "", "/api/auth", 0)?,
        ])
    }
}

fn authenticated_status(
    csrf_token: String,
    expires_at: u64,
    license_expires_at: f64,
    now: u64,
) -> SessionStatus {
    SessionStatus {
        authenticated: true,
        auth_state: "authorized",
        server_time: now,
        license_expires_at: Some(license_expires_at),
        principal: Some("local-operator"),
        role: Some("operator"),
        permissions: OPERATOR_PERMISSIONS.to_vec(),
        csrf_token: Some(csrf_token),
        expires_at: Some(expires_at),
        session_lifetime_seconds: SESSION_SECONDS,
    }
}

fn store_error(error: impl std::fmt::Display) -> AuthError {
    AuthError::Store(error.to_string())
}

fn private_file(path: &Path, create_new: bool) -> Result<File, AuthError> {
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(AuthError::Store(
            "session path must not be a symlink".into(),
        ));
    }
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create(!create_new)
        .create_new(create_new);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(store_error)
}

fn required_permission(method: &Method, path: &str) -> Option<&'static str> {
    if *method == Method::DELETE && path == "/api/activity" {
        return Some("runtime:operate");
    }
    if matches!(*method, Method::GET | Method::HEAD)
        && [
            "/api/license",
            "/api/runtime",
            "/api/studio/v1",
            "/api/config",
            "/api/capture",
            "/api/executors",
            "/api/models",
            "/api/v1/status",
            "/api/v1/config",
            "/api/v1/events",
            "/api/v1/runtime",
            "/api/activity",
            "/ws/status",
            "/ws/studio/v1",
            "/ws/activity",
        ]
        .iter()
        .any(|prefix| path_is(prefix, path))
    {
        return Some("studio:read");
    }
    for (prefix, permission) in [
        ("/api/runtime", "runtime:operate"),
        ("/api/studio/v1/runtime", "runtime:operate"),
        ("/api/v1/runtime", "runtime:operate"),
        ("/api/config", "configuration:write"),
        ("/api/v1/config", "configuration:write"),
        ("/api/capture", "configuration:write"),
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

pub fn anonymous_status(auth_state: &'static str, now: u64) -> SessionStatus {
    SessionStatus {
        authenticated: false,
        auth_state,
        server_time: now,
        license_expires_at: None,
        principal: None,
        role: None,
        permissions: Vec::new(),
        csrf_token: None,
        expires_at: None,
        session_lifetime_seconds: SESSION_SECONDS,
    }
}

fn prune(state: &mut AuthState, now: u64) {
    state.license_activation_attempts.retain(|_, window| {
        now.saturating_sub(window.window_started_at) < LICENSE_ACTIVATION_WINDOW_SECONDS
    });
}

fn random_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
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

pub fn unix_timestamp() -> Result<f64, AuthError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .map_err(|_| AuthError::Clock)
}

pub fn unix_seconds() -> Result<u64, AuthError> {
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
    fn test_binding() -> super::LicenseBinding {
        super::LicenseBinding {
            license_id: "test".into(),
            fingerprint: "machine".into(),
            created_at: Some(1.0),
            expires_at: (super::unix_seconds().unwrap() + 86400) as f64,
        }
    }

    #[test]
    fn session_requires_its_csrf_token_for_mutations() {
        let auth = AuthService::new(false);
        let issued = auth
            .issue_session(test_binding(), super::unix_seconds().unwrap())
            .unwrap();
        let cookie = issued.cookie.to_str().unwrap().split(';').next().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_str(cookie).unwrap());
        for path in ["/api/runtime/start", "/api/executors/kmnet/disconnect"] {
            assert!(matches!(
                auth.authorize_proxy(&Method::POST, path, &HeaderMap::new()),
                Err(AuthError::Required)
            ));
            assert!(matches!(
                auth.authorize_proxy(&Method::POST, path, &headers),
                Err(AuthError::CsrfRejected)
            ));
        }
        headers.insert(
            CSRF_HEADER,
            HeaderValue::from_str(issued.status.csrf_token.as_deref().unwrap()).unwrap(),
        );
        for path in ["/api/runtime/start", "/api/executors/kmnet/disconnect"] {
            auth.authorize_proxy(&Method::POST, path, &headers).unwrap();
        }
    }

    #[test]
    fn web_operator_can_clear_activity_history() {
        let auth = AuthService::new(false);
        let issued = auth
            .issue_session(test_binding(), super::unix_seconds().unwrap())
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(issued.cookie.to_str().unwrap().split(';').next().unwrap())
                .unwrap(),
        );
        headers.insert(
            CSRF_HEADER,
            HeaderValue::from_str(issued.status.csrf_token.as_deref().unwrap()).unwrap(),
        );

        auth.authorize_proxy(&Method::DELETE, "/api/activity", &headers)
            .unwrap();
    }

    #[test]
    fn local_daemon_shutdown_is_never_web_authorized() {
        let auth = AuthService::new(false);
        let issued = auth
            .issue_session(test_binding(), super::unix_seconds().unwrap())
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
        let auth = AuthService::new(false);
        let issued = auth
            .issue_session(test_binding(), super::unix_seconds().unwrap())
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
        auth.authorize_proxy(&Method::GET, "/api/activity", &headers)
            .unwrap();
        auth.authorize_proxy(&Method::GET, "/ws/activity", &headers)
            .unwrap();
    }

    #[test]
    fn dns_hosts_require_an_exact_allowlist_and_matching_origin() {
        let auth = AuthService::new(false);
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
    fn license_activation_attempts_are_rate_limited_per_peer() {
        let auth = AuthService::new(false);
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
    fn cookies(issued: &super::IssuedSession) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!(
                "{}; {}",
                issued.cookie.to_str().unwrap().split(';').next().unwrap(),
                issued
                    .refresh_cookie
                    .to_str()
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap()
            ))
            .unwrap(),
        );
        headers.insert(
            CSRF_HEADER,
            HeaderValue::from_str(issued.status.csrf_token.as_deref().unwrap()).unwrap(),
        );
        headers
    }

    #[test]
    fn elapsed_access_and_two_hour_sleep_restore_identity_without_extending_license() {
        for elapsed in [3601, 7200] {
            let auth = AuthService::new(false);
            let now = super::unix_seconds().unwrap();
            let binding = test_binding();
            let first = auth.issue_session(binding.clone(), now - elapsed).unwrap();
            let headers = cookies(&first);
            assert!(matches!(auth.session(&headers), Err(AuthError::Required)));
            assert_eq!(auth.status(&headers).auth_state, "checking");
            let renewed = auth.refresh(&headers, &binding).unwrap();
            assert!(auth.session(&cookies(&renewed)).is_ok());
            assert_eq!(renewed.status.license_expires_at, Some(binding.expires_at));
        }
    }

    #[test]
    fn parallel_tabs_receive_one_rotation_and_revoked_families_cannot_return() {
        let auth = AuthService::new(false);
        let binding = test_binding();
        let original = auth
            .issue_session(binding.clone(), super::unix_seconds().unwrap())
            .unwrap();
        let old = cookies(&original);
        let first = auth.refresh(&old, &binding).unwrap();
        let second = auth.refresh(&old, &binding).unwrap();
        assert_eq!(first.refresh_cookie, second.refresh_cookie);
        assert_eq!(first.cookie, second.cookie);
        assert!(auth.session(&old).is_ok());
        let newest = auth.refresh(&cookies(&first), &binding).unwrap();
        assert!(auth.refresh(&old, &binding).is_ok());
        auth.logout(&cookies(&newest)).unwrap();
        assert!(matches!(
            auth.session(&cookies(&newest)),
            Err(AuthError::Revoked)
        ));
    }

    #[test]
    fn restart_preserves_refresh_and_logout_remains_revoked_after_another_restart() {
        let directory = std::env::temp_dir().join(format!("ns-auth-{}", super::random_token()));
        let path = directory.join("sessions.json");
        let auth = AuthService::open(&path, false).unwrap();
        let binding = test_binding();
        let issued = auth
            .issue_session(binding.clone(), super::unix_seconds().unwrap() - 7200)
            .unwrap();
        let old = cookies(&issued);
        drop(auth);
        let auth = AuthService::open(&path, false).unwrap();
        let renewed = auth.refresh(&old, &binding).unwrap();
        let headers = cookies(&renewed);
        auth.logout(&headers).unwrap();
        drop(auth);
        let auth = AuthService::open(&path, false).unwrap();
        assert!(matches!(
            auth.refresh_binding(&headers),
            Err(AuthError::Revoked)
        ));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(auth);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn machine_deadline_and_binding_are_authoritative() {
        let auth = AuthService::new(false);
        let now = super::unix_seconds().unwrap();
        let mut binding = test_binding();
        binding.expires_at = (now + 20) as f64;
        let issued = auth.issue_session(binding.clone(), now).unwrap();
        assert_eq!(issued.status.expires_at, Some(binding.expires_at as u64));
        let mut replacement = binding.clone();
        replacement.fingerprint = "replaced".into();
        assert!(matches!(
            auth.refresh(&cookies(&issued), &replacement),
            Err(AuthError::Revoked)
        ));
        binding.expires_at = now as f64;
        assert!(matches!(
            auth.issue_session(binding, now),
            Err(AuthError::Expired)
        ));
    }

    #[test]
    fn persistence_failure_does_not_confirm_a_rotation() {
        let directory = std::env::temp_dir().join(format!("ns-auth-{}", super::random_token()));
        let path = directory.join("sessions.json");
        let auth = AuthService::open(&path, false).unwrap();
        let binding = test_binding();
        let issued = auth
            .issue_session(binding.clone(), super::unix_seconds().unwrap())
            .unwrap();
        let headers = cookies(&issued);
        std::fs::rename(&directory, directory.with_extension("saved")).unwrap();
        assert!(matches!(
            auth.refresh(&headers, &binding),
            Err(AuthError::Store(_))
        ));
        assert!(auth.session(&headers).is_ok());
        std::fs::rename(directory.with_extension("saved"), &directory).unwrap();
        assert!(auth.refresh(&headers, &binding).is_ok());
        drop(auth);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn fractional_machine_deadlines_are_not_rounded_into_an_early_expiration() {
        let auth = AuthService::new(false);
        let now = super::unix_seconds().unwrap();
        let mut binding = test_binding();
        binding.expires_at = now as f64 + 1.999;
        let issued = auth.issue_session(binding.clone(), now).unwrap();
        assert_eq!(issued.status.license_expires_at, Some(binding.expires_at));
        assert_eq!(issued.status.expires_at, Some(now + 2));
        assert!(auth.refresh_binding(&cookies(&issued)).is_ok());
        binding.expires_at = super::unix_timestamp().unwrap() - 0.001;
        let expired = auth.issue_session(binding.clone(), now - 1).unwrap();
        assert!(matches!(
            auth.refresh_binding(&cookies(&expired)),
            Err(AuthError::Expired)
        ));
    }

    #[test]
    fn explicit_license_replacement_rebinds_the_same_browser_without_another_prompt() {
        let auth = AuthService::new(false);
        let original = test_binding();
        let issued = auth
            .issue_session(original.clone(), super::unix_seconds().unwrap())
            .unwrap();
        let mut replacement = original;
        replacement.fingerprint = "new-user-submitted-license".into();
        let rebound = auth.rebind(&cookies(&issued), replacement.clone()).unwrap();
        assert_eq!(
            auth.refresh_binding(&cookies(&rebound)).unwrap(),
            replacement
        );
        assert!(auth.refresh(&cookies(&rebound), &replacement).is_ok());
    }
    #[test]
    fn a_lost_refresh_response_survives_a_restart_and_a_long_network_outage() {
        let directory = std::env::temp_dir().join(format!("ns-lost-{}", super::random_token()));
        let path = directory.join("sessions.json");
        let auth = AuthService::open(&path, false).unwrap();
        let binding = test_binding();
        let original = auth
            .issue_session(binding.clone(), super::unix_seconds().unwrap())
            .unwrap();
        let parent = cookies(&original);
        let lost = auth.refresh(&parent, &binding).unwrap();
        drop(auth);
        // Fault fixture advances only the persisted rotation/access age; the original license deadline is unchanged.
        let mut document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for family in document["families"].as_object_mut().unwrap().values_mut() {
            family["rotated_at"] = (super::unix_seconds().unwrap() - 7200).into();
            family["issued_at"] = (super::unix_seconds().unwrap() - 7200).into();
        }
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let auth = AuthService::open(&path, false).unwrap();
        let recovered = auth.refresh(&parent, &binding).unwrap();
        assert!(auth.session(&cookies(&recovered)).is_ok());
        assert_eq!(
            recovered.status.license_expires_at,
            lost.status.license_expires_at
        );
        // Another late parent from the same recovery wave must not revoke the browser.
        assert!(auth.refresh(&parent, &binding).is_ok());
        drop(auth);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
