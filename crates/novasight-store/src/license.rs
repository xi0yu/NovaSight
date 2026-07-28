use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use fs2::FileExt;
use rsa::RsaPublicKey;
use rsa::pkcs1::DecodeRsaPublicKey;
use rsa::pkcs1v15::{Signature, VerifyingKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub mod jwt;

use jwt::verify_license_jwt;

const TEMPORARY_LICENSE_TIER: &str = "temporary";
const DEVELOPMENT_SESSION_ID: &str = "debug-process-session";
const LEGACY_CREDENTIAL_FORMAT: &str = "legacy_ns1";
const JWT_CREDENTIAL_FORMAT: &str = "jwt_rs256";
const DEBUG_CREDENTIAL_FORMAT: &str = "debug_session";
const MAX_LICENSE_CREDENTIAL_BYTES: usize = 64 * 1024;
const MAX_UNIX_TIMESTAMP: u64 = 253_402_300_799;
const LICENSE_CLOCK_SKEW_SECONDS: f64 = 60.0;
const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 60.0 * 60.0;
const ALL_FEATURES: [&str; 7] = [
    "capture",
    "runtime",
    "models",
    "tensorrt",
    "hardware_control",
    "config_read",
    "config_write",
];
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct LicensePolicy {
    temporary_access_supported: bool,
    public_key_pem: Option<String>,
}

impl LicensePolicy {
    pub fn new(temporary_access_supported: bool, public_key_pem: Option<String>) -> Self {
        Self {
            temporary_access_supported,
            public_key_pem,
        }
    }

    pub const fn temporary_access_supported(&self) -> bool {
        self.temporary_access_supported
    }

    /// Validate the verifier before daemon readiness. Production requires a
    /// key; dry-run may omit it but still rejects malformed configured PEM.
    pub fn validate_public_key(&self, required: bool) -> Result<(), LicenseError> {
        match self.public_key_pem.as_deref() {
            Some(pem) => parse_public_key(pem).map(|_| ()),
            None if required => Err(LicenseError::PublicKeyMissing),
            None => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LicenseStatus {
    pub configured: bool,
    pub valid: bool,
    #[serde(default)]
    pub temporary_access_supported: bool,
    pub fingerprint: String,
    pub tier: String,
    pub features: Vec<String>,
    pub license_id: String,
    #[serde(default)]
    pub credential_format: String,
    #[serde(default)]
    pub token_id: String,
    #[serde(default)]
    pub key_id: String,
    pub created_at: Option<f64>,
    #[serde(default)]
    pub not_before: Option<f64>,
    pub activated_at: Option<f64>,
    pub expires_at: Option<f64>,
    pub duration_value: Option<u64>,
    pub duration_unit: String,
    pub updated_at: Option<f64>,
    pub message: String,
}

impl LicenseStatus {
    pub fn unconfigured() -> Self {
        Self {
            configured: false,
            valid: false,
            temporary_access_supported: false,
            fingerprint: String::new(),
            tier: String::new(),
            features: Vec::new(),
            license_id: String::new(),
            credential_format: String::new(),
            token_id: String::new(),
            key_id: String::new(),
            created_at: None,
            not_before: None,
            activated_at: None,
            expires_at: None,
            duration_value: None,
            duration_unit: String::new(),
            updated_at: None,
            message: String::new(),
        }
    }

    fn invalid_configured(message: impl Into<String>) -> Self {
        Self {
            configured: true,
            valid: false,
            message: message.into(),
            ..Self::unconfigured()
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileLicenseRepository {
    path: PathBuf,
    policy: LicensePolicy,
    operation_lock: Arc<Mutex<()>>,
    development_session: Arc<Mutex<Option<f64>>>,
}

impl FileLicenseRepository {
    pub fn new(path: impl Into<PathBuf>, policy: LicensePolicy) -> Self {
        Self {
            path: path.into(),
            policy,
            operation_lock: Arc::new(Mutex::new(())),
            development_session: Arc::new(Mutex::new(None)),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn temporary_access_supported(&self) -> bool {
        self.policy.temporary_access_supported()
    }

    pub fn status(&self) -> Result<LicenseStatus, LicenseError> {
        if let Some(granted_at) = *self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Ok(development_session_status(granted_at));
        }
        let _operation = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(granted_at) = *self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Ok(development_session_status(granted_at));
        }
        self.storage_status()
    }

    fn storage_status(&self) -> Result<LicenseStatus, LicenseError> {
        let mut status = match self.read_status(now_seconds()?) {
            Ok(status) => status,
            Err(error) if error.is_invalid_license_state() => {
                LicenseStatus::invalid_configured(error.to_string())
            }
            Err(error) if self.temporary_access_supported() => {
                let mut status = LicenseStatus::unconfigured();
                status.message = format!(
                    "formal license storage is unavailable; Debug temporary access remains available: {error}"
                );
                status
            }
            Err(error) => return Err(error),
        };
        status.temporary_access_supported = self.temporary_access_supported();
        Ok(status)
    }

    pub fn activate(&self, key: &str) -> Result<LicenseStatus, LicenseError> {
        let key = key.trim();
        if key.len() < 8 {
            return Err(LicenseError::InvalidKey(
                "license key must be at least 8 characters".to_owned(),
            ));
        }
        if key.len() > MAX_LICENSE_CREDENTIAL_BYTES {
            return Err(LicenseError::InvalidKey(
                "license credential is too large".to_owned(),
            ));
        }
        let _operation = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _lock = self.lock()?;
        let now = now_seconds()?;
        let credential = verify_license_credential(
            key,
            self.policy
                .public_key_pem
                .as_deref()
                .ok_or(LicenseError::PublicKeyMissing)?,
        )?;
        let key_hash = sha256_hex(key.as_bytes());
        let document = PersistedLicense {
            schema_version: credential.schema_version,
            fingerprint: key_hash.chars().take(12).collect(),
            key_hash,
            license_id: credential.license_id,
            tier: credential.tier,
            features: credential.features,
            credential_format: credential.credential_format.to_owned(),
            token_id: credential.token_id,
            key_id: credential.key_id,
            created_at: credential.created_at,
            not_before: credential.not_before,
            activated_at: now,
            duration_value: credential.duration_value,
            duration_unit: credential.duration_unit,
            expires_at: credential.expires_at,
            updated_at: now,
            verification: credential.verification,
        };
        self.write_document(&document)?;
        *self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        let mut status = self.status_from_document(document, now)?;
        status.temporary_access_supported = self.temporary_access_supported();
        Ok(status)
    }

    /// Enable full access for the lifetime of the current debug daemon.
    /// This state is deliberately process-local and never written to disk.
    pub fn grant_temporary(&self) -> Result<LicenseStatus, LicenseError> {
        if !self.temporary_access_supported() {
            return Err(LicenseError::TemporaryGrantDisabled);
        }
        let _operation = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let now = now_seconds()?;
        if let Some(granted_at) = *self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Ok(development_session_status(granted_at));
        }
        // Development access does not create or lock authorization storage.
        // An already readable signed license remains authoritative.
        match self.read_status(now) {
            Ok(current) if current.valid => {
                return Err(LicenseError::TemporaryGrantWouldReplaceActiveLicense);
            }
            Ok(_) | Err(_) => {}
        }
        *self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(now);
        Ok(development_session_status(now))
    }

    pub fn clear(&self) -> Result<LicenseStatus, LicenseError> {
        let _operation = self
            .operation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let had_development_session = self
            .development_session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .is_some();
        if had_development_session {
            return self.storage_status();
        }
        match fs::symlink_metadata(&self.path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let mut status = LicenseStatus::unconfigured();
                status.temporary_access_supported = self.temporary_access_supported();
                return Ok(status);
            }
            Ok(_) => {}
            Err(source) => return Err(LicenseError::io("inspect", &self.path, source)),
        }
        let _lock = self.lock()?;
        match fs::symlink_metadata(&self.path) {
            Ok(metadata) => {
                validate_regular_file(&self.path, &metadata)?;
                fs::remove_file(&self.path)
                    .map_err(|source| LicenseError::io("remove", &self.path, source))?;
                sync_parent(&self.path)?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(LicenseError::io("inspect", &self.path, source)),
        }
        let mut status = LicenseStatus::unconfigured();
        status.temporary_access_supported = self.temporary_access_supported();
        Ok(status)
    }

    fn read_status(&self, now: f64) -> Result<LicenseStatus, LicenseError> {
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(LicenseStatus::unconfigured());
            }
            Err(source) => return Err(LicenseError::io("inspect", &self.path, source)),
        };
        validate_regular_file(&self.path, &metadata)?;
        let bytes =
            fs::read(&self.path).map_err(|source| LicenseError::io("read", &self.path, source))?;
        let value = serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|source| {
            LicenseError::InvalidDocument {
                path: self.path.clone(),
                source,
            }
        })?;
        if value.get("schema_version").is_none() {
            let legacy = serde_json::from_value::<LegacyLicense>(value).map_err(|source| {
                LicenseError::InvalidDocument {
                    path: self.path.clone(),
                    source,
                }
            })?;
            return Ok(legacy.into_status());
        }
        let document = serde_json::from_value::<PersistedLicense>(value).map_err(|source| {
            LicenseError::InvalidDocument {
                path: self.path.clone(),
                source,
            }
        })?;
        self.status_from_document(document, now)
    }

    fn status_from_document(
        &self,
        document: PersistedLicense,
        now: f64,
    ) -> Result<LicenseStatus, LicenseError> {
        let public_key_pem = self
            .policy
            .public_key_pem
            .as_deref()
            .ok_or(LicenseError::PublicKeyMissing)?;
        let (credential_format, token_id, key_id, not_before, expires_at) =
            match (document.schema_version, &document.verification) {
                (2, Verification::RsaPkcs1v15Sha256 { payload, signature }) => {
                    let key = format!("NS1.{payload}.{signature}");
                    let (claims, _, _) = verify_legacy_signed_key(&key, public_key_pem)?;
                    let expires_at = expiration(
                        claims.created_at,
                        claims.duration.value,
                        &claims.duration.unit,
                    )?;
                    if document.key_hash != sha256_hex(key.as_bytes())
                        || document.license_id != claims.license_id
                        || document.tier != claims.tier
                        || document.features != claims.features
                        || document.created_at != claims.created_at
                        || document.duration_value != claims.duration.value
                        || document.duration_unit != claims.duration.unit
                    {
                        return Err(LicenseError::VerificationFailed(
                            "persisted license claims do not match the signed payload".to_owned(),
                        ));
                    }
                    if !document.credential_format.is_empty()
                        && document.credential_format != LEGACY_CREDENTIAL_FORMAT
                    {
                        return Err(LicenseError::VerificationFailed(
                            "persisted license credential format was modified".to_owned(),
                        ));
                    }
                    (
                        LEGACY_CREDENTIAL_FORMAT,
                        String::new(),
                        String::new(),
                        None,
                        expires_at,
                    )
                }
                (
                    3,
                    Verification::JwtRs256 {
                        header,
                        payload,
                        signature,
                    },
                ) => {
                    let key = format!("{header}.{payload}.{signature}");
                    let claims =
                        verify_license_jwt(&key, public_key_pem).map_err(map_license_jwt_error)?;
                    let created_at = claims.issued_at as f64;
                    let expires_at = claims.expires_at as f64;
                    let not_before = claims.not_before.map(|value| value as f64);
                    let duration_value = claims.expires_at - claims.issued_at;
                    validate_license_features(&claims.features)?;
                    if document.key_hash != sha256_hex(key.as_bytes())
                        || document.license_id != claims.license_id
                        || document.tier != claims.tier
                        || document.features != claims.features
                        || document.credential_format != JWT_CREDENTIAL_FORMAT
                        || document.token_id != claims.token_id
                        || document.key_id != claims.key_id
                        || document.created_at != created_at
                        || document.not_before != not_before
                        || document.duration_value != duration_value
                        || document.duration_unit != "seconds"
                    {
                        return Err(LicenseError::VerificationFailed(
                            "persisted license claims do not match the signed JWT".to_owned(),
                        ));
                    }
                    (
                        JWT_CREDENTIAL_FORMAT,
                        claims.token_id,
                        claims.key_id,
                        not_before,
                        expires_at,
                    )
                }
                (schema_version, _) if schema_version != 2 && schema_version != 3 => {
                    return Err(LicenseError::UnsupportedSchema(schema_version));
                }
                _ => {
                    return Err(LicenseError::VerificationFailed(
                        "persisted license schema and credential format do not match".to_owned(),
                    ));
                }
            };
        let expected_fingerprint: String = document.key_hash.chars().take(12).collect();
        if document.fingerprint != expected_fingerprint || document.expires_at != expires_at {
            return Err(LicenseError::VerificationFailed(
                "persisted fingerprint or expiration was modified".to_owned(),
            ));
        }
        let not_active_yet = document.created_at > now + LICENSE_CLOCK_SKEW_SECONDS
            || not_before.is_some_and(|value| value > now + LICENSE_CLOCK_SKEW_SECONDS);
        let valid = !not_active_yet && expires_at > now;
        Ok(LicenseStatus {
            configured: true,
            valid,
            temporary_access_supported: self.temporary_access_supported(),
            fingerprint: document.fingerprint,
            tier: document.tier,
            features: document.features,
            license_id: document.license_id,
            credential_format: credential_format.to_owned(),
            token_id,
            key_id,
            created_at: Some(document.created_at),
            not_before,
            activated_at: Some(document.activated_at),
            expires_at: Some(expires_at),
            duration_value: Some(document.duration_value),
            duration_unit: document.duration_unit,
            updated_at: Some(document.updated_at),
            message: if valid {
                String::new()
            } else if not_active_yet {
                "license is not active yet".to_owned()
            } else {
                "license expired".to_owned()
            },
        })
    }

    fn write_document(&self, document: &PersistedLicense) -> Result<(), LicenseError> {
        if let Ok(metadata) = fs::symlink_metadata(&self.path) {
            validate_regular_file(&self.path, &metadata)?;
        }
        let parent = parent_directory(&self.path);
        fs::create_dir_all(parent)
            .map_err(|source| LicenseError::io("create directory", parent, source))?;
        let bytes = serde_json::to_vec_pretty(document).map_err(LicenseError::Serialize)?;
        let temporary = parent.join(format!(
            ".{}.tmp-{}-{}",
            self.path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("license"),
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(|source| LicenseError::io("create temporary", &temporary, source))?;
        let result = (|| {
            file.write_all(&bytes)
                .map_err(|source| LicenseError::io("write temporary", &temporary, source))?;
            file.write_all(b"\n")
                .map_err(|source| LicenseError::io("write temporary", &temporary, source))?;
            file.sync_all()
                .map_err(|source| LicenseError::io("sync temporary", &temporary, source))?;
            fs::rename(&temporary, &self.path)
                .map_err(|source| LicenseError::io("replace", &self.path, source))?;
            sync_parent(&self.path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn lock(&self) -> Result<LicenseLock, LicenseError> {
        let parent = parent_directory(&self.path);
        fs::create_dir_all(parent)
            .map_err(|source| LicenseError::io("create directory", parent, source))?;
        let lock_path = parent.join(format!(
            ".{}.lock",
            self.path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("license")
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_path)
            .map_err(|source| LicenseError::io("open lock", &lock_path, source))?;
        file.lock_exclusive()
            .map_err(|source| LicenseError::io("lock", &lock_path, source))?;
        Ok(LicenseLock(file))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedLicense {
    schema_version: u32,
    key_hash: String,
    fingerprint: String,
    license_id: String,
    tier: String,
    features: Vec<String>,
    #[serde(default)]
    credential_format: String,
    #[serde(default)]
    token_id: String,
    #[serde(default)]
    key_id: String,
    created_at: f64,
    #[serde(default)]
    not_before: Option<f64>,
    activated_at: f64,
    duration_value: u64,
    duration_unit: String,
    expires_at: f64,
    updated_at: f64,
    verification: Verification,
}

#[derive(Debug, Deserialize)]
struct LegacyLicense {
    #[serde(default)]
    key_hash: String,
    #[serde(default)]
    fingerprint: String,
    #[serde(default)]
    license_id: String,
    #[serde(default)]
    tier: String,
    #[serde(default)]
    features: Vec<String>,
    created_at: Option<f64>,
    activated_at: Option<f64>,
    expires_at: Option<f64>,
    duration_value: Option<u64>,
    #[serde(default)]
    duration_unit: String,
    updated_at: Option<f64>,
}

impl LegacyLicense {
    fn into_status(self) -> LicenseStatus {
        let configured = !self.key_hash.trim().is_empty();
        LicenseStatus {
            configured,
            valid: false,
            temporary_access_supported: false,
            fingerprint: self.fingerprint,
            tier: self.tier,
            features: self.features,
            license_id: self.license_id,
            credential_format: "legacy_document".to_owned(),
            token_id: String::new(),
            key_id: String::new(),
            created_at: self.created_at,
            not_before: None,
            activated_at: self.activated_at,
            expires_at: self.expires_at,
            duration_value: self.duration_value,
            duration_unit: self.duration_unit,
            updated_at: self.updated_at,
            message: if configured {
                "legacy Python license must be reactivated so Rust can verify its signed claims"
                    .to_owned()
            } else {
                String::new()
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Verification {
    RsaPkcs1v15Sha256 {
        payload: String,
        signature: String,
    },
    JwtRs256 {
        header: String,
        payload: String,
        signature: String,
    },
}

fn development_session_status(granted_at: f64) -> LicenseStatus {
    LicenseStatus {
        configured: true,
        valid: true,
        temporary_access_supported: true,
        fingerprint: String::new(),
        tier: TEMPORARY_LICENSE_TIER.to_owned(),
        features: ALL_FEATURES.iter().map(ToString::to_string).collect(),
        license_id: DEVELOPMENT_SESSION_ID.to_owned(),
        credential_format: DEBUG_CREDENTIAL_FORMAT.to_owned(),
        token_id: String::new(),
        key_id: String::new(),
        created_at: Some(granted_at),
        not_before: None,
        activated_at: Some(granted_at),
        expires_at: None,
        duration_value: None,
        duration_unit: "process".to_owned(),
        updated_at: Some(granted_at),
        message: "Debug development access is active for the current novasightd process".to_owned(),
    }
}

#[derive(Debug, Deserialize)]
struct SignedLicenseClaims {
    license_id: String,
    tier: String,
    created_at: f64,
    duration: SignedDuration,
    #[serde(default)]
    features: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SignedDuration {
    value: u64,
    unit: String,
}

struct VerifiedCredential {
    schema_version: u32,
    credential_format: &'static str,
    license_id: String,
    token_id: String,
    key_id: String,
    tier: String,
    features: Vec<String>,
    created_at: f64,
    not_before: Option<f64>,
    expires_at: f64,
    duration_value: u64,
    duration_unit: String,
    verification: Verification,
}

struct LicenseLock(File);

impl Drop for LicenseLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

fn expiration(created_at: f64, value: u64, unit: &str) -> Result<f64, LicenseError> {
    if value == 0 || !created_at.is_finite() || created_at < 0.0 {
        return Err(LicenseError::VerificationFailed(
            "license duration or creation time is invalid".to_owned(),
        ));
    }
    let multiplier = match unit.to_ascii_lowercase().as_str() {
        "second" | "seconds" => 1.0,
        "day" | "days" => 24.0 * 60.0 * 60.0,
        "month" | "months" => 30.0 * 24.0 * 60.0 * 60.0,
        "year" | "years" => SECONDS_PER_YEAR,
        _ => return Err(LicenseError::UnsupportedDuration(unit.to_owned())),
    };
    Ok(created_at + value as f64 * multiplier)
}

fn verify_license_credential(
    key: &str,
    public_key_pem: &str,
) -> Result<VerifiedCredential, LicenseError> {
    if key.starts_with("NS1.") {
        let (claims, payload, signature) = verify_legacy_signed_key(key, public_key_pem)?;
        validate_license_features(&claims.features)?;
        let expires_at = expiration(
            claims.created_at,
            claims.duration.value,
            &claims.duration.unit,
        )?;
        return Ok(VerifiedCredential {
            schema_version: 2,
            credential_format: LEGACY_CREDENTIAL_FORMAT,
            license_id: claims.license_id,
            token_id: String::new(),
            key_id: String::new(),
            tier: claims.tier,
            features: claims.features,
            created_at: claims.created_at,
            not_before: None,
            expires_at,
            duration_value: claims.duration.value,
            duration_unit: claims.duration.unit,
            verification: Verification::RsaPkcs1v15Sha256 { payload, signature },
        });
    }

    let claims = verify_license_jwt(key, public_key_pem).map_err(map_license_jwt_error)?;
    if claims.issued_at > MAX_UNIX_TIMESTAMP
        || claims.expires_at > MAX_UNIX_TIMESTAMP
        || claims
            .not_before
            .is_some_and(|value| value > MAX_UNIX_TIMESTAMP)
    {
        return Err(LicenseError::InvalidKey(
            "license JWT timestamp is outside the supported range".to_owned(),
        ));
    }
    validate_license_features(&claims.features)?;
    let mut parts = key.split('.');
    let header = parts.next().unwrap_or_default().to_owned();
    let payload = parts.next().unwrap_or_default().to_owned();
    let signature = parts.next().unwrap_or_default().to_owned();

    Ok(VerifiedCredential {
        schema_version: 3,
        credential_format: JWT_CREDENTIAL_FORMAT,
        license_id: claims.license_id,
        token_id: claims.token_id,
        key_id: claims.key_id,
        tier: claims.tier,
        features: claims.features,
        created_at: claims.issued_at as f64,
        not_before: claims.not_before.map(|value| value as f64),
        expires_at: claims.expires_at as f64,
        duration_value: claims.expires_at - claims.issued_at,
        duration_unit: "seconds".to_owned(),
        verification: Verification::JwtRs256 {
            header,
            payload,
            signature,
        },
    })
}

fn verify_legacy_signed_key(
    key: &str,
    public_key_pem: &str,
) -> Result<(SignedLicenseClaims, String, String), LicenseError> {
    let mut parts = key.split('.');
    if parts.next() != Some("NS1") {
        return Err(LicenseError::InvalidKey(
            "signed license must use the NS1 format".to_owned(),
        ));
    }
    let payload = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| LicenseError::InvalidKey("signed license payload is missing".to_owned()))?;
    let signature = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| {
            LicenseError::InvalidKey("signed license signature is missing".to_owned())
        })?;
    if parts.next().is_some() {
        return Err(LicenseError::InvalidKey(
            "signed license has unexpected segments".to_owned(),
        ));
    }
    let payload_bytes = decode_base64url(payload, "payload")?;
    let signature_bytes = decode_base64url(signature, "signature")?;
    let public_key = parse_public_key(public_key_pem)?;
    let signature = Signature::try_from(signature_bytes.as_slice()).map_err(|error| {
        LicenseError::InvalidKey(format!("invalid RSA signature length: {error}"))
    })?;
    VerifyingKey::<Sha256>::new(public_key)
        .verify(&payload_bytes, &signature)
        .map_err(|_| LicenseError::VerificationFailed("RSA signature is invalid".to_owned()))?;
    let claims = serde_json::from_slice::<SignedLicenseClaims>(&payload_bytes)
        .map_err(|error| LicenseError::InvalidKey(format!("invalid signed claims: {error}")))?;
    if claims.license_id.trim().is_empty() || claims.tier.trim().is_empty() {
        return Err(LicenseError::InvalidKey(
            "signed license_id and tier must not be empty".to_owned(),
        ));
    }
    expiration(
        claims.created_at,
        claims.duration.value,
        &claims.duration.unit,
    )?;
    Ok((
        claims,
        payload.to_owned(),
        key.rsplit_once('.').unwrap().1.to_owned(),
    ))
}

fn validate_license_features(features: &[String]) -> Result<(), LicenseError> {
    let mut seen = std::collections::HashSet::with_capacity(features.len());
    for feature in features {
        if !ALL_FEATURES.contains(&feature.as_str()) {
            return Err(LicenseError::InvalidKey(format!(
                "license contains unsupported feature {feature:?}"
            )));
        }
        if !seen.insert(feature) {
            return Err(LicenseError::InvalidKey(format!(
                "license contains duplicate feature {feature:?}"
            )));
        }
    }
    Ok(())
}

fn map_license_jwt_error(error: jwt::LicenseJwtError) -> LicenseError {
    match error {
        jwt::LicenseJwtError::InvalidPublicKey(message) => LicenseError::PublicKeyInvalid(message),
        jwt::LicenseJwtError::InvalidToken(message)
        | jwt::LicenseJwtError::InvalidClaims(message) => LicenseError::InvalidKey(message),
        jwt::LicenseJwtError::VerificationFailed(message) => {
            LicenseError::VerificationFailed(message)
        }
    }
}

fn parse_public_key(public_key_pem: &str) -> Result<RsaPublicKey, LicenseError> {
    Ok(match RsaPublicKey::from_public_key_pem(public_key_pem) {
        Ok(key) => key,
        Err(spki_error) => RsaPublicKey::from_pkcs1_pem(public_key_pem).map_err(|pkcs1_error| {
            LicenseError::PublicKeyInvalid(format!(
                "cannot parse RSA public key as SPKI ({spki_error}) or PKCS#1 ({pkcs1_error})"
            ))
        })?,
    })
}

fn decode_base64url(value: &str, field: &str) -> Result<Vec<u8>, LicenseError> {
    URL_SAFE_NO_PAD
        .decode(value)
        .or_else(|_| URL_SAFE.decode(value))
        .map_err(|error| LicenseError::InvalidKey(format!("invalid license {field}: {error}")))
}

fn now_seconds() -> Result<f64, LicenseError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .map_err(|_| LicenseError::ClockBeforeEpoch)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn validate_regular_file(path: &Path, metadata: &fs::Metadata) -> Result<(), LicenseError> {
    if !metadata.file_type().is_file() {
        return Err(LicenseError::UnsafePath(path.to_owned()));
    }
    if metadata.nlink() != 1 {
        return Err(LicenseError::UnsafeHardlinks {
            path: path.to_owned(),
            links: metadata.nlink(),
        });
    }
    Ok(())
}

fn sync_parent(path: &Path) -> Result<(), LicenseError> {
    let parent = parent_directory(path);
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| LicenseError::io("sync directory", parent, source))
}

#[derive(Debug, Error)]
pub enum LicenseError {
    #[error("invalid license key: {0}")]
    InvalidKey(String),
    #[error("temporary development access is unavailable in this build")]
    TemporaryGrantDisabled,
    #[error("a signed license is already active; development access was not enabled")]
    TemporaryGrantWouldReplaceActiveLicense,
    #[error("NOVASIGHT_LICENSE_PUBLIC_KEY is required for signed licenses")]
    PublicKeyMissing,
    #[error("configured license public key is invalid: {0}")]
    PublicKeyInvalid(String),
    #[error("license verification failed: {0}")]
    VerificationFailed(String),
    #[error("unsupported license duration unit: {0}")]
    UnsupportedDuration(String),
    #[error("unsupported license file schema version: {0}")]
    UnsupportedSchema(u32),
    #[error("system clock is before the Unix epoch")]
    ClockBeforeEpoch,
    #[error("license path is not a regular file: {}", .0.display())]
    UnsafePath(PathBuf),
    #[error("license path has {links} hard links: {}", path.display())]
    UnsafeHardlinks { path: PathBuf, links: u64 },
    #[error("failed to parse license document {}: {source}", path.display())]
    InvalidDocument {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to serialize license document: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to {operation} license path {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
}

impl LicenseError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidKey(_) => "LICENSE_KEY_INVALID",
            Self::TemporaryGrantDisabled => "LICENSE_TEMPORARY_DISABLED",
            Self::TemporaryGrantWouldReplaceActiveLicense => "LICENSE_ACTIVE_CREDENTIAL_PRESENT",
            Self::PublicKeyMissing => "LICENSE_PUBLIC_KEY_MISSING",
            Self::PublicKeyInvalid(_) => "LICENSE_PUBLIC_KEY_INVALID",
            Self::VerificationFailed(_) => "LICENSE_VERIFICATION_FAILED",
            Self::UnsupportedDuration(_) => "LICENSE_DURATION_INVALID",
            Self::UnsupportedSchema(_) => "LICENSE_SCHEMA_UNSUPPORTED",
            Self::ClockBeforeEpoch => "LICENSE_CLOCK_INVALID",
            Self::UnsafePath(_) | Self::UnsafeHardlinks { .. } => "LICENSE_PATH_UNSAFE",
            Self::InvalidDocument { .. } => "LICENSE_DOCUMENT_INVALID",
            Self::Serialize(_) | Self::Io { .. } => "LICENSE_STORAGE_FAILED",
        }
    }

    pub const fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidKey(_)
                | Self::TemporaryGrantDisabled
                | Self::TemporaryGrantWouldReplaceActiveLicense
                | Self::VerificationFailed(_)
                | Self::UnsupportedDuration(_)
        )
    }

    pub const fn is_configuration_error(&self) -> bool {
        matches!(self, Self::PublicKeyMissing | Self::PublicKeyInvalid(_))
    }

    fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_owned(),
            source,
        }
    }

    const fn is_invalid_license_state(&self) -> bool {
        matches!(
            self,
            Self::InvalidKey(_)
                | Self::TemporaryGrantDisabled
                | Self::VerificationFailed(_)
                | Self::UnsupportedDuration(_)
                | Self::UnsupportedSchema(_)
                | Self::InvalidDocument { .. }
        )
    }
}
