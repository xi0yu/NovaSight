use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, HeaderValue, header::COOKIE};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use novasight_store::license::LicenseStatus;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};

const COOKIE_NAME: &str = "novasight_session";
const SESSION_SECONDS: u64 = 60 * 60;
const REFRESH_BEFORE_SECONDS: u64 = 10 * 60;

#[derive(Clone)]
pub(crate) struct LicenseSession {
    secret: Arc<[u8; 32]>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionClaims {
    sub: String,
    exp: u64,
    license_fingerprint: String,
}

pub(crate) struct SessionCheck {
    pub refresh: bool,
}

impl LicenseSession {
    pub(crate) fn new() -> Self {
        let mut secret = [0_u8; 32];
        OsRng.fill_bytes(&mut secret);
        Self {
            secret: Arc::new(secret),
        }
    }

    pub(crate) fn issue_cookie(&self, status: &LicenseStatus) -> Result<HeaderValue, String> {
        if !status.configured || !status.valid {
            return Ok(Self::clear_cookie());
        }
        let now = unix_seconds()?;
        let license_remaining = status
            .expires_at
            .map(|expires_at| expires_at.max(now as f64) as u64 - now);
        let max_age = license_remaining
            .map(|remaining| remaining.min(SESSION_SECONDS))
            .unwrap_or(SESSION_SECONDS)
            .max(1);
        let claims = SessionClaims {
            sub: status.license_id.clone(),
            exp: now + max_age,
            license_fingerprint: status.fingerprint.clone(),
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(self.secret.as_ref()),
        )
        .map_err(|error| format!("failed to encode license session: {error}"))?;
        HeaderValue::from_str(&format!(
            "{COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}"
        ))
        .map_err(|error| format!("failed to create license session cookie: {error}"))
    }

    pub(crate) fn verify(
        &self,
        headers: &HeaderMap,
        status: &LicenseStatus,
    ) -> Result<SessionCheck, String> {
        let token = cookie_value(headers, COOKIE_NAME)
            .ok_or_else(|| "license session cookie is missing".to_owned())?;
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_required_spec_claims(&["exp", "sub"]);
        validation.leeway = 5;
        let claims = decode::<SessionClaims>(
            &token,
            &DecodingKey::from_secret(self.secret.as_ref()),
            &validation,
        )
        .map_err(|error| format!("license session is invalid: {error}"))?
        .claims;
        if claims.sub != status.license_id || claims.license_fingerprint != status.fingerprint {
            return Err("license session does not match the active license".to_owned());
        }
        let now = unix_seconds()?;
        Ok(SessionCheck {
            refresh: claims.exp.saturating_sub(now) <= REFRESH_BEFORE_SECONDS,
        })
    }

    pub(crate) const fn clear_cookie() -> HeaderValue {
        HeaderValue::from_static("novasight_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
    }
}

fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get_all(COOKIE).iter().find_map(|header| {
        header.to_str().ok()?.split(';').find_map(|cookie| {
            let (cookie_name, value) = cookie.trim().split_once('=')?;
            (cookie_name == name && !value.is_empty()).then(|| value.to_owned())
        })
    })
}

fn unix_seconds() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| "system clock is before the Unix epoch".to_owned())
}
