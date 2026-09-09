//! Formal NovaSight license JWT contract.
//!
//! The external issuer signs `RS256` tokens for issuer
//! `novasight-license` and audience `novasightd`. Registered claims are
//! `sub`, `iat`, optional `nbf`, `exp`, and `jti`; private claims are `tier`
//! and `features`. The daemon only contains the public verification key.

use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use thiserror::Error;

pub const LICENSE_JWT_ISSUER: &str = "novasight-license";
pub const LICENSE_JWT_AUDIENCE: &str = "novasightd";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedLicenseJwt {
    pub license_id: String,
    pub token_id: String,
    pub key_id: String,
    pub tier: String,
    pub features: Vec<String>,
    pub issued_at: u64,
    pub not_before: Option<u64>,
    pub expires_at: u64,
}

#[derive(Debug, Deserialize)]
struct LicenseJwtClaims {
    iss: String,
    sub: String,
    aud: Audience,
    exp: u64,
    iat: u64,
    #[serde(default)]
    nbf: Option<u64>,
    jti: String,
    tier: String,
    #[serde(default)]
    features: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Audience {
    One(String),
    Many(Vec<String>),
}

impl Audience {
    fn contains(&self, expected: &str) -> bool {
        match self {
            Self::One(value) => value == expected,
            Self::Many(values) => values.iter().any(|value| value == expected),
        }
    }
}

/// Verify a formal NovaSight license JWT without treating browser-side parsing
/// as authorization. Expiry and not-before are returned to the repository so
/// an expired persisted credential can still produce an informative status.
pub fn verify_license_jwt(
    token: &str,
    public_key_pem: &str,
) -> Result<VerifiedLicenseJwt, LicenseJwtError> {
    if token.split('.').count() != 3 {
        return Err(LicenseJwtError::InvalidToken(
            "JWT compact serialization must contain three segments".to_owned(),
        ));
    }

    let header =
        decode_header(token).map_err(|error| LicenseJwtError::InvalidToken(error.to_string()))?;
    if header.alg != Algorithm::RS256 {
        return Err(LicenseJwtError::InvalidToken(format!(
            "license JWT must use RS256, got {:?}",
            header.alg
        )));
    }
    if header
        .typ
        .as_deref()
        .is_some_and(|value| !value.eq_ignore_ascii_case("JWT"))
    {
        return Err(LicenseJwtError::InvalidToken(
            "license JWT typ must be JWT when present".to_owned(),
        ));
    }

    let decoding_key = DecodingKey::from_rsa_pem(public_key_pem.as_bytes())
        .map_err(|error| LicenseJwtError::InvalidPublicKey(error.to_string()))?;
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_required_spec_claims(&["exp", "iss", "aud", "sub"]);
    validation.set_issuer(&[LICENSE_JWT_ISSUER]);
    validation.set_audience(&[LICENSE_JWT_AUDIENCE]);
    // The repository evaluates time validity after signature and persisted
    // claim verification so it can report "expired" instead of losing all
    // trustworthy metadata behind a generic decoding error.
    validation.validate_exp = false;
    validation.validate_nbf = false;

    let claims = decode::<LicenseJwtClaims>(token, &decoding_key, &validation)
        .map_err(|error| LicenseJwtError::VerificationFailed(error.to_string()))?
        .claims;

    if claims.iss != LICENSE_JWT_ISSUER || !claims.aud.contains(LICENSE_JWT_AUDIENCE) {
        return Err(LicenseJwtError::InvalidClaims(
            "license JWT issuer or audience is invalid".to_owned(),
        ));
    }
    if claims.sub.trim().is_empty() || claims.jti.trim().is_empty() || claims.tier.trim().is_empty()
    {
        return Err(LicenseJwtError::InvalidClaims(
            "license JWT sub, jti, and tier must not be empty".to_owned(),
        ));
    }
    if claims.iat >= claims.exp {
        return Err(LicenseJwtError::InvalidClaims(
            "license JWT exp must be later than iat".to_owned(),
        ));
    }
    if claims
        .nbf
        .is_some_and(|not_before| not_before >= claims.exp)
    {
        return Err(LicenseJwtError::InvalidClaims(
            "license JWT nbf must be earlier than exp".to_owned(),
        ));
    }
    if header
        .kid
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(LicenseJwtError::InvalidClaims(
            "license JWT kid must not be empty when present".to_owned(),
        ));
    }

    Ok(VerifiedLicenseJwt {
        license_id: claims.sub,
        token_id: claims.jti,
        key_id: header.kid.unwrap_or_default(),
        tier: claims.tier,
        features: claims.features,
        issued_at: claims.iat,
        not_before: claims.nbf,
        expires_at: claims.exp,
    })
}

#[derive(Debug, Error)]
pub enum LicenseJwtError {
    #[error("configured JWT public key is invalid: {0}")]
    InvalidPublicKey(String),
    #[error("invalid license JWT: {0}")]
    InvalidToken(String),
    #[error("invalid license JWT claims: {0}")]
    InvalidClaims(String),
    #[error("license JWT verification failed: {0}")]
    VerificationFailed(String),
}
