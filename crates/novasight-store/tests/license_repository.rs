use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use novasight_store::license::{FileLicenseRepository, LicensePolicy};
use rand::thread_rng;
use rsa::RsaPrivateKey;
use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};
use serde::Serialize;

const PUBLIC_KEY: &str = include_str!("../../../testdata/license-public.pem");
const SIGNED_KEY: &str = include_str!("../../../testdata/license-signed.key");
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-license-repository-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn development_access_is_process_local_and_never_written_to_the_license_file() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    let policy = LicensePolicy::new(true, None);
    let repository = FileLicenseRepository::new(&path, policy.clone());

    let activated = repository.grant_temporary().unwrap();

    assert!(activated.configured);
    assert!(activated.valid);
    assert_eq!(activated.tier, "temporary");
    assert_eq!(activated.credential_format, "debug_session");
    assert_eq!(activated.license_id, "debug-process-session");
    assert!(activated.features.contains(&"hardware_control".to_owned()));
    assert!(activated.activated_at.is_some());
    assert!(activated.expires_at.is_none());
    assert!(
        !path.exists(),
        "debug access must not create a license file"
    );
    assert!(
        !directory.0.join(".license.json.lock").exists(),
        "debug access must not create a license lock file"
    );

    let repeated = repository.grant_temporary().unwrap();
    assert_eq!(repeated, activated);

    let cleared = repository.clear().unwrap();
    assert!(!cleared.configured);
    assert!(!cleared.valid);
    assert!(
        !directory.0.join(".license.json.lock").exists(),
        "clearing process-local access must not create a license lock file"
    );

    let reopened = FileLicenseRepository::new(path, policy).status().unwrap();
    assert!(!reopened.configured);
    assert!(!reopened.valid);
    assert!(!directory.0.join(".license.json.lock").exists());
}

#[test]
fn signed_license_is_reverified_after_restart_and_rejects_disk_claim_tampering() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    let policy = LicensePolicy::new(true, Some(PUBLIC_KEY.to_owned()));
    let repository = FileLicenseRepository::new(&path, policy.clone());

    let activated = repository.activate(SIGNED_KEY).unwrap();

    assert!(activated.valid);
    assert_eq!(activated.license_id, "commercial-001");
    assert_eq!(activated.tier, "pro");
    assert_eq!(activated.credential_format, "legacy_ns1");
    assert_eq!(activated.features, ["runtime", "models"]);
    assert!(!fs::read_to_string(&path).unwrap().contains(SIGNED_KEY));
    assert_eq!(
        FileLicenseRepository::new(&path, policy.clone())
            .status()
            .unwrap(),
        activated
    );
    assert!(matches!(
        repository.grant_temporary(),
        Err(novasight_store::license::LicenseError::TemporaryGrantWouldReplaceActiveLicense)
    ));

    let mut document: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    document["tier"] = serde_json::Value::String("ultimate".to_owned());
    fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let status = FileLicenseRepository::new(path, policy).status().unwrap();
    assert!(status.configured);
    assert!(!status.valid);
    assert!(status.message.contains("do not match"));
}

#[test]
fn rs256_jwt_license_is_reverified_from_persisted_signed_segments() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    let private_key = RsaPrivateKey::new(&mut thread_rng(), 2048).unwrap();
    let private_pem = private_key.to_pkcs8_pem(LineEnding::LF).unwrap();
    let public_pem = private_key
        .to_public_key()
        .to_public_key_pem(LineEnding::LF)
        .unwrap();
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some("license-key-2026-01".to_owned());
    let token = encode(
        &header,
        &TestJwtClaims {
            iss: "novasight-license",
            sub: "commercial-jwt-001",
            aud: "novasightd",
            exp: 4_102_444_800,
            iat: 1_700_000_000,
            jti: "entitlement-001",
            tier: "pro",
            features: &["runtime", "models"],
        },
        &EncodingKey::from_rsa_pem(private_pem.as_bytes()).unwrap(),
    )
    .unwrap();
    let policy = LicensePolicy::new(false, Some(public_pem));
    let repository = FileLicenseRepository::new(&path, policy.clone());

    let activated = repository.activate(&token).unwrap();

    assert!(activated.valid);
    assert_eq!(activated.license_id, "commercial-jwt-001");
    assert_eq!(activated.credential_format, "jwt_rs256");
    assert_eq!(activated.token_id, "entitlement-001");
    assert_eq!(activated.key_id, "license-key-2026-01");
    assert_eq!(activated.duration_unit, "seconds");
    assert!(!fs::read_to_string(&path).unwrap().contains(&token));
    assert_eq!(
        FileLicenseRepository::new(&path, policy).status().unwrap(),
        activated
    );
}

#[derive(Serialize)]
struct TestJwtClaims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    exp: u64,
    iat: u64,
    jti: &'a str,
    tier: &'a str,
    features: &'a [&'a str],
}

#[test]
fn corrupt_document_is_reported_as_invalid_without_hiding_recovery_routes() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    fs::write(&path, "{not-json").unwrap();

    let status = FileLicenseRepository::new(path, LicensePolicy::new(false, None))
        .status()
        .unwrap();

    assert!(status.configured);
    assert!(!status.valid);
    assert!(status.message.contains("failed to parse license document"));
}

#[test]
fn legacy_python_license_is_readable_but_cannot_authorize_without_reactivation() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    fs::write(
        &path,
        r#"{
  "key_hash": "0123456789abcdef",
  "fingerprint": "0123456789ab",
  "license_id": "legacy-license",
  "tier": "pro",
  "features": ["runtime"],
  "created_at": 1783036800.0,
  "activated_at": 1783036801.0,
  "expires_at": 1814572800.0,
  "duration_value": 1,
  "duration_unit": "years",
  "updated_at": 1783036801.0
}"#,
    )
    .unwrap();

    let status = FileLicenseRepository::new(path, LicensePolicy::new(false, None))
        .status()
        .unwrap();

    assert!(status.configured);
    assert!(!status.valid);
    assert_eq!(status.license_id, "legacy-license");
    assert_eq!(status.tier, "pro");
    assert!(status.message.contains("reactivated"));
}
