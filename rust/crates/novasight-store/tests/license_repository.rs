use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::license::{FileLicenseRepository, LicensePolicy};

const TEST_KEY: &str = "NOVASIGHT-TEST-MAX-ACCESS-2026";
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
fn activation_persists_no_plaintext_and_round_trips_verified_status() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    let policy = LicensePolicy::new(true, None);
    let repository = FileLicenseRepository::new(&path, policy.clone());

    let activated = repository.activate(TEST_KEY).unwrap();

    assert!(activated.configured);
    assert!(activated.valid);
    assert_eq!(activated.tier, "test_max");
    assert_eq!(activated.license_id, "test-max-access");
    assert!(activated.features.contains(&"hardware_control".to_owned()));
    assert!(activated.activated_at.is_some());
    assert!(activated.expires_at.is_some());
    let document = fs::read_to_string(&path).unwrap();
    assert!(!document.contains(TEST_KEY));
    assert!(document.contains("key_hash"));
    assert!(document.contains("built_in_test"));
    let python_compatible: serde_json::Value = serde_json::from_str(&document).unwrap();
    assert_eq!(
        python_compatible["fingerprint"].as_str(),
        Some(activated.fingerprint.as_str())
    );
    assert_eq!(
        python_compatible["expires_at"].as_f64(),
        activated.expires_at
    );

    let reopened = FileLicenseRepository::new(path, policy).status().unwrap();
    assert_eq!(reopened, activated);
}

#[test]
fn signed_license_is_reverified_after_restart_and_rejects_disk_claim_tampering() {
    let directory = TestDirectory::new();
    let path = directory.0.join("license.json");
    let policy = LicensePolicy::new(false, Some(PUBLIC_KEY.to_owned()));
    let repository = FileLicenseRepository::new(&path, policy.clone());

    let activated = repository.activate(SIGNED_KEY).unwrap();

    assert!(activated.valid);
    assert_eq!(activated.license_id, "commercial-001");
    assert_eq!(activated.tier, "pro");
    assert_eq!(activated.features, ["runtime", "models"]);
    assert!(!fs::read_to_string(&path).unwrap().contains(SIGNED_KEY));
    assert_eq!(
        FileLicenseRepository::new(&path, policy.clone())
            .status()
            .unwrap(),
        activated
    );

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
