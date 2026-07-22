use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn help_exposes_the_production_daemon_contract() {
    let output = binary().arg("--help").output().expect("run --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: relink_server [OPTIONS]"));
    assert!(stdout.contains("--config <CONFIG>"));
    assert!(stdout.contains("--model-job-script <MODEL_JOB_SCRIPT>"));
    assert!(stdout.contains("--live-perception"));
    assert!(stdout.contains("/etc/novasight/novasight.yaml"));
}

#[test]
fn missing_config_keeps_the_stable_production_error() {
    let output = binary()
        .args(["--config", "/definitely/missing/novasight.yaml", "--check"])
        .output()
        .expect("run relink_server");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("CONFIG_NOT_FOUND"));
}

#[test]
fn check_runs_the_shared_config_and_model_helper_preflight() {
    let root = repository_root();
    let config = root.join("rust/config/novasightd.example.yaml");
    let helper = root.join("scripts/model_ingress_job.py");
    let output = binary()
        .arg("--config")
        .arg(config)
        .arg("--check")
        .arg("--model-job-script")
        .arg(helper)
        .arg("--model-job-workdir")
        .arg(&root)
        .output()
        .expect("run relink_server preflight");

    assert!(
        output.status.success(),
        "preflight failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("model_ingress_helper=ready"));
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_relink_server"))
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("resolve repository root")
}
