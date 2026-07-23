use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEEPSTREAM");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_BUILD_REVISION");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_BUILD_DIRTY");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo provides manifest dir");
    let repository = Path::new(&manifest).join("../../..");
    let revision = match std::env::var("NOVASIGHT_BUILD_REVISION") {
        Ok(value) => {
            assert!(
                valid_revision(&value),
                "NOVASIGHT_BUILD_REVISION must be a full Git SHA-1"
            );
            value
        }
        Err(_) => git_output(&repository, &["rev-parse", "HEAD"])
            .filter(|value| valid_revision(value))
            .unwrap_or_else(|| "unknown".to_owned()),
    };
    let dirty = match std::env::var("NOVASIGHT_BUILD_DIRTY") {
        Ok(value) => match value.as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => panic!("NOVASIGHT_BUILD_DIRTY must be true, false, 1, or 0"),
        },
        Err(_) => git_output(
            &repository,
            &["status", "--porcelain", "--untracked-files=no"],
        )
        .is_none_or(|value| !value.is_empty()),
    };
    let target = std::env::var("TARGET").expect("Cargo provides target triple");
    let profile = std::env::var("PROFILE").expect("Cargo provides build profile");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_REVISION={revision}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_TARGET={target}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_PROFILE={profile}");
    if target.contains("linux") && std::env::var_os("CARGO_FEATURE_DEEPSTREAM").is_some() {
        // Production layout: /opt/novasight/bin/novasightd and
        // /opt/novasight/lib/libnovasight_deepstream_bridge.so.
        println!("cargo:rustc-link-arg-bin=novasightd=-Wl,-rpath,$ORIGIN/../lib");
    }
}

fn git_output(repository: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
