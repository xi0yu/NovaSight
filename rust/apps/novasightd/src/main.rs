//! Canonical NovaSight daemon executable.
//!
//! The composition code lives in the `relink-server` library crate, but this
//! is the only server executable. Legacy Python remains the rollback path.

use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
struct BuildInfo {
    schema_version: u32,
    binary: &'static str,
    version: &'static str,
    source_revision: &'static str,
    source_dirty: bool,
    target: &'static str,
    profile: &'static str,
    features: Vec<&'static str>,
}

fn build_info() -> BuildInfo {
    let mut features = Vec::new();
    if cfg!(feature = "deepstream") {
        features.push("deepstream");
    }
    if cfg!(feature = "cuda-preprocess") {
        features.push("cuda-preprocess");
    }
    if cfg!(feature = "tensorrt") {
        features.push("tensorrt");
    }
    if cfg!(feature = "experimental-kmnet-native") {
        features.push("experimental-kmnet-native");
    }
    BuildInfo {
        schema_version: 1,
        binary: "novasightd",
        version: env!("CARGO_PKG_VERSION"),
        source_revision: env!("NOVASIGHT_BUILD_REVISION"),
        source_dirty: env!("NOVASIGHT_BUILD_DIRTY") == "true",
        target: env!("NOVASIGHT_BUILD_TARGET"),
        profile: env!("NOVASIGHT_BUILD_PROFILE"),
        features,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    if std::env::args_os().len() == 2
        && std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--build-info-json"))
    {
        println!(
            "{}",
            serde_json::to_string(&build_info()).expect("build identity must serialize")
        );
        return ExitCode::SUCCESS;
    }
    relink_server::entry().await
}
