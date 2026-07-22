use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=NOVASIGHT_DEEPSTREAM_BRIDGE_DIR");
    if std::env::var_os("CARGO_FEATURE_FFI").is_none() || std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    let directory = std::env::var_os("NOVASIGHT_DEEPSTREAM_BRIDGE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_bridge_directory);
    println!("cargo:rustc-link-search=native={}", directory.display());
    let library = directory.join("libnovasight_deepstream_bridge.so");
    if !library.is_file() {
        println!(
            "cargo:warning=DeepStream bridge is not built at {}; run scripts/build_deepstream_bridge.sh or set NOVASIGHT_DEEPSTREAM_BRIDGE_DIR",
            library.display()
        );
    }
}

fn default_bridge_directory() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(3)
        .expect("bridge crate remains inside <repo>/rust/crates")
        .join("build/deepstream-bridge")
}
