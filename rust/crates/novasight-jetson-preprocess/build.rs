use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=NOVASIGHT_JETSON_PREPROCESS_DIR");
    if std::env::var_os("CARGO_FEATURE_FFI").is_none() || std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    let directory = std::env::var_os("NOVASIGHT_JETSON_PREPROCESS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_library_directory);
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=dylib=novasight_preprocess");
    let suffix = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "dylib",
        Ok("windows") => "dll",
        _ => "so",
    };
    let library = directory.join(format!("libnovasight_preprocess.{suffix}"));
    if !library.is_file() {
        println!(
            "cargo:warning=Jetson preprocess library is not built at {}; run scripts/build_jetson_preprocess.sh on the target Jetson or set NOVASIGHT_JETSON_PREPROCESS_DIR",
            library.display()
        );
    }
}

fn default_library_directory() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(3)
        .expect("preprocess crate remains inside <repo>/rust/crates")
        .join("build/jetson-native")
}
