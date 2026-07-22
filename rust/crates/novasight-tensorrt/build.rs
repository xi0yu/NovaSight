use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=NOVASIGHT_TENSORRT_RUNTIME_DIR");
    if std::env::var_os("CARGO_FEATURE_FFI").is_none() || std::env::var_os("DOCS_RS").is_some() {
        return;
    }
    let directory = std::env::var_os("NOVASIGHT_TENSORRT_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_library_directory);
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=dylib=novasight_tensorrt");
    let suffix = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "dylib",
        Ok("windows") => "dll",
        _ => "so",
    };
    let library = directory.join(format!("libnovasight_tensorrt.{suffix}"));
    if !library.is_file() {
        println!(
            "cargo:warning=TensorRT runtime library is not built at {}; build native/tensorrt-runtime on Jetson or set NOVASIGHT_TENSORRT_RUNTIME_DIR",
            library.display()
        );
    }
}

fn default_library_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("TensorRT crate remains inside <repo>/rust/crates")
        .join("build/jetson-native")
}
