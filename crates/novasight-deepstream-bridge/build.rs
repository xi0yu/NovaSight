use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=NOVASIGHT_DEEPSTREAM_ROOT");
    if env::var_os("CARGO_FEATURE_FFI").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!("the DeepStream FFI bridge is supported only for Linux targets");
    }

    let source = bridge_source_directory();
    let implementation = source.join("src/novasight_deepstream_bridge.cpp");
    let header = source.join("include/novasight_deepstream_bridge.h");
    for path in [&implementation, &header] {
        if !path.is_file() {
            panic!(
                "required DeepStream bridge source is missing: {}; sync the complete NovaSight checkout",
                path.display()
            );
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let deepstream = deepstream_root();
    let deepstream_include = deepstream.join("sources/includes");
    require_file(
        &deepstream_include.join("gstnvdsmeta.h"),
        "DeepStream headers",
    );
    require_file(
        &deepstream.join("lib/libnvdsgst_meta.so"),
        "DeepStream metadata runtime",
    );
    require_file(
        &deepstream.join("lib/libnvds_meta.so"),
        "DeepStream metadata runtime",
    );

    let gstreamer = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("gstreamer-1.0")
        .unwrap_or_else(|error| panic!("GStreamer development package is required: {error}"));

    let output_directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
    cc::Build::new()
        .cargo_metadata(false)
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wpedantic")
        .include(source.join("include"))
        .include(deepstream_include)
        .includes(&gstreamer.include_paths)
        .file(implementation)
        .compile("novasight_deepstream_bridge");

    println!(
        "cargo:rustc-link-search=native={}",
        output_directory.display()
    );
    println!("cargo:rustc-link-lib=static=novasight_deepstream_bridge");
    // `cargo_metadata(false)` keeps the native link contract explicit, so the
    // C++ runtime must be declared here as well. The bridge catches exceptions
    // at the C ABI boundary and therefore requires libstdc++ exception support.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    for path in gstreamer.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for library in gstreamer.libs {
        println!("cargo:rustc-link-lib={library}");
    }
    println!(
        "cargo:rustc-link-search=native={}",
        deepstream.join("lib").display()
    );
    println!("cargo:rustc-link-lib=nvdsgst_meta");
    println!("cargo:rustc-link-lib=nvds_meta");
}

fn bridge_source_directory() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .expect("bridge crate remains inside <repo>/crates")
        .join("native/deepstream-bridge")
}

fn deepstream_root() -> PathBuf {
    if let Some(configured) = env::var_os("NOVASIGHT_DEEPSTREAM_ROOT") {
        return PathBuf::from(configured);
    }
    let stable = PathBuf::from("/opt/nvidia/deepstream/deepstream");
    if stable.is_dir() {
        return stable;
    }
    let parent = Path::new("/opt/nvidia/deepstream");
    let mut candidates = fs::read_dir(parent)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("deepstream-"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop().unwrap_or_else(|| {
        panic!(
            "DeepStream SDK was not found under /opt/nvidia/deepstream; install DeepStream or set NOVASIGHT_DEEPSTREAM_ROOT for a non-standard installation"
        )
    })
}

fn require_file(path: &Path, label: &str) {
    if !path.is_file() {
        panic!("{label} not found at {}", path.display());
    }
}
