use std::env;
use std::path::{Path, PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_FFI").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }
    let source = source_directory();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let jetson =
        target_os == "linux" && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");
    let implementation = source.join(if jetson {
        "src/jetson.cpp"
    } else {
        "src/reference.cpp"
    });
    let header = source.join("include/novasight_tensorrt_runtime.h");
    for path in [&implementation, &header] {
        require_file(path, "NovaSight TensorRT ABI source");
        println!("cargo:rerun-if-changed={}", path.display());
    }

    if !jetson {
        cc::Build::new()
            .cargo_metadata(false)
            .cpp(true)
            .std("c++17")
            .include(source.join("include"))
            .file(implementation)
            .compile("novasight_tensorrt");
        let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
        println!("cargo:rustc-link-search=native={}", output.display());
        println!("cargo:rustc-link-lib=static=novasight_tensorrt");
        if target_os == "macos" {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target_os != "windows" {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        return;
    }

    let tensorrt_include = first_existing_directory(&[
        "/usr/include/aarch64-linux-gnu",
        "/usr/include",
        "/usr/local/cuda/include",
    ])
    .unwrap_or_else(|| panic!("TensorRT NvInfer.h include directory was not found"));
    require_file(&tensorrt_include.join("NvInfer.h"), "TensorRT headers");
    let cuda_include = first_file_parent(&[
        "/usr/local/cuda/include/cuda_runtime_api.h",
        "/usr/include/cuda_runtime_api.h",
    ])
    .unwrap_or_else(|| panic!("CUDA cuda_runtime_api.h was not found"));

    cc::Build::new()
        .cargo_metadata(false)
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-fPIC")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wpedantic")
        .include(source.join("include"))
        .include(tensorrt_include)
        .include(cuda_include)
        .file(implementation)
        .compile("novasight_tensorrt");

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=novasight_tensorrt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    for directory in [
        "/usr/lib/aarch64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/lib",
    ] {
        if Path::new(directory).is_dir() {
            println!("cargo:rustc-link-search=native={directory}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn source_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("TensorRT crate remains inside <repo>/rust/crates")
        .join("native/tensorrt-runtime")
}

fn first_existing_directory(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.join("NvInfer.h").is_file())
}

fn first_file_parent(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
        .and_then(|path| path.parent().map(Path::to_owned))
}

fn require_file(path: &Path, label: &str) {
    if !path.is_file() {
        panic!("{label} is missing: {}", path.display());
    }
}
