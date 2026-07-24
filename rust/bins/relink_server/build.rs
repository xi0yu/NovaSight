use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=NOVASIGHT_DEEPSTREAM_ROOT");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CXX");
    if env::var_os("CARGO_FEATURE_DEEPSTREAM").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!("the DeepStream parser is supported only for Linux targets");
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("relink-server remains inside <repo>/rust/bins");
    let source = repository.join("native/deepstream-parser/src/novasight_parser.cpp");
    if !source.is_file() {
        panic!(
            "required DeepStream parser source is missing: {}; sync the complete NovaSight checkout",
            source.display()
        );
    }
    println!("cargo:rerun-if-changed={}", source.display());

    let deepstream = deepstream_root();
    let deepstream_include = deepstream.join("sources/includes");
    require_file(
        &deepstream_include.join("nvdsinfer_custom_impl.h"),
        "DeepStream inference headers",
    );
    let cuda = cuda_root();
    let cuda_include = cuda.join("include");
    require_file(
        &cuda_include.join("cuda_runtime_api.h"),
        "CUDA runtime headers",
    );

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
        .join("libnovasight_parser.so");
    let compiler = env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let result = Command::new(&compiler)
        .arg("-std=c++17")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O2")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Wpedantic")
        .arg(format!("-I{}", deepstream_include.display()))
        .arg(format!("-I{}", cuda_include.display()))
        .arg(&source)
        .arg("-Wl,-soname,libnovasight_parser.so")
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to execute the C++ compiler {}: {error}",
                PathBuf::from(&compiler).display()
            )
        });
    if !result.status.success() {
        panic!(
            "DeepStream parser compilation failed:\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    println!(
        "cargo:rustc-env=NOVASIGHT_DEEPSTREAM_PARSER_LIBRARY={}",
        output.display()
    );
}

fn deepstream_root() -> PathBuf {
    discover_root(
        "NOVASIGHT_DEEPSTREAM_ROOT",
        Path::new("/opt/nvidia/deepstream/deepstream"),
        Path::new("/opt/nvidia/deepstream"),
        "deepstream-",
        "DeepStream SDK",
    )
}

fn cuda_root() -> PathBuf {
    discover_root(
        "NOVASIGHT_CUDA_ROOT",
        Path::new("/usr/local/cuda"),
        Path::new("/usr/local"),
        "cuda-",
        "CUDA Toolkit",
    )
}

fn discover_root(
    variable: &str,
    stable: &Path,
    parent: &Path,
    prefix: &str,
    label: &str,
) -> PathBuf {
    if let Some(configured) = env::var_os(variable) {
        return PathBuf::from(configured);
    }
    if stable.is_dir() {
        return stable.to_owned();
    }
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
                    .is_some_and(|name| name.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop().unwrap_or_else(|| {
        panic!(
            "{label} was not found under {}; install it or set {variable} for a non-standard installation",
            parent.display()
        )
    })
}

fn require_file(path: &Path, label: &str) {
    if !path.is_file() {
        panic!("{label} not found at {}", path.display());
    }
}
