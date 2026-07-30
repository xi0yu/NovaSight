use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEEPSTREAM");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_BUILD_REVISION");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_BUILD_DIRTY");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_DEEPSTREAM_ROOT");
    println!("cargo:rerun-if-env-changed=NOVASIGHT_CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CXX");

    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo provides manifest dir");
    let repository = Path::new(&manifest).join("../..");
    emit_build_identity(&repository);

    if env::var_os("CARGO_FEATURE_DEEPSTREAM").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && env::var_os("DOCS_RS").is_none()
    {
        compile_deepstream_parser(&repository);
    }
}

fn emit_build_identity(repository: &Path) {
    let revision = match env::var("NOVASIGHT_BUILD_REVISION") {
        Ok(value) => {
            assert!(
                valid_revision(&value),
                "NOVASIGHT_BUILD_REVISION must be a full Git SHA-1"
            );
            value
        }
        Err(_) => git_output(repository, &["rev-parse", "HEAD"])
            .filter(|value| valid_revision(value))
            .unwrap_or_else(|| "unknown".to_owned()),
    };
    let dirty = match env::var("NOVASIGHT_BUILD_DIRTY") {
        Ok(value) => match value.as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => panic!("NOVASIGHT_BUILD_DIRTY must be true, false, 1, or 0"),
        },
        Err(_) => git_output(
            repository,
            &["status", "--porcelain", "--untracked-files=no"],
        )
        .is_none_or(|value| !value.is_empty()),
    };
    let target = env::var("TARGET").expect("Cargo provides target triple");
    let profile = env::var("PROFILE").expect("Cargo provides build profile");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_REVISION={revision}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_TARGET={target}");
    println!("cargo:rustc-env=NOVASIGHT_BUILD_PROFILE={profile}");
}

fn compile_deepstream_parser(repository: &Path) {
    let source = repository.join("native/deepstream-parser/src/novasight_parser.cpp");
    require_file(&source, "DeepStream parser source");
    println!("cargo:rerun-if-changed={}", source.display());

    let deepstream_include = deepstream_root().join("sources/includes");
    require_file(
        &deepstream_include.join("nvdsinfer_custom_impl.h"),
        "DeepStream inference headers",
    );
    let cuda_include = cuda_root().join("include");
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
