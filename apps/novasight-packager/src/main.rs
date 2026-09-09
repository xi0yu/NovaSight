use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use serde::Deserialize;

const DEFAULT_OUTPUT: &str = "out/package/NovaSight";
const READY_FILE: &str = "run/ready.json";
const CONTROL_SOCKET: &str = "run/novasightd.sock";
const RUST_PACKAGES: [&str; 4] = ["novasight", "novasight-web", "novasightctl", "novasightd"];

#[derive(Parser, Debug)]
#[command(
    name = "novasight-packager",
    about = "Build a NovaSight portable package"
)]
struct Args {
    /// Package build profile.
    #[arg(long, value_enum, default_value_t = PackageProfile::Debug)]
    profile: PackageProfile,

    /// Output package directory. It must stay below the workspace out/ tree.
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum PackageProfile {
    /// Developer package with debug Rust binaries.
    Debug,
    /// Optimized package with release Rust binaries.
    Release,
}

#[derive(Clone, Debug)]
struct PackageLayout {
    root: PathBuf,
    bin: PathBuf,
    web: PathBuf,
    data: PathBuf,
    logs: PathBuf,
    run: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ReadyDocument {
    address: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("NOVASIGHT_PACKAGE_FAILED: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let workspace = find_workspace_root(&std::env::current_dir().context("read current dir")?)?;
    let output = resolve_output_path(&workspace, &args.output)?;
    build_artifacts(&workspace, args.profile)?;
    assemble_package(&workspace, args.profile, &output)?;
    validate_package(&output)?;
    println!("{}", output.display());
    Ok(())
}

fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut cursor = start.to_owned();
    loop {
        if cursor.join("Cargo.toml").is_file()
            && cursor.join("apps").is_dir()
            && cursor.join("web").is_dir()
        {
            return Ok(cursor);
        }
        if !cursor.pop() {
            bail!(
                "could not find NovaSight workspace root from {}",
                start.display()
            );
        }
    }
}

fn resolve_output_path(workspace: &Path, output: &Path) -> Result<PathBuf> {
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        workspace.join(output)
    };
    if output
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!("package output cannot contain '..': {}", output.display());
    }
    let out_root = workspace.join("out");
    if !output.starts_with(&out_root) {
        bail!(
            "package output must stay below {}; got {}",
            out_root.display(),
            output.display()
        );
    }
    Ok(output)
}

fn build_artifacts(workspace: &Path, profile: PackageProfile) -> Result<()> {
    build_rust_artifacts(workspace, profile)?;
    run_command(
        workspace,
        "pnpm",
        ["--dir", "web", "build"],
        &[("CI", "true")],
    )
}

fn build_rust_artifacts(workspace: &Path, profile: PackageProfile) -> Result<()> {
    run_command(workspace, "cargo", rust_build_args(profile), &[])
}

fn rust_build_args(profile: PackageProfile) -> Vec<&'static str> {
    let mut args = vec!["build", "--locked"];
    for package in RUST_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    if profile == PackageProfile::Release {
        args.push("--release");
    }
    args
}

fn run_command<I, S>(cwd: &Path, program: &str, args: I, envs: &[(&str, &str)]) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(program);
    command.args(args).current_dir(cwd);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command.status().with_context(|| format!("run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}

fn assemble_package(workspace: &Path, profile: PackageProfile, output: &Path) -> Result<()> {
    prepare_output_for_rebuild(output)?;
    if output.exists() {
        fs::remove_dir_all(output)
            .with_context(|| format!("remove previous package {}", output.display()))?;
    }

    let layout = PackageLayout::new(output);
    fs::create_dir_all(&layout.bin).with_context(|| format!("create {}", layout.bin.display()))?;
    fs::create_dir_all(&layout.web).with_context(|| format!("create {}", layout.web.display()))?;
    fs::create_dir_all(layout.data.join("models"))
        .with_context(|| format!("create {}", layout.data.join("models").display()))?;
    fs::create_dir_all(layout.data.join("runtime/deepstream")).with_context(|| {
        format!(
            "create {}",
            layout.data.join("runtime/deepstream").display()
        )
    })?;
    fs::create_dir_all(&layout.logs)
        .with_context(|| format!("create {}", layout.logs.display()))?;
    fs::create_dir_all(&layout.run).with_context(|| format!("create {}", layout.run.display()))?;

    let artifact_dir = workspace.join("out/cargo").join(profile.artifact_dir());
    copy_executable(
        &artifact_dir.join(executable_name("novasight")),
        &layout.root.join("NovaSight"),
    )?;
    copy_executable(
        &artifact_dir.join(executable_name("novasightd")),
        &layout.bin.join(executable_name("novasightd")),
    )?;
    copy_executable(
        &artifact_dir.join(executable_name("novasight-web")),
        &layout.bin.join(executable_name("novasight-web")),
    )?;
    copy_executable(
        &artifact_dir.join(executable_name("novasightctl")),
        &layout.bin.join(executable_name("novasightctl")),
    )?;
    copy_runtime_web_tree(&workspace.join("out/web"), &layout.web)?;
    copy_model_asset_tree_if_present(&workspace.join("data/models"), &layout.data.join("models"))?;
    copy_file(
        &workspace.join("deploy/novasight.production.yaml"),
        &layout.data.join("novasight.yaml"),
    )?;
    copy_file(
        &workspace.join("deploy/deepstream-tracker-iou.yml"),
        &layout
            .data
            .join("runtime/deepstream/deepstream-tracker-iou.yml"),
    )?;
    copy_file(
        &workspace.join("USER_MANUAL.md"),
        &layout.root.join("USER_MANUAL.md"),
    )?;
    fs::write(
        layout.root.join("README-USER.txt"),
        "Run ./NovaSight from this directory. NovaSight listens on 0.0.0.0:7351; use the printed authenticated LAN URL from another computer on the same network. The browser exchanges its one-time URL fragment for an HttpOnly operator session. Press Ctrl+C in the launcher terminal to stop NovaSight. Open USER_MANUAL.md for the user guide.\n",
    )
    .with_context(|| format!("write {}", layout.root.join("README-USER.txt").display()))?;
    Ok(())
}

fn prepare_output_for_rebuild(output: &Path) -> Result<()> {
    prepare_output_for_rebuild_with(output, health_check, control_socket_is_live)
}

fn prepare_output_for_rebuild_with(
    output: &Path,
    mut http_ready_is_live: impl FnMut(&str) -> bool,
    mut control_socket_is_live: impl FnMut(&Path) -> bool,
) -> Result<()> {
    let ready_file = output.join(READY_FILE);
    let control_socket = output.join(CONTROL_SOCKET);
    let ready_exists = ready_file.exists();
    let socket_exists = control_socket.exists();
    if !ready_exists && !socket_exists {
        return Ok(());
    }
    let ready_is_live = ready_exists
        && read_ready_file(&ready_file)
            .ok()
            .is_some_and(|ready| http_ready_is_live(&ready.address));
    let control_is_live = socket_exists && control_socket_is_live(&control_socket);
    if ready_is_live || control_is_live {
        bail!(
            "{} appears to be running; press Ctrl+C in the NovaSight launcher terminal before rebuilding the package",
            output.display(),
        );
    }
    if ready_exists {
        fs::remove_file(&ready_file)
            .with_context(|| format!("remove stale {}", ready_file.display()))?;
    }
    if socket_exists {
        fs::remove_file(&control_socket)
            .with_context(|| format!("remove stale {}", control_socket.display()))?;
    }
    eprintln!(
        "NOVASIGHT_PACKAGE_STALE_RUN_STATE: removed stale run markers below {}",
        output.display()
    );
    Ok(())
}

fn read_ready_file(path: &Path) -> Result<ReadyDocument> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parse {}", path.display()))
}

fn health_check(address: &str) -> bool {
    let Ok(address) = address.parse::<SocketAddr>().map(connectable_local_address) else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(300)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));
    if stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buffer = [0_u8; 64];
    let Ok(read) = stream.read(&mut buffer) else {
        return false;
    };
    buffer[..read].starts_with(b"HTTP/1.1 200") || buffer[..read].starts_with(b"HTTP/1.0 200")
}

fn connectable_local_address(address: SocketAddr) -> SocketAddr {
    if !address.ip().is_unspecified() {
        return address;
    }
    match address {
        SocketAddr::V4(address) => SocketAddr::new(Ipv4Addr::LOCALHOST.into(), address.port()),
        SocketAddr::V6(address) => SocketAddr::new(Ipv6Addr::LOCALHOST.into(), address.port()),
    }
}

#[cfg(unix)]
fn control_socket_is_live(path: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

#[cfg(not(unix))]
fn control_socket_is_live(_path: &Path) -> bool {
    false
}

impl PackageProfile {
    const fn artifact_dir(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

impl PackageLayout {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_owned(),
            bin: root.join("bin"),
            web: root.join("web"),
            data: root.join("data"),
            logs: root.join("logs"),
            run: root.join("run"),
        }
    }
}

fn executable_name(name: &'static str) -> &'static str {
    if cfg!(windows) {
        match name {
            "novasight" => "novasight.exe",
            "novasightd" => "novasightd.exe",
            "novasight-web" => "novasight-web.exe",
            "novasightctl" => "novasightctl.exe",
            _ => name,
        }
    } else {
        name
    }
}

fn copy_executable(source: &Path, destination: &Path) -> Result<()> {
    copy_file(source, destination)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o755))
            .with_context(|| format!("set executable permissions on {}", destination.display()))?;
    }
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", destination.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    fs::copy(source, destination)
        .with_context(|| format!("copy {} to {}", source.display(), destination.display()))?;
    Ok(())
}

fn copy_runtime_web_tree(source: &Path, destination: &Path) -> Result<()> {
    copy_runtime_web_tree_from_root(source, destination)
}

fn copy_model_asset_tree_if_present(source: &Path, destination: &Path) -> Result<()> {
    if !source.exists() {
        return Ok(());
    }
    copy_model_asset_tree_from_root(source, source, destination)
}

fn copy_model_asset_tree_from_root(
    model_root: &Path,
    source: &Path,
    destination: &Path,
) -> Result<()> {
    if !source.is_dir() {
        bail!("{} is not a directory", source.display());
    }
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry.with_context(|| format!("read entry below {}", source.display()))?;
        let source_path = entry.path();
        let metadata = entry
            .metadata()
            .with_context(|| format!("inspect {}", source_path.display()))?;
        if metadata.is_dir() {
            copy_model_asset_tree_from_root(
                model_root,
                &source_path,
                &destination.join(entry.file_name()),
            )?;
        } else if metadata.is_file() && should_package_model_asset_path(model_root, &source_path) {
            copy_file(&source_path, &destination.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn should_package_model_asset_path(model_root: &Path, path: &Path) -> bool {
    if path.strip_prefix(model_root).is_err() {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name.ends_with(".engine")
        || name.ends_with(".onnx")
        || name.ends_with(".engine.manifest.json")
        || name.ends_with(".onnx.manifest.json")
}

fn copy_runtime_web_tree_from_root(source: &Path, destination: &Path) -> Result<()> {
    if !source.is_dir() {
        bail!("{} is not a directory", source.display());
    }
    fs::create_dir_all(destination).with_context(|| format!("create {}", destination.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry.with_context(|| format!("read entry below {}", source.display()))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = entry
            .metadata()
            .with_context(|| format!("inspect {}", source_path.display()))?;
        if metadata.is_dir() {
            copy_runtime_web_tree_from_root(&source_path, &destination_path)?;
        } else if metadata.is_file() {
            copy_file(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn validate_package(output: &Path) -> Result<()> {
    for path in [
        output.join("NovaSight"),
        output.join("bin").join(executable_name("novasightd")),
        output.join("bin").join(executable_name("novasight-web")),
        output.join("bin").join(executable_name("novasightctl")),
        output.join("web/index.html"),
        output.join("data/novasight.yaml"),
        output.join("USER_MANUAL.md"),
        output.join("logs"),
        output.join("run"),
    ] {
        if !path.exists() {
            bail!("packaged artifact is missing {}", path.display());
        }
    }
    Ok(())
}
