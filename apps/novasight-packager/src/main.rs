use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    /// Assemble already-built Rust binaries and Web assets without rebuilding.
    #[arg(long)]
    skip_build: bool,
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
    fs::create_dir_all(output.parent().context("package output has no parent")?)?;
    let update_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output.with_extension("update.lock"))
        .context("open package update lock")?;
    fs2::FileExt::try_lock_exclusive(&update_lock)
        .context("NovaSight 正在运行或更新；请先安全退出，用户数据不会被覆盖")?;
    if output.exists() {
        validate_package(&output).context("旧目录不是完整的 NovaSight 启动包；拒绝覆盖")?;
    }
    prepare_output_for_rebuild(&output)?;
    if !args.skip_build {
        build_artifacts(&workspace, args.profile)?;
    }
    validate_artifact_revision(&workspace, args.profile)?;
    update_package(&workspace, args.profile, &output)?;
    println!("{}", output.display());
    Ok(())
}

fn update_package(workspace: &Path, profile: PackageProfile, output: &Path) -> Result<()> {
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    let staging = output.with_extension(format!("staging-{nonce}"));
    let backup = output.with_extension(format!("previous-{nonce}"));
    eprintln!("NovaSight：准备新版本，原版本保持不变…");
    assemble_package(workspace, profile, &staging)?;
    if output.exists() {
        validate_package(output).context("旧目录不是完整的 NovaSight 启动包；拒绝覆盖")?;
        eprintln!("NovaSight：保留配置、模型、数据库和日志…");
        for directory in ["data", "logs"] {
            copy_runtime_web_tree(&output.join(directory), &staging.join(directory))?;
        }
    }
    write_desktop_entry(&staging, output)?;
    validate_package(&staging)?;
    install_package(&staging, output, &backup)?;
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
    if output == out_root || !output.starts_with(&out_root) {
        bail!(
            "package output must stay below {}; got {}",
            out_root.display(),
            output.display()
        );
    }
    // Reject symlink ancestors before moving any package directory.
    let relative = output.strip_prefix(workspace)?;
    let mut ancestor = workspace.to_owned();
    for component in relative.components() {
        ancestor.push(component);
        if fs::symlink_metadata(&ancestor).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            bail!(
                "package output cannot traverse a symlink: {}",
                ancestor.display()
            );
        }
    }
    Ok(output)
}

fn install_package(staging: &Path, output: &Path, backup: &Path) -> Result<()> {
    if backup.exists() {
        bail!(
            "backup already exists; refusing to overwrite {}",
            backup.display()
        );
    }
    prepare_output_for_rebuild(output)?;
    let replacing = output.exists();
    if replacing {
        fs::rename(output, backup)
            .with_context(|| format!("backup {} to {}", output.display(), backup.display()))?;
    }
    if let Err(error) = fs::rename(staging, output) {
        if replacing {
            fs::rename(backup, output).with_context(|| {
                format!(
                    "安装失败（{error}），自动恢复失败；旧版本仍在 {}，新版本仍在 {}",
                    backup.display(),
                    staging.display()
                )
            })?;
        }
        return Err(error).context("安装新版本失败；原版本已保留");
    }
    if replacing {
        eprintln!("NovaSight：更新完成，旧版本备份：{}", backup.display());
    }
    Ok(())
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

fn validate_artifact_revision(workspace: &Path, profile: PackageProfile) -> Result<()> {
    let git = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()
        .context("read package source revision")?;
    if !git.status.success() {
        bail!("package source revision is unavailable");
    }
    let expected = String::from_utf8(git.stdout).context("decode package source revision")?;
    let daemon = workspace
        .join("out/cargo")
        .join(profile.artifact_dir())
        .join(executable_name("novasightd"));
    let info = Command::new(&daemon)
        .arg("--build-info-json")
        .output()
        .with_context(|| format!("read build identity from {}", daemon.display()))?;
    if !info.status.success() {
        bail!(
            "daemon build identity is unavailable from {}",
            daemon.display()
        );
    }
    let info: serde_json::Value =
        serde_json::from_slice(&info.stdout).context("decode daemon build identity")?;
    let actual = info["source_revision"].as_str().unwrap_or("unknown");
    require_matching_revision(expected.trim(), actual)
}

fn require_matching_revision(expected: &str, actual: &str) -> Result<()> {
    if actual != expected {
        bail!(
            "stale daemon build: source revision {expected}, binary revision {actual}; rebuild before packaging"
        );
    }
    Ok(())
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
    if output.exists() {
        bail!(
            "new package staging directory already exists: {}",
            output.display()
        );
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
        "直接运行 NovaSight，无需 Cargo 或 pnpm。Jetson 桌面也可使用 NovaSight.desktop（首次需允许启动；该快捷方式对应打包时的位置）。重复启动只打开现有界面，不重启后台。打开首页总开关才开始采集、推理与控制。其他电脑通过 Jetson 的局域网地址访问。桌面启动失败请查看 logs/launcher.log；终端启动可用 Ctrl+C 安全退出。更新会保留 data/ 与 logs/，并保留旧版本备份。详见 USER_MANUAL.md。\n",
    )
    .with_context(|| format!("write {}", layout.root.join("README-USER.txt").display()))?;
    Ok(())
}

fn write_desktop_entry(staging: &Path, output: &Path) -> Result<()> {
    let executable = output
        .join("NovaSight")
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('`', "\\`")
        .replace('$', "\\$")
        .replace('\\', "\\\\")
        .replace('%', "%%");
    if executable.contains(['\n', '\r', '=']) {
        bail!("desktop launcher path cannot contain a newline or '='");
    }
    let path = staging.join("NovaSight.desktop");
    fs::write(
        &path,
        format!(
            "[Desktop Entry]\nType=Application\nName=NovaSight Studio\nComment=打开 NovaSight 界面\nExec=\"{executable}\"\nTerminal=false\nCategories=Utility;\nActions=Quit;\n\n[Desktop Action Quit]\nName=退出 NovaSight\nExec=\"{executable}\" --quit\n"
        ),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    }
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
        if entry.file_type()?.is_symlink() {
            bail!(
                "不能静默跳过符号链接 {}；原版本保持不变",
                source_path.display()
            );
        }
        let metadata = entry
            .metadata()
            .with_context(|| format!("inspect {}", source_path.display()))?;
        if metadata.is_dir() {
            copy_runtime_web_tree_from_root(&source_path, &destination_path)?;
        } else if metadata.is_file() {
            copy_file(&source_path, &destination_path)?;
        } else {
            bail!(
                "不支持复制特殊文件 {}；原版本保持不变",
                source_path.display()
            );
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

#[cfg(test)]
mod tests {
    use super::require_matching_revision;

    #[test]
    fn package_update_preserves_user_data_and_rolls_back_failed_installation() {
        use super::*;
        let workspace = std::env::temp_dir().join(format!(
            "novasight-package-update-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for (path, contents) in [
            ("out/cargo/debug/novasight", "old-launcher"),
            ("out/cargo/debug/novasightd", "daemon"),
            ("out/cargo/debug/novasight-web", "web"),
            ("out/cargo/debug/novasightctl", "control"),
            ("out/web/index.html", "interface"),
            ("deploy/novasight.production.yaml", "default-config"),
            ("deploy/deepstream-tracker-iou.yml", "tracker"),
            ("USER_MANUAL.md", "manual"),
        ] {
            let destination = workspace.join(path);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::write(destination, contents).unwrap();
        }
        let output = workspace.join(DEFAULT_OUTPUT);
        update_package(&workspace, PackageProfile::Debug, &output).unwrap();
        for (path, contents) in [
            ("data/novasight.yaml", "user-config"),
            ("data/novasight.db", "user-database"),
            ("data/models/user.engine", "user-model"),
            ("logs/novasightd.log", "user-log"),
        ] {
            fs::write(output.join(path), contents).unwrap();
        }
        fs::write(workspace.join("out/cargo/debug/novasight"), "new-launcher").unwrap();
        update_package(&workspace, PackageProfile::Debug, &output).unwrap();
        assert_eq!(
            fs::read_to_string(output.join("NovaSight")).unwrap(),
            "new-launcher"
        );
        for (path, contents) in [
            ("data/novasight.yaml", "user-config"),
            ("data/novasight.db", "user-database"),
            ("data/models/user.engine", "user-model"),
            ("logs/novasightd.log", "user-log"),
        ] {
            assert_eq!(fs::read_to_string(output.join(path)).unwrap(), contents);
        }
        let backup = fs::read_dir(output.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("NovaSight.previous-")
            })
            .unwrap();
        assert_eq!(
            fs::read_to_string(backup.join("NovaSight")).unwrap(),
            "old-launcher"
        );
        assert!(
            install_package(
                &workspace.join("missing-staging"),
                &output,
                &workspace.join("rollback-backup")
            )
            .is_err()
        );
        assert_eq!(
            fs::read_to_string(output.join("data/novasight.db")).unwrap(),
            "user-database"
        );
        assert!(resolve_output_path(&workspace, &workspace.join("out")).is_err());
        fs::remove_dir_all(workspace).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn package_update_does_not_silently_drop_linked_user_files() {
        let root =
            std::env::temp_dir().join(format!("novasight-package-link-{}", std::process::id()));
        std::fs::create_dir_all(root.join("data")).unwrap();
        std::fs::write(root.join("model.engine"), "model").unwrap();
        std::os::unix::fs::symlink(root.join("model.engine"), root.join("data/model.engine"))
            .unwrap();
        assert!(super::copy_runtime_web_tree(&root.join("data"), &root.join("staging")).is_err());
        assert_eq!(
            std::fs::read_to_string(root.join("data/model.engine")).unwrap(),
            "model"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_daemon_revision_is_rejected() {
        assert!(require_matching_revision("current", "current").is_ok());
        assert!(
            require_matching_revision("current", "old")
                .unwrap_err()
                .to_string()
                .contains("stale daemon build")
        );
    }
}
