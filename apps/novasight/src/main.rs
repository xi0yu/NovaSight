use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, UdpSocket};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use novasight_config::{YamlConfigRepository, studio_endpoint_contract};
use rand::{RngCore, rngs::OsRng};
use serde::Deserialize;
use serde_yaml::{Mapping, Number, Value};
use tokio::process::{Child, Command};
use tokio::time;

mod frontend_dev;
mod lifecycle;

const CONFIG_PATH: &str = "data/novasight.yaml";
const DEVELOPMENT_CONFIG_PATH: &str = "data/novasight.frontend-dev.yaml";
const DATA_DIR: &str = "data";
const MODEL_DIR: &str = "data/models";
const DATABASE_PATH: &str = "data/novasight.db";
const LICENSE_PATH: &str = "data/license.json";
const LOG_DIR: &str = "logs";
const RUN_DIR: &str = "run";
const CONTROL_SOCKET: &str = "run/novasightd.sock";
const WEB_READY_FILE: &str = "run/ready.json";
const DAEMON_READY_FILE: &str = "run/novasightd-ready.json";
const TEMPORARY_LICENSE_ACCESS_FILE: &str = "run/temporary-license-code";
const TEMPORARY_LICENSE_CODE_ENV: &str = "NOVASIGHT_TEMPORARY_LICENSE_CODE";
const PROCESS_READY_TIMEOUT: Duration = Duration::from_secs(20);
const PROCESS_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const LOG_TAIL_BYTES: u64 = 3 * 1024;

#[derive(Clone, Debug)]
struct PortableLayout {
    mode: LayoutMode,
    root: PathBuf,
    daemon: PathBuf,
    web: PathBuf,
    control: PathBuf,
    config: PathBuf,
    data_dir: PathBuf,
    model_dir: PathBuf,
    log_dir: PathBuf,
    run_dir: PathBuf,
    web_ready_file: PathBuf,
    daemon_ready_file: PathBuf,
    daemon_log: PathBuf,
    web_log: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayoutMode {
    Package,
    Developer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StartupRequest {
    Open,
    Restart,
    Quit,
}

#[derive(Clone, Debug, Deserialize)]
struct ReadyDocument {
    address: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct DaemonReadyDocument {
    control_socket: String,
    transport: String,
}

#[derive(Clone, Debug)]
struct TemporaryLicenseAccess {
    code: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("NOVASIGHT_LAUNCH_FAILED: {error:#}");
            if let Ok(layout) = PortableLayout::discover()
                && let Ok(_guard) = lock_package_for_run(&layout)
            {
                let _ = fs::create_dir_all(&layout.log_dir);
                if let Ok(mut log) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(layout.log_dir.join("launcher.log"))
                {
                    let _ = writeln!(log, "NOVASIGHT_LAUNCH_FAILED: {error:#}");
                }
            }
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let request = parse_startup_request(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
    let layout = PortableLayout::discover()?;
    if request == StartupRequest::Restart && layout.mode != LayoutMode::Developer {
        bail!("--restart 仅用于源码开发；正式版本请先安全退出，再更新启动包");
    }
    let _package_lock = lock_package_for_run(&layout)?;
    if request == StartupRequest::Quit {
        if daemon_status_succeeds(&layout).await? {
            request_daemon_shutdown(&layout).await?;
            eprintln!("NovaSight：已请求后台安全退出");
        } else {
            eprintln!("NovaSight：当前没有运行中的后台");
        }
        return Ok(());
    }
    prepare_layout(&layout)?;
    if request == StartupRequest::Open
        && let Some(ready) = existing_studio(&layout).await?
    {
        eprintln!("NOVASIGHT_EXISTING_STACK: 已在运行，打开现有界面；不会编译或重启后台");
        open_studio(&ready);
        return Ok(());
    }
    if layout.mode == LayoutMode::Developer {
        return frontend_dev::run(&layout).await;
    }
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set bundle root {}", layout.root.display()))?;
    eprintln!("NovaSight：检查配置…");
    remove_stale_ready_files(&layout)?;
    ensure_portable_config(&layout)?;
    let temporary_license = create_temporary_license_access(&layout)?;
    eprintln!("NovaSight：启动后台服务…");
    let mut daemon = spawn_daemon(&layout, temporary_license.as_ref())?;
    if let Err(error) = wait_for_daemon_ready(&layout, &mut daemon, PROCESS_READY_TIMEOUT).await {
        let _ = stop_child("novasightd", &mut daemon).await;
        return Err(error);
    }
    eprintln!("NovaSight：启动界面服务…");
    let mut web = match spawn_web(&layout, false) {
        Ok(web) => web,
        Err(error) => {
            let _ = stop_owned_daemon(&layout, &mut daemon).await;
            return Err(error);
        }
    };
    let ready =
        match wait_for_web_ready(&layout, &mut web, &mut daemon, PROCESS_READY_TIMEOUT).await {
            Ok(ready) => ready,
            Err(error) => {
                let _ = stop_child("novasight-web", &mut web).await;
                let _ = stop_owned_daemon(&layout, &mut daemon).await;
                return Err(error);
            }
        };
    eprintln!("NovaSight：界面已就绪，采集与控制需在首页开启");
    open_studio(&ready);
    print_temporary_license_access(temporary_license.as_ref());
    lifecycle::supervise(&layout, daemon, web, None).await
}

fn parse_startup_request(args: &[OsString]) -> Result<StartupRequest> {
    match args {
        [] => Ok(StartupRequest::Open),
        [arg] if arg == "--restart" => Ok(StartupRequest::Restart),
        [arg] if arg == "--quit" => Ok(StartupRequest::Quit),
        _ => bail!("直接启动 NovaSight；安全退出使用 --quit，源码开发更新后台使用 --restart"),
    }
}

fn lock_package_for_run(layout: &PortableLayout) -> Result<Option<File>> {
    if layout.mode != LayoutMode::Package {
        return Ok(None);
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(layout.root.with_extension("update.lock"))
        .context("open package update lock")?;
    fs2::FileExt::try_lock_shared(&file).context("NovaSight 正在更新，请更新完成后再启动")?;
    Ok(Some(file))
}

async fn existing_studio(layout: &PortableLayout) -> Result<Option<ReadyDocument>> {
    let ready = read_ready_file(&layout.web_ready_file).ok();
    if !ready
        .as_ref()
        .is_some_and(|ready| health_check(&ready.address))
    {
        if layout.control.is_file() && daemon_status_succeeds(layout).await? {
            bail!(
                "已有后台正在启动或界面异常；不会自动重启，请安全退出后重试，源码更新可使用 --restart"
            );
        }
        return Ok(None);
    }
    if !daemon_status_succeeds(layout).await? {
        bail!("界面端口已被占用，但本目录的后台不可达；不会接管或关闭其他实例");
    }
    let studio = if layout.mode == LayoutMode::Developer {
        frontend_dev::studio_ready_document()?
    } else {
        ready.context("healthy gateway did not publish its address")?
    };
    if !health_check(&studio.address) {
        bail!("后台已运行，但界面不可用；请安全退出原实例后重试，源码开发可使用 --restart");
    }
    Ok(Some(studio))
}

async fn daemon_status_succeeds(layout: &PortableLayout) -> Result<bool> {
    // License status is available even before activation. Pin the local socket
    // so an inherited diagnostic override cannot redirect instance ownership.
    let status = time::timeout(
        Duration::from_secs(3),
        Command::new(&layout.control)
            .args(["license", "status"])
            .env("NOVASIGHT_CONTROL_SOCKET", layout.root.join(CONTROL_SOCKET))
            .current_dir(&layout.root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status(),
    )
    .await
    .context("检查现有后台超时；不会重启或接管该实例")?
    .with_context(|| {
        format!(
            "inspect existing daemon through {}",
            layout.control.display()
        )
    })?;
    Ok(status.success())
}

impl PortableLayout {
    fn discover() -> Result<Self> {
        let current_exe = std::env::current_exe().context("read current executable path")?;
        let executable_dir = current_exe
            .parent()
            .ok_or_else(|| anyhow!("current executable has no parent directory"))?;
        if let Some(workspace) = developer_workspace_root(executable_dir) {
            return Ok(Self::new(
                LayoutMode::Developer,
                workspace,
                executable_dir.join(executable_name("novasightd")),
                executable_dir.join(executable_name("novasight-web")),
                executable_dir.join(executable_name("novasightctl")),
            ));
        }
        let root = infer_bundle_root(executable_dir);
        let daemon = resolve_binary(&root, executable_dir, "novasightd")?;
        let web = resolve_binary(&root, executable_dir, "novasight-web")?;
        let control = resolve_binary(&root, executable_dir, "novasightctl")?;
        Ok(Self::new(LayoutMode::Package, root, daemon, web, control))
    }

    fn new(
        mode: LayoutMode,
        root: PathBuf,
        daemon: PathBuf,
        web: PathBuf,
        control: PathBuf,
    ) -> Self {
        let log_dir = root.join(LOG_DIR);
        let config_path = match mode {
            LayoutMode::Package => CONFIG_PATH,
            LayoutMode::Developer => DEVELOPMENT_CONFIG_PATH,
        };
        Self {
            mode,
            daemon,
            web,
            control,
            config: root.join(config_path),
            data_dir: root.join(DATA_DIR),
            model_dir: root.join(MODEL_DIR),
            run_dir: root.join(RUN_DIR),
            web_ready_file: root.join(WEB_READY_FILE),
            daemon_ready_file: root.join(DAEMON_READY_FILE),
            daemon_log: log_dir.join("novasightd.log"),
            web_log: log_dir.join("novasight-web.log"),
            log_dir,
            root,
        }
    }
}

fn developer_workspace_root(executable_dir: &Path) -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent()?.parent()?.to_owned();
    if !looks_like_workspace_root(&workspace) {
        return None;
    }
    executable_dir
        .starts_with(workspace.join("out/cargo"))
        .then_some(workspace)
}

fn looks_like_workspace_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file() && path.join("apps").is_dir() && path.join("web").is_dir()
}

fn infer_bundle_root(executable_dir: &Path) -> PathBuf {
    if executable_dir.file_name().is_some_and(|name| name == "bin") {
        return executable_dir
            .parent()
            .map(Path::to_owned)
            .unwrap_or_else(|| executable_dir.to_owned());
    }
    executable_dir.to_owned()
}

fn resolve_binary(root: &Path, executable_dir: &Path, binary: &'static str) -> Result<PathBuf> {
    let name = executable_name(binary);
    [
        root.join("bin").join(name),
        root.join(name),
        executable_dir.join(name),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| anyhow!("{binary} was not found below {}", root.display()))
}

fn executable_name(name: &'static str) -> &'static str {
    if cfg!(windows) {
        match name {
            "novasightd" => "novasightd.exe",
            "novasight-web" => "novasight-web.exe",
            "novasightctl" => "novasightctl.exe",
            _ => name,
        }
    } else {
        name
    }
}

fn prepare_layout(layout: &PortableLayout) -> Result<()> {
    fs::create_dir_all(&layout.data_dir)
        .with_context(|| format!("create {}", layout.data_dir.display()))?;
    fs::create_dir_all(&layout.model_dir)
        .with_context(|| format!("create {}", layout.model_dir.display()))?;
    fs::create_dir_all(&layout.log_dir)
        .with_context(|| format!("create {}", layout.log_dir.display()))?;
    ensure_private_directory(&layout.run_dir)
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("set private permissions on {}", path.display()))?;
    Ok(())
}

fn remove_stale_ready_files(layout: &PortableLayout) -> Result<()> {
    for path in [&layout.web_ready_file, &layout.daemon_ready_file] {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).with_context(|| format!("remove {}", path.display())),
        }
    }
    Ok(())
}

fn ensure_portable_config(layout: &PortableLayout) -> Result<()> {
    if !layout.config.exists() {
        let production_config = layout.root.join(CONFIG_PATH);
        if layout.mode == LayoutMode::Developer && production_config.is_file() {
            fs::copy(&production_config, &layout.config).with_context(|| {
                format!(
                    "initialize {} from {}",
                    layout.config.display(),
                    production_config.display()
                )
            })?;
        } else {
            YamlConfigRepository::initialize_default(&layout.config)
                .with_context(|| format!("initialize {}", layout.config.display()))?;
        }
    }
    let mut document: Value = serde_yaml::from_reader(
        File::open(&layout.config).with_context(|| format!("open {}", layout.config.display()))?,
    )
    .with_context(|| format!("parse {}", layout.config.display()))?;
    let studio = &studio_endpoint_contract().studio;
    let mut changed = false;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "host",
        Value::String(studio.host.clone()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "port",
        Value::Number(Number::from(studio.port)),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "control_socket",
        Value::String(CONTROL_SOCKET.to_owned()),
    )?;
    for (section, key, value) in [
        ("paths", "data_dir", DATA_DIR),
        ("paths", "model_dir", MODEL_DIR),
        ("paths", "database", DATABASE_PATH),
        ("paths", "license", LICENSE_PATH),
    ] {
        changed |= set_mapping_field(&mut document, section, key, Value::String(value.to_owned()))?;
    }
    if changed {
        let current = YamlConfigRepository::load(&layout.config)
            .with_context(|| format!("load {}", layout.config.display()))?;
        YamlConfigRepository::new(&layout.config)
            .replace_document(document, current.revision)
            .with_context(|| format!("save {}", layout.config.display()))?;
    }
    Ok(())
}

fn set_mapping_field(document: &mut Value, section: &str, key: &str, value: Value) -> Result<bool> {
    let root = document
        .as_mapping_mut()
        .ok_or_else(|| anyhow!("configuration root must be a mapping"))?;
    let section_value = root
        .entry(Value::String(section.to_owned()))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    let section_mapping = section_value
        .as_mapping_mut()
        .ok_or_else(|| anyhow!("configuration section {section} must be a mapping"))?;
    let key = Value::String(key.to_owned());
    if section_mapping.get(&key) == Some(&value) {
        return Ok(false);
    }
    section_mapping.insert(key, value);
    Ok(true)
}

fn spawn_logged_process(
    executable: &Path,
    root: &Path,
    log_path: &Path,
    args: &[OsString],
    environment: &[(&str, &str)],
) -> Result<Child> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("open {}", log_path.display()))?;
    let stdout = log
        .try_clone()
        .with_context(|| format!("clone {}", log_path.display()))?;
    let mut command = Command::new(executable);
    command
        .args(args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log))
        .kill_on_drop(true);
    for (name, value) in environment {
        command.env(name, value);
    }
    command
        .spawn()
        .with_context(|| format!("start {}", executable.display()))
}

fn spawn_daemon(
    layout: &PortableLayout,
    temporary_license: Option<&TemporaryLicenseAccess>,
) -> Result<Child> {
    let args = [
        OsString::from("--config"),
        layout.config.as_os_str().to_owned(),
    ];
    match temporary_license {
        Some(access) => spawn_logged_process(
            &layout.daemon,
            &layout.root,
            &layout.daemon_log,
            &args,
            &[(TEMPORARY_LICENSE_CODE_ENV, access.code.as_str())],
        ),
        None => spawn_logged_process(&layout.daemon, &layout.root, &layout.daemon_log, &args, &[]),
    }
}

fn spawn_web(layout: &PortableLayout, frontend_dev: bool) -> Result<Child> {
    let mut args = vec![
        OsString::from("--config"),
        layout.config.as_os_str().to_owned(),
    ];
    if frontend_dev {
        args.push(OsString::from("--frontend-dev"));
    }
    spawn_logged_process(&layout.web, &layout.root, &layout.web_log, &args, &[])
}

async fn wait_for_daemon_ready(
    layout: &PortableLayout,
    daemon: &mut Child,
    timeout: Duration,
) -> Result<DaemonReadyDocument> {
    let started = Instant::now();
    loop {
        if let Some(status) = daemon.try_wait().context("check novasightd process")? {
            bail!(
                "novasightd exited before ready with status {status}; {}",
                log_tail(&layout.daemon_log)
            );
        }
        if let Ok(ready) = read_daemon_ready_file(&layout.daemon_ready_file)
            && ready.transport == "http1-unix"
            && !ready.control_socket.trim().is_empty()
        {
            return Ok(ready);
        }
        if started.elapsed() >= timeout {
            bail!(
                "novasightd did not publish Unix control readiness within {timeout:?}; {}",
                log_tail(&layout.daemon_log)
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_web_ready(
    layout: &PortableLayout,
    web: &mut Child,
    daemon: &mut Child,
    timeout: Duration,
) -> Result<ReadyDocument> {
    let started = Instant::now();
    loop {
        if let Some(status) = web.try_wait().context("check novasight-web process")? {
            bail!(
                "novasight-web exited before ready with status {status}; {}",
                log_tail(&layout.web_log)
            );
        }
        if let Some(status) = daemon.try_wait().context("check novasightd process")? {
            bail!(
                "novasightd exited while Web/API was starting with status {status}; {}",
                log_tail(&layout.daemon_log)
            );
        }
        if let Ok(ready) = read_ready_file(&layout.web_ready_file)
            && health_check(&ready.address)
        {
            return Ok(ready);
        }
        if started.elapsed() >= timeout {
            bail!(
                "novasight-web did not become healthy within {timeout:?}; {}",
                log_tail(&layout.web_log)
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

fn log_tail(path: &Path) -> String {
    match read_file_tail(path, LOG_TAIL_BYTES) {
        Ok(text) if !text.trim().is_empty() => format!("last log output:\n{}", text.trim_end()),
        Ok(_) => format!("{} is empty", path.display()),
        Err(error) => format!("{} unavailable: {error}", path.display()),
    }
}

fn read_file_tail(path: &Path, max_bytes: u64) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let start = len.saturating_sub(max_bytes);
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::with_capacity((len - start).min(max_bytes) as usize);
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

async fn stop_owned_daemon(layout: &PortableLayout, child: &mut Child) -> Result<()> {
    eprintln!("NOVASIGHT_SHUTDOWN_REQUESTED: stopping novasightd through local IPC");
    if let Err(error) = request_daemon_shutdown(layout).await {
        eprintln!("NOVASIGHT_SHUTDOWN_FALLBACK: {error:#}");
    }
    match time::timeout(PROCESS_SHUTDOWN_TIMEOUT, child.wait()).await {
        Ok(status) => {
            let _ = status.context("wait for novasightd process")?;
            Ok(())
        }
        Err(_) => stop_child("novasightd", child).await,
    }
}

async fn stop_child(label: &str, child: &mut Child) -> Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    eprintln!("NOVASIGHT_SHUTDOWN_REQUESTED: stopping {label}");
    child
        .start_kill()
        .with_context(|| format!("stop {label}"))?;
    let _ = child
        .wait()
        .await
        .with_context(|| format!("wait for {label}"))?;
    Ok(())
}

async fn request_daemon_shutdown(layout: &PortableLayout) -> Result<()> {
    let status = Command::new(&layout.control)
        .arg("shutdown")
        .env("NOVASIGHT_CONTROL_SOCKET", layout.root.join(CONTROL_SOCKET))
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .await
        .with_context(|| format!("run {}", layout.control.display()))?;
    if !status.success() {
        bail!("novasightctl shutdown exited with {status}");
    }
    Ok(())
}

async fn wait_for_shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt()).context("register SIGINT handler")?;
        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            _ = interrupt.recv() => Ok(()),
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl+C")
}

fn read_ready_file(path: &Path) -> Result<ReadyDocument> {
    serde_json::from_reader(File::open(path)?).with_context(|| format!("parse {}", path.display()))
}

fn read_daemon_ready_file(path: &Path) -> Result<DaemonReadyDocument> {
    serde_json::from_reader(File::open(path)?).with_context(|| format!("parse {}", path.display()))
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

fn tcp_port_accepts_connections(address: &str) -> bool {
    address
        .parse::<SocketAddr>()
        .map(connectable_local_address)
        .is_ok_and(|address| {
            TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_ok()
        })
}

fn open_studio(ready: &ReadyDocument) {
    print_studio_urls(ready);
    let url = ready_address(ready)
        .map(connectable_local_address)
        .map(http_url)
        .unwrap_or_else(|| ready.url.clone());
    if let Err(error) = open_browser(&url) {
        eprintln!(
            "NOVASIGHT_BROWSER_OPEN_SKIPPED: {error:#}; open the authenticated Studio URL above manually"
        );
    }
}

fn print_studio_urls(ready: &ReadyDocument) {
    let lan_ips = detect_lan_ips();
    for message in studio_ready_messages(ready, &lan_ips) {
        println!("{message}");
    }
}

fn studio_ready_messages(ready: &ReadyDocument, lan_ips: &[IpAddr]) -> Vec<String> {
    let Some(address) = ready_address(ready) else {
        return vec![format!("NovaSight Studio 访问地址：{}", ready.url)];
    };
    if !address.ip().is_unspecified() {
        return vec![format!("NovaSight Studio 访问地址：{}", ready.url)];
    }
    let mut unique_lan_ips = Vec::new();
    for ip in lan_ips.iter().copied().filter_map(candidate_lan_ip) {
        if ip.is_ipv4() {
            unique_lan_ips.push(ip);
        }
    }
    unique_lan_ips.sort_by_key(lan_ip_sort_key);
    unique_lan_ips.dedup();
    let lan_url = unique_lan_ips
        .first()
        .map(|ip| http_url(SocketAddr::new(*ip, address.port())))
        .unwrap_or_else(|| format!("http://<本机局域网IP>:{}/", address.port()));
    let mut messages = vec![format!("NovaSight Studio 局域网访问地址：{}", lan_url)];
    messages.push(format!(
        "  ➜  Local:   http://localhost:{}/",
        address.port()
    ));
    messages.extend(unique_lan_ips.into_iter().map(|ip| {
        format!(
            "  ➜  Network: {}",
            http_url(SocketAddr::new(ip, address.port()))
        )
    }));
    messages
}

fn create_temporary_license_access(
    layout: &PortableLayout,
) -> Result<Option<TemporaryLicenseAccess>> {
    let path = layout.root.join(TEMPORARY_LICENSE_ACCESS_FILE);
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("remove stale {}", path.display()));
        }
    }
    if !cfg!(debug_assertions) {
        return Ok(None);
    }
    Ok(Some(TemporaryLicenseAccess {
        code: random_access_code(),
    }))
}

fn random_access_code() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn print_temporary_license_access(access: Option<&TemporaryLicenseAccess>) {
    let Some(access) = access else {
        return;
    };
    eprintln!("NovaSight 临时授权码 {}", access.code);
}

fn ready_address(ready: &ReadyDocument) -> Option<SocketAddr> {
    ready.address.parse().ok()
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

fn http_url(address: SocketAddr) -> String {
    format!("http://{address}/")
}

fn detect_lan_ips() -> Vec<IpAddr> {
    let mut ips = detect_lan_ips_from_interfaces();
    if let Some(ip) = detect_lan_ip_by_udp_route()
        && !ips.contains(&ip)
    {
        ips.push(ip);
    }
    ips
}

fn detect_lan_ip_by_udp_route() -> Option<IpAddr> {
    let socket = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).ok()?;
    socket.connect(SocketAddr::from(([8, 8, 8, 8], 80))).ok()?;
    candidate_lan_ip(socket.local_addr().ok()?.ip())
}

#[cfg(target_os = "linux")]
fn detect_lan_ips_from_interfaces() -> Vec<IpAddr> {
    let Ok(output) = StdCommand::new("hostname").arg("-I").output() else {
        return Vec::new();
    };
    parse_whitespace_lan_ips(&output.stdout)
}

#[cfg(target_os = "macos")]
fn detect_lan_ips_from_interfaces() -> Vec<IpAddr> {
    let Ok(output) = StdCommand::new("/sbin/ifconfig").arg("-a").output() else {
        return Vec::new();
    };
    parse_ifconfig_lan_ips(&output.stdout)
}

#[cfg(target_os = "windows")]
fn detect_lan_ips_from_interfaces() -> Vec<IpAddr> {
    let Ok(output) = StdCommand::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.AddressState -eq 'Preferred' } | Select-Object -ExpandProperty IPAddress",
        ])
        .output()
    else {
        return Vec::new();
    };
    parse_whitespace_lan_ips(&output.stdout)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn detect_lan_ips_from_interfaces() -> Vec<IpAddr> {
    Vec::new()
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn parse_whitespace_lan_ips(output: &[u8]) -> Vec<IpAddr> {
    String::from_utf8_lossy(output)
        .split_whitespace()
        .filter_map(|candidate| candidate.parse().ok())
        .filter_map(candidate_lan_ip)
        .filter(IpAddr::is_ipv4)
        .collect()
}

#[cfg(any(target_os = "macos", test))]
fn parse_ifconfig_lan_ips(output: &[u8]) -> Vec<IpAddr> {
    String::from_utf8_lossy(output)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            if fields.next()? != "inet" {
                return None;
            }
            fields.next()?.parse().ok()
        })
        .filter_map(candidate_lan_ip)
        .filter(IpAddr::is_ipv4)
        .collect()
}

fn candidate_lan_ip(ip: IpAddr) -> Option<IpAddr> {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return None;
    }
    match ip {
        IpAddr::V4(ip) if ip.is_link_local() => None,
        IpAddr::V6(ip) if ip.is_unicast_link_local() => None,
        _ => Some(ip),
    }
}

fn lan_ip_sort_key(ip: &IpAddr) -> (u8, u32) {
    let IpAddr::V4(ip) = ip else {
        return (u8::MAX, u32::MAX);
    };
    let [first, second, _, _] = ip.octets();
    let network_rank = if ip.is_private() {
        0
    } else if first == 100 && (64..=127).contains(&second) {
        1
    } else {
        2
    };
    (network_rank, u32::from(*ip))
}

#[cfg(target_os = "linux")]
fn open_browser(url: &str) -> Result<()> {
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        bail!("no graphical session detected");
    }
    StdCommand::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start browser")?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) -> Result<()> {
    StdCommand::new("open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start browser")?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) -> Result<()> {
    StdCommand::new("cmd")
        .args(["/C", "start", "", url])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start browser")?;
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn open_browser(_url: &str) -> Result<()> {
    bail!("opening the browser is not supported on this platform")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::net::{IpAddr, Ipv4Addr};
    use std::path::PathBuf;

    use super::{
        CONFIG_PATH, DEVELOPMENT_CONFIG_PATH, LayoutMode, PortableLayout, ReadyDocument,
        TEMPORARY_LICENSE_ACCESS_FILE, create_temporary_license_access, ensure_portable_config,
        parse_ifconfig_lan_ips, parse_startup_request, random_access_code, studio_ready_messages,
    };
    use novasight_config::YamlConfigRepository;
    use serde_yaml::Value;

    struct TestRoot(PathBuf);

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn only_explicit_restart_requests_update_the_running_development_stack() {
        assert_eq!(
            parse_startup_request(&[]).unwrap(),
            super::StartupRequest::Open
        );
        assert_eq!(
            parse_startup_request(&["--restart".into()]).unwrap(),
            super::StartupRequest::Restart
        );
        assert_eq!(
            parse_startup_request(&["--quit".into()]).unwrap(),
            super::StartupRequest::Quit
        );
        assert!(parse_startup_request(&["--restart".into(), "--restart".into()]).is_err());
        assert!(parse_startup_request(&["--unknown".into()]).is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn existing_package_is_reused_only_when_its_local_daemon_is_reachable() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::os::unix::fs::PermissionsExt;

        let root = TestRoot(
            std::env::temp_dir().join(format!("novasight-launcher-reuse-{}", random_access_code())),
        );
        fs::create_dir_all(root.0.join("run")).unwrap();
        let control = root.0.join("novasightctl");
        fs::write(&control, "#!/bin/sh\ntest \"$1 $2\" = 'license status' && test \"$NOVASIGHT_CONTROL_SOCKET\" = \"${0%/*}/run/novasightd.sock\" && test ! -f daemon-unavailable\n").unwrap();
        fs::set_permissions(&control, fs::Permissions::from_mode(0o755)).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        fs::write(
            root.0.join(super::WEB_READY_FILE),
            format!(r#"{{"address":"{address}","url":"http://{address}/"}}"#),
        )
        .unwrap();
        let gateway = std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = stream.read(&mut [0_u8; 256]);
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
                    )
                    .unwrap();
            }
        });
        let layout = PortableLayout::new(
            LayoutMode::Package,
            root.0.clone(),
            root.0.join("novasightd"),
            root.0.join("novasight-web"),
            control,
        );
        assert_eq!(
            super::existing_studio(&layout)
                .await
                .unwrap()
                .unwrap()
                .address,
            address.to_string()
        );
        fs::write(root.0.join("daemon-unavailable"), "").unwrap();
        assert!(super::existing_studio(&layout).await.is_err());
        gateway.join().unwrap();
        fs::remove_file(root.0.join("daemon-unavailable")).unwrap();
        fs::remove_file(&layout.web_ready_file).unwrap();
        assert!(super::existing_studio(&layout).await.is_err());
    }

    #[test]
    fn generated_access_codes_are_full_width_hex_and_not_reused() {
        let first = random_access_code();
        let second = random_access_code();
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn developer_and_package_layouts_use_separate_config_files() {
        let root = std::env::temp_dir().join("novasight-layout-test");
        let developer = PortableLayout::new(
            LayoutMode::Developer,
            root.clone(),
            root.join("novasightd"),
            root.join("novasight-web"),
            root.join("novasightctl"),
        );
        let package = PortableLayout::new(
            LayoutMode::Package,
            root.clone(),
            root.join("novasightd"),
            root.join("novasight-web"),
            root.join("novasightctl"),
        );

        assert_eq!(developer.config, root.join(DEVELOPMENT_CONFIG_PATH));
        assert_eq!(package.config, root.join(CONFIG_PATH));
    }

    #[test]
    fn first_developer_config_inherits_production_without_rewriting_it() {
        let root = TestRoot(std::env::temp_dir().join(format!(
            "novasight-launcher-config-test-{}",
            random_access_code()
        )));
        let production_config = root.0.join(CONFIG_PATH);
        YamlConfigRepository::initialize_default(&production_config).unwrap();
        let mut production_document = fs::read_to_string(&production_config).unwrap();
        production_document.push_str("deployment_marker: jetson-calibrated\n");
        fs::write(&production_config, production_document).unwrap();
        let production_before = fs::read(&production_config).unwrap();
        let layout = PortableLayout::new(
            LayoutMode::Developer,
            root.0.clone(),
            root.0.join("novasightd"),
            root.0.join("novasight-web"),
            root.0.join("novasightctl"),
        );

        ensure_portable_config(&layout).unwrap();

        assert_eq!(fs::read(&production_config).unwrap(), production_before);
        let development = YamlConfigRepository::load(&layout.config).unwrap();
        assert_eq!(
            development.extra.get("deployment_marker"),
            Some(&Value::String("jetson-calibrated".to_owned()))
        );
    }

    #[test]
    fn unspecified_listener_reports_plain_local_and_all_network_urls() {
        let ready = ReadyDocument {
            address: "0.0.0.0:7351".to_owned(),
            url: "http://0.0.0.0:7351/".to_owned(),
        };
        let messages = studio_ready_messages(
            &ready,
            &[
                IpAddr::V4(Ipv4Addr::new(100, 106, 210, 36)),
                IpAddr::V4(Ipv4Addr::new(192, 168, 31, 248)),
                IpAddr::V4(Ipv4Addr::new(192, 168, 31, 248)),
            ],
        );

        assert_eq!(
            messages,
            vec![
                "NovaSight Studio 局域网访问地址：http://192.168.31.248:7351/".to_owned(),
                "  ➜  Local:   http://localhost:7351/".to_owned(),
                "  ➜  Network: http://192.168.31.248:7351/".to_owned(),
                "  ➜  Network: http://100.106.210.36:7351/".to_owned(),
            ]
        );

        assert_eq!(
            studio_ready_messages(&ready, &[]),
            vec![
                "NovaSight Studio 局域网访问地址：http://<本机局域网IP>:7351/".to_owned(),
                "  ➜  Local:   http://localhost:7351/".to_owned(),
            ]
        );
    }

    #[test]
    fn macos_interface_output_discovers_lan_and_tailscale_ipv4s() {
        let output = b"\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST>\n\
    inet 127.0.0.1 netmask 0xff000000\n\
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST>\n\
    inet 192.168.31.248 netmask 0xffffff00 broadcast 192.168.31.255\n\
utun4: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST>\n\
    inet 100.106.210.36 --> 100.106.210.36 netmask 0xffffffff\n\
    inet6 fe80::1%utun4 prefixlen 64 scopeid 0x16\n";

        assert_eq!(
            parse_ifconfig_lan_ips(output),
            vec![
                IpAddr::V4(Ipv4Addr::new(192, 168, 31, 248)),
                IpAddr::V4(Ipv4Addr::new(100, 106, 210, 36)),
            ]
        );
    }

    #[test]
    fn temporary_license_is_unique_per_start_and_never_persisted() {
        let root = TestRoot(std::env::temp_dir().join(format!(
            "novasight-launcher-license-test-{}",
            random_access_code()
        )));
        fs::create_dir_all(root.0.join("run")).unwrap();
        let stale_path = root.0.join(TEMPORARY_LICENSE_ACCESS_FILE);
        fs::write(&stale_path, "stale-license").unwrap();
        let layout = PortableLayout::new(
            LayoutMode::Developer,
            root.0.clone(),
            root.0.join("novasightd"),
            root.0.join("novasight-web"),
            root.0.join("novasightctl"),
        );

        let first = create_temporary_license_access(&layout)
            .unwrap()
            .expect("debug launcher must create temporary authorization");
        assert!(!stale_path.exists());
        let second = create_temporary_license_access(&layout)
            .unwrap()
            .expect("debug launcher must create temporary authorization");

        assert_ne!(first.code, second.code);
        assert!(!stale_path.exists());
    }
}
