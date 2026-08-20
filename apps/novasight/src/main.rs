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

const CONFIG_PATH: &str = "data/novasight.yaml";
const DATA_DIR: &str = "data";
const MODEL_DIR: &str = "data/models";
const DATABASE_PATH: &str = "data/novasight.db";
const LICENSE_PATH: &str = "data/license.json";
const LOG_DIR: &str = "logs";
const RUN_DIR: &str = "run";
const CONTROL_SOCKET: &str = "run/novasightd.sock";
const WEB_READY_FILE: &str = "run/ready.json";
const DAEMON_READY_FILE: &str = "run/novasightd-ready.json";
const WEB_ACCESS_FILE: &str = "run/web-access-code";
const TEMPORARY_LICENSE_ACCESS_FILE: &str = "run/temporary-license-code";
const WEB_ACCESS_CODE_ENV: &str = "NOVASIGHT_WEB_ACCESS_CODE";
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
struct WebAccess {
    code: String,
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
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    ensure_zero_arguments()?;
    let layout = PortableLayout::discover()?;
    prepare_layout(&layout)?;
    if layout.mode == LayoutMode::Developer {
        return frontend_dev::run(&layout).await;
    }
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set bundle root {}", layout.root.display()))?;
    if let Ok(ready) = read_ready_file(&layout.web_ready_file)
        && health_check(&ready.address)
    {
        bail!(
            "NovaSight Web/API is already running at {}; stop the existing launcher before starting another instance",
            ready.url
        );
    }

    remove_stale_ready_files(&layout)?;
    ensure_portable_config(&layout)?;
    let access = create_web_access(&layout)?;
    let temporary_license = create_temporary_license_access(&layout)?;
    let mut daemon = spawn_daemon(&layout, temporary_license.as_ref())?;
    if let Err(error) = wait_for_daemon_ready(&layout, &mut daemon, PROCESS_READY_TIMEOUT).await {
        let _ = stop_child("novasightd", &mut daemon).await;
        return Err(error);
    }
    let mut web = match spawn_web(&layout, &access, false) {
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
    open_studio(&ready, &access);
    print_temporary_license_access(&layout, temporary_license.as_ref());
    supervise_stack(&layout, daemon, web).await
}

fn ensure_zero_arguments() -> Result<()> {
    if std::env::args_os().nth(1).is_some() {
        bail!("NovaSight does not accept startup arguments; run NovaSight directly");
    }
    Ok(())
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
        Self {
            mode,
            daemon,
            web,
            control,
            config: root.join(CONFIG_PATH),
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
        YamlConfigRepository::initialize_default(&layout.config)
            .with_context(|| format!("initialize {}", layout.config.display()))?;
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
    args: &[&str],
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
    match temporary_license {
        Some(access) => spawn_logged_process(
            &layout.daemon,
            &layout.root,
            &layout.daemon_log,
            &[],
            &[(TEMPORARY_LICENSE_CODE_ENV, access.code.as_str())],
        ),
        None => spawn_logged_process(&layout.daemon, &layout.root, &layout.daemon_log, &[], &[]),
    }
}

fn spawn_web(layout: &PortableLayout, access: &WebAccess, frontend_dev: bool) -> Result<Child> {
    let args = if frontend_dev {
        &["--frontend-dev"][..]
    } else {
        &[][..]
    };
    spawn_logged_process(
        &layout.web,
        &layout.root,
        &layout.web_log,
        args,
        &[(WEB_ACCESS_CODE_ENV, access.code.as_str())],
    )
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

async fn supervise_stack(layout: &PortableLayout, mut daemon: Child, mut web: Child) -> Result<()> {
    eprintln!("NOVASIGHT_RUNNING: press Ctrl+C to stop Web/API and novasightd");
    tokio::select! {
        status = daemon.wait() => {
            let status = status.context("wait for novasightd process")?;
            let _ = stop_child("novasight-web", &mut web).await;
            bail!("novasightd exited with status {status}")
        }
        status = web.wait() => {
            let status = status.context("wait for novasight-web process")?;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("novasight-web exited with status {status}")
        }
        signal = wait_for_shutdown_signal() => {
            signal?;
            let web_stop = stop_child("novasight-web", &mut web).await;
            let daemon_stop = stop_owned_daemon(layout, &mut daemon).await;
            web_stop.and(daemon_stop)
        }
    }
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

fn open_studio(ready: &ReadyDocument, access: &WebAccess) {
    print_studio_urls(ready, access);
    let url = access_url(
        &ready_address(ready)
            .map(connectable_local_address)
            .map(http_url)
            .unwrap_or_else(|| ready.url.clone()),
        access,
    );
    if let Err(error) = open_browser(&url) {
        eprintln!(
            "NOVASIGHT_BROWSER_OPEN_SKIPPED: {error:#}; open the authenticated Studio URL above manually"
        );
    }
}

fn print_studio_urls(ready: &ReadyDocument, access: &WebAccess) {
    for message in studio_ready_messages(ready, access, detect_lan_ip()) {
        println!("{message}");
    }
}

fn studio_ready_messages(
    ready: &ReadyDocument,
    access: &WebAccess,
    lan_ip: Option<IpAddr>,
) -> Vec<String> {
    let Some(address) = ready_address(ready) else {
        return vec![format!(
            "NovaSight Studio authenticated URL: {}",
            access_url(&ready.url, access)
        )];
    };
    if !address.ip().is_unspecified() {
        return vec![format!(
            "NovaSight Studio authenticated URL: {}",
            access_url(&ready.url, access)
        )];
    }
    let lan_url = lan_ip
        .map(|ip| http_url(SocketAddr::new(ip, address.port())))
        .unwrap_or_else(|| format!("http://<this-machine-ip>:{}/", address.port()));
    vec![
        format!(
            "NovaSight Studio LAN authenticated URL: {}",
            access_url(&lan_url, access)
        ),
        format!(
            "NovaSight Studio local authenticated URL: {}",
            access_url(&http_url(connectable_local_address(address)), access)
        ),
    ]
}

fn access_url(url: &str, access: &WebAccess) -> String {
    format!("{url}#access={}", access.code)
}

fn create_web_access(layout: &PortableLayout) -> Result<WebAccess> {
    let access = WebAccess {
        code: random_access_code(),
    };
    persist_web_access(layout, &access)?;
    Ok(access)
}

fn persist_web_access(layout: &PortableLayout, access: &WebAccess) -> Result<()> {
    let path = layout.root.join(WEB_ACCESS_FILE);
    persist_private_code(&path, &access.code, "Web access")
}

fn create_temporary_license_access(
    layout: &PortableLayout,
) -> Result<Option<TemporaryLicenseAccess>> {
    let path = layout.root.join(TEMPORARY_LICENSE_ACCESS_FILE);
    if !cfg!(debug_assertions) {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("remove stale {}", path.display()));
            }
        }
        return Ok(None);
    }
    let access = TemporaryLicenseAccess {
        code: random_access_code(),
    };
    persist_private_code(&path, &access.code, "temporary license")?;
    Ok(Some(access))
}

fn random_access_code() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn persist_private_code(path: &Path, code: &str, label: &str) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path)
        && !metadata.file_type().is_file()
    {
        bail!("{label} path is not a regular file: {}", path.display());
    }
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("open private {label} file {}", path.display()))?;
    file.write_all(code.as_bytes())?;
    file.sync_all()?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

fn print_temporary_license_access(
    layout: &PortableLayout,
    access: Option<&TemporaryLicenseAccess>,
) {
    let Some(access) = access else {
        return;
    };
    eprintln!(
        "NovaSight temporary license code (current debug daemon only): {}",
        access.code
    );
    eprintln!(
        "NovaSight temporary license code file: {}",
        layout.root.join(TEMPORARY_LICENSE_ACCESS_FILE).display()
    );
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

fn detect_lan_ip() -> Option<IpAddr> {
    detect_lan_ip_by_udp_route().or_else(detect_lan_ip_from_hostname)
}

fn detect_lan_ip_by_udp_route() -> Option<IpAddr> {
    let socket = UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))).ok()?;
    socket.connect(SocketAddr::from(([8, 8, 8, 8], 80))).ok()?;
    candidate_lan_ip(socket.local_addr().ok()?.ip())
}

#[cfg(target_os = "linux")]
fn detect_lan_ip_from_hostname() -> Option<IpAddr> {
    let output = StdCommand::new("hostname").arg("-I").output().ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .split_whitespace()
        .filter_map(|candidate| candidate.parse().ok())
        .find_map(candidate_lan_ip)
}

#[cfg(not(target_os = "linux"))]
fn detect_lan_ip_from_hostname() -> Option<IpAddr> {
    None
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
    use super::random_access_code;

    #[test]
    fn generated_access_codes_are_full_width_hex_and_not_reused() {
        let first = random_access_code();
        let second = random_access_code();
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }
}
