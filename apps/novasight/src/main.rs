use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use novasight_config::YamlConfigRepository;
use serde::Deserialize;
use serde_yaml::{Mapping, Number, Value};
use tokio::process::{Child, Command};
use tokio::time;

const CONFIG_PATH: &str = "data/novasight.yaml";
const DATA_DIR: &str = "data";
const MODEL_DIR: &str = "data/models";
const DATABASE_PATH: &str = "data/novasight.db";
const LICENSE_PATH: &str = "data/license.json";
const LOG_DIR: &str = "logs";
const RUN_DIR: &str = "run";
const CONTROL_SOCKET: &str = "run/novasightd.sock";
const READY_FILE: &str = "run/ready.json";
const PORTABLE_BIND_HOST: &str = "0.0.0.0";
const DAEMON_READY_TIMEOUT: Duration = Duration::from_secs(20);
const DAEMON_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Parser, Debug)]
#[command(name = "novasight", about = "NovaSight portable launcher")]
struct Args {}

#[derive(Clone, Debug)]
struct PortableLayout {
    mode: LayoutMode,
    root: PathBuf,
    daemon: PathBuf,
    control: PathBuf,
    config: PathBuf,
    data_dir: PathBuf,
    model_dir: PathBuf,
    log_dir: PathBuf,
    run_dir: PathBuf,
    ready_file: PathBuf,
    daemon_log: PathBuf,
    web_root_override: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayoutMode {
    Package,
    Developer,
}

#[derive(Debug, Deserialize)]
struct ReadyDocument {
    address: String,
    url: String,
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
    let _args = Args::parse();
    let layout = PortableLayout::discover()?;
    prepare_layout(&layout)?;
    ensure_developer_artifacts(&layout)?;
    ensure_portable_config(&layout)?;
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set bundle root {}", layout.root.display()))?;

    if let Some(ready) = read_ready_file(&layout.ready_file).ok()
        && health_check(&ready.address)
    {
        open_studio(&ready);
        supervise_existing_daemon(&layout, &ready).await?;
        return Ok(());
    }

    let _ = fs::remove_file(&layout.ready_file);
    let mut child = spawn_daemon(&layout)?;
    let ready = wait_for_ready(&layout.ready_file, &mut child, DAEMON_READY_TIMEOUT).await?;
    open_studio(&ready);
    supervise_spawned_daemon(&layout, child).await?;
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
                workspace.clone(),
                executable_dir.join(executable_name("novasightd")),
                executable_dir.join(executable_name("novasightctl")),
                Some(PathBuf::from("out/web")),
            ));
        }
        let root = infer_bundle_root(executable_dir);
        let daemon = resolve_binary(&root, executable_dir, "novasightd")?;
        let control = resolve_binary(&root, executable_dir, "novasightctl")?;
        Ok(Self::new(LayoutMode::Package, root, daemon, control, None))
    }

    fn new(
        mode: LayoutMode,
        root: PathBuf,
        daemon: PathBuf,
        control: PathBuf,
        web_root_override: Option<PathBuf>,
    ) -> Self {
        let config = root.join(CONFIG_PATH);
        let data_dir = root.join(DATA_DIR);
        let model_dir = root.join(MODEL_DIR);
        let log_dir = root.join(LOG_DIR);
        let run_dir = root.join(RUN_DIR);
        let ready_file = root.join(READY_FILE);
        let daemon_log = log_dir.join("novasightd.log");
        Self {
            mode,
            root,
            daemon,
            control,
            config,
            data_dir,
            model_dir,
            log_dir,
            run_dir,
            ready_file,
            daemon_log,
            web_root_override,
        }
    }
}

fn developer_workspace_root(executable_dir: &Path) -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent()?.parent()?.to_owned();
    if !looks_like_workspace_root(&workspace) {
        return None;
    }
    let cargo_artifacts = workspace.join("out/cargo");
    executable_dir
        .starts_with(cargo_artifacts)
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
    let candidates = [
        root.join("bin").join(name),
        root.join(name),
        executable_dir.join(name),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| anyhow!("{binary} was not found below {}", root.display()))
}

fn executable_name(name: &'static str) -> &'static str {
    if cfg!(windows) {
        match name {
            "novasightd" => "novasightd.exe",
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
    ensure_private_directory(&layout.run_dir)?;
    Ok(())
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("set private permissions on {}", path.display()))?;
    }
    Ok(())
}

fn ensure_developer_artifacts(layout: &PortableLayout) -> Result<()> {
    if layout.mode != LayoutMode::Developer {
        return Ok(());
    }
    run_developer_command(
        &layout.root,
        "cargo",
        [
            "build",
            "--locked",
            "-p",
            "novasightd",
            "-p",
            "novasightctl",
        ],
        &[],
    )?;
    run_developer_command(
        &layout.root,
        "pnpm",
        ["--dir", "web", "build"],
        &[("CI", "true")],
    )
}

fn run_developer_command<I, S>(
    cwd: &Path,
    program: &str,
    args: I,
    envs: &[(&str, &str)],
) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut command = StdCommand::new(program);
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

fn ensure_portable_config(layout: &PortableLayout) -> Result<()> {
    if !layout.config.exists() {
        YamlConfigRepository::initialize_default(&layout.config)
            .with_context(|| format!("initialize {}", layout.config.display()))?;
    }

    let mut document: Value = serde_yaml::from_reader(
        File::open(&layout.config).with_context(|| format!("open {}", layout.config.display()))?,
    )
    .with_context(|| format!("parse {}", layout.config.display()))?;
    let mut changed = false;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "host",
        Value::String(PORTABLE_BIND_HOST.to_owned()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "port",
        Value::Number(Number::from(0)),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "server",
        "control_socket",
        Value::String(CONTROL_SOCKET.to_owned()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "paths",
        "data_dir",
        Value::String(DATA_DIR.to_owned()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "paths",
        "model_dir",
        Value::String(MODEL_DIR.to_owned()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "paths",
        "database",
        Value::String(DATABASE_PATH.to_owned()),
    )?;
    changed |= set_mapping_field(
        &mut document,
        "paths",
        "license",
        Value::String(LICENSE_PATH.to_owned()),
    )?;
    if changed {
        let current = YamlConfigRepository::load(&layout.config)
            .with_context(|| format!("load {}", layout.config.display()))?;
        let repository = YamlConfigRepository::new(&layout.config);
        repository
            .replace_document(document, current.revision)
            .with_context(|| format!("save {}", layout.config.display()))?;
    }
    Ok(())
}

fn set_mapping_field(document: &mut Value, section: &str, key: &str, value: Value) -> Result<bool> {
    let root = document
        .as_mapping_mut()
        .ok_or_else(|| anyhow!("configuration root must be a mapping"))?;
    let section_key = Value::String(section.to_owned());
    let section_value = root
        .entry(section_key)
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

fn spawn_daemon(layout: &PortableLayout) -> Result<Child> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&layout.daemon_log)
        .with_context(|| format!("open {}", layout.daemon_log.display()))?;
    let stdout = log
        .try_clone()
        .with_context(|| format!("clone {}", layout.daemon_log.display()))?;
    let mut command = Command::new(&layout.daemon);
    command
        .current_dir(&layout.root)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log));
    if let Some(web_root) = &layout.web_root_override {
        command.env("NOVASIGHT_WEB_ROOT", web_root);
    }
    command
        .spawn()
        .with_context(|| format!("start {}", layout.daemon.display()))
}

async fn wait_for_ready(
    ready_file: &Path,
    child: &mut Child,
    timeout: Duration,
) -> Result<ReadyDocument> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().context("check novasightd process")? {
            bail!("novasightd exited before ready with status {status}");
        }
        if let Ok(ready) = read_ready_file(ready_file)
            && health_check(&ready.address)
        {
            return Ok(ready);
        }
        if started.elapsed() >= timeout {
            bail!(
                "novasightd did not become ready within {:?}; see {}",
                timeout,
                ready_file.display()
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn supervise_existing_daemon(layout: &PortableLayout, ready: &ReadyDocument) -> Result<()> {
    eprintln!("NOVASIGHT_RUNNING: press Ctrl+C to stop NovaSight");
    wait_for_shutdown_signal().await?;
    request_daemon_shutdown(layout).await?;
    wait_until_daemon_stops(&ready.address, DAEMON_SHUTDOWN_TIMEOUT).await
}

async fn supervise_spawned_daemon(layout: &PortableLayout, mut child: Child) -> Result<()> {
    eprintln!("NOVASIGHT_RUNNING: press Ctrl+C to stop NovaSight");
    tokio::select! {
        status = child.wait() => {
            let status = status.context("wait for novasightd process")?;
            if status.success() {
                Ok(())
            } else {
                bail!("novasightd exited with status {status}")
            }
        }
        signal = wait_for_shutdown_signal() => {
            signal?;
            stop_owned_daemon(layout, &mut child).await
        }
    }
}

async fn stop_owned_daemon(layout: &PortableLayout, child: &mut Child) -> Result<()> {
    eprintln!("NOVASIGHT_SHUTDOWN_REQUESTED: stopping novasightd");
    if let Err(error) = request_daemon_shutdown(layout).await {
        eprintln!("NOVASIGHT_SHUTDOWN_FALLBACK: {error:#}");
    }
    match time::timeout(DAEMON_SHUTDOWN_TIMEOUT, child.wait()).await {
        Ok(status) => {
            let _ = status.context("wait for novasightd process")?;
            Ok(())
        }
        Err(_) => {
            child
                .start_kill()
                .context("force stop unresponsive novasightd")?;
            let _ = child.wait().await.context("wait for killed novasightd")?;
            Ok(())
        }
    }
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

async fn wait_until_daemon_stops(address: &str, timeout: Duration) -> Result<()> {
    let started = Instant::now();
    loop {
        if !health_check(address) {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            bail!("novasightd did not stop within {timeout:?}");
        }
        time::sleep(Duration::from_millis(100)).await;
    }
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
    {
        tokio::signal::ctrl_c().await.context("wait for Ctrl+C")
    }
}

fn read_ready_file(path: &Path) -> Result<ReadyDocument> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
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

fn open_studio(ready: &ReadyDocument) {
    for message in studio_ready_messages(ready, detect_lan_ip()) {
        println!("{message}");
    }
    let url = browser_open_url(ready);
    if let Err(error) = open_browser(&url) {
        eprintln!(
            "NOVASIGHT_BROWSER_OPEN_SKIPPED: {error:#}; open the NovaSight Studio Web UI URL above manually"
        );
    }
}

fn studio_ready_messages(ready: &ReadyDocument, lan_ip: Option<IpAddr>) -> Vec<String> {
    let Some(address) = ready_address(ready) else {
        return vec![format!("NovaSight Studio Web UI: {}", ready.url)];
    };
    if !address.ip().is_unspecified() {
        return vec![format!("NovaSight Studio Web UI: {}", ready.url)];
    }

    let mut messages = vec![format!("NovaSight Studio Web UI: {}", http_url(address))];
    let lan_url = lan_ip
        .map(|ip| http_url(SocketAddr::new(ip, address.port())))
        .unwrap_or_else(|| format!("http://<this-machine-ip>:{}/", address.port()));
    messages.push(format!("NovaSight Studio Web UI (LAN): {lan_url}"));
    messages.push(format!(
        "NovaSight Studio Web UI (this machine): {}",
        http_url(connectable_local_address(address))
    ));
    messages
}

fn browser_open_url(ready: &ReadyDocument) -> String {
    ready_address(ready)
        .map(connectable_local_address)
        .map(http_url)
        .unwrap_or_else(|| ready.url.clone())
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
    let ip = socket.local_addr().ok()?.ip();
    candidate_lan_ip(ip)
}

#[cfg(target_os = "linux")]
fn detect_lan_ip_from_hostname() -> Option<IpAddr> {
    let output = StdCommand::new("hostname").arg("-I").output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    first_lan_ip_from_whitespace(&text)
}

#[cfg(not(target_os = "linux"))]
fn detect_lan_ip_from_hostname() -> Option<IpAddr> {
    None
}

#[cfg(target_os = "linux")]
fn first_lan_ip_from_whitespace(text: &str) -> Option<IpAddr> {
    text.split_whitespace()
        .filter_map(|candidate| candidate.parse().ok())
        .find_map(candidate_lan_ip)
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
        .context("start browser with xdg-open")?;
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
