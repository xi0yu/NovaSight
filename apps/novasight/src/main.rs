use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use novasight_store::config::YamlConfigRepository;
use serde::Deserialize;
use serde_yaml::{Mapping, Number, Value};

const CONFIG_PATH: &str = "data/novasight.yaml";
const DATA_DIR: &str = "data";
const MODEL_DIR: &str = "data/models";
const DATABASE_PATH: &str = "data/novasight.db";
const LICENSE_PATH: &str = "data/license.json";
const LOG_DIR: &str = "logs";
const RUN_DIR: &str = "run";
const CONTROL_SOCKET: &str = "run/novasightd.sock";
const READY_FILE: &str = "run/ready.json";
const DAEMON_READY_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Parser, Debug)]
#[command(name = "novasight", about = "NovaSight portable launcher")]
struct Args {}

#[derive(Clone, Debug)]
struct PortableLayout {
    root: PathBuf,
    daemon: PathBuf,
    config: PathBuf,
    data_dir: PathBuf,
    model_dir: PathBuf,
    log_dir: PathBuf,
    run_dir: PathBuf,
    ready_file: PathBuf,
    daemon_log: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ReadyDocument {
    address: String,
    url: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("NOVASIGHT_LAUNCH_FAILED: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let _args = Args::parse();
    let layout = PortableLayout::discover()?;
    prepare_layout(&layout)?;
    ensure_portable_config(&layout)?;
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set bundle root {}", layout.root.display()))?;

    if let Some(ready) = read_ready_file(&layout.ready_file).ok()
        && health_check(&ready.address)
    {
        open_studio(&ready.url);
        return Ok(());
    }

    let _ = fs::remove_file(&layout.ready_file);
    let mut child = spawn_daemon(&layout)?;
    let ready = wait_for_ready(&layout.ready_file, &mut child, DAEMON_READY_TIMEOUT)?;
    open_studio(&ready.url);
    Ok(())
}

impl PortableLayout {
    fn discover() -> Result<Self> {
        let current_exe = std::env::current_exe().context("read current executable path")?;
        let executable_dir = current_exe
            .parent()
            .ok_or_else(|| anyhow!("current executable has no parent directory"))?;
        let root = infer_bundle_root(executable_dir);
        let daemon = resolve_daemon(&root, executable_dir)?;
        let config = root.join(CONFIG_PATH);
        let data_dir = root.join(DATA_DIR);
        let model_dir = root.join(MODEL_DIR);
        let log_dir = root.join(LOG_DIR);
        let run_dir = root.join(RUN_DIR);
        let ready_file = root.join(READY_FILE);
        let daemon_log = log_dir.join("novasightd.log");
        Ok(Self {
            root,
            daemon,
            config,
            data_dir,
            model_dir,
            log_dir,
            run_dir,
            ready_file,
            daemon_log,
        })
    }
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

fn resolve_daemon(root: &Path, executable_dir: &Path) -> Result<PathBuf> {
    let name = executable_name("novasightd");
    let candidates = [
        root.join("bin").join(name),
        root.join(name),
        executable_dir.join(name),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| anyhow!("novasightd was not found below {}", root.display()))
}

fn executable_name(name: &'static str) -> &'static str {
    if cfg!(windows) {
        match name {
            "novasightd" => "novasightd.exe",
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
        Value::String("127.0.0.1".to_owned()),
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

fn spawn_daemon(layout: &PortableLayout) -> Result<std::process::Child> {
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
    command
        .spawn()
        .with_context(|| format!("start {}", layout.daemon.display()))
}

fn wait_for_ready(
    ready_file: &Path,
    child: &mut std::process::Child,
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
        thread::sleep(Duration::from_millis(100));
    }
}

fn read_ready_file(path: &Path) -> Result<ReadyDocument> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parse {}", path.display()))
}

fn health_check(address: &str) -> bool {
    let Ok(address) = address.parse::<SocketAddr>() else {
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

fn open_studio(url: &str) {
    println!("{url}");
    if let Err(error) = open_browser(url) {
        eprintln!("NOVASIGHT_BROWSER_OPEN_SKIPPED: {error:#}; open {url} manually");
    }
}

#[cfg(target_os = "linux")]
fn open_browser(url: &str) -> Result<()> {
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        bail!("no graphical session detected");
    }
    Command::new("xdg-open")
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
    Command::new("open")
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
    Command::new("cmd")
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

    use super::*;

    #[test]
    fn bin_launcher_uses_parent_as_bundle_root() {
        assert_eq!(
            infer_bundle_root(Path::new("/opt/NovaSight/bin")),
            PathBuf::from("/opt/NovaSight")
        );
    }

    #[test]
    fn root_launcher_uses_own_directory_as_bundle_root() {
        assert_eq!(
            infer_bundle_root(Path::new("/opt/NovaSight")),
            PathBuf::from("/opt/NovaSight")
        );
    }

    #[test]
    fn mapping_field_update_is_idempotent() {
        let mut document = Value::Mapping(Mapping::new());
        assert!(
            set_mapping_field(
                &mut document,
                "server",
                "control_socket",
                Value::String(CONTROL_SOCKET.to_owned()),
            )
            .unwrap()
        );
        assert!(
            !set_mapping_field(
                &mut document,
                "server",
                "control_socket",
                Value::String(CONTROL_SOCKET.to_owned()),
            )
            .unwrap()
        );
    }

    #[test]
    fn portable_config_is_rewritten_to_package_local_paths() {
        let root =
            std::env::temp_dir().join(format!("novasight-launcher-config-{}", std::process::id()));
        let layout = PortableLayout {
            root: root.clone(),
            daemon: root.join("bin/novasightd"),
            config: root.join(CONFIG_PATH),
            data_dir: root.join(DATA_DIR),
            model_dir: root.join(MODEL_DIR),
            log_dir: root.join(LOG_DIR),
            run_dir: root.join(RUN_DIR),
            ready_file: root.join(READY_FILE),
            daemon_log: root.join(LOG_DIR).join("novasightd.log"),
        };

        prepare_layout(&layout).unwrap();
        ensure_portable_config(&layout).unwrap();

        let config = YamlConfigRepository::load(&layout.config).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 0);
        assert_eq!(config.server.control_socket, Path::new(CONTROL_SOCKET));
        assert_eq!(config.paths.data_dir, Path::new(DATA_DIR));
        assert_eq!(config.paths.model_dir, Path::new(MODEL_DIR));
        assert_eq!(config.paths.database, Path::new(DATABASE_PATH));
        assert_eq!(config.paths.license, Path::new(LICENSE_PATH));

        fs::remove_dir_all(root).unwrap();
    }
}
