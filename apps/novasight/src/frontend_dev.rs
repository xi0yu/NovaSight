use std::fs::{self, OpenOptions};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use novasight_config::studio_endpoint_contract;
use tokio::process::{Child, Command};
use tokio::time;

use super::{
    DAEMON_READY_TIMEOUT, LayoutMode, PortableLayout, ReadyDocument, health_check, http_url,
    print_studio_urls, read_ready_file, stop_owned_daemon, wait_for_ready,
    wait_for_shutdown_signal,
};

const FRONTEND_READY_TIMEOUT: Duration = Duration::from_secs(20);

pub(super) async fn run(layout: &PortableLayout) -> Result<()> {
    validate_artifacts(layout)?;
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set workspace root {}", layout.root.display()))?;

    if let Some(ready) = read_ready_file(&layout.ready_file).ok()
        && health_check(&ready.address)
    {
        bail!(
            "another NovaSight daemon is already running at {}; stop it before starting frontend development",
            ready.url
        );
    }

    let _ = fs::remove_file(&layout.ready_file);
    let mut daemon = spawn_daemon(layout)?;
    let api_ready = match wait_for_ready(
        &layout.ready_file,
        &layout.daemon_log,
        &mut daemon,
        DAEMON_READY_TIMEOUT,
    )
    .await
    {
        Ok(ready) => ready,
        Err(error) => {
            let _ = stop_child("novasightd", &mut daemon).await;
            return Err(error);
        }
    };

    let mut vite = match spawn_vite(layout) {
        Ok(child) => child,
        Err(error) => {
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            return Err(error);
        }
    };
    let studio_ready = studio_ready_document()?;
    if let Err(error) = wait_until_ready(
        &studio_ready.address,
        &mut daemon,
        &mut vite,
        FRONTEND_READY_TIMEOUT,
    )
    .await
    {
        let _ = stop_child("Vite", &mut vite).await;
        let _ = stop_owned_daemon(layout, &mut daemon).await;
        return Err(error);
    }

    eprintln!(
        "NOVASIGHT_FRONTEND_DEV_READY: Rust API {} · Vite {}",
        api_ready.url, studio_ready.url
    );
    print_studio_urls(&studio_ready);
    supervise(layout, daemon, vite).await
}

fn validate_artifacts(layout: &PortableLayout) -> Result<()> {
    if layout.mode != LayoutMode::Developer {
        bail!("--frontend-dev is available only from a NovaSight source workspace");
    }

    let required = [
        ("novasightd", layout.daemon.clone()),
        ("novasightctl", layout.control.clone()),
        ("Vite", vite_executable(layout)),
    ];
    let missing = required
        .iter()
        .filter(|(_, path)| !path.is_file())
        .map(|(label, path)| format!("{label}: {}", path.display()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "frontend development artifacts are missing; startup does not build or install anything implicitly:\n{}\nprepare them explicitly before running NovaSight",
            missing.join("\n")
        );
    }
    Ok(())
}

fn vite_executable(layout: &PortableLayout) -> PathBuf {
    let executable = if cfg!(windows) { "vite.cmd" } else { "vite" };
    layout.root.join("web/node_modules/.bin").join(executable)
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
        .arg("--frontend-dev")
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log))
        .kill_on_drop(true);
    command
        .spawn()
        .with_context(|| format!("start {} --frontend-dev", layout.daemon.display()))
}

fn spawn_vite(layout: &PortableLayout) -> Result<Child> {
    let executable = vite_executable(layout);
    let mut command = Command::new(&executable);
    command
        .current_dir(layout.root.join("web"))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command
        .spawn()
        .with_context(|| format!("start Vite from {}", executable.display()))
}

fn studio_ready_document() -> Result<ReadyDocument> {
    let endpoint = &studio_endpoint_contract().studio;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let parsed = address
        .parse::<SocketAddr>()
        .with_context(|| format!("parse Studio endpoint {address}"))?;
    Ok(ReadyDocument {
        address,
        url: http_url(parsed),
    })
}

async fn wait_until_ready(
    address: &str,
    daemon: &mut Child,
    vite: &mut Child,
    timeout: Duration,
) -> Result<()> {
    let started = Instant::now();
    loop {
        if let Some(status) = daemon.try_wait().context("check novasightd process")? {
            bail!("novasightd exited before frontend readiness with status {status}");
        }
        if let Some(status) = vite.try_wait().context("check Vite process")? {
            bail!("Vite exited before frontend readiness with status {status}");
        }
        if health_check(address) {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            bail!("Vite did not proxy a healthy Studio within {timeout:?} at {address}");
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn supervise(layout: &PortableLayout, mut daemon: Child, mut vite: Child) -> Result<()> {
    eprintln!("NOVASIGHT_RUNNING: press Ctrl+C to stop Rust API and Vite");
    tokio::select! {
        status = daemon.wait() => {
            let status = status.context("wait for novasightd process")?;
            let _ = stop_child("Vite", &mut vite).await;
            bail!("novasightd exited while frontend development was running with status {status}")
        }
        status = vite.wait() => {
            let status = status.context("wait for Vite process")?;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("Vite exited while frontend development was running with status {status}")
        }
        signal = wait_for_shutdown_signal() => {
            signal?;
            let vite_stop = stop_child("Vite", &mut vite).await;
            let daemon_stop = stop_owned_daemon(layout, &mut daemon).await;
            vite_stop.and(daemon_stop)
        }
    }
}

async fn stop_child(label: &str, child: &mut Child) -> Result<()> {
    if child
        .try_wait()
        .with_context(|| format!("check {label} process"))?
        .is_some()
    {
        return Ok(());
    }
    eprintln!("NOVASIGHT_SHUTDOWN_REQUESTED: stopping {label}");
    child
        .start_kill()
        .with_context(|| format!("stop {label} process"))?;
    let _ = child
        .wait()
        .await
        .with_context(|| format!("wait for {label} process"))?;
    Ok(())
}
