use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use novasight_config::studio_endpoint_contract;
use tokio::process::{Child, Command};
use tokio::time;

use super::{
    LayoutMode, PROCESS_READY_TIMEOUT, PortableLayout, ReadyDocument,
    create_temporary_license_access, create_web_access, health_check, http_url, print_studio_urls,
    print_temporary_license_access, read_ready_file, spawn_daemon, spawn_web, stop_child,
    stop_owned_daemon, wait_for_daemon_ready, wait_for_shutdown_signal, wait_for_web_ready,
};

const FRONTEND_READY_TIMEOUT: Duration = Duration::from_secs(20);

pub(super) async fn run(layout: &PortableLayout) -> Result<()> {
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set workspace root {}", layout.root.display()))?;
    if let Ok(ready) = read_ready_file(&layout.web_ready_file)
        && health_check(&ready.address)
    {
        bail!(
            "another NovaSight Web/API gateway is already running at {}; stop it before starting frontend development",
            ready.url
        );
    }
    validate_artifacts(layout)?;
    let _ = fs::remove_file(&layout.web_ready_file);
    let _ = fs::remove_file(&layout.daemon_ready_file);
    let access = create_web_access(layout)?;
    let temporary_license = create_temporary_license_access(layout)?;

    let mut daemon = spawn_daemon(layout, temporary_license.as_ref())?;
    if let Err(error) = wait_for_daemon_ready(layout, &mut daemon, PROCESS_READY_TIMEOUT).await {
        let _ = stop_child("novasightd", &mut daemon).await;
        return Err(error);
    }
    let mut web = match spawn_web(layout, &access, true) {
        Ok(web) => web,
        Err(error) => {
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            return Err(error);
        }
    };
    let api_ready =
        match wait_for_web_ready(layout, &mut web, &mut daemon, PROCESS_READY_TIMEOUT).await {
            Ok(ready) => ready,
            Err(error) => {
                let _ = stop_child("novasight-web", &mut web).await;
                let _ = stop_owned_daemon(layout, &mut daemon).await;
                return Err(error);
            }
        };
    let mut vite = match spawn_vite(layout) {
        Ok(vite) => vite,
        Err(error) => {
            let _ = stop_child("novasight-web", &mut web).await;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            return Err(error);
        }
    };
    let studio_ready = studio_ready_document()?;
    if let Err(error) = wait_until_ready(
        &studio_ready.address,
        &mut daemon,
        &mut web,
        &mut vite,
        FRONTEND_READY_TIMEOUT,
    )
    .await
    {
        let _ = stop_child("Vite", &mut vite).await;
        let _ = stop_child("novasight-web", &mut web).await;
        let _ = stop_owned_daemon(layout, &mut daemon).await;
        return Err(error);
    }
    eprintln!(
        "NOVASIGHT_FRONTEND_DEV_READY: authenticated API {} · Vite {} · daemon IPC",
        api_ready.url, studio_ready.url
    );
    print_studio_urls(&studio_ready, &access);
    print_temporary_license_access(layout, temporary_license.as_ref());
    supervise(layout, daemon, web, vite).await
}

fn validate_artifacts(layout: &PortableLayout) -> Result<()> {
    if layout.mode != LayoutMode::Developer {
        bail!("frontend development is available only from a NovaSight source workspace");
    }
    let vite = vite_executable(layout);
    if !vite.is_file() {
        bail!(
            "Vite is missing at {}; install frontend dependencies explicitly before running NovaSight",
            vite.display()
        );
    }
    refresh_rust_artifacts(layout)?;
    let required = [
        ("novasightd", layout.daemon.clone()),
        ("novasight-web", layout.web.clone()),
        ("novasightctl", layout.control.clone()),
    ];
    let missing = required
        .iter()
        .filter(|(_, path)| !path.is_file())
        .map(|(label, path)| format!("{label}: {}", path.display()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "Cargo completed but frontend development artifacts are missing:\n{}",
            missing.join("\n")
        );
    }
    validate_web_capability(layout)
}

fn refresh_rust_artifacts(layout: &PortableLayout) -> Result<()> {
    eprintln!("NOVASIGHT_DEV_REFRESH: checking daemon, Web/API, and local client with Cargo");
    let status = StdCommand::new("cargo")
        .args([
            "build",
            "--locked",
            "-p",
            "novasightd",
            "-p",
            "novasight-web",
            "-p",
            "novasightctl",
        ])
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("refresh frontend-development Rust binaries with Cargo")?;
    if !status.success() {
        bail!("Cargo could not refresh frontend-development Rust binaries: {status}");
    }
    Ok(())
}

fn validate_web_capability(layout: &PortableLayout) -> Result<()> {
    let output = StdCommand::new(&layout.web)
        .arg("--help")
        .current_dir(&layout.root)
        .output()
        .with_context(|| format!("inspect {} capabilities", layout.web.display()))?;
    let help = String::from_utf8_lossy(&output.stdout);
    if output.status.success() && help.contains("--frontend-dev") {
        return Ok(());
    }
    bail!(
        "Cargo produced a novasight-web without the required --frontend-dev capability: {}",
        layout.web.display()
    )
}

fn vite_executable(layout: &PortableLayout) -> PathBuf {
    let executable = if cfg!(windows) { "vite.cmd" } else { "vite" };
    layout.root.join("web/node_modules/.bin").join(executable)
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
    web: &mut Child,
    vite: &mut Child,
    timeout: Duration,
) -> Result<()> {
    let started = Instant::now();
    loop {
        for (label, child) in [
            ("novasightd", &mut *daemon),
            ("novasight-web", &mut *web),
            ("Vite", &mut *vite),
        ] {
            if let Some(status) = child.try_wait().with_context(|| format!("check {label}"))? {
                bail!("{label} exited before frontend readiness with status {status}");
            }
        }
        if health_check(address) {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            bail!(
                "Vite did not proxy a healthy authenticated Studio within {timeout:?} at {address}"
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn supervise(
    layout: &PortableLayout,
    mut daemon: Child,
    mut web: Child,
    mut vite: Child,
) -> Result<()> {
    eprintln!("NOVASIGHT_RUNNING: press Ctrl+C to stop Vite, Web/API, and daemon");
    tokio::select! {
        status = daemon.wait() => {
            let status = status.context("wait for novasightd")?;
            let _ = stop_child("Vite", &mut vite).await;
            let _ = stop_child("novasight-web", &mut web).await;
            bail!("novasightd exited during frontend development with status {status}")
        }
        status = web.wait() => {
            let status = status.context("wait for novasight-web")?;
            let _ = stop_child("Vite", &mut vite).await;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("novasight-web exited during frontend development with status {status}")
        }
        status = vite.wait() => {
            let status = status.context("wait for Vite")?;
            let _ = stop_child("novasight-web", &mut web).await;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("Vite exited during frontend development with status {status}")
        }
        signal = wait_for_shutdown_signal() => {
            signal?;
            let vite_stop = stop_child("Vite", &mut vite).await;
            let web_stop = stop_child("novasight-web", &mut web).await;
            let daemon_stop = stop_owned_daemon(layout, &mut daemon).await;
            vite_stop.and(web_stop).and(daemon_stop)
        }
    }
}
