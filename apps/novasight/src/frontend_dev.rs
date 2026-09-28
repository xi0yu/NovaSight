use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use novasight_config::studio_endpoint_contract;
use tokio::process::Child;
use tokio::time;

use super::{
    LayoutMode, PROCESS_SHUTDOWN_TIMEOUT, PortableLayout, ReadyDocument,
    create_temporary_license_access, ensure_portable_config, health_check, http_url, lifecycle,
    log_tail, print_studio_urls, print_temporary_license_access, read_daemon_ready_file,
    read_ready_file, request_daemon_shutdown, spawn_daemon, spawn_logged_process, spawn_web,
    stop_child, stop_owned_daemon,
};

const FRONTEND_READY_TIMEOUT: Duration = Duration::from_secs(20);

pub(super) async fn run(layout: &PortableLayout) -> Result<()> {
    std::env::set_current_dir(&layout.root)
        .with_context(|| format!("set workspace root {}", layout.root.display()))?;
    ensure_portable_config(layout)?;
    validate_artifacts(layout)?;
    stop_preexisting_stack(layout).await?;
    stop_preexisting_daemon(layout).await?;
    let _ = fs::remove_file(&layout.web_ready_file);
    let _ = fs::remove_file(&layout.daemon_ready_file);
    let temporary_license = create_temporary_license_access(layout)?;

    let mut daemon = spawn_daemon(layout, temporary_license.as_ref())?;
    let mut web = match spawn_web(layout, true) {
        Ok(web) => web,
        Err(error) => {
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
        layout,
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
    print_studio_urls(&studio_ready);
    print_temporary_license_access(temporary_license.as_ref());
    lifecycle::supervise(layout, daemon, web, Some(vite)).await
}

async fn stop_preexisting_stack(layout: &PortableLayout) -> Result<()> {
    let Ok(ready) = read_ready_file(&layout.web_ready_file) else {
        return Ok(());
    };
    if !health_check(&ready.address) {
        return Ok(());
    }
    if !daemon_status_succeeds(layout).await? {
        bail!(
            "NovaSight Web/API is running at {}, but its daemon is not reachable through the local control channel; stop that standalone gateway before restarting frontend development",
            ready.url
        );
    }
    eprintln!("NOVASIGHT_EXISTING_STACK: 检测到已有开发实例；正在安全退出旧实例并启动当前代码");
    request_daemon_shutdown(layout).await?;
    let started = Instant::now();
    while health_check(&ready.address) {
        if started.elapsed() >= PROCESS_SHUTDOWN_TIMEOUT {
            bail!(
                "旧 NovaSight Web/API 在 daemon 退出后仍未释放 {}",
                ready.address
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

async fn stop_preexisting_daemon(layout: &PortableLayout) -> Result<()> {
    if !daemon_status_succeeds(layout).await? {
        return Ok(());
    }
    eprintln!(
        "NOVASIGHT_EXISTING_DAEMON: Web 已停止，但检测到已有 novasightd；先通过本地控制通道有序停止，再启动本次工作区实例"
    );
    request_daemon_shutdown(layout).await?;
    let started = Instant::now();
    while daemon_status_succeeds(layout).await? {
        if started.elapsed() >= PROCESS_SHUTDOWN_TIMEOUT {
            bail!("已有 novasightd 在收到关闭请求后仍未退出；请先结束旧实例，再重新启动 NovaSight");
        }
        time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

async fn daemon_status_succeeds(layout: &PortableLayout) -> Result<bool> {
    let status = tokio::process::Command::new(&layout.control)
        // Runtime status is license-gated; license status remains available on
        // the trusted local socket and is enough to prove daemon ownership.
        .args(["license", "status"])
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .with_context(|| {
            format!(
                "inspect existing daemon through {}",
                layout.control.display()
            )
        })?;
    Ok(status.success())
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
    let output = StdCommand::new("cargo")
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
        .output()
        .context("refresh frontend-development Rust binaries with Cargo")?;
    if !output.status.success() {
        bail!(
            "Cargo could not refresh frontend-development Rust binaries: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim_end()
        );
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
    spawn_logged_process(
        &executable,
        &layout.root.join("web"),
        &layout.log_dir.join("vite.log"),
        &[],
        &[],
    )
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
    layout: &PortableLayout,
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
                let log = match label {
                    "novasightd" => log_tail(&layout.daemon_log),
                    "novasight-web" => log_tail(&layout.web_log),
                    _ => log_tail(&layout.log_dir.join("vite.log")),
                };
                bail!("{label} exited before frontend readiness with status {status}; {log}");
            }
        }
        let daemon_ready = read_daemon_ready_file(&layout.daemon_ready_file).is_ok_and(|ready| {
            ready.transport == "http1-unix" && !ready.control_socket.trim().is_empty()
        });
        if daemon_ready && health_check(address) {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            bail!(
                "NovaSight frontend stack did not become healthy within {timeout:?} at {address}; daemon: {}; Web/API: {}; Vite: {}",
                log_tail(&layout.daemon_log),
                log_tail(&layout.web_log),
                log_tail(&layout.log_dir.join("vite.log")),
            );
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::os::unix::fs::PermissionsExt;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{
        daemon_status_succeeds, health_check, stop_preexisting_daemon, stop_preexisting_stack,
    };
    use crate::{LayoutMode, PortableLayout};

    #[tokio::test]
    async fn frontend_start_stops_a_daemon_left_running_without_the_web_gateway() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("novasight-existing-daemon-{suffix}"));
        fs::create_dir_all(&root).unwrap();
        let control = root.join("novasightctl");
        fs::write(
            &control,
            "#!/bin/sh\ncase \"$1\" in\n  license) test \"$2\" = status && test ! -f \"$0.stopped\" ;;\n  status) exit 77 ;;\n  shutdown) : > \"$0.stopped\" ;;\n  *) exit 2 ;;\nesac\n",
        )
        .unwrap();
        fs::set_permissions(&control, fs::Permissions::from_mode(0o755)).unwrap();
        let layout = PortableLayout::new(
            LayoutMode::Developer,
            root.clone(),
            root.join("novasightd"),
            root.join("novasight-web"),
            control.clone(),
        );

        assert!(daemon_status_succeeds(&layout).await.unwrap());
        stop_preexisting_daemon(&layout).await.unwrap();
        assert!(!daemon_status_succeeds(&layout).await.unwrap());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn frontend_start_restarts_an_existing_owned_stack() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("novasight-existing-stack-{suffix}"));
        fs::create_dir_all(root.join("run")).unwrap();
        let control = root.join("novasightctl");
        fs::write(
            &control,
            "#!/bin/sh\ncase \"$1\" in\n  license) test \"$2\" = status && test ! -f \"$0.stopped\" ;;\n  status) exit 77 ;;\n  shutdown) : > \"$0.stopped\" ;;\n  *) exit 2 ;;\nesac\n",
        )
        .unwrap();
        fs::set_permissions(&control, fs::Permissions::from_mode(0o755)).unwrap();
        let layout = PortableLayout::new(
            LayoutMode::Developer,
            root.clone(),
            root.join("novasightd"),
            root.join("novasight-web"),
            control.clone(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        fs::write(
            &layout.web_ready_file,
            format!(r#"{{"address":"{address}","url":"http://{address}/"}}"#),
        )
        .unwrap();
        let stopped = control.with_file_name("novasightctl.stopped");
        let gateway = thread::spawn(move || {
            loop {
                if stopped.exists() {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0; 256];
                        let _ = stream.read(&mut request);
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("fake gateway failed: {error}"),
                }
            }
        });

        assert!(health_check(&address.to_string()));
        stop_preexisting_stack(&layout).await.unwrap();
        gateway.join().unwrap();
        assert!(!daemon_status_succeeds(&layout).await.unwrap());

        fs::remove_dir_all(root).unwrap();
    }
}
