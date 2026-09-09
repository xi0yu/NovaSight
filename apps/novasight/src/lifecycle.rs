use std::future;
use std::io::{self, BufRead, IsTerminal};
use std::thread;

use anyhow::{Context, Result, bail};
use tokio::process::Child;
use tokio::sync::mpsc;

use super::{PortableLayout, stop_child, stop_owned_daemon, wait_for_shutdown_signal};

pub(super) async fn supervise(
    layout: &PortableLayout,
    mut daemon: Child,
    mut web: Child,
    mut vite: Option<Child>,
) -> Result<()> {
    let shutdown = wait_for_shutdown_request();
    tokio::pin!(shutdown);

    tokio::select! {
        status = daemon.wait() => {
            let status = status.context("wait for novasightd process")?;
            stop_ingress(&mut vite, &mut web).await;
            bail!("novasightd exited with status {status}")
        }
        status = web.wait() => {
            let status = status.context("wait for novasight-web process")?;
            let _ = stop_optional_vite(&mut vite).await;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("novasight-web exited with status {status}")
        }
        status = wait_optional_vite(&mut vite) => {
            let status = status.context("wait for Vite process")?;
            let _ = stop_child("novasight-web", &mut web).await;
            let _ = stop_owned_daemon(layout, &mut daemon).await;
            bail!("Vite exited with status {status}")
        }
        request = &mut shutdown => {
            request?;
            eprintln!("NovaSight 正在安全退出…");
            let vite_stop = stop_optional_vite(&mut vite).await;
            let web_stop = stop_child("novasight-web", &mut web).await;
            let daemon_stop = stop_owned_daemon(layout, &mut daemon).await;
            vite_stop.and(web_stop).and(daemon_stop)?;
            eprintln!("NovaSight 已安全退出");
            Ok(())
        }
    }
}

async fn wait_optional_vite(vite: &mut Option<Child>) -> std::io::Result<std::process::ExitStatus> {
    match vite {
        Some(vite) => vite.wait().await,
        None => future::pending().await,
    }
}

async fn stop_optional_vite(vite: &mut Option<Child>) -> Result<()> {
    match vite {
        Some(vite) => stop_child("Vite", vite).await,
        None => Ok(()),
    }
}

async fn stop_ingress(vite: &mut Option<Child>, web: &mut Child) {
    let _ = stop_optional_vite(vite).await;
    let _ = stop_child("novasight-web", web).await;
}

async fn wait_for_shutdown_request() -> Result<()> {
    if !io::stdin().is_terminal() {
        return wait_for_shutdown_signal().await;
    }

    eprintln!("novasight> 输入 /exit 安全退出");
    let mut exit_commands = spawn_exit_command_reader()?;
    tokio::select! {
        signal = wait_for_shutdown_signal() => signal,
        command = exit_commands.recv() => match command {
            Some(()) => Ok(()),
            None => wait_for_shutdown_signal().await,
        },
    }
}

fn spawn_exit_command_reader() -> Result<mpsc::UnboundedReceiver<()>> {
    let (sender, receiver) = mpsc::unbounded_channel();
    thread::Builder::new()
        .name("novasight-terminal".to_owned())
        .spawn(move || {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else {
                    break;
                };
                let command = line.trim();
                if is_exit_command(command) {
                    let _ = sender.send(());
                    break;
                }
                if command == "/help" || command == "help" {
                    eprintln!("可用命令：/exit");
                } else if !command.is_empty() {
                    eprintln!("未知命令。输入 /exit 安全退出");
                }
            }
        })
        .context("start launcher terminal reader")?;
    Ok(receiver)
}

fn is_exit_command(command: &str) -> bool {
    matches!(command, "/exit" | "exit" | "/quit" | "quit")
}

#[cfg(test)]
mod tests {
    use super::is_exit_command;

    #[test]
    fn only_explicit_exit_commands_request_shutdown() {
        for command in ["/exit", "exit", "/quit", "quit"] {
            assert!(is_exit_command(command));
        }
        for command in ["", "/status", "stop", "exit now", "/exit --force"] {
            assert!(!is_exit_command(command));
        }
    }
}
