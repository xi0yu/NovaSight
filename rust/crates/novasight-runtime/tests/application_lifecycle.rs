use std::fs;
use std::path::{Path, PathBuf};

use novasight_runtime::{Application, DaemonState, LoadedApplication, RuntimeDependencies};
use tokio::sync::oneshot;
use uuid::Uuid;

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("novasight-application-{}", Uuid::new_v4()));
        fs::create_dir(&path).expect("create temp directory");
        Self(path)
    }

    fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn bootstrap_loads_yaml_before_spawning_the_supervisor() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(
        &path,
        "server:\n  host: 127.0.0.1\n  port: 9070\nreplay:\n  enabled: false\n",
    )
    .expect("write config");

    let loaded = LoadedApplication::load(&path)
        .await
        .expect("load application config");
    assert_eq!(loaded.config_path(), path);
    assert_eq!(loaded.config().server.host, "127.0.0.1");
    assert_eq!(loaded.config().server.port, 9070);
    assert!(!loaded.config().replay.enabled);

    let application = loaded.start(RuntimeDependencies::recording());
    assert_eq!(application.config_path(), path);
    assert_eq!(application.config().server.host, "127.0.0.1");
    assert_eq!(application.config().server.port, 9070);
    assert!(!application.config().replay.enabled);
    assert_eq!(
        application.runtime().snapshot().daemon.state,
        DaemonState::Ready
    );

    application.shutdown().await.expect("shutdown application");
}

#[tokio::test]
async fn missing_config_fails_without_creating_a_default() {
    let directory = TempDirectory::new();
    let path = directory.join("missing.yaml");

    let error = Application::bootstrap(&path, RuntimeDependencies::recording())
        .await
        .expect_err("missing configuration");
    assert_eq!(error.code(), "CONFIG_NOT_FOUND");
    assert!(!path.exists());
}

#[tokio::test]
async fn run_until_waits_for_shutdown_then_joins_the_supervisor() {
    let directory = TempDirectory::new();
    let path = directory.join("novasight.yaml");
    fs::write(&path, "{}\n").expect("write config");
    let application = Application::bootstrap(&path, RuntimeDependencies::recording())
        .await
        .expect("bootstrap application");
    let handle = application.runtime();
    handle.start().await.expect("start pipeline");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task = tokio::spawn(application.run_until(async move {
        let _ = shutdown_rx.await;
    }));

    assert!(!task.is_finished());
    shutdown_tx.send(()).expect("signal shutdown");
    task.await
        .expect("application task")
        .expect("application shutdown");
    assert_eq!(handle.snapshot().daemon.state, DaemonState::ShuttingDown);
}
