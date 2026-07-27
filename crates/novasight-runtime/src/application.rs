use std::future::Future;
use std::path::{Path, PathBuf};

use novasight_store::config::{AppConfig, ConfigError, YamlConfigRepository};
use thiserror::Error;

use crate::{RuntimeDependencies, RuntimeError, RuntimeHandle, RuntimeSupervisor};

/// Configuration loaded and parsed without starting any runtime owner.
#[derive(Debug)]
pub struct LoadedApplication {
    config_path: PathBuf,
    config: AppConfig,
}

impl LoadedApplication {
    pub async fn load(config_path: impl AsRef<Path>) -> Result<Self, ApplicationError> {
        let config_path = config_path.as_ref().to_path_buf();
        let load_path = config_path.clone();
        let config = tokio::task::spawn_blocking(move || YamlConfigRepository::load(load_path))
            .await
            .map_err(ApplicationError::ConfigLoadTask)??;
        Ok(Self {
            config_path,
            config,
        })
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn start(self, dependencies: RuntimeDependencies) -> Application {
        let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
        Application {
            config_path: self.config_path,
            config: self.config,
            supervisor,
            runtime,
        }
    }
}

/// Process-level owner of configuration and the one runtime supervisor.
pub struct Application {
    config_path: PathBuf,
    config: AppConfig,
    supervisor: RuntimeSupervisor,
    runtime: RuntimeHandle,
}

impl std::fmt::Debug for Application {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Application")
            .field("config_path", &self.config_path)
            .field("config", &self.config)
            .field("supervisor", &self.supervisor)
            .field("runtime", &self.runtime)
            .finish()
    }
}

impl Application {
    /// Load the persisted configuration before spawning any long-lived owner.
    pub async fn bootstrap(
        config_path: impl AsRef<Path>,
        dependencies: RuntimeDependencies,
    ) -> Result<Self, ApplicationError> {
        Ok(LoadedApplication::load(config_path)
            .await?
            .start(dependencies))
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn runtime(&self) -> RuntimeHandle {
        self.runtime.clone()
    }

    /// Testable shutdown seam used by signal handling and service managers.
    pub async fn run_until<F>(self, shutdown: F) -> Result<(), ApplicationError>
    where
        F: Future<Output = ()>,
    {
        shutdown.await;
        self.shutdown().await
    }

    pub async fn run_until_shutdown(self) -> Result<(), ApplicationError> {
        self.run_until_shutdown_with_ready(|| {}).await
    }

    /// Register process signals before publishing readiness.
    pub async fn run_until_shutdown_with_ready(
        self,
        ready: impl FnOnce(),
    ) -> Result<(), ApplicationError> {
        let signals = ShutdownSignals::register().map_err(ApplicationError::Signal);
        let signal = match signals {
            Ok(signals) => {
                ready();
                signals.wait().await
            }
            Err(error) => Err(error),
        };
        let shutdown = self.shutdown().await;
        match (signal, shutdown) {
            (Ok(()), result) => result,
            (Err(signal), Ok(())) => Err(signal),
            (Err(signal), Err(shutdown)) => Err(ApplicationError::SignalAndShutdown {
                signal: Box::new(signal),
                shutdown: Box::new(shutdown),
            }),
        }
    }

    pub async fn shutdown(self) -> Result<(), ApplicationError> {
        let supervisor_already_exited = !self.runtime.is_supervisor_alive();
        if supervisor_already_exited {
            return match self.supervisor.join().await {
                Ok(()) => Err(ApplicationError::RuntimeExitedUnexpectedly),
                Err(error) => Err(error.into()),
            };
        }

        let command = self.runtime.shutdown_daemon().await;
        let join = self.supervisor.join().await;
        match (command, join) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error.into()),
            (Err(command), Err(join)) => Err(ApplicationError::RuntimeShutdown { command, join }),
        }
    }
}

#[cfg(unix)]
struct ShutdownSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl ShutdownSignals {
    fn register() -> Result<Self, std::io::Error> {
        use tokio::signal::unix::{SignalKind, signal};

        Ok(Self {
            interrupt: signal(SignalKind::interrupt())?,
            terminate: signal(SignalKind::terminate())?,
        })
    }

    async fn wait(mut self) -> Result<(), ApplicationError> {
        tokio::select! {
            _ = self.interrupt.recv() => Ok(()),
            _ = self.terminate.recv() => Ok(()),
        }
    }
}

#[cfg(not(unix))]
struct ShutdownSignals;

#[cfg(not(unix))]
impl ShutdownSignals {
    fn register() -> Result<Self, std::io::Error> {
        Ok(Self)
    }

    async fn wait(self) -> Result<(), ApplicationError> {
        tokio::signal::ctrl_c()
            .await
            .map_err(ApplicationError::Signal)
    }
}

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("configuration load task failed: {0}")]
    ConfigLoadTask(tokio::task::JoinError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error("runtime supervisor exited while the application was still serving")]
    RuntimeExitedUnexpectedly,
    #[error("failed to register or receive shutdown signal: {0}")]
    Signal(std::io::Error),
    #[error("runtime shutdown command failed: {command}; supervisor join failed: {join}")]
    RuntimeShutdown {
        command: RuntimeError,
        join: RuntimeError,
    },
    #[error("shutdown signal failed: {signal}; cleanup also failed: {shutdown}")]
    SignalAndShutdown {
        signal: Box<ApplicationError>,
        shutdown: Box<ApplicationError>,
    },
}

impl ApplicationError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Config(error) => error.code(),
            Self::ConfigLoadTask(_) => "CONFIG_LOAD_TASK_FAILED",
            Self::Runtime(error) => error.kind.code(),
            Self::RuntimeExitedUnexpectedly => "RUNTIME_SUPERVISOR_EXITED",
            Self::Signal(_) => "SHUTDOWN_SIGNAL_FAILED",
            Self::RuntimeShutdown { .. } => "RUNTIME_SHUTDOWN_FAILED",
            Self::SignalAndShutdown { .. } => "SIGNAL_AND_SHUTDOWN_FAILED",
        }
    }
}
