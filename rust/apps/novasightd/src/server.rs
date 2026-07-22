use std::future::IntoFuture;
use std::io;

use novasight_api::build_control_router_with_shutdown;
use novasight_runtime::{ApplicationError, LoadedApplication, RuntimeDependencies};
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::watch;

pub(super) async fn run_dry_run(loaded: LoadedApplication) -> Result<(), DaemonRunError> {
    let host = loaded.config().server.host.clone();
    let port = loaded.config().server.port;
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|source| DaemonRunError::Bind {
            host: host.clone(),
            port,
            source,
        })?;
    let address = listener
        .local_addr()
        .map_err(DaemonRunError::LocalAddress)?;
    let mut signals = ShutdownSignals::register()?;
    let application = loaded.start(RuntimeDependencies::recording());
    let (server_shutdown_tx, mut server_shutdown_rx) = watch::channel(false);
    let router =
        build_control_router_with_shutdown(application.runtime(), Some(server_shutdown_rx.clone()));
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            if !*server_shutdown_rx.borrow() {
                let _ = server_shutdown_rx.changed().await;
            }
        })
        .into_future();
    let mut server = Box::pin(server);

    tracing::warn!("novasightd is running in explicit dry-run mode; hardware output is disabled");
    eprintln!("novasightd ready mode=dry-run address={address}");

    let service_result = tokio::select! {
        result = server.as_mut() => result.map_err(DaemonRunError::Serve),
        () = signals.wait() => {
            server_shutdown_tx.send_replace(true);
            server.as_mut().await.map_err(DaemonRunError::Serve)
        }
    };
    drop(server);
    let shutdown_result = application
        .shutdown()
        .await
        .map_err(DaemonRunError::Application);
    match (service_result, shutdown_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(primary), Err(cleanup)) => Err(DaemonRunError::ServiceAndShutdown {
            primary: Box::new(primary),
            cleanup: Box::new(cleanup),
        }),
    }
}

struct ShutdownSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

impl ShutdownSignals {
    fn register() -> Result<Self, DaemonRunError> {
        use tokio::signal::unix::{SignalKind, signal};

        Ok(Self {
            interrupt: signal(SignalKind::interrupt()).map_err(DaemonRunError::Signal)?,
            terminate: signal(SignalKind::terminate()).map_err(DaemonRunError::Signal)?,
        })
    }

    async fn wait(&mut self) {
        tokio::select! {
            _ = self.interrupt.recv() => {}
            _ = self.terminate.recv() => {}
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum DaemonRunError {
    #[error("failed to bind HTTP server at {host}:{port}: {source}")]
    Bind {
        host: String,
        port: u16,
        #[source]
        source: io::Error,
    },
    #[error("failed to read bound HTTP address: {0}")]
    LocalAddress(io::Error),
    #[error("failed to register shutdown signal: {0}")]
    Signal(io::Error),
    #[error("HTTP server failed: {0}")]
    Serve(io::Error),
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error("service failed: {primary}; runtime cleanup also failed: {cleanup}")]
    ServiceAndShutdown {
        primary: Box<DaemonRunError>,
        cleanup: Box<DaemonRunError>,
    },
}

impl DaemonRunError {
    pub(super) const fn code(&self) -> &'static str {
        match self {
            Self::Bind { .. } => "SERVER_BIND_FAILED",
            Self::LocalAddress(_) => "SERVER_LOCAL_ADDRESS_FAILED",
            Self::Signal(_) => "SHUTDOWN_SIGNAL_FAILED",
            Self::Serve(_) => "SERVER_FAILED",
            Self::Application(error) => error.code(),
            Self::ServiceAndShutdown { .. } => "SERVER_AND_SHUTDOWN_FAILED",
        }
    }
}
