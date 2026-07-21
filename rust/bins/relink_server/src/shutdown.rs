use std::{future::IntoFuture, io};

use axum::Router;
use novasight_core::{AppError, RuntimeHandle};
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

pub(super) async fn serve(
    listener: TcpListener,
    app: Router,
    runtime: RuntimeHandle,
) -> Result<(), ShutdownError> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.await;
        })
        .into_future();
    tokio::pin!(server);

    let signal_result = tokio::select! {
        server_result = &mut server => {
            let stop_result = runtime.stop().await;
            stop_result.map_err(ShutdownError::RuntimeStop)?;
            return server_result.map_err(ShutdownError::Server);
        }
        signal_result = wait_for_shutdown_signal() => signal_result,
    };

    // Axum must stop accepting work before the runtime session is revoked.
    let _ = shutdown_tx.send(());
    let stop_result = runtime.stop().await;
    let server_result = server.await;

    signal_result.map_err(ShutdownError::Signal)?;
    stop_result.map_err(ShutdownError::RuntimeStop)?;
    server_result.map_err(ShutdownError::Server)
}

async fn wait_for_shutdown_signal() -> io::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result,
            _ = terminate.recv() => Ok(()),
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await
    }
}

#[derive(Debug, Error)]
pub(super) enum ShutdownError {
    #[error("failed to listen for process shutdown signal: {0}")]
    Signal(#[source] io::Error),
    #[error("backend server failed: {0}")]
    Server(#[source] io::Error),
    #[error("runtime stop failed during process shutdown: {0}")]
    RuntimeStop(#[source] AppError),
}

impl ShutdownError {
    pub(super) const fn code(&self) -> &'static str {
        match self {
            Self::Signal(_) => "SHUTDOWN_SIGNAL_FAILED",
            Self::Server(_) => "SERVER_SERVE_FAILED",
            Self::RuntimeStop(_) => "RUNTIME_STOP_FAILED",
        }
    }
}
