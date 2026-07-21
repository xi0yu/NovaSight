use std::{future::Future, future::IntoFuture, io};

use axum::Router;
use novasight_core::{AppError, RuntimeHandle};
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

pub(super) struct ShutdownSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(windows)]
    ctrl_c: tokio::signal::windows::CtrlC,
}

impl ShutdownSignals {
    pub(super) fn register() -> Result<Self, ShutdownError> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            Ok(Self {
                interrupt: signal(SignalKind::interrupt()).map_err(ShutdownError::Signal)?,
                terminate: signal(SignalKind::terminate()).map_err(ShutdownError::Signal)?,
            })
        }

        #[cfg(windows)]
        {
            Ok(Self {
                ctrl_c: tokio::signal::windows::ctrl_c().map_err(ShutdownError::Signal)?,
            })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(ShutdownError::Signal(io::Error::new(
                io::ErrorKind::Unsupported,
                "process signals are unsupported on this platform",
            )))
        }
    }

    async fn wait(&mut self) {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.interrupt.recv() => {}
                _ = self.terminate.recv() => {}
            }
        }

        #[cfg(windows)]
        {
            let _ = self.ctrl_c.recv().await;
        }
    }
}

pub(super) async fn serve(
    listener: TcpListener,
    app: Router,
    runtime: RuntimeHandle,
    mut signals: ShutdownSignals,
) -> Result<(), ShutdownError> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.await;
        })
        .into_future();
    tokio::pin!(server);

    tokio::select! {
        server_result = &mut server => {
            return finish_server_exit(server_result, runtime.stop()).await;
        }
        () = signals.wait() => {}
    }

    // Axum must stop accepting work before the runtime session is revoked.
    let _ = shutdown_tx.send(());
    finish_signal_shutdown(server, &runtime).await
}

async fn finish_server_exit<T>(
    server_result: io::Result<()>,
    cleanup: impl Future<Output = Result<T, AppError>>,
) -> Result<(), ShutdownError> {
    let cleanup_result = cleanup.await;
    prioritize_server_error(server_result, cleanup_result)
}

async fn finish_signal_shutdown(
    server: impl Future<Output = io::Result<()>>,
    runtime: &RuntimeHandle,
) -> Result<(), ShutdownError> {
    let prompt_stop_result = runtime.stop().await;
    let server_result = server.await;
    let final_stop_result = runtime.stop().await;

    if let Err(prompt_stop_error) = prompt_stop_result {
        tracing::error!(
            code = "RUNTIME_STOP_FAILED",
            error = %prompt_stop_error,
            "prompt runtime stop failed; final stop was still attempted after server drain"
        );
    }

    prioritize_server_error(server_result, final_stop_result)
}

fn prioritize_server_error<T>(
    server_result: io::Result<()>,
    cleanup_result: Result<T, AppError>,
) -> Result<(), ShutdownError> {
    match server_result {
        Err(server_error) => {
            if let Err(cleanup_error) = cleanup_result {
                tracing::error!(
                    code = "RUNTIME_STOP_FAILED",
                    error = %cleanup_error,
                    "runtime cleanup failed after backend server failure"
                );
            }
            Err(ShutdownError::Server(server_error))
        }
        Ok(()) => cleanup_result
            .map(|_| ())
            .map_err(ShutdownError::RuntimeStop),
    }
}

#[derive(Debug, Error)]
pub(super) enum ShutdownError {
    #[error("failed to register process shutdown signal: {0}")]
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

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use novasight_core::{AppError, RuntimeDependencies, RuntimeManager, RuntimePhase};

    use super::{ShutdownSignals, finish_server_exit, finish_signal_shutdown};

    #[tokio::test]
    async fn signal_receivers_are_registered_during_synchronous_construction() {
        let signals = ShutdownSignals::register();

        assert!(signals.is_ok());
    }

    #[tokio::test]
    async fn server_error_remains_primary_after_failed_cleanup_is_awaited() {
        let cleanup_was_awaited = Arc::new(AtomicBool::new(false));
        let cleanup_observation = cleanup_was_awaited.clone();

        let error = finish_server_exit(Err(io::Error::other("serve failed")), async move {
            cleanup_observation.store(true, Ordering::SeqCst);
            Err::<(), _>(AppError::RuntimeManagerUnavailable)
        })
        .await
        .expect_err("server failure must be returned");

        assert!(cleanup_was_awaited.load(Ordering::SeqCst));
        assert_eq!(error.code(), "SERVER_SERVE_FAILED");
    }

    #[tokio::test]
    async fn signal_shutdown_stops_a_session_started_by_a_draining_handler() {
        let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
        let original = runtime.start().await.expect("start initial session");
        let handler_runtime = runtime.clone();

        finish_signal_shutdown(
            async move {
                assert_eq!(handler_runtime.snapshot().phase, RuntimePhase::Stopped);
                let restarted = handler_runtime
                    .start()
                    .await
                    .expect("draining handler starts a new session");
                assert!(restarted.epoch > original.epoch);
                Ok(())
            },
            &runtime,
        )
        .await
        .expect("shutdown cleanup");

        assert_eq!(runtime.snapshot().phase, RuntimePhase::Stopped);
        assert!(!runtime.snapshot().running);
    }
}
