use std::{io, time::Duration};

use novasight_api::{ApiState, build_router};
use novasight_core::{
    Clock, Detection, DetectionBatch, FrameStamp, RuntimeDependencies, RuntimeEpoch, RuntimeManager,
};
use novasight_platform_jetson::SystemMonotonicClock;
use novasight_store::config::{AppConfig, ConfigError, YamlConfigRepository};
use thiserror::Error;
use tokio::net::TcpListener;

use crate::{Cli, shutdown};

pub(super) async fn run(cli: Cli) -> Result<(), BootstrapError> {
    let config = YamlConfigRepository::load(&cli.config)?;
    validate_phase_one(&config)?;
    let signals = shutdown::ShutdownSignals::register()?;

    let clock = SystemMonotonicClock::default();
    let dependencies = replay_dependencies(
        &clock,
        Duration::from_millis(config.replay.frame_interval_ms),
    );
    let runtime = RuntimeManager::spawn(dependencies);
    let app = build_router(ApiState::new(runtime.clone()));

    let listener = bind_listener(&config.server.host, config.server.port).await?;
    let local_address = listener
        .local_addr()
        .map_err(|source| BootstrapError::Bind {
            host: config.server.host.clone(),
            port: config.server.port,
            source,
        })?;
    eprintln!("relink_server listening address={local_address}");

    shutdown::serve(listener, app, runtime, signals).await?;
    Ok(())
}

fn validate_phase_one(config: &AppConfig) -> Result<(), BootstrapError> {
    if !config.replay.enabled {
        return Err(BootstrapError::ReplayDisabled);
    }
    if config.replay.output_gate_open {
        return Err(BootstrapError::OutputGateOpen);
    }
    Ok(())
}

async fn bind_listener(host: &str, port: u16) -> Result<TcpListener, BootstrapError> {
    TcpListener::bind((host, port))
        .await
        .map_err(|source| BootstrapError::Bind {
            host: host.to_owned(),
            port,
            source,
        })
}

fn replay_dependencies(clock: &dyn Clock, frame_interval: Duration) -> RuntimeDependencies {
    let detection = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9)
        .expect("the production replay seed is statically valid");
    let batch = DetectionBatch::fixture(
        FrameStamp::new(RuntimeEpoch(0), 1, clock.now().0),
        640,
        640,
        vec![detection],
    )
    .expect("the production replay batch is statically valid");
    RuntimeDependencies::replay([batch], frame_interval)
}

#[derive(Debug, Error)]
pub(super) enum BootstrapError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("Phase 1 requires replay.enabled=true")]
    ReplayDisabled,
    #[error("Phase 1 forbids replay.output_gate_open=true")]
    OutputGateOpen,
    #[error("failed to bind backend server at host {host} port {port}: {source}")]
    Bind {
        host: String,
        port: u16,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Shutdown(#[from] shutdown::ShutdownError),
}

impl BootstrapError {
    pub(super) const fn code(&self) -> &'static str {
        match self {
            Self::Config(error) => error.code(),
            Self::ReplayDisabled => "REPLAY_DISABLED",
            Self::OutputGateOpen => "OUTPUT_GATE_OPEN",
            Self::Bind { .. } => "SERVER_BIND_FAILED",
            Self::Shutdown(error) => error.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bind_listener;

    #[tokio::test]
    async fn ipv6_loopback_binds_with_an_ephemeral_port() {
        let listener = bind_listener("::1", 0).await.expect("bind IPv6 loopback");
        let address = listener.local_addr().expect("read bound address");

        assert!(address.is_ipv6());
        assert_ne!(address.port(), 0);
    }
}
