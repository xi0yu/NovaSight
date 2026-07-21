use std::io;

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

    let clock = SystemMonotonicClock::default();
    let dependencies = replay_dependencies(&clock);
    let runtime = RuntimeManager::spawn(dependencies);
    let app = build_router(ApiState::new(runtime.clone()));

    let bind_address = format!("{}:{}", config.server.host, config.server.port);
    let listener =
        TcpListener::bind(&bind_address)
            .await
            .map_err(|source| BootstrapError::Bind {
                address: bind_address.clone(),
                source,
            })?;
    let local_address = listener
        .local_addr()
        .map_err(|source| BootstrapError::Bind {
            address: bind_address,
            source,
        })?;
    tracing::info!(address = %local_address, "relink_server listening");

    shutdown::serve(listener, app, runtime).await?;
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

fn replay_dependencies(clock: &dyn Clock) -> RuntimeDependencies {
    let detection = Detection::new(1, 0, 300.0, 300.0, 40.0, 80.0, 0.9)
        .expect("the production replay seed is statically valid");
    let batch = DetectionBatch::fixture(
        FrameStamp::new(RuntimeEpoch(0), 1, clock.now().0),
        640,
        640,
        vec![detection],
    )
    .expect("the production replay batch is statically valid");
    RuntimeDependencies::replay([batch])
}

#[derive(Debug, Error)]
pub(super) enum BootstrapError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("Phase 1 requires replay.enabled=true")]
    ReplayDisabled,
    #[error("Phase 1 forbids replay.output_gate_open=true")]
    OutputGateOpen,
    #[error("failed to bind backend server at {address}: {source}")]
    Bind {
        address: String,
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
