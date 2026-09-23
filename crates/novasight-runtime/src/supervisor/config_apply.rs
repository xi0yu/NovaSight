//! Epoch-scoped configuration installation and compensation.
//!
//! The supervisor remains the sole lifecycle owner. This module only groups
//! the configuration transaction so persistence, dependency replacement, and
//! previous-epoch recovery do not become independent command paths.

use std::sync::Arc;

use novasight_core::PointerDeviceMode;
use novasight_pipeline::PipelineIngress;
use novasight_store::config::AppConfig;
use tokio::sync::{mpsc, watch};

use crate::config_service::ConfigService;
use crate::error::RuntimeError;
use crate::snapshot::RuntimeSnapshot;
use crate::state::PipelineState;

use super::{
    ActivePipeline, PipelineNotice, RuntimeDependencies, SupervisorState, now_ms, publish,
    publish_with_result, start_state, stop_state,
};

#[derive(Clone, Debug)]
pub(crate) struct RuntimeConfigApplyFailure {
    pub apply: RuntimeError,
    pub previous_runtime_restored: bool,
    pub recovery: Option<RuntimeError>,
}

impl RuntimeConfigApplyFailure {
    pub(super) fn before_install(apply: RuntimeError) -> Self {
        Self {
            apply,
            previous_runtime_restored: false,
            recovery: None,
        }
    }
}

pub(super) fn install_config_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    state: &mut SupervisorState,
    dependencies: &RuntimeDependencies,
    config: &AppConfig,
    rebuild_device: bool,
    rebuild_crosshair: bool,
) -> Result<(), RuntimeError> {
    let device_mode = dependencies.install_app_config(config, rebuild_device, rebuild_crosshair)?;
    state.device_mode = device_mode;
    state.output_enabled =
        config.control.output_enabled && device_mode == PointerDeviceMode::Commissioned;
    if device_mode == PointerDeviceMode::Uncommissioned {
        state.mark_device_uncommissioned();
    } else {
        state.subsystems.device.state = crate::protocol::SubsystemState::Stopped;
        state.subsystems.device.last_error = None;
    }
    publish(snapshot_tx, state, now_ms());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn restore_previous_config_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
    rollback_config: &AppConfig,
    was_running: bool,
    rebuild_device: bool,
    rebuild_crosshair: bool,
) -> Result<(), RuntimeError> {
    if active.is_some() {
        stop_state(snapshot_tx, ingress_tx, state, active).await?;
    }
    install_config_state(
        snapshot_tx,
        state,
        dependencies,
        rollback_config,
        rebuild_device,
        rebuild_crosshair,
    )?;
    if was_running {
        start_state(
            snapshot_tx,
            ingress_tx,
            notice_tx,
            state,
            active,
            dependencies,
        )
        .await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_config_state(
    snapshot_tx: &watch::Sender<Arc<RuntimeSnapshot>>,
    ingress_tx: &watch::Sender<Option<PipelineIngress>>,
    notice_tx: &mpsc::Sender<PipelineNotice>,
    state: &mut SupervisorState,
    active: &mut Option<ActivePipeline>,
    dependencies: &RuntimeDependencies,
    service: &ConfigService,
    config: &AppConfig,
    rollback_config: &AppConfig,
    rebuild_device: bool,
    rebuild_crosshair: bool,
) -> Result<RuntimeSnapshot, RuntimeConfigApplyFailure> {
    let was_running = matches!(
        state.pipeline,
        PipelineState::Running | PipelineState::Standby
    );
    if was_running {
        stop_state(snapshot_tx, ingress_tx, state, active)
            .await
            .map_err(RuntimeConfigApplyFailure::before_install)?;
    }

    let apply_result = match install_config_state(
        snapshot_tx,
        state,
        dependencies,
        config,
        rebuild_device,
        rebuild_crosshair,
    ) {
        Err(error) => Err(error),
        Ok(()) if was_running => {
            start_state(
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active,
                dependencies,
            )
            .await
        }
        Ok(()) => Ok(publish_with_result(snapshot_tx, state, now_ms())),
    };
    match apply_result {
        Ok(snapshot) => Ok(snapshot),
        Err(apply) => {
            // Perception resolves its runtime contract from ConfigService.
            // Restore that source before restarting the previous epoch.
            service.stage_effective_runtime_config(rollback_config.clone());
            let recovery = restore_previous_config_state(
                snapshot_tx,
                ingress_tx,
                notice_tx,
                state,
                active,
                dependencies,
                rollback_config,
                was_running,
                rebuild_device,
                rebuild_crosshair,
            )
            .await
            .err();
            Err(RuntimeConfigApplyFailure {
                apply,
                previous_runtime_restored: recovery.is_none(),
                recovery,
            })
        }
    }
}
