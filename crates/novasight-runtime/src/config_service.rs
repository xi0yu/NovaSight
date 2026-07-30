use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_core::SelectedCaptureProfile;
use novasight_store::config::{
    AppConfig, CapturePreference, ConfigError, ConfigRepository,
    RecoilConfig as ConfigRecoilConfig, TriggerMode as ConfigTriggerMode, YamlConfigRepository,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
#[cfg(test)]
use tokio::sync::Notify;
use tokio::sync::{Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use crate::error::RuntimeError;
use crate::supervisor::RuntimeHandle;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConfigFieldUpdate {
    pub section: String,
    pub key: String,
    pub value: Value,
    #[serde(default)]
    pub expected_revision: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConfigUpdate {
    pub config: AppConfig,
    pub restart_required: bool,
    pub applied: bool,
    pub rolled_back: bool,
    pub message: String,
}

pub(crate) struct PersistedOutputGate<'a> {
    _update_guard: MutexGuard<'a, ()>,
    current_guard: RwLockWriteGuard<'a, AppConfig>,
    effective_revision: &'a AtomicU64,
    effective_config: &'a std::sync::RwLock<AppConfig>,
    output_gate_consistent: &'a std::sync::atomic::AtomicBool,
    config: AppConfig,
    advance_effective_revision: bool,
}

impl PersistedOutputGate<'_> {
    pub(crate) fn commit(mut self) -> ConfigUpdate {
        *self.current_guard = self.config.clone();
        if self.advance_effective_revision {
            *self
                .effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = self.config.clone();
            self.effective_revision
                .store(self.config.revision, Ordering::Release);
        } else {
            // An output-only update may be applied while unrelated desired
            // configuration is waiting for process restart. Reflect only the
            // live gate in the effective view; never claim that the pending
            // capture/model/device settings entered the running process.
            self.effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .control
                .output_enabled = self.config.control.output_enabled;
        }
        self.output_gate_consistent.store(true, Ordering::Release);
        ConfigUpdate {
            config: self.config,
            restart_required: !self.advance_effective_revision,
            applied: true,
            rolled_back: false,
            message: if self.advance_effective_revision {
                "output gate persisted and applied to the live runtime".to_owned()
            } else {
                "output gate applied live; other configuration remains pending daemon restart"
                    .to_owned()
            },
        }
    }
}

pub(crate) struct PersistedTriggerMode<'a> {
    _update_guard: MutexGuard<'a, ()>,
    current_guard: RwLockWriteGuard<'a, AppConfig>,
    effective_revision: &'a AtomicU64,
    effective_config: &'a std::sync::RwLock<AppConfig>,
    config: AppConfig,
    advance_effective_revision: bool,
}

impl PersistedTriggerMode<'_> {
    pub(crate) fn commit(mut self) -> ConfigUpdate {
        *self.current_guard = self.config.clone();
        if self.advance_effective_revision {
            *self
                .effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = self.config.clone();
            self.effective_revision
                .store(self.config.revision, Ordering::Release);
        } else {
            self.effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .control
                .trigger_mode = self.config.control.trigger_mode;
        }
        ConfigUpdate {
            config: self.config,
            restart_required: !self.advance_effective_revision,
            applied: true,
            rolled_back: false,
            message: if self.advance_effective_revision {
                "trigger mode persisted and applied to the live runtime".to_owned()
            } else {
                "trigger mode applied live; other configuration remains pending daemon restart"
                    .to_owned()
            },
        }
    }
}

pub(crate) struct PersistedRecoil<'a> {
    _update_guard: MutexGuard<'a, ()>,
    current_guard: RwLockWriteGuard<'a, AppConfig>,
    effective_revision: &'a AtomicU64,
    effective_config: &'a std::sync::RwLock<AppConfig>,
    config: AppConfig,
    advance_effective_revision: bool,
}

impl PersistedRecoil<'_> {
    pub(crate) fn commit(mut self) -> ConfigUpdate {
        *self.current_guard = self.config.clone();
        if self.advance_effective_revision {
            *self
                .effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = self.config.clone();
            self.effective_revision
                .store(self.config.revision, Ordering::Release);
        } else {
            self.effective_config
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .control
                .recoil = self.config.control.recoil.clone();
        }
        ConfigUpdate {
            config: self.config,
            restart_required: !self.advance_effective_revision,
            applied: true,
            rolled_back: false,
            message: if self.advance_effective_revision {
                "recoil configuration persisted and applied to the live runtime".to_owned()
            } else {
                "recoil configuration applied live; other configuration remains pending daemon restart"
                    .to_owned()
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConfigService {
    inner: Arc<ConfigServiceInner>,
}

#[derive(Debug)]
struct ConfigServiceInner {
    repository: YamlConfigRepository,
    current: RwLock<AppConfig>,
    effective_revision: AtomicU64,
    effective_config: std::sync::RwLock<AppConfig>,
    output_gate_consistent: std::sync::atomic::AtomicBool,
    update_lock: Mutex<()>,
    #[cfg(test)]
    output_gate_checkpoint: std::sync::Mutex<Option<Arc<OutputGateCommitCheckpoint>>>,
}

#[cfg(test)]
#[derive(Default)]
struct OutputGateCommitCheckpoint {
    persistence_started: Notify,
    allow_persistence: Notify,
    persisted: Notify,
    release: Notify,
    committed: Notify,
}

#[cfg(test)]
impl std::fmt::Debug for OutputGateCommitCheckpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OutputGateCommitCheckpoint")
    }
}

impl ConfigService {
    pub fn new(path: impl Into<PathBuf>, initial: AppConfig) -> Self {
        let effective_revision = initial.revision;
        Self {
            inner: Arc::new(ConfigServiceInner {
                repository: YamlConfigRepository::new(path),
                current: RwLock::new(initial.clone()),
                effective_revision: AtomicU64::new(effective_revision),
                effective_config: std::sync::RwLock::new(initial),
                output_gate_consistent: std::sync::atomic::AtomicBool::new(true),
                update_lock: Mutex::new(()),
                #[cfg(test)]
                output_gate_checkpoint: std::sync::Mutex::new(None),
            }),
        }
    }

    pub async fn snapshot(&self) -> AppConfig {
        self.inner.current.read().await.clone()
    }

    pub fn effective_revision(&self) -> u64 {
        self.inner.effective_revision.load(Ordering::Acquire)
    }

    /// Read the latest desired configuration from a blocking control task.
    /// Runtime adapters should use [`Self::blocking_effective_snapshot`].
    pub fn blocking_snapshot(&self) -> AppConfig {
        self.inner.current.blocking_read().clone()
    }

    /// The configuration actually installed in the running process. Model
    /// publication uses this view so unrelated desired edits cannot block or
    /// silently alter candidate validation before a restart.
    pub fn blocking_effective_snapshot(&self) -> AppConfig {
        self.inner
            .effective_config
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub async fn pending_process_restart_sections(
        &self,
    ) -> Result<Vec<&'static str>, ConfigServiceError> {
        let effective = self.blocking_effective_snapshot();
        let desired = self.inner.current.read().await.clone();
        process_restart_sections(&effective, &desired)
    }

    pub async fn ensure_effective(&self) -> Result<(), ConfigServiceError> {
        let desired_revision = self.inner.current.read().await.revision;
        let effective_revision = self.effective_revision();
        if !self.inner.output_gate_consistent.load(Ordering::Acquire) {
            return Err(ConfigServiceError::RuntimeConfigDiverged {
                effective_revision,
                desired_revision,
            });
        }
        if desired_revision == effective_revision {
            Ok(())
        } else {
            Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision,
            })
        }
    }

    /// Install saved vision/control settings into the next runtime epoch.
    ///
    /// This is intentionally a stopped-runtime control-plane operation. It
    /// does not add locks or configuration reads to DetectionBatch handling,
    /// controller evaluation, or device output. Process-owned resources such
    /// as listeners, database roots, and the concrete pointer adapter retain
    /// the honest daemon-restart boundary.
    pub async fn prepare_runtime_start(
        &self,
        runtime: &RuntimeHandle,
    ) -> Result<(), ConfigServiceError> {
        let desired = self.inner.current.read().await.clone();
        let effective_revision = self.effective_revision();
        if !self.inner.output_gate_consistent.load(Ordering::Acquire) {
            return Err(ConfigServiceError::RuntimeConfigDiverged {
                effective_revision,
                desired_revision: desired.revision,
            });
        }
        if desired.revision == effective_revision {
            return Ok(());
        }
        let effective = self
            .inner
            .effective_config
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let process_sections = process_restart_sections(&effective, &desired)?;
        if !process_sections.is_empty() {
            return Err(ConfigServiceError::ProcessRestartRequired {
                effective_revision,
                desired_revision: desired.revision,
                sections: process_sections.join(", "),
            });
        }

        runtime
            .install_stopped_config(desired.clone())
            .await
            .map_err(ConfigServiceError::Runtime)?;

        // Recheck under the writer lock. If another writer won while the
        // stopped runtime was being prepared, never claim its newer revision
        // is active; the next start attempt will install that exact snapshot.
        let _update_guard = self.inner.update_lock.lock().await;
        let latest_revision = self.inner.current.read().await.revision;
        if latest_revision != desired.revision {
            return Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision: latest_revision,
            });
        }
        *self
            .inner
            .effective_config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = desired.clone();
        self.inner
            .effective_revision
            .store(desired.revision, Ordering::Release);
        Ok(())
    }

    /// Commit a persisted revision only after its runtime-side mutation has
    /// succeeded. Keeping this separate from persistence prevents a failed hot
    /// apply from being reported as effective.
    pub async fn commit_effective_revision(
        &self,
        applied_revision: u64,
    ) -> Result<(), ConfigServiceError> {
        let _update_guard = self.inner.update_lock.lock().await;
        let desired_revision = self.inner.current.read().await.revision;
        let effective_revision = self.effective_revision();
        if desired_revision != applied_revision {
            return Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision,
            });
        }
        let effective = self.inner.current.read().await.clone();
        *self
            .inner
            .effective_config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = effective;
        self.inner
            .effective_revision
            .store(applied_revision, Ordering::Release);
        Ok(())
    }

    /// Persist a kernel-validated concrete capture profile and make that
    /// capture-only revision visible to the stopped runtime immediately.
    /// Other pending configuration edits are never swept into the effective
    /// revision by this operation.
    pub async fn apply_capture_profile(
        &self,
        selected: &SelectedCaptureProfile,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        let _update_guard = self.inner.update_lock.lock().await;
        let current = self.inner.current.read().await.clone();
        let effective_revision = self.effective_revision();
        if current.revision != effective_revision {
            return Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision: current.revision,
            });
        }
        let mut candidate = current.clone();
        let capture = candidate
            .capture
            .as_mut()
            .ok_or(ConfigServiceError::CaptureNotConfigured)?;
        capture.device = PathBuf::from(&selected.device);
        capture.preference = CapturePreference::Manual;
        capture.pixel_format.clone_from(&selected.pixel_format);
        capture.width = selected.width;
        capture.height = selected.height;
        capture.fps = selected.fps;
        candidate
            .validate_configured_adapters()
            .map_err(ConfigServiceError::CaptureValidation)?;

        let repository = self.inner.repository.clone();
        let config = tokio::task::spawn_blocking(move || {
            repository.save_config(&candidate, current.revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)??;
        *self.inner.current.write().await = config.clone();
        *self
            .inner
            .effective_config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config.clone();
        self.inner
            .effective_revision
            .store(config.revision, Ordering::Release);
        Ok(ConfigUpdate {
            config,
            restart_required: false,
            applied: true,
            rolled_back: false,
            message: format!(
                "capture profile applied: {} {}x{}@{} ({})",
                selected.pixel_format,
                selected.width,
                selected.height,
                selected.fps,
                selected.selection_reason
            ),
        })
    }

    pub async fn update_field(
        &self,
        update: ConfigFieldUpdate,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        let _update_guard = self.inner.update_lock.lock().await;
        let coordinated_control_update = update.section == "control"
            && matches!(
                update.key.as_str(),
                "output_enabled" | "trigger_mode" | "recoil"
            );
        if coordinated_control_update {
            return Err(ConfigServiceError::HotUpdateTransactionRequired);
        }
        let current_revision = self.inner.current.read().await.revision;
        let expected_revision = match update.expected_revision {
            Some(revision) => revision,
            None => current_revision,
        };
        let yaml_value =
            serde_yaml::to_value(update.value).map_err(ConfigServiceError::SerializeFieldValue)?;
        let repository = self.inner.repository.clone();
        let section = update.section;
        let key = update.key;
        let config = tokio::task::spawn_blocking(move || {
            repository.save_field(&section, &key, yaml_value, expected_revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)??;
        *self.inner.current.write().await = config.clone();
        Ok(ConfigUpdate {
            config,
            restart_required: true,
            applied: false,
            rolled_back: false,
            message: "configuration persisted; restart novasightd to apply it".to_owned(),
        })
    }

    /// Submit a hot output update to the sole runtime actor.
    ///
    /// Once the command enters the actor queue it owns the transaction even if
    /// the requesting HTTP future is cancelled. The actor persists the
    /// candidate before opening the physical gate and cannot process shutdown
    /// concurrently with that commit.
    pub async fn update_output_gate(
        &self,
        runtime: &RuntimeHandle,
        update: ConfigFieldUpdate,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        runtime.update_output_config(self.clone(), update).await
    }

    pub async fn update_trigger_mode(
        &self,
        runtime: &RuntimeHandle,
        update: ConfigFieldUpdate,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        runtime
            .update_trigger_mode_config(self.clone(), update)
            .await
    }

    pub async fn update_recoil(
        &self,
        runtime: &RuntimeHandle,
        update: ConfigFieldUpdate,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        runtime.update_recoil_config(self.clone(), update).await
    }

    pub(crate) fn output_gate_value(
        update: &ConfigFieldUpdate,
    ) -> Result<bool, ConfigServiceError> {
        if update.section != "control" || update.key != "output_enabled" {
            return Err(ConfigServiceError::OutputGateUpdateInvalid);
        }
        update
            .value
            .as_bool()
            .ok_or(ConfigServiceError::OutputGateUpdateInvalid)
    }

    pub(crate) fn trigger_mode_value(
        update: &ConfigFieldUpdate,
    ) -> Result<ConfigTriggerMode, ConfigServiceError> {
        if update.section != "control" || update.key != "trigger_mode" {
            return Err(ConfigServiceError::TriggerModeUpdateInvalid);
        }
        serde_json::from_value(update.value.clone())
            .map_err(|_| ConfigServiceError::TriggerModeUpdateInvalid)
    }

    pub(crate) fn recoil_value(
        update: &ConfigFieldUpdate,
    ) -> Result<ConfigRecoilConfig, ConfigServiceError> {
        if update.section != "control" || update.key != "recoil" {
            return Err(ConfigServiceError::RecoilUpdateInvalid);
        }
        serde_json::from_value(update.value.clone())
            .map_err(|_| ConfigServiceError::RecoilUpdateInvalid)
    }

    pub(crate) async fn persist_output_gate(
        &self,
        update: ConfigFieldUpdate,
        enabled: bool,
    ) -> Result<PersistedOutputGate<'_>, ConfigServiceError> {
        if Self::output_gate_value(&update)? != enabled {
            return Err(ConfigServiceError::OutputGateUpdateInvalid);
        }
        let _update_guard = self.inner.update_lock.lock().await;
        let current_guard = self.inner.current.write().await;
        let current = current_guard.clone();
        let effective_revision = self.effective_revision();
        let advance_effective_revision = current.revision == effective_revision;
        if !advance_effective_revision && enabled {
            let effective = self
                .inner
                .effective_config
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let desired_device = serde_yaml::to_value(current.device.as_ref())
                .map_err(ConfigServiceError::SerializeFieldValue)?;
            let effective_device = serde_yaml::to_value(effective.device.as_ref())
                .map_err(ConfigServiceError::SerializeFieldValue)?;
            if desired_device != effective_device {
                return Err(ConfigServiceError::RestartRequired {
                    effective_revision,
                    desired_revision: current.revision,
                });
            }
        }
        let expected_revision = update.expected_revision.unwrap_or(current.revision);
        if expected_revision != current.revision {
            return Err(ConfigError::RevisionConflict {
                path: self.inner.repository.path().to_owned(),
                expected: expected_revision,
                actual: current.revision,
            }
            .into());
        }

        let mut candidate = current.clone();
        candidate.control.output_enabled = enabled;
        candidate
            .validate_configured_adapters()
            .map_err(ConfigServiceError::OutputGateValidation)?;

        #[cfg(test)]
        let checkpoint = self
            .inner
            .output_gate_checkpoint
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        #[cfg(test)]
        if let Some(checkpoint) = &checkpoint {
            checkpoint.persistence_started.notify_one();
            checkpoint.allow_persistence.notified().await;
        }

        let repository = self.inner.repository.clone();
        let config = tokio::task::spawn_blocking(move || {
            repository.save_config(&candidate, expected_revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)
        .and_then(|result| result.map_err(ConfigServiceError::Config))?;

        #[cfg(test)]
        if let Some(checkpoint) = &checkpoint {
            checkpoint.persisted.notify_one();
            checkpoint.release.notified().await;
        }

        Ok(PersistedOutputGate {
            _update_guard,
            current_guard,
            effective_revision: &self.inner.effective_revision,
            effective_config: &self.inner.effective_config,
            output_gate_consistent: &self.inner.output_gate_consistent,
            config,
            advance_effective_revision,
        })
    }

    pub(crate) async fn persist_trigger_mode(
        &self,
        update: ConfigFieldUpdate,
        mode: ConfigTriggerMode,
    ) -> Result<PersistedTriggerMode<'_>, ConfigServiceError> {
        if Self::trigger_mode_value(&update)? != mode {
            return Err(ConfigServiceError::TriggerModeUpdateInvalid);
        }
        let _update_guard = self.inner.update_lock.lock().await;
        let current_guard = self.inner.current.write().await;
        let current = current_guard.clone();
        let effective_revision = self.effective_revision();
        let advance_effective_revision = current.revision == effective_revision;
        let expected_revision = update.expected_revision.unwrap_or(current.revision);
        if expected_revision != current.revision {
            return Err(ConfigError::RevisionConflict {
                path: self.inner.repository.path().to_owned(),
                expected: expected_revision,
                actual: current.revision,
            }
            .into());
        }

        let mut candidate = current;
        candidate.control.trigger_mode = mode;
        candidate
            .validate_configured_adapters()
            .map_err(ConfigServiceError::TriggerModeValidation)?;
        let repository = self.inner.repository.clone();
        let config = tokio::task::spawn_blocking(move || {
            repository.save_config(&candidate, expected_revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)
        .and_then(|result| result.map_err(ConfigServiceError::Config))?;

        Ok(PersistedTriggerMode {
            _update_guard,
            current_guard,
            effective_revision: &self.inner.effective_revision,
            effective_config: &self.inner.effective_config,
            config,
            advance_effective_revision,
        })
    }

    pub(crate) async fn persist_recoil(
        &self,
        update: ConfigFieldUpdate,
        recoil: ConfigRecoilConfig,
    ) -> Result<PersistedRecoil<'_>, ConfigServiceError> {
        let _ = Self::recoil_value(&update)?;
        let _update_guard = self.inner.update_lock.lock().await;
        let current_guard = self.inner.current.write().await;
        let current = current_guard.clone();
        let effective_revision = self.effective_revision();
        let advance_effective_revision = current.revision == effective_revision;
        let expected_revision = update.expected_revision.unwrap_or(current.revision);
        if expected_revision != current.revision {
            return Err(ConfigError::RevisionConflict {
                path: self.inner.repository.path().to_owned(),
                expected: expected_revision,
                actual: current.revision,
            }
            .into());
        }

        let mut candidate = current;
        candidate.control.recoil = recoil;
        candidate
            .validate_configured_adapters()
            .map_err(ConfigServiceError::RecoilValidation)?;
        let repository = self.inner.repository.clone();
        let config = tokio::task::spawn_blocking(move || {
            repository.save_config(&candidate, expected_revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)
        .and_then(|result| result.map_err(ConfigServiceError::Config))?;

        Ok(PersistedRecoil {
            _update_guard,
            current_guard,
            effective_revision: &self.inner.effective_revision,
            effective_config: &self.inner.effective_config,
            config,
            advance_effective_revision,
        })
    }

    pub(crate) fn mark_output_gate_diverged(&self) {
        self.inner
            .output_gate_consistent
            .store(false, Ordering::Release);
    }

    pub(crate) fn output_gate_is_consistent(&self) -> bool {
        self.inner.output_gate_consistent.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn notify_output_gate_applied(&self) {
        let checkpoint = self
            .inner
            .output_gate_checkpoint
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if let Some(checkpoint) = checkpoint {
            checkpoint.committed.notify_one();
        }
    }

    pub async fn replace(&self, replacement: Value) -> Result<ConfigUpdate, ConfigServiceError> {
        let expected_revision = replacement
            .get("revision")
            .and_then(Value::as_u64)
            .ok_or(ConfigServiceError::ReplacementRevisionRequired)?;
        let replacement_output_enabled = replacement
            .get("control")
            .and_then(|control| control.get("output_enabled"))
            .and_then(Value::as_bool);
        let replacement =
            serde_yaml::to_value(replacement).map_err(ConfigServiceError::SerializeFieldValue)?;
        let _update_guard = self.inner.update_lock.lock().await;
        let current_output_enabled = self.inner.current.read().await.control.output_enabled;
        if replacement_output_enabled.is_some_and(|enabled| enabled != current_output_enabled) {
            return Err(ConfigServiceError::HotUpdateTransactionRequired);
        }
        let repository = self.inner.repository.clone();
        let config = tokio::task::spawn_blocking(move || {
            repository.replace_document(replacement, expected_revision)
        })
        .await
        .map_err(ConfigServiceError::SaveTask)??;
        *self.inner.current.write().await = config.clone();
        Ok(ConfigUpdate {
            config,
            restart_required: true,
            applied: false,
            rolled_back: false,
            message: "configuration persisted; restart novasightd to apply it".to_owned(),
        })
    }
}

#[derive(Debug, Error)]
pub enum ConfigServiceError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("configuration field value cannot be represented in YAML: {0}")]
    SerializeFieldValue(serde_yaml::Error),
    #[error("configuration replacement must include its current numeric revision")]
    ReplacementRevisionRequired,
    #[error("capture adapter is not configured")]
    CaptureNotConfigured,
    #[error("selected capture profile is incompatible with the current configuration: {0}")]
    CaptureValidation(novasight_store::config::ConfigValidationError),
    #[error("output gate update must target control.output_enabled with a boolean value")]
    OutputGateUpdateInvalid,
    #[error("output gate update is incompatible with the current configuration: {0}")]
    OutputGateValidation(novasight_store::config::ConfigValidationError),
    #[error("trigger mode update must target control.trigger_mode with always or hardware")]
    TriggerModeUpdateInvalid,
    #[error("trigger mode update is incompatible with the current configuration: {0}")]
    TriggerModeValidation(novasight_store::config::ConfigValidationError),
    #[error("recoil update must target control.recoil with a valid recoil object")]
    RecoilUpdateInvalid,
    #[error("recoil update is incompatible with the current configuration: {0}")]
    RecoilValidation(novasight_store::config::ConfigValidationError),
    #[error(
        "control.output_enabled, control.trigger_mode, and control.recoil require the coordinated runtime configuration transaction"
    )]
    HotUpdateTransactionRequired,
    #[error(
        "physical output was disabled, but the matching configuration could not be persisted: {source}"
    )]
    OutputGateDisabledButNotPersisted { source: Box<ConfigServiceError> },
    #[error(
        "runtime output is fail-closed and differs from configuration revision {desired_revision}; last fully effective revision is {effective_revision}"
    )]
    RuntimeConfigDiverged {
        effective_revision: u64,
        desired_revision: u64,
    },
    #[error(transparent)]
    Runtime(RuntimeError),
    #[error(
        "novasightd restart required: process uses configuration revision {effective_revision}, persisted revision is {desired_revision}"
    )]
    RestartRequired {
        effective_revision: u64,
        desired_revision: u64,
    },
    #[error(
        "novasightd restart required for process-owned configuration sections [{sections}]: process uses revision {effective_revision}, persisted revision is {desired_revision}"
    )]
    ProcessRestartRequired {
        effective_revision: u64,
        desired_revision: u64,
        sections: String,
    },
    #[error("configuration save task failed: {0}")]
    SaveTask(tokio::task::JoinError),
}

impl ConfigServiceError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Config(error) => error.code(),
            Self::SerializeFieldValue(_) => "CONFIG_FIELD_VALUE_INVALID",
            Self::ReplacementRevisionRequired => "CONFIG_REPLACEMENT_REVISION_REQUIRED",
            Self::CaptureNotConfigured => "CAPTURE_NOT_CONFIGURED",
            Self::CaptureValidation(_) => "CAPTURE_PROFILE_INVALID",
            Self::OutputGateUpdateInvalid => "CONFIG_FIELD_VALUE_INVALID",
            Self::OutputGateValidation(_) => "CONFIG_VALIDATION_ERROR",
            Self::TriggerModeUpdateInvalid => "CONFIG_FIELD_VALUE_INVALID",
            Self::TriggerModeValidation(_) => "CONFIG_VALIDATION_ERROR",
            Self::RecoilUpdateInvalid => "CONFIG_FIELD_VALUE_INVALID",
            Self::RecoilValidation(_) => "CONFIG_VALIDATION_ERROR",
            Self::HotUpdateTransactionRequired => "CONFIG_HOT_UPDATE_TRANSACTION_REQUIRED",
            Self::OutputGateDisabledButNotPersisted { .. } => {
                "CONFIG_OUTPUT_GATE_DISABLED_NOT_PERSISTED"
            }
            Self::RuntimeConfigDiverged { .. } => "CONFIG_RUNTIME_DIVERGED",
            Self::Runtime(error) => error.kind.code(),
            Self::RestartRequired { .. } | Self::ProcessRestartRequired { .. } => {
                "CONFIG_RESTART_REQUIRED"
            }
            Self::SaveTask(_) => "CONFIG_SAVE_TASK_FAILED",
        }
    }
}

fn process_restart_sections(
    effective: &AppConfig,
    desired: &AppConfig,
) -> Result<Vec<&'static str>, ConfigServiceError> {
    let mut changed = Vec::new();
    for (name, differs) in [
        (
            "schema_version",
            config_value_differs(&effective.schema_version, &desired.schema_version)?,
        ),
        (
            "server",
            config_value_differs(&effective.server, &desired.server)?,
        ),
        (
            "replay",
            config_value_differs(&effective.replay, &desired.replay)?,
        ),
        (
            "paths",
            config_value_differs(&effective.paths, &desired.paths)?,
        ),
        (
            "crosshair",
            config_value_differs(&effective.crosshair, &desired.crosshair)?,
        ),
        (
            "hardware",
            config_value_differs(&effective.device, &desired.device)?,
        ),
        (
            "legacy",
            config_value_differs(&effective.legacy, &desired.legacy)?,
        ),
    ] {
        if differs {
            changed.push(name);
        }
    }
    Ok(changed)
}

fn config_value_differs(
    effective: &impl Serialize,
    desired: &impl Serialize,
) -> Result<bool, ConfigServiceError> {
    let effective =
        serde_yaml::to_value(effective).map_err(ConfigServiceError::SerializeFieldValue)?;
    let desired = serde_yaml::to_value(desired).map_err(ConfigServiceError::SerializeFieldValue)?;
    Ok(effective != desired)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use novasight_store::config::YamlConfigRepository;

    use super::*;
    use crate::{RuntimeDependencies, RuntimeSupervisor};

    struct TestConfigDirectory(PathBuf);

    impl TestConfigDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "novasight-output-transaction-{}",
                uuid::Uuid::new_v4()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestConfigDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn configured_service(
        output_enabled: bool,
    ) -> (
        TestConfigDirectory,
        PathBuf,
        ConfigService,
        Arc<OutputGateCommitCheckpoint>,
    ) {
        let directory = TestConfigDirectory::new();
        let path = directory.0.join("novasight.yaml");
        fs::write(
            &path,
            format!(
                "revision: 0\ncontrol:\n  output_enabled: {output_enabled}\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n"
            ),
        )
        .unwrap();
        let initial = YamlConfigRepository::load(&path).unwrap();
        let service = ConfigService::new(&path, initial);
        let checkpoint = Arc::new(OutputGateCommitCheckpoint::default());
        *service
            .inner
            .output_gate_checkpoint
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::clone(&checkpoint));
        (directory, path, service, checkpoint)
    }

    fn output_update(enabled: bool) -> ConfigFieldUpdate {
        ConfigFieldUpdate {
            section: "control".to_owned(),
            key: "output_enabled".to_owned(),
            value: Value::Bool(enabled),
            expected_revision: Some(0),
        }
    }

    async fn wait_for_checkpoint(notify: &Notify, message: &str) {
        tokio::time::timeout(Duration::from_secs(5), notify.notified())
            .await
            .unwrap_or_else(|_| panic!("{message}"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_finishes_a_persisted_output_commit_after_the_caller_is_cancelled() {
        let (_directory, path, service, checkpoint) = configured_service(false);

        let (supervisor, runtime) =
            RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(false));
        tokio::time::timeout(Duration::from_secs(5), runtime.start())
            .await
            .expect("runtime start timed out")
            .unwrap();

        let mut caller = tokio::spawn({
            let service = service.clone();
            let runtime = runtime.clone();
            async move {
                service
                    .update_output_gate(&runtime, output_update(true))
                    .await
            }
        });

        tokio::select! {
            () = checkpoint.persistence_started.notified() => {}
            result = &mut caller => {
                panic!("output transaction ended before persistence started: {result:?}");
            }
            () = tokio::time::sleep(Duration::from_secs(5)) => {
                panic!("output transaction did not reach persistence");
            }
        }
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
        checkpoint.allow_persistence.notify_one();
        wait_for_checkpoint(
            &checkpoint.persisted,
            "output transaction did not reach the durable checkpoint",
        )
        .await;
        let persisted = YamlConfigRepository::load(&path).unwrap();
        assert_eq!(persisted.revision, 1);
        assert!(persisted.control.output_enabled);
        assert_eq!(service.effective_revision(), 0);
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);

        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        checkpoint.release.notify_one();
        tokio::time::timeout(Duration::from_secs(5), checkpoint.committed.notified())
            .await
            .expect("actor did not finish the caller-independent commit");

        assert_eq!(service.effective_revision(), 1);
        assert_eq!(service.snapshot().await.revision, 1);
        assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

        tokio::time::timeout(Duration::from_secs(5), runtime.shutdown_daemon())
            .await
            .expect("runtime shutdown timed out")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), supervisor.join())
            .await
            .expect("supervisor join timed out")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disabling_output_closes_the_gate_before_persistence_can_run() {
        let (_directory, _path, service, checkpoint) = configured_service(true);
        let (supervisor, runtime) =
            RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
        runtime.start().await.unwrap();
        assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

        let caller = tokio::spawn({
            let service = service.clone();
            let runtime = runtime.clone();
            async move {
                service
                    .update_output_gate(&runtime, output_update(false))
                    .await
            }
        });

        wait_for_checkpoint(
            &checkpoint.persistence_started,
            "disable transaction did not reach persistence",
        )
        .await;
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
        assert!(!caller.is_finished());

        checkpoint.allow_persistence.notify_one();
        wait_for_checkpoint(&checkpoint.persisted, "disable was not persisted").await;
        checkpoint.release.notify_one();
        let update = caller.await.unwrap().unwrap();
        assert!(!update.config.control.output_enabled);
        assert_eq!(service.effective_revision(), 1);

        runtime.shutdown_daemon().await.unwrap();
        supervisor.join().await.unwrap();
    }

    #[tokio::test]
    async fn output_can_reopen_while_unrelated_configuration_awaits_restart() {
        let directory = TestConfigDirectory::new();
        let path = directory.0.join("novasight.yaml");
        fs::write(
            &path,
            "revision: 0\ncontrol:\n  output_enabled: true\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n",
        )
        .unwrap();
        let initial = YamlConfigRepository::load(&path).unwrap();
        let service = ConfigService::new(&path, initial);
        let (supervisor, runtime) =
            RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
        runtime.start().await.unwrap();

        service
            .update_field(ConfigFieldUpdate {
                section: "replay".to_owned(),
                key: "frame_interval_ms".to_owned(),
                value: Value::from(24),
                expected_revision: Some(0),
            })
            .await
            .unwrap();
        assert_eq!(service.effective_revision(), 0);

        service
            .update_output_gate(
                &runtime,
                ConfigFieldUpdate {
                    expected_revision: Some(1),
                    ..output_update(false)
                },
            )
            .await
            .unwrap();
        assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);

        let reopened = service
            .update_output_gate(
                &runtime,
                ConfigFieldUpdate {
                    expected_revision: Some(2),
                    ..output_update(true)
                },
            )
            .await
            .unwrap();
        assert!(reopened.restart_required);
        assert!(reopened.applied);
        assert!(runtime.snapshot().pipeline_metrics.output_gate_open);
        assert_eq!(service.effective_revision(), 0);

        runtime.shutdown_daemon().await.unwrap();
        supervisor.join().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_supervisor_latches_output_closed_during_an_enable_commit() {
        let (_directory, _path, service, checkpoint) = configured_service(false);
        let (supervisor, runtime) =
            RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(false));
        runtime.start().await.unwrap();
        let mut snapshots = runtime.subscribe();

        let caller = tokio::spawn({
            let service = service.clone();
            async move {
                service
                    .update_output_gate(&runtime, output_update(true))
                    .await
            }
        });
        wait_for_checkpoint(
            &checkpoint.persistence_started,
            "enable transaction did not reach persistence",
        )
        .await;
        checkpoint.allow_persistence.notify_one();
        wait_for_checkpoint(&checkpoint.persisted, "enable was not persisted").await;

        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        drop(supervisor);
        assert!(!snapshots.borrow().pipeline_metrics.output_gate_open);
        checkpoint.release.notify_one();
        wait_for_checkpoint(
            &checkpoint.committed,
            "enable transaction did not commit after supervisor drop",
        )
        .await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while snapshots.changed().await.is_ok() {}
        })
        .await
        .expect("supervisor did not exit after owner drop");

        assert_eq!(service.effective_revision(), 1);
        assert!(!snapshots.borrow().pipeline_metrics.output_gate_open);
    }
}
