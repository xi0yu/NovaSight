use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_core::SelectedCaptureProfile;
use novasight_store::config::{
    AppConfig, CapturePreference, ConfigError, ConfigRepository, YamlConfigRepository,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

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

#[derive(Clone, Debug)]
pub struct ConfigService {
    inner: Arc<ConfigServiceInner>,
}

#[derive(Debug)]
struct ConfigServiceInner {
    repository: YamlConfigRepository,
    current: RwLock<AppConfig>,
    effective_revision: AtomicU64,
    update_lock: Mutex<()>,
}

impl ConfigService {
    pub fn new(path: impl Into<PathBuf>, initial: AppConfig) -> Self {
        let effective_revision = initial.revision;
        Self {
            inner: Arc::new(ConfigServiceInner {
                repository: YamlConfigRepository::new(path),
                current: RwLock::new(initial),
                effective_revision: AtomicU64::new(effective_revision),
                update_lock: Mutex::new(()),
            }),
        }
    }

    pub async fn snapshot(&self) -> AppConfig {
        self.inner.current.read().await.clone()
    }

    pub fn effective_revision(&self) -> u64 {
        self.inner.effective_revision.load(Ordering::Acquire)
    }

    /// Read the daemon's current configuration from a blocking adapter task.
    /// Runtime preflight and startup use `spawn_blocking`, so they can safely
    /// consume the same revision that the control plane owns.
    pub fn blocking_snapshot(&self) -> AppConfig {
        self.inner.current.blocking_read().clone()
    }

    pub async fn ensure_effective(&self) -> Result<(), ConfigServiceError> {
        let desired_revision = self.inner.current.read().await.revision;
        let effective_revision = self.effective_revision();
        if desired_revision == effective_revision {
            Ok(())
        } else {
            Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision,
            })
        }
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
        let hot_output_gate = update.section == "control" && update.key == "output_enabled";
        let current_revision = self.inner.current.read().await.revision;
        let effective_revision = self.effective_revision();
        if hot_output_gate && current_revision != effective_revision {
            return Err(ConfigServiceError::RestartRequired {
                effective_revision,
                desired_revision: current_revision,
            });
        }
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
        if hot_output_gate {
            self.inner
                .effective_revision
                .store(config.revision, Ordering::Release);
        }
        Ok(ConfigUpdate {
            config,
            restart_required: !hot_output_gate,
            applied: hot_output_gate,
            rolled_back: false,
            message: if hot_output_gate {
                "output gate persisted and applied to the live runtime".to_owned()
            } else {
                "configuration persisted; restart novasightd to apply it".to_owned()
            },
        })
    }

    pub async fn replace(&self, replacement: Value) -> Result<ConfigUpdate, ConfigServiceError> {
        let expected_revision = replacement
            .get("revision")
            .and_then(Value::as_u64)
            .ok_or(ConfigServiceError::ReplacementRevisionRequired)?;
        let replacement =
            serde_yaml::to_value(replacement).map_err(ConfigServiceError::SerializeFieldValue)?;
        let _update_guard = self.inner.update_lock.lock().await;
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
    #[error(
        "novasightd restart required: process uses configuration revision {effective_revision}, persisted revision is {desired_revision}"
    )]
    RestartRequired {
        effective_revision: u64,
        desired_revision: u64,
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
            Self::RestartRequired { .. } => "CONFIG_RESTART_REQUIRED",
            Self::SaveTask(_) => "CONFIG_SAVE_TASK_FAILED",
        }
    }
}
