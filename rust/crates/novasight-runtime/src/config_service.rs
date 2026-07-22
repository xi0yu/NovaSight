use std::path::PathBuf;
use std::sync::Arc;

use novasight_store::config::{AppConfig, ConfigError, YamlConfigRepository};
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
    update_lock: Mutex<()>,
}

impl ConfigService {
    pub fn new(path: impl Into<PathBuf>, initial: AppConfig) -> Self {
        Self {
            inner: Arc::new(ConfigServiceInner {
                repository: YamlConfigRepository::new(path),
                current: RwLock::new(initial),
                update_lock: Mutex::new(()),
            }),
        }
    }

    pub async fn snapshot(&self) -> AppConfig {
        self.inner.current.read().await.clone()
    }

    pub async fn update_field(
        &self,
        update: ConfigFieldUpdate,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        let _update_guard = self.inner.update_lock.lock().await;
        let expected_revision = match update.expected_revision {
            Some(revision) => revision,
            None => self.inner.current.read().await.revision,
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
    #[error("configuration save task failed: {0}")]
    SaveTask(tokio::task::JoinError),
}

impl ConfigServiceError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Config(error) => error.code(),
            Self::SerializeFieldValue(_) => "CONFIG_FIELD_VALUE_INVALID",
            Self::ReplacementRevisionRequired => "CONFIG_REPLACEMENT_REVISION_REQUIRED",
            Self::SaveTask(_) => "CONFIG_SAVE_TASK_FAILED",
        }
    }
}
