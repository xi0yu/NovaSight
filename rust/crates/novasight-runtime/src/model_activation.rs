use novasight_pipeline::PerceptionModelContract;
use novasight_store::model_catalog::{
    ActiveModelDeployment, Deployment, ModelCatalogError, RuntimeModelArtifact,
};
use thiserror::Error;

use crate::{RuntimeError, RuntimeSnapshot};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelActivationRequest {
    Publish {
        project_id: i64,
        artifact_id: i64,
        parser_preset: String,
    },
    Rollback {
        project_id: i64,
    },
}

impl ModelActivationRequest {
    pub const fn project_id(&self) -> i64 {
        match self {
            Self::Publish { project_id, .. } | Self::Rollback { project_id } => *project_id,
        }
    }

    pub const fn label(&self) -> &'static str {
        match self {
            Self::Publish { .. } => "publish",
            Self::Rollback { .. } => "rollback",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelActivationResult {
    pub action: &'static str,
    pub deployment: Deployment,
    pub active: ActiveModelDeployment,
    pub candidate: RuntimeModelArtifact,
    pub contract: Option<PerceptionModelContract>,
    pub runtime: RuntimeSnapshot,
    pub restarted: bool,
    pub changed: bool,
}

#[derive(Debug, Error)]
pub enum ModelActivationError {
    #[error(transparent)]
    Catalog(#[from] ModelCatalogError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error("model {action} failed: {message}")]
    Failed {
        action: &'static str,
        message: String,
    },
    #[error("model catalog is not configured in the runtime supervisor")]
    Unavailable,
    #[error("model activation requires a configured perception preflight adapter")]
    PerceptionUnavailable,
}
