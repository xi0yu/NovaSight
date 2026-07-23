//! Typed local client for the daemon-owned control surface.
//!
//! The CLI uses HTTP/1.1 over a Unix domain socket. The paths and JSON
//! bodies are identical to the Web control API, so local and remote
//! callers cannot create competing runtime semantics.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::{Method, Request, StatusCode, client::conn::http1};
use hyper_util::rt::TokioIo;
pub use novasight_core::control::humanized_motion::MotionProfile;
use novasight_core::{CaptureCapabilities, CaptureSelectionPreference, DeviceReceipt};
use novasight_runtime::{
    AppConfig, ConfigFieldUpdate, ConfigUpdate, CrosshairSnapshot, ModelIngressResult,
    ModelProbeInputMode, ModelProfileConfigureRequest, MotionProfileStatus, PreviewSnapshot,
    RuntimeSnapshot,
};
pub use novasight_store::license::LicenseStatus;
pub use novasight_store::model_catalog::{
    CatalogEngineRegistration, Deployment, ModelArtifact, ModelProject, ModelVersion,
};
pub use novasight_store::motion_profile::{
    MotionSampleInput, MotionSampleResult, MotionSessionSummary,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use thiserror::Error;
use tokio::net::UnixStream;

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelSwitchResponse {
    pub deployment: Deployment,
    pub inference: Value,
    pub parser_contract: Option<Value>,
    pub preparation: ModelPreparation,
    pub report: ModelSwitchReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelPreparation {
    pub manifest_action: String,
    pub reason: String,
    pub input_shape: String,
    pub classes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelSwitchReport {
    pub action: String,
    pub applied: bool,
    pub rolled_back: bool,
    pub message: String,
    pub runtime_error: String,
    pub artifact_id: i64,
    pub previous_artifact_id: Option<i64>,
    pub artifact_path: String,
    pub backend: String,
    pub input_shape: String,
    pub classes: usize,
    pub sections: Vec<ModelSwitchSection>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelSwitchSection {
    pub section: String,
    pub impact: String,
    pub status: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutorStatus {
    pub selected: String,
    pub executors: std::collections::BTreeMap<String, ExecutorAvailability>,
    pub state: String,
    pub last_error: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutorAvailability {
    pub available: bool,
    pub connected: bool,
    pub connecting: bool,
    pub monitoring: bool,
    pub connection_state: String,
    pub retryable: bool,
    pub last_error: Option<String>,
    pub managed_by_runtime: bool,
    pub move_count: u64,
    pub last_dx: Option<i32>,
    pub last_dy: Option<i32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeviceButtons {
    pub available: bool,
    pub left: bool,
    pub right: bool,
    pub managed_by_runtime: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiagnosticMoveResponse {
    pub sent: bool,
    pub queued: bool,
    pub steps_sent: u32,
    pub message: String,
    pub receipt: DeviceReceipt,
    pub status: DiagnosticDeviceStatus,
    pub metadata: DiagnosticMetadata,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiagnosticDeviceStatus {
    pub available: bool,
    pub connected: bool,
    pub connection_state: String,
    pub move_count: u64,
    pub last_dx: i32,
    pub last_dy: i32,
    pub managed_by_runtime: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiagnosticMetadata {
    pub api_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ActivatedMotionProfile {
    pub profile: MotionProfile,
    pub runtime: MotionProfileStatus,
}

#[derive(Clone, Debug)]
pub struct ControlClient {
    socket: PathBuf,
    request_timeout: Duration,
}

impl ControlClient {
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub async fn status(&self) -> Result<RuntimeSnapshot, ClientError> {
        self.request(Method::GET, "/api/v1/status", None::<&()>)
            .await
    }

    pub async fn start(&self) -> Result<RuntimeSnapshot, ClientError> {
        self.request(Method::POST, "/api/v1/runtime/start", None::<&()>)
            .await
    }

    pub async fn stop(&self) -> Result<RuntimeSnapshot, ClientError> {
        self.request(Method::POST, "/api/v1/runtime/stop", None::<&()>)
            .await
    }

    pub async fn restart(&self) -> Result<RuntimeSnapshot, ClientError> {
        self.request(Method::POST, "/api/v1/runtime/restart", None::<&()>)
            .await
    }

    pub async fn emergency_stop(&self) -> Result<RuntimeSnapshot, ClientError> {
        self.request(Method::POST, "/api/v1/runtime/emergency-stop", None::<&()>)
            .await
    }

    pub async fn config(&self) -> Result<AppConfig, ClientError> {
        self.request(Method::GET, "/api/v1/config", None::<&()>)
            .await
    }

    pub async fn update_config_field(
        &self,
        section: impl Into<String>,
        key: impl Into<String>,
        value: Value,
        expected_revision: Option<u64>,
    ) -> Result<ConfigUpdate, ClientError> {
        let update = ConfigFieldUpdate {
            section: section.into(),
            key: key.into(),
            value,
            expected_revision,
        };
        self.request(Method::PATCH, "/api/v1/config", Some(&update))
            .await
    }

    pub async fn license_status(&self) -> Result<LicenseStatus, ClientError> {
        self.request(Method::GET, "/api/license", None::<&()>).await
    }

    pub async fn activate_license(&self, key: &str) -> Result<LicenseStatus, ClientError> {
        #[derive(Serialize)]
        struct ActivateLicenseRequest<'a> {
            key: &'a str,
        }
        self.request(
            Method::PUT,
            "/api/license",
            Some(&ActivateLicenseRequest { key }),
        )
        .await
    }

    pub async fn clear_license(&self) -> Result<LicenseStatus, ClientError> {
        self.request(Method::DELETE, "/api/license", None::<&()>)
            .await
    }

    pub async fn model_projects(&self) -> Result<Vec<ModelProject>, ClientError> {
        self.request(Method::GET, "/api/models/projects", None::<&()>)
            .await
    }

    pub async fn model_versions(&self, project_id: i64) -> Result<Vec<ModelVersion>, ClientError> {
        self.request(
            Method::GET,
            &format!("/api/models/projects/{project_id}/versions"),
            None::<&()>,
        )
        .await
    }

    pub async fn model_artifacts(
        &self,
        version_id: i64,
    ) -> Result<Vec<ModelArtifact>, ClientError> {
        self.request(
            Method::GET,
            &format!("/api/models/versions/{version_id}/artifacts"),
            None::<&()>,
        )
        .await
    }

    pub async fn register_catalog_engine(
        &self,
        relative_path: &str,
    ) -> Result<CatalogEngineRegistration, ClientError> {
        #[derive(Serialize)]
        struct RegisterCatalogEngineRequest<'a> {
            relative_path: &'a str,
        }
        self.request(
            Method::POST,
            "/api/models/catalog/register",
            Some(&RegisterCatalogEngineRequest { relative_path }),
        )
        .await
    }

    pub async fn publish_model(
        &self,
        project_id: i64,
        artifact_id: i64,
        parser_preset: &str,
    ) -> Result<ModelSwitchResponse, ClientError> {
        #[derive(Serialize)]
        struct PublishModelRequest<'a> {
            artifact_id: i64,
            parser_preset: &'a str,
        }
        self.request(
            Method::POST,
            &format!("/api/models/projects/{project_id}/publish"),
            Some(&PublishModelRequest {
                artifact_id,
                parser_preset,
            }),
        )
        .await
    }

    pub async fn rollback_model(
        &self,
        project_id: i64,
    ) -> Result<ModelSwitchResponse, ClientError> {
        self.request(
            Method::POST,
            &format!("/api/models/projects/{project_id}/rollback"),
            None::<&()>,
        )
        .await
    }

    pub async fn executor_status(&self) -> Result<ExecutorStatus, ClientError> {
        self.request(Method::GET, "/api/executors", None::<&()>)
            .await
    }

    pub async fn diagnostic_move(
        &self,
        dx: i32,
        dy: i32,
    ) -> Result<DiagnosticMoveResponse, ClientError> {
        #[derive(Serialize)]
        struct DiagnosticMoveRequest {
            dx: i32,
            dy: i32,
            repeat: u32,
            interval_ms: u64,
            move_kind: &'static str,
        }
        self.request(
            Method::POST,
            "/api/executors/kmnet/diagnostic-move",
            Some(&DiagnosticMoveRequest {
                dx,
                dy,
                repeat: 1,
                interval_ms: 0,
                move_kind: "raw",
            }),
        )
        .await
    }

    pub async fn device_buttons(&self) -> Result<DeviceButtons, ClientError> {
        self.request(Method::GET, "/api/executors/kmnet/buttons", None::<&()>)
            .await
    }

    pub async fn capture_state(&self) -> Result<Value, ClientError> {
        self.request(Method::GET, "/api/capture/state", None::<&()>)
            .await
    }

    pub async fn preview_status(&self) -> Result<PreviewSnapshot, ClientError> {
        self.request(Method::GET, "/api/capture/preview", None::<&()>)
            .await
    }

    pub async fn set_preview_active(&self, enabled: bool) -> Result<PreviewSnapshot, ClientError> {
        #[derive(Serialize)]
        struct PreviewRequest {
            enabled: bool,
        }
        self.request(
            Method::POST,
            "/api/capture/preview",
            Some(&PreviewRequest { enabled }),
        )
        .await
    }

    pub async fn crosshair_status(&self) -> Result<CrosshairSnapshot, ClientError> {
        self.request(Method::GET, "/api/crosshair", None::<&()>)
            .await
    }

    pub async fn learn_crosshair(&self) -> Result<Value, ClientError> {
        self.request(Method::POST, "/api/crosshair/learn", None::<&()>)
            .await
    }

    pub async fn clear_crosshair(&self) -> Result<CrosshairSnapshot, ClientError> {
        self.request(Method::DELETE, "/api/crosshair/template", None::<&()>)
            .await
    }

    pub async fn motion_sessions(&self) -> Result<Vec<MotionSessionSummary>, ClientError> {
        self.request(Method::GET, "/api/motion/sessions", None::<&()>)
            .await
    }

    pub async fn create_motion_session(
        &self,
        name: &str,
    ) -> Result<MotionSessionSummary, ClientError> {
        #[derive(Serialize)]
        struct Request<'a> {
            name: &'a str,
        }
        self.request(
            Method::POST,
            "/api/motion/sessions",
            Some(&Request { name }),
        )
        .await
    }

    pub async fn add_motion_sample(
        &self,
        session_id: &str,
        sample: &MotionSampleInput,
    ) -> Result<MotionSampleResult, ClientError> {
        self.request(
            Method::POST,
            &format!("/api/motion/sessions/{session_id}/samples"),
            Some(sample),
        )
        .await
    }

    pub async fn train_motion_profile(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<MotionProfile, ClientError> {
        #[derive(Serialize)]
        struct Request<'a> {
            session_id: &'a str,
            name: &'a str,
        }
        self.request(
            Method::POST,
            "/api/motion/profiles/train",
            Some(&Request { session_id, name }),
        )
        .await
    }

    pub async fn motion_profiles(&self) -> Result<Vec<MotionProfile>, ClientError> {
        self.request(Method::GET, "/api/motion/profiles", None::<&()>)
            .await
    }

    pub async fn activate_motion_profile(
        &self,
        profile_id: &str,
    ) -> Result<ActivatedMotionProfile, ClientError> {
        self.request(
            Method::POST,
            &format!("/api/motion/profiles/{profile_id}/activate"),
            None::<&()>,
        )
        .await
    }

    pub async fn activate_builtin_motion(&self) -> Result<MotionProfileStatus, ClientError> {
        self.request(
            Method::POST,
            "/api/motion/runtime/builtin/activate",
            None::<&()>,
        )
        .await
    }

    pub async fn disable_motion_profile(&self) -> Result<MotionProfileStatus, ClientError> {
        self.request(Method::POST, "/api/motion/profiles/disable", None::<&()>)
            .await
    }

    pub async fn motion_profile_status(&self) -> Result<MotionProfileStatus, ClientError> {
        self.request(Method::GET, "/api/motion/runtime", None::<&()>)
            .await
    }

    pub async fn capture_capabilities(
        &self,
        device: &str,
    ) -> Result<CaptureCapabilities, ClientError> {
        #[derive(Serialize)]
        struct CaptureCapabilitiesRequest<'a> {
            device: &'a str,
        }
        self.request(
            Method::POST,
            "/api/capture/capabilities",
            Some(&CaptureCapabilitiesRequest { device }),
        )
        .await
    }

    pub async fn select_capture(
        &self,
        device: &str,
        preference: CaptureSelectionPreference,
        manual: Option<(&str, u32, u32, u32)>,
    ) -> Result<Value, ClientError> {
        #[derive(Serialize)]
        struct CaptureSelectRequest<'a> {
            device: &'a str,
            preference: CaptureSelectionPreference,
            #[serde(skip_serializing_if = "Option::is_none")]
            pixel_format: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            width: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            height: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            fps: Option<u32>,
        }
        let (pixel_format, width, height, fps) = manual
            .map(|(format, width, height, fps)| {
                (Some(format), Some(width), Some(height), Some(fps))
            })
            .unwrap_or((None, None, None, None));
        self.request(
            Method::POST,
            "/api/capture/select",
            Some(&CaptureSelectRequest {
                device,
                preference,
                pixel_format,
                width,
                height,
                fps,
            }),
        )
        .await
    }

    pub async fn inspect_model(&self, artifact_id: i64) -> Result<ModelIngressResult, ClientError> {
        self.request(
            Method::POST,
            &format!("/api/models/artifacts/{artifact_id}/inspect"),
            None::<&()>,
        )
        .await
    }

    pub async fn model_profile(&self, artifact_id: i64) -> Result<ModelIngressResult, ClientError> {
        self.request(
            Method::GET,
            &format!("/api/models/artifacts/{artifact_id}/profile"),
            None::<&()>,
        )
        .await
    }

    pub async fn deepstream_recommendation(&self, artifact_id: i64) -> Result<Value, ClientError> {
        self.request(
            Method::GET,
            &format!("/api/models/artifacts/{artifact_id}/deepstream/recommendation"),
            None::<&()>,
        )
        .await
    }

    pub async fn configure_model(
        &self,
        artifact_id: i64,
        profile: &ModelProfileConfigureRequest,
    ) -> Result<ModelIngressResult, ClientError> {
        self.request(
            Method::PUT,
            &format!("/api/models/artifacts/{artifact_id}/profile"),
            Some(profile),
        )
        .await
    }

    pub async fn probe_model(
        &self,
        artifact_id: i64,
        input_mode: ModelProbeInputMode,
    ) -> Result<ModelIngressResult, ClientError> {
        #[derive(Serialize)]
        struct ProbeRequest {
            input_mode: ModelProbeInputMode,
        }
        self.request(
            Method::POST,
            &format!("/api/models/artifacts/{artifact_id}/probe"),
            Some(&ProbeRequest { input_mode }),
        )
        .await
    }

    async fn request<T, B>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
    ) -> Result<T, ClientError>
    where
        T: DeserializeOwned,
        B: Serialize + ?Sized,
    {
        let body = body
            .map(serde_json::to_vec)
            .transpose()
            .map_err(ClientError::EncodeRequest)?;
        tokio::time::timeout(self.request_timeout, self.request_inner(method, path, body))
            .await
            .map_err(|_| ClientError::Timeout(self.request_timeout))?
    }

    async fn request_inner<T>(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<T, ClientError>
    where
        T: DeserializeOwned,
    {
        let stream =
            UnixStream::connect(&self.socket)
                .await
                .map_err(|source| ClientError::Connect {
                    path: self.socket.clone(),
                    source,
                })?;
        let (mut sender, connection) = http1::handshake(TokioIo::new(stream))
            .await
            .map_err(ClientError::Handshake)?;
        let mut request = Request::builder()
            .method(method)
            .uri(path)
            .header("host", "localhost")
            .header("connection", "close");
        let body = match body {
            Some(body) => {
                request = request.header("content-type", "application/json");
                Bytes::from(body)
            }
            None => Bytes::new(),
        };
        let request = request
            .body(Full::new(body))
            .map_err(ClientError::BuildRequest)?;
        let exchange = async {
            let response = sender
                .send_request(request)
                .await
                .map_err(ClientError::Request)?;
            let status = response.status();
            let body = Limited::new(response.into_body(), MAX_RESPONSE_BYTES)
                .collect()
                .await
                .map_err(|error| ClientError::ResponseBody(error.to_string()))?
                .to_bytes();
            Ok::<_, ClientError>((status, body))
        };
        tokio::pin!(exchange);
        let connection = connection.without_shutdown();
        tokio::pin!(connection);
        let (status, body) = tokio::select! {
            biased;
            result = exchange.as_mut() => result?,
            result = connection.as_mut() => {
                result.map_err(ClientError::Connection)?;
                exchange.await?
            },
        };

        if status.is_success() {
            serde_json::from_slice(&body).map_err(ClientError::DecodeResponse)
        } else {
            let error: DaemonErrorBody =
                serde_json::from_slice(&body).map_err(ClientError::DecodeDaemonError)?;
            let code = error
                .code
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| format!("HTTP_{}", status.as_u16()));
            let message = error
                .message
                .filter(|value| !value.trim().is_empty())
                .or_else(|| error.detail.filter(|value| !value.trim().is_empty()))
                .unwrap_or_else(|| {
                    status
                        .canonical_reason()
                        .unwrap_or("daemon rejected the command")
                        .to_owned()
                });
            Err(ClientError::Daemon {
                status,
                code,
                message,
            })
        }
    }
}

#[derive(Debug, Deserialize)]
struct DaemonErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    detail: Option<String>,
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("failed to connect to NovaSight daemon socket {}: {source}", path.display())]
    Connect {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("daemon request exceeded the {0:?} timeout")]
    Timeout(Duration),
    #[error("failed to establish HTTP over the daemon socket: {0}")]
    Handshake(#[source] hyper::Error),
    #[error("failed to build daemon request: {0}")]
    BuildRequest(#[source] hyper::http::Error),
    #[error("failed to encode daemon request: {0}")]
    EncodeRequest(#[source] serde_json::Error),
    #[error("daemon request failed: {0}")]
    Request(#[source] hyper::Error),
    #[error("daemon response body failed: {0}")]
    ResponseBody(String),
    #[error("daemon connection failed: {0}")]
    Connection(#[source] hyper::Error),
    #[error("daemon returned an invalid JSON response: {0}")]
    DecodeResponse(#[source] serde_json::Error),
    #[error("daemon returned an invalid error body: {0}")]
    DecodeDaemonError(#[source] serde_json::Error),
    #[error("daemon rejected the command ({status} {code}): {message}")]
    Daemon {
        status: StatusCode,
        code: String,
        message: String,
    },
}

impl ClientError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Connect { .. } => "daemon_connect_failed",
            Self::Timeout(_) => "daemon_request_timed_out",
            Self::Handshake(_) => "daemon_handshake_failed",
            Self::BuildRequest(_) => "daemon_request_build_failed",
            Self::EncodeRequest(_) => "daemon_request_encode_failed",
            Self::Request(_) => "daemon_request_failed",
            Self::ResponseBody(_) => "daemon_response_failed",
            Self::Connection(_) => "daemon_connection_failed",
            Self::DecodeResponse(_) => "daemon_response_invalid",
            Self::DecodeDaemonError(_) => "daemon_error_response_invalid",
            Self::Daemon { .. } => "daemon_command_rejected",
        }
    }
}
