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
use novasight_runtime::{AppConfig, ConfigFieldUpdate, ConfigUpdate, RuntimeSnapshot};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use thiserror::Error;
use tokio::net::UnixStream;

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

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

    async fn request<T, B>(
        &self,
        method: Method,
        path: &'static str,
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
        path: &'static str,
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
            Err(ClientError::Daemon {
                status,
                code: error.code,
                message: error.message,
            })
        }
    }
}

#[derive(Debug, Deserialize)]
struct DaemonErrorBody {
    code: String,
    message: String,
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
