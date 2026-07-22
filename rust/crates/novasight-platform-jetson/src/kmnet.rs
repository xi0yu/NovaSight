//! Crash-isolated kmNet adapter.
//!
//! The vendor artifact is a CPython extension, not a stable C ABI. This client
//! keeps the Rust daemon authoritative while owning a narrow JSON-lines helper
//! process that can be killed on timeout or protocol corruption.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TrySendError, sync_channel};
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use novasight_core::{AppError, DeviceCommand, DeviceReceipt, PointerButtons, PointerDevice};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

const PROTOCOL_VERSION: u64 = 1;
type HostResponseResult = Result<HostResponse, String>;
type HostResponseReceiver = Receiver<HostResponseResult>;

#[derive(Clone, Debug)]
pub struct KmNetHostConfig {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub host: String,
    pub port: u16,
    pub uuid: String,
    pub monitor_port: u16,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub reconnect_cooldown: Duration,
}

impl KmNetHostConfig {
    pub fn validate(&self) -> Result<(), KmNetError> {
        if self.program.as_os_str().is_empty() {
            return Err(KmNetError::InvalidConfig("helper program is empty"));
        }
        if self.host.trim().is_empty() {
            return Err(KmNetError::InvalidConfig("kmNet host is empty"));
        }
        if self.port == 0 {
            return Err(KmNetError::InvalidConfig("kmNet port must be non-zero"));
        }
        if self.uuid.trim().is_empty() {
            return Err(KmNetError::InvalidConfig("kmNet UUID is empty"));
        }
        if self.startup_timeout.is_zero() || self.request_timeout.is_zero() {
            return Err(KmNetError::InvalidConfig("kmNet timeouts must be non-zero"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct KmNetHostClient {
    config: KmNetHostConfig,
    state: Mutex<ClientState>,
    successful_sends: AtomicU64,
}

#[derive(Debug)]
struct ClientState {
    session: Option<HostSession>,
    last_failure: Option<Instant>,
}

impl KmNetHostClient {
    /// Construct a validated but disconnected adapter. Production composition
    /// uses this so daemon control surfaces remain available while hardware is
    /// offline; the runtime epoch owns the actual helper session.
    pub fn new(config: KmNetHostConfig) -> Result<Self, KmNetError> {
        config.validate()?;
        Ok(Self {
            config,
            state: Mutex::new(ClientState {
                session: None,
                last_failure: None,
            }),
            successful_sends: AtomicU64::new(0),
        })
    }

    /// Eager compatibility constructor used by focused diagnostics and tests.
    pub fn connect(config: KmNetHostConfig) -> Result<Self, KmNetError> {
        let client = Self::new(config)?;
        client.connect_inner()?;
        Ok(client)
    }

    fn state(&self) -> MutexGuard<'_, ClientState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn send_inner(&self, command: DeviceCommand) -> Result<DeviceReceipt, KmNetError> {
        if i16::try_from(command.delta_x_counts).is_err()
            || i16::try_from(command.delta_y_counts).is_err()
        {
            return Err(KmNetError::CountsOutOfRange {
                dx: command.delta_x_counts,
                dy: command.delta_y_counts,
            });
        }
        let mut state = self.state();
        let result = self.ensure_session(&mut state)?.request(
            "move",
            json!({
                "dx": command.delta_x_counts,
                "dy": command.delta_y_counts,
            }),
            self.config.request_timeout,
        );
        if let Err(error) = result {
            if let Some(mut session) = state.session.take() {
                session.abort();
            }
            state.last_failure = Some(Instant::now());
            return Err(error);
        }
        state.last_failure = None;
        let attempt = self.successful_sends.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(DeviceReceipt::accepted(attempt, command))
    }

    fn buttons_inner(&self) -> Result<PointerButtons, KmNetError> {
        let mut state = self.state();
        let result = self
            .ensure_session(&mut state)?
            .request("buttons", json!({}), self.config.request_timeout)
            .and_then(|value| {
                if value.get("available").and_then(Value::as_bool) != Some(true) {
                    return Err(KmNetError::Protocol(
                        "helper does not expose a hardware trigger".to_owned(),
                    ));
                }
                let left = value.get("left").and_then(Value::as_bool).ok_or_else(|| {
                    KmNetError::Protocol("buttons result is missing boolean left".to_owned())
                })?;
                let right = value.get("right").and_then(Value::as_bool).ok_or_else(|| {
                    KmNetError::Protocol("buttons result is missing boolean right".to_owned())
                })?;
                Ok(PointerButtons { left, right })
            });
        match result {
            Ok(buttons) => {
                state.last_failure = None;
                Ok(buttons)
            }
            Err(error) => {
                if let Some(mut session) = state.session.take() {
                    session.abort();
                }
                state.last_failure = Some(Instant::now());
                Err(error)
            }
        }
    }

    fn connect_inner(&self) -> Result<(), KmNetError> {
        let mut state = self.state();
        self.ensure_session(&mut state).map(|_| ())
    }

    fn ensure_session<'a>(
        &self,
        state: &'a mut ClientState,
    ) -> Result<&'a mut HostSession, KmNetError> {
        if state.session.is_none() {
            if let Some(remaining) =
                cooldown_remaining(state.last_failure, self.config.reconnect_cooldown)
            {
                return Err(KmNetError::ReconnectCooldown(remaining));
            }
            match HostSession::start(&self.config) {
                Ok(session) => state.session = Some(session),
                Err(error) => {
                    state.last_failure = Some(Instant::now());
                    return Err(error);
                }
            }
        }
        Ok(state
            .session
            .as_mut()
            .expect("session was established above"))
    }

    fn disconnect_inner(&self) {
        let mut state = self.state();
        if let Some(mut session) = state.session.take() {
            session.abort();
        }
        state.last_failure = None;
    }
}

impl PointerDevice for KmNetHostClient {
    fn connect(&self) -> Result<(), AppError> {
        self.connect_inner()
            .map_err(|error| AppError::PointerDevice {
                code: error.code(),
                message: error.to_string(),
            })
    }

    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        self.send_inner(command)
            .map_err(|error| AppError::PointerDevice {
                code: error.code(),
                message: error.to_string(),
            })
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        self.buttons_inner()
            .map(|buttons| Some(buttons.trigger_active()))
            .map_err(|error| AppError::PointerDevice {
                code: error.code(),
                message: error.to_string(),
            })
    }

    fn buttons(&self) -> Result<Option<PointerButtons>, AppError> {
        self.buttons_inner()
            .map(Some)
            .map_err(|error| AppError::PointerDevice {
                code: error.code(),
                message: error.to_string(),
            })
    }

    fn disconnect(&self) -> Result<(), AppError> {
        self.disconnect_inner();
        Ok(())
    }
}

impl Drop for KmNetHostClient {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(mut session) = state.session.take() {
            session.abort();
        }
    }
}

fn cooldown_remaining(last_failure: Option<Instant>, cooldown: Duration) -> Option<Duration> {
    let elapsed = last_failure?.elapsed();
    (elapsed < cooldown).then(|| cooldown - elapsed)
}

#[derive(Debug)]
struct HostSession {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    responses: HostResponseReceiver,
    reader: Option<JoinHandle<()>>,
    next_request_id: u64,
}

impl HostSession {
    fn start(config: &KmNetHostConfig) -> Result<Self, KmNetError> {
        let mut child = Command::new(&config.program)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|source| KmNetError::Spawn {
                program: config.program.clone(),
                source,
            })?;
        let stdin = child.stdin.take().ok_or(KmNetError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(KmNetError::MissingPipe("stdout"))?;
        let (responses, reader) = spawn_response_reader(stdout)?;
        let mut session = Self {
            child,
            stdin: BufWriter::new(stdin),
            responses,
            reader: Some(reader),
            next_request_id: 0,
        };
        let result = (|| {
            let hello = session.request(
                "hello",
                json!({ "protocol": PROTOCOL_VERSION }),
                config.startup_timeout,
            )?;
            if hello.get("protocol").and_then(Value::as_u64) != Some(PROTOCOL_VERSION) {
                return Err(KmNetError::Protocol(
                    "helper protocol version mismatch".to_owned(),
                ));
            }
            session.request(
                "connect",
                json!({
                    "host": config.host,
                    "port": config.port,
                    "uuid": config.uuid,
                    "monitor_port": config.monitor_port,
                }),
                config.startup_timeout,
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            session.abort();
            return Err(error);
        }
        Ok(session)
    }

    fn request(
        &mut self,
        operation: &'static str,
        fields: Value,
        timeout: Duration,
    ) -> Result<Value, KmNetError> {
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(KmNetError::RequestIdExhausted)?;
        let request_id = self.next_request_id;
        let mut request = fields
            .as_object()
            .cloned()
            .ok_or_else(|| KmNetError::Protocol("request fields are not an object".to_owned()))?;
        request.insert("id".to_owned(), json!(request_id));
        request.insert("op".to_owned(), json!(operation));
        serde_json::to_writer(&mut self.stdin, &request).map_err(KmNetError::Encode)?;
        self.stdin.write_all(b"\n").map_err(KmNetError::Write)?;
        self.stdin.flush().map_err(KmNetError::Write)?;
        let response = match self.responses.recv_timeout(timeout) {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => return Err(KmNetError::Protocol(error)),
            Err(RecvTimeoutError::Timeout) => return Err(KmNetError::Timeout(operation)),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(KmNetError::HelperExited {
                    operation,
                    status: self
                        .child
                        .try_wait()
                        .ok()
                        .flatten()
                        .map(|status| status.to_string()),
                });
            }
        };
        if response.id != request_id {
            return Err(KmNetError::ResponseMismatch {
                expected: request_id,
                actual: response.id,
            });
        }
        if !response.ok {
            return Err(KmNetError::Driver {
                operation,
                message: response
                    .error
                    .unwrap_or_else(|| "unknown driver error".to_owned()),
            });
        }
        Ok(response.result.unwrap_or(Value::Null))
    }

    fn abort(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for HostSession {
    fn drop(&mut self) {
        self.abort();
    }
}

fn spawn_response_reader(
    stdout: ChildStdout,
) -> Result<(HostResponseReceiver, JoinHandle<()>), KmNetError> {
    let (sender, receiver) = sync_channel(1);
    let reader = thread::Builder::new()
        .name("novasight-kmnet-host-reader".to_owned())
        .spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let response = match line {
                    Ok(line) => serde_json::from_str::<HostResponse>(&line)
                        .map_err(|error| format!("invalid helper response: {error}")),
                    Err(error) => Err(format!("failed to read helper response: {error}")),
                };
                match sender.try_send(response) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => return,
                }
            }
        })
        .map_err(KmNetError::SpawnReader)?;
    Ok((receiver, reader))
}

#[derive(Debug, Deserialize)]
struct HostResponse {
    id: u64,
    ok: bool,
    result: Option<Value>,
    error: Option<String>,
}

#[derive(Debug, Error)]
pub enum KmNetError {
    #[error("invalid kmNet host configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("failed to spawn kmNet helper {}: {source}", program.display())]
    Spawn {
        program: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("kmNet helper did not expose {0}")]
    MissingPipe(&'static str),
    #[error("failed to spawn kmNet response reader: {0}")]
    SpawnReader(std::io::Error),
    #[error("failed to encode kmNet helper request: {0}")]
    Encode(serde_json::Error),
    #[error("failed to write kmNet helper request: {0}")]
    Write(std::io::Error),
    #[error("kmNet helper protocol error: {0}")]
    Protocol(String),
    #[error("kmNet helper request ID exhausted")]
    RequestIdExhausted,
    #[error("kmNet helper response mismatch: expected {expected}, got {actual}")]
    ResponseMismatch { expected: u64, actual: u64 },
    #[error("kmNet driver call timed out: {0}")]
    Timeout(&'static str),
    #[error("kmNet helper exited during {operation}: {status:?}")]
    HelperExited {
        operation: &'static str,
        status: Option<String>,
    },
    #[error("kmNet driver {operation} failed: {message}")]
    Driver {
        operation: &'static str,
        message: String,
    },
    #[error("kmNet reconnect cooldown active for {0:?}")]
    ReconnectCooldown(Duration),
    #[error("kmNet movement ({dx}, {dy}) is outside the signed 16-bit device range")]
    CountsOutOfRange { dx: i32, dy: i32 },
}

impl KmNetError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig(_) => "config_invalid",
            Self::Spawn { .. } | Self::MissingPipe(_) | Self::SpawnReader(_) => {
                "helper_spawn_failed"
            }
            Self::Encode(_)
            | Self::Write(_)
            | Self::Protocol(_)
            | Self::RequestIdExhausted
            | Self::ResponseMismatch { .. } => "helper_protocol_failed",
            Self::Timeout(_) => "driver_timeout",
            Self::HelperExited { .. } => "helper_exited",
            Self::Driver { .. } => "driver_rejected",
            Self::ReconnectCooldown(_) => "reconnect_cooldown",
            Self::CountsOutOfRange { .. } => "counts_out_of_range",
        }
    }
}
