//! Production HTTP, Unix-socket, and shutdown server ownership.

use std::fs::File;
use std::future::IntoFuture;
use std::io;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use novasight_api::build_control_router_with_platform_queries;
use novasight_core::CaptureCapabilityProbe;
use novasight_runtime::{
    ApplicationError, ConfigService, LoadedApplication, PipelineState, RuntimeDependencies,
    RuntimeError, RuntimeHandle,
};
use novasight_store::license::{FileLicenseRepository, LicenseError, LicensePolicy};
use novasight_store::model_catalog::SqliteModelCatalog;
use thiserror::Error;
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DaemonMode {
    DryRun,
    Production,
}

impl DaemonMode {
    const fn label(self) -> &'static str {
        match self {
            Self::DryRun => "dry-run",
            Self::Production => "production",
        }
    }

    const fn hardware_output_enabled(self) -> bool {
        match self {
            Self::DryRun => false,
            Self::Production => true,
        }
    }
}

pub(super) async fn run_daemon(
    loaded: LoadedApplication,
    dependencies: RuntimeDependencies,
    config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    mode: DaemonMode,
) -> Result<(), DaemonRunError> {
    let _instance_lock = acquire_instance_lock(mode.hardware_output_enabled())?;
    let host = loaded.config().server.host.clone();
    let port = loaded.config().server.port;
    let control_socket = loaded.config().server.control_socket.clone();
    let license_repository =
        FileLicenseRepository::new(loaded.config().paths.license.clone(), license_policy(mode)?);
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|source| DaemonRunError::Bind {
            host: host.clone(),
            port,
            source,
        })?;
    let address = listener
        .local_addr()
        .map_err(DaemonRunError::LocalAddress)?;
    let (control_listener, _control_socket_guard, _control_directory_lock) =
        bind_control_socket(&control_socket).await?;
    let mut signals = ShutdownSignals::register()?;
    let application = loaded.start(dependencies);
    let (server_shutdown_tx, mut server_shutdown_rx) = watch::channel(false);
    let capture_probe = platform_capture_probe();
    let runtime = application.runtime();
    let router = build_control_router_with_platform_queries(
        runtime.clone(),
        config_service,
        license_repository.clone(),
        model_catalog,
        capture_probe,
        mode.hardware_output_enabled(),
        server_shutdown_rx.clone(),
    );
    let http_server = axum::serve(listener, router.clone())
        .with_graceful_shutdown(async move {
            if !*server_shutdown_rx.borrow() {
                let _ = server_shutdown_rx.changed().await;
            }
        })
        .into_future();
    let mut control_shutdown_rx = server_shutdown_tx.subscribe();
    let control_server = axum::serve(control_listener, router)
        .with_graceful_shutdown(async move {
            if !*control_shutdown_rx.borrow() {
                let _ = control_shutdown_rx.changed().await;
            }
        })
        .into_future();
    let mut http_server = Box::pin(http_server);
    let mut control_server = Box::pin(control_server);
    let mut license_watchdog = Box::pin(monitor_runtime_license(
        license_repository,
        runtime.clone(),
        mode.hardware_output_enabled(),
        server_shutdown_tx.subscribe(),
        Duration::from_secs(1),
    ));
    let mut runtime_exit = Box::pin(runtime.wait_for_supervisor_exit());

    if mode == DaemonMode::DryRun {
        tracing::warn!(
            "novasightd is running in explicit dry-run mode; hardware output is disabled"
        );
    }
    eprintln!(
        "novasightd ready mode={} address={address} socket={}",
        mode.label(),
        control_socket.display()
    );

    let first_exit = tokio::select! {
        result = http_server.as_mut() => FirstExit::Http(result),
        result = control_server.as_mut() => FirstExit::Control(result),
        result = license_watchdog.as_mut() => FirstExit::License(result),
        () = runtime_exit.as_mut() => FirstExit::Runtime,
        () = signals.wait() => FirstExit::Signal,
    };
    server_shutdown_tx.send_replace(true);
    let (service_result, license_result) = match first_exit {
        FirstExit::Signal | FirstExit::Runtime => {
            let (http, control, license) = tokio::join!(
                http_server.as_mut(),
                control_server.as_mut(),
                license_watchdog.as_mut()
            );
            (combine_server_results(http, control), license)
        }
        FirstExit::Http(http) => {
            let (control, license) =
                tokio::join!(control_server.as_mut(), license_watchdog.as_mut());
            (combine_server_results(http, control), license)
        }
        FirstExit::Control(control) => {
            let (http, license) = tokio::join!(http_server.as_mut(), license_watchdog.as_mut());
            (combine_server_results(http, control), license)
        }
        FirstExit::License(license) => {
            let (http, control) = tokio::join!(http_server.as_mut(), control_server.as_mut());
            (combine_server_results(http, control), license)
        }
    };
    let service_result = service_result.and(license_result);
    drop(http_server);
    drop(control_server);
    drop(license_watchdog);
    drop(runtime_exit);
    let shutdown_result = application
        .shutdown()
        .await
        .map_err(DaemonRunError::Application);
    match (service_result, shutdown_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(primary), Err(cleanup)) => Err(DaemonRunError::ServiceAndShutdown {
            primary: Box::new(primary),
            cleanup: Box::new(cleanup),
        }),
    }
}

#[cfg(all(feature = "deepstream", target_os = "linux"))]
fn platform_capture_probe() -> Option<Arc<dyn CaptureCapabilityProbe>> {
    Some(
        Arc::new(novasight_platform_jetson::v4l2::V4l2CapabilityProbe)
            as Arc<dyn CaptureCapabilityProbe>,
    )
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
fn platform_capture_probe() -> Option<Arc<dyn CaptureCapabilityProbe>> {
    None
}

fn license_policy(mode: DaemonMode) -> Result<LicensePolicy, DaemonRunError> {
    let allow_test_key = mode == DaemonMode::DryRun;
    let inline_public_key = std::env::var("NOVASIGHT_LICENSE_PUBLIC_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let public_key = match inline_public_key {
        Some(public_key) => Some(public_key),
        None => std::env::var_os("NOVASIGHT_LICENSE_PUBLIC_KEY_FILE")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .map(|path| {
                std::fs::read_to_string(&path).map_err(|source| {
                    DaemonRunError::LicensePublicKeyRead {
                        path: path.clone(),
                        source,
                    }
                })
            })
            .transpose()?,
    };
    let policy = LicensePolicy::new(allow_test_key, public_key);
    match policy.validate_public_key(mode.hardware_output_enabled()) {
        Ok(()) => Ok(policy),
        Err(LicenseError::PublicKeyMissing) => Err(DaemonRunError::LicensePublicKeyMissing),
        Err(error) => Err(DaemonRunError::LicensePublicKeyInvalid(error)),
    }
}

pub(super) fn preflight_license_policy(mode: DaemonMode) -> Result<(), DaemonRunError> {
    license_policy(mode).map(drop)
}

pub(super) fn preflight_instance_lock(mode: DaemonMode) -> Result<(), DaemonRunError> {
    acquire_instance_lock(mode.hardware_output_enabled()).map(drop)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const INSTANCE_LOCK_SOCKET: &str = "@novasightd-instance";

#[cfg(any(target_os = "linux", target_os = "android"))]
fn acquire_instance_lock(required: bool) -> Result<Option<InstanceLock>, DaemonRunError> {
    if !required {
        return Ok(None);
    }
    let address = abstract_socket_path(Path::new(INSTANCE_LOCK_SOCKET))?;
    UnixListener::bind(address)
        .map(InstanceLock)
        .map(Some)
        .map_err(|source| {
            if source.kind() == io::ErrorKind::AddrInUse {
                DaemonRunError::InstanceLockInUse
            } else {
                DaemonRunError::InstanceLockAcquire(source)
            }
        })
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn acquire_instance_lock(_required: bool) -> Result<Option<InstanceLock>, DaemonRunError> {
    Ok(None)
}

#[derive(Debug)]
struct InstanceLock(#[allow(dead_code)] UnixListener);

async fn monitor_runtime_license(
    repository: FileLicenseRepository,
    runtime: RuntimeHandle,
    hardware_output_enabled: bool,
    mut shutdown: watch::Receiver<bool>,
    interval: Duration,
) -> Result<(), DaemonRunError> {
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return Ok(());
                }
            }
            _ = tick.tick() => {
                let status_repository = repository.clone();
                let authorization = tokio::task::spawn_blocking(move || {
                    status_repository.status().map(|status| {
                        let runtime = status.features.iter().any(|feature| feature == "runtime");
                        let hardware = status
                            .features
                            .iter()
                            .any(|feature| feature == "hardware_control");
                        let authorized = status.configured
                            && status.valid
                            && runtime
                            && (!hardware_output_enabled || hardware);
                        (authorized, status.message)
                    })
                })
                .await;
                let (authorized, reason) = match authorization {
                    Ok(Ok(result)) => result,
                    Ok(Err(error)) => (false, error.to_string()),
                    Err(error) => (false, format!("license verification task failed: {error}")),
                };
                let pipeline = runtime.snapshot().pipeline.state;
                if !authorized
                    && matches!(
                        pipeline,
                        PipelineState::Starting | PipelineState::Running | PipelineState::Standby
                    )
                {
                    tracing::error!(
                        reason = %if reason.is_empty() { "license is missing or lacks required runtime features" } else { &reason },
                        "runtime license became invalid; closing hardware output"
                    );
                    runtime
                        .emergency_stop()
                        .await
                        .map_err(DaemonRunError::LicenseEnforcement)?;
                }
            }
        }
    }
}

enum FirstExit {
    Signal,
    Runtime,
    Http(Result<(), io::Error>),
    Control(Result<(), io::Error>),
    License(Result<(), DaemonRunError>),
}

fn combine_server_results(
    http: Result<(), io::Error>,
    control: Result<(), io::Error>,
) -> Result<(), DaemonRunError> {
    match (http, control) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(DaemonRunError::HttpServe(error)),
        (Ok(()), Err(error)) => Err(DaemonRunError::ControlServe(error)),
        (Err(http), Err(control)) => Err(DaemonRunError::ServersFailed { http, control }),
    }
}

async fn bind_control_socket(
    path: &Path,
) -> Result<
    (
        UnixListener,
        Option<ControlSocketGuard>,
        Option<ControlDirectoryLock>,
    ),
    DaemonRunError,
> {
    if is_abstract_socket(path) {
        let address = abstract_socket_path(path)?;
        let listener =
            UnixListener::bind(address).map_err(|source| DaemonRunError::ControlSocketBind {
                path: path.to_path_buf(),
                source,
            })?;
        return Ok((listener, None, None));
    }

    let directory_lock = lock_control_directory(path)?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_socket() => {
            return Err(DaemonRunError::ControlSocketPathOccupied(
                path.to_path_buf(),
            ));
        }
        Ok(_) => match UnixStream::connect(path).await {
            Ok(_) => return Err(DaemonRunError::ControlSocketInUse(path.to_path_buf())),
            Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {
                std::fs::remove_file(path).map_err(|source| {
                    DaemonRunError::RemoveStaleControlSocket {
                        path: path.to_path_buf(),
                        source,
                    }
                })?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(DaemonRunError::InspectControlSocket {
                    path: path.to_path_buf(),
                    source,
                });
            }
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(DaemonRunError::InspectControlSocket {
                path: path.to_path_buf(),
                source,
            });
        }
    }

    let listener =
        UnixListener::bind(path).map_err(|source| DaemonRunError::ControlSocketBind {
            path: path.to_path_buf(),
            source,
        })?;
    let metadata =
        std::fs::symlink_metadata(path).map_err(|source| DaemonRunError::InspectControlSocket {
            path: path.to_path_buf(),
            source,
        })?;
    let guard = ControlSocketGuard {
        path: path.to_path_buf(),
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660)).map_err(|source| {
        DaemonRunError::ControlSocketPermissions {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok((listener, Some(guard), Some(directory_lock)))
}

fn is_abstract_socket(path: &Path) -> bool {
    path.as_os_str().as_encoded_bytes().starts_with(b"@")
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn abstract_socket_path(path: &Path) -> Result<PathBuf, DaemonRunError> {
    let configured = path.as_os_str().as_bytes();
    if configured.len() == 1 {
        return Err(DaemonRunError::AbstractControlSocketNameEmpty);
    }
    let mut address = Vec::with_capacity(configured.len());
    address.push(0);
    address.extend_from_slice(&configured[1..]);
    Ok(PathBuf::from(std::ffi::OsString::from_vec(address)))
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn abstract_socket_path(_path: &Path) -> Result<PathBuf, DaemonRunError> {
    Err(DaemonRunError::AbstractControlSocketUnsupported)
}

fn lock_control_directory(path: &Path) -> Result<ControlDirectoryLock, DaemonRunError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| DaemonRunError::ControlSocketParentMissing(path.to_path_buf()))?;
    match std::fs::symlink_metadata(parent) {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match std::fs::create_dir(parent) {
                Ok(()) => {
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o750))
                        .map_err(|source| DaemonRunError::ControlSocketDirectoryPermissions {
                            path: parent.to_path_buf(),
                            source,
                        })?;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(source) => {
                    return Err(DaemonRunError::CreateControlSocketDirectory {
                        path: parent.to_path_buf(),
                        source,
                    });
                }
            }
        }
        Err(source) => {
            return Err(DaemonRunError::InspectControlSocketDirectory {
                path: parent.to_path_buf(),
                source,
            });
        }
    }

    let path_metadata = std::fs::symlink_metadata(parent).map_err(|source| {
        DaemonRunError::InspectControlSocketDirectory {
            path: parent.to_path_buf(),
            source,
        }
    })?;
    if !path_metadata.file_type().is_dir() {
        return Err(DaemonRunError::ControlSocketParentNotDirectory(
            parent.to_path_buf(),
        ));
    }
    let mode = path_metadata.mode();
    let writable_by_non_owner = mode & 0o022 != 0;
    let sticky = mode & 0o1000 != 0;
    if writable_by_non_owner && !sticky {
        return Err(DaemonRunError::InsecureControlSocketDirectory {
            path: parent.to_path_buf(),
            mode: mode & 0o7777,
        });
    }

    let directory =
        File::open(parent).map_err(|source| DaemonRunError::OpenControlSocketDirectory {
            path: parent.to_path_buf(),
            source,
        })?;
    let opened_metadata =
        directory
            .metadata()
            .map_err(|source| DaemonRunError::InspectControlSocketDirectory {
                path: parent.to_path_buf(),
                source,
            })?;
    if opened_metadata.dev() != path_metadata.dev() || opened_metadata.ino() != path_metadata.ino()
    {
        return Err(DaemonRunError::ControlSocketDirectoryChanged(
            parent.to_path_buf(),
        ));
    }
    fs2::FileExt::try_lock_exclusive(&directory).map_err(|source| {
        if source.kind() == io::ErrorKind::WouldBlock {
            DaemonRunError::ControlDirectoryInUse(parent.to_path_buf())
        } else {
            DaemonRunError::ControlDirectoryLock {
                path: parent.to_path_buf(),
                source,
            }
        }
    })?;
    Ok(ControlDirectoryLock(directory))
}

#[derive(Debug)]
struct ControlDirectoryLock(#[allow(dead_code)] File);

#[derive(Debug)]
struct ControlSocketGuard {
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl Drop for ControlSocketGuard {
    fn drop(&mut self) {
        let Ok(metadata) = std::fs::symlink_metadata(&self.path) else {
            return;
        };
        if metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

struct ShutdownSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

impl ShutdownSignals {
    fn register() -> Result<Self, DaemonRunError> {
        use tokio::signal::unix::{SignalKind, signal};

        Ok(Self {
            interrupt: signal(SignalKind::interrupt()).map_err(DaemonRunError::Signal)?,
            terminate: signal(SignalKind::terminate()).map_err(DaemonRunError::Signal)?,
        })
    }

    async fn wait(&mut self) {
        tokio::select! {
            _ = self.interrupt.recv() => {}
            _ = self.terminate.recv() => {}
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum DaemonRunError {
    #[error(
        "production mode requires NOVASIGHT_LICENSE_PUBLIC_KEY or NOVASIGHT_LICENSE_PUBLIC_KEY_FILE"
    )]
    LicensePublicKeyMissing,
    #[error("failed to read license public key {}: {source}", path.display())]
    LicensePublicKeyRead {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("configured production license public key is invalid: {0}")]
    LicensePublicKeyInvalid(LicenseError),
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[error("another NovaSight daemon is already running")]
    InstanceLockInUse,
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[error("failed to acquire the automatic NovaSight instance guard: {0}")]
    InstanceLockAcquire(io::Error),
    #[error("failed to stop runtime after license invalidation: {0}")]
    LicenseEnforcement(RuntimeError),
    #[error("failed to bind HTTP server at {host}:{port}: {source}")]
    Bind {
        host: String,
        port: u16,
        #[source]
        source: io::Error,
    },
    #[error("failed to read bound HTTP address: {0}")]
    LocalAddress(io::Error),
    #[error("failed to register shutdown signal: {0}")]
    Signal(io::Error),
    #[error("HTTP server failed: {0}")]
    HttpServe(io::Error),
    #[error("Unix control server failed: {0}")]
    ControlServe(io::Error),
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[error("abstract control socket name after @ cannot be empty")]
    AbstractControlSocketNameEmpty,
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    #[error("abstract control sockets are supported only on Linux and Android")]
    AbstractControlSocketUnsupported,
    #[error("HTTP server failed: {http}; Unix control server also failed: {control}")]
    ServersFailed { http: io::Error, control: io::Error },
    #[error("control socket path is occupied by a non-socket file: {}", .0.display())]
    ControlSocketPathOccupied(PathBuf),
    #[error("another daemon is already listening at control socket {}", .0.display())]
    ControlSocketInUse(PathBuf),
    #[error("failed to inspect control socket {}: {source}", path.display())]
    InspectControlSocket {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to remove stale control socket {}: {source}", path.display())]
    RemoveStaleControlSocket {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to bind control socket {}: {source}", path.display())]
    ControlSocketBind {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to set control socket permissions at {}: {source}", path.display())]
    ControlSocketPermissions {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("control socket has no parent directory: {}", .0.display())]
    ControlSocketParentMissing(PathBuf),
    #[error("failed to create control socket directory {}: {source}", path.display())]
    CreateControlSocketDirectory {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to set control socket directory permissions at {}: {source}", path.display())]
    ControlSocketDirectoryPermissions {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to inspect control socket directory {}: {source}", path.display())]
    InspectControlSocketDirectory {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("control socket parent is not a directory: {}", .0.display())]
    ControlSocketParentNotDirectory(PathBuf),
    #[error("control socket directory {} has insecure mode {mode:o}", path.display())]
    InsecureControlSocketDirectory { path: PathBuf, mode: u32 },
    #[error("failed to open control socket directory {}: {source}", path.display())]
    OpenControlSocketDirectory {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("control socket directory changed while opening: {}", .0.display())]
    ControlSocketDirectoryChanged(PathBuf),
    #[error("another daemon owns control directory {}", .0.display())]
    ControlDirectoryInUse(PathBuf),
    #[error("failed to lock control socket directory {}: {source}", path.display())]
    ControlDirectoryLock {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error("service failed: {primary}; runtime cleanup also failed: {cleanup}")]
    ServiceAndShutdown {
        primary: Box<DaemonRunError>,
        cleanup: Box<DaemonRunError>,
    },
}

impl DaemonRunError {
    pub(super) const fn code(&self) -> &'static str {
        match self {
            Self::LicensePublicKeyMissing => "LICENSE_PUBLIC_KEY_MISSING",
            Self::LicensePublicKeyRead { .. } => "LICENSE_PUBLIC_KEY_READ_FAILED",
            Self::LicensePublicKeyInvalid(_) => "LICENSE_PUBLIC_KEY_INVALID",
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Self::InstanceLockInUse => "INSTANCE_ALREADY_RUNNING",
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Self::InstanceLockAcquire(_) => "INSTANCE_GUARD_FAILED",
            Self::LicenseEnforcement(_) => "LICENSE_ENFORCEMENT_FAILED",
            Self::Bind { .. } => "SERVER_BIND_FAILED",
            Self::LocalAddress(_) => "SERVER_LOCAL_ADDRESS_FAILED",
            Self::Signal(_) => "SHUTDOWN_SIGNAL_FAILED",
            Self::HttpServe(_) => "SERVER_FAILED",
            Self::ControlServe(_) => "CONTROL_SERVER_FAILED",
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Self::AbstractControlSocketNameEmpty => "CONTROL_SOCKET_ABSTRACT_NAME_EMPTY",
            #[cfg(not(any(target_os = "linux", target_os = "android")))]
            Self::AbstractControlSocketUnsupported => "CONTROL_SOCKET_ABSTRACT_UNSUPPORTED",
            Self::ServersFailed { .. } => "SERVERS_FAILED",
            Self::ControlSocketPathOccupied(_) => "CONTROL_SOCKET_PATH_OCCUPIED",
            Self::ControlSocketInUse(_) => "CONTROL_SOCKET_IN_USE",
            Self::InspectControlSocket { .. } => "CONTROL_SOCKET_INSPECT_FAILED",
            Self::RemoveStaleControlSocket { .. } => "CONTROL_SOCKET_STALE_REMOVE_FAILED",
            Self::ControlSocketBind { .. } => "CONTROL_SOCKET_BIND_FAILED",
            Self::ControlSocketPermissions { .. } => "CONTROL_SOCKET_PERMISSIONS_FAILED",
            Self::ControlSocketParentMissing(_) => "CONTROL_SOCKET_PARENT_MISSING",
            Self::CreateControlSocketDirectory { .. } => "CONTROL_SOCKET_DIRECTORY_CREATE_FAILED",
            Self::ControlSocketDirectoryPermissions { .. } => {
                "CONTROL_SOCKET_DIRECTORY_PERMISSIONS_FAILED"
            }
            Self::InspectControlSocketDirectory { .. } => "CONTROL_SOCKET_DIRECTORY_INSPECT_FAILED",
            Self::ControlSocketParentNotDirectory(_) => "CONTROL_SOCKET_PARENT_NOT_DIRECTORY",
            Self::InsecureControlSocketDirectory { .. } => "CONTROL_SOCKET_DIRECTORY_INSECURE",
            Self::OpenControlSocketDirectory { .. } => "CONTROL_SOCKET_DIRECTORY_OPEN_FAILED",
            Self::ControlSocketDirectoryChanged(_) => "CONTROL_SOCKET_DIRECTORY_CHANGED",
            Self::ControlDirectoryInUse(_) => "CONTROL_DIRECTORY_IN_USE",
            Self::ControlDirectoryLock { .. } => "CONTROL_DIRECTORY_LOCK_FAILED",
            Self::Application(error) => error.code(),
            Self::ServiceAndShutdown { .. } => "SERVER_AND_SHUTDOWN_FAILED",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPath(PathBuf);

    impl TestPath {
        fn new() -> Self {
            let directory =
                PathBuf::from("/tmp").join(format!("novasightd-server-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&directory).expect("create test socket directory");
            Self(directory.join("control.sock"))
        }

        fn with_missing_parent() -> Self {
            let directory =
                PathBuf::from("/tmp").join(format!("novasightd-server-{}", uuid::Uuid::new_v4()));
            Self(directory.join("control.sock"))
        }
    }

    impl Drop for TestPath {
        fn drop(&mut self) {
            if let Some(parent) = self.0.parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    #[tokio::test]
    async fn refuses_to_replace_a_non_socket_control_path() {
        let path = TestPath::new();
        std::fs::write(&path.0, "do not remove").expect("create occupied path");

        let error = bind_control_socket(&path.0)
            .await
            .expect_err("regular file must be preserved");

        assert!(matches!(
            error,
            DaemonRunError::ControlSocketPathOccupied(_)
        ));
        assert_eq!(
            std::fs::read_to_string(&path.0).expect("occupied file remains"),
            "do not remove"
        );
    }

    #[tokio::test]
    async fn refuses_to_unlink_a_live_daemon_socket() {
        let path = TestPath::new();
        let live = UnixListener::bind(&path.0).expect("bind live socket");

        let error = bind_control_socket(&path.0)
            .await
            .expect_err("live socket must not be replaced");

        assert!(matches!(error, DaemonRunError::ControlSocketInUse(_)));
        assert!(path.0.exists());
        drop(live);
    }

    #[tokio::test]
    async fn replaces_a_stale_socket_and_removes_only_its_own_inode() {
        let path = TestPath::new();
        let stale = UnixListener::bind(&path.0).expect("bind stale socket");
        drop(stale);

        let (listener, guard, directory_lock) = bind_control_socket(&path.0)
            .await
            .expect("replace stale socket");
        assert!(path.0.exists());

        drop(listener);
        drop(guard);
        drop(directory_lock);
        assert!(!path.0.exists());
    }

    #[tokio::test]
    async fn directory_lock_serializes_concurrent_stale_socket_recovery() {
        let path = TestPath::new();
        let stale = UnixListener::bind(&path.0).expect("bind stale socket");
        drop(stale);
        let (_listener, _guard, _directory_lock) = bind_control_socket(&path.0)
            .await
            .expect("first daemon recovers stale socket");

        let error = bind_control_socket(&path.0)
            .await
            .expect_err("second daemon must not enter stale recovery");

        assert!(matches!(error, DaemonRunError::ControlDirectoryInUse(_)));
    }

    #[tokio::test]
    async fn creates_a_missing_private_control_directory() {
        let path = TestPath::with_missing_parent();

        let (_listener, _guard, _directory_lock) = bind_control_socket(&path.0)
            .await
            .expect("create and bind control directory");
        let mode = std::fs::symlink_metadata(path.0.parent().expect("parent"))
            .expect("directory metadata")
            .mode()
            & 0o7777;

        assert_eq!(mode, 0o750);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn binds_a_linux_abstract_control_socket_without_a_filesystem_entry() {
        let name = format!("novasightd-server-{}", uuid::Uuid::new_v4());
        let configured = PathBuf::from(format!("@{name}"));

        let (listener, guard, directory_lock) = bind_control_socket(&configured)
            .await
            .expect("bind abstract control socket");
        let address = listener.local_addr().expect("abstract local address");

        assert_eq!(address.as_abstract_name(), Some(name.as_bytes()));
        assert!(guard.is_none());
        assert!(directory_lock.is_none());
        assert!(!configured.exists());
    }

    #[tokio::test]
    async fn license_watchdog_emergency_stops_a_running_pipeline_after_clear() {
        let path = TestPath::new();
        let repository = FileLicenseRepository::new(
            path.0.with_file_name("license.json"),
            LicensePolicy::new(true, None),
        );
        repository
            .activate("NOVASIGHT-TEST-MAX-ACCESS-2026")
            .expect("activate test license");
        let (supervisor, runtime) = novasight_runtime::RuntimeSupervisor::spawn_recording();
        runtime.start().await.expect("start licensed pipeline");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let watchdog = tokio::spawn(monitor_runtime_license(
            repository.clone(),
            runtime.clone(),
            true,
            shutdown_rx,
            Duration::from_millis(5),
        ));

        repository.clear().expect("clear active license");
        let mut snapshots = runtime.subscribe();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if snapshots.borrow().pipeline.state == PipelineState::Stopped {
                    break;
                }
                snapshots.changed().await.expect("runtime snapshot");
            }
        })
        .await
        .expect("license watchdog stops runtime");
        assert_eq!(
            runtime
                .snapshot()
                .subsystems
                .control
                .last_error
                .as_ref()
                .map(|error| error.code.as_str()),
            Some("emergency_stop")
        );

        shutdown_tx.send_replace(true);
        watchdog
            .await
            .expect("watchdog joins")
            .expect("watchdog exits");
        runtime.shutdown_daemon().await.expect("shutdown runtime");
        supervisor.join().await.expect("supervisor joins");
    }
}
