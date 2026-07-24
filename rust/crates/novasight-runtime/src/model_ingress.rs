use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use novasight_store::model_catalog::{ModelIngressCatalogUpdate, RuntimeModelArtifact};

const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_HELPER_BYTES: u64 = 1024 * 1024;
static NEXT_JOB_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfileConfigureRequest {
    pub color_format: String,
    pub scale: f64,
    #[serde(default)]
    pub offsets: Vec<f64>,
    #[serde(default)]
    pub mean: Vec<f64>,
    #[serde(default)]
    pub std: Vec<f64>,
    pub resize_mode: String,
    #[serde(default)]
    pub symmetric_padding: bool,
    #[serde(default)]
    pub padding_value: f64,
    pub parser_type: String,
    pub class_count: u32,
    pub labels: Vec<String>,
    pub bbox_format: String,
    pub has_objectness: bool,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f64,
    #[serde(default = "default_max_detections")]
    pub max_detections: u32,
}

impl ModelProfileConfigureRequest {
    pub fn validate(&self) -> Result<(), ModelIngressError> {
        if !matches!(
            self.color_format.trim().to_ascii_uppercase().as_str(),
            "RGB" | "BGR" | "GRAY" | "GREY"
        ) {
            return Err(ModelIngressError::InvalidRequest(
                "color_format must be RGB, BGR, or GRAY".to_owned(),
            ));
        }
        finite_positive(self.scale, "scale")?;
        channel_values(&self.offsets, "offsets", true)?;
        channel_values(&self.mean, "mean", true)?;
        channel_values(&self.std, "std", false)?;
        if !matches!(
            self.resize_mode.trim().to_ascii_lowercase().as_str(),
            "direct" | "letterbox"
        ) {
            return Err(ModelIngressError::InvalidRequest(
                "resize_mode must be direct or letterbox".to_owned(),
            ));
        }
        if !self.padding_value.is_finite() {
            return Err(ModelIngressError::InvalidRequest(
                "padding_value must be finite".to_owned(),
            ));
        }
        if self.parser_type.trim().is_empty() {
            return Err(ModelIngressError::InvalidRequest(
                "parser_type must not be blank".to_owned(),
            ));
        }
        if self.class_count == 0
            || self.labels.len() != self.class_count as usize
            || self.labels.iter().any(|label| label.trim().is_empty())
        {
            return Err(ModelIngressError::InvalidRequest(
                "labels must contain exactly class_count non-empty values".to_owned(),
            ));
        }
        if !matches!(
            self.bbox_format.trim().to_ascii_lowercase().as_str(),
            "xywh" | "xyxy"
        ) {
            return Err(ModelIngressError::InvalidRequest(
                "bbox_format must be xywh or xyxy".to_owned(),
            ));
        }
        unit_interval(self.confidence_threshold, "confidence_threshold")?;
        unit_interval(self.nms_threshold, "nms_threshold")?;
        if self.max_detections == 0 {
            return Err(ModelIngressError::InvalidRequest(
                "max_detections must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelProbeInputMode {
    Fixed,
    Latest,
}

#[derive(Clone, Debug)]
pub enum ModelIngressRequest {
    Inspect {
        artifact_id: i64,
    },
    GetProfile {
        artifact_id: i64,
    },
    Configure {
        artifact_id: i64,
        profile: Box<ModelProfileConfigureRequest>,
    },
    Probe {
        artifact_id: i64,
        input_mode: ModelProbeInputMode,
    },
}

impl ModelIngressRequest {
    pub const fn artifact_id(&self) -> i64 {
        match self {
            Self::Inspect { artifact_id }
            | Self::GetProfile { artifact_id }
            | Self::Configure { artifact_id, .. }
            | Self::Probe { artifact_id, .. } => *artifact_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelIngressResult {
    pub artifact_id: i64,
    pub profile_path: String,
    pub profile: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelIngressStage {
    Inspect,
    Configure,
    Probe,
}

pub(crate) struct ModelManifestTransaction {
    path: PathBuf,
    previous: Option<Vec<u8>>,
    committed: bool,
}

impl ModelManifestTransaction {
    pub(crate) fn begin(engine_path: &Path, max_bytes: usize) -> Result<Self, ModelIngressError> {
        let path = profile_path_for_engine(engine_path)?;
        let previous = match std::fs::read(&path) {
            Ok(bytes) if bytes.len() <= max_bytes => Some(bytes),
            Ok(_) => return Err(ModelIngressError::OutputLimitExceeded(max_bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(ModelIngressError::ReadProfile(error)),
        };
        Ok(Self {
            path,
            previous,
            committed: false,
        })
    }

    pub(crate) fn commit(mut self) {
        self.committed = true;
    }

    pub(crate) fn rollback(mut self) -> Result<(), ModelIngressError> {
        self.restore()?;
        self.committed = true;
        Ok(())
    }

    fn restore(&self) -> Result<(), ModelIngressError> {
        match &self.previous {
            Some(bytes) => {
                let sequence = NEXT_JOB_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let name = self.path.file_name().unwrap_or_default().to_string_lossy();
                let temporary = self
                    .path
                    .with_file_name(format!(".{name}.restore-{}-{sequence}", std::process::id()));
                let mut file =
                    std::fs::File::create(&temporary).map_err(ModelIngressError::ReadProfile)?;
                std::io::Write::write_all(&mut file, bytes)
                    .map_err(ModelIngressError::ReadProfile)?;
                file.sync_all().map_err(ModelIngressError::ReadProfile)?;
                std::fs::rename(&temporary, &self.path).map_err(ModelIngressError::ReadProfile)
            }
            None => match std::fs::remove_file(&self.path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(ModelIngressError::ReadProfile(error)),
            },
        }
    }
}

impl Drop for ModelManifestTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.restore();
        }
    }
}

#[derive(Clone, Debug)]
pub struct OfflineModelJobRunner {
    executable: PathBuf,
    helper_script: PathBuf,
    helper_sha256: String,
    python_code_identity: Option<(PathBuf, String)>,
    working_directory: PathBuf,
    parser_library: PathBuf,
    timeout: Duration,
    max_output_bytes: usize,
}

impl OfflineModelJobRunner {
    pub fn new(
        executable: impl Into<PathBuf>,
        helper_script: impl Into<PathBuf>,
        working_directory: impl Into<PathBuf>,
        parser_library: impl Into<PathBuf>,
        timeout: Duration,
        max_output_bytes: usize,
    ) -> Result<Self, ModelIngressError> {
        let executable = executable.into();
        let executable = executable.canonicalize().map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress executable {} cannot be resolved: {error}",
                executable.display()
            ))
        })?;
        if !executable.is_file() {
            return Err(ModelIngressError::InvalidRunner(format!(
                "model-ingress executable is not a file: {}",
                executable.display()
            )));
        }
        let helper_script = helper_script.into();
        let helper_script = helper_script.canonicalize().map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress helper {} cannot be resolved: {error}",
                helper_script.display()
            ))
        })?;
        if !helper_script.is_file() {
            return Err(ModelIngressError::InvalidRunner(format!(
                "model-ingress helper is not a file: {}",
                helper_script.display()
            )));
        }
        let helper_sha256 = sha256_file(&helper_script).map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress helper {} cannot be hashed: {error}",
                helper_script.display()
            ))
        })?;
        let working_directory = working_directory.into();
        let working_directory = working_directory.canonicalize().map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress working directory {} cannot be resolved: {error}",
                working_directory.display()
            ))
        })?;
        if !working_directory.is_dir() {
            return Err(ModelIngressError::InvalidRunner(format!(
                "model-ingress working directory is not a directory: {}",
                working_directory.display()
            )));
        }
        if timeout.is_zero() {
            return Err(ModelIngressError::InvalidRunner(
                "model-ingress timeout must be positive".to_owned(),
            ));
        }
        if max_output_bytes == 0 {
            return Err(ModelIngressError::InvalidRunner(
                "model-ingress output limit must be positive".to_owned(),
            ));
        }
        Ok(Self {
            executable,
            helper_script,
            helper_sha256,
            python_code_identity: None,
            working_directory,
            parser_library: parser_library.into(),
            timeout,
            max_output_bytes,
        })
    }

    pub fn pin_python_code(mut self, root: impl Into<PathBuf>) -> Result<Self, ModelIngressError> {
        let root = root.into().canonicalize().map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress Python code root cannot be resolved: {error}"
            ))
        })?;
        if !root.is_dir() {
            return Err(ModelIngressError::InvalidRunner(format!(
                "model-ingress Python code root is not a directory: {}",
                root.display()
            )));
        }
        let digest = sha256_python_tree(&root).map_err(|error| {
            ModelIngressError::InvalidRunner(format!(
                "model-ingress Python code root {} cannot be hashed: {error}",
                root.display()
            ))
        })?;
        self.python_code_identity = Some((root, digest));
        Ok(self)
    }

    /// Execute the packaged worker without touching a model artifact. This
    /// proves that the configured interpreter can execute the pinned helper,
    /// import its retained Python dependencies, and speak the expected
    /// bounded JSON protocol before the daemon reports readiness.
    pub async fn preflight(&self) -> Result<(), ModelIngressError> {
        let report: ModelWorkerPreflight = self
            .run("preflight", None, &[], None, Arc::new(AtomicUsize::new(0)))
            .await?;
        if report.protocol != 1
            || report.worker != "novasight.model_ingress"
            || report.operations != ["inspect", "configure", "probe"]
        {
            return Err(ModelIngressError::Protocol(
                "model-ingress preflight returned an incompatible protocol".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) async fn inspect(
        &self,
        engine_path: &Path,
        display_name: &str,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        self.run(
            "inspect",
            Some(engine_path),
            &["--display-name", display_name],
            None,
            cancellation,
        )
        .await
    }

    pub(crate) async fn configure(
        &self,
        engine_path: &Path,
        profile: &ModelProfileConfigureRequest,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        profile.validate()?;
        let request = serde_json::to_vec(profile).map_err(ModelIngressError::EncodeRequest)?;
        if request.len() > MAX_REQUEST_BYTES {
            return Err(ModelIngressError::InvalidRequest(format!(
                "model profile request exceeds {MAX_REQUEST_BYTES} bytes"
            )));
        }
        self.run(
            "configure",
            Some(engine_path),
            &[],
            Some(request),
            cancellation,
        )
        .await
    }

    pub(crate) async fn probe(
        &self,
        engine_path: &Path,
        input_mode: ModelProbeInputMode,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        if input_mode == ModelProbeInputMode::Latest {
            return Err(ModelIngressError::LatestFrameUnavailable);
        }
        let parser_library = self.parser_library.to_string_lossy().into_owned();
        self.run(
            "probe",
            Some(engine_path),
            &["--parser-library", &parser_library, "--input-mode", "fixed"],
            None,
            cancellation,
        )
        .await
    }

    async fn run<T: DeserializeOwned>(
        &self,
        operation: &'static str,
        engine_path: Option<&Path>,
        extra_args: &[&str],
        request: Option<Vec<u8>>,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<T, ModelIngressError> {
        if cancellation.load(Ordering::Acquire) != 0 {
            return Err(ModelIngressError::Cancelled);
        }
        let current_helper_sha256 = sha256_file(&self.helper_script).map_err(|error| {
            ModelIngressError::HelperIdentityChanged {
                path: self.helper_script.clone(),
                message: error.to_string(),
            }
        })?;
        if current_helper_sha256 != self.helper_sha256 {
            return Err(ModelIngressError::HelperIdentityChanged {
                path: self.helper_script.clone(),
                message: "SHA-256 changed after daemon startup".to_owned(),
            });
        }
        if let Some((root, expected)) = &self.python_code_identity {
            let actual = sha256_python_tree(root).map_err(|error| {
                ModelIngressError::HelperIdentityChanged {
                    path: root.clone(),
                    message: error.to_string(),
                }
            })?;
            if &actual != expected {
                return Err(ModelIngressError::HelperIdentityChanged {
                    path: root.clone(),
                    message: "Python code bundle changed after daemon startup".to_owned(),
                });
            }
        }
        let job_directory = JobDirectory::create(&self.working_directory)?;
        let mut command = Command::new(&self.executable);
        command.arg(&self.helper_script).arg(operation);
        if let Some(engine_path) = engine_path {
            command.arg("--engine").arg(engine_path);
        }
        command
            .args(extra_args)
            .current_dir(job_directory.path())
            .stdin(if request.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn().map_err(|source| ModelIngressError::Spawn {
            executable: self.executable.clone(),
            source,
        })?;
        let child_pid = child.id();
        let deadline = tokio::time::Instant::now() + self.timeout;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ModelIngressError::Protocol("worker stdout was not piped".to_owned()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| ModelIngressError::Protocol("worker stderr was not piped".to_owned()))?;
        let stdout_task = tokio::spawn(read_limited(stdout, self.max_output_bytes));
        let stderr_task = tokio::spawn(read_limited(stderr, self.max_output_bytes));
        if let Some(request) = request {
            let mut stdin = child.stdin.take().ok_or_else(|| {
                ModelIngressError::Protocol("worker stdin was not piped".to_owned())
            })?;
            let write_request = async {
                stdin
                    .write_all(&request)
                    .await
                    .map_err(ModelIngressError::WriteRequest)?;
                stdin
                    .shutdown()
                    .await
                    .map_err(ModelIngressError::WriteRequest)
            };
            let interrupted = tokio::select! {
                result = write_request => {
                    result?;
                    None
                }
                () = wait_for_cancellation(Arc::clone(&cancellation)) => {
                    Some(ModelIngressError::Cancelled)
                }
                () = tokio::time::sleep_until(deadline) => {
                    Some(ModelIngressError::TimedOut(self.timeout))
                }
            };
            if let Some(error) = interrupted {
                terminate_worker(child_pid, &mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                return Err(error);
            }
        }
        let outcome = tokio::select! {
            status = child.wait() => JobOutcome::Exited(status.map_err(ModelIngressError::Wait)?),
            () = wait_for_cancellation(cancellation) => JobOutcome::Cancelled,
            () = tokio::time::sleep_until(deadline) => JobOutcome::TimedOut,
        };
        if !matches!(outcome, JobOutcome::Exited(_)) {
            terminate_worker(child_pid, &mut child).await;
            stdout_task.abort();
            stderr_task.abort();
            return match outcome {
                JobOutcome::Cancelled => Err(ModelIngressError::Cancelled),
                JobOutcome::TimedOut => Err(ModelIngressError::TimedOut(self.timeout)),
                JobOutcome::Exited(_) => unreachable!(),
            };
        }
        kill_process_group(child_pid);
        let stdout = join_output(stdout_task, "stdout").await?;
        let stderr = join_output(stderr_task, "stderr").await?;
        match outcome {
            JobOutcome::Exited(status) if !status.success() => {
                return Err(ModelIngressError::WorkerFailed {
                    status: status.code(),
                    message: worker_error_message(&stderr),
                });
            }
            JobOutcome::Exited(_) => {}
            JobOutcome::Cancelled | JobOutcome::TimedOut => unreachable!(),
        }
        serde_json::from_slice(&stdout).map_err(ModelIngressError::DecodeResponse)
    }
}

#[derive(Debug, Deserialize)]
struct ModelWorkerPreflight {
    protocol: u32,
    worker: String,
    operations: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelWorkerOutput {
    pub profile_path: PathBuf,
    pub profile: Value,
    #[serde(default)]
    pub report: Option<Value>,
}

pub(crate) fn validate_worker_output(
    artifact: &RuntimeModelArtifact,
    display_root: &Path,
    stage: ModelIngressStage,
    output: ModelWorkerOutput,
) -> Result<(ModelIngressResult, ModelIngressCatalogUpdate), ModelIngressError> {
    let expected_profile_path = profile_path_for_engine(&artifact.artifact_path)?;
    let actual_profile_path = canonical_file(&output.profile_path, "profile_path")?;
    let expected_profile_path = canonical_file(&expected_profile_path, "expected profile path")?;
    if actual_profile_path != expected_profile_path {
        return Err(ModelIngressError::Protocol(format!(
            "worker wrote profile {} instead of {}",
            actual_profile_path.display(),
            expected_profile_path.display()
        )));
    }
    let profile = &output.profile;
    let engine = object_field(profile, "engine")?;
    let worker_engine_path = string_field(engine, "path")?;
    let worker_engine_path = canonical_file(Path::new(worker_engine_path), "profile engine path")?;
    let expected_engine_path = canonical_file(&artifact.artifact_path, "artifact engine path")?;
    if worker_engine_path != expected_engine_path {
        return Err(ModelIngressError::Protocol(format!(
            "worker inspected engine {} instead of {}",
            worker_engine_path.display(),
            expected_engine_path.display()
        )));
    }
    let actual_size = expected_engine_path
        .metadata()
        .map_err(ModelIngressError::ReadProfile)?
        .len();
    let worker_size = u64_field(engine, "file_size")?;
    if worker_size != actual_size {
        return Err(ModelIngressError::Protocol(format!(
            "worker engine size {worker_size} does not match artifact size {actual_size}"
        )));
    }
    let checksum = string_field(engine, "sha256")?.to_owned();
    let profile_object = profile.as_object().ok_or_else(|| {
        ModelIngressError::Protocol("worker profile must be an object".to_owned())
    })?;
    let profile_status = string_field(profile_object, "status")?;
    let input_shape = runtime_shape(profile)?;
    let labels = string_array(profile.get("labels"), "labels")?;
    let status = match stage {
        ModelIngressStage::Inspect => match profile_status {
            "NEEDS_CONFIGURATION" => "pending",
            "INCOMPATIBLE" => "failed",
            other => {
                return Err(ModelIngressError::Protocol(format!(
                    "inspect returned unexpected profile status {other}"
                )));
            }
        },
        ModelIngressStage::Configure => {
            if profile_status != "READY_FOR_PROBE" {
                return Err(ModelIngressError::Protocol(format!(
                    "configure returned unexpected profile status {profile_status}"
                )));
            }
            "pending"
        }
        ModelIngressStage::Probe => {
            if profile_status == "VALIDATED" {
                validate_probe_contract(profile, &output.report)?;
                "ready"
            } else if profile_status == "INVALID" {
                "failed"
            } else {
                return Err(ModelIngressError::Protocol(format!(
                    "probe returned unexpected profile status {profile_status}"
                )));
            }
        }
    };
    let classes = match stage {
        ModelIngressStage::Inspect => None,
        ModelIngressStage::Configure | ModelIngressStage::Probe => Some(labels),
    };
    let profile_path = actual_profile_path
        .strip_prefix(display_root)
        .map(Path::to_string_lossy)
        .map(|value| value.replace('\\', "/"))
        .unwrap_or_else(|_| actual_profile_path.to_string_lossy().into_owned());
    let result = ModelIngressResult {
        artifact_id: artifact.artifact.id,
        profile_path,
        profile: output.profile,
        report: output.report,
    };
    let update = ModelIngressCatalogUpdate {
        artifact_id: artifact.artifact.id,
        status: status.to_owned(),
        checksum,
        classes,
        input_shape,
    };
    Ok((result, update))
}

pub(crate) fn load_profile(
    artifact: &RuntimeModelArtifact,
    display_root: &Path,
    max_bytes: usize,
) -> Result<ModelIngressResult, ModelIngressError> {
    let path = profile_path_for_engine(&artifact.artifact_path)?;
    let mut file = std::fs::File::open(&path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            ModelIngressError::ProfileNotFound(artifact.artifact.id)
        } else {
            ModelIngressError::ReadProfile(source)
        }
    })?;
    let mut bytes = Vec::new();
    let mut limited = std::io::Read::take(&mut file, max_bytes.saturating_add(1) as u64);
    std::io::Read::read_to_end(&mut limited, &mut bytes).map_err(ModelIngressError::ReadProfile)?;
    if bytes.len() > max_bytes {
        return Err(ModelIngressError::OutputLimitExceeded(max_bytes));
    }
    let root: Value = serde_json::from_slice(&bytes).map_err(ModelIngressError::DecodeResponse)?;
    let profile = root
        .get("model_profile")
        .cloned()
        .ok_or_else(|| ModelIngressError::Protocol("manifest has no model_profile".to_owned()))?;
    let output = ModelWorkerOutput {
        profile_path: path,
        profile,
        report: None,
    };
    validate_existing_profile(artifact, display_root, output)
}

fn validate_existing_profile(
    artifact: &RuntimeModelArtifact,
    display_root: &Path,
    mut output: ModelWorkerOutput,
) -> Result<ModelIngressResult, ModelIngressError> {
    let expected_profile_path = profile_path_for_engine(&artifact.artifact_path)?;
    let actual_profile_path = canonical_file(&output.profile_path, "profile_path")?;
    let expected_profile_path = canonical_file(&expected_profile_path, "expected profile path")?;
    if actual_profile_path != expected_profile_path {
        return Err(ModelIngressError::Protocol(
            "profile path identity mismatch".to_owned(),
        ));
    }
    let engine = object_field(&output.profile, "engine")?;
    let worker_engine_path = canonical_file(
        Path::new(string_field(engine, "path")?),
        "profile engine path",
    )?;
    let expected_engine_path = canonical_file(&artifact.artifact_path, "artifact engine path")?;
    if worker_engine_path != expected_engine_path {
        return Err(ModelIngressError::Protocol(
            "profile engine identity mismatch".to_owned(),
        ));
    }
    let checksum = string_field(engine, "sha256")?.to_owned();
    let recorded_size = u64_field(engine, "file_size")?;
    let actual_size = std::fs::metadata(&artifact.artifact_path)
        .map_err(ModelIngressError::ReadProfile)?
        .len();
    let actual_checksum = format!(
        "sha256:{}",
        sha256_unbounded(&artifact.artifact_path).map_err(ModelIngressError::ReadProfile)?
    );
    if recorded_size != actual_size || checksum != actual_checksum {
        output.profile["status"] = Value::String("UNINSPECTED".to_owned());
        let validation = output
            .profile
            .get_mut("validation")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                ModelIngressError::Protocol("profile.validation must be an object".to_owned())
            })?;
        validation.insert("status".to_owned(), Value::String("not_run".to_owned()));
        validation.insert("validated_at".to_owned(), Value::String(String::new()));
        for field in [
            "engine_execution_ok",
            "decoder_ok",
            "nms_ok",
            "detection_batch_ok",
        ] {
            validation.insert(field.to_owned(), Value::Bool(false));
        }
        validation.insert(
            "profile_fingerprint".to_owned(),
            Value::String(String::new()),
        );
        validation.insert(
            "issues".to_owned(),
            Value::Array(vec![Value::String("ENGINE_CONTENT_CHANGED".to_owned())]),
        );
    }
    let profile_path = actual_profile_path
        .strip_prefix(display_root)
        .map(Path::to_string_lossy)
        .map(|value| value.replace('\\', "/"))
        .unwrap_or_else(|_| actual_profile_path.to_string_lossy().into_owned());
    Ok(ModelIngressResult {
        artifact_id: artifact.artifact.id,
        profile_path,
        profile: output.profile,
        report: output.report,
    })
}

fn profile_path_for_engine(engine_path: &Path) -> Result<PathBuf, ModelIngressError> {
    let name = engine_path.file_name().ok_or_else(|| {
        ModelIngressError::Protocol(format!(
            "engine path has no filename: {}",
            engine_path.display()
        ))
    })?;
    let mut profile_name = name.to_os_string();
    profile_name.push(".manifest.json");
    Ok(engine_path.with_file_name(profile_name))
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf, ModelIngressError> {
    path.canonicalize().map_err(|error| {
        ModelIngressError::Protocol(format!(
            "{label} {} cannot be resolved: {error}",
            path.display()
        ))
    })
}

fn object_field<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a serde_json::Map<String, Value>, ModelIngressError> {
    value
        .get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| ModelIngressError::Protocol(format!("profile.{name} must be an object")))
}

fn string_field<'a>(
    value: &'a serde_json::Map<String, Value>,
    name: &str,
) -> Result<&'a str, ModelIngressError> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ModelIngressError::Protocol(format!("profile field {name} is missing")))
}

fn u64_field(value: &serde_json::Map<String, Value>, name: &str) -> Result<u64, ModelIngressError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelIngressError::Protocol(format!("profile field {name} is invalid")))
}

fn runtime_shape(profile: &Value) -> Result<Option<String>, ModelIngressError> {
    let input = object_field(profile, "input")?;
    let Some(shape) = input.get("runtime_shape").and_then(Value::as_array) else {
        return Err(ModelIngressError::Protocol(
            "profile input.runtime_shape is missing".to_owned(),
        ));
    };
    if shape.is_empty() {
        return Ok(None);
    }
    let values = shape
        .iter()
        .map(|value| value.as_i64().filter(|value| *value > 0))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            ModelIngressError::Protocol(
                "profile input.runtime_shape must contain positive integers".to_owned(),
            )
        })?;
    Ok(Some(
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x"),
    ))
}

fn string_array(value: Option<&Value>, name: &str) -> Result<Vec<String>, ModelIngressError> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        ModelIngressError::Protocol(format!("profile field {name} must be an array"))
    })?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    ModelIngressError::Protocol(format!(
                        "profile field {name} contains a blank value"
                    ))
                })
        })
        .collect()
}

fn validate_probe_contract(
    profile: &Value,
    report: &Option<Value>,
) -> Result<(), ModelIngressError> {
    let validation = object_field(profile, "validation")?;
    for field in [
        "engine_execution_ok",
        "decoder_ok",
        "nms_ok",
        "detection_batch_ok",
    ] {
        if validation.get(field).and_then(Value::as_bool) != Some(true) {
            return Err(ModelIngressError::Protocol(format!(
                "validated profile has false validation.{field}"
            )));
        }
    }
    let report = report.as_ref().and_then(Value::as_object).ok_or_else(|| {
        ModelIngressError::Protocol("validated probe response has no report".to_owned())
    })?;
    if report.get("status").and_then(Value::as_str) != Some("validated") {
        return Err(ModelIngressError::Protocol(
            "validated profile has a non-validated report".to_owned(),
        ));
    }
    Ok(())
}

enum JobOutcome {
    Exited(std::process::ExitStatus),
    Cancelled,
    TimedOut,
}

struct JobDirectory(PathBuf);

impl JobDirectory {
    fn create(root: &Path) -> Result<Self, ModelIngressError> {
        let sequence = NEXT_JOB_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".novasight-model-job-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).map_err(|error| {
            ModelIngressError::Failed(format!(
                "failed to create isolated model job directory {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for JobDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn terminate_worker(pid: Option<u32>, child: &mut tokio::process::Child) {
    kill_process_group(pid);
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
}

fn kill_process_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid.and_then(|pid| i32::try_from(pid).ok()) {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(pid),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    #[cfg(not(unix))]
    let _ = pid;
}

async fn join_output(
    mut task: tokio::task::JoinHandle<Result<Vec<u8>, ModelIngressError>>,
    stream: &'static str,
) -> Result<Vec<u8>, ModelIngressError> {
    tokio::select! {
        result = &mut task => result.map_err(|error| {
            ModelIngressError::Protocol(format!("{stream} task failed: {error}"))
        })?,
        () = tokio::time::sleep(Duration::from_secs(1)) => {
            task.abort();
            Err(ModelIngressError::Protocol(format!(
                "{stream} pipe remained open after worker exit"
            )))
        }
    }
}

async fn wait_for_cancellation(cancellation: Arc<AtomicUsize>) {
    let mut interval = tokio::time::interval(Duration::from_millis(10));
    loop {
        interval.tick().await;
        if cancellation.load(Ordering::Acquire) != 0 {
            return;
        }
    }
}

async fn read_limited(
    reader: impl AsyncRead + Unpin,
    limit: usize,
) -> Result<Vec<u8>, ModelIngressError> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .await
        .map_err(ModelIngressError::ReadOutput)?;
    if bytes.len() > limit {
        return Err(ModelIngressError::OutputLimitExceeded(limit));
    }
    Ok(bytes)
}

fn worker_error_message(stderr: &[u8]) -> String {
    serde_json::from_slice::<Value>(stderr)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(stderr).trim().to_owned())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size > MAX_HELPER_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("helper exceeds {MAX_HELPER_BYTES} bytes"),
        ));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_unbounded(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_python_tree(root: &Path) -> Result<String, std::io::Error> {
    let mut pending = vec![root.to_owned()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Python code bundle contains symlink: {}",
                        entry.path().display()
                    ),
                ));
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file()
                && entry.path().extension().is_some_and(|value| value == "py")
            {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Python code bundle contains no .py files",
        ));
    }
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root).map_err(std::io::Error::other)?;
        digest.update(relative.as_os_str().as_encoded_bytes());
        digest.update([0]);
        let bytes = std::fs::read(&path)?;
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn finite_positive(value: f64, name: &str) -> Result<(), ModelIngressError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ModelIngressError::InvalidRequest(format!(
            "{name} must be finite and positive"
        )))
    }
}

fn unit_interval(value: f64, name: &str) -> Result<(), ModelIngressError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ModelIngressError::InvalidRequest(format!(
            "{name} must be finite and in [0, 1]"
        )))
    }
}

fn channel_values(values: &[f64], name: &str, allow_zero: bool) -> Result<(), ModelIngressError> {
    if !matches!(values.len(), 0 | 1 | 3)
        || values.iter().any(|value| !value.is_finite())
        || (!allow_zero && values.contains(&0.0))
    {
        return Err(ModelIngressError::InvalidRequest(format!(
            "{name} must contain zero, one, or three finite{} values",
            if allow_zero { "" } else { " non-zero" }
        )));
    }
    Ok(())
}

const fn default_confidence_threshold() -> f64 {
    0.25
}

const fn default_nms_threshold() -> f64 {
    0.45
}

const fn default_max_detections() -> u32 {
    256
}

#[derive(Debug, Error)]
pub enum ModelIngressError {
    #[error("invalid model-ingress runner: {0}")]
    InvalidRunner(String),
    #[error("model-ingress helper identity changed at {}: {message}", path.display())]
    HelperIdentityChanged { path: PathBuf, message: String },
    #[error("invalid model profile request: {0}")]
    InvalidRequest(String),
    #[error("model-ingress worker is not configured")]
    Unavailable,
    #[error("model profile for artifact {0} has not been inspected")]
    ProfileNotFound(i64),
    #[error("latest-frame model diagnostics require the Rust LatestFrame phase")]
    LatestFrameUnavailable,
    #[error("model-ingress operation was cancelled by stop or emergency-stop")]
    Cancelled,
    #[error("model-ingress operation exceeded {0:?}")]
    TimedOut(Duration),
    #[error("failed to spawn model-ingress executable {}: {source}", executable.display())]
    Spawn {
        executable: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write model-ingress request: {0}")]
    WriteRequest(#[source] std::io::Error),
    #[error("failed to encode model-ingress request: {0}")]
    EncodeRequest(#[source] serde_json::Error),
    #[error("failed to wait for model-ingress worker: {0}")]
    Wait(#[source] std::io::Error),
    #[error("failed to read model-ingress output: {0}")]
    ReadOutput(#[source] std::io::Error),
    #[error("failed to read model profile: {0}")]
    ReadProfile(#[source] std::io::Error),
    #[error("model-ingress worker output exceeded {0} bytes")]
    OutputLimitExceeded(usize),
    #[error("model-ingress worker failed with status {status:?}: {message}")]
    WorkerFailed {
        status: Option<i32>,
        message: String,
    },
    #[error("model-ingress worker returned invalid JSON: {0}")]
    DecodeResponse(#[source] serde_json::Error),
    #[error("model-ingress protocol violation: {0}")]
    Protocol(String),
    #[error(transparent)]
    Catalog(#[from] novasight_store::model_catalog::ModelCatalogError),
    #[error(transparent)]
    Runtime(#[from] crate::RuntimeError),
    #[error("model-ingress operation failed: {0}")]
    Failed(String),
}
