use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use novasight_store::model_catalog::{ModelIngressCatalogUpdate, RuntimeModelArtifact};

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
    Admit {
        artifact_id: i64,
        parser_preset: String,
        catalog_labels: Vec<String>,
    },
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
            Self::Admit { artifact_id, .. }
            | Self::Inspect { artifact_id }
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

/// Rust-native model inspection, configuration, and probe backend.
#[derive(Clone, Debug)]
pub struct ModelJobRunner {
    #[cfg(feature = "tensorrt-model-ingress")]
    native: crate::model_ingress_native::NativeModelJobRunner,
}

#[cfg(feature = "tensorrt-model-ingress")]
impl From<crate::model_ingress_native::NativeModelJobRunner> for ModelJobRunner {
    fn from(value: crate::model_ingress_native::NativeModelJobRunner) -> Self {
        Self { native: value }
    }
}

impl ModelJobRunner {
    pub async fn preflight(&self) -> Result<(), ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self.native.preflight().await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        Err(ModelIngressError::Unavailable)
    }

    pub(crate) async fn validation_receipt_is_current(
        &self,
        engine_path: &Path,
    ) -> Result<bool, ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self.native.validation_receipt_is_current(engine_path).await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        {
            let _ = engine_path;
            Err(ModelIngressError::Unavailable)
        }
    }

    pub(crate) async fn inspect(
        &self,
        engine_path: &Path,
        display_name: &str,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self
            .native
            .inspect(engine_path, display_name, cancellation)
            .await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        {
            let _ = (engine_path, display_name, cancellation);
            Err(ModelIngressError::Unavailable)
        }
    }

    pub(crate) async fn admit(
        &self,
        engine_path: &Path,
        display_name: &str,
        parser_preset: &str,
        catalog_labels: &[String],
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self
            .native
            .admit(
                engine_path,
                display_name,
                parser_preset,
                catalog_labels,
                cancellation,
            )
            .await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        {
            let _ = (
                engine_path,
                display_name,
                parser_preset,
                catalog_labels,
                cancellation,
            );
            Err(ModelIngressError::Unavailable)
        }
    }

    pub(crate) async fn configure(
        &self,
        engine_path: &Path,
        profile: &ModelProfileConfigureRequest,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self
            .native
            .configure(engine_path, profile, cancellation)
            .await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        {
            let _ = (engine_path, profile, cancellation);
            Err(ModelIngressError::Unavailable)
        }
    }

    pub(crate) async fn probe(
        &self,
        engine_path: &Path,
        input_mode: ModelProbeInputMode,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        #[cfg(feature = "tensorrt-model-ingress")]
        return self
            .native
            .probe(engine_path, input_mode, cancellation)
            .await;
        #[cfg(not(feature = "tensorrt-model-ingress"))]
        {
            let _ = (engine_path, input_mode, cancellation);
            Err(ModelIngressError::Unavailable)
        }
    }
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
    mut output: ModelWorkerOutput,
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
    normalize_lossless_json_boundary(&mut output.profile);
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

/// Build the minimum complete YOLO profile needed by the native TensorRT
/// ingress path when a referenced Engine has no sidecar yet. Tensor identity
/// and dimensions come from the Engine inspection result; class labels are
/// descriptive only and are generated from the real output channel count when
/// the catalog has no matching labels.
#[cfg(test)]
fn automatic_activation_profile(
    inspected: &ModelIngressResult,
    parser_preset: &str,
    catalog_labels: &[String],
) -> Result<ModelProfileConfigureRequest, ModelIngressError> {
    automatic_activation_profile_from_value(&inspected.profile, parser_preset, catalog_labels)
}

#[cfg(any(test, feature = "tensorrt-model-ingress"))]
pub(crate) fn automatic_activation_profile_from_value(
    profile: &Value,
    parser_preset: &str,
    catalog_labels: &[String],
) -> Result<ModelProfileConfigureRequest, ModelIngressError> {
    let outputs = profile
        .get("outputs")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ModelIngressError::Protocol("profile.outputs must be an array".to_owned())
        })?;
    if outputs.len() != 1 {
        return Err(ModelIngressError::InvalidRequest(format!(
            "automatic YOLO activation requires one output tensor, Engine reports {}",
            outputs.len()
        )));
    }
    let output = &outputs[0];
    let shape = output
        .get("engine_shape")
        .and_then(Value::as_array)
        .filter(|shape| !shape.is_empty())
        .or_else(|| output.get("shape").and_then(Value::as_array))
        .ok_or_else(|| {
            ModelIngressError::Protocol("profile.outputs[0] has no inspected shape".to_owned())
        })?
        .iter()
        .map(|value| value.as_u64().filter(|dimension| *dimension > 0))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            ModelIngressError::Protocol(
                "profile.outputs[0].engine_shape must contain positive integers".to_owned(),
            )
        })?;
    let semantic_shape = if shape.first() == Some(&1) {
        &shape[1..]
    } else {
        shape.as_slice()
    };
    if semantic_shape.len() != 2 {
        return Err(ModelIngressError::InvalidRequest(format!(
            "automatic raw YOLO activation requires [C,N], [N,C], or [1,C,N], got {shape:?}"
        )));
    }
    let channels = semantic_shape.iter().copied().min().unwrap_or(0);
    let normalized = parser_preset.trim().to_ascii_lowercase().replace('-', "_");
    let catalog_count = u64::try_from(catalog_labels.len()).unwrap_or(u64::MAX);
    let (parser_type, has_objectness) = match normalized.as_str() {
        "yolov5" | "yolo_v5" | "yolov5_raw" => ("yolov5_raw", true),
        "yolo11" | "yolo_11" | "yolo11_raw" => ("yolo11_raw", false),
        "yolov8" | "yolo_v8" | "yolov8_raw" => ("yolov8_raw", false),
        "" | "auto" | "automatic" | "novasight_generic" | "generic" | "custom" => {
            if !catalog_labels.is_empty() && channels == catalog_count.saturating_add(5) {
                ("yolov5_raw", true)
            } else {
                ("yolov8_raw", false)
            }
        }
        _ => {
            return Err(ModelIngressError::InvalidRequest(format!(
                "unsupported parser preset {parser_preset}"
            )));
        }
    };
    let base_channels = if has_objectness { 5 } else { 4 };
    let class_count = channels.checked_sub(base_channels).ok_or_else(|| {
        ModelIngressError::InvalidRequest(format!(
            "output channel count {channels} is too small for {parser_type}"
        ))
    })?;
    let class_count = u32::try_from(class_count)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            ModelIngressError::InvalidRequest(
                "automatic YOLO class count is outside the supported range".to_owned(),
            )
        })?;
    let labels = if usize::try_from(class_count).ok() == Some(catalog_labels.len())
        && catalog_labels.iter().all(|label| !label.trim().is_empty())
    {
        catalog_labels.to_vec()
    } else {
        (0..class_count)
            .map(|class_id| format!("class_{class_id}"))
            .collect()
    };
    Ok(ModelProfileConfigureRequest {
        color_format: "RGB".to_owned(),
        scale: 1.0 / 255.0,
        offsets: Vec::new(),
        mean: Vec::new(),
        std: Vec::new(),
        resize_mode: "direct".to_owned(),
        symmetric_padding: false,
        padding_value: 0.0,
        parser_type: parser_type.to_owned(),
        class_count,
        labels,
        bbox_format: "xywh".to_owned(),
        has_objectness,
        confidence_threshold: default_confidence_threshold(),
        nms_threshold: default_nms_threshold(),
        max_detections: default_max_detections(),
    })
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
    normalize_lossless_json_boundary(&mut output.profile);
    Ok(ModelIngressResult {
        artifact_id: artifact.artifact.id,
        profile_path,
        profile: output.profile,
        report: output.report,
    })
}

fn normalize_lossless_json_boundary(profile: &mut Value) {
    let Some(modified_at_ns) = profile
        .get_mut("engine")
        .and_then(Value::as_object_mut)
        .and_then(|engine| engine.get_mut("modified_at_ns"))
    else {
        return;
    };
    if modified_at_ns.is_number() {
        *modified_at_ns = Value::String(modified_at_ns.to_string());
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn inspected(shape: &[u64]) -> ModelIngressResult {
        ModelIngressResult {
            artifact_id: 1,
            profile_path: "detector.engine.manifest.json".to_owned(),
            profile: serde_json::json!({
                "outputs": [{"engine_shape": shape}],
            }),
            report: None,
        }
    }

    fn inspected_dynamic(shape: &[u64]) -> ModelIngressResult {
        ModelIngressResult {
            artifact_id: 1,
            profile_path: "detector.engine.manifest.json".to_owned(),
            profile: serde_json::json!({
                "outputs": [{"shape": shape, "engine_shape": []}],
            }),
            report: None,
        }
    }

    #[test]
    fn automatic_profile_derives_yolov8_classes_from_engine_channels() {
        let profile = automatic_activation_profile(
            &inspected(&[1, 7, 1_344]),
            "auto",
            &["target".to_owned()],
        )
        .expect("three-class YOLOv8 profile");

        assert_eq!(profile.parser_type, "yolov8_raw");
        assert!(!profile.has_objectness);
        assert_eq!(profile.class_count, 3);
        assert_eq!(profile.labels, ["class_0", "class_1", "class_2"]);
    }

    #[test]
    fn automatic_profile_uses_catalog_labels_to_identify_objectness() {
        let labels = (0..80).map(|id| format!("label_{id}")).collect::<Vec<_>>();
        let profile = automatic_activation_profile(&inspected(&[1, 8_400, 85]), "auto", &labels)
            .expect("eighty-class YOLOv5 profile");

        assert_eq!(profile.parser_type, "yolov5_raw");
        assert!(profile.has_objectness);
        assert_eq!(profile.class_count, 80);
        assert_eq!(profile.labels, labels);
    }

    #[test]
    fn explicit_parser_preset_controls_objectness_without_a_manifest() {
        let profile = automatic_activation_profile(&inspected(&[1, 8, 1_344]), "yolov5", &[])
            .expect("three-class YOLOv5 profile");

        assert_eq!(profile.class_count, 3);
        assert!(profile.has_objectness);
    }

    #[test]
    fn automatic_profile_uses_selected_runtime_shape_for_dynamic_engines() {
        let profile = automatic_activation_profile(&inspected_dynamic(&[1, 7, 1_344]), "auto", &[])
            .expect("dynamic YOLOv8 profile");

        assert_eq!(profile.class_count, 3);
        assert_eq!(profile.parser_type, "yolov8_raw");
    }
}

#[derive(Debug, Error)]
pub enum ModelIngressError {
    #[error("invalid model profile request: {0}")]
    InvalidRequest(String),
    #[error("native TensorRT model diagnostics are not available in this build")]
    Unavailable,
    #[error("model profile for artifact {0} has not been inspected")]
    ProfileNotFound(i64),
    #[error("latest-frame model diagnostics require the Rust LatestFrame phase")]
    LatestFrameUnavailable,
    #[error("model-ingress operation was cancelled by stop or emergency-stop")]
    Cancelled,
    #[error("failed to read model profile: {0}")]
    ReadProfile(#[source] std::io::Error),
    #[error("model manifest exceeded {0} bytes")]
    OutputLimitExceeded(usize),
    #[error("model manifest contains invalid JSON: {0}")]
    DecodeResponse(#[source] serde_json::Error),
    #[error("model manifest contract violation: {0}")]
    Protocol(String),
    #[error(transparent)]
    Catalog(#[from] novasight_store::model_catalog::ModelCatalogError),
    #[error(transparent)]
    Runtime(#[from] crate::RuntimeError),
    #[error("model-ingress operation failed: {0}")]
    Failed(String),
}
