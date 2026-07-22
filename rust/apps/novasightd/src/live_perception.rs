use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use novasight_core::{Clock, PointerDevice, RecordingPointerDevice};
use novasight_pipeline::PipelineConfig;
use novasight_platform_jetson::SystemMonotonicClock;
use novasight_platform_jetson::deepstream::{
    CaptureFormat, CaptureProfile, DeepStreamAdapter, DeepStreamPipelineSpec,
    DeepStreamSessionConfig, ModelInput, Roi,
};
use novasight_platform_jetson::kmnet::{KmNetError, KmNetHostClient, KmNetHostConfig};
use novasight_platform_jetson::kmnet_native::{
    KmNetNativeConfig, KmNetNativeDevice, KmNetNativeError,
};
use novasight_runtime::RuntimeDependencies;
use novasight_store::config::{AppConfig, CapturePreference, ConfigValidationError, DeviceBackend};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub(super) fn build_live_recording_dependencies(
    config: &AppConfig,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    build_live_dependencies(config, device, None)
}

pub(super) fn build_live_production_dependencies(
    config: &AppConfig,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let adapters = config
        .require_production_adapters()
        .map_err(LivePerceptionError::Config)?;
    if !adapters.device.auto_connect {
        return Err(LivePerceptionError::DeviceAutoConnectDisabled);
    }
    let device: Arc<dyn PointerDevice> = match adapters.device.backend {
        DeviceBackend::NativeUdp => {
            let host =
                adapters.device.host.parse().map_err(|_| {
                    LivePerceptionError::InvalidKmNetHost(adapters.device.host.clone())
                })?;
            Arc::new(
                KmNetNativeDevice::connect(KmNetNativeConfig {
                    host,
                    port: adapters.device.port,
                    uuid: adapters.device.uuid.clone(),
                    monitor_port: adapters.device.monitor_port,
                    connect_timeout: Duration::from_millis(adapters.device.connect_timeout_ms),
                    request_timeout: Duration::from_millis(adapters.device.send_timeout_ms),
                    monitor_timeout: Duration::from_millis(adapters.device.monitor_timeout_ms),
                })
                .map_err(LivePerceptionError::NativeKmNet)?,
            )
        }
        DeviceBackend::PythonHost => Arc::new(
            KmNetHostClient::connect(KmNetHostConfig {
                program: config.paths.python_executable.clone(),
                args: vec![
                    OsString::from("-u"),
                    OsString::from("-m"),
                    OsString::from(adapters.device.helper_module.trim()),
                ],
                host: adapters.device.host.clone(),
                port: adapters.device.port,
                uuid: adapters.device.uuid.clone(),
                monitor_port: adapters.device.monitor_port,
                startup_timeout: Duration::from_millis(adapters.device.connect_timeout_ms),
                request_timeout: Duration::from_millis(adapters.device.send_timeout_ms),
                reconnect_cooldown: Duration::from_millis(adapters.device.reconnect_cooldown_ms),
            })
            .map_err(LivePerceptionError::KmNet)?,
        ),
    };
    build_live_dependencies(
        config,
        device,
        Some(adapters.device.trigger_poll_interval_ms),
    )
}

fn build_live_dependencies(
    config: &AppConfig,
    device: Arc<dyn PointerDevice>,
    trigger_poll_interval_ms: Option<u64>,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let adapters = config
        .require_production_adapters()
        .map_err(LivePerceptionError::Config)?;
    if !adapters.inference.enabled {
        return Err(LivePerceptionError::InferenceDisabled);
    }
    if adapters.capture.preference != CapturePreference::Manual {
        return Err(LivePerceptionError::AutomaticCaptureUnsupported);
    }
    let format = parse_capture_format(&adapters.capture.pixel_format)?;
    let io_mode = u32::try_from(adapters.inference.deepstream_io_mode)
        .map_err(|_| LivePerceptionError::InvalidIoMode(adapters.inference.deepstream_io_mode))?;
    let nvinfer_config = resolve_data_artifact(
        &config.paths.data_dir,
        &adapters.inference.deepstream_nvinfer_config,
    )?;
    if !nvinfer_config.is_file() {
        return Err(LivePerceptionError::NvinferConfigMissing(nvinfer_config));
    }
    let parser_library = resolve_process_path(&adapters.inference.deepstream_parser_library)?;
    validate_nvinfer_contract(
        &nvinfer_config,
        &parser_library,
        adapters.inference.confidence_threshold,
        adapters.inference.nms_threshold,
        adapters.inference.model_width,
        adapters.inference.model_height,
        adapters.inference.deepstream_component_id,
    )?;

    let pipeline = DeepStreamPipelineSpec {
        device: adapters.capture.device.clone(),
        capture: CaptureProfile {
            width: adapters.capture.width,
            height: adapters.capture.height,
            fps: adapters.capture.fps,
            format,
        },
        io_mode,
        roi: Roi {
            left: adapters.capture.roi_left,
            top: adapters.capture.roi_top,
            width: adapters.capture.roi_width,
            height: adapters.capture.roi_height,
        },
        model_input: ModelInput {
            width: adapters.inference.model_width,
            height: adapters.inference.model_height,
        },
        nvinfer_config,
        batched_push_timeout_us: adapters.inference.deepstream_batched_push_timeout_us,
        inference_element: adapters.inference.deepstream_probe_element.clone(),
    };
    pipeline
        .build()
        .map_err(|error| LivePerceptionError::Pipeline(error.to_string()))?;

    let session = DeepStreamSessionConfig {
        pipeline,
        probe_pad: adapters.inference.deepstream_probe_pad.clone(),
        source_id: adapters.inference.deepstream_source_id,
        inference_component_id: adapters.inference.deepstream_component_id,
        max_batch_age_ns: deadline_ns(adapters.inference.inference_input_deadline_ms),
        startup_timeout: Duration::from_millis(adapters.inference.deepstream_startup_timeout_ms),
        shutdown_timeout: Duration::from_millis(adapters.inference.deepstream_shutdown_timeout_ms),
    };
    let clock: Arc<dyn Clock> = Arc::new(SystemMonotonicClock::default());
    Ok(RuntimeDependencies::new(
        clock,
        device,
        PipelineConfig {
            trigger_poll_interval_ms,
            ..PipelineConfig::default()
        },
    )
    .with_perception(Arc::new(DeepStreamAdapter::new(session))))
}

fn resolve_data_artifact(data_dir: &Path, artifact: &Path) -> Result<PathBuf, LivePerceptionError> {
    if artifact.is_absolute() {
        Ok(artifact.to_owned())
    } else {
        let data_dir = if data_dir.is_absolute() {
            data_dir.to_owned()
        } else {
            std::env::current_dir()
                .map_err(LivePerceptionError::CurrentDirectory)?
                .join(data_dir)
        };
        Ok(data_dir.join(artifact))
    }
}

fn resolve_process_path(path: &Path) -> Result<PathBuf, LivePerceptionError> {
    let resolved = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(LivePerceptionError::CurrentDirectory)?
            .join(path)
    };
    if !resolved.is_file() {
        return Err(LivePerceptionError::ParserLibraryMissing(resolved));
    }
    Ok(resolved)
}

fn parse_capture_format(value: &str) -> Result<CaptureFormat, LivePerceptionError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "MJPEG" | "MJPG" => Ok(CaptureFormat::Mjpeg),
        "NV12" => Ok(CaptureFormat::Nv12),
        "YUY2" | "YUYV" => Ok(CaptureFormat::Yuy2),
        _ => Err(LivePerceptionError::UnsupportedCaptureFormat(
            value.to_owned(),
        )),
    }
}

fn deadline_ns(milliseconds: f64) -> Option<u64> {
    if milliseconds == 0.0 {
        None
    } else {
        Some(
            (milliseconds * 1_000_000.0)
                .round()
                .clamp(1.0, u64::MAX as f64) as u64,
        )
    }
}

fn validate_nvinfer_contract(
    path: &Path,
    expected_parser: &Path,
    confidence_threshold: f64,
    nms_threshold: f64,
    model_width: u32,
    model_height: u32,
    component_id: i32,
) -> Result<(), LivePerceptionError> {
    let source = fs::read_to_string(path).map_err(|source| LivePerceptionError::ReadNvinfer {
        path: path.to_owned(),
        source,
    })?;
    let values = parse_ini_values(&source);
    let parser = required_value(&values, "custom-lib-path", path)?;
    let parser = PathBuf::from(parser);
    let parser = if parser.is_absolute() {
        parser
    } else {
        path.parent().unwrap_or_else(|| Path::new(".")).join(parser)
    };
    let actual_parser =
        parser
            .canonicalize()
            .map_err(|source| LivePerceptionError::CanonicalizeParser {
                path: parser,
                source,
            })?;
    let expected_parser = expected_parser.canonicalize().map_err(|source| {
        LivePerceptionError::CanonicalizeParser {
            path: expected_parser.to_owned(),
            source,
        }
    })?;
    if actual_parser != expected_parser {
        return Err(LivePerceptionError::ParserMismatch {
            expected: expected_parser,
            actual: actual_parser,
        });
    }
    validate_threshold(&values, "pre-cluster-threshold", confidence_threshold, path)?;
    validate_threshold(&values, "nms-iou-threshold", nms_threshold, path)?;
    validate_integer(&values, "gie-unique-id", i64::from(component_id), path)?;
    let engine = resolve_ini_path(path, required_value(&values, "model-engine-file", path)?);
    if !engine.is_file() {
        return Err(LivePerceptionError::EngineMissing(engine));
    }
    validate_model_manifest(
        &engine,
        &source,
        confidence_threshold,
        nms_threshold,
        model_width,
        model_height,
    )?;
    Ok(())
}

fn resolve_ini_path(config: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        config.parent().unwrap_or_else(|| Path::new(".")).join(path)
    }
}

fn parse_ini_values(source: &str) -> BTreeMap<String, String> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with(';')
                || line.starts_with('[')
            {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

fn required_value<'a>(
    values: &'a BTreeMap<String, String>,
    key: &'static str,
    path: &Path,
) -> Result<&'a str, LivePerceptionError> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| LivePerceptionError::NvinferKeyMissing {
            path: path.to_owned(),
            key,
        })
}

fn validate_threshold(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: f64,
    path: &Path,
) -> Result<(), LivePerceptionError> {
    let raw = required_value(values, key, path)?;
    let actual = raw
        .parse::<f64>()
        .map_err(|_| LivePerceptionError::NvinferValueInvalid {
            path: path.to_owned(),
            key,
            value: raw.to_owned(),
        })?;
    if (actual - expected).abs() > 1e-6 {
        return Err(LivePerceptionError::ThresholdMismatch {
            key,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_integer(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: i64,
    path: &Path,
) -> Result<(), LivePerceptionError> {
    let raw = required_value(values, key, path)?;
    let actual = raw
        .parse::<i64>()
        .map_err(|_| LivePerceptionError::NvinferValueInvalid {
            path: path.to_owned(),
            key,
            value: raw.to_owned(),
        })?;
    if actual != expected {
        return Err(LivePerceptionError::IntegerMismatch {
            key,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_model_manifest(
    engine: &Path,
    nvinfer_source: &str,
    confidence_threshold: f64,
    nms_threshold: f64,
    model_width: u32,
    model_height: u32,
) -> Result<(), LivePerceptionError> {
    let file_name = engine
        .file_name()
        .ok_or_else(|| LivePerceptionError::EngineFileName(engine.to_owned()))?;
    let manifest = engine.with_file_name(format!("{}.manifest.json", file_name.to_string_lossy()));
    let source =
        fs::read_to_string(&manifest).map_err(|source| LivePerceptionError::ReadManifest {
            path: manifest.clone(),
            source,
        })?;
    let document: ModelManifest =
        serde_json::from_str(&source).map_err(|source| LivePerceptionError::ParseManifest {
            path: manifest.clone(),
            source,
        })?;
    validate_manifest_shape(&document, model_width, model_height)?;
    if !document.validated {
        return Err(manifest_error("model manifest must have validated=true"));
    }
    let fingerprint = require_nonempty(&document.model_fingerprint, "model_fingerprint")?;
    let computed_fingerprint = compute_model_fingerprint(&document)?;
    if fingerprint != computed_fingerprint {
        return Err(manifest_error(format!(
            "model_fingerprint {fingerprint} does not match canonical content {computed_fingerprint}"
        )));
    }
    let configured_fingerprint = nvinfer_source
        .lines()
        .find_map(|line| line.trim().strip_prefix("# novasight-model-fingerprint="))
        .ok_or_else(|| {
            LivePerceptionError::ManifestContract(
                "nvinfer config has no novasight-model-fingerprint header".to_owned(),
            )
        })?;
    if configured_fingerprint != fingerprint {
        return Err(LivePerceptionError::ManifestContract(format!(
            "nvinfer fingerprint {configured_fingerprint} does not match manifest {fingerprint}"
        )));
    }
    validate_manifest_threshold(
        "postprocess.confidence_threshold",
        document.postprocess.confidence_threshold,
        confidence_threshold,
    )?;
    validate_manifest_threshold(
        "postprocess.nms_iou_threshold",
        document.postprocess.nms_iou_threshold,
        nms_threshold,
    )?;
    let engine_name = engine
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| manifest_error("engine file name is not valid UTF-8"))?;
    if document.artifact.engine_path != engine_name {
        return Err(manifest_error(format!(
            "artifact.engine_path {} does not match engine file {engine_name}",
            document.artifact.engine_path
        )));
    }
    let actual_size = fs::metadata(engine)
        .map_err(|source| LivePerceptionError::ReadEngine {
            path: engine.to_owned(),
            source,
        })?
        .len();
    if actual_size != document.artifact.size_bytes {
        return Err(manifest_error(format!(
            "engine size {actual_size} does not match manifest {}",
            document.artifact.size_bytes
        )));
    }
    let actual_sha = sha256_file(engine)?;
    if actual_sha != document.artifact.sha256 {
        return Err(manifest_error(format!(
            "engine sha256 {actual_sha} does not match manifest {}",
            document.artifact.sha256
        )));
    }
    validate_nvinfer_manifest_contract(nvinfer_source, &document)?;
    Ok(())
}

fn validate_manifest_threshold(
    field: &'static str,
    actual: f64,
    expected: f64,
) -> Result<(), LivePerceptionError> {
    if !actual.is_finite() || (actual - expected).abs() > 1e-6 {
        return Err(manifest_error(format!(
            "manifest {field}={actual} does not match YAML {expected}"
        )));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ModelManifest {
    schema_version: u32,
    model_id: String,
    display_name: String,
    artifact: ManifestArtifact,
    runtime: ManifestRuntime,
    input: ManifestInput,
    output: ManifestOutput,
    postprocess: ManifestPostprocess,
    validated: bool,
    model_fingerprint: String,
}

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    engine_path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct ManifestRuntime {
    precision: String,
    batch_size: u32,
}

#[derive(Debug, Deserialize, Serialize)]
struct ManifestTensor {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    layout: String,
}

#[derive(Debug, Deserialize)]
struct ManifestInput {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    layout: String,
    color_format: String,
    scale_factor: f64,
    maintain_aspect_ratio: bool,
    symmetric_padding: bool,
}

#[derive(Debug, Deserialize)]
struct ManifestOutput {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    layout: String,
    format: String,
    class_count: u32,
    class_names: Vec<String>,
    has_objectness: bool,
    coordinate_mode: String,
    #[serde(default)]
    bindings: Vec<ManifestTensor>,
    #[serde(default)]
    strides: Vec<u32>,
    #[serde(default)]
    anchors: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct ManifestPostprocess {
    parser: String,
    parser_preset: String,
    confidence_threshold: f64,
    nms_iou_threshold: f64,
    max_detections: u32,
}

#[derive(Clone, Copy)]
struct ParserContract {
    function: &'static str,
    cluster_mode: i64,
}

fn validate_manifest_shape(
    manifest: &ModelManifest,
    model_width: u32,
    model_height: u32,
) -> Result<(), LivePerceptionError> {
    if manifest.schema_version != 1 {
        return Err(manifest_error(format!(
            "unsupported schema_version {}; expected 1",
            manifest.schema_version
        )));
    }
    require_nonempty(&manifest.model_id, "model_id")?;
    require_nonempty(&manifest.display_name, "display_name")?;
    require_nonempty(&manifest.artifact.engine_path, "artifact.engine_path")?;
    require_nonempty(&manifest.artifact.sha256, "artifact.sha256")?;
    for (label, tensor) in [
        (
            "input",
            ManifestTensorRef {
                name: &manifest.input.name,
                shape: &manifest.input.shape,
                dtype: &manifest.input.dtype,
                layout: &manifest.input.layout,
            },
        ),
        (
            "output",
            ManifestTensorRef {
                name: &manifest.output.name,
                shape: &manifest.output.shape,
                dtype: &manifest.output.dtype,
                layout: &manifest.output.layout,
            },
        ),
    ] {
        validate_tensor(label, tensor)?;
    }
    for binding in &manifest.output.bindings {
        validate_tensor(
            "output.bindings",
            ManifestTensorRef {
                name: &binding.name,
                shape: &binding.shape,
                dtype: &binding.dtype,
                layout: &binding.layout,
            },
        )?;
    }
    if manifest.runtime.batch_size != 1 {
        return Err(manifest_error("runtime.batch_size must be 1"));
    }
    if !manifest.input.layout.eq_ignore_ascii_case("NCHW")
        || manifest.input.shape.len() != 4
        || manifest.input.shape[0] != 1
        || manifest.input.shape[2] != u64::from(model_height)
        || manifest.input.shape[3] != u64::from(model_width)
    {
        return Err(manifest_error(format!(
            "input {:?} {} does not match configured 1xCx{model_height}x{model_width} NCHW",
            manifest.input.shape, manifest.input.layout
        )));
    }
    if manifest.output.class_count == 0 {
        return Err(manifest_error("output.class_count must be positive"));
    }
    Ok(())
}

struct ManifestTensorRef<'a> {
    name: &'a str,
    shape: &'a [u64],
    dtype: &'a str,
    layout: &'a str,
}

fn validate_tensor(label: &str, tensor: ManifestTensorRef<'_>) -> Result<(), LivePerceptionError> {
    require_nonempty(tensor.name, &format!("{label}.name"))?;
    require_nonempty(tensor.dtype, &format!("{label}.dtype"))?;
    require_nonempty(tensor.layout, &format!("{label}.layout"))?;
    if tensor.shape.is_empty() || tensor.shape.contains(&0) {
        return Err(manifest_error(format!(
            "{label}.shape must contain positive dimensions"
        )));
    }
    Ok(())
}

fn require_nonempty<'a>(value: &'a str, field: &str) -> Result<&'a str, LivePerceptionError> {
    let value = value.trim();
    if value.is_empty() {
        Err(manifest_error(format!("{field} must not be empty")))
    } else {
        Ok(value)
    }
}

fn compute_model_fingerprint(manifest: &ModelManifest) -> Result<String, LivePerceptionError> {
    let mut output = serde_json::json!({
        "name": manifest.output.name,
        "shape": manifest.output.shape,
        "dtype": manifest.output.dtype,
        "layout": manifest.output.layout,
        "format": manifest.output.format,
        "class_count": manifest.output.class_count,
        "has_objectness": manifest.output.has_objectness,
        "coordinate_mode": manifest.output.coordinate_mode,
        "class_names": manifest.output.class_names,
    });
    let output = output
        .as_object_mut()
        .expect("json object construction is infallible");
    if !manifest.output.bindings.is_empty() {
        output.insert(
            "bindings".to_owned(),
            serde_json::to_value(&manifest.output.bindings)
                .map_err(LivePerceptionError::SerializeFingerprint)?,
        );
    }
    if !manifest.output.strides.is_empty() {
        output.insert(
            "strides".to_owned(),
            serde_json::json!(manifest.output.strides),
        );
    }
    if !manifest.output.anchors.is_empty() {
        output.insert(
            "anchors".to_owned(),
            serde_json::json!(manifest.output.anchors),
        );
    }
    let payload = serde_json::json!({
        "artifact_sha256": manifest.artifact.sha256,
        "input": {
            "name": manifest.input.name,
            "shape": manifest.input.shape,
            "dtype": manifest.input.dtype,
            "layout": manifest.input.layout,
        },
        "output": output,
        "parser_schema": "yolo-v1",
    });
    let stable =
        serde_json::to_string(&payload).map_err(LivePerceptionError::SerializeFingerprint)?;
    let stable = python_json_numbers(&stable);
    Ok(sha256_bytes(stable.as_bytes()))
}

/// Match Python's stable_json float spelling at the two ryu differences used
/// by manifests: exponent padding and scientific notation below `1e-4`.
/// Only JSON number tokens outside strings are rewritten.
fn python_json_numbers(json: &str) -> String {
    let bytes = json.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            output.push(byte);
            index += 1;
            continue;
        }
        if !byte.is_ascii_digit() && byte != b'-' {
            output.push(byte);
            index += 1;
            continue;
        }
        let number_start = index;
        index += 1;
        while index < bytes.len()
            && (bytes[index].is_ascii_digit()
                || matches!(bytes[index], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            index += 1;
        }
        output.extend_from_slice(python_number_token(&bytes[number_start..index]).as_bytes());
    }
    String::from_utf8(output).expect("valid JSON remains UTF-8")
}

fn python_number_token(token: &[u8]) -> String {
    let token = std::str::from_utf8(token).expect("JSON numbers are ASCII");
    if let Some(exponent_at) = token.find(['e', 'E']) {
        let (mantissa, exponent) = token.split_at(exponent_at);
        let exponent = &exponent[1..];
        let (sign, digits) = exponent
            .strip_prefix(['+', '-'])
            .map_or(("", exponent), |digits| (&exponent[..1], digits));
        return format!(
            "{mantissa}e{sign}{}{digits}",
            if digits.len() == 1 { "0" } else { "" }
        );
    }
    let (sign, unsigned) = token
        .strip_prefix('-')
        .map_or(("", token), |value| ("-", value));
    let Some(fraction) = unsigned.strip_prefix("0.") else {
        return token.to_owned();
    };
    let leading_zeros = fraction.bytes().take_while(|byte| *byte == b'0').count();
    if leading_zeros < 4 || leading_zeros == fraction.len() {
        return token.to_owned();
    }
    let significant = &fraction[leading_zeros..];
    let (first, rest) = significant.split_at(1);
    let mantissa = if rest.is_empty() {
        first.to_owned()
    } else {
        format!("{first}.{rest}")
    };
    let exponent = leading_zeros + 1;
    format!(
        "{sign}{mantissa}e-{}{exponent}",
        if exponent < 10 { "0" } else { "" }
    )
}

fn validate_nvinfer_manifest_contract(
    source: &str,
    manifest: &ModelManifest,
) -> Result<(), LivePerceptionError> {
    let values = parse_ini_values(source);
    let parser = resolve_parser_contract(manifest)?;
    validate_ini_integer(&values, "batch-size", 1)?;
    validate_ini_integer(&values, "network-type", 0)?;
    validate_ini_integer(&values, "process-mode", 1)?;
    validate_ini_integer(&values, "interval", 0)?;
    validate_ini_integer(
        &values,
        "num-detected-classes",
        i64::from(manifest.output.class_count),
    )?;
    validate_ini_integer(&values, "output-tensor-meta", 0)?;
    validate_ini_integer(&values, "cluster-mode", parser.cluster_mode)?;
    validate_ini_integer(
        &values,
        "network-mode",
        match manifest
            .runtime
            .precision
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "fp32" => 0,
            "int8" => 1,
            "fp16" => 2,
            value => return Err(manifest_error(format!("unsupported precision {value}"))),
        },
    )?;
    validate_ini_integer(
        &values,
        "model-color-format",
        match manifest
            .input
            .color_format
            .trim()
            .to_ascii_uppercase()
            .as_str()
        {
            "RGB" => 0,
            "BGR" => 1,
            "GRAY" | "GREY" => 2,
            value => return Err(manifest_error(format!("unsupported color format {value}"))),
        },
    )?;
    validate_ini_integer(
        &values,
        "maintain-aspect-ratio",
        i64::from(manifest.input.maintain_aspect_ratio),
    )?;
    validate_ini_integer(
        &values,
        "symmetric-padding",
        i64::from(manifest.input.symmetric_padding),
    )?;
    validate_ini_integer(
        &values,
        "topk",
        i64::from(manifest.postprocess.max_detections.max(1)),
    )?;
    validate_ini_float(&values, "net-scale-factor", manifest.input.scale_factor)?;
    validate_ini_string(&values, "parse-bbox-func-name", parser.function)?;
    let output_names = if manifest.output.bindings.is_empty() {
        manifest.output.name.clone()
    } else {
        manifest
            .output
            .bindings
            .iter()
            .map(|binding| binding.name.as_str())
            .collect::<Vec<_>>()
            .join(";")
    };
    validate_ini_string(&values, "output-blob-names", &output_names)?;
    Ok(())
}

fn resolve_parser_contract(
    manifest: &ModelManifest,
) -> Result<ParserContract, LivePerceptionError> {
    let parser = manifest.postprocess.parser.trim().to_ascii_lowercase();
    let format = manifest.output.format.trim().to_ascii_lowercase();
    let preset = normalize_parser_preset(&manifest.postprocess.parser_preset)?;
    match parser.as_str() {
        "decoded_nms" => {
            if format != "decoded_boxes6"
                || manifest.output.shape.len() != 3
                || manifest.output.shape[0] != 1
                || manifest.output.shape[2] != 6
            {
                return Err(manifest_error("decoded_nms requires output shape [1,N,6]"));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightDecodedNms",
                cluster_mode: 4,
            })
        }
        "rockchip_yolov5" => {
            if format != "rockchip_yolov5_three_scale"
                || manifest.output.bindings.len() != 3
                || manifest.output.strides != [8, 16, 32]
                || manifest.output.anchors.len() != 3
                || manifest.output.anchors.iter().any(|scale| scale.len() != 6)
                || matches!(preset, "yolov8" | "yolo11")
            {
                return Err(manifest_error("invalid Rockchip YOLOv5 parser contract"));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightRockchipYoloV5",
                cluster_mode: 2,
            })
        }
        "efficientnms" => {
            if format != "efficientnms_boxes_scores_classes" || manifest.output.bindings.len() != 4
            {
                return Err(manifest_error("invalid EfficientNMS parser contract"));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightEfficientNms",
                cluster_mode: 4,
            })
        }
        "yolo" => {
            if format != "yolo_cxcywh_class_scores"
                || !manifest
                    .output
                    .coordinate_mode
                    .eq_ignore_ascii_case("pixel")
            {
                return Err(manifest_error("invalid raw YOLO output contract"));
            }
            let shape = match manifest.output.shape.as_slice() {
                [1, left, right] => [*left, *right],
                [left, right] => [*left, *right],
                _ => {
                    return Err(manifest_error(
                        "YOLO output must be [C,N], [N,C], or [1,C,N]",
                    ));
                }
            };
            let preset_objectness = match preset {
                "yolov5" => Some(true),
                "yolov8" | "yolo11" => Some(false),
                _ => None,
            };
            if preset_objectness.is_some_and(|value| value != manifest.output.has_objectness) {
                return Err(manifest_error(
                    "parser preset conflicts with output objectness",
                ));
            }
            let has_objectness = preset_objectness.unwrap_or(manifest.output.has_objectness);
            let channels =
                u64::from(manifest.output.class_count) + if has_objectness { 5 } else { 4 };
            if !shape.contains(&channels) || shape.iter().max().copied().unwrap_or(0) <= channels {
                return Err(manifest_error(
                    "YOLO output shape does not match class/objectness contract",
                ));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightRaw",
                cluster_mode: 2,
            })
        }
        _ => Err(manifest_error(format!("unsupported parser {parser}"))),
    }
}

fn normalize_parser_preset(value: &str) -> Result<&str, LivePerceptionError> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    let canonical = match normalized.as_str() {
        "" | "automatic" => "auto",
        "yolo_v5" | "yolov5_raw" => "yolov5",
        "yolo_v8" | "yolov8_raw" => "yolov8",
        "yolo_11" | "yolo11_raw" => "yolo11",
        "generic" | "custom" | "novasight" => "novasight_generic",
        "auto" | "yolov5" | "yolov8" | "yolo11" | "novasight_generic" => normalized.as_str(),
        _ => return Err(manifest_error(format!("unsupported parser preset {value}"))),
    };
    Ok(match canonical {
        "auto" => "auto",
        "yolov5" => "yolov5",
        "yolov8" => "yolov8",
        "yolo11" => "yolo11",
        _ => "novasight_generic",
    })
}

fn validate_ini_string(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: &str,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .ok_or_else(|| manifest_error(format!("nvinfer is missing {key}")))?;
    if actual != expected {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn validate_ini_integer(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: i64,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| manifest_error(format!("nvinfer has no valid {key}")))?;
    if actual != expected {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn validate_ini_float(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: f64,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .ok_or_else(|| manifest_error(format!("nvinfer has no valid {key}")))?;
    if !actual.is_finite() || (actual - expected).abs() > 1e-12 {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn manifest_error(message: impl Into<String>) -> LivePerceptionError {
    LivePerceptionError::ManifestContract(message.into())
}

fn sha256_file(path: &Path) -> Result<String, LivePerceptionError> {
    let mut file = fs::File::open(path).map_err(|source| LivePerceptionError::ReadEngine {
        path: path.to_owned(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| LivePerceptionError::ReadEngine {
                path: path.to_owned(),
                source,
            })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(encode_sha256(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    encode_sha256(digest.finalize())
}

fn encode_sha256(digest: impl IntoIterator<Item = u8>) -> String {
    let digest = digest.into_iter();
    let mut encoded = String::with_capacity(digest.size_hint().0 * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            model_id: "detector".to_owned(),
            display_name: "Detector".to_owned(),
            artifact: ManifestArtifact {
                engine_path: "detector.engine".to_owned(),
                sha256: "ab".repeat(32),
                size_bytes: 123,
            },
            runtime: ManifestRuntime {
                precision: "fp16".to_owned(),
                batch_size: 1,
            },
            input: ManifestInput {
                name: "images".to_owned(),
                shape: vec![1, 3, 640, 640],
                dtype: "float32".to_owned(),
                layout: "NCHW".to_owned(),
                color_format: "RGB".to_owned(),
                scale_factor: 1.0 / 255.0,
                maintain_aspect_ratio: false,
                symmetric_padding: false,
            },
            output: ManifestOutput {
                name: "output0".to_owned(),
                shape: vec![1, 84, 8400],
                dtype: "float32".to_owned(),
                layout: "NCHW".to_owned(),
                format: "yolo_cxcywh_class_scores".to_owned(),
                class_count: 80,
                class_names: (0..80).map(|value| value.to_string()).collect(),
                has_objectness: false,
                coordinate_mode: "pixel".to_owned(),
                bindings: Vec::new(),
                strides: Vec::new(),
                anchors: Vec::new(),
            },
            postprocess: ManifestPostprocess {
                parser: "yolo".to_owned(),
                parser_preset: "yolov8".to_owned(),
                confidence_threshold: 0.25,
                nms_iou_threshold: 0.45,
                max_detections: 300,
            },
            validated: true,
            model_fingerprint: String::new(),
        }
    }

    #[test]
    fn fingerprint_matches_python_canonical_json_contract() {
        assert_eq!(
            compute_model_fingerprint(&canonical_manifest()).expect("fingerprint"),
            "3866800678bc7e9031b148c4ec8f5314457c93f211f780557abaea5212e56d9b"
        );
    }

    #[test]
    fn fingerprint_matches_python_exponent_float_format() {
        let mut manifest = canonical_manifest();
        manifest.output.anchors = vec![vec![1e-7]];
        assert_eq!(
            compute_model_fingerprint(&manifest).expect("fingerprint"),
            "3baa62909243392b4bd0aeec3a23726a23112973a346654f5f6b9fd3b280858d"
        );
        assert_eq!(
            python_json_numbers(r#"{"value":1e-7,"text":"1e-7"}"#),
            r#"{"value":1e-07,"text":"1e-7"}"#
        );
    }

    #[test]
    fn fingerprint_matches_python_small_fixed_float_format() {
        let mut manifest = canonical_manifest();
        manifest.output.anchors = vec![vec![1e-5]];
        assert_eq!(
            compute_model_fingerprint(&manifest).expect("fingerprint"),
            "efdfa2dc09af968ea57a7a7bcbf9c098a7e83335897e8b2e95574dfbbdeb5696"
        );
        assert_eq!(python_number_token(b"0.00001234"), "1.234e-05");
    }

    #[test]
    fn nvinfer_contract_rejects_parser_semantic_drift() {
        let manifest = canonical_manifest();
        let valid = format!(
            "batch-size=1\nnetwork-mode=2\nnetwork-type=0\nprocess-mode=1\ninterval=0\nnum-detected-classes=80\nnet-scale-factor={:.17}\nmodel-color-format=0\nmaintain-aspect-ratio=0\nsymmetric-padding=0\noutput-tensor-meta=0\noutput-blob-names=output0\nparse-bbox-func-name=NvDsInferParseNovaSightRaw\ncluster-mode=2\ntopk=300\n",
            1.0 / 255.0
        );
        validate_nvinfer_manifest_contract(&valid, &manifest).expect("canonical contract");
        let drifted = valid.replace(
            "NvDsInferParseNovaSightRaw",
            "NvDsInferParseNovaSightEfficientNms",
        );
        assert!(validate_nvinfer_manifest_contract(&drifted, &manifest).is_err());
    }
}

#[derive(Debug, Error)]
pub(super) enum LivePerceptionError {
    #[error("production adapter configuration is invalid: {0}")]
    Config(ConfigValidationError),
    #[error("live DeepStream requires inference.enabled=true")]
    InferenceDisabled,
    #[error("live DeepStream currently requires capture.preference=manual")]
    AutomaticCaptureUnsupported,
    #[error("production kmNet output requires hardware.auto_connect=true")]
    DeviceAutoConnectDisabled,
    #[error("kmNet production adapter failed: {0}")]
    KmNet(KmNetError),
    #[error("hardware.host must be an IPv4 address for native kmNet: {0}")]
    InvalidKmNetHost(String),
    #[error("native kmNet production adapter failed: {0}")]
    NativeKmNet(KmNetNativeError),
    #[error("unsupported capture pixel format: {0}")]
    UnsupportedCaptureFormat(String),
    #[error("invalid DeepStream I/O mode: {0}")]
    InvalidIoMode(i32),
    #[error("nvinfer configuration file does not exist: {}", .0.display())]
    NvinferConfigMissing(PathBuf),
    #[error("DeepStream parser library does not exist: {}", .0.display())]
    ParserLibraryMissing(PathBuf),
    #[error("failed to read current working directory: {0}")]
    CurrentDirectory(std::io::Error),
    #[error("failed to read nvinfer configuration {}: {source}", path.display())]
    ReadNvinfer {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("nvinfer configuration {} is missing required key {key}", path.display())]
    NvinferKeyMissing { path: PathBuf, key: &'static str },
    #[error("nvinfer configuration {} has invalid {key} value {value}", path.display())]
    NvinferValueInvalid {
        path: PathBuf,
        key: &'static str,
        value: String,
    },
    #[error("failed to resolve DeepStream parser library {}: {source}", path.display())]
    CanonicalizeParser {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("nvinfer parser mismatch: expected {}, got {}", expected.display(), actual.display())]
    ParserMismatch { expected: PathBuf, actual: PathBuf },
    #[error("nvinfer {key} mismatch: expected {expected}, got {actual}")]
    ThresholdMismatch {
        key: &'static str,
        expected: f64,
        actual: f64,
    },
    #[error("nvinfer {key} mismatch: expected {expected}, got {actual}")]
    IntegerMismatch {
        key: &'static str,
        expected: i64,
        actual: i64,
    },
    #[error("TensorRT engine does not exist: {}", .0.display())]
    EngineMissing(PathBuf),
    #[error("TensorRT engine path has no file name: {}", .0.display())]
    EngineFileName(PathBuf),
    #[error("failed to read model manifest {}: {source}", path.display())]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model manifest {}: {source}", path.display())]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize canonical model fingerprint: {0}")]
    SerializeFingerprint(serde_json::Error),
    #[error("failed to read TensorRT engine {}: {source}", path.display())]
    ReadEngine {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("model deployment contract invalid: {0}")]
    ManifestContract(String),
    #[error("DeepStream pipeline configuration is invalid: {0}")]
    Pipeline(String),
}
