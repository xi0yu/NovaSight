//! Native TensorRT model admission for the Jetson daemon.
//!
//! TensorRT/CUDA ownership stays behind `novasight-tensorrt`; this module owns
//! the user-supplied model semantics, the canonical manifest transaction and
//! the compatibility profile returned to the current WebUI.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use novasight_store::model_manifest::{
    ManifestArtifact, ManifestInput, ManifestOutput, ManifestPostprocess, ManifestRuntime,
    ModelManifest, compute_model_fingerprint,
};
use novasight_tensorrt::{
    DecodeContract, DetectionDecoder, EngineContract, TensorDtype, TensorRtEngine,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::model_ingress::{
    ModelIngressError, ModelProbeInputMode, ModelProfileConfigureRequest, ModelWorkerOutput,
    automatic_activation_profile_from_value,
};

const MANIFEST_LIMIT: usize = 1024 * 1024;
const RUNTIME_MAX_DETECTIONS: u32 = 256;

#[derive(Clone, Debug, Default)]
pub struct NativeModelJobRunner;

impl NativeModelJobRunner {
    pub const fn new() -> Self {
        Self
    }

    pub async fn preflight(&self) -> Result<(), ModelIngressError> {
        TensorRtEngine::preflight().map_err(native_failure)
    }

    pub(crate) async fn admit(
        &self,
        engine_path: &Path,
        display_name: &str,
        parser_preset: &str,
        catalog_labels: &[String],
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        check_cancelled(&cancellation)?;
        let engine_path = engine_path.to_owned();
        let display_name = display_name.to_owned();
        let parser_preset = parser_preset.to_owned();
        let catalog_labels = catalog_labels.to_vec();
        let cancellation_for_job = Arc::clone(&cancellation);
        let output = tokio::task::spawn_blocking(move || {
            check_cancelled(&cancellation_for_job)?;
            admit_sync(&engine_path, &display_name, &parser_preset, &catalog_labels)
        })
        .await
        .map_err(|error| {
            ModelIngressError::Failed(format!("native admission task failed: {error}"))
        })??;
        check_cancelled(&cancellation)?;
        Ok(output)
    }

    pub(crate) async fn inspect(
        &self,
        engine_path: &Path,
        display_name: &str,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        check_cancelled(&cancellation)?;
        let engine_path = engine_path.to_owned();
        let display_name = display_name.to_owned();
        let cancellation_for_job = Arc::clone(&cancellation);
        let output = tokio::task::spawn_blocking(move || {
            check_cancelled(&cancellation_for_job)?;
            inspect_sync(&engine_path, &display_name)
        })
        .await
        .map_err(|error| {
            ModelIngressError::Failed(format!("native inspect task failed: {error}"))
        })??;
        check_cancelled(&cancellation)?;
        Ok(output)
    }

    pub(crate) async fn configure(
        &self,
        engine_path: &Path,
        request: &ModelProfileConfigureRequest,
        cancellation: Arc<AtomicUsize>,
    ) -> Result<ModelWorkerOutput, ModelIngressError> {
        request.validate()?;
        check_cancelled(&cancellation)?;
        let engine_path = engine_path.to_owned();
        let request = request.clone();
        let cancellation_for_job = Arc::clone(&cancellation);
        let output = tokio::task::spawn_blocking(move || {
            check_cancelled(&cancellation_for_job)?;
            configure_sync(&engine_path, &request)
        })
        .await
        .map_err(|error| {
            ModelIngressError::Failed(format!("native configure task failed: {error}"))
        })??;
        check_cancelled(&cancellation)?;
        Ok(output)
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
        check_cancelled(&cancellation)?;
        let engine_path = engine_path.to_owned();
        let cancellation_for_job = Arc::clone(&cancellation);
        let output = tokio::task::spawn_blocking(move || {
            check_cancelled(&cancellation_for_job)?;
            probe_sync(&engine_path)
        })
        .await
        .map_err(|error| {
            ModelIngressError::Failed(format!("native probe task failed: {error}"))
        })??;
        check_cancelled(&cancellation)?;
        Ok(output)
    }
}

fn inspect_sync(
    engine_path: &Path,
    display_name: &str,
) -> Result<ModelWorkerOutput, ModelIngressError> {
    let engine = TensorRtEngine::inspect(engine_path).map_err(native_failure)?;
    let profile = inspected_profile(engine_path, display_name, engine.contract())?;
    let profile_path = manifest_path(engine_path)?;
    write_manifest(
        &profile_path,
        &json!({
            "schema_version": 1,
            "manifest_kind": "novasight_model",
            "model_profile": profile,
        }),
    )?;
    Ok(ModelWorkerOutput {
        profile_path,
        profile,
        report: None,
    })
}

fn admit_sync(
    engine_path: &Path,
    display_name: &str,
    parser_preset: &str,
    catalog_labels: &[String],
) -> Result<ModelWorkerOutput, ModelIngressError> {
    let mut engine = TensorRtEngine::inspect(engine_path).map_err(native_failure)?;
    let mut profile = inspected_profile(engine_path, display_name, engine.contract())?;
    let request = automatic_activation_profile_from_value(&profile, parser_preset, catalog_labels)?;
    request.validate()?;
    configure_profile(&mut profile, engine.contract(), &request)?;
    probe_loaded(engine_path, profile, &mut engine)
}

fn configure_sync(
    engine_path: &Path,
    request: &ModelProfileConfigureRequest,
) -> Result<ModelWorkerOutput, ModelIngressError> {
    let profile_path = manifest_path(engine_path)?;
    let existing = load_manifest(&profile_path)?;
    let display_name = existing
        .get("model_profile")
        .and_then(|profile| profile.get("display_name"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            engine_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("model")
        });
    let engine = TensorRtEngine::inspect(engine_path).map_err(native_failure)?;
    let contract = engine.contract().clone();
    let mut profile = inspected_profile(engine_path, display_name, &contract)?;
    configure_profile(&mut profile, &contract, request)?;
    write_manifest(
        &profile_path,
        &json!({
            "schema_version": 1,
            "manifest_kind": "novasight_model",
            "model_profile": profile,
        }),
    )?;
    Ok(ModelWorkerOutput {
        profile_path,
        profile,
        report: None,
    })
}

fn configure_profile(
    profile: &mut Value,
    contract: &EngineContract,
    request: &ModelProfileConfigureRequest,
) -> Result<(), ModelIngressError> {
    validate_semantics(contract, request)?;
    profile["status"] = json!("READY_FOR_PROBE");
    profile["preprocess"] = json!({
        "color_format": request.color_format.trim().to_ascii_uppercase(),
        "scale": request.scale,
        "offsets": request.offsets,
        "mean": request.mean,
        "std": request.std,
        "resize_mode": request.resize_mode.trim().to_ascii_lowercase(),
        "symmetric_padding": request.symmetric_padding,
        "padding_value": request.padding_value,
    });
    profile["decoder"] = json!({
        "parser_type": normalize_parser(&request.parser_type)?,
        "class_count": request.class_count,
        "bbox_format": request.bbox_format.trim().to_ascii_lowercase(),
        "has_objectness": request.has_objectness,
    });
    profile["postprocess"] = json!({
        "confidence_threshold": request.confidence_threshold,
        "nms_threshold": request.nms_threshold,
        "max_detections": request.max_detections.min(RUNTIME_MAX_DETECTIONS),
    });
    profile["labels"] = json!(request.labels);
    Ok(())
}

fn probe_sync(engine_path: &Path) -> Result<ModelWorkerOutput, ModelIngressError> {
    let profile_path = manifest_path(engine_path)?;
    let root = load_manifest(&profile_path)?;
    let profile = root.get("model_profile").cloned().ok_or_else(|| {
        ModelIngressError::Protocol("inspect and configure the Engine first".to_owned())
    })?;
    let mut engine = TensorRtEngine::inspect(engine_path).map_err(native_failure)?;
    probe_loaded(engine_path, profile, &mut engine)
}

fn probe_loaded(
    engine_path: &Path,
    mut profile: Value,
    engine: &mut TensorRtEngine,
) -> Result<ModelWorkerOutput, ModelIngressError> {
    let profile_path = manifest_path(engine_path)?;
    if !matches!(
        profile.get("status").and_then(Value::as_str),
        Some("READY_FOR_PROBE" | "VALIDATED")
    ) {
        return Err(ModelIngressError::InvalidRequest(
            "model profile must be READY_FOR_PROBE before diagnostics".to_owned(),
        ));
    }

    let parser = normalize_parser(&profile_string(&profile, &["decoder", "parser_type"])?)?;
    let class_count = profile_u32(&profile, &["decoder", "class_count"])?;
    let has_objectness = profile_bool(&profile, &["decoder", "has_objectness"])?;
    let confidence = profile_f32(&profile, &["postprocess", "confidence_threshold"])?;
    let nms = profile_f32(&profile, &["postprocess", "nms_threshold"])?;
    let max_detections = profile_u32(&profile, &["postprocess", "max_detections"])?;
    let input_shape = profile_shape(&profile, &["input", "runtime_shape"])?;
    let output_name = profile_string(&profile, &["outputs", "0", "name"])?;

    let contract = engine.contract().clone();
    if input_shape.as_slice() != contract.input().dimensions() {
        return Err(ModelIngressError::InvalidRequest(
            "configured input shape no longer matches the TensorRT Engine".to_owned(),
        ));
    }
    let expected_objectness = parser == "yolov5_raw";
    if has_objectness != expected_objectness {
        return Err(ModelIngressError::InvalidRequest(format!(
            "{parser} requires has_objectness={expected_objectness}"
        )));
    }
    let decoder = DetectionDecoder::new(
        DecodeContract::RawYolo {
            output_name: output_name.clone(),
            class_count,
            has_objectness,
        },
        confidence,
        nms,
        max_detections as usize,
        input_shape[3] as u32,
        input_shape[2] as u32,
    )
    .map_err(|error| ModelIngressError::InvalidRequest(error.to_string()))?;
    decoder
        .validate_engine_contract(&contract)
        .map_err(|error| ModelIngressError::InvalidRequest(error.to_string()))?;

    let inference_started = Instant::now();
    let outputs = engine.probe_zero().map_err(native_failure)?;
    let inference_ms = inference_started.elapsed().as_secs_f64() * 1000.0;
    let output_tensor_ok = outputs.iter().all(|tensor| {
        (0..tensor.element_count()).all(|index| tensor.value_f32(index).is_ok_and(f32::is_finite))
    });
    if !output_tensor_ok {
        return Err(ModelIngressError::InvalidRequest(
            "TensorRT probe output contains NaN, Inf, or an unsupported dtype".to_owned(),
        ));
    }
    let decode_started = Instant::now();
    let batch = decoder
        .decode(&outputs)
        .map_err(|error| ModelIngressError::InvalidRequest(error.to_string()))?;
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    drop(outputs);

    let mut manifest = canonical_manifest(engine_path, &profile, &contract, parser)?;
    manifest.model_fingerprint = compute_model_fingerprint(&manifest).map_err(|error| {
        ModelIngressError::Protocol(format!("manifest fingerprint failed: {error}"))
    })?;
    profile["status"] = json!("VALIDATED");
    profile["validation"] = json!({
        "status": "validated",
        "validated_at": format!("unix-ms:{}", unix_time_ms()),
        "engine_execution_ok": true,
        "decoder_ok": true,
        "nms_ok": true,
        "detection_batch_ok": true,
        "profile_fingerprint": manifest.model_fingerprint,
        "issues": [],
    });
    let report = json!({
        "status": "validated",
        "engine_execution_ok": true,
        "output_tensor_ok": true,
        "decoder_ok": true,
        "nms_ok": true,
        "detection_batch_ok": true,
        "preprocess_ms": null,
        "inference_ms": inference_ms,
        "decode_ms": decode_ms,
        "nms_ms": null,
        "decode_includes_nms": true,
        "probe_input": "zero_model_tensor",
        "detection_count": batch.detections().len(),
        "issues": [],
    });
    let mut written = serde_json::to_value(manifest).map_err(ModelIngressError::DecodeResponse)?;
    let written_object = written.as_object_mut().ok_or_else(|| {
        ModelIngressError::Protocol("canonical manifest must serialize as an object".to_owned())
    })?;
    written_object.insert("manifest_kind".to_owned(), json!("novasight_model"));
    written_object.insert("model_profile".to_owned(), profile.clone());
    write_manifest(&profile_path, &written)?;
    Ok(ModelWorkerOutput {
        profile_path,
        profile,
        report: Some(report),
    })
}

fn inspected_profile(
    engine_path: &Path,
    display_name: &str,
    contract: &EngineContract,
) -> Result<Value, ModelIngressError> {
    let canonical = engine_path
        .canonicalize()
        .map_err(ModelIngressError::ReadProfile)?;
    let metadata = canonical
        .metadata()
        .map_err(ModelIngressError::ReadProfile)?;
    let sha = sha256_file(&canonical)?;
    let display_name = if display_name.trim().is_empty() {
        canonical
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("model")
    } else {
        display_name.trim()
    };
    let input = contract.input();
    let input_shape = input.dimensions();
    let declared_input_shape = if contract.input_dynamic() {
        &[][..]
    } else {
        input_shape
    };
    let input_descriptor = tensor_descriptor(
        "input",
        input.name(),
        declared_input_shape,
        dtype_name(input.dtype()),
    );
    let outputs = contract
        .outputs()
        .iter()
        .map(|output| {
            let declared_shape = if contract.input_dynamic() {
                &[][..]
            } else {
                output.dimensions()
            };
            json!({
                "name": output.name(),
                "shape": output.dimensions(),
                "engine_shape": declared_shape,
                "dtype": dtype_name(output.dtype()),
            })
        })
        .collect::<Vec<_>>();
    let inspection_outputs = contract
        .outputs()
        .iter()
        .map(|output| {
            let declared_shape = if contract.input_dynamic() {
                &[][..]
            } else {
                output.dimensions()
            };
            tensor_descriptor(
                "output",
                output.name(),
                declared_shape,
                dtype_name(output.dtype()),
            )
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema_version": 1,
        "model_id": format!("sha256:{sha}"),
        "display_name": display_name,
        "status": "NEEDS_CONFIGURATION",
        "engine": {
            "path": canonical,
            "sha256": format!("sha256:{sha}"),
            "file_size": metadata.len(),
            "modified_at_ns": metadata.modified().ok().and_then(|value| value.duration_since(UNIX_EPOCH).ok()).map(|value| value.as_nanos()).unwrap_or(0),
        },
        "inspection": {
            "deserialize_ok": true,
            "compatible": true,
            "engine_name": null,
            "inputs": [input_descriptor],
            "outputs": inspection_outputs,
            "profiles": null,
            "selected_profile": contract.selected_profile(),
            "has_dynamic_shape": contract.input_dynamic(),
            "has_shape_input": null,
            "requires_plugin": null,
            "error_code": null,
            "raw_error": "",
            "warnings": if contract.input_dynamic() {
                vec![
                    "runtime_shape is optimization profile 0 OPT; declared dynamic dimensions are intentionally not guessed",
                    "the narrow runtime ABI does not report profile min/max ranges or tensor vectorization metadata",
                ]
            } else {
                vec!["the narrow runtime ABI does not report tensor vectorization metadata"]
            },
        },
        "input": {
            "name": input.name(),
            "runtime_shape": input_shape,
            "engine_shape": declared_input_shape,
            "dtype": dtype_name(input.dtype()),
            "layout": "NCHW",
            "profile_index": contract.selected_profile(),
        },
        "outputs": outputs,
        "preprocess": {"color_format":"", "scale":null, "offsets":[], "mean":[], "std":[], "resize_mode":"", "symmetric_padding":false, "padding_value":0.0},
        "decoder": {"parser_type":"", "class_count":0, "bbox_format":"", "has_objectness":null},
        "postprocess": {"confidence_threshold":0.25, "nms_threshold":0.45, "max_detections":RUNTIME_MAX_DETECTIONS},
        "labels": [],
        "parser_candidates": parser_candidates(contract),
        "validation": {"status":"not_run", "validated_at":"", "engine_execution_ok":false, "decoder_ok":false, "nms_ok":false, "detection_batch_ok":false, "profile_fingerprint":"", "issues":[]},
    }))
}

fn validate_semantics(
    contract: &EngineContract,
    request: &ModelProfileConfigureRequest,
) -> Result<(), ModelIngressError> {
    if contract.outputs().len() != 1 {
        return Err(ModelIngressError::InvalidRequest(
            "the native YOLO runtime currently requires exactly one output tensor".to_owned(),
        ));
    }
    let parser = normalize_parser(&request.parser_type)?;
    if !request.offsets.is_empty() || !request.mean.is_empty() || !request.std.is_empty() {
        return Err(ModelIngressError::InvalidRequest(
            "the production manifest cannot represent offsets/mean/std; leave them empty"
                .to_owned(),
        ));
    }
    if request.padding_value != 0.0 {
        return Err(ModelIngressError::InvalidRequest(
            "the production manifest currently requires padding_value=0".to_owned(),
        ));
    }
    if request.bbox_format.trim().to_ascii_lowercase() != "xywh" {
        return Err(ModelIngressError::InvalidRequest(format!(
            "{parser} requires bbox_format=xywh"
        )));
    }
    let expected_objectness = parser == "yolov5_raw";
    if request.has_objectness != expected_objectness {
        return Err(ModelIngressError::InvalidRequest(format!(
            "{parser} requires has_objectness={expected_objectness}"
        )));
    }
    let input = contract.input().dimensions();
    let decoder = DetectionDecoder::new(
        DecodeContract::RawYolo {
            output_name: contract.outputs()[0].name().to_owned(),
            class_count: request.class_count,
            has_objectness: request.has_objectness,
        },
        request.confidence_threshold as f32,
        request.nms_threshold as f32,
        request.max_detections.min(RUNTIME_MAX_DETECTIONS) as usize,
        input[3] as u32,
        input[2] as u32,
    )
    .map_err(|error| ModelIngressError::InvalidRequest(error.to_string()))?;
    decoder
        .validate_engine_contract(contract)
        .map_err(|error| ModelIngressError::InvalidRequest(error.to_string()))
}

fn canonical_manifest(
    engine_path: &Path,
    profile: &Value,
    contract: &EngineContract,
    parser: String,
) -> Result<ModelManifest, ModelIngressError> {
    let metadata = engine_path
        .metadata()
        .map_err(ModelIngressError::ReadProfile)?;
    let sha = sha256_file(engine_path)?;
    let input = contract.input();
    let output = &contract.outputs()[0];
    let labels = profile
        .get("labels")
        .and_then(Value::as_array)
        .ok_or_else(|| ModelIngressError::Protocol("profile.labels is missing".to_owned()))?
        .iter()
        .map(|value| value.as_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| ModelIngressError::Protocol("profile.labels is invalid".to_owned()))?;
    let input_dtype = dtype_name(input.dtype()).to_owned();
    let parser_preset = match parser.as_str() {
        "yolov5_raw" => "yolov5",
        "yolov8_raw" => "yolov8",
        "yolo11_raw" => "yolo11",
        _ => unreachable!("normalized parser"),
    };
    Ok(ModelManifest {
        schema_version: 1,
        model_id: profile_string(profile, &["model_id"])?,
        display_name: profile_string(profile, &["display_name"])?,
        artifact: ManifestArtifact {
            engine_path: engine_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            sha256: sha,
            size_bytes: metadata.len(),
        },
        runtime: ManifestRuntime {
            backend: "custom_tensorrt".to_owned(),
            precision: if input_dtype == "float16" {
                "fp16"
            } else {
                "fp32"
            }
            .to_owned(),
            batch_size: 1,
        },
        input: ManifestInput {
            name: input.name().to_owned(),
            shape: input.dimensions().to_vec(),
            dtype: input_dtype,
            layout: "NCHW".to_owned(),
            color_format: profile_string(profile, &["preprocess", "color_format"])?,
            scale_factor: profile_f64(profile, &["preprocess", "scale"])?,
            maintain_aspect_ratio: profile_string(profile, &["preprocess", "resize_mode"])?
                == "letterbox",
            symmetric_padding: profile_bool(profile, &["preprocess", "symmetric_padding"])?,
        },
        output: ManifestOutput {
            name: output.name().to_owned(),
            shape: output.dimensions().to_vec(),
            dtype: dtype_name(output.dtype()).to_owned(),
            layout: "NCHW".to_owned(),
            format: "yolo_cxcywh_class_scores".to_owned(),
            class_count: labels.len() as u32,
            class_names: labels,
            has_objectness: profile_bool(profile, &["decoder", "has_objectness"])?,
            scores_are_sigmoid: true,
            coordinate_mode: "pixel".to_owned(),
            bindings: Vec::new(),
            strides: Vec::new(),
            anchors: Vec::new(),
        },
        postprocess: ManifestPostprocess {
            parser: "yolo".to_owned(),
            parser_preset: parser_preset.to_owned(),
            confidence_threshold: profile_f64(profile, &["postprocess", "confidence_threshold"])?,
            nms_iou_threshold: profile_f64(profile, &["postprocess", "nms_threshold"])?,
            class_aware_nms: true,
            max_detections: profile_u32(profile, &["postprocess", "max_detections"])?,
        },
        validated: true,
        model_fingerprint: String::new(),
    })
}

fn tensor_descriptor(mode: &str, name: &str, shape: &[u64], dtype: &str) -> Value {
    json!({
        "name": name,
        "io_mode": mode,
        "engine_shape": shape,
        "data_type": dtype,
        "tensor_format": null,
        "is_shape_tensor": null,
        "bytes_per_component": if dtype == "float16" { 2 } else { 4 },
        "components_per_element": null,
        "vectorized_dim": null,
    })
}

fn parser_candidates(contract: &EngineContract) -> Vec<Value> {
    let Some(output) = contract.outputs().first() else {
        return Vec::new();
    };
    let shape = output.dimensions();
    let semantic_shape = if shape.first() == Some(&1) {
        &shape[1..]
    } else {
        shape
    };
    let probable_channels = semantic_shape.iter().copied().min().unwrap_or(0);
    let mut candidates = Vec::new();
    if probable_channels >= 5 {
        candidates.push(json!({"parser_type":"yolov8_raw", "confidence":"medium", "reason":"single raw YOLO tensor", "requires_confirmation":true}));
        candidates.push(json!({"parser_type":"yolo11_raw", "confidence":"medium", "reason":"single raw YOLO tensor", "requires_confirmation":true}));
        candidates.push(json!({"parser_type":"yolov5_raw", "confidence":"low", "reason":"objectness cannot be inferred from shape alone", "requires_confirmation":true}));
    }
    candidates
}

fn normalize_parser(value: &str) -> Result<String, ModelIngressError> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    matches!(
        normalized.as_str(),
        "yolov5_raw" | "yolov8_raw" | "yolo11_raw"
    )
    .then_some(normalized)
    .ok_or_else(|| {
        ModelIngressError::InvalidRequest(
            "parser_type must be yolov5_raw, yolov8_raw, or yolo11_raw".to_owned(),
        )
    })
}

fn dtype_name(dtype: TensorDtype) -> &'static str {
    match dtype {
        TensorDtype::Float16 => "float16",
        TensorDtype::Float32 => "float32",
    }
}

fn manifest_path(engine_path: &Path) -> Result<PathBuf, ModelIngressError> {
    let name = engine_path.file_name().ok_or_else(|| {
        ModelIngressError::Protocol(format!(
            "Engine path has no filename: {}",
            engine_path.display()
        ))
    })?;
    let mut manifest = name.to_os_string();
    manifest.push(".manifest.json");
    Ok(engine_path.with_file_name(manifest))
}

fn load_manifest(path: &Path) -> Result<Value, ModelIngressError> {
    let bytes = fs::read(path).map_err(ModelIngressError::ReadProfile)?;
    if bytes.len() > MANIFEST_LIMIT {
        return Err(ModelIngressError::OutputLimitExceeded(MANIFEST_LIMIT));
    }
    serde_json::from_slice(&bytes).map_err(ModelIngressError::DecodeResponse)
}

fn write_manifest(path: &Path, value: &Value) -> Result<(), ModelIngressError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(ModelIngressError::DecodeResponse)?;
    bytes.push(b'\n');
    if bytes.len() > MANIFEST_LIMIT {
        return Err(ModelIngressError::OutputLimitExceeded(MANIFEST_LIMIT));
    }
    let temporary = path.with_file_name(format!(
        ".{}.native-{}-{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        unix_time_ms(),
    ));
    let operation = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(ModelIngressError::ReadProfile)?;
        file.write_all(&bytes)
            .map_err(ModelIngressError::ReadProfile)?;
        file.sync_all().map_err(ModelIngressError::ReadProfile)?;
        fs::rename(&temporary, path).map_err(ModelIngressError::ReadProfile)
    })();
    if operation.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    operation
}

fn sha256_file(path: &Path) -> Result<String, ModelIngressError> {
    let mut file = File::open(path).map_err(ModelIngressError::ReadProfile)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(ModelIngressError::ReadProfile)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn profile_value<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(root, |value, key| {
        if let Ok(index) = key.parse::<usize>() {
            value.as_array()?.get(index)
        } else {
            value.get(*key)
        }
    })
}

fn profile_string(root: &Value, path: &[&str]) -> Result<String, ModelIngressError> {
    profile_value(root, path)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} is missing", path.join(".")))
        })
}

fn profile_bool(root: &Value, path: &[&str]) -> Result<bool, ModelIngressError> {
    profile_value(root, path)
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} is invalid", path.join(".")))
        })
}

fn profile_f64(root: &Value, path: &[&str]) -> Result<f64, ModelIngressError> {
    profile_value(root, path)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} is invalid", path.join(".")))
        })
}

fn profile_f32(root: &Value, path: &[&str]) -> Result<f32, ModelIngressError> {
    let value = profile_f64(root, path)? as f32;
    value.is_finite().then_some(value).ok_or_else(|| {
        ModelIngressError::Protocol(format!("profile.{} exceeds f32", path.join(".")))
    })
}

fn profile_u32(root: &Value, path: &[&str]) -> Result<u32, ModelIngressError> {
    profile_value(root, path)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} is invalid", path.join(".")))
        })
}

fn profile_shape(root: &Value, path: &[&str]) -> Result<[u64; 4], ModelIngressError> {
    let values = profile_value(root, path)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} is missing", path.join(".")))
        })?;
    let values = values
        .iter()
        .map(Value::as_u64)
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.try_into().ok())
        .ok_or_else(|| {
            ModelIngressError::Protocol(format!("profile.{} must be rank-4", path.join(".")))
        })?;
    Ok(values)
}

fn check_cancelled(cancellation: &AtomicUsize) -> Result<(), ModelIngressError> {
    (cancellation.load(Ordering::Acquire) == 0)
        .then_some(())
        .ok_or(ModelIngressError::Cancelled)
}

fn native_failure(error: impl std::fmt::Display) -> ModelIngressError {
    ModelIngressError::InvalidRequest(format!("TensorRT model contract failed: {error}"))
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
