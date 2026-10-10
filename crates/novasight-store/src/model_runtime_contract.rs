use thiserror::Error;

use crate::model_manifest::{ModelManifest, compute_model_fingerprint, model_fingerprint_matches};

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct ModelRuntimeContractError {
    message: String,
}

impl ModelRuntimeContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Validate the platform-independent model contract once before artifact I/O
/// or adapter composition. File name, size and digest binding remain the
/// caller's responsibility because they require filesystem access.
pub fn validate_model_runtime_contract(
    manifest: &ModelManifest,
    output_class_names_present: bool,
) -> Result<(), ModelRuntimeContractError> {
    validate_manifest_shape(manifest)?;
    if !manifest.validated {
        return Err(error("model manifest must have validated=true"));
    }
    let fingerprint = require_nonempty(&manifest.model_fingerprint, "model_fingerprint")?;
    let canonical = compute_model_fingerprint(manifest)
        .map_err(|source| error(format!("failed to serialize model fingerprint: {source}")))?;
    if fingerprint != canonical
        && !model_fingerprint_matches(manifest, fingerprint, output_class_names_present)
            .map_err(|source| error(format!("failed to serialize model fingerprint: {source}")))?
    {
        return Err(error(format!(
            "model_fingerprint {fingerprint} does not match canonical content {canonical}"
        )));
    }
    validate_probability(
        "postprocess.confidence_threshold",
        manifest.postprocess.confidence_threshold,
    )?;
    validate_probability(
        "postprocess.nms_iou_threshold",
        manifest.postprocess.nms_iou_threshold,
    )?;
    Ok(())
}

fn validate_manifest_shape(manifest: &ModelManifest) -> Result<(), ModelRuntimeContractError> {
    if manifest.schema_version != 1 {
        return Err(error(format!(
            "unsupported schema_version {}; expected 1",
            manifest.schema_version
        )));
    }
    require_nonempty(&manifest.model_id, "model_id")?;
    require_nonempty(&manifest.display_name, "display_name")?;
    require_nonempty(&manifest.artifact.engine_path, "artifact.engine_path")?;
    require_nonempty(&manifest.artifact.sha256, "artifact.sha256")?;
    validate_tensor(
        "input",
        &manifest.input.name,
        &manifest.input.shape,
        &manifest.input.dtype,
        &manifest.input.layout,
    )?;
    validate_tensor(
        "output",
        &manifest.output.name,
        &manifest.output.shape,
        &manifest.output.dtype,
        &manifest.output.layout,
    )?;
    for binding in &manifest.output.bindings {
        validate_tensor(
            "output.bindings",
            &binding.name,
            &binding.shape,
            &binding.dtype,
            &binding.layout,
        )?;
    }
    if manifest.runtime.batch_size != 1 {
        return Err(error("runtime.batch_size must be 1"));
    }
    if !manifest.input.layout.eq_ignore_ascii_case("NCHW")
        || manifest.input.shape.len() != 4
        || manifest.input.shape[0] != 1
    {
        return Err(error(format!(
            "input {:?} {} must be batch-one NCHW with positive dimensions",
            manifest.input.shape, manifest.input.layout
        )));
    }
    if manifest.output.class_count == 0 {
        return Err(error("output.class_count must be positive"));
    }
    if manifest.postprocess.max_detections == 0 {
        return Err(error("postprocess.max_detections must be positive"));
    }
    Ok(())
}

fn validate_tensor(
    label: &str,
    name: &str,
    shape: &[u64],
    dtype: &str,
    layout: &str,
) -> Result<(), ModelRuntimeContractError> {
    require_nonempty(name, &format!("{label}.name"))?;
    require_nonempty(dtype, &format!("{label}.dtype"))?;
    require_nonempty(layout, &format!("{label}.layout"))?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(error(format!(
            "{label}.shape must contain positive dimensions"
        )));
    }
    Ok(())
}

fn validate_probability(field: &str, actual: f64) -> Result<(), ModelRuntimeContractError> {
    if !actual.is_finite() || !(0.0..=1.0).contains(&actual) {
        return Err(error(format!("manifest {field} must be in [0, 1]")));
    }
    Ok(())
}

fn require_nonempty<'a>(value: &'a str, field: &str) -> Result<&'a str, ModelRuntimeContractError> {
    let value = value.trim();
    if value.is_empty() {
        Err(error(format!("{field} must not be empty")))
    } else {
        Ok(value)
    }
}

fn error(message: impl Into<String>) -> ModelRuntimeContractError {
    ModelRuntimeContractError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manifest::{
        ManifestArtifact, ManifestInput, ManifestOutput, ManifestPostprocess, ManifestRuntime,
    };

    fn manifest() -> ModelManifest {
        let mut manifest = ModelManifest {
            schema_version: 1,
            model_id: "detector".to_owned(),
            display_name: "Detector".to_owned(),
            artifact: ManifestArtifact {
                engine_path: "detector.engine".to_owned(),
                sha256: "ab".repeat(32),
                size_bytes: 123,
            },
            runtime: ManifestRuntime {
                backend: "custom_tensorrt".to_owned(),
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
                scores_are_sigmoid: true,
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
                class_aware_nms: true,
                max_detections: 300,
            },
            validated: true,
            model_fingerprint: String::new(),
        };
        manifest.model_fingerprint = compute_model_fingerprint(&manifest).unwrap();
        manifest
    }

    #[test]
    fn accepts_a_complete_canonical_manifest() {
        validate_model_runtime_contract(&manifest(), true).unwrap();
    }

    #[test]
    fn rejects_semantic_drift_after_fingerprinting() {
        let mut manifest = manifest();
        manifest.output.has_objectness = true;
        assert!(validate_model_runtime_contract(&manifest, true).is_err());
    }
}
