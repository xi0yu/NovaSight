use novasight_jetson_preprocess::{TensorContract, TensorDtype};
use novasight_store::model_manifest::ModelManifest;
use novasight_tensorrt::{DecodeContract, DetectionDecoder};
use thiserror::Error;

#[derive(Clone, Debug)]
pub(crate) struct RustTensorRtContract {
    pub(crate) input: TensorContract,
    pub(crate) decoder: DetectionDecoder,
}

pub(crate) fn resolve_rust_tensorrt_contract(
    manifest: &ModelManifest,
    confidence_threshold: f64,
    nms_threshold: f64,
) -> Result<RustTensorRtContract, ModelContractError> {
    if !manifest
        .runtime
        .backend
        .eq_ignore_ascii_case("custom_tensorrt")
    {
        return Err(ModelContractError::Invalid(format!(
            "Rust TensorRT requires runtime.backend=custom_tensorrt, got {}",
            manifest.runtime.backend
        )));
    }
    if manifest.input.shape.len() != 4
        || manifest.input.shape[0] != 1
        || manifest.input.shape[1] != 3
        || !manifest.input.layout.eq_ignore_ascii_case("NCHW")
        || !manifest.input.color_format.eq_ignore_ascii_case("RGB")
        || manifest.input.maintain_aspect_ratio
        || manifest.input.symmetric_padding
        || !manifest.input.scale_factor.is_finite()
        || (manifest.input.scale_factor - 1.0 / 255.0).abs() > 1e-12
    {
        return Err(ModelContractError::Invalid(
            "Rust CUDA preprocess requires batch-one RGB NCHW, direct resize, and 1/255 scale"
                .to_owned(),
        ));
    }
    let dtype = match manifest.input.dtype.trim().to_ascii_lowercase().as_str() {
        "float16" | "fp16" => TensorDtype::Float16,
        "float32" | "fp32" => TensorDtype::Float32,
        value => {
            return Err(ModelContractError::Invalid(format!(
                "Rust CUDA preprocess does not support input dtype {value}"
            )));
        }
    };
    let height = u32::try_from(manifest.input.shape[2])
        .map_err(|_| ModelContractError::Invalid("input height exceeds CUDA limits".to_owned()))?;
    let width = u32::try_from(manifest.input.shape[3])
        .map_err(|_| ModelContractError::Invalid("input width exceeds CUDA limits".to_owned()))?;
    let input = TensorContract::rgb_nchw(height, width, dtype)
        .map_err(|error| ModelContractError::Invalid(error.to_string()))?;
    if !manifest.output.scores_are_sigmoid {
        return Err(ModelContractError::Invalid(
            "Rust TensorRT decoder requires probability scores; logits are unsupported".to_owned(),
        ));
    }
    let parser = manifest.postprocess.parser.trim().to_ascii_lowercase();
    let coordinate_mode = manifest.output.coordinate_mode.trim().to_ascii_lowercase();
    let decoder_contract = match parser.as_str() {
        "yolo" => {
            if coordinate_mode != "pixel" {
                return Err(ModelContractError::Invalid(
                    "raw YOLO Rust decoder requires pixel coordinates".to_owned(),
                ));
            }
            if !manifest.postprocess.class_aware_nms {
                return Err(ModelContractError::Invalid(
                    "raw YOLO Rust decoder requires class-aware NMS".to_owned(),
                ));
            }
            DecodeContract::RawYolo {
                output_name: manifest.output.name.clone(),
                class_count: manifest.output.class_count,
                has_objectness: manifest.output.has_objectness,
            }
        }
        "decoded_nms" => DecodeContract::DecodedBoxes6 {
            output_name: manifest.output.name.clone(),
            class_count: manifest.output.class_count,
            normalized_coordinates: match coordinate_mode.as_str() {
                "normalized" => true,
                "pixel" => false,
                _ => {
                    return Err(ModelContractError::Invalid(format!(
                        "decoded boxes coordinate mode {coordinate_mode} is unsupported"
                    )));
                }
            },
        },
        "efficientnms" | "rockchip_yolov5" => {
            return Err(ModelContractError::Invalid(format!(
                "Rust TensorRT decoder does not yet support {parser}; select deepstream_nvinfer"
            )));
        }
        _ => {
            return Err(ModelContractError::Invalid(format!(
                "unsupported parser {parser}"
            )));
        }
    };
    let max_detections = usize::try_from(manifest.postprocess.max_detections).map_err(|_| {
        ModelContractError::Invalid("postprocess.max_detections exceeds platform limits".to_owned())
    })?;
    let decoder = DetectionDecoder::new(
        decoder_contract,
        confidence_threshold as f32,
        nms_threshold as f32,
        max_detections,
        width,
        height,
    )
    .map_err(|error| ModelContractError::Invalid(error.to_string()))?;
    Ok(RustTensorRtContract { input, decoder })
}

#[derive(Debug, Error)]
pub(crate) enum ModelContractError {
    #[error("invalid Rust TensorRT model contract: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use novasight_store::model_manifest::{
        ManifestArtifact, ManifestInput, ManifestOutput, ManifestPostprocess, ManifestRuntime,
    };

    use super::*;

    fn manifest() -> ModelManifest {
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
        }
    }

    #[test]
    fn reuses_validated_manifest_semantics() {
        let contract = resolve_rust_tensorrt_contract(&manifest(), 0.25, 0.45).unwrap();
        assert_eq!(contract.input.shape(), [1, 3, 640, 640]);
        assert_eq!(contract.input.dtype(), TensorDtype::Float32);
        assert!(format!("{:?}", contract.decoder).contains("DetectionDecoder"));
    }

    #[test]
    fn fails_closed_for_unsupported_parser() {
        let mut manifest = manifest();
        manifest.postprocess.parser = "efficientnms".to_owned();
        let error = resolve_rust_tensorrt_contract(&manifest, 0.25, 0.45).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not yet support efficientnms")
        );
    }

    #[test]
    fn rejects_preprocess_semantic_drift() {
        let mut manifest = manifest();
        manifest.input.maintain_aspect_ratio = true;
        let error = resolve_rust_tensorrt_contract(&manifest, 0.25, 0.45).unwrap_err();
        assert!(error.to_string().contains("direct resize"));
    }
}
