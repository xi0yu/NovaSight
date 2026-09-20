use novasight_core::MAX_DETECTIONS;
use novasight_store::model_manifest::ModelManifest;
use novasight_tensorrt::gpu::{GpuFrameConfig, GpuModelConfig};
use thiserror::Error;

/// Admit only contracts implemented by the CUDA path. Never select a CPU parser.
pub fn gpu_model_config(
    manifest: &ModelManifest,
    engine: std::path::PathBuf,
    confidence_threshold: f64,
    nms_threshold: f64,
) -> Result<GpuModelConfig, DeepStreamModelContractError> {
    validate_probability("confidence_threshold", confidence_threshold)?;
    validate_probability("nms_threshold", nms_threshold)?;
    if !manifest.postprocess.parser.eq_ignore_ascii_case("yolo") {
        return Err(gpu_contract_error(
            "postprocess.parser",
            &manifest.postprocess.parser,
            "yolo (single raw output)",
        ));
    }
    for (field, actual, expected) in [
        (
            "output.scores_are_sigmoid",
            manifest.output.scores_are_sigmoid,
            true,
        ),
        (
            "postprocess.class_aware_nms",
            manifest.postprocess.class_aware_nms,
            true,
        ),
        (
            "input.maintain_aspect_ratio",
            manifest.input.maintain_aspect_ratio,
            false,
        ),
    ] {
        if actual != expected {
            return Err(gpu_contract_error(field, actual, expected));
        }
    }
    if !manifest.output.bindings.is_empty() {
        return Err(gpu_contract_error(
            "output.bindings.len",
            manifest.output.bindings.len(),
            0,
        ));
    }
    let [1, 3, height, width] = manifest.input.shape.as_slice() else {
        return Err(gpu_contract_error(
            "input.shape",
            format!("{:?}", manifest.input.shape),
            "[1,3,H,W]",
        ));
    };
    if !manifest.input.layout.eq_ignore_ascii_case("NCHW") {
        return Err(gpu_contract_error(
            "input.layout",
            &manifest.input.layout,
            "NCHW",
        ));
    }
    for (field, actual, maximum) in [
        ("input.width", *width, 16384),
        ("input.height", *height, 16384),
        (
            "runtime.batch_size",
            u64::from(manifest.runtime.batch_size),
            1,
        ),
        (
            "output.class_count",
            u64::from(manifest.output.class_count),
            1024,
        ),
    ] {
        if actual == 0 || actual > maximum {
            return Err(gpu_contract_error(field, actual, format!("1..={maximum}")));
        }
    }
    // Reject unsupported parsers before entering their unrelated shape logic.
    deepstream_parser_contract(manifest).map_err(|cause| {
        error(format!(
            "GPU model contract: {cause}; output.shape={:?}, class_count={}, has_objectness={}; CPU fallback forbidden",
            manifest.output.shape, manifest.output.class_count, manifest.output.has_objectness,
        ))
    })?;
    let shape = &manifest.output.shape[manifest.output.shape.len() - 2..];
    let channels =
        u64::from(manifest.output.class_count) + if manifest.output.has_objectness { 5 } else { 4 };
    let channels_first = shape[0] == channels;
    let candidates = shape[usize::from(channels_first)];
    if candidates == 0 || candidates > 32768 {
        return Err(gpu_contract_error(
            "output.candidates",
            candidates,
            "1..=32768",
        ));
    }
    if manifest.postprocess.max_detections == 0 {
        return Err(gpu_contract_error(
            "postprocess.max_detections",
            0,
            "positive per-class Top-K (capped at 256)",
        ));
    }
    let dtype = |field: &str, value: &str| match value.to_ascii_lowercase().as_str() {
        "float32" | "fp32" => Ok(1),
        "float16" | "fp16" => Ok(2),
        _ => Err(gpu_contract_error(field, value, "FP32 or FP16")),
    };
    let bgr = match color_format(manifest)? {
        0 => 0,
        1 => 1,
        _ => return Err(error("CUDA preprocessing requires RGB/BGR")),
    };
    let input_dtype = dtype("input.dtype", &manifest.input.dtype)?;
    let scale = manifest.input.scale_factor as f32;
    let maximum_pixel = 255.0 * scale;
    if !scale.is_finite()
        || scale <= 0.0
        || !maximum_pixel.is_finite()
        || (input_dtype == 2 && maximum_pixel > 65504.0)
    {
        return Err(gpu_contract_error(
            "input.scale_factor",
            manifest.input.scale_factor,
            "positive scale with every uint8 pixel representable in the input dtype",
        ));
    }
    Ok(GpuModelConfig {
        engine,
        input_name: manifest.input.name.clone(),
        output_name: manifest.output.name.clone(),
        frame: GpuFrameConfig {
            abi_version: 1,
            width: *width as u32,
            height: *height as u32,
            candidates: candidates as u32,
            classes: manifest.output.class_count,
            channels_first: u32::from(channels_first),
            has_objectness: u32::from(manifest.output.has_objectness),
            input_dtype,
            output_dtype: dtype("output.dtype", &manifest.output.dtype)?,
            bgr,
            top_k: manifest
                .postprocess
                .max_detections
                .min(MAX_DETECTIONS as u32),
            cuda_graph: 1,
            scale,
            confidence_threshold: confidence_threshold as f32,
            nms_threshold: nms_threshold as f32,
        },
    })
}

fn gpu_contract_error(
    field: &str,
    actual: impl std::fmt::Display,
    expected: impl std::fmt::Display,
) -> DeepStreamModelContractError {
    error(format!(
        "GPU model contract: {field}={actual}; expected {expected}; CPU fallback forbidden"
    ))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct DeepStreamModelContractError {
    message: String,
}

impl DeepStreamModelContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepStreamParserContract {
    pub function: &'static str,
    pub cluster_mode: i64,
}

/// Render nvinfer from one typed adapter contract. The composition root owns
/// path canonicalization and file I/O; this module owns DeepStream semantics.
pub fn render_deepstream_nvinfer_config(
    manifest: &ModelManifest,
    engine_path: &str,
    parser_library_path: &str,
    component_id: i32,
    confidence_threshold: f64,
    nms_threshold: f64,
) -> Result<String, DeepStreamModelContractError> {
    validate_probability("confidence_threshold", confidence_threshold)?;
    validate_probability("nms_threshold", nms_threshold)?;
    let contract = DeepStreamNvinferContract::from_manifest(manifest)?;
    Ok(contract.render(
        manifest,
        engine_path,
        parser_library_path,
        component_id,
        confidence_threshold,
        nms_threshold,
    ))
}

pub fn deepstream_parser_contract(
    manifest: &ModelManifest,
) -> Result<DeepStreamParserContract, DeepStreamModelContractError> {
    let parser = manifest.postprocess.parser.trim().to_ascii_lowercase();
    let format = manifest.output.format.trim().to_ascii_lowercase();
    let preset = normalize_parser_preset(&manifest.postprocess.parser_preset)?;
    match parser.as_str() {
        "decoded_nms" => {
            let elements = manifest.output.shape.iter().try_fold(1_u64, |size, &dim| {
                if dim == 0 {
                    None
                } else {
                    size.checked_mul(dim)
                }
            });
            if format != "decoded_boxes6"
                || manifest.output.shape.last().copied() != Some(6)
                || elements.is_none()
            {
                return Err(error(
                    "decoded_nms requires a positive output shape with no u64 overflow whose last dimension is 6",
                ));
            }
            Ok(DeepStreamParserContract {
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
                return Err(error("invalid Rockchip YOLOv5 parser contract"));
            }
            Ok(DeepStreamParserContract {
                function: "NvDsInferParseNovaSightRockchipYoloV5",
                cluster_mode: 2,
            })
        }
        "efficientnms" => {
            if format != "efficientnms_boxes_scores_classes" || manifest.output.bindings.len() != 4
            {
                return Err(error("invalid EfficientNMS parser contract"));
            }
            Ok(DeepStreamParserContract {
                function: "NvDsInferParseNovaSightEfficientNms",
                cluster_mode: 4,
            })
        }
        "yolo" => raw_yolo_parser_contract(manifest, &format, preset),
        _ => Err(error(format!("unsupported parser {parser}"))),
    }
}

#[derive(Clone, Debug)]
struct DeepStreamNvinferContract {
    parser: DeepStreamParserContract,
    network_mode: i64,
    color_format: i64,
    output_names: String,
    max_detections: u32,
}

impl DeepStreamNvinferContract {
    fn from_manifest(manifest: &ModelManifest) -> Result<Self, DeepStreamModelContractError> {
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
        if output_names.is_empty() || output_names.contains(['\r', '\n']) {
            return Err(error("output binding names are not safe for nvinfer"));
        }
        Ok(Self {
            parser: deepstream_parser_contract(manifest)?,
            network_mode: network_mode(manifest)?,
            color_format: color_format(manifest)?,
            output_names,
            max_detections: manifest
                .postprocess
                .max_detections
                .min(MAX_DETECTIONS as u32),
        })
    }

    fn render(
        &self,
        manifest: &ModelManifest,
        engine_path: &str,
        parser_library_path: &str,
        component_id: i32,
        confidence_threshold: f64,
        nms_threshold: f64,
    ) -> String {
        format!(
            "# Generated by NovaSight Rust. Do not hand-edit.\n# novasight-model-fingerprint={}\n[property]\ngpu-id=0\nmodel-engine-file={}\nbatch-size=1\nnetwork-mode={}\nnetwork-type=0\nprocess-mode=1\ngie-unique-id={}\ninterval=0\nnum-detected-classes={}\nnet-scale-factor={:.17}\nmodel-color-format={}\nmaintain-aspect-ratio={}\nsymmetric-padding={}\noutput-tensor-meta=0\noutput-blob-names={}\ncustom-lib-path={}\nparse-bbox-func-name={}\ncluster-mode={}\n\n[class-attrs-all]\npre-cluster-threshold={:.8}\nnms-iou-threshold={:.8}\ntopk={}\n",
            manifest.model_fingerprint,
            engine_path,
            self.network_mode,
            component_id,
            manifest.output.class_count,
            manifest.input.scale_factor,
            self.color_format,
            i32::from(manifest.input.maintain_aspect_ratio),
            i32::from(manifest.input.symmetric_padding),
            self.output_names,
            parser_library_path,
            self.parser.function,
            self.parser.cluster_mode,
            confidence_threshold,
            nms_threshold,
            self.max_detections,
        )
    }
}

fn raw_yolo_parser_contract(
    manifest: &ModelManifest,
    format: &str,
    preset: &str,
) -> Result<DeepStreamParserContract, DeepStreamModelContractError> {
    if format != "yolo_cxcywh_class_scores"
        || !manifest
            .output
            .coordinate_mode
            .eq_ignore_ascii_case("pixel")
    {
        return Err(error("invalid raw YOLO output contract"));
    }
    let shape = match manifest.output.shape.as_slice() {
        [1, left, right] => [*left, *right],
        [left, right] => [*left, *right],
        _ => return Err(error("YOLO output must be [C,N], [N,C], or [1,C,N]")),
    };
    let preset_objectness = match preset {
        "yolov5" => Some(true),
        "yolov8" | "yolo11" => Some(false),
        _ => None,
    };
    if preset_objectness.is_some_and(|value| value != manifest.output.has_objectness) {
        return Err(error("parser preset conflicts with output objectness"));
    }
    let has_objectness = preset_objectness.unwrap_or(manifest.output.has_objectness);
    let channels = u64::from(manifest.output.class_count) + if has_objectness { 5 } else { 4 };
    if !shape.contains(&channels) || shape.iter().max().copied().unwrap_or(0) <= channels {
        return Err(error(
            "YOLO output shape does not match class/objectness contract",
        ));
    }
    Ok(DeepStreamParserContract {
        function: "NvDsInferParseNovaSightRaw",
        cluster_mode: 2,
    })
}

fn network_mode(manifest: &ModelManifest) -> Result<i64, DeepStreamModelContractError> {
    match manifest
        .runtime
        .precision
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "fp32" => Ok(0),
        "int8" => Ok(1),
        "fp16" => Ok(2),
        value => Err(error(format!("unsupported precision {value}"))),
    }
}

fn color_format(manifest: &ModelManifest) -> Result<i64, DeepStreamModelContractError> {
    match manifest
        .input
        .color_format
        .trim()
        .to_ascii_uppercase()
        .as_str()
    {
        "RGB" => Ok(0),
        "BGR" => Ok(1),
        "GRAY" | "GREY" => Ok(2),
        value => Err(error(format!("unsupported color format {value}"))),
    }
}

fn normalize_parser_preset(value: &str) -> Result<&str, DeepStreamModelContractError> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    let canonical = match normalized.as_str() {
        "" | "automatic" => "auto",
        "yolo_v5" | "yolov5_raw" => "yolov5",
        "yolo_v8" | "yolov8_raw" => "yolov8",
        "yolo_11" | "yolo11_raw" => "yolo11",
        "generic" | "custom" | "novasight" => "novasight_generic",
        "auto" | "yolov5" | "yolov8" | "yolo11" | "novasight_generic" => normalized.as_str(),
        _ => return Err(error(format!("unsupported parser preset {value}"))),
    };
    Ok(match canonical {
        "auto" => "auto",
        "yolov5" => "yolov5",
        "yolov8" => "yolov8",
        "yolo11" => "yolo11",
        _ => "novasight_generic",
    })
}

fn validate_probability(field: &str, value: f64) -> Result<(), DeepStreamModelContractError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(error(format!("{field} must be in [0, 1]")))
    }
}

fn error(message: impl Into<String>) -> DeepStreamModelContractError {
    DeepStreamModelContractError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use novasight_store::model_manifest::{
        ManifestArtifact, ManifestInput, ManifestOutput, ManifestPostprocess, ManifestRuntime,
    };

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
                class_names: Vec::new(),
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
            model_fingerprint: "fingerprint".to_owned(),
        }
    }

    #[test]
    fn renders_one_typed_nvinfer_contract_without_runtime_reparse() {
        let source = render_deepstream_nvinfer_config(
            &manifest(),
            "/models/detector.engine",
            "/lib/libparser.so",
            1,
            0.30,
            0.50,
        )
        .unwrap();

        assert!(source.contains("topk=256"));
        assert!(source.contains("parse-bbox-func-name=NvDsInferParseNovaSightRaw"));
    }

    #[test]
    fn rejects_parser_semantic_drift() {
        let mut manifest = manifest();
        manifest.output.has_objectness = true;
        assert!(deepstream_parser_contract(&manifest).is_err());
    }

    #[test]
    fn gpu_admission_preserves_contract_and_rejects_cpu_only_modes() {
        let mut m = manifest();
        let c = gpu_model_config(&m, "detector.engine".into(), 0.65, 0.45).unwrap();
        assert_eq!(
            (c.frame.channels_first, c.frame.candidates, c.frame.classes),
            (1, 8400, 80)
        );
        assert_eq!(c.frame.confidence_threshold, 0.65);
        m.output.shape = vec![1, 8400, 84];
        assert_eq!(
            gpu_model_config(&m, "x".into(), 0.65, 0.45)
                .unwrap()
                .frame
                .channels_first,
            0
        );
        m.input.maintain_aspect_ratio = true;
        assert!(gpu_model_config(&m, "x".into(), 0.65, 0.45).is_err());
        m.input.maintain_aspect_ratio = false;
        m.output.scores_are_sigmoid = false;
        assert!(gpu_model_config(&m, "x".into(), 0.65, 0.45).is_err());
    }

    #[test]
    fn gpu_normalization_cannot_overflow_the_input_tensor() {
        let mut m = manifest();
        for (dtype, safe, overflow) in [
            ("float16", 1.0, 300.0),
            ("float32", 1.0 / 255.0, f64::from(f32::MAX)),
        ] {
            m.input.dtype = dtype.into();
            m.input.scale_factor = safe;
            assert!(gpu_model_config(&m, "x".into(), 0.65, 0.45).is_ok());
            m.input.scale_factor = overflow;
            let error = gpu_model_config(&m, "x".into(), 0.65, 0.45).unwrap_err();
            assert!(error.to_string().contains("input.scale_factor"));
        }
    }

    #[test]
    fn gpu_boundary_errors_identify_the_field_and_keep_existing_limits() {
        let mut m = manifest();
        let detail = |m: &ModelManifest| {
            gpu_model_config(m, "x".into(), 0.65, 0.45)
                .unwrap_err()
                .to_string()
        };
        m.input.maintain_aspect_ratio = true;
        assert!(detail(&m).contains("input.maintain_aspect_ratio=true; expected false"));
        m.input.maintain_aspect_ratio = false;
        m.output.shape = vec![1, 84, 32769];
        assert!(detail(&m).contains("output.candidates=32769; expected 1..=32768"));
        m.output.shape[2] = 32768;
        let config = gpu_model_config(&m, "x".into(), 0.65, 0.45).unwrap();
        assert_eq!(config.frame.candidates, 32768);
        assert_eq!(config.frame.top_k, 256); // Existing per-class clamp, not global Top-K.
        m.output.dtype = "int8".into();
        assert!(detail(&m).contains("output.dtype=int8; expected FP32 or FP16"));
        m.output.dtype = "float32".into();
        m.output.shape = vec![1, 7, 8400];
        assert!(detail(&m).contains("output.shape=[1, 7, 8400], class_count=80"));
        // An unsupported parser must fail before unrelated shape arithmetic can panic.
        m.postprocess.parser = "decoded_nms".into();
        m.output.format = "decoded_boxes6".into();
        m.output.shape = vec![u64::MAX, 6];
        assert!(detail(&m).contains("postprocess.parser=decoded_nms"));
        assert!(deepstream_parser_contract(&m).is_err());
        m.output.shape = vec![1, 0, 6];
        assert!(deepstream_parser_contract(&m).is_err());
        m.output.shape = vec![1, 256, 6];
        assert!(deepstream_parser_contract(&m).is_ok());
    }
}
