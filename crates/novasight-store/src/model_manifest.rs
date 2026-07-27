use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub model_id: String,
    pub display_name: String,
    pub artifact: ManifestArtifact,
    #[serde(default)]
    pub runtime: ManifestRuntime,
    pub input: ManifestInput,
    pub output: ManifestOutput,
    #[serde(default)]
    pub postprocess: ManifestPostprocess,
    #[serde(default)]
    pub validated: bool,
    #[serde(default)]
    pub model_fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestArtifact {
    pub engine_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestRuntime {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
}

impl Default for ManifestRuntime {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            precision: default_precision(),
            batch_size: default_batch_size(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    #[serde(default = "default_layout")]
    pub layout: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestInput {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    #[serde(default = "default_layout")]
    pub layout: String,
    #[serde(default = "default_color_format")]
    pub color_format: String,
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,
    #[serde(default)]
    pub maintain_aspect_ratio: bool,
    #[serde(default)]
    pub symmetric_padding: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestOutput {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    #[serde(default = "default_layout")]
    pub layout: String,
    #[serde(default = "default_output_format")]
    pub format: String,
    #[serde(default)]
    pub class_count: u32,
    #[serde(default)]
    pub class_names: Vec<String>,
    #[serde(default)]
    pub has_objectness: bool,
    #[serde(default = "default_true")]
    pub scores_are_sigmoid: bool,
    #[serde(default = "default_coordinate_mode")]
    pub coordinate_mode: String,
    #[serde(default)]
    pub bindings: Vec<ManifestTensor>,
    #[serde(default)]
    pub strides: Vec<u32>,
    #[serde(default)]
    pub anchors: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestPostprocess {
    #[serde(default = "default_parser")]
    pub parser: String,
    #[serde(default = "default_parser_preset")]
    pub parser_preset: String,
    #[serde(default = "default_confidence")]
    pub confidence_threshold: f64,
    #[serde(default = "default_nms")]
    pub nms_iou_threshold: f64,
    #[serde(default = "default_true")]
    pub class_aware_nms: bool,
    #[serde(default = "default_max_detections")]
    pub max_detections: u32,
}

impl Default for ManifestPostprocess {
    fn default() -> Self {
        Self {
            parser: default_parser(),
            parser_preset: default_parser_preset(),
            confidence_threshold: default_confidence(),
            nms_iou_threshold: default_nms(),
            class_aware_nms: true,
            max_detections: default_max_detections(),
        }
    }
}

const fn default_schema_version() -> u32 {
    1
}
fn default_backend() -> String {
    "custom_tensorrt".to_owned()
}
fn default_precision() -> String {
    "fp16".to_owned()
}
const fn default_batch_size() -> u32 {
    1
}
fn default_layout() -> String {
    "NCHW".to_owned()
}
fn default_color_format() -> String {
    "RGB".to_owned()
}
const fn default_scale_factor() -> f64 {
    1.0 / 255.0
}
fn default_output_format() -> String {
    "yolo_cxcywh_class_scores".to_owned()
}
const fn default_true() -> bool {
    true
}
fn default_coordinate_mode() -> String {
    "pixel".to_owned()
}
fn default_parser() -> String {
    "yolo".to_owned()
}
fn default_parser_preset() -> String {
    "auto".to_owned()
}
const fn default_confidence() -> f64 {
    0.25
}
const fn default_nms() -> f64 {
    0.45
}
const fn default_max_detections() -> u32 {
    256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_matches_runtime_defaults() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{
                "model_id":"detector",
                "display_name":"Detector",
                "artifact":{"engine_path":"detector.engine","sha256":"abc","size_bytes":3},
                "input":{"name":"images","shape":[1,3,640,640],"dtype":"float32"},
                "output":{"name":"output0","shape":[1,84,8400],"dtype":"float32"}
            }"#,
        )
        .expect("valid runtime manifest");

        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.runtime.backend, "custom_tensorrt");
        assert_eq!(manifest.runtime.precision, "fp16");
        assert_eq!(manifest.input.layout, "NCHW");
        assert_eq!(manifest.output.layout, "NCHW");
        assert!(manifest.output.scores_are_sigmoid);
        assert_eq!(manifest.postprocess.max_detections, 256);
    }

    #[test]
    fn decoder_rejects_unknown_nested_fields() {
        let result = serde_json::from_str::<ModelManifest>(
            r#"{
                "model_id":"detector",
                "display_name":"Detector",
                "artifact":{"engine_path":"detector.engine","sha256":"abc","size_bytes":3},
                "runtime":{"unsupported":true},
                "input":{"name":"images","shape":[1,3,640,640],"dtype":"float32"},
                "output":{"name":"output0","shape":[1,84,8400],"dtype":"float32"}
            }"#,
        );

        assert!(result.is_err());
    }
}
