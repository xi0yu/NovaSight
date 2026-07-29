use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

/// Compute the canonical fingerprint shared by model ingestion, catalog
/// verification, and the live DeepStream composition path.
pub fn compute_model_fingerprint(manifest: &ModelManifest) -> Result<String, serde_json::Error> {
    manifest_fingerprint(manifest, true)
}

/// Verify one persisted fingerprint. Manifests written before `class_names`
/// became explicit may use the legacy payload only when that key was absent
/// from the source document.
pub fn model_fingerprint_matches(
    manifest: &ModelManifest,
    expected: &str,
    output_class_names_present: bool,
) -> Result<bool, serde_json::Error> {
    if manifest_fingerprint(manifest, true)? == expected {
        return Ok(true);
    }
    if output_class_names_present {
        return Ok(false);
    }
    Ok(manifest_fingerprint(manifest, false)? == expected)
}

fn manifest_fingerprint(
    manifest: &ModelManifest,
    include_class_names: bool,
) -> Result<String, serde_json::Error> {
    let mut output = serde_json::Map::new();
    output.insert("name".to_owned(), serde_json::json!(manifest.output.name));
    output.insert("shape".to_owned(), serde_json::json!(manifest.output.shape));
    output.insert("dtype".to_owned(), serde_json::json!(manifest.output.dtype));
    output.insert(
        "layout".to_owned(),
        serde_json::json!(manifest.output.layout),
    );
    output.insert(
        "format".to_owned(),
        serde_json::json!(manifest.output.format),
    );
    output.insert(
        "class_count".to_owned(),
        serde_json::json!(manifest.output.class_count),
    );
    output.insert(
        "has_objectness".to_owned(),
        serde_json::json!(manifest.output.has_objectness),
    );
    output.insert(
        "coordinate_mode".to_owned(),
        serde_json::json!(manifest.output.coordinate_mode),
    );
    if !manifest.output.bindings.is_empty() {
        output.insert(
            "bindings".to_owned(),
            serde_json::to_value(&manifest.output.bindings)?,
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
    if include_class_names {
        output.insert(
            "class_names".to_owned(),
            serde_json::json!(manifest.output.class_names),
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
    let stable = serde_json::to_string(&payload)?;
    let mut digest = Sha256::new();
    digest.update(python_json_numbers(&stable).as_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

/// Match Python's stable JSON float spelling at the two ryu differences used
/// by existing manifests: exponent padding and scientific notation below
/// `1e-4`. Only JSON number tokens outside strings are rewritten.
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
        let start = index;
        index += 1;
        while index < bytes.len()
            && (bytes[index].is_ascii_digit()
                || matches!(bytes[index], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            index += 1;
        }
        output.extend_from_slice(python_number_token(&bytes[start..index]).as_bytes());
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

    #[test]
    fn fingerprint_number_normalization_matches_existing_python_manifests() {
        assert_eq!(
            python_json_numbers(r#"{"value":1e-7,"text":"1e-7"}"#),
            r#"{"value":1e-07,"text":"1e-7"}"#
        );
        assert_eq!(python_number_token(b"0.00001234"), "1.234e-05");
    }
}
