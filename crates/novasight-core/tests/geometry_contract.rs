//! Phase 2 coordinate-geometry contract tests. These pin the behaviour of
//! the Rust `CoordinateTransform` against the captured regression
//! fixtures in `tests/fixtures/*.jsonl`; this test parses those files and
//! asserts that the Rust transform produces the same scale, the same
//! shifted-ROI point, and the same round-trip coordinates within the
//! per-field tolerance declared in the schema.

use std::path::PathBuf;

use novasight_core::geometry::{BBox, CoordinateTransform, Point, standard_transform};
use novasight_core::units::{
    CapturePixels, ControlCoordinate, ModelPixels, RoiCoordinate, RoiPixels, Scale,
};
use serde_json::Value;

const GEOMETRY_F64_ABS: f64 = 1e-9;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_jsonl(name: &str) -> Vec<Value> {
    let path = fixture_dir().join(name);
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("missing fixture {name}: {err}"));
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("valid JSON"))
        .collect()
}

fn transform_from_frame(frame: &Value) -> CoordinateTransform {
    CoordinateTransform::new(
        ModelPixels::new(frame["model_width"].as_f64().unwrap()).expect("model_width"),
        ModelPixels::new(frame["model_height"].as_f64().unwrap()).expect("model_height"),
        RoiCoordinate::new(frame["roi_x"].as_f64().unwrap()).expect("roi_x"),
        RoiCoordinate::new(frame["roi_y"].as_f64().unwrap()).expect("roi_y"),
        RoiPixels::new(frame["roi_width"].as_f64().unwrap()).expect("roi_width"),
        RoiPixels::new(frame["roi_height"].as_f64().unwrap()).expect("roi_height"),
        CapturePixels::new(frame["capture_width"].as_f64().unwrap()).expect("capture_width"),
        CapturePixels::new(frame["capture_height"].as_f64().unwrap()).expect("capture_height"),
        ControlCoordinate::new(frame["control_origin_x"].as_f64().unwrap())
            .expect("control_origin_x"),
        ControlCoordinate::new(frame["control_origin_y"].as_f64().unwrap())
            .expect("control_origin_y"),
        Scale::new(frame["display_scale_x"].as_f64().unwrap()).expect("display_scale_x"),
        Scale::new(frame["display_scale_y"].as_f64().unwrap()).expect("display_scale_y"),
    )
    .expect("transform")
}

#[test]
fn static_target_scales_match_python_reference() {
    let records = load_jsonl("static-target.jsonl");
    assert!(
        !records.is_empty(),
        "static-target.jsonl must contain records"
    );

    let transform = transform_from_frame(&records[0]["frame"]);
    let expected = &records[0]["transform"];
    let scale_x = transform.model_to_roi_scale_x();
    let scale_y = transform.model_to_roi_scale_y();
    let recorded_x = expected["model_to_roi_scale_x"].as_f64().unwrap();
    let recorded_y = expected["model_to_roi_scale_y"].as_f64().unwrap();
    assert!(
        (scale_x - recorded_x).abs() < GEOMETRY_F64_ABS,
        "scale_x mismatch: {scale_x} vs {recorded_x}"
    );
    assert!(
        (scale_y - recorded_y).abs() < GEOMETRY_F64_ABS,
        "scale_y mismatch: {scale_y} vs {recorded_y}"
    );
}

#[test]
fn moving_target_round_trip_is_identity() {
    let records = load_jsonl("moving-target.jsonl");
    assert!(
        !records.is_empty(),
        "moving-target.jsonl must contain records"
    );

    let transform = transform_from_frame(&records[0]["frame"]);
    let det = &records[0]["detections"][0];
    let box_model = BBox::from_xyxy(
        det["x"].as_f64().unwrap(),
        det["y"].as_f64().unwrap(),
        det["x"].as_f64().unwrap() + det["w"].as_f64().unwrap(),
        det["y"].as_f64().unwrap() + det["h"].as_f64().unwrap(),
    );
    let box_roi = transform.model_to_roi_box(box_model);
    let box_back = BBox::from_xyxy(
        box_roi.x1 / transform.model_to_roi_scale_x(),
        box_roi.y1 / transform.model_to_roi_scale_y(),
        box_roi.x2 / transform.model_to_roi_scale_x(),
        box_roi.y2 / transform.model_to_roi_scale_y(),
    );
    assert!((box_back.x1 - box_model.x1).abs() < GEOMETRY_F64_ABS);
    assert!((box_back.y1 - box_model.y1).abs() < GEOMETRY_F64_ABS);
    assert!((box_back.x2 - box_model.x2).abs() < GEOMETRY_F64_ABS);
    assert!((box_back.y2 - box_model.y2).abs() < GEOMETRY_F64_ABS);
}

#[test]
fn shifted_roi_translates_model_origin_to_capture_origin() {
    let transform = standard_transform(640.0, 320.0, 320.0).expect("transform");
    let origin = Point::new(0.0, 0.0);
    let shifted = transform.roi_to_capture_point(origin.x, origin.y);
    assert!((shifted.x - 0.0).abs() < GEOMETRY_F64_ABS);
    assert!((shifted.y - 0.0).abs() < GEOMETRY_F64_ABS);
    let model_centre = transform.model_to_roi_point(320.0, 320.0);
    let capture_centre = transform.roi_to_capture_point(model_centre.x, model_centre.y);
    assert!((capture_centre.x - 160.0).abs() < GEOMETRY_F64_ABS);
    assert!((capture_centre.y - 160.0).abs() < GEOMETRY_F64_ABS);
}

#[test]
fn integer_widths_above_f32_range_stay_exact_in_f64() {
    let transform = standard_transform(2_000_000.0, 2_000_000.0, 2_000_000.0).expect("transform");
    let box_model = BBox::from_xyxy(0.0, 0.0, 1_999_999.0, 1_999_999.0);
    let box_roi = transform.model_to_roi_box(box_model);
    assert!((box_roi.x1 - 0.0).abs() < GEOMETRY_F64_ABS);
    assert!((box_roi.x2 - 1_999_999.0).abs() < GEOMETRY_F64_ABS);
}

#[test]
fn invalid_dimensions_are_rejected_at_the_boundary() {
    assert!(ModelPixels::new(0.0).is_err());
    assert!(ModelPixels::new(f64::NAN).is_err());
    assert!(ModelPixels::new(f64::INFINITY).is_err());
    assert!(RoiPixels::new(-1.0).is_err());
    assert!(Scale::new(0.0).is_err());
    assert!(Scale::new(f64::NAN).is_err());
}
