//! Phase 2 freshness contract tests. Pins the Rust `FreshnessPolicy`
//! against the captured `FreshnessGate` contract in
//! `freshness-reset.jsonl`; this test parses those records and
//! asserts the Rust gate produces the same decision for each case.

use std::path::PathBuf;

use novasight_core::freshness::{
    FreshnessOutcome, FreshnessPolicy, FreshnessRejection, age_ms, evaluate,
};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_records(name: &str) -> Vec<Value> {
    let path = fixture_dir().join(name);
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("missing fixture {name}: {err}"));
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("valid JSON"))
        .collect()
}

fn assert_matches_python(record: &Value) {
    let policy_json = &record["policy"];
    let policy = FreshnessPolicy::new(policy_json["threshold_ms"].as_f64().expect("threshold_ms"))
        .expect("policy");
    let frame = &record["frame"];
    let capture = frame["captured_at_ns"].as_u64().expect("captured_at_ns");
    let now = frame["control_now_ns"].as_u64().expect("control_now_ns");
    let expected = &record["outcome"];
    let actual = evaluate(&policy, capture, now);

    let expected_admit = expected["admit"].as_bool().expect("admit");
    assert_eq!(
        actual.admit,
        expected_admit,
        "admit flag for {label}",
        label = record["label"]
    );

    if !expected_admit {
        let expected_reason = expected["reason"].as_str().expect("reason");
        assert!(
            !expected_reason.is_empty(),
            "non-admit record must declare a reason for {label}",
            label = record["label"]
        );
        match actual.rejection.expect("rejection") {
            FreshnessRejection::Stale {
                age_ms,
                threshold_ms,
            } => {
                assert!(
                    age_ms >= threshold_ms,
                    "stale rejection must report age >= threshold (age={age_ms}, threshold={threshold_ms})"
                );
            }
            FreshnessRejection::ClockRegression => {}
        }
    }
}

#[test]
fn freshness_reset_fixture_matches_python_reference() {
    let records = load_records("freshness-reset.jsonl");
    assert!(
        !records.is_empty(),
        "freshness-reset.jsonl must contain records"
    );
    for record in &records {
        assert_matches_python(record);
    }
}

#[test]
fn strictest_constructor_picks_smallest_positive_threshold() {
    let policy = FreshnessPolicy::strictest(&[80.0, 55.0, 0.0, -1.0, f64::NAN]);
    assert_eq!(policy.threshold_ms(), 55.0);
}

#[test]
fn zero_threshold_admits_every_frame() {
    let policy = FreshnessPolicy::new(0.0).expect("policy");
    let outcome = evaluate(&policy, 1_000_000_000, 1_000_000_000 + 5_000_000_000);
    assert!(outcome.admit);
    assert!(outcome.rejection.is_none());
}

#[test]
fn exactly_at_threshold_admits() {
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    let outcome = evaluate(&policy, 1_000_000_000, 1_000_000_000 + 55_000_000);
    assert!(outcome.rejection.is_none());
}

#[test]
fn one_ns_over_threshold_rejects_as_stale() {
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    let outcome = evaluate(&policy, 1_000_000_000, 1_000_000_000 + 55_000_001);
    assert!(!outcome.admit);
    match outcome.rejection {
        Some(FreshnessRejection::Stale {
            age_ms,
            threshold_ms,
        }) => {
            assert!(age_ms >= 55);
            assert_eq!(threshold_ms, 55);
        }
        other => panic!("expected stale rejection, got {other:?}"),
    }
}

#[test]
fn clock_regression_is_a_typed_rejection() {
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    let outcome = evaluate(&policy, 1_000_000_000, 999_999_999);
    assert!(!outcome.admit);
    assert!(matches!(
        outcome.rejection,
        Some(FreshnessRejection::ClockRegression)
    ));
}

#[test]
fn age_ms_uses_saturating_subtraction() {
    assert!((age_ms(1_000_000_000, 999_999_999) - 0.0).abs() < 1e-9);
    assert!((age_ms(0, 0) - 0.0).abs() < 1e-9);
    assert!((age_ms(0, 1_000_000_000) - 1000.0).abs() < 1e-9);
}

#[test]
fn synthetic_timestamps_above_plausibility_window_are_not_policed() {
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    // 10 hours in nanoseconds exceeds the default plausibility window.
    let huge = 10 * 60 * 60 * 1_000_000_000_u64;
    let outcome = evaluate(&policy, 0, huge);
    assert!(outcome.admit);
}

#[test]
fn property_age_is_never_negative() {
    let policy = FreshnessPolicy::new(55.0).expect("policy");
    for delta in [0_u64, 1, 1_000, 1_000_000, 1_000_000_000] {
        let outcome = evaluate(&policy, 1_000_000_000, 1_000_000_000 + delta);
        assert!(outcome.age_ms >= 0.0);
    }
}

#[test]
fn property_strictest_prefers_smallest_positive_value() {
    for (inputs, expected) in [
        (vec![5.0, 10.0, 15.0], 5.0),
        (vec![15.0, 5.0, 10.0], 5.0),
        (vec![10.0, 10.0, 10.0], 10.0),
        (vec![0.0, 0.0, 0.0], 0.0),
        (vec![-5.0, 0.0, 5.0], 5.0),
    ] {
        let policy = FreshnessPolicy::strictest(&inputs);
        assert_eq!(policy.threshold_ms(), expected);
    }
}

#[test]
fn typed_rejection_serializes_to_stable_string_keys() {
    let rejection = FreshnessRejection::Stale {
        age_ms: 100,
        threshold_ms: 55,
    };
    let json = serde_json::to_value(rejection).expect("serialize");
    assert_eq!(json["Stale"]["age_ms"], 100);
    assert_eq!(json["Stale"]["threshold_ms"], 55);
}

#[test]
fn outcome_admit_helper_keeps_rejection_none() {
    let outcome = FreshnessOutcome::admit(12.0);
    assert!(outcome.admit);
    assert!(outcome.rejection.is_none());
}
