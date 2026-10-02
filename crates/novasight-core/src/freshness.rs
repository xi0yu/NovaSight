//! Freshness gate.
//!
//! Pure admission logic. The gate chooses the strictest positive
//! threshold from a set of inputs and returns a typed rejection reason
//! when a frame's age exceeds the threshold. It does not read any clock;
//! callers supply `now` from the typed `Clock` port. Rejection reasons
//! reset downstream targeting/control state through a typed
//! `FreshnessRejection` value.

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::units::Nanoseconds;

/// Maximum plausible frame age, in milliseconds, before the timestamp is
/// treated as synthetic replay or unit-test seam rather than monotonic.
const DEFAULT_MAX_PLAUSIBLE_AGE_MS: f64 = 3_600_000.0;
const PLAUSIBILITY_MULTIPLIER: f64 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreshnessPolicy {
    threshold_ms: f64,
    max_plausible_age_ms: f64,
}

impl FreshnessPolicy {
    pub fn new(threshold_ms: f64) -> Result<Self, AppError> {
        Self::with_max_plausible(threshold_ms, None)
    }

    pub fn with_max_plausible(
        threshold_ms: f64,
        max_plausible_age_ms: Option<f64>,
    ) -> Result<Self, AppError> {
        if !threshold_ms.is_finite() || threshold_ms < 0.0 {
            return Err(AppError::InvalidCoordinateSpace {
                width: 0,
                height: 0,
            });
        }
        let max_plausible = match max_plausible_age_ms {
            Some(value) if value.is_finite() && value >= 0.0 => value,
            _ => DEFAULT_MAX_PLAUSIBLE_AGE_MS.max(threshold_ms * PLAUSIBILITY_MULTIPLIER),
        };
        Ok(Self {
            threshold_ms,
            max_plausible_age_ms: max_plausible,
        })
    }

    pub fn strictest(thresholds_ms: &[f64]) -> Self {
        let mut chosen = 0.0_f64;
        for value in thresholds_ms {
            if value.is_finite() && *value > 0.0 && (*value < chosen || chosen == 0.0) {
                chosen = *value;
            }
        }
        Self {
            threshold_ms: chosen,
            max_plausible_age_ms: DEFAULT_MAX_PLAUSIBLE_AGE_MS
                .max(chosen * PLAUSIBILITY_MULTIPLIER),
        }
    }

    pub fn threshold_ms(&self) -> f64 {
        self.threshold_ms
    }

    pub fn evaluate(&self, capture: Nanoseconds, now: Nanoseconds) -> FreshnessOutcome {
        evaluate(self, capture.0, now.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessRejection {
    /// Age exceeded the threshold and the timestamp is plausibly monotonic.
    Stale {
        /// Truncated age in whole milliseconds, as captured by the Python
        /// reference (which formats ``age_ms`` as ``{:.3f}``). The
        /// comparison is always done on the full ``f64`` value first.
        age_ms: u64,
        threshold_ms: u64,
    },
    /// `now` preceded the capture timestamp. Callers should reset
    /// downstream state and treat the frame as out of the temporal domain.
    ClockRegression,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreshnessOutcome {
    pub admit: bool,
    pub age_ms: f64,
    pub rejection: Option<FreshnessRejection>,
}

impl FreshnessOutcome {
    pub const fn admit(age_ms: f64) -> Self {
        Self {
            admit: true,
            age_ms,
            rejection: None,
        }
    }

    pub const fn reject(rejection: FreshnessRejection, age_ms: f64) -> Self {
        Self {
            admit: false,
            age_ms,
            rejection: Some(rejection),
        }
    }
}

pub fn age_ms(capture_ts_ns: u64, now_ns: u64) -> f64 {
    now_ns.saturating_sub(capture_ts_ns) as f64 / 1_000_000.0
}

pub fn evaluate(policy: &FreshnessPolicy, capture_ts_ns: u64, now_ns: u64) -> FreshnessOutcome {
    if policy.threshold_ms <= 0.0 {
        return FreshnessOutcome::admit(age_ms(capture_ts_ns, now_ns));
    }
    if now_ns < capture_ts_ns {
        return FreshnessOutcome::reject(FreshnessRejection::ClockRegression, 0.0);
    }
    let age = age_ms(capture_ts_ns, now_ns);
    if !age.is_finite() {
        return FreshnessOutcome::admit(age);
    }
    if age > policy.max_plausible_age_ms {
        // Synthetic timestamps from replay and unit-test seams are not policed.
        return FreshnessOutcome::admit(age);
    }
    if age > policy.threshold_ms {
        return FreshnessOutcome::reject(
            FreshnessRejection::Stale {
                age_ms: age_as_u64(age),
                threshold_ms: age_as_u64(policy.threshold_ms),
            },
            age,
        );
    }
    FreshnessOutcome::admit(age)
}

fn age_as_u64(value: f64) -> u64 {
    if !value.is_finite() || value < 0.0 {
        0
    } else {
        value as u64
    }
}
