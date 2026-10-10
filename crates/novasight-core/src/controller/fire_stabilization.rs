use serde::{Deserialize, Serialize};

/// Optional near-target correction while the physical left button is held.
/// The estimator lives in the device lane; these are its only user settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FireStabilizationConfig {
    pub enabled: bool,
    /// 0 disables learning; 1 permits the full bounded correction.
    pub strength: f64,
}

impl Default for FireStabilizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 0.5,
        }
    }
}

impl FireStabilizationConfig {
    pub fn is_valid(self) -> bool {
        self.strength.is_finite() && (0.0..=1.0).contains(&self.strength)
    }
}
