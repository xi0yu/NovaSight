use std::sync::{Arc, RwLock};

use novasight_core::output::humanized_motion::{MotionProfile, MotionRuntimeParameters};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct MotionProfileHub {
    inner: Arc<RwLock<MotionProfileState>>,
    builtin: Arc<MotionProfile>,
    tuning: MotionRuntimeParameters,
}

#[derive(Clone, Debug)]
struct MotionProfileState {
    active: Option<Arc<MotionProfile>>,
    source: MotionProfileSource,
    revision: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionProfileSource {
    #[default]
    StartupConfig,
    RuntimeMemory,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionProfileStatus {
    pub enabled: bool,
    pub trajectory_source: String,
    pub active_profile: String,
    pub profile_name: String,
    pub sample_count: usize,
    pub profile_version: u32,
    pub spatial_curve_available: bool,
    pub effective_runtime_parameters: MotionRuntimeParameters,
    pub source: MotionProfileSource,
    pub revision: u64,
}

impl MotionProfileHub {
    pub fn new(
        builtin_fitts_a_ms: f64,
        builtin_fitts_b_ms: f64,
        builtin_side_ratio: f64,
        runtime_parameters: MotionRuntimeParameters,
    ) -> Self {
        let builtin = Arc::new(MotionProfile::builtin(
            builtin_fitts_a_ms,
            builtin_fitts_b_ms,
            builtin_side_ratio,
            runtime_parameters.clone(),
        ));
        Self {
            inner: Arc::new(RwLock::new(MotionProfileState {
                active: None,
                source: MotionProfileSource::StartupConfig,
                revision: 0,
            })),
            builtin,
            tuning: runtime_parameters,
        }
    }

    pub fn active(&self) -> Option<Arc<MotionProfile>> {
        self.inner
            .read()
            .unwrap_or_else(|value| value.into_inner())
            .active
            .clone()
    }

    pub fn activate(
        &self,
        mut profile: MotionProfile,
    ) -> Result<MotionProfileStatus, &'static str> {
        profile.validate()?;
        overlay_tuning(&mut profile.runtime_parameters, &self.tuning);
        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|value| value.into_inner());
        state.active = Some(Arc::new(profile));
        state.source = MotionProfileSource::RuntimeMemory;
        state.revision = state.revision.saturating_add(1);
        Ok(snapshot(&state))
    }

    pub fn activate_builtin(&self) -> MotionProfileStatus {
        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|value| value.into_inner());
        state.active = Some(Arc::clone(&self.builtin));
        state.source = MotionProfileSource::RuntimeMemory;
        state.revision = state.revision.saturating_add(1);
        snapshot(&state)
    }

    pub fn activate_startup(&self, mut profile: Option<MotionProfile>) -> Result<(), &'static str> {
        if let Some(profile) = &mut profile {
            profile.validate()?;
            overlay_tuning(&mut profile.runtime_parameters, &self.tuning);
        }
        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|value| value.into_inner());
        state.active = profile.map(Arc::new);
        state.source = MotionProfileSource::StartupConfig;
        state.revision = state.revision.saturating_add(1);
        Ok(())
    }

    pub fn activate_builtin_startup(&self) {
        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|value| value.into_inner());
        state.active = Some(Arc::clone(&self.builtin));
        state.source = MotionProfileSource::StartupConfig;
        state.revision = state.revision.saturating_add(1);
    }

    pub fn disable(&self) -> MotionProfileStatus {
        let mut state = self
            .inner
            .write()
            .unwrap_or_else(|value| value.into_inner());
        state.active = None;
        state.source = MotionProfileSource::RuntimeMemory;
        state.revision = state.revision.saturating_add(1);
        snapshot(&state)
    }

    pub fn status(&self) -> MotionProfileStatus {
        snapshot(&self.inner.read().unwrap_or_else(|value| value.into_inner()))
    }
}

fn overlay_tuning(target: &mut MotionRuntimeParameters, tuning: &MotionRuntimeParameters) {
    target.spatial_curve_enabled = tuning.spatial_curve_enabled;
    target.side_scale = tuning.side_scale;
    target.max_side_ratio = tuning.max_side_ratio;
    target.near_fade_start_px = tuning.near_fade_start_px;
    target.micro_bypass_px = tuning.micro_bypass_px;
    target.dynamic_rebase_ratio = tuning.dynamic_rebase_ratio;
    target.minimum_jerk_fallback = tuning.minimum_jerk_fallback;
    target.terminal_feedback_gain = tuning.terminal_feedback_gain;
}

impl Default for MotionProfileHub {
    fn default() -> Self {
        Self::new(35.0, 55.0, 0.012, MotionRuntimeParameters::default())
    }
}

fn snapshot(state: &MotionProfileState) -> MotionProfileStatus {
    let profile = state.active.as_deref();
    MotionProfileStatus {
        enabled: profile.is_some(),
        trajectory_source: profile
            .map_or("static", |value| {
                if value.profile_id == "builtin" {
                    "builtin"
                } else {
                    "trained"
                }
            })
            .to_owned(),
        active_profile: profile.map_or_else(String::new, |value| value.profile_id.clone()),
        profile_name: profile.map_or_else(String::new, |value| value.name.clone()),
        sample_count: profile.map_or(0, |value| value.sample_count),
        profile_version: profile.map_or(0, |value| value.profile_version),
        spatial_curve_available: profile.is_some(),
        effective_runtime_parameters: profile
            .map_or_else(MotionRuntimeParameters::default, |value| {
                value.runtime_parameters.clone()
            }),
        source: state.source,
        revision: state.revision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_is_immediately_visible_to_realtime_readers() {
        let hub = MotionProfileHub::default();
        assert!(!hub.status().enabled);
        let status = hub.activate_builtin();
        assert_eq!(status.active_profile, "builtin");
        assert_eq!(hub.active().unwrap().profile_id, "builtin");
        let disabled = hub.disable();
        assert!(!disabled.enabled);
        assert!(hub.active().is_none());
    }
}
