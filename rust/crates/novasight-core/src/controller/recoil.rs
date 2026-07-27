//! Target-relative recoil controller and tracking/recoil mixer.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecoilConfig {
    pub enabled: bool,
    pub base_rate_counts_s: f64,
    pub max_rate_counts_s: f64,
    pub startup_ms: f64,
    pub positive_deadzone_norm: f64,
    pub negative_deadzone_norm: f64,
    pub full_brake_error_norm: f64,
    pub fast_add_gain_counts_s: f64,
    pub max_fast_add_ratio: f64,
    pub stale_threshold_ms: f64,
}

impl Default for RecoilConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_rate_counts_s: 0.0,
            max_rate_counts_s: 0.0,
            startup_ms: 35.0,
            positive_deadzone_norm: 0.04,
            negative_deadzone_norm: 0.04,
            full_brake_error_norm: 0.12,
            fast_add_gain_counts_s: 0.0,
            max_fast_add_ratio: 0.30,
            stale_threshold_ms: 55.0,
        }
    }
}

impl RecoilConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        for value in [
            self.base_rate_counts_s,
            self.max_rate_counts_s,
            self.startup_ms,
            self.positive_deadzone_norm,
            self.negative_deadzone_norm,
            self.full_brake_error_norm,
            self.fast_add_gain_counts_s,
            self.max_fast_add_ratio,
            self.stale_threshold_ms,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err("recoil values must be finite and non-negative");
            }
        }
        if self.max_rate_counts_s < self.base_rate_counts_s {
            return Err("recoil max rate must be at least the base rate");
        }
        for value in [
            self.positive_deadzone_norm,
            self.negative_deadzone_norm,
            self.full_brake_error_norm,
            self.max_fast_add_ratio,
        ] {
            if value > 1.0 {
                return Err("normalized recoil values must not exceed one");
            }
        }
        if self.full_brake_error_norm <= self.negative_deadzone_norm {
            return Err("recoil full brake threshold must exceed the negative deadzone");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoilState {
    #[default]
    Idle,
    Startup,
    Active,
    Hold,
    Brake,
    Stale,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoilBlockReason {
    #[default]
    RecoilDisabled,
    FiringInactive,
    DtInvalid,
    ObservationAgeInvalid,
    TargetStale,
    TargetInvalid,
    ErrorInvalid,
    PositionBrake,
    RecoilRateZero,
    #[serde(rename = "")]
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoilInput {
    pub firing: bool,
    pub now_ns: u64,
    pub dt_s: f64,
    pub target_valid: bool,
    pub target_id: Option<u64>,
    pub source_generation: Option<u64>,
    pub observation_age_ms: f64,
    pub error_y_norm: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecoilDecision {
    pub state: RecoilState,
    pub base_rate_counts_s: f64,
    pub fast_add_rate_counts_s: f64,
    pub gate: f64,
    pub final_rate_counts_s: f64,
    pub requested_counts_y: f64,
    pub emitted_counts_y: i32,
    pub residual_counts_y: f64,
    pub error_y_norm: Option<f64>,
    pub observation_age_ms: Option<f64>,
    pub source_generation: Option<u64>,
    pub block_reason: RecoilBlockReason,
}

impl RecoilDecision {
    pub fn engaged(self) -> bool {
        matches!(
            self.state,
            RecoilState::Startup | RecoilState::Active | RecoilState::Hold | RecoilState::Brake
        ) && (self.gate > 0.0 || self.final_rate_counts_s > 0.0 || self.residual_counts_y > 0.0)
    }
}

#[derive(Clone, Debug)]
pub struct TargetRelativeRecoilController {
    config: RecoilConfig,
    state: RecoilState,
    fire_start_ns: Option<u64>,
    residual: f64,
    target_id: Option<u64>,
}

impl TargetRelativeRecoilController {
    pub fn new(config: RecoilConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            state: RecoilState::Idle,
            fire_start_ns: None,
            residual: 0.0,
            target_id: None,
        })
    }

    pub fn reset(&mut self) {
        self.state = RecoilState::Idle;
        self.fire_start_ns = None;
        self.residual = 0.0;
        self.target_id = None;
    }

    pub fn calculate(&mut self, input: RecoilInput) -> RecoilDecision {
        let config = self.config;
        if !config.enabled {
            self.reset();
            return self.blocked(RecoilBlockReason::RecoilDisabled, input);
        }
        if !input.firing {
            self.reset();
            return self.blocked(RecoilBlockReason::FiringInactive, input);
        }
        if !input.dt_s.is_finite() || input.dt_s < 0.0 {
            self.reset();
            return self.blocked(RecoilBlockReason::DtInvalid, input);
        }
        if !input.observation_age_ms.is_finite() || input.observation_age_ms < 0.0 {
            self.reset();
            return self.blocked(RecoilBlockReason::ObservationAgeInvalid, input);
        }
        if input.observation_age_ms > config.stale_threshold_ms || !input.target_valid {
            self.state = RecoilState::Stale;
            self.residual = 0.0;
            return self.blocked(
                if input.target_valid {
                    RecoilBlockReason::TargetStale
                } else {
                    RecoilBlockReason::TargetInvalid
                },
                input,
            );
        }
        if !input.error_y_norm.is_finite() {
            self.state = RecoilState::Stale;
            self.residual = 0.0;
            return self.blocked(RecoilBlockReason::ErrorInvalid, input);
        }
        let fire_start_ns = *self.fire_start_ns.get_or_insert(input.now_ns);
        if self.target_id.is_some() && input.target_id != self.target_id {
            self.residual = 0.0;
        }
        self.target_id = input.target_id;
        let elapsed_ms = input.now_ns.saturating_sub(fire_start_ns) as f64 / 1_000_000.0;
        let startup_gate = if config.startup_ms <= 0.0 {
            1.0
        } else {
            ((elapsed_ms + input.dt_s * 1_000.0) / config.startup_ms).clamp(0.0, 1.0)
        };
        let error = input.error_y_norm;
        let gate = if error <= -config.full_brake_error_norm {
            0.0
        } else if error < -config.negative_deadzone_norm {
            1.0 - (-error - config.negative_deadzone_norm)
                / (config.full_brake_error_norm - config.negative_deadzone_norm)
        } else {
            1.0
        };
        self.state = if error < -config.negative_deadzone_norm {
            RecoilState::Brake
        } else if startup_gate < 1.0 {
            RecoilState::Startup
        } else if error <= config.positive_deadzone_norm {
            RecoilState::Hold
        } else {
            RecoilState::Active
        };
        let fast_add_rate_counts_s = if error > config.positive_deadzone_norm {
            (config.fast_add_gain_counts_s * (error - config.positive_deadzone_norm))
                .min(config.max_rate_counts_s * config.max_fast_add_ratio)
        } else {
            0.0
        };
        let final_rate_counts_s =
            ((config.base_rate_counts_s + fast_add_rate_counts_s) * gate * startup_gate)
                .clamp(0.0, config.max_rate_counts_s);
        let requested_counts_y = final_rate_counts_s * input.dt_s;
        let accumulated = self.residual + requested_counts_y;
        let emitted_counts_y = accumulated.trunc().max(0.0) as i32;
        self.residual = accumulated - f64::from(emitted_counts_y);
        RecoilDecision {
            state: self.state,
            base_rate_counts_s: config.base_rate_counts_s,
            fast_add_rate_counts_s,
            gate,
            final_rate_counts_s,
            requested_counts_y,
            emitted_counts_y,
            residual_counts_y: self.residual,
            error_y_norm: Some(error),
            observation_age_ms: Some(input.observation_age_ms),
            source_generation: input.source_generation,
            block_reason: if gate <= 0.0 {
                RecoilBlockReason::PositionBrake
            } else if final_rate_counts_s <= 0.0 {
                RecoilBlockReason::RecoilRateZero
            } else {
                RecoilBlockReason::None
            },
        }
    }

    fn blocked(&self, reason: RecoilBlockReason, input: RecoilInput) -> RecoilDecision {
        RecoilDecision {
            state: self.state,
            residual_counts_y: self.residual,
            error_y_norm: input.error_y_norm.is_finite().then_some(input.error_y_norm),
            observation_age_ms: input
                .observation_age_ms
                .is_finite()
                .then_some(input.observation_age_ms),
            source_generation: input.source_generation,
            block_reason: reason,
            ..RecoilDecision::default()
        }
    }
}

pub fn mix_tracking_and_recoil(tracking_y: i32, recoil: RecoilDecision) -> i32 {
    if recoil.engaged() {
        recoil
            .emitted_counts_y
            .max(tracking_y.max(0))
            .saturating_add(tracking_y.min(0))
    } else {
        tracking_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> RecoilConfig {
        RecoilConfig {
            enabled: true,
            base_rate_counts_s: 600.0,
            max_rate_counts_s: 1_000.0,
            startup_ms: 0.0,
            fast_add_gain_counts_s: 500.0,
            ..RecoilConfig::default()
        }
    }

    fn input(now_ns: u64, dt_s: f64, error_y_norm: f64) -> RecoilInput {
        RecoilInput {
            firing: true,
            now_ns,
            dt_s,
            target_valid: true,
            target_id: Some(7),
            source_generation: Some(1),
            observation_age_ms: 4.0,
            error_y_norm,
        }
    }

    #[test]
    fn rate_integration_is_independent_of_tick_frequency() {
        for hz in [60_u64, 120] {
            let mut controller = TargetRelativeRecoilController::new(enabled()).unwrap();
            let dt = 1.0 / hz as f64;
            let mut now = 1_000_000_000;
            let mut total = 0.0;
            for tick in 0..hz {
                now += 1_000_000_000 / hz;
                let decision = controller.calculate(input(now, dt, 0.0));
                total += f64::from(decision.emitted_counts_y);
                if tick == hz - 1 {
                    total += decision.residual_counts_y;
                }
            }
            assert!((total - 600.0).abs() < 1e-6);
        }
    }

    #[test]
    fn target_error_brakes_and_mixer_keeps_upward_recovery() {
        let mut controller = TargetRelativeRecoilController::new(enabled()).unwrap();
        let active = controller.calculate(input(1_000_000_000, 0.004, 0.20));
        let brake = controller.calculate(input(1_004_000_000, 0.004, -0.08));
        let stopped = controller.calculate(input(1_008_000_000, 0.004, -0.12));
        assert_eq!(active.state, RecoilState::Active);
        assert!(active.fast_add_rate_counts_s > 0.0);
        assert_eq!(brake.state, RecoilState::Brake);
        assert!(brake.gate > 0.0 && brake.gate < 1.0);
        assert_eq!(stopped.final_rate_counts_s, 0.0);
        assert_eq!(mix_tracking_and_recoil(-3, active), -1);
        assert_eq!(mix_tracking_and_recoil(4, active), 4);
    }

    #[test]
    fn target_switch_clears_fractional_debt_and_release_resets_fire_state() {
        let mut config = enabled();
        config.base_rate_counts_s = 125.0;
        config.max_rate_counts_s = 125.0;
        let mut controller = TargetRelativeRecoilController::new(config).unwrap();
        let first = controller.calculate(input(1_000_000_000, 0.004, 0.0));
        assert_eq!(first.emitted_counts_y, 0);
        assert_eq!(first.residual_counts_y, 0.5);

        let mut switched = input(1_004_000_000, 0.004, 0.0);
        switched.target_id = Some(8);
        let switched = controller.calculate(switched);
        assert_eq!(switched.emitted_counts_y, 0);
        assert_eq!(switched.residual_counts_y, 0.5);

        let mut released = input(1_008_000_000, 0.004, 0.0);
        released.firing = false;
        let released = controller.calculate(released);
        assert_eq!(released.state, RecoilState::Idle);
        assert_eq!(released.residual_counts_y, 0.0);
        assert_eq!(released.block_reason, RecoilBlockReason::FiringInactive);
    }
}
