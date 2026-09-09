//! Interval-gated recoil contribution for the final pointer command.
//!
//! Recoil never owns a timer-driven device lane. On each newest safe visual
//! observation, a due contribution is combined with that observation's
//! tracking demand (which may be zero) and emitted as at most one command.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecoilConfig {
    pub enabled: bool,
    pub require_target: bool,
    pub interval_ms: u64,
    pub y_counts: i32,
}

impl Default for RecoilConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_target: true,
            interval_ms: 16,
            y_counts: 1,
        }
    }
}

impl RecoilConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1..=5_000).contains(&self.interval_ms) {
            return Err("recoil interval must be within 1..=5000 ms");
        }
        if !(1..=i16::MAX as i32).contains(&self.y_counts) {
            return Err("recoil positive-Y contribution must be within 1..=32767 counts");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoilState {
    #[default]
    Idle,
    Waiting,
    Ready,
    Applied,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecoilBlockReason {
    #[default]
    RecoilDisabled,
    FiringInactive,
    TargetRequired,
    IntervalPending,
    OutputSaturated,
    #[serde(rename = "")]
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoilInput {
    pub firing: bool,
    pub now_ns: u64,
    pub target_valid: bool,
    pub source_generation: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecoilDecision {
    pub state: RecoilState,
    pub interval_ms: u64,
    pub configured_y_counts: i32,
    pub elapsed_since_output_ms: Option<f64>,
    pub remaining_ms: f64,
    pub requested_counts_y: i32,
    pub emitted_counts_y: i32,
    pub source_generation: Option<u64>,
    pub block_reason: RecoilBlockReason,
}

impl RecoilDecision {
    pub fn engaged(self) -> bool {
        matches!(
            self.state,
            RecoilState::Waiting | RecoilState::Ready | RecoilState::Applied
        )
    }

    pub fn should_add(self) -> bool {
        self.state == RecoilState::Ready && self.requested_counts_y > 0
    }

    pub fn mark_output_result(&mut self, applied_counts_y: i32) -> bool {
        if self.should_add() && applied_counts_y > 0 {
            self.state = RecoilState::Applied;
            self.emitted_counts_y = applied_counts_y;
            true
        } else if self.should_add() {
            self.block_reason = RecoilBlockReason::OutputSaturated;
            false
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntervalRecoilController {
    config: RecoilConfig,
    cadence_started_ns: Option<u64>,
    last_output_ns: Option<u64>,
}

impl IntervalRecoilController {
    pub fn new(config: RecoilConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            cadence_started_ns: None,
            last_output_ns: None,
        })
    }

    pub fn reset(&mut self) {
        self.cadence_started_ns = None;
        self.last_output_ns = None;
    }

    /// Decide whether the current safe output plan should receive +Y.
    ///
    /// This method never consumes a due recoil step. Call `mark_output_sent`
    /// only after the combined device command has been accepted, so a failed
    /// send cannot advance the cadence or lose a recoil step.
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

        let cadence_start = *self.cadence_started_ns.get_or_insert(input.now_ns);
        let baseline = self.last_output_ns.unwrap_or(cadence_start);
        let elapsed_ns = input.now_ns.saturating_sub(baseline);
        let elapsed_ms = elapsed_ns as f64 / 1_000_000.0;
        let required_interval_ns = config.interval_ms.saturating_mul(1_000_000);
        let remaining_ms = required_interval_ns.saturating_sub(elapsed_ns) as f64 / 1_000_000.0;

        if config.require_target && !input.target_valid {
            return RecoilDecision {
                state: RecoilState::Waiting,
                interval_ms: config.interval_ms,
                configured_y_counts: config.y_counts,
                elapsed_since_output_ms: Some(elapsed_ms),
                remaining_ms,
                source_generation: input.source_generation,
                block_reason: RecoilBlockReason::TargetRequired,
                ..RecoilDecision::default()
            };
        }
        if elapsed_ns < required_interval_ns {
            return RecoilDecision {
                state: RecoilState::Waiting,
                interval_ms: config.interval_ms,
                configured_y_counts: config.y_counts,
                elapsed_since_output_ms: Some(elapsed_ms),
                remaining_ms,
                source_generation: input.source_generation,
                block_reason: RecoilBlockReason::IntervalPending,
                ..RecoilDecision::default()
            };
        }

        RecoilDecision {
            state: RecoilState::Ready,
            interval_ms: config.interval_ms,
            configured_y_counts: config.y_counts,
            elapsed_since_output_ms: Some(elapsed_ms),
            remaining_ms: 0.0,
            requested_counts_y: config.y_counts,
            source_generation: input.source_generation,
            block_reason: RecoilBlockReason::None,
            ..RecoilDecision::default()
        }
    }

    pub fn mark_output_sent(&mut self, now_ns: u64) {
        self.last_output_ns = Some(now_ns);
    }

    fn blocked(&self, reason: RecoilBlockReason, input: RecoilInput) -> RecoilDecision {
        RecoilDecision {
            interval_ms: self.config.interval_ms,
            configured_y_counts: self.config.y_counts,
            source_generation: input.source_generation,
            block_reason: reason,
            ..RecoilDecision::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoilMix {
    pub command_y: i32,
    pub applied_counts_y: i32,
    /// Portion of the signed tracking demand that remains after the positive-Y
    /// recoil contribution is composed within the device integer range. The
    /// configured fixed Y limit is applied to the combined command downstream.
    pub surviving_tracking_counts_y: i32,
}

/// Mix the recoil contribution into the current command exactly once and
/// report how much positive Y survived the representable device-range clamp.
/// The configured fixed X/Y output limits remain the final downstream boundary.
pub fn mix_tracking_and_recoil(tracking_y: i32, recoil: RecoilDecision) -> RecoilMix {
    let command_y = if recoil.should_add() {
        tracking_y
            .saturating_add(recoil.requested_counts_y)
            .clamp(i16::MIN as i32, i16::MAX as i32)
    } else {
        tracking_y
    };
    RecoilMix {
        command_y,
        applied_counts_y: command_y.saturating_sub(tracking_y).max(0),
        surviving_tracking_counts_y: if command_y.signum() == tracking_y.signum() {
            tracking_y.signum() * command_y.abs().min(tracking_y.abs())
        } else {
            0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> RecoilConfig {
        RecoilConfig {
            enabled: true,
            interval_ms: 10,
            y_counts: 3,
            ..RecoilConfig::default()
        }
    }

    fn input(now_ns: u64) -> RecoilInput {
        RecoilInput {
            firing: true,
            now_ns,
            target_valid: true,
            source_generation: Some(1),
        }
    }

    #[test]
    fn waits_for_interval_and_never_catches_up_multiple_steps() {
        let mut controller = IntervalRecoilController::new(enabled()).unwrap();
        let first = controller.calculate(input(1_000_000_000));
        assert_eq!(first.state, RecoilState::Waiting);
        assert_eq!(first.remaining_ms, 10.0);

        let due = controller.calculate(input(1_010_000_000));
        assert!(due.should_add());
        assert_eq!(mix_tracking_and_recoil(-2, due).command_y, 1);
        controller.mark_output_sent(1_010_000_000);

        let late = controller.calculate(input(1_035_000_000));
        assert_eq!(late.requested_counts_y, 3);
    }

    #[test]
    fn failed_send_does_not_consume_a_due_step() {
        let mut controller = IntervalRecoilController::new(enabled()).unwrap();
        controller.calculate(input(1_000_000_000));
        let first_attempt = controller.calculate(input(1_010_000_000));
        let retry = controller.calculate(input(1_011_000_000));
        assert!(first_attempt.should_add());
        assert!(retry.should_add());

        controller.mark_output_sent(1_011_000_000);
        assert!(!controller.calculate(input(1_012_000_000)).should_add());
    }

    #[test]
    fn target_gate_blocks_addition_without_resetting_elapsed_time() {
        let mut controller = IntervalRecoilController::new(enabled()).unwrap();
        let mut no_target = input(1_000_000_000);
        no_target.target_valid = false;
        controller.calculate(no_target);
        no_target.now_ns = 1_020_000_000;
        let blocked = controller.calculate(no_target);
        assert_eq!(blocked.block_reason, RecoilBlockReason::TargetRequired);

        let ready = controller.calculate(input(1_021_000_000));
        assert!(ready.should_add());
    }

    #[test]
    fn release_resets_the_first_interval() {
        let mut controller = IntervalRecoilController::new(enabled()).unwrap();
        controller.calculate(input(1_000_000_000));
        let mut released = input(1_020_000_000);
        released.firing = false;
        assert_eq!(
            controller.calculate(released).block_reason,
            RecoilBlockReason::FiringInactive
        );

        let pressed_again = controller.calculate(input(1_030_000_000));
        assert_eq!(pressed_again.remaining_ms, 10.0);
        assert!(!pressed_again.should_add());
    }

    #[test]
    fn saturation_reports_only_the_contribution_that_reaches_the_command() {
        let ready = RecoilDecision {
            state: RecoilState::Ready,
            requested_counts_y: 3,
            ..RecoilDecision::default()
        };
        assert_eq!(
            mix_tracking_and_recoil(i16::MAX as i32, ready),
            RecoilMix {
                command_y: i16::MAX as i32,
                applied_counts_y: 0,
                surviving_tracking_counts_y: i16::MAX as i32,
            }
        );
        assert_eq!(
            mix_tracking_and_recoil(i16::MAX as i32 - 1, ready),
            RecoilMix {
                command_y: i16::MAX as i32,
                applied_counts_y: 1,
                surviving_tracking_counts_y: i16::MAX as i32 - 1,
            }
        );
    }

    #[test]
    fn opposite_tracking_and_recoil_report_only_the_physical_tracking_remainder() {
        let ready = RecoilDecision {
            state: RecoilState::Ready,
            requested_counts_y: 3,
            ..RecoilDecision::default()
        };
        assert_eq!(
            mix_tracking_and_recoil(-5, ready),
            RecoilMix {
                command_y: -2,
                applied_counts_y: 3,
                surviving_tracking_counts_y: -2,
            }
        );
        assert_eq!(
            mix_tracking_and_recoil(-3, ready),
            RecoilMix {
                command_y: 0,
                applied_counts_y: 3,
                surviving_tracking_counts_y: 0,
            }
        );
        assert_eq!(
            mix_tracking_and_recoil(-1, ready),
            RecoilMix {
                command_y: 2,
                applied_counts_y: 3,
                surviving_tracking_counts_y: 0,
            }
        );
    }
}
