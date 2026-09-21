//! Raw detector classes stay factual; policy uses them only as soft evidence.

pub(super) fn preference_score(class_id: u32, priority: &[u32]) -> f64 {
    priority
        .iter()
        .position(|candidate| *candidate == class_id)
        .map_or(0.0, |rank| 0.5_f64.powi(rank as i32))
}

pub(super) const fn association_cost(previous: u32, observed: u32) -> f64 {
    if previous == observed { 0.0 } else { 1.0 }
}
