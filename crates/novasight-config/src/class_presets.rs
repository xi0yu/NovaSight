use std::collections::{BTreeMap, BTreeSet};

use novasight_core::tracking::TargetPartRole;
use serde::{Deserialize, Serialize};

/// A named class mapping. Built-ins are code-owned; user snapshots live in
/// `inference.detection_custom_presets` and are included in config backups.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassPreset {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub verified: bool,
    pub class_names: Vec<String>,
    pub enabled_ids: Vec<u32>,
    pub weights: BTreeMap<u32, f64>,
    #[serde(default)]
    pub roles: BTreeMap<u32, TargetPartRole>,
}

impl ClassPreset {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.id.is_empty()
            || self.id.len() > 64
            || !self.id.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
            })
        {
            return Err("preset id must contain 1-64 lowercase ASCII letters, digits, _ or -");
        }
        if self.label.trim().is_empty()
            || self.label.chars().count() > 40
            || self.note.chars().count() > 160
            || !(1..=8).contains(&self.class_names.len())
            || self
                .class_names
                .iter()
                .any(|name| name.trim().is_empty() || name.chars().count() > 40)
        {
            return Err("preset must have a short label and 1-8 nonempty class names");
        }
        let count = self.class_names.len() as u32;
        let enabled: BTreeSet<u32> = self.enabled_ids.iter().copied().collect();
        if enabled.is_empty()
            || enabled.len() != self.enabled_ids.len()
            || enabled.iter().any(|id| *id >= count)
            || self.weights.iter().any(|(id, value)| {
                *id >= count || !value.is_finite() || !(0.0..=1.0).contains(value)
            })
            || self.roles.keys().any(|id| *id >= count)
        {
            return Err("preset class ids or weights are invalid");
        }
        Ok(())
    }
}

pub fn builtin_class_presets() -> Vec<ClassPreset> {
    vec![ClassPreset {
        id: "babi_20260920_device_record".into(),
        label: "Babi 七类 · 设备记录".into(),
        note: "来自 Jetson 现有配置；模型清单只有 class_0～class_6，应用前请核对画面中的类别。"
            .into(),
        verified: false,
        class_names: [
            "敌人/身体",
            "头部",
            "队友",
            "小兵",
            "倒地",
            "靶场",
            "靶场头",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        enabled_ids: vec![0, 1],
        weights: BTreeMap::from([(0, 0.5), (1, 1.0)]),
        roles: BTreeMap::from([(0, TargetPartRole::Body), (1, TargetPartRole::Head)]),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_valid_but_not_claimed_verified() {
        for preset in builtin_class_presets() {
            assert!(preset.validate().is_ok());
            assert!(!preset.verified);
        }
    }
}
