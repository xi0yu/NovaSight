use super::{ConfigSectionSchema, boolean, float, hot_section};

pub(super) fn section() -> ConfigSectionSchema {
    hot_section(
        "control.fire_stabilization",
        "开火稳定",
        vec![
            boolean("control.fire_stabilization.enabled", "启用近区自适应下压"),
            float(
                "control.fire_stabilization.strength",
                "学习强度",
                0.0,
                1.0,
                None,
            ),
        ],
    )
}
