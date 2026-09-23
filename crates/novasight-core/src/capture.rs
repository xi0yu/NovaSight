use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Frames per second, preserving the device's fraction through caps negotiation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CaptureFrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

impl CaptureFrameRate {
    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Some(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }

    /// UI/selection preference only; never use this value as negotiated caps.
    pub fn rounded_fps(self) -> Option<u32> {
        if self.numerator == 0 || self.denominator == 0 {
            return None;
        }
        let fps = (u64::from(self.numerator) + u64::from(self.denominator) / 2)
            / u64::from(self.denominator);
        u32::try_from(fps).ok().filter(|fps| *fps > 0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaptureCapability {
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps_list: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frame_rates: Vec<CaptureFrameRate>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaptureCapabilities {
    pub available: bool,
    pub device: String,
    pub capabilities: Vec<CaptureCapability>,
    pub reason: String,
}

pub trait CaptureCapabilityProbe: Send + Sync + 'static {
    fn probe(&self, device: &str) -> Result<CaptureCapabilities, CaptureProbeError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureSelectionPreference {
    #[default]
    AutoHighFps,
    AutoLowLatency,
    AutoBalanced,
    Manual,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SelectedCaptureProfile {
    pub device: String,
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frame_rate: CaptureFrameRate,
    pub preference: CaptureSelectionPreference,
    pub selection_reason: &'static str,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CaptureSelectionError {
    #[error("no capture capabilities available for {device}")]
    NoCapabilities { device: String },
    #[error(
        "capture device {device} has profiles, but the selected adapter supports only {supported_formats}"
    )]
    AdapterUnsupportedFormats {
        device: String,
        supported_formats: String,
    },
    #[error("manual capture selection requires pixel_format, width, height, and fps")]
    ManualFieldsRequired,
    #[error("unsupported capture profile for {device}: {pixel_format} {width}x{height}@{fps}")]
    UnsupportedProfile {
        device: String,
        pixel_format: String,
        width: u32,
        height: u32,
        fps: u32,
    },
}

pub fn select_capture_profile(
    capabilities: &CaptureCapabilities,
    preference: CaptureSelectionPreference,
    manual: Option<(&str, u32, u32, u32)>,
) -> Result<SelectedCaptureProfile, CaptureSelectionError> {
    select_capture_profile_for_formats(capabilities, preference, manual, &[])
}

/// Resolve a concrete profile from the intersection of device capabilities
/// and the formats implemented by the selected capture adapter. An empty
/// allow-list preserves the platform-neutral selector behavior.
pub fn select_capture_profile_for_formats(
    capabilities: &CaptureCapabilities,
    preference: CaptureSelectionPreference,
    manual: Option<(&str, u32, u32, u32)>,
    supported_formats: &[&str],
) -> Result<SelectedCaptureProfile, CaptureSelectionError> {
    let device_has_profiles = capabilities
        .capabilities
        .iter()
        .any(|capability| !capability.fps_list.is_empty());
    let choices = capabilities
        .capabilities
        .iter()
        .filter(|capability| {
            supported_formats.is_empty()
                || supported_formats.iter().any(|supported| {
                    canonical_pixel_format(supported)
                        == canonical_pixel_format(&capability.pixel_format)
                })
        })
        .flat_map(|capability| {
            capability.fps_list.iter().map(move |fps| {
                (
                    canonical_pixel_format(&capability.pixel_format),
                    capability.width,
                    capability.height,
                    *fps,
                    capability
                        .frame_rates
                        .iter()
                        .copied()
                        .filter(|rate| rate.rounded_fps() == Some(*fps))
                        .min_by(|left, right| {
                            // Compare distance in FPS, independent of denominator.
                            let distance = |rate: &CaptureFrameRate| {
                                u128::from(rate.numerator)
                                    .abs_diff(u128::from(*fps) * u128::from(rate.denominator))
                            };
                            (distance(left) * u128::from(right.denominator))
                                .cmp(&(distance(right) * u128::from(left.denominator)))
                        })
                        .or_else(|| {
                            capability
                                .frame_rates
                                .is_empty()
                                .then(|| CaptureFrameRate::new(*fps, 1))
                                .flatten()
                        }),
                )
            })
        })
        .filter(|choice| choice.4.is_some())
        .collect::<Vec<_>>();
    if choices.is_empty() {
        if device_has_profiles && !supported_formats.is_empty() {
            return Err(CaptureSelectionError::AdapterUnsupportedFormats {
                device: capabilities.device.clone(),
                supported_formats: supported_formats.join(", "),
            });
        }
        return Err(CaptureSelectionError::NoCapabilities {
            device: capabilities.device.clone(),
        });
    }

    let (choice, selection_reason) = match preference {
        CaptureSelectionPreference::Manual => {
            let (pixel_format, width, height, fps) =
                manual.ok_or(CaptureSelectionError::ManualFieldsRequired)?;
            let pixel_format = canonical_pixel_format(pixel_format);
            let choice = choices
                .iter()
                .find(|choice| {
                    (choice.0.as_str(), choice.1, choice.2, choice.3)
                        == (pixel_format.as_str(), width, height, fps)
                })
                .cloned()
                .ok_or_else(|| CaptureSelectionError::UnsupportedProfile {
                    device: capabilities.device.clone(),
                    pixel_format,
                    width,
                    height,
                    fps,
                })?;
            (choice, "manual profile matched device capabilities")
        }
        CaptureSelectionPreference::AutoLowLatency => {
            let choice = choices
                .iter()
                .min_by_key(|choice| {
                    (
                        low_latency_format_rank(&choice.0),
                        std::cmp::Reverse(choice.3),
                        std::cmp::Reverse(u64::from(choice.1) * u64::from(choice.2)),
                    )
                })
                .expect("non-empty choices")
                .clone();
            (choice, "auto_low_latency selected raw-friendly profile")
        }
        CaptureSelectionPreference::AutoBalanced => {
            let target = 1920_u64 * 1080;
            let choice = choices
                .iter()
                .min_by_key(|choice| {
                    let pixels = u64::from(choice.1) * u64::from(choice.2);
                    (
                        pixels.abs_diff(target),
                        std::cmp::Reverse(choice.3),
                        high_fps_format_rank(&choice.0, choice.3),
                    )
                })
                .expect("non-empty choices")
                .clone();
            (
                choice,
                "auto_balanced selected profile closest to 1080p with high fps",
            )
        }
        CaptureSelectionPreference::AutoHighFps => {
            let choice = choices
                .iter()
                .min_by_key(|choice| {
                    jetson_nvmm_rank(&(choice.0.clone(), choice.1, choice.2, choice.3))
                })
                .expect("non-empty choices")
                .clone();
            (choice, "auto_high_fps selected jetson nvmm profile")
        }
    };

    Ok(SelectedCaptureProfile {
        device: capabilities.device.clone(),
        pixel_format: choice.0,
        width: choice.1,
        height: choice.2,
        fps: choice.3,
        frame_rate: choice.4.expect("filtered valid device rate"),
        preference,
        selection_reason,
    })
}

fn canonical_pixel_format(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "MJPEG" => "MJPG".to_owned(),
        "YUY2" => "YUYV".to_owned(),
        canonical => canonical.to_owned(),
    }
}

fn high_fps_format_rank(pixel_format: &str, fps: u32) -> u8 {
    match (pixel_format, fps) {
        ("MJPG", 120..) => 0,
        ("NV12", 60..) => 1,
        ("YUYV", _) => 2,
        ("MJPG", _) => 3,
        _ => 4,
    }
}

fn low_latency_format_rank(pixel_format: &str) -> u8 {
    match pixel_format {
        "NV12" => 0,
        "YUYV" => 1,
        "MJPG" => 2,
        _ => 3,
    }
}

fn jetson_nvmm_rank(choice: &(String, u32, u32, u32)) -> (u8, i64, i64, i64) {
    let (pixel_format, width, height, fps) = choice;
    let pixels = u64::from(*width) * u64::from(*height);
    let is_1080p = pixels == 1920 * 1080;
    match (pixel_format.as_str(), is_1080p, *fps) {
        ("MJPG", true, 120) => (0, 0, 0, 0),
        ("MJPG", true, 121..) => (1, i64::from(*fps) - 120, 0, 0),
        ("MJPG", true, 60..=119) => (2, 120 - i64::from(*fps), 0, 0),
        ("NV12", true, 120..) => (3, i64::from(*fps) - 120, 0, 0),
        ("NV12", true, 60..=119) => (4, 120 - i64::from(*fps), 0, 0),
        ("YUYV", true, 60..) => (5, 120 - i64::from(*fps), 0, 0),
        _ => (
            6,
            i64::from(high_fps_format_rank(pixel_format, *fps)),
            -i64::from(*fps),
            -i64::try_from(pixels).unwrap_or(i64::MAX),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> CaptureCapabilities {
        CaptureCapabilities {
            available: true,
            device: "/dev/video7".to_owned(),
            capabilities: vec![
                CaptureCapability {
                    pixel_format: "NV12".to_owned(),
                    width: 1920,
                    height: 1080,
                    fps_list: vec![60],
                    frame_rates: vec![],
                },
                CaptureCapability {
                    pixel_format: "MJPG".to_owned(),
                    width: 1920,
                    height: 1080,
                    fps_list: vec![60, 120],
                    frame_rates: vec![],
                },
            ],
            reason: "kernel capabilities".to_owned(),
        }
    }

    #[test]
    fn fractional_mode_preference_compares_fps_not_numerator_error() {
        let mut caps = capabilities();
        caps.capabilities[0].frame_rates = vec![
            CaptureFrameRate::new(601, 10).unwrap(),
            CaptureFrameRate::new(59997, 1000).unwrap(),
        ];
        let selected = select_capture_profile(
            &caps,
            CaptureSelectionPreference::Manual,
            Some(("NV12", 1920, 1080, 60)),
        )
        .unwrap();
        assert_eq!(
            selected.frame_rate,
            CaptureFrameRate::new(59997, 1000).unwrap()
        );
        caps.capabilities[0]
            .frame_rates
            .push(CaptureFrameRate::new(60, 1).unwrap());
        assert_eq!(
            select_capture_profile(
                &caps,
                CaptureSelectionPreference::Manual,
                Some(("NV12", 1920, 1080, 60))
            )
            .unwrap()
            .frame_rate,
            CaptureFrameRate::new(60, 1).unwrap()
        );
    }

    #[test]
    fn high_fps_matches_the_jetson_1080p_120_policy() {
        let selected = select_capture_profile(
            &capabilities(),
            CaptureSelectionPreference::AutoHighFps,
            None,
        )
        .unwrap();
        assert_eq!(selected.pixel_format, "MJPG");
        assert_eq!(selected.fps, 120);
    }

    #[test]
    fn low_latency_prefers_nv12_over_faster_mjpeg() {
        let selected = select_capture_profile(
            &capabilities(),
            CaptureSelectionPreference::AutoLowLatency,
            None,
        )
        .unwrap();
        assert_eq!(selected.pixel_format, "NV12");
        assert_eq!(selected.fps, 60);
    }

    #[test]
    fn manual_requires_an_exact_kernel_reported_tuple() {
        let error = select_capture_profile(
            &capabilities(),
            CaptureSelectionPreference::Manual,
            Some(("mjpg", 1280, 720, 120)),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CaptureSelectionError::UnsupportedProfile { .. }
        ));
    }

    #[test]
    fn manual_selection_accepts_common_v4l2_format_aliases() {
        let selected = select_capture_profile(
            &capabilities(),
            CaptureSelectionPreference::Manual,
            Some(("MJPEG", 1920, 1080, 120)),
        )
        .unwrap();

        assert_eq!(selected.pixel_format, "MJPG");
        assert_eq!(selected.fps, 120);
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct CaptureProbeError {
    code: &'static str,
    message: String,
}

impl CaptureProbeError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }
}
