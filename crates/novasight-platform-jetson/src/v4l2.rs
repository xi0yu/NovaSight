//! Direct V4L2 capability enumeration for the production capture control plane.
//!
//! The probe uses the kernel ABI rather than parsing `v4l2-ctl` output. Only
//! discrete sizes and intervals are projected because that is the existing
//! Studio contract; continuous/stepwise ranges are never invented as profiles.

#[cfg(target_os = "linux")]
mod linux {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};

    use novasight_core::{
        CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureProbeError,
    };

    const VIDEO_CAPTURE: u32 = 1;
    const VIDEO_CAPTURE_MPLANE: u32 = 9;
    const FRAME_SIZE_DISCRETE: u32 = 1;
    const FRAME_INTERVAL_DISCRETE: u32 = 1;
    const MAX_ENUMERATED_ITEMS: u32 = 4_096;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FormatDescription {
        index: u32,
        buffer_type: u32,
        flags: u32,
        description: [u8; 32],
        pixel_format: u32,
        media_bus_code: u32,
        reserved: [u32; 3],
    }

    impl V4l2FormatDescription {
        const fn new(index: u32, buffer_type: u32) -> Self {
            Self {
                index,
                buffer_type,
                flags: 0,
                description: [0; 32],
                pixel_format: 0,
                media_bus_code: 0,
                reserved: [0; 3],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FrameSizeDiscrete {
        width: u32,
        height: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FrameSizeStepwise {
        min_width: u32,
        max_width: u32,
        step_width: u32,
        min_height: u32,
        max_height: u32,
        step_height: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    union V4l2FrameSize {
        discrete: V4l2FrameSizeDiscrete,
        stepwise: V4l2FrameSizeStepwise,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FrameSizeEnum {
        index: u32,
        pixel_format: u32,
        frame_size_type: u32,
        frame_size: V4l2FrameSize,
        reserved: [u32; 2],
    }

    impl V4l2FrameSizeEnum {
        const fn new(index: u32, pixel_format: u32) -> Self {
            Self {
                index,
                pixel_format,
                frame_size_type: 0,
                frame_size: V4l2FrameSize {
                    discrete: V4l2FrameSizeDiscrete {
                        width: 0,
                        height: 0,
                    },
                },
                reserved: [0; 2],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Fraction {
        numerator: u32,
        denominator: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FrameIntervalStepwise {
        min: V4l2Fraction,
        max: V4l2Fraction,
        step: V4l2Fraction,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    union V4l2FrameInterval {
        discrete: V4l2Fraction,
        stepwise: V4l2FrameIntervalStepwise,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2FrameIntervalEnum {
        index: u32,
        pixel_format: u32,
        width: u32,
        height: u32,
        frame_interval_type: u32,
        frame_interval: V4l2FrameInterval,
        reserved: [u32; 2],
    }

    impl V4l2FrameIntervalEnum {
        const fn new(index: u32, pixel_format: u32, width: u32, height: u32) -> Self {
            Self {
                index,
                pixel_format,
                width,
                height,
                frame_interval_type: 0,
                frame_interval: V4l2FrameInterval {
                    discrete: V4l2Fraction {
                        numerator: 0,
                        denominator: 0,
                    },
                },
                reserved: [0; 2],
            }
        }
    }

    nix::ioctl_readwrite!(enumerate_format, b'V', 2, V4l2FormatDescription);
    nix::ioctl_readwrite!(enumerate_frame_sizes, b'V', 74, V4l2FrameSizeEnum);
    nix::ioctl_readwrite!(enumerate_frame_intervals, b'V', 75, V4l2FrameIntervalEnum);

    #[derive(Clone, Copy, Debug, Default)]
    pub struct V4l2CapabilityProbe;

    impl CaptureCapabilityProbe for V4l2CapabilityProbe {
        fn probe(&self, device: &str) -> Result<CaptureCapabilities, CaptureProbeError> {
            let device = device.trim();
            if device.is_empty() {
                return Err(CaptureProbeError::new(
                    "CAPTURE_DEVICE_INVALID",
                    "capture device must not be blank",
                ));
            }
            if device.len() > 4_096 || device.as_bytes().contains(&0) {
                return Err(CaptureProbeError::new(
                    "CAPTURE_DEVICE_INVALID",
                    "capture device path is not a valid bounded Unix path",
                ));
            }
            let file = match OpenOptions::new()
                .read(true)
                .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_CLOEXEC)
                .open(device)
            {
                Ok(file) => file,
                Err(error) => {
                    return Ok(unavailable(
                        device,
                        format!("failed to open V4L2 device: {error}"),
                    ));
                }
            };
            let metadata = match file.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    return Ok(unavailable(
                        device,
                        format!("failed to inspect opened V4L2 device: {error}"),
                    ));
                }
            };
            if !metadata.file_type().is_char_device() {
                return Ok(unavailable(
                    device,
                    "opened capture path is not a character device".to_owned(),
                ));
            }
            let descriptor = file.as_raw_fd();
            let formats = match enumerate_formats(descriptor) {
                Ok(formats) => formats,
                Err(error) => {
                    return Ok(unavailable(
                        device,
                        format!("failed to enumerate V4L2 formats: {error}"),
                    ));
                }
            };
            let mut profiles: BTreeMap<(String, u32, u32), BTreeSet<u32>> = BTreeMap::new();
            for pixel_format in formats {
                for (width, height) in enumerate_sizes(descriptor, pixel_format)? {
                    let frame_rates = enumerate_intervals(descriptor, pixel_format, width, height)?;
                    if !frame_rates.is_empty() {
                        profiles
                            .entry((fourcc(pixel_format), width, height))
                            .or_default()
                            .extend(frame_rates);
                    }
                }
            }
            let capabilities = profiles
                .into_iter()
                .map(|((pixel_format, width, height), frame_rates)| {
                    let mut fps_list = frame_rates.into_iter().collect::<Vec<_>>();
                    fps_list.sort_unstable_by(|left, right| right.cmp(left));
                    CaptureCapability {
                        pixel_format,
                        width,
                        height,
                        fps_list,
                    }
                })
                .collect::<Vec<_>>();
            Ok(CaptureCapabilities {
                available: !capabilities.is_empty(),
                device: device.to_owned(),
                reason: if capabilities.is_empty() {
                    "no discrete V4L2 formats, sizes, and frame intervals were reported".to_owned()
                } else {
                    String::new()
                },
                capabilities,
            })
        }
    }

    fn unavailable(device: &str, reason: String) -> CaptureCapabilities {
        CaptureCapabilities {
            available: false,
            device: device.to_owned(),
            capabilities: Vec::new(),
            reason,
        }
    }

    fn enumerate_formats(descriptor: i32) -> Result<BTreeSet<u32>, CaptureProbeError> {
        let mut formats = BTreeSet::new();
        for buffer_type in [VIDEO_CAPTURE, VIDEO_CAPTURE_MPLANE] {
            for index in 0..MAX_ENUMERATED_ITEMS {
                let mut description = V4l2FormatDescription::new(index, buffer_type);
                // SAFETY: the descriptor remains open for the call and the repr(C)
                // value is fully initialized with the layout from linux/videodev2.h.
                match unsafe { enumerate_format(descriptor, &mut description) } {
                    Ok(_) => {
                        formats.insert(description.pixel_format);
                    }
                    Err(nix::errno::Errno::EINVAL) => break,
                    Err(error) if index == 0 => {
                        if buffer_type == VIDEO_CAPTURE_MPLANE {
                            break;
                        }
                        return Err(probe_error("VIDIOC_ENUM_FMT", error));
                    }
                    Err(error) => return Err(probe_error("VIDIOC_ENUM_FMT", error)),
                }
                if index + 1 == MAX_ENUMERATED_ITEMS {
                    return Err(limit_error("formats"));
                }
            }
        }
        Ok(formats)
    }

    fn enumerate_sizes(
        descriptor: i32,
        pixel_format: u32,
    ) -> Result<Vec<(u32, u32)>, CaptureProbeError> {
        let mut sizes = Vec::new();
        for index in 0..MAX_ENUMERATED_ITEMS {
            let mut frame_size = V4l2FrameSizeEnum::new(index, pixel_format);
            // SAFETY: the open descriptor and fully initialized repr(C) value
            // satisfy the V4L2 ioctl contract.
            match unsafe { enumerate_frame_sizes(descriptor, &mut frame_size) } {
                Ok(_) if frame_size.frame_size_type == FRAME_SIZE_DISCRETE => {
                    // SAFETY: the kernel selected the discrete union variant.
                    let discrete = unsafe { frame_size.frame_size.discrete };
                    if discrete.width > 0 && discrete.height > 0 {
                        sizes.push((discrete.width, discrete.height));
                    }
                }
                Ok(_) => {}
                Err(nix::errno::Errno::EINVAL) => break,
                Err(error) => return Err(probe_error("VIDIOC_ENUM_FRAMESIZES", error)),
            }
            if index + 1 == MAX_ENUMERATED_ITEMS {
                return Err(limit_error("frame sizes"));
            }
        }
        sizes.sort_unstable();
        sizes.dedup();
        Ok(sizes)
    }

    fn enumerate_intervals(
        descriptor: i32,
        pixel_format: u32,
        width: u32,
        height: u32,
    ) -> Result<BTreeSet<u32>, CaptureProbeError> {
        let mut frame_rates = BTreeSet::new();
        for index in 0..MAX_ENUMERATED_ITEMS {
            let mut interval = V4l2FrameIntervalEnum::new(index, pixel_format, width, height);
            // SAFETY: the open descriptor and fully initialized repr(C) value
            // satisfy the V4L2 ioctl contract.
            match unsafe { enumerate_frame_intervals(descriptor, &mut interval) } {
                Ok(_) if interval.frame_interval_type == FRAME_INTERVAL_DISCRETE => {
                    // SAFETY: the kernel selected the discrete union variant.
                    let fraction = unsafe { interval.frame_interval.discrete };
                    if let Some(fps) = rounded_fps(fraction) {
                        frame_rates.insert(fps);
                    }
                }
                Ok(_) => {}
                Err(nix::errno::Errno::EINVAL) => break,
                Err(error) => return Err(probe_error("VIDIOC_ENUM_FRAMEINTERVALS", error)),
            }
            if index + 1 == MAX_ENUMERATED_ITEMS {
                return Err(limit_error("frame intervals"));
            }
        }
        Ok(frame_rates)
    }

    fn rounded_fps(fraction: V4l2Fraction) -> Option<u32> {
        if fraction.numerator == 0 || fraction.denominator == 0 {
            return None;
        }
        let numerator = u64::from(fraction.numerator);
        let denominator = u64::from(fraction.denominator);
        let rounded = denominator.saturating_add(numerator / 2) / numerator;
        u32::try_from(rounded).ok().filter(|fps| *fps > 0)
    }

    fn fourcc(value: u32) -> String {
        value
            .to_le_bytes()
            .into_iter()
            .map(|byte| {
                if byte.is_ascii_graphic() {
                    char::from(byte)
                } else {
                    '?'
                }
            })
            .collect()
    }

    fn probe_error(operation: &'static str, error: nix::errno::Errno) -> CaptureProbeError {
        CaptureProbeError::new(
            "CAPTURE_CAPABILITY_IOCTL_FAILED",
            format!("{operation} failed: {error}"),
        )
    }

    fn limit_error(kind: &'static str) -> CaptureProbeError {
        CaptureProbeError::new(
            "CAPTURE_CAPABILITY_LIMIT_EXCEEDED",
            format!("V4L2 reported more than {MAX_ENUMERATED_ITEMS} {kind}"),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn kernel_abi_layouts_match_videodev2() {
            assert_eq!(std::mem::size_of::<V4l2FormatDescription>(), 64);
            assert_eq!(std::mem::size_of::<V4l2FrameSizeEnum>(), 44);
            assert_eq!(std::mem::size_of::<V4l2FrameIntervalEnum>(), 52);
        }

        #[test]
        fn fourcc_and_fraction_match_v4l2_ctl_projection() {
            assert_eq!(fourcc(u32::from_le_bytes(*b"MJPG")), "MJPG");
            assert_eq!(
                rounded_fps(V4l2Fraction {
                    numerator: 1_001,
                    denominator: 30_000,
                }),
                Some(30)
            );
        }

        #[test]
        fn non_v4l2_file_reports_unavailable_without_inventing_profiles() {
            let result = V4l2CapabilityProbe.probe("/dev/null").unwrap();
            assert!(!result.available);
            assert!(result.capabilities.is_empty());
            assert!(result.reason.contains("VIDIOC_ENUM_FMT"));
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::V4l2CapabilityProbe;

#[cfg(not(target_os = "linux"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct V4l2CapabilityProbe;

#[cfg(not(target_os = "linux"))]
impl novasight_core::CaptureCapabilityProbe for V4l2CapabilityProbe {
    fn probe(
        &self,
        _device: &str,
    ) -> Result<novasight_core::CaptureCapabilities, novasight_core::CaptureProbeError> {
        Err(novasight_core::CaptureProbeError::new(
            "CAPTURE_PROBE_UNSUPPORTED_PLATFORM",
            "V4L2 capture probing is only available on Linux",
        ))
    }
}
