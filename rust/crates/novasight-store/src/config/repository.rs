use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::os::darwin::fs::MetadataExt as DarwinMetadataExt;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::AsRawFd;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::MetadataExt;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use xattr::FileExt as XattrFileExt;

use serde_yaml::Value;

use super::{AppConfig, ConfigValidationError};

pub trait ConfigRepository {
    type Error: Error + Send + Sync + 'static;

    fn load_config(&self) -> Result<AppConfig, Self::Error>;

    fn save_config(
        &self,
        config: &AppConfig,
        expected_revision: u64,
    ) -> Result<AppConfig, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct YamlConfigRepository {
    path: PathBuf,
}

impl YamlConfigRepository {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(path: impl AsRef<Path>) -> Result<AppConfig, ConfigError> {
        load_document(path.as_ref()).map(|(_, _, config)| config)
    }

    pub fn save(
        path: impl AsRef<Path>,
        config: &AppConfig,
        expected_revision: u64,
    ) -> Result<AppConfig, ConfigError> {
        save_document(path.as_ref(), config, expected_revision)
    }

    pub fn save_field(
        &self,
        section: &str,
        key: &str,
        value: Value,
        expected_revision: u64,
    ) -> Result<AppConfig, ConfigError> {
        save_document_field(&self.path, section, key, value, expected_revision)
    }

    pub fn replace_document(
        &self,
        replacement: Value,
        expected_revision: u64,
    ) -> Result<AppConfig, ConfigError> {
        replace_document(&self.path, replacement, expected_revision)
    }
}

impl ConfigRepository for YamlConfigRepository {
    type Error = ConfigError;

    fn load_config(&self) -> Result<AppConfig, Self::Error> {
        Self::load(&self.path)
    }

    fn save_config(
        &self,
        config: &AppConfig,
        expected_revision: u64,
    ) -> Result<AppConfig, Self::Error> {
        Self::save(&self.path, config, expected_revision)
    }
}

#[derive(Debug)]
pub enum ConfigError {
    NotFound {
        path: PathBuf,
    },
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    Validation {
        path: PathBuf,
        source: ConfigValidationError,
    },
    Serialize {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    RevisionConflict {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    RevisionOverflow {
        path: PathBuf,
        revision: u64,
    },
    ReservedLegacyKey {
        path: PathBuf,
        section: &'static str,
        key: String,
    },
    InvalidFieldTarget {
        path: PathBuf,
        section: String,
        key: String,
        reason: &'static str,
    },
    InvalidReplacementDocument {
        path: PathBuf,
        reason: &'static str,
    },
    SymlinkUnsupported {
        path: PathBuf,
    },
    HardlinkUnsupported {
        path: PathBuf,
        links: u64,
    },
    PathChanged {
        path: PathBuf,
    },
    SecurityMetadataUnsupported {
        path: PathBuf,
        reason: String,
    },
}

impl ConfigError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "CONFIG_NOT_FOUND",
            Self::Io { .. } => "CONFIG_IO_ERROR",
            Self::Parse { .. } => "CONFIG_PARSE_ERROR",
            Self::Validation { .. } => "CONFIG_VALIDATION_ERROR",
            Self::Serialize { .. } => "CONFIG_SERIALIZE_ERROR",
            Self::RevisionConflict { .. } => "CONFIG_REVISION_CONFLICT",
            Self::RevisionOverflow { .. } => "CONFIG_REVISION_OVERFLOW",
            Self::ReservedLegacyKey { .. } => "CONFIG_RESERVED_LEGACY_KEY",
            Self::InvalidFieldTarget { .. } => "CONFIG_INVALID_FIELD_TARGET",
            Self::InvalidReplacementDocument { .. } => "CONFIG_REPLACEMENT_INVALID",
            Self::SymlinkUnsupported { .. } => "CONFIG_SYMLINK_UNSUPPORTED",
            Self::HardlinkUnsupported { .. } => "CONFIG_HARDLINK_UNSUPPORTED",
            Self::PathChanged { .. } => "CONFIG_PATH_CHANGED",
            Self::SecurityMetadataUnsupported { .. } => "CONFIG_SECURITY_METADATA_UNSUPPORTED",
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::NotFound { path }
            | Self::Io { path, .. }
            | Self::Parse { path, .. }
            | Self::Validation { path, .. }
            | Self::Serialize { path, .. }
            | Self::RevisionConflict { path, .. }
            | Self::RevisionOverflow { path, .. }
            | Self::ReservedLegacyKey { path, .. }
            | Self::InvalidFieldTarget { path, .. }
            | Self::InvalidReplacementDocument { path, .. }
            | Self::SymlinkUnsupported { path }
            | Self::HardlinkUnsupported { path, .. }
            | Self::PathChanged { path }
            | Self::SecurityMetadataUnsupported { path, .. } => path,
        }
    }

    pub const fn expected_revision(&self) -> Option<u64> {
        match self {
            Self::RevisionConflict { expected, .. } => Some(*expected),
            _ => None,
        }
    }

    pub const fn actual_revision(&self) -> Option<u64> {
        match self {
            Self::RevisionConflict { actual, .. } => Some(*actual),
            _ => None,
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { path } => {
                write!(
                    formatter,
                    "configuration file not found: {}",
                    path.display()
                )
            }
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "configuration {operation} failed for {}: {source}",
                path.display()
            ),
            Self::Parse { path, source } => write!(
                formatter,
                "configuration parse failed for {}: {source}",
                path.display()
            ),
            Self::Validation { path, source } => write!(
                formatter,
                "configuration validation failed for {}: {source}",
                path.display()
            ),
            Self::Serialize { path, source } => write!(
                formatter,
                "configuration serialization failed for {}: {source}",
                path.display()
            ),
            Self::RevisionConflict {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "configuration revision conflict for {}: expected {expected}, found {actual}",
                path.display()
            ),
            Self::RevisionOverflow { path, revision } => write!(
                formatter,
                "configuration revision overflow for {} at {revision}",
                path.display()
            ),
            Self::ReservedLegacyKey { path, section, key } => write!(
                formatter,
                "configuration legacy key {section}.{key} is reserved at {}",
                path.display()
            ),
            Self::InvalidFieldTarget {
                path,
                section,
                key,
                reason,
            } => write!(
                formatter,
                "configuration field target {section}.{key} is invalid at {}: {reason}",
                path.display()
            ),
            Self::InvalidReplacementDocument { path, reason } => write!(
                formatter,
                "configuration replacement is invalid at {}: {reason}",
                path.display()
            ),
            Self::SymlinkUnsupported { path } => write!(
                formatter,
                "configuration symlinks are unsupported: {}",
                path.display()
            ),
            Self::HardlinkUnsupported { path, links } => write!(
                formatter,
                "configuration has {links} hard links and cannot be atomically replaced: {}",
                path.display()
            ),
            Self::PathChanged { path } => write!(
                formatter,
                "configuration path changed while it was being saved: {}",
                path.display()
            ),
            Self::SecurityMetadataUnsupported { path, reason } => write!(
                formatter,
                "configuration security metadata cannot be preserved for {}: {reason}",
                path.display()
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } | Self::Serialize { source, .. } => Some(source),
            Self::Validation { source, .. } => Some(source),
            Self::NotFound { .. }
            | Self::RevisionConflict { .. }
            | Self::RevisionOverflow { .. }
            | Self::ReservedLegacyKey { .. }
            | Self::InvalidFieldTarget { .. }
            | Self::InvalidReplacementDocument { .. }
            | Self::SymlinkUnsupported { .. }
            | Self::HardlinkUnsupported { .. }
            | Self::PathChanged { .. }
            | Self::SecurityMetadataUnsupported { .. } => None,
        }
    }
}

fn load_document(path: &Path) -> Result<(File, Value, AppConfig), ConfigError> {
    let file = open_config_file(path)?;
    let mut source = String::new();
    (&file)
        .read_to_string(&mut source)
        .map_err(|source| io_error("read", path, source))?;
    let mut document: Value =
        serde_yaml::from_str(&source).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
    normalize_root_alias(path, &mut document, "device", "hardware")?;
    let mut config: AppConfig =
        serde_yaml::from_value(document.clone()).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
    mark_production_fields(&document, &mut config);
    config
        .validate_configured_adapters()
        .map_err(|source| ConfigError::Validation {
            path: path.to_owned(),
            source,
        })?;
    Ok((file, document, config))
}

fn mark_production_fields(document: &Value, config: &mut AppConfig) {
    config.pipeline.production_fields_explicit = section_has_fields(
        document,
        "pipeline",
        &[
            "freshness_threshold_ms",
            "near_threshold_px",
            "projection_fov_x_deg",
            "projection_counts_per_360",
            "projection_invert_y",
            "atan_scale_counts",
            "far_kp",
            "far_max_counts_per_update",
            "near_kp",
            "near_max_counts_per_update",
            "velocity_smoothing_frames",
            "velocity_history_reset_gap_ms",
            "velocity_spread_base_px_ms",
            "velocity_spread_relative",
            "velocity_change_base_px_ms",
            "velocity_change_relative",
            "prediction_lead_frames",
            "prediction_far_absolute_cap_px",
            "prediction_far_base_cap_px",
            "prediction_far_relative_cap",
            "prediction_near_absolute_cap_px",
            "prediction_near_base_cap_px",
            "prediction_near_relative_cap",
            "residual_cap",
            "target_debounce_distance_px",
            "target_min_confidence",
            "target_track_max_age",
            "tracker_max_match_distance",
            "tracker_position_cost_weight",
            "tracker_iou_cost_weight",
            "max_command_age_ms",
            "output_interval_ms",
        ],
    );
    if let Some(capture) = &mut config.capture {
        capture.production_fields_explicit = section_has_fields(
            document,
            "capture",
            &[
                "device",
                "backend",
                "memory",
                "preference",
                "latest_only",
                "appsink_max_buffers",
                "queue_leaky",
                "width",
                "height",
                "fps",
                "pixel_format",
                "roi_left",
                "roi_top",
                "roi_width",
                "roi_height",
            ],
        );
    }
    if let Some(inference) = &mut config.inference {
        inference.production_fields_explicit = section_has_fields(
            document,
            "inference",
            &[
                "enabled",
                "backend",
                "device",
                "require_gpu",
                "allow_cpu_fallback",
                "confidence_threshold",
                "nms_threshold",
                "inference_input_deadline_ms",
                "deepstream_parser_library",
                "deepstream_io_mode",
                "deepstream_batched_push_timeout_us",
                "deepstream_component_id",
                "deepstream_source_id",
                "deepstream_probe_element",
                "deepstream_probe_pad",
                "deepstream_nvinfer_config",
                "model_width",
                "model_height",
                "deepstream_startup_timeout_ms",
                "deepstream_shutdown_timeout_ms",
                "input_source",
            ],
        );
    }
    if let Some(device) = &mut config.device {
        let fields: &[&str] = match device.backend {
            super::DeviceBackend::NativeUdp => &[
                "auto_connect",
                "backend",
                "host",
                "port",
                "uuid",
                "monitor_port",
                "connect_timeout_ms",
                "send_timeout_ms",
                "monitor_timeout_ms",
                "trigger_poll_interval_ms",
            ],
            super::DeviceBackend::PythonHost => &[
                "auto_connect",
                "backend",
                "host",
                "port",
                "uuid",
                "monitor_port",
                "helper_module",
                "connect_timeout_ms",
                "send_timeout_ms",
                "reconnect_cooldown_ms",
                "trigger_poll_interval_ms",
            ],
        };
        device.production_fields_explicit = section_has_fields(document, "hardware", fields);
    }
}

fn section_has_fields(document: &Value, section: &str, fields: &[&str]) -> bool {
    document
        .get(section)
        .and_then(Value::as_mapping)
        .is_some_and(|mapping| {
            fields
                .iter()
                .all(|field| mapping.contains_key(Value::String((*field).to_owned())))
        })
}

fn normalize_root_alias(
    path: &Path,
    document: &mut Value,
    alias: &'static str,
    canonical: &'static str,
) -> Result<(), ConfigError> {
    let Some(mapping) = document.as_mapping_mut() else {
        return Ok(());
    };
    let alias_key = Value::String(alias.to_owned());
    let canonical_key = Value::String(canonical.to_owned());
    let Some(value) = mapping.remove(&alias_key) else {
        return Ok(());
    };
    if mapping.contains_key(&canonical_key) {
        return Err(ConfigError::Validation {
            path: path.to_owned(),
            source: ConfigValidationError {
                field: "device",
                message: "device and hardware cannot both be present".to_owned(),
            },
        });
    }
    mapping.insert(canonical_key, value);
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_config_file(path: &Path) -> Result<File, ConfigError> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|source| {
            if source.raw_os_error() == Some(libc::ELOOP) {
                ConfigError::SymlinkUnsupported {
                    path: path.to_owned(),
                }
            } else {
                read_error(path, source)
            }
        })
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_config_file(path: &Path) -> Result<File, ConfigError> {
    reject_symlink(path)?;
    File::open(path).map_err(|source| read_error(path, source))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn ensure_save_supported(_path: &Path) -> Result<(), ConfigError> {
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ensure_save_supported(path: &Path) -> Result<(), ConfigError> {
    Err(ConfigError::SecurityMetadataUnsupported {
        path: path.to_owned(),
        reason: "atomic replacement security metadata verification is unsupported on this platform"
            .to_owned(),
    })
}

fn save_document(
    path: &Path,
    config: &AppConfig,
    expected_revision: u64,
) -> Result<AppConfig, ConfigError> {
    ensure_save_supported(path)?;
    let parent = parent_directory(path);
    let directory = File::open(parent).map_err(|source| read_error(path, source))?;
    directory
        .lock()
        .map_err(|source| io_error("lock parent directory", parent, source))?;

    let (destination, mut document, current) = load_document(path)?;
    if current.revision != expected_revision {
        return Err(ConfigError::RevisionConflict {
            path: path.to_owned(),
            expected: expected_revision,
            actual: current.revision,
        });
    }
    config
        .validate_configured_adapters()
        .map_err(|source| ConfigError::Validation {
            path: path.to_owned(),
            source,
        })?;
    config
        .validate_explicit_adapter_fields()
        .map_err(|source| ConfigError::Validation {
            path: path.to_owned(),
            source,
        })?;
    validate_legacy_keys(path, config)?;
    let mut next = config.clone();
    next.revision =
        current
            .revision
            .checked_add(1)
            .ok_or_else(|| ConfigError::RevisionOverflow {
                path: path.to_owned(),
                revision: current.revision,
            })?;
    let replacement = serde_yaml::to_value(&next).map_err(|source| ConfigError::Serialize {
        path: path.to_owned(),
        source,
    })?;
    merge_value(&mut document, replacement);
    let mut persisted: AppConfig =
        serde_yaml::from_value(document.clone()).map_err(|source| ConfigError::Serialize {
            path: path.to_owned(),
            source,
        })?;
    mark_production_fields(&document, &mut persisted);
    let serialized = serde_yaml::to_string(&document).map_err(|source| ConfigError::Serialize {
        path: path.to_owned(),
        source,
    })?;

    atomic_replace(path, serialized.as_bytes(), &directory, &destination)?;
    Ok(persisted)
}

fn save_document_field(
    path: &Path,
    section: &str,
    key: &str,
    value: Value,
    expected_revision: u64,
) -> Result<AppConfig, ConfigError> {
    validate_field_target(path, section, key)?;
    ensure_save_supported(path)?;
    let parent = parent_directory(path);
    let directory = File::open(parent).map_err(|source| read_error(path, source))?;
    directory
        .lock()
        .map_err(|source| io_error("lock parent directory", parent, source))?;

    let (destination, mut document, current) = load_document(path)?;
    if current.revision != expected_revision {
        return Err(ConfigError::RevisionConflict {
            path: path.to_owned(),
            expected: expected_revision,
            actual: current.revision,
        });
    }
    let next_revision =
        current
            .revision
            .checked_add(1)
            .ok_or_else(|| ConfigError::RevisionOverflow {
                path: path.to_owned(),
                revision: current.revision,
            })?;
    let root = document
        .as_mapping_mut()
        .ok_or_else(|| ConfigError::InvalidFieldTarget {
            path: path.to_owned(),
            section: section.to_owned(),
            key: key.to_owned(),
            reason: "configuration root must be a mapping",
        })?;
    let section_key = Value::String(section.to_owned());
    let section_value = root
        .entry(section_key)
        .or_insert_with(|| Value::Mapping(Default::default()));
    let section_mapping =
        section_value
            .as_mapping_mut()
            .ok_or_else(|| ConfigError::InvalidFieldTarget {
                path: path.to_owned(),
                section: section.to_owned(),
                key: key.to_owned(),
                reason: "target section must be a mapping",
            })?;
    section_mapping.insert(Value::String(key.to_owned()), value);
    root.insert(
        Value::String("revision".to_owned()),
        Value::Number(next_revision.into()),
    );

    let mut persisted: AppConfig =
        serde_yaml::from_value(document.clone()).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
    mark_production_fields(&document, &mut persisted);
    persisted
        .validate_configured_adapters()
        .and_then(|()| persisted.validate_explicit_adapter_fields())
        .map_err(|source| ConfigError::Validation {
            path: path.to_owned(),
            source,
        })?;
    validate_legacy_keys(path, &persisted)?;
    let serialized = serde_yaml::to_string(&document).map_err(|source| ConfigError::Serialize {
        path: path.to_owned(),
        source,
    })?;
    atomic_replace(path, serialized.as_bytes(), &directory, &destination)?;
    Ok(persisted)
}

fn validate_field_target(path: &Path, section: &str, key: &str) -> Result<(), ConfigError> {
    let valid_component = |component: &str| {
        !component.is_empty()
            && component.len() <= 128
            && component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    };
    if !valid_component(section) || !valid_component(key) {
        return Err(ConfigError::InvalidFieldTarget {
            path: path.to_owned(),
            section: section.to_owned(),
            key: key.to_owned(),
            reason: "section and key must be simple non-empty names of at most 128 characters",
        });
    }
    if matches!(section, "schema_version" | "revision" | "device") {
        return Err(ConfigError::InvalidFieldTarget {
            path: path.to_owned(),
            section: section.to_owned(),
            key: key.to_owned(),
            reason: "root scalar and alias fields cannot be updated as sections",
        });
    }
    Ok(())
}

fn replace_document(
    path: &Path,
    mut replacement: Value,
    expected_revision: u64,
) -> Result<AppConfig, ConfigError> {
    if !replacement.is_mapping() {
        return Err(ConfigError::InvalidReplacementDocument {
            path: path.to_owned(),
            reason: "replacement must be a mapping",
        });
    }
    ensure_save_supported(path)?;
    let parent = parent_directory(path);
    let directory = File::open(parent).map_err(|source| read_error(path, source))?;
    directory
        .lock()
        .map_err(|source| io_error("lock parent directory", parent, source))?;
    let (destination, mut document, current) = load_document(path)?;
    if current.revision != expected_revision {
        return Err(ConfigError::RevisionConflict {
            path: path.to_owned(),
            expected: expected_revision,
            actual: current.revision,
        });
    }
    let next_revision =
        current
            .revision
            .checked_add(1)
            .ok_or_else(|| ConfigError::RevisionOverflow {
                path: path.to_owned(),
                revision: current.revision,
            })?;
    normalize_root_alias(path, &mut replacement, "device", "hardware")?;
    if let Some(mapping) = replacement.as_mapping_mut() {
        mapping.remove(Value::String("revision".to_owned()));
        mapping.remove(Value::String("schema_version".to_owned()));
    }
    merge_value(&mut document, replacement);
    let root =
        document
            .as_mapping_mut()
            .ok_or_else(|| ConfigError::InvalidReplacementDocument {
                path: path.to_owned(),
                reason: "persisted configuration root must be a mapping",
            })?;
    root.insert(
        Value::String("revision".to_owned()),
        Value::Number(next_revision.into()),
    );
    let mut persisted: AppConfig =
        serde_yaml::from_value(document.clone()).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
    mark_production_fields(&document, &mut persisted);
    persisted
        .validate_configured_adapters()
        .and_then(|()| persisted.validate_explicit_adapter_fields())
        .map_err(|source| ConfigError::Validation {
            path: path.to_owned(),
            source,
        })?;
    validate_legacy_keys(path, &persisted)?;
    let serialized = serde_yaml::to_string(&document).map_err(|source| ConfigError::Serialize {
        path: path.to_owned(),
        source,
    })?;
    atomic_replace(path, serialized.as_bytes(), &directory, &destination)?;
    Ok(persisted)
}

fn merge_value(document: &mut Value, replacement: Value) {
    match (document, replacement) {
        (Value::Mapping(document), Value::Mapping(replacement)) => {
            for (key, value) in replacement {
                match document.get_mut(&key) {
                    Some(existing) => merge_value(existing, value),
                    None => {
                        document.insert(key, value);
                    }
                }
            }
        }
        (document, replacement) => *document = replacement,
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn reject_symlink(path: &Path) -> Result<(), ConfigError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| read_error(path, source))?;
    if metadata.file_type().is_symlink() {
        Err(ConfigError::SymlinkUnsupported {
            path: path.to_owned(),
        })
    } else {
        Ok(())
    }
}

fn validate_legacy_keys(path: &Path, config: &AppConfig) -> Result<(), ConfigError> {
    for (section, legacy, reserved) in [
        (
            "root",
            &config.legacy,
            &[
                "schema_version",
                "revision",
                "server",
                "replay",
                "paths",
                "capture",
                "inference",
                "hardware",
                "device",
            ][..],
        ),
        ("server", &config.server.legacy, &["host", "port"][..]),
        (
            "replay",
            &config.replay.legacy,
            &["enabled", "frame_interval_ms", "output_gate_open"][..],
        ),
        (
            "paths",
            &config.paths.legacy,
            &[
                "data_dir",
                "model_dir",
                "database",
                "license",
                "python_executable",
            ][..],
        ),
    ] {
        if let Some(key) = reserved.iter().find(|key| legacy.contains_key(**key)) {
            return Err(ConfigError::ReservedLegacyKey {
                path: path.to_owned(),
                section,
                key: (*key).to_owned(),
            });
        }
    }
    if let Some(capture) = &config.capture {
        validate_reserved_legacy(
            path,
            "capture",
            &capture.legacy,
            &[
                "device",
                "backend",
                "memory",
                "preference",
                "latest_only",
                "appsink_max_buffers",
                "queue_leaky",
                "width",
                "height",
                "fps",
                "pixel_format",
            ],
        )?;
    }
    if let Some(inference) = &config.inference {
        validate_reserved_legacy(
            path,
            "inference",
            &inference.legacy,
            &[
                "enabled",
                "backend",
                "device",
                "require_gpu",
                "allow_cpu_fallback",
                "confidence_threshold",
                "nms_threshold",
                "inference_input_deadline_ms",
                "deepstream_parser_library",
                "deepstream_io_mode",
                "deepstream_batched_push_timeout_us",
                "deepstream_component_id",
                "deepstream_probe_element",
                "deepstream_probe_pad",
                "input_source",
            ],
        )?;
    }
    if let Some(device) = &config.device {
        validate_reserved_legacy(
            path,
            "hardware",
            &device.legacy,
            &[
                "auto_connect",
                "backend",
                "host",
                "port",
                "uuid",
                "monitor_port",
                "helper_module",
                "connect_timeout_ms",
                "send_timeout_ms",
                "monitor_timeout_ms",
                "trigger_poll_interval_ms",
                "reconnect_cooldown_ms",
            ],
        )?;
    }
    Ok(())
}

fn validate_reserved_legacy(
    path: &Path,
    section: &'static str,
    legacy: &BTreeMap<String, Value>,
    reserved: &[&str],
) -> Result<(), ConfigError> {
    if let Some(key) = reserved.iter().find(|key| legacy.contains_key(**key)) {
        Err(ConfigError::ReservedLegacyKey {
            path: path.to_owned(),
            section,
            key: (*key).to_owned(),
        })
    } else {
        Ok(())
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn validate_security_metadata(
    path: &Path,
    destination: &File,
    temporary: &File,
) -> Result<(), ConfigError> {
    let destination_metadata = read_security_metadata(destination, path)?;
    let temporary_metadata = read_security_metadata(temporary, path)?;
    if destination_metadata == temporary_metadata {
        Ok(())
    } else {
        Err(ConfigError::SecurityMetadataUnsupported {
            path: path.to_owned(),
            reason: "destination owner, group, mode, ACL, inode flags, or extended attributes differ from the atomic replacement".to_owned(),
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn validate_security_metadata(
    path: &Path,
    _destination: &File,
    _temporary: &File,
) -> Result<(), ConfigError> {
    ensure_save_supported(path)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(PartialEq)]
struct SecurityMetadata {
    owner: u32,
    group: u32,
    mode: u32,
    acl: Vec<exacl::AclEntry>,
    inode_flags: u32,
    extended_attributes: BTreeMap<OsString, Vec<u8>>,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn read_security_metadata(file: &File, path: &Path) -> Result<SecurityMetadata, ConfigError> {
    let metadata = file
        .metadata()
        .map_err(|source| security_metadata_error(path, "inspect file metadata", source))?;
    if metadata.nlink() != 1 {
        return Err(ConfigError::HardlinkUnsupported {
            path: path.to_owned(),
            links: metadata.nlink(),
        });
    }
    let acl = exacl::getfacl(descriptor_path(file), None)
        .map_err(|source| security_metadata_error(path, "inspect access-control list", source))?;
    let attributes = file
        .list_xattr()
        .map_err(|source| security_metadata_error(path, "list extended attributes", source))?;
    let mut extended_attributes = BTreeMap::new();
    for name in attributes {
        let value = file
            .get_xattr(&name)
            .map_err(|source| security_metadata_error(path, "read extended attribute", source))?
            .ok_or_else(|| ConfigError::SecurityMetadataUnsupported {
                path: path.to_owned(),
                reason: format!(
                    "extended attribute {:?} disappeared while it was inspected",
                    name
                ),
            })?;
        extended_attributes.insert(name, value);
    }

    Ok(SecurityMetadata {
        owner: metadata.uid(),
        group: metadata.gid(),
        mode: metadata.mode(),
        acl,
        inode_flags: inode_flags(file, &metadata, path)?,
        extended_attributes,
    })
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn inode_flags(file: &File, _metadata: &fs::Metadata, path: &Path) -> Result<u32, ConfigError> {
    let mut flags: libc::c_long = 0;
    // SAFETY: libc::FS_IOC_GETFLAGS is generated from the target Linux UAPI's
    // architecture-correct _IOR('f', 1, long) encoding. `file` owns a live
    // descriptor and `flags` is a correctly sized writable c_long for the
    // kernel to initialize. The ioctl does not retain the pointer.
    let result = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            libc::FS_IOC_GETFLAGS,
            &mut flags as *mut libc::c_long,
        )
    };
    if result < 0 {
        Err(security_metadata_error(
            path,
            "inspect inode flags",
            io::Error::last_os_error(),
        ))
    } else {
        Ok(flags as u32)
    }
}

#[cfg(target_os = "macos")]
fn inode_flags(_file: &File, metadata: &fs::Metadata, _path: &Path) -> Result<u32, ConfigError> {
    Ok(metadata.st_flags())
}

#[cfg(target_os = "linux")]
fn descriptor_path(file: &File) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

#[cfg(target_os = "macos")]
fn descriptor_path(file: &File) -> PathBuf {
    PathBuf::from(format!("/dev/fd/{}", file.as_raw_fd()))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn verify_path_identity(path: &Path, destination: &File) -> Result<(), ConfigError> {
    let current = open_config_file(path)?;
    let expected = destination
        .metadata()
        .map_err(|source| security_metadata_error(path, "inspect opened identity", source))?;
    let actual = current
        .metadata()
        .map_err(|source| security_metadata_error(path, "inspect path identity", source))?;
    if actual.nlink() != 1 {
        return Err(ConfigError::HardlinkUnsupported {
            path: path.to_owned(),
            links: actual.nlink(),
        });
    }
    if expected.dev() == actual.dev() && expected.ino() == actual.ino() {
        Ok(())
    } else {
        Err(ConfigError::PathChanged {
            path: path.to_owned(),
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn verify_path_identity(path: &Path, _destination: &File) -> Result<(), ConfigError> {
    ensure_save_supported(path)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn security_metadata_error(path: &Path, operation: &'static str, source: io::Error) -> ConfigError {
    ConfigError::SecurityMetadataUnsupported {
        path: path.to_owned(),
        reason: format!("{operation} failed: {source}"),
    }
}

fn atomic_replace(
    path: &Path,
    contents: &[u8],
    directory: &File,
    destination: &File,
) -> Result<(), ConfigError> {
    let parent = parent_directory(path);
    let permissions = destination
        .metadata()
        .map_err(|source| io_error("inspect opened metadata", path, source))?
        .permissions();
    let (temporary_path, mut temporary_file) = create_temporary_file(parent, path.file_name())?;
    let mut cleanup = TemporaryCleanup::new(temporary_path.clone());

    temporary_file
        .set_permissions(permissions)
        .map_err(|source| io_error("set temporary permissions", &temporary_path, source))?;
    validate_security_metadata(path, destination, &temporary_file)?;
    temporary_file
        .write_all(contents)
        .map_err(|source| io_error("write temporary file", &temporary_path, source))?;
    temporary_file
        .flush()
        .map_err(|source| io_error("flush temporary file", &temporary_path, source))?;
    temporary_file
        .sync_all()
        .map_err(|source| io_error("sync temporary file", &temporary_path, source))?;
    validate_security_metadata(path, destination, &temporary_file)?;
    verify_path_identity(path, destination)?;

    fs::rename(&temporary_path, path).map_err(|source| io_error("replace", path, source))?;
    drop(temporary_file);
    cleanup.disarm();

    directory
        .sync_all()
        .map_err(|source| io_error("sync parent directory", parent, source))?;
    Ok(())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn create_temporary_file(
    parent: &Path,
    destination_name: Option<&OsStr>,
) -> Result<(PathBuf, File), ConfigError> {
    let destination_name = destination_name.unwrap_or_else(|| OsStr::new("config"));
    for sequence in 0_u8..=u8::MAX {
        let mut temporary_name = OsString::from(".");
        temporary_name.push(destination_name);
        temporary_name.push(format!(".tmp.{}.{sequence}", std::process::id()));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(io_error("create temporary file", &temporary_path, source));
            }
        }
    }

    Err(io_error(
        "create temporary file",
        parent,
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "all temporary configuration names are occupied",
        ),
    ))
}

fn read_error(path: &Path, source: io::Error) -> ConfigError {
    if source.kind() == io::ErrorKind::NotFound {
        ConfigError::NotFound {
            path: path.to_owned(),
        }
    } else {
        io_error("read", path, source)
    }
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> ConfigError {
    ConfigError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

struct TemporaryCleanup {
    path: Option<PathBuf>,
}

impl TemporaryCleanup {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for TemporaryCleanup {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;

    #[test]
    fn path_identity_check_rejects_an_inode_replaced_after_open() {
        let directory = std::env::temp_dir().join(format!(
            "novasight-config-identity-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("novasight.yaml");
        let replacement = directory.join("replacement.yaml");
        fs::write(&path, "revision: 0\n").unwrap();
        fs::write(&replacement, "revision: 1\n").unwrap();
        let opened = open_config_file(&path).unwrap();
        fs::rename(&replacement, &path).unwrap();

        let error = verify_path_identity(&path, &opened).unwrap_err();

        assert_eq!(error.code(), "CONFIG_PATH_CHANGED");
        fs::remove_dir_all(directory).unwrap();
    }
}
