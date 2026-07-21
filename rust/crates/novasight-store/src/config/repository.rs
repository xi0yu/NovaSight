use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_yaml::Value;

use super::AppConfig;

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
        load_document(path.as_ref()).map(|(_, config)| config)
    }

    pub fn save(
        path: impl AsRef<Path>,
        config: &AppConfig,
        expected_revision: u64,
    ) -> Result<AppConfig, ConfigError> {
        save_document(path.as_ref(), config, expected_revision)
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
}

impl ConfigError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "CONFIG_NOT_FOUND",
            Self::Io { .. } => "CONFIG_IO_ERROR",
            Self::Parse { .. } => "CONFIG_PARSE_ERROR",
            Self::Serialize { .. } => "CONFIG_SERIALIZE_ERROR",
            Self::RevisionConflict { .. } => "CONFIG_REVISION_CONFLICT",
            Self::RevisionOverflow { .. } => "CONFIG_REVISION_OVERFLOW",
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::NotFound { path }
            | Self::Io { path, .. }
            | Self::Parse { path, .. }
            | Self::Serialize { path, .. }
            | Self::RevisionConflict { path, .. }
            | Self::RevisionOverflow { path, .. } => path,
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
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } | Self::Serialize { source, .. } => Some(source),
            Self::NotFound { .. }
            | Self::RevisionConflict { .. }
            | Self::RevisionOverflow { .. } => None,
        }
    }
}

fn load_document(path: &Path) -> Result<(Value, AppConfig), ConfigError> {
    let source = fs::read_to_string(path).map_err(|source| read_error(path, source))?;
    let document: Value = serde_yaml::from_str(&source).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        source,
    })?;
    let config = serde_yaml::from_value(document.clone()).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        source,
    })?;
    Ok((document, config))
}

fn save_document(
    path: &Path,
    config: &AppConfig,
    expected_revision: u64,
) -> Result<AppConfig, ConfigError> {
    let parent = parent_directory(path);
    let directory = File::open(parent).map_err(|source| read_error(path, source))?;
    directory
        .lock()
        .map_err(|source| io_error("lock parent directory", parent, source))?;

    let (mut document, current) = load_document(path)?;
    if current.revision != expected_revision {
        return Err(ConfigError::RevisionConflict {
            path: path.to_owned(),
            expected: expected_revision,
            actual: current.revision,
        });
    }

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
    let serialized = serde_yaml::to_string(&document).map_err(|source| ConfigError::Serialize {
        path: path.to_owned(),
        source,
    })?;

    atomic_replace(path, serialized.as_bytes(), &directory)?;
    Ok(next)
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

fn atomic_replace(path: &Path, contents: &[u8], directory: &File) -> Result<(), ConfigError> {
    let parent = parent_directory(path);
    let permissions = fs::metadata(path)
        .map_err(|source| io_error("metadata", path, source))?
        .permissions();
    let (temporary_path, mut temporary_file) = create_temporary_file(parent, path.file_name())?;
    let mut cleanup = TemporaryCleanup::new(temporary_path.clone());

    fs::set_permissions(&temporary_path, permissions)
        .map_err(|source| io_error("set temporary permissions", &temporary_path, source))?;
    temporary_file
        .write_all(contents)
        .map_err(|source| io_error("write temporary file", &temporary_path, source))?;
    temporary_file
        .flush()
        .map_err(|source| io_error("flush temporary file", &temporary_path, source))?;
    temporary_file
        .sync_all()
        .map_err(|source| io_error("sync temporary file", &temporary_path, source))?;
    drop(temporary_file);

    fs::rename(&temporary_path, path).map_err(|source| io_error("replace", path, source))?;
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
