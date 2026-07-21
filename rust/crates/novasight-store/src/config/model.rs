use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub replay: ReplayConfig,
    #[serde(default)]
    pub paths: PathConfig,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            revision: 0,
            server: ServerConfig::default(),
            replay: ReplayConfig::default(),
            paths: PathConfig::default(),
            legacy: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub host: String,
    #[serde(default = "default_server_port")]
    pub port: u16,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
            legacy: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayConfig {
    #[serde(default = "default_replay_enabled")]
    pub enabled: bool,
    #[serde(default = "default_frame_interval_ms")]
    pub frame_interval_ms: u64,
    #[serde(default)]
    pub output_gate_open: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enabled: default_replay_enabled(),
            frame_interval_ms: default_frame_interval_ms(),
            output_gate_open: false,
            legacy: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default = "default_database")]
    pub database: PathBuf,
    #[serde(default = "default_license")]
    pub license: PathBuf,
    #[serde(default = "default_python_executable")]
    pub python_executable: PathBuf,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            model_dir: default_model_dir(),
            database: default_database(),
            license: default_license(),
            python_executable: default_python_executable(),
            legacy: BTreeMap::new(),
        }
    }
}

const fn default_schema_version() -> u32 {
    1
}

fn default_server_host() -> String {
    "0.0.0.0".to_owned()
}

const fn default_server_port() -> u16 {
    5174
}

const fn default_replay_enabled() -> bool {
    true
}

const fn default_frame_interval_ms() -> u64 {
    16
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}

fn default_model_dir() -> PathBuf {
    PathBuf::from("data/models")
}

fn default_database() -> PathBuf {
    PathBuf::from("data/novasight.db")
}

fn default_license() -> PathBuf {
    PathBuf::from("data/license.json")
}

fn default_python_executable() -> PathBuf {
    PathBuf::from("python3")
}
