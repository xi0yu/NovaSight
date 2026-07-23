use std::fs;
use std::path::PathBuf;

use novasight_runtime::{
    ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate, RuntimeHandle,
};
use novasight_store::config::YamlConfigRepository;
use serde_json::Value;

pub struct TestConfig {
    root: PathBuf,
    service: ConfigService,
}

impl TestConfig {
    pub fn commissioned(output_enabled: bool) -> Self {
        let root = std::env::temp_dir().join(format!(
            "novasight-runtime-test-config-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&root).unwrap();
        let path = root.join("novasight.yaml");
        fs::write(
            &path,
            format!(
                "revision: 0\ncontrol:\n  output_enabled: {output_enabled}\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n"
            ),
        )
        .unwrap();
        let initial = YamlConfigRepository::load(&path).unwrap();
        Self {
            root,
            service: ConfigService::new(path, initial),
        }
    }

    pub async fn set_output(
        &self,
        runtime: &RuntimeHandle,
        enabled: bool,
    ) -> Result<ConfigUpdate, ConfigServiceError> {
        self.service
            .update_output_gate(
                runtime,
                ConfigFieldUpdate {
                    section: "control".to_owned(),
                    key: "output_enabled".to_owned(),
                    value: Value::Bool(enabled),
                    expected_revision: None,
                },
            )
            .await
    }
}

impl Drop for TestConfig {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
