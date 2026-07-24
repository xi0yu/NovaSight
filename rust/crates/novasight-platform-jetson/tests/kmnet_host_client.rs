use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use novasight_core::{DeviceCommand, Generation, MonotonicNanos, PointerDevice, RuntimeEpoch};
use novasight_platform_jetson::kmnet::{KmNetHostClient, KmNetHostConfig};

const GOOD_HELPER: &str = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    ok = (
        op == 'hello'
        or (op == 'connect' and request['host'] == '192.0.2.10' and request['port'] == 8888 and request['uuid'] == 'box')
        or (op == 'move' and request['dx'] == 120 and request['dy'] == -45)
        or op == 'shutdown'
    )
    print(json.dumps({'id': request['id'], 'ok': ok, 'result': {'protocol': 1}, 'error': None if ok else 'unexpected request'}), flush=True)
    if op == 'shutdown':
        break
"#;

fn config(script: &str) -> KmNetHostConfig {
    KmNetHostConfig {
        program: PathBuf::from("python3"),
        args: vec![
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ],
        working_directory: std::env::current_dir().expect("current directory"),
        host: "192.0.2.10".to_owned(),
        port: 8888,
        uuid: "box".to_owned(),
        monitor_port: 0,
        startup_timeout: Duration::from_secs(1),
        request_timeout: Duration::from_millis(100),
        reconnect_cooldown: Duration::from_millis(30),
    }
}

fn command() -> DeviceCommand {
    DeviceCommand {
        epoch: RuntimeEpoch(4),
        generation: Generation(9),
        source_captured_at: MonotonicNanos(100),
        issued_at: MonotonicNanos(100),
        target_object_id: 7,
        delta_x_counts: 120,
        delta_y_counts: -45,
    }
}

#[test]
fn performs_handshake_connect_and_real_move_round_trip() {
    let client = KmNetHostClient::connect(config(GOOD_HELPER)).expect("connect helper");

    let receipt = client.send(command()).expect("send movement");

    assert_eq!(receipt.epoch, RuntimeEpoch(4));
    assert_eq!(receipt.generation, Generation(9));
    assert_eq!(receipt.delta_x_counts, 120);
    assert_eq!(receipt.delta_y_counts, -45);
}

#[test]
fn preflight_proves_helper_protocol_without_connecting_hardware() {
    let script = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'connect':
        print(json.dumps({'id': request['id'], 'ok': False, 'result': None, 'error': 'hardware must not be opened by preflight'}), flush=True)
        continue
    result = {'protocol': 1, 'driver_available': True} if op == 'hello' else {}
    print(json.dumps({'id': request['id'], 'ok': op in ('hello', 'shutdown'), 'result': result, 'error': None}), flush=True)
    if op == 'shutdown':
        break
"#;
    let config = config(script);

    KmNetHostClient::preflight(&config).expect("helper-only preflight");
}

#[test]
fn helper_module_is_resolved_from_the_pinned_working_directory() {
    let root = std::env::temp_dir().join(format!(
        "novasight-kmnet-helper-root-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("pinned_helper.py"),
        r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    result = {'protocol': 1, 'driver_available': True} if op == 'hello' else {}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result, 'error': None}), flush=True)
    if op == 'shutdown':
        break
"#,
    )
    .unwrap();
    let mut config = config(GOOD_HELPER);
    config.args = vec![
        OsString::from("-u"),
        OsString::from("-m"),
        OsString::from("pinned_helper"),
    ];
    config.working_directory = root.clone();

    KmNetHostClient::preflight(&config).expect("module from pinned release root");

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn preflight_rejects_a_packaged_helper_with_the_wrong_protocol() {
    let script = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'id': request['id'], 'ok': True, 'result': {'protocol': 2}, 'error': None}), flush=True)
"#;
    let config = config(script);

    let error = KmNetHostClient::preflight(&config).expect_err("protocol drift must fail");

    assert!(error.to_string().contains("protocol version mismatch"));
}

#[test]
fn preflight_rejects_a_helper_without_the_vendor_driver() {
    let script = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'id': request['id'], 'ok': True, 'result': {'protocol': 1, 'driver_available': False}, 'error': None}), flush=True)
"#;
    let config = config(script);

    let error = KmNetHostClient::preflight(&config).expect_err("missing driver must fail");

    assert_eq!(error.code(), "driver_unavailable");
}

#[test]
fn lazy_adapter_defers_helper_start_until_runtime_connect() {
    let mut lazy = config(GOOD_HELPER);
    lazy.program = PathBuf::from("/definitely/missing/novasight-kmnet-helper");

    let client = KmNetHostClient::new(lazy).expect("configuration is valid without hardware");
    let error = PointerDevice::connect(&client).expect_err("runtime connect starts helper");

    assert!(error.to_string().contains("failed to spawn"));
}

#[test]
fn disconnect_releases_the_helper_and_next_epoch_reconnects() {
    let client = KmNetHostClient::connect(config(GOOD_HELPER)).expect("connect helper");

    PointerDevice::disconnect(&client).expect("disconnect helper");
    PointerDevice::connect(&client).expect("reconnect helper");
    let receipt = client.send(command()).expect("send after reconnect");

    assert_eq!(receipt.generation, Generation(9));
}

#[test]
fn timed_out_driver_is_aborted_and_cooldown_is_nonblocking() {
    let script = GOOD_HELPER.replace(
        "or (op == 'move' and request['dx'] == 120 and request['dy'] == -45)",
        "or (op == 'move' and (__import__('time').sleep(2) is None))",
    );
    let client = KmNetHostClient::connect(config(&script)).expect("connect helper");

    let started = Instant::now();
    let first = client.send(command()).expect_err("move must time out");
    assert!(first.to_string().contains("timed out"));
    assert!(started.elapsed() < Duration::from_secs(1));

    let retry_started = Instant::now();
    let retry = client
        .send(command())
        .expect_err("cooldown must reject retry");
    assert!(retry.to_string().contains("cooldown"));
    assert!(retry_started.elapsed() < Duration::from_millis(20));
}

#[test]
fn crashed_helper_reconnects_after_cooldown() {
    let marker =
        std::env::temp_dir().join(format!("novasight-kmnet-reconnect-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let quoted = serde_json::to_string(&marker).expect("marker JSON");
    let script = format!(
        r#"
import json, os, sys
marker = {quoted}
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'move' and not os.path.exists(marker):
        open(marker, 'w').close()
        os._exit(17)
    print(json.dumps({{'id': request['id'], 'ok': True, 'result': {{'protocol': 1}}, 'error': None}}), flush=True)
"#
    );
    let client = KmNetHostClient::connect(config(&script)).expect("connect helper");
    client.send(command()).expect_err("first helper crashes");
    std::thread::sleep(Duration::from_millis(40));

    let receipt = client.send(command()).expect("reconnected move");

    assert_eq!(receipt.target_object_id, 7);
    let _ = std::fs::remove_file(marker);
}

#[test]
fn rejects_counts_outside_the_kmnet_signed_16_bit_contract() {
    let client = KmNetHostClient::connect(config(GOOD_HELPER)).expect("connect helper");
    let mut invalid = command();
    invalid.delta_x_counts = i32::from(i16::MAX) + 1;

    let error = client.send(invalid).expect_err("out of range");

    assert!(error.to_string().contains("signed 16-bit"));
}

#[test]
fn unavailable_button_api_is_fail_closed_and_destroys_the_session() {
    let script = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    result = {'protocol': 1} if op == 'hello' else ({'available': False, 'left': False, 'right': False} if op == 'buttons' else {})
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result, 'error': None}), flush=True)
"#;
    let client = KmNetHostClient::connect(config(script)).expect("connect helper");

    let error = client
        .trigger_active()
        .expect_err("missing hardware trigger must fail closed");
    assert!(error.to_string().contains("does not expose"));
    let retry = client
        .trigger_active()
        .expect_err("failed protocol session enters cooldown");
    assert!(retry.to_string().contains("cooldown"));
}

#[test]
fn preserves_individual_button_state_from_the_helper() {
    let script = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'hello':
        result = {'protocol': 1}
    elif op == 'buttons':
        result = {'available': True, 'left': False, 'right': True}
    else:
        result = {}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result, 'error': None}), flush=True)
"#;
    let client = KmNetHostClient::connect(config(script)).expect("connect helper");

    let buttons = client.buttons().expect("read buttons").expect("available");

    assert!(!buttons.left);
    assert!(buttons.right);
    assert!(buttons.trigger_active());
}
