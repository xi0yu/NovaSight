//! Native kmBoxNet UDP adapter.
//!
//! This is an independent Rust implementation of the public device protocol.
//! It intentionally does not compile or redistribute the vendor's Windows C++
//! source. The daemon owns timeouts, response validation and monitor lifetime.

use std::io;
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use novasight_core::{AppError, DeviceCommand, DeviceReceipt, PointerButtons, PointerDevice};
use thiserror::Error;

const CMD_CONNECT: u32 = 0xaf3c_2828;
const CMD_MOUSE_MOVE: u32 = 0xaede_7345;
const CMD_MONITOR: u32 = 0x2738_8020;
const HEADER_LEN: usize = 16;
const MOVE_PACKET_LEN: usize = HEADER_LEN + 56;
// Upstream's packed monitor datagram is one 8-byte mouse report followed by
// one 12-byte keyboard report.  We only consume the mouse buttons today, but
// accepting a truncated datagram would turn corruption into trigger state.
const MONITOR_PACKET_LEN: usize = 20;
const MONITOR_PORT_RANGE: std::ops::RangeInclusive<u16> = 1024..=49_151;
const IDEMPOTENT_SETUP_ATTEMPTS: usize = 3;

#[derive(Clone, Debug)]
pub struct KmNetNativeConfig {
    pub host: Ipv4Addr,
    pub port: u16,
    pub uuid: String,
    pub monitor_port: u16,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub monitor_timeout: Duration,
}

#[derive(Debug)]
pub struct KmNetNativeDevice {
    config: KmNetNativeConfig,
    session: Mutex<Option<KmNetNativeSession>>,
    successful_sends: AtomicU64,
}

#[derive(Debug)]
struct KmNetNativeSession {
    control: Mutex<ControlSocket>,
    buttons: Arc<AtomicU8>,
    monitor_healthy: Arc<AtomicBool>,
    last_monitor_update: Arc<Mutex<Option<Instant>>>,
    monitor_timeout: Duration,
    monitor_port: u16,
    stop: Arc<AtomicBool>,
    monitor: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Debug)]
struct ControlSocket {
    socket: UdpSocket,
    mac: u32,
    sequence: u32,
    random_state: u32,
}

impl KmNetNativeDevice {
    /// Build a disconnected adapter. Network resources are acquired by the
    /// runtime epoch through [`PointerDevice::connect`].
    pub fn new(config: KmNetNativeConfig) -> Result<Self, KmNetNativeError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            session: Mutex::new(None),
            successful_sends: AtomicU64::new(0),
        })
    }
}

impl KmNetNativeSession {
    fn connect(config: &KmNetNativeConfig) -> Result<Self, KmNetNativeError> {
        validate_config(config)?;
        let mac = parse_uuid(&config.uuid)?;
        let remote = SocketAddrV4::new(config.host, config.port);
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(KmNetNativeError::Bind)?;
        socket.connect(remote).map_err(KmNetNativeError::Connect)?;
        socket
            .set_read_timeout(Some(
                config.connect_timeout / u32::try_from(IDEMPOTENT_SETUP_ATTEMPTS).unwrap(),
            ))
            .map_err(KmNetNativeError::Configure)?;
        let mut control = ControlSocket {
            socket,
            mac,
            // The public reference starts the first connect request at zero.
            sequence: u32::MAX,
            random_state: random_seed(mac),
        };
        control.exchange_idempotent(CMD_CONNECT, None, &[], "connect")?;

        let monitor_socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, config.monitor_port))
            .map_err(KmNetNativeError::MonitorBind)?;
        monitor_socket
            .set_read_timeout(Some(config.request_timeout))
            .map_err(KmNetNativeError::Configure)?;
        let monitor_value = u32::from(config.monitor_port) | (0xaa55_u32 << 16);
        control.exchange_idempotent(CMD_MONITOR, Some(monitor_value), &[], "monitor")?;
        control
            .socket
            .set_read_timeout(Some(config.request_timeout))
            .map_err(KmNetNativeError::Configure)?;

        let buttons = Arc::new(AtomicU8::new(0));
        let monitor_healthy = Arc::new(AtomicBool::new(true));
        let last_monitor_update = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let monitor = spawn_monitor(
            monitor_socket,
            Arc::clone(&buttons),
            Arc::clone(&monitor_healthy),
            Arc::clone(&last_monitor_update),
            Arc::clone(&stop),
            config.host,
        )?;
        Ok(Self {
            control: Mutex::new(control),
            buttons,
            monitor_healthy,
            last_monitor_update,
            monitor_timeout: config.monitor_timeout,
            monitor_port: config.monitor_port,
            stop,
            monitor: Mutex::new(Some(monitor)),
        })
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Wake recv_from immediately; production shutdown must not inherit the
        // device request timeout.
        if let Ok(waker) = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)) {
            let _ = waker.send_to(&[0], (Ipv4Addr::LOCALHOST, self.monitor_port));
        }
        if let Some(handle) = self
            .monitor
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }
}

fn validate_config(config: &KmNetNativeConfig) -> Result<(), KmNetNativeError> {
    if config.port == 0 {
        return Err(KmNetNativeError::InvalidConfig(
            "control port must be non-zero",
        ));
    }
    if !MONITOR_PORT_RANGE.contains(&config.monitor_port) {
        return Err(KmNetNativeError::InvalidConfig(
            "monitor port must be within the vendor range 1024..=49151",
        ));
    }
    if config.connect_timeout.is_zero()
        || config.request_timeout.is_zero()
        || config.monitor_timeout.is_zero()
    {
        return Err(KmNetNativeError::InvalidConfig("timeouts must be non-zero"));
    }
    parse_uuid(&config.uuid)?;
    Ok(())
}

impl PointerDevice for KmNetNativeDevice {
    fn connect(&self) -> Result<(), AppError> {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if session.is_none() {
            *session = Some(
                KmNetNativeSession::connect(&self.config)
                    .map_err(|error| pointer_error(error.code(), error.to_string()))?,
            );
        }
        Ok(())
    }

    fn send(&self, command: DeviceCommand) -> Result<DeviceReceipt, AppError> {
        let dx = i16::try_from(command.delta_x_counts).map_err(|_| {
            pointer_error(
                "counts_out_of_range",
                format!("kmNet x movement {} is outside i16", command.delta_x_counts),
            )
        })?;
        let dy = i16::try_from(command.delta_y_counts).map_err(|_| {
            pointer_error(
                "counts_out_of_range",
                format!("kmNet y movement {} is outside i16", command.delta_y_counts),
            )
        })?;
        let mut payload = [0_u8; 56];
        payload[4..8].copy_from_slice(&i32::from(dx).to_le_bytes());
        payload[8..12].copy_from_slice(&i32::from(dy).to_le_bytes());
        let session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = session.as_ref().ok_or_else(|| {
            pointer_error(
                "not_connected",
                "native kmNet session is not connected".to_owned(),
            )
        })?;
        session
            .control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .exchange(CMD_MOUSE_MOVE, None, &payload, "move")
            .map_err(|error| pointer_error(error.code(), error.to_string()))?;
        let attempt = self.successful_sends.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(DeviceReceipt::accepted(attempt, command))
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        self.buttons()
            .map(|buttons| buttons.map(PointerButtons::trigger_active))
    }

    fn buttons(&self) -> Result<Option<PointerButtons>, AppError> {
        let session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = session.as_ref().ok_or_else(|| {
            pointer_error(
                "not_connected",
                "native kmNet session is not connected".to_owned(),
            )
        })?;
        read_monitor_buttons(
            &session.monitor_healthy,
            &session.last_monitor_update,
            &session.buttons,
            session.monitor_timeout,
        )
        .map(Some)
    }

    fn disconnect(&self) -> Result<(), AppError> {
        if let Some(mut session) = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            session.shutdown();
        }
        Ok(())
    }
}

impl Drop for KmNetNativeDevice {
    fn drop(&mut self) {
        if let Some(mut session) = self
            .session
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            session.shutdown();
        }
    }
}

impl ControlSocket {
    fn exchange(
        &mut self,
        command: u32,
        random_override: Option<u32>,
        payload: &[u8],
        operation: &'static str,
    ) -> Result<(), KmNetNativeError> {
        self.exchange_with_attempts(command, random_override, payload, operation, 1)
    }

    fn exchange_idempotent(
        &mut self,
        command: u32,
        random_override: Option<u32>,
        payload: &[u8],
        operation: &'static str,
    ) -> Result<(), KmNetNativeError> {
        self.exchange_with_attempts(
            command,
            random_override,
            payload,
            operation,
            IDEMPOTENT_SETUP_ATTEMPTS,
        )
    }

    fn exchange_with_attempts(
        &mut self,
        command: u32,
        random_override: Option<u32>,
        payload: &[u8],
        operation: &'static str,
        attempts: usize,
    ) -> Result<(), KmNetNativeError> {
        self.sequence = self.sequence.wrapping_add(1);
        let random = random_override.unwrap_or_else(|| self.next_random());
        let packet = encode_packet(self.mac, random, self.sequence, command, payload);
        debug_assert!(packet.len() == HEADER_LEN || packet.len() == MOVE_PACKET_LEN);
        for attempt in 0..attempts {
            self.socket
                .send(&packet)
                .map_err(|source| KmNetNativeError::Send { operation, source })?;
            loop {
                let mut response = [0_u8; 1024];
                let received = match self.socket.recv(&mut response) {
                    Ok(received) => received,
                    Err(source)
                        if attempt + 1 < attempts
                            && matches!(
                                source.kind(),
                                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                            ) =>
                    {
                        break;
                    }
                    Err(source) => return Err(KmNetNativeError::Receive { operation, source }),
                };
                if received < HEADER_LEN {
                    return Err(KmNetNativeError::ShortResponse {
                        operation,
                        received,
                    });
                }
                let response_sequence = u32::from_le_bytes(response[8..12].try_into().unwrap());
                let response_command = u32::from_le_bytes(response[12..16].try_into().unwrap());
                let sequence_age = self.sequence.wrapping_sub(response_sequence);
                if sequence_age != 0 && sequence_age < (1_u32 << 31) {
                    continue;
                }
                if response_sequence != self.sequence || response_command != command {
                    return Err(KmNetNativeError::ResponseMismatch {
                        operation,
                        expected_sequence: self.sequence,
                        actual_sequence: response_sequence,
                        expected_command: command,
                        actual_command: response_command,
                    });
                }
                return Ok(());
            }
        }
        unreachable!("exchange attempts are always non-zero")
    }

    fn next_random(&mut self) -> u32 {
        // The vendor protocol refreshes this field for ordinary commands.  It
        // is packet diversity, not an authentication primitive, so a local
        // xorshift stream is sufficient and avoids adding a crypto dependency
        // to the realtime device adapter.
        let mut value = self.random_state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        if value == 0 {
            value = 0xa5a5_5a5a;
        }
        self.random_state = value;
        value
    }
}

fn random_seed(mac: u32) -> u32 {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let folded_time = since_epoch as u32 ^ (since_epoch >> 32) as u32;
    let seed = folded_time ^ std::process::id().rotate_left(11) ^ mac.rotate_right(7);
    if seed == 0 { 0x6d2b_79f5 } else { seed }
}

fn encode_packet(mac: u32, random: u32, sequence: u32, command: u32, payload: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(HEADER_LEN + payload.len());
    packet.extend_from_slice(&mac.to_le_bytes());
    packet.extend_from_slice(&random.to_le_bytes());
    packet.extend_from_slice(&sequence.to_le_bytes());
    packet.extend_from_slice(&command.to_le_bytes());
    packet.extend_from_slice(payload);
    packet
}

fn spawn_monitor(
    socket: UdpSocket,
    buttons: Arc<AtomicU8>,
    healthy: Arc<AtomicBool>,
    last_update: Arc<Mutex<Option<Instant>>>,
    stop: Arc<AtomicBool>,
    expected_host: Ipv4Addr,
) -> Result<JoinHandle<()>, KmNetNativeError> {
    thread::Builder::new()
        .name("novasight-kmnet-monitor".to_owned())
        .spawn(move || {
            let mut packet = [0_u8; 1024];
            while !stop.load(Ordering::Acquire) {
                match socket.recv_from(&mut packet) {
                    Ok((received, peer))
                        if received >= MONITOR_PACKET_LEN
                            && peer.ip() == std::net::IpAddr::V4(expected_host) =>
                    {
                        buttons.store(packet[1], Ordering::Release);
                        *last_update
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            Some(Instant::now());
                    }
                    Ok(_) => {}
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                        ) => {}
                    Err(_) => {
                        healthy.store(false, Ordering::Release);
                        return;
                    }
                }
            }
        })
        .map_err(KmNetNativeError::SpawnMonitor)
}

fn parse_uuid(value: &str) -> Result<u32, KmNetNativeError> {
    let value = value.trim();
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(KmNetNativeError::InvalidUuid(value.to_owned()));
    }
    u32::from_str_radix(value, 16).map_err(|_| KmNetNativeError::InvalidUuid(value.to_owned()))
}

fn pointer_error(code: &'static str, message: String) -> AppError {
    AppError::PointerDevice { code, message }
}

fn read_monitor_buttons(
    healthy: &AtomicBool,
    last_update: &Mutex<Option<Instant>>,
    buttons: &AtomicU8,
    monitor_timeout: Duration,
) -> Result<PointerButtons, AppError> {
    if !healthy.load(Ordering::Acquire) {
        return Err(pointer_error(
            "monitor_failed",
            "kmNet monitor socket terminated".to_owned(),
        ));
    }
    let last_update = *last_update
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(last_update) = last_update else {
        return Err(pointer_error(
            "monitor_stale",
            "kmNet monitor is waiting for its first hardware report".to_owned(),
        ));
    };
    if last_update.elapsed() > monitor_timeout {
        return Err(pointer_error(
            "monitor_stale",
            format!(
                "kmNet monitor has been silent for more than {:?}",
                monitor_timeout
            ),
        ));
    }
    let buttons = buttons.load(Ordering::Acquire);
    Ok(PointerButtons {
        left: buttons & 0x01 != 0,
        right: buttons & 0x02 != 0,
    })
}

#[derive(Debug, Error)]
pub enum KmNetNativeError {
    #[error("invalid native kmNet configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("kmNet UUID must contain exactly eight hexadecimal digits, got {0:?}")]
    InvalidUuid(String),
    #[error("failed to bind kmNet control socket: {0}")]
    Bind(io::Error),
    #[error("failed to connect kmNet control socket: {0}")]
    Connect(io::Error),
    #[error("failed to bind kmNet monitor socket: {0}")]
    MonitorBind(io::Error),
    #[error("failed to configure kmNet socket: {0}")]
    Configure(io::Error),
    #[error("failed to send kmNet {operation} request: {source}")]
    Send {
        operation: &'static str,
        source: io::Error,
    },
    #[error("failed to receive kmNet {operation} response: {source}")]
    Receive {
        operation: &'static str,
        source: io::Error,
    },
    #[error("kmNet {operation} response was only {received} bytes")]
    ShortResponse {
        operation: &'static str,
        received: usize,
    },
    #[error(
        "kmNet {operation} response mismatch: sequence {actual_sequence:#x}/{expected_sequence:#x}, command {actual_command:#x}/{expected_command:#x}"
    )]
    ResponseMismatch {
        operation: &'static str,
        expected_sequence: u32,
        actual_sequence: u32,
        expected_command: u32,
        actual_command: u32,
    },
    #[error("failed to spawn kmNet monitor thread: {0}")]
    SpawnMonitor(io::Error),
}

impl KmNetNativeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig(_) | Self::InvalidUuid(_) => "config_invalid",
            Self::Bind(_) | Self::Connect(_) | Self::MonitorBind(_) | Self::Configure(_) => {
                "socket_setup_failed"
            }
            Self::Send { .. } => "driver_send_failed",
            Self::Receive { source, .. }
                if matches!(
                    source.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                "driver_timeout"
            }
            Self::Receive { .. } | Self::ShortResponse { .. } | Self::ResponseMismatch { .. } => {
                "driver_protocol_failed"
            }
            Self::SpawnMonitor(_) => "monitor_failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, UdpSocket};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU8};
    use std::thread;
    use std::time::{Duration, Instant};

    use novasight_core::{
        AppError, DeviceCommand, Generation, MonotonicNanos, PointerButtons, PointerDevice,
        RuntimeEpoch,
    };

    use super::{
        CMD_CONNECT, CMD_MONITOR, CMD_MOUSE_MOVE, KmNetNativeConfig, KmNetNativeDevice,
        encode_packet, parse_uuid, read_monitor_buttons,
    };

    fn available_monitor_port() -> u16 {
        for port in (40_000..=49_151).rev() {
            if UdpSocket::bind((Ipv4Addr::LOCALHOST, port)).is_ok() {
                return port;
            }
        }
        panic!("no free UDP port in the kmNet monitor range");
    }

    fn send_monitor_report(monitor_port: u16) {
        let monitor = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut report = [0_u8; 20];
        report[0] = 1;
        report[1] = 0x02;
        report[8] = 2;
        monitor
            .send_to(&report, (Ipv4Addr::LOCALHOST, monitor_port))
            .unwrap();
    }

    #[test]
    fn uuid_is_exactly_the_vendor_four_byte_identifier() {
        assert_eq!(parse_uuid("01FBC068").unwrap(), 0x01fb_c068);
        assert!(parse_uuid("12345").is_err());
        assert!(parse_uuid("1234567z").is_err());
    }

    #[test]
    fn packet_bytes_match_the_public_cpp_packed_little_endian_abi() {
        let packet = encode_packet(0x01fb_c068, 0x1122_3344, 7, CMD_MOUSE_MOVE, &[0xaa, 0xbb]);

        assert_eq!(
            packet,
            [
                0x68, 0xc0, 0xfb, 0x01, // UUID/mac
                0x44, 0x33, 0x22, 0x11, // rand/control value
                0x07, 0x00, 0x00, 0x00, // indexpts
                0x45, 0x73, 0xde, 0xae, // cmd_mouse_move
                0xaa, 0xbb,
            ]
        );
    }

    #[test]
    fn rejects_monitor_ports_outside_the_public_cpp_contract() {
        for monitor_port in [0, 1023, 49_152, u16::MAX] {
            let error = KmNetNativeDevice::new(KmNetNativeConfig {
                host: Ipv4Addr::LOCALHOST,
                port: 8888,
                uuid: "01FBC068".to_owned(),
                monitor_port,
                connect_timeout: Duration::from_secs(1),
                request_timeout: Duration::from_secs(1),
                monitor_timeout: Duration::from_secs(1),
            })
            .unwrap_err();

            assert!(error.to_string().contains("1024..=49151"));
        }
    }

    #[test]
    fn construction_validates_without_opening_a_device_session() {
        let device = KmNetNativeDevice::new(KmNetNativeConfig {
            host: Ipv4Addr::LOCALHOST,
            port: 9,
            uuid: "01FBC068".to_owned(),
            monitor_port: available_monitor_port(),
            connect_timeout: Duration::from_millis(1),
            request_timeout: Duration::from_millis(1),
            monitor_timeout: Duration::from_millis(1),
        })
        .expect("construction must not contact the configured endpoint");

        let error = device
            .send(DeviceCommand {
                epoch: RuntimeEpoch(1),
                generation: Generation(1),
                issued_at: MonotonicNanos(1),
                target_object_id: 1,
                delta_x_counts: 1,
                delta_y_counts: 1,
            })
            .expect_err("commands outside an epoch-owned session must fail closed");
        assert!(error.to_string().contains("not connected"));
    }

    #[test]
    fn monitor_is_unavailable_until_a_real_hardware_report_arrives() {
        let error = read_monitor_buttons(
            &AtomicBool::new(true),
            &Mutex::new(None),
            &AtomicU8::new(0),
            Duration::from_secs(2),
        )
        .expect_err("the connect acknowledgement is not a monitor report");
        assert!(error.to_string().contains("first hardware report"));
    }

    #[test]
    fn native_device_exchanges_real_udp_packets_and_caches_monitor_buttons() {
        let server = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let server_port = server.local_addr().unwrap().port();
        let monitor_port = available_monitor_port();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        let responder = thread::spawn(move || {
            let mut packet = [0_u8; 1024];
            let mut ordinary_randoms = Vec::new();
            let commands = [CMD_CONNECT, CMD_MONITOR, CMD_MOUSE_MOVE];
            let mut expected_sequence = 0_usize;
            ready_tx.send(()).unwrap();
            while expected_sequence < commands.len() {
                let (received, peer) = server.recv_from(&mut packet).unwrap();
                assert!(received >= 16);
                let actual_sequence =
                    u32::from_le_bytes(packet[8..12].try_into().unwrap()) as usize;
                let actual_command = u32::from_le_bytes(packet[12..16].try_into().unwrap());
                if actual_sequence < expected_sequence {
                    assert_eq!(commands[actual_sequence], actual_command);
                    server.send_to(&packet[..received], peer).unwrap();
                    if actual_command == CMD_MONITOR {
                        send_monitor_report(monitor_port);
                    }
                    continue;
                }
                let expected_command = commands[expected_sequence];
                assert_eq!(
                    actual_sequence, expected_sequence,
                    "setup retries must retain the previous sequence"
                );
                assert_eq!(actual_command, expected_command);
                if expected_command == CMD_MONITOR {
                    let encoded_port = u32::from_le_bytes(packet[4..8].try_into().unwrap());
                    assert_eq!(encoded_port, 0xaa55_0000 | u32::from(monitor_port));
                    assert_eq!(received, 16);
                } else {
                    ordinary_randoms.push(u32::from_le_bytes(packet[4..8].try_into().unwrap()));
                }
                server.send_to(&packet[..received], peer).unwrap();
                if expected_command == CMD_MONITOR {
                    send_monitor_report(monitor_port);
                }
                if expected_command == CMD_MOUSE_MOVE {
                    assert_eq!(received, 72);
                    assert_eq!(i32::from_le_bytes(packet[20..24].try_into().unwrap()), 12);
                    assert_eq!(i32::from_le_bytes(packet[24..28].try_into().unwrap()), -7);
                }
                expected_sequence += 1;
            }
            assert_eq!(ordinary_randoms.len(), 2);
            assert!(ordinary_randoms.iter().all(|value| *value != 0));
            assert_ne!(ordinary_randoms[0], ordinary_randoms[1]);
        });
        let device = KmNetNativeDevice::new(KmNetNativeConfig {
            host: Ipv4Addr::LOCALHOST,
            port: server_port,
            uuid: "01FBC068".to_owned(),
            monitor_port,
            // CI hosts can heavily deschedule loopback responder threads; the
            // production retry budget remains supplied by configuration.
            connect_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(5),
            monitor_timeout: Duration::from_secs(2),
        })
        .unwrap();
        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("mock device responder must be scheduled before the client handshake");
        device.connect().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match device.trigger_active() {
                Ok(Some(true)) => break,
                Ok(Some(false))
                | Err(AppError::PointerDevice {
                    code: "monitor_stale",
                    ..
                }) if Instant::now() < deadline => thread::yield_now(),
                result => panic!("monitor did not publish the hardware report: {result:?}"),
            }
        }
        assert_eq!(device.trigger_active().unwrap(), Some(true));
        assert_eq!(
            device.buttons().unwrap(),
            Some(PointerButtons {
                left: false,
                right: true,
            })
        );
        thread::sleep(Duration::from_millis(2_100));
        assert!(
            device
                .trigger_active()
                .expect_err("silent monitor cache must expire")
                .to_string()
                .contains("silent")
        );
        device
            .send(DeviceCommand {
                epoch: RuntimeEpoch(1),
                generation: Generation(2),
                issued_at: MonotonicNanos(3),
                target_object_id: 4,
                delta_x_counts: 12,
                delta_y_counts: -7,
            })
            .unwrap();
        responder.join().unwrap();
        let shutdown_started = Instant::now();
        device.disconnect().unwrap();
        assert!(shutdown_started.elapsed() < Duration::from_millis(250));
        assert!(
            device
                .trigger_active()
                .unwrap_err()
                .to_string()
                .contains("not connected")
        );
    }
}
