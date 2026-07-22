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
use std::time::Duration;

use novasight_core::{AppError, DeviceCommand, DeviceReceipt, PointerDevice};
use thiserror::Error;

const CMD_CONNECT: u32 = 0xaf3c_2828;
const CMD_MOUSE_MOVE: u32 = 0xaede_7345;
const CMD_MONITOR: u32 = 0x2738_8020;
const HEADER_LEN: usize = 16;
const MOVE_PACKET_LEN: usize = HEADER_LEN + 56;
const MONITOR_PACKET_MIN_LEN: usize = 8;

#[derive(Clone, Debug)]
pub struct KmNetNativeConfig {
    pub host: Ipv4Addr,
    pub port: u16,
    pub uuid: String,
    pub monitor_port: u16,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
}

#[derive(Debug)]
pub struct KmNetNativeDevice {
    control: Mutex<ControlSocket>,
    buttons: Arc<AtomicU8>,
    monitor_healthy: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    monitor: Mutex<Option<JoinHandle<()>>>,
    successful_sends: AtomicU64,
}

#[derive(Debug)]
struct ControlSocket {
    socket: UdpSocket,
    mac: u32,
    sequence: u32,
}

impl KmNetNativeDevice {
    pub fn connect(config: KmNetNativeConfig) -> Result<Self, KmNetNativeError> {
        if config.port == 0 || config.monitor_port == 0 {
            return Err(KmNetNativeError::InvalidConfig(
                "control and monitor ports must be non-zero",
            ));
        }
        if config.connect_timeout.is_zero() || config.request_timeout.is_zero() {
            return Err(KmNetNativeError::InvalidConfig("timeouts must be non-zero"));
        }
        let mac = parse_uuid(&config.uuid)?;
        let remote = SocketAddrV4::new(config.host, config.port);
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(KmNetNativeError::Bind)?;
        socket.connect(remote).map_err(KmNetNativeError::Connect)?;
        socket
            .set_read_timeout(Some(config.connect_timeout))
            .map_err(KmNetNativeError::Configure)?;
        let mut control = ControlSocket {
            socket,
            mac,
            sequence: 0,
        };
        control.exchange(CMD_CONNECT, 0, &[], "connect")?;

        let monitor_socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, config.monitor_port))
            .map_err(KmNetNativeError::MonitorBind)?;
        monitor_socket
            .set_read_timeout(Some(config.request_timeout))
            .map_err(KmNetNativeError::Configure)?;
        let monitor_value = u32::from(config.monitor_port) | (0xaa55_u32 << 16);
        control.exchange(CMD_MONITOR, monitor_value, &[], "monitor")?;
        control
            .socket
            .set_read_timeout(Some(config.request_timeout))
            .map_err(KmNetNativeError::Configure)?;

        let buttons = Arc::new(AtomicU8::new(0));
        let monitor_healthy = Arc::new(AtomicBool::new(true));
        let stop = Arc::new(AtomicBool::new(false));
        let monitor = spawn_monitor(
            monitor_socket,
            Arc::clone(&buttons),
            Arc::clone(&monitor_healthy),
            Arc::clone(&stop),
        )?;
        Ok(Self {
            control: Mutex::new(control),
            buttons,
            monitor_healthy,
            stop,
            monitor: Mutex::new(Some(monitor)),
            successful_sends: AtomicU64::new(0),
        })
    }
}

impl PointerDevice for KmNetNativeDevice {
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
        self.control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .exchange(CMD_MOUSE_MOVE, 0, &payload, "move")
            .map_err(|error| pointer_error(error.code(), error.to_string()))?;
        let attempt = self.successful_sends.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(DeviceReceipt::accepted(attempt, command))
    }

    fn trigger_active(&self) -> Result<Option<bool>, AppError> {
        if !self.monitor_healthy.load(Ordering::Acquire) {
            return Err(pointer_error(
                "monitor_failed",
                "kmNet monitor socket terminated".to_owned(),
            ));
        }
        Ok(Some(self.buttons.load(Ordering::Acquire) & 0x03 != 0))
    }
}

impl Drop for KmNetNativeDevice {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
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

impl ControlSocket {
    fn exchange(
        &mut self,
        command: u32,
        random: u32,
        payload: &[u8],
        operation: &'static str,
    ) -> Result<(), KmNetNativeError> {
        self.sequence = self.sequence.wrapping_add(1);
        let mut packet = Vec::with_capacity(HEADER_LEN + payload.len());
        packet.extend_from_slice(&self.mac.to_le_bytes());
        packet.extend_from_slice(&random.to_le_bytes());
        packet.extend_from_slice(&self.sequence.to_le_bytes());
        packet.extend_from_slice(&command.to_le_bytes());
        packet.extend_from_slice(payload);
        debug_assert!(packet.len() == HEADER_LEN || packet.len() == MOVE_PACKET_LEN);
        self.socket
            .send(&packet)
            .map_err(|source| KmNetNativeError::Send { operation, source })?;
        let mut response = [0_u8; 1024];
        let received = self
            .socket
            .recv(&mut response)
            .map_err(|source| KmNetNativeError::Receive { operation, source })?;
        if received < HEADER_LEN {
            return Err(KmNetNativeError::ShortResponse {
                operation,
                received,
            });
        }
        let response_sequence = u32::from_le_bytes(response[8..12].try_into().unwrap());
        let response_command = u32::from_le_bytes(response[12..16].try_into().unwrap());
        if response_sequence != self.sequence || response_command != command {
            return Err(KmNetNativeError::ResponseMismatch {
                operation,
                expected_sequence: self.sequence,
                actual_sequence: response_sequence,
                expected_command: command,
                actual_command: response_command,
            });
        }
        Ok(())
    }
}

fn spawn_monitor(
    socket: UdpSocket,
    buttons: Arc<AtomicU8>,
    healthy: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, KmNetNativeError> {
    thread::Builder::new()
        .name("novasight-kmnet-monitor".to_owned())
        .spawn(move || {
            let mut packet = [0_u8; 1024];
            while !stop.load(Ordering::Acquire) {
                match socket.recv_from(&mut packet) {
                    Ok((received, _)) if received >= MONITOR_PACKET_MIN_LEN => {
                        buttons.store(packet[1], Ordering::Release);
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
    use std::thread;
    use std::time::{Duration, Instant};

    use novasight_core::{DeviceCommand, Generation, MonotonicNanos, PointerDevice, RuntimeEpoch};

    use super::{
        CMD_CONNECT, CMD_MONITOR, CMD_MOUSE_MOVE, KmNetNativeConfig, KmNetNativeDevice, parse_uuid,
    };

    #[test]
    fn uuid_is_exactly_the_vendor_four_byte_identifier() {
        assert_eq!(parse_uuid("01FBC068").unwrap(), 0x01fb_c068);
        assert!(parse_uuid("12345").is_err());
        assert!(parse_uuid("1234567z").is_err());
    }

    #[test]
    fn native_device_exchanges_real_udp_packets_and_caches_monitor_buttons() {
        let server = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let server_port = server.local_addr().unwrap().port();
        let monitor_probe = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let monitor_port = monitor_probe.local_addr().unwrap().port();
        drop(monitor_probe);
        let responder = thread::spawn(move || {
            let mut packet = [0_u8; 1024];
            for expected_command in [CMD_CONNECT, CMD_MONITOR, CMD_MOUSE_MOVE] {
                let (received, peer) = server.recv_from(&mut packet).unwrap();
                assert!(received >= 16);
                assert_eq!(
                    u32::from_le_bytes(packet[12..16].try_into().unwrap()),
                    expected_command
                );
                if expected_command == CMD_MONITOR {
                    let encoded_port = u32::from_le_bytes(packet[4..8].try_into().unwrap());
                    assert_eq!(encoded_port & 0xffff, u32::from(monitor_port));
                }
                server.send_to(&packet[..received], peer).unwrap();
                if expected_command == CMD_MONITOR {
                    let monitor = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
                    monitor
                        .send_to(
                            &[1, 0x02, 0, 0, 0, 0, 0, 0],
                            (Ipv4Addr::LOCALHOST, monitor_port),
                        )
                        .unwrap();
                }
                if expected_command == CMD_MOUSE_MOVE {
                    assert_eq!(i32::from_le_bytes(packet[20..24].try_into().unwrap()), 12);
                    assert_eq!(i32::from_le_bytes(packet[24..28].try_into().unwrap()), -7);
                }
            }
        });
        let device = KmNetNativeDevice::connect(KmNetNativeConfig {
            host: Ipv4Addr::LOCALHOST,
            port: server_port,
            uuid: "01FBC068".to_owned(),
            monitor_port,
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(2),
        })
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while device.trigger_active().unwrap() != Some(true) && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(device.trigger_active().unwrap(), Some(true));
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
    }
}
