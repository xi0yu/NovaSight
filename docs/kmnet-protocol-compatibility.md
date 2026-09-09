# kmBoxNet protocol compatibility evidence

NovaSight's native Rust adapter is an independent implementation of the
kmBoxNet UDP wire contract. It does not compile, copy, vendor, or distribute
the manufacturer's C++ implementation.

The current compatibility baseline is the public `c++_demo` at upstream commit
`9b62283c6271e8e594f3b97986fed291deb4318f`:

- <https://github.com/kvmaibox/kmboxnet/tree/9b62283c6271e8e594f3b97986fed291deb4318f/c%2B%2B_demo>
- <https://github.com/kvmaibox/kmboxnet/blob/9b62283c6271e8e594f3b97986fed291deb4318f/c%2B%2B_demo/NetConfig/kmboxNet.cpp>
- <https://github.com/kvmaibox/kmboxnet/blob/9b62283c6271e8e594f3b97986fed291deb4318f/c%2B%2B_demo/NetConfig/kmboxNet.h>

The Rust compatibility contract is deliberately narrow:

| Behavior | Upstream evidence | Rust behavior |
| --- | --- | --- |
| Header | four native `unsigned int` fields: mac, rand, indexpts, cmd | four packed little-endian `u32` fields |
| Connect | sequence 0, `cmd_connect`, refreshed rand | identical wire values; bounded receive timeout |
| Mouse move | signed-short API written into 32-bit x/y fields of a 56-byte mouse payload | accepts only `i16`-representable counts and emits the same 72-byte datagram |
| Monitor | rand is `0xaa55 << 16 | port`; port range 1024 through 49151 | identical encoding and range validation |
| Monitor report | packed 8-byte mouse report plus 12-byte keyboard report | requires at least the complete 20-byte report before publishing buttons |
| Response | command and sequence correspond to the request | also enforces source, minimum length, timeout, command, and sequence |

The `rand` field is refreshed for ordinary commands to match upstream packet
behavior. It is not treated as authentication. The monitor command is the one
exception because that field transports its tagged listener port.

`native_udp` is the production adapter and is included by the normal
`deepstream` build. `novasightd --check` validates its IPv4, ports, UUID,
timeouts and monitor contract without contacting the hardware; the runtime
epoch owns the real UDP sockets, monitor thread, connect exchange and shutdown.

At runtime, recoverable UDP/device failures close the output gate and mark
the device disconnected while the pipeline remains available for bounded
reconnect attempts. The supervisor projects this as `device.state=degraded`
with a `device_reconnecting` error instead of leaving a false `ready` state.
`pipeline_metrics` records the connection bit, failure count, recovery count,
and last concrete driver error; a successful buttons or move receipt returns
the subsystem to `ready` and clears its active error without erasing that
historical evidence. Reconnect-cooldown polls do not overwrite the originating
failure or inflate its failure count.

The upstream repository does not publish a standard open-source license, and
its copyright notice restricts use to official kmBox hardware. NovaSight's
adapter therefore targets official kmBox hardware only; commercial terms still
need to be confirmed with the vendor.
