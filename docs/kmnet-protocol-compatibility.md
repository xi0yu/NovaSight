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

The retained `python_host` adapter is still the production default while this
native path remains evidence-gated. Production `novasightd --check` now starts
that exact configured helper, validates its JSON-lines protocol and
`driver_available` receipt, and requests a clean shutdown without sending the
hardware `connect` operation. This catches a missing interpreter, packaged
module, incompatible helper, or unavailable vendor extension before systemd
starts the daemon. Selecting `native_udp` without the explicit experimental
feature fails the same preflight instead of producing a false PASS.
The Rust composition root also pins the helper's working directory to the
canonical release that contains the configured model worker. Helper startup no
longer depends on the operator's shell directory or systemd's ambient working
directory, and reconnect cannot cross an atomic `current` symlink switch into
another release's Python package.

The upstream repository does not publish a standard open-source license, and
its copyright notice restricts use to official kmBox hardware. Protocol
compatibility does not by itself grant commercial redistribution rights.
Commercial NovaSight releases therefore require written vendor authorization;
the native adapter remains evidence-gated until real Jetson and hardware runs
also pass the sustained-output acceptance test.
