# Web Authentication Architecture

## Ownership

NovaSight separates the LAN trust boundary from runtime authority:

```text
Browser -- LAN HTTP/HTTPS --> novasight-web -- HTTP/1 over Unix socket --> novasightd
                                  |                                      |
                                  |                                      +-- runtime/config/license SSOT
                                  +-- authentication/session/authz/CSRF
```

- `novasight-web` is the only TCP listener. It serves static assets and owns all
  browser security state.
- `novasightd` never listens on a LAN socket. It owns runtime, configuration,
  model, license, capture, and hardware state behind `run/novasightd.sock`.
- `novasightctl` remains a local Unix-socket client. Browser credentials are not
  accepted by or forwarded to the daemon.

## Authentication and session contract

1. The Debug launcher generates one random 256-bit temporary license code for
   each start and passes it to `novasightd`; it is not written to disk.
2. Studio accepts either this temporary code or a signed license in the same
   authorization field. No separate access code is required. Old `#access=...`
   fragments are cleared and cannot authenticate a session.
3. `POST /api/auth/session` asks the daemon to validate/activate the submitted
   license before issuing a Web session, with per-source-IP rate limiting.
4. A successful login creates an opaque random server-side session. The browser
   receives only `novasight_web_session` with `HttpOnly`, `SameSite=Strict`,
   `Path=/`, one-hour `Max-Age`, and optional `Secure`.
5. `GET /api/auth/session` restores page state after refresh. `DELETE` requires
   CSRF and removes the server-side session. Restarting `novasight-web`
   invalidates all sessions.

The Web session authenticates the caller; daemon-side license features remain
authoritative for hardware operations.

## Authorization and CSRF

The first version exposes one explicit role, `operator`, with named permissions:

| Permission | Surface |
| --- | --- |
| `studio:read` | authenticated API reads and status WebSockets |
| `runtime:operate` | runtime start, stop, restart, emergency stop |
| `configuration:write` | config, capture, and crosshair mutations |
| `models:manage` | catalog, metadata, profiling, publish, rollback |
| `license:manage` | temporary/formal credential activation and clear |
| `hardware:operate` | executor connect/disconnect/diagnostic operations |

Route authorization is deny-by-default. An unknown future daemon route is not
automatically exposed to LAN operators. `/api/v1/daemon/shutdown` is always
local-only and is rejected by the Web/API layer.

Every method other than `GET`, `HEAD`, and `OPTIONS` requires the session's
random CSRF token in `X-NovaSight-CSRF`. Studio keeps that token in memory only,
adds it centrally in the API client, refreshes session state after a CSRF
rejection, and never stores it in local storage.

## Request boundary protections

- IP literals and `localhost` are valid Host values. DNS/reverse-proxy names
  must be listed exactly in `NOVASIGHT_WEB_ALLOWED_HOSTS`.
- When `Origin` is present, its host must equal the request Host. Loopback host
  aliases are treated as the same local site.
- Cookies, browser authorization, origin, referrer, forwarding headers, and the
  CSRF header are stripped before proxying to the Unix socket.
- Daemon `Set-Cookie` and proxy-authentication headers are stripped from the
  response.
- Static responses set CSP, frame denial, no-sniff, no-referrer, and restrictive
  permissions headers. HTML and all auth/API responses are not cached.
- WebSocket upgrades require a valid session and are closed with code `4403`
  when that session expires.
- License activation is independently limited to six submissions per source IP
  per 60 seconds after session, authorization, and CSRF checks pass.

## License activation boundary

Browser session authentication and product licensing remain separate internal
responsibilities, but one validated license submission establishes both.

Debug launchers generate the temporary 256-bit value in memory and pass
it only to `novasightd` through `NOVASIGHT_TEMPORARY_LICENSE_CODE` without
writing it to disk. Studio submits both that ephemeral value and
formal signed licenses through `POST /api/license/activate`. The repository
compares only a SHA-256 digest of the temporary value in constant time; a match
creates process-local authorization without writing a license file. A mismatch
continues through the normal signed-license verifier. Release builds ignore the
temporary credential environment and never enable process-local authorization.

Temporary authorization includes `hardware_control`, expires with the daemon,
and is regenerated on restart. Granting it does not connect a device or enable
output by itself; Studio retains explicit output confirmation and backend
connection/output checks. No fixed temporary code exists in frontend or backend
source. Signed licenses without that feature remain restricted.

Direct HTTP is suitable only for a controlled LAN. Across routed, shared, or
untrusted networks, terminate HTTPS at an authenticated reverse proxy, enable
`NOVASIGHT_WEB_SECURE_COOKIE=true`, and explicitly allow the proxy Host name.

## Frontend state model

The UI uses one authentication gate around every protected page:

| State | Page behavior |
| --- | --- |
| checking | show the access station and test the existing HttpOnly session |
| anonymous | accept one temporary or signed license; no runtime API is called |
| authenticated | mount the existing license/workspace pages and show the persistent session bar |
| session expired / API 401 | unmount protected pages; preserve daemon runtime; ask for authentication again |
| CSRF rejected | refresh session/CSRF state; tell the operator to retry the mutation |
| daemon unavailable | keep the Web session, show Unix IPC as unavailable, disable progression through the license gate |
| logout | revoke the server session, clear in-memory CSRF, unmount protected pages |

The authenticated session bar keeps browser, Web/API, and daemon status visible
on every page. Authentication never implies license validity, runtime health, or
hardware output permission; those remain separate daemon-owned gates.

## Jetson acceptance

The protected Jetson job uses job-local authorization and verifies:

- `novasightd` becomes ready only on the Unix socket;
- `novasight-web` serves static/health traffic and rejects anonymous API calls;
- invalid and valid license-login paths, HttpOnly session creation, and CSRF on
  mutations;
- signed license activation and the existing runtime/model/hardware-safe receipt;
- logout revocation and local-socket-only daemon shutdown;
- `novasightd --check` and `novasight-web --check` from the packaged release.
