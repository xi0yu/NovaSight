# Target-host presence power saving

NovaSight can stop the Jetson DeepStream pipeline when the target/game host is
offline while preserving the user's run intent. The runtime resumes only after
the target host heartbeat returns and `auto_resume` is enabled.

## Jetson configuration

```yaml
power_saving:
  host_presence_enabled: true
  target_host_id: gaming-pc
  heartbeat_timeout_s: 6.0
  offline_grace_s: 15.0
  auto_resume: true
```

Configuration changes require restarting the NovaSight process. The default is
disabled so existing deployments do not require a host agent.

## Target Host Heartbeat

Send the heartbeat from the target/game host with any HTTP client or service
manager. The active contract is the Rust daemon endpoint:

```bash
while true; do
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d '{"host_id":"gaming-pc"}' \
    http://JETSON_IP:5174/api/runtime/presence/heartbeat
  sleep 2
done
```

The host sends a heartbeat every two seconds. If the host shuts down, sleeps,
or loses network connectivity, NovaSight first enters an offline grace period
and then stops the complete DeepStream pipeline. A later heartbeat automatically
starts it again only when the user previously requested the runtime to run.

## Runtime states

- `active`: DeepStream inference is running and the heartbeat lease is valid.
- `grace`: the heartbeat expired but the offline grace period has not elapsed.
- `cold_standby`: the pipeline is stopped while run intent remains active.
- `interrupted`: the runtime stopped for a non-policy failure and heartbeat cannot restart it.
- `stopped`: the user explicitly stopped the runtime; heartbeats cannot resume it.
- `disabled`: host-presence supervision is disabled and manual behavior is unchanged.

Status is exposed through `GET /api/runtime/power`, `/api/runtime/state`, and the
status WebSocket. The heartbeat endpoint is
`POST /api/runtime/presence/heartbeat` with JSON `{"host_id":"gaming-pc"}`.
