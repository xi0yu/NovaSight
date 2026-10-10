# NovaSight D1A Engineering Review Test Plan

Status: pre-implementation acceptance contract
Target: D1A safe runtime golden journey
Environment boundary: mocked browser behavior on host; authoritative gateway/runtime checks in Rust; release build and real runtime receipt on Jetson.

## Critical user journeys

### 1. LAN access and authentication

1. Open `/` without a session and confirm only the access-code entry surface is available.
2. Submit an invalid access code and confirm the page reports rejection without exposing server or credential details.
3. Submit the current launch-generated access code and confirm the URL fragment is cleared, a server-side session is established, and the user proceeds to license status.
4. Expire the session during an ordinary action and confirm the UI returns to authentication without claiming that daemon state changed.
5. Re-authenticate after an unconfirmed safety operation and confirm the first action is an authoritative state reread, never an automatic replay.

### 2. Temporary and formal license activation

1. Use the same activation form for a temporary code and a signed formal license.
2. Confirm invalid, expired, rate-limited, and daemon-unavailable outcomes each name the problem and a safe next action.
3. While runtime may be active, request license clearing and confirm the dialog states that NovaSight will stop and verify safety before clearing.
4. Inject stop or verification failure and confirm the credential and session cookie remain present.
5. Inject repository-clear failure and confirm the UI reports the partial outcome: runtime stopped, credential retained.

### 3. Runtime Overview and golden journey

1. Open `?page=overview`; verify the first screen shows one conclusion, lifecycle/perception/output axes, one main action, and blockers before diagnostics.
2. Follow a blocker link to License, Models, Capture, Params, or Control; resolve it and return to Overview with focus restored to the changed blocker/result.
3. Start runtime and confirm a successful request remains pending until a newer authoritative snapshot confirms the result.
4. Verify `will_emit=true` is described only as current-sample eligibility and never as physical delivery.
5. Stop runtime and confirm success requires the strict safe tuple, not `stopping` or `faulted`.
6. Open every legacy deep link and confirm it remains usable; open an unknown page and confirm explicit fallback to Overview.

### 4. Emergency Stop

1. Trigger Emergency Stop while an ordinary mutation and a dialog are active; confirm it stays reachable and supersedes the ordinary local operation.
2. Confirm HTTP 204 alone never produces a success message.
3. On the same daemon instance, provide a higher-sequence snapshot with the strict safe tuple and confirm “紧急停止已确认”.
4. Restart the daemon before reconciliation and provide a safe snapshot; confirm the UI says the current system is safe but original-request causality is unknown.
5. Return `stopping`, `faulted`, stale, lower/equal sequence, invalid contract, or transport loss; confirm the critical unconfirmed state remains visible through re-authentication.
6. At 320 px, 375 px, 200% zoom, keyboard-only navigation, and a screen-reader landmark pass, confirm Emergency Stop is visible, focusable, labelled, and not covered by a drawer/dialog/safe area.

### 5. Models, parameters, and incidents

1. Select the already-active model and confirm a no-op success rather than an error.
2. Interrupt publish refresh, return to Models, and confirm active deployment/runtime are reconciled; conversion jobs remain separately labelled.
3. Create a multi-tab parameter revision conflict and confirm the local draft/diff survives; retry sends only selected remaining fields.
4. Generate repeated incidents and confirm grouping uses source, code, resource, and operation context; acknowledgement does not claim authoritative resolution.
5. Export diagnostics and confirm credentials, cookies, CSRF values, absolute paths, and license contents are absent.

## Temporal and transport edges

- Same daemon instance: lower or equal sequence is rejected; a higher complete snapshot may advance state.
- New daemon instance: sequence may establish a new baseline and the UI announces reconciliation.
- Old browser request generation: late REST or WebSocket callback is dropped even if its sequence is larger.
- Heartbeat-only frames never activate a new daemon instance or overwrite a full snapshot.
- Invalid or partial frames preserve the last trusted value as stale and disable unsafe optimistic conclusions.
- Reconnect, timeout, double-click, refresh, and session expiry never auto-replay Run, Stop, Emergency Stop, license clear, publish, or save.

## CI and production evidence

### Host CI

- Locked dependency install; removed dependencies are absent from direct importers, scripts, and CI build items.
- Frontend unit/contract suite passes before the Studio production build.
- Browser behavior job runs Chromium with a managed web server and uploads trace/screenshots on failure.
- Rust gateway/API integration proves session, CSRF, Host/Origin, authorization, Unix proxy, emergency barrier, and license-clear ordering.
- Browser mocks are labelled UI evidence and never reported as LAN-security or hardware proof.

### Jetson build receipt

- Record workflow URL, commit SHA, Jetson/L4T/JetPack identity, toolchain versions, package profile, and artifact identity.
- Run the locked metadata, Rust checks, frontend unit suite, portable release build, unauthenticated/CSRF/authorization checks, and strict safe tuple.
- A build receipt proves package composition and safe startup only; production readiness remains unknown.

### Jetson production receipt

- Require a current model fixture, camera/DeepStream readiness, runtime start, fresh detections, controlled stop, Emergency Stop, and the strict final safe tuple.
- Do not move physical kmNet hardware in CI; keep supervised physical movement as a separate commissioning receipt.
- A production claim is valid only when the current commit has both the Jetson build receipt and production receipt.

## Release blockers

- Any false-success safety state, missing daemon incarnation fence, or credential-first license clear.
- Any golden-journey or narrow-screen Emergency Stop failure.
- Any frontend behavior suite omitted from the required host/Jetson build gates.
- Missing workflow URL/SHA/platform identity in the receipt.
- Any claim that mocked E2E, host build, or a build-only Jetson receipt proves physical production behavior.
