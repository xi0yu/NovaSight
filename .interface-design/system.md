# NovaSight Product Interface System

## Product promise

NovaSight is a consumer-grade product that controls professional Edge AI systems.

The visible experience is light, fast, natural, alive, reliable, and easy to understand. CUDA, TensorRT, NVMM, V4L2, tracking, control, and deployment remain available, but appear only when the user asks to go deeper.

> Professional capability stays inside. Consumer-grade experience stays outside.

## Product split

- **NovaSight Cloud** is object-first and remote: Home, Devices, Models, Activity, Me.
- **NovaSight Studio** is a professional workstation: Home, Device, Models, Activity, License. Capture, inference, tracking, control, parameters, performance, and hardware tests live one level deeper inside the current device.
- Cloud must not mirror every Studio setting. Both products share typography, status language, motion, and progressive disclosure.

## Human and task

The human is an operator who wants to know, without learning backend architecture:

1. Which device or model am I looking at?
2. Is it running and safe?
3. What just happened?
4. What needs my attention?
5. What is the next useful action?

The first screen answers those questions. It does not begin with analytics, registries, telemetry, or infrastructure.

## Information architecture

- Organize around real objects: my device, my model, what happened, my account or license.
- Prefer `首页 / 设备 / 模型 / 活动 / 我的` in Cloud.
- In Studio, expose device-specific depth only after entering the device: `画面 / 推理 / 目标与控制 / 参数 / 性能 / 控制测试`.
- A core object or task must be visible as text on the first relevant layer. Do not require users to guess a tab, hover target, right-click menu, or collapsed section.
- Low-frequency actions may move behind `···`; the current state and primary action may not.

## Progressive disclosure

Use three levels:

1. **Human conclusion** — `正在运行 · 198 FPS · 21 ms · 一切正常`.
2. **Operational detail** — capture, inference, tracking, control, GPU, temperature, current model.
3. **Engineering evidence** — V4L2, NVMM, TensorRT, FP16, revisions, generations, request IDs, and raw errors.

Professional information is not removed. It is revealed in response to user intent.

## Layout and hierarchy

- One focal point per view: the current object, conclusion, or task.
- Prefer open compositions, inline metrics, lists, and timelines over grids of equal cards.
- Use whitespace, type weight, alignment, and quiet separators before containers.
- Do not create a card for every number. `198 FPS · 21 ms · 74% GPU` should read as one natural status line when they belong together.
- Avoid dense tables and filter walls on the first layer. Dense comparison belongs in professional depth, such as the Studio model asset view.
- Pages should feel like an everyday product, not an IT department console.

## Visual direction

- Platform typography first: SF Pro / system UI / Noto Sans SC / PingFang SC.
- Large headings use tighter tracking; body copy uses comfortable leading.
- Dominant surfaces are quiet white, glacier gray, or restrained graphite.
- Color communicates state. Cyan may mean live signal, green healthy, amber attention, red fault or physical risk. No decorative purple-blue AI gradients.
- Depth is subtle surface separation. Avoid stacked glass, large shadows, excessive radius, or decorative glow.
- Preserve only the three approved appearances: 专业工作台, 粉白清昼, 黑灰红.

## Interaction and state

- Feedback begins immediately and remains visible through connecting, validating, applying, deploying, and reconciling.
- Motion explains state change; it never decorates idle pages.
- Only dangerous or externally consequential actions require confirmation: opening physical output, live model switching when impact is real, hardware test movement, and safe license exit.
- Stop, pause output, ordinary save, retry, and navigation do not ask repeatedly.
- Interrupted tasks retain the user’s intent and return path. Reconnect reads back authoritative state before retrying.
- A saved value, requested value, and effective runtime value are different states and must be labeled honestly.

## Error and activity language

- Default user message: what happened, what it affects, whether the system is safe, and what to do next.
- Keep `原始错误与开发者详情` one level deeper with source, HTTP status, request ID, and backend text.
- Activity is a human timeline — `模型已更新`, `设备已上线`, `检测到异常` — not an event-management table.
- Toasts confirm brief completion. Persistent faults and uncertain results remain in Activity until resolved or cleared.

## Studio-specific professional depth

- Use the real chain `采集 → 推理 → 目标 → 控制 → 输出` as the device-level mental model.
- No-target and waiting-for-trigger are normal waiting states, not faults.
- Unknown or stale runtime state locks physical output and says `未确认`; frontend configuration must never impersonate runtime proof.
- The model workspace distinguishes discovered file, registered artifact, validated Engine, deployed model, and runtime-loaded model.
- Model metadata, tags, paths, parser settings, revisions, and raw performance counters belong below the current model and deployment task.

## Accessibility and motion

- Interactive targets are at least 40px on desktop and 44px for touch layouts.
- Every control has visible hover, active, focus, disabled, loading, success, and error states as applicable.
- Use semantic HTML and keep visible labels aligned with accessible names.
- Respect reduced motion, reduced transparency, and increased contrast.
- Animate only transform and opacity; no ornamental loops. Live pulses are allowed only when they encode genuinely fresh data.

## Rejected defaults

- Enterprise SaaS navigation and admin dashboards.
- Dashboard / Analytics / Device Management / Model Registry / Observability as the product’s first impression.
- Equal card walls, giant KPI numbers, dense filters, and database-first model views.
- AI-template gradients, stars, glassmorphism, oversized shadows, and decorative animation.
- Hiding primary actions inside tabs, hover-only controls, or unexplained icon menus.
- Marketing copy where a direct status is clearer.

## Shipping checks

- **Daily-use check:** does this feel like a product someone would choose to open every day?
- **First-glance check:** can a new user identify the object, state, attention item, and next action in five seconds?
- **Depth check:** can an expert still reach the real pipeline evidence without the beginner seeing it first?
- **Runtime honesty check:** is every displayed fact labeled as runtime, saved configuration, requested state, or diagnostic evidence?
- **Discoverability check:** are first-layer tasks visible without guessing a sub-tab or menu?
- **Browser check:** verify desktop and narrow layouts, keyboard focus, real state transitions, errors, and zero horizontal overflow.
