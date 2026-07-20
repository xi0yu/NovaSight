# NovaSight Frontend Visual System

Version: v1
Scope: React + TypeScript + Vite + Tauri frontend assets and UI implementation.

## 1. Visual Position

NovaSight should read as professional desktop software for realtime vision, inference, tracking, prediction, control, device connectivity, and runtime monitoring.

The product visual language is:

```text
cool white commercial software
+ glacier gray workspace
+ deep blue-gray text
+ restrained blue-violet brand color
+ cyan-blue realtime signal color
+ light glass surfaces
+ precise SVG icons
+ modular data panels
+ technical feature illustrations
+ subtle data-flow and scan patterns
```

It must not look like a generic admin template, a game overlay panel, a neon cyberpunk dashboard, a Bootstrap/Ant-style default UI, or a decorative concept page that hurts reading speed.

Primary product references are macOS Sonoma, ChatGPT Desktop, NVIDIA App, Linear, Raycast, Stripe Dashboard, Vercel, OpenAI product surfaces, Adobe professional tools, and high-end device-control software.

## 2. Brand Graphic System

The NovaSight brand mark is built from four primitives:

- Four-corner vision frame.
- Soft center light point.
- Data trajectory from lower-left to upper-right.
- Abstract `N` structure embedded into the path geometry.

Avoid direct weapon crosshairs. Use optical framing, data flow, and recognition geometry instead.

### Logo Variants

| Variant | Usage | File |
| --- | --- | --- |
| Full horizontal | Sidebar header, about dialog, documents | `assets/brand/logos/novasight-logo-horizontal.svg` |
| Mark only | App icon, favicon, compact nav, loading | `assets/brand/logos/novasight-mark.svg` |
| Light background | Default product UI | `assets/brand/logos/novasight-logo-light.svg` |
| Dark background | Splash, dark mode, about dialog | `assets/brand/logos/novasight-logo-dark.svg` |
| Monochrome dark | Print, disabled branding, low-color contexts | `assets/brand/logos/novasight-logo-mono-dark.svg` |
| Monochrome light | Dark surfaces, watermark | `assets/brand/logos/novasight-logo-mono-light.svg` |

### Required Sizes

| Size | Use |
| --- | --- |
| 16px | favicon small, dense table identity |
| 20px | compact nav mark |
| 24px | sidebar logo icon |
| 32px | app header, about dialog |
| 64px | splash small |
| 128px | app icon source |
| 512px | app-store/Tauri icon source |

Clear space around the mark equals at least half the mark height in dense UI and one mark height in splash/about contexts.

## 3. Token Architecture

Use three token layers:

```text
Primitive tokens: raw values
Semantic tokens: product meaning
Component tokens: component-specific usage
```

Existing `web/src/design/tokens.css` should evolve toward this structure. New components should avoid raw hex values except inside the token source file.

## 4. Color Tokens

### Core Palette

| Token | Value | Usage |
| --- | --- | --- |
| `--bg-app` | `#F4F6F9` | Main app background |
| `--bg-surface` | `#FFFFFF` | Panels, cards, dialogs |
| `--bg-subtle` | `#F8F9FB` | Subtle sections |
| `--bg-elevated` | `rgba(255, 255, 255, 0.88)` | Glass panels |
| `--bg-sidebar` | `#F7F8FA` | Sidebar |
| `--bg-hover` | `#F0F3F8` | Hover surfaces |
| `--bg-active` | `#EAF0FF` | Selected nav, selected rows |
| `--text-primary` | `#182033` | Main text |
| `--text-secondary` | `#566074` | Body/supporting text |
| `--text-tertiary` | `#818A9B` | Muted metadata |
| `--text-disabled` | `#AAB1BD` | Disabled text |
| `--brand-primary` | `#4F67EC` | Main action, selected UI |
| `--brand-primary-hover` | `#465FE4` | Hover action |
| `--brand-primary-active` | `#394FCB` | Pressed action |
| `--brand-soft` | `#EEF1FF` | Brand-tinted background |
| `--brand-border` | `#CFD7FF` | Brand-tinted border |
| `--realtime-primary` | `#18A6B8` | Realtime stream, FPS, current frame |
| `--realtime-soft` | `#E9F9FB` | Realtime subtle background |
| `--realtime-border` | `#BDEBF0` | Realtime border |

### Status Palette

| Status | Main | Background | Border | Text |
| --- | --- | --- | --- | --- |
| Normal | `#16A56A` | `#EAF8F1` | `#BFEBD5` | `#117A50` |
| Running | `#3478F6` | `#EBF3FF` | `#C9DCFF` | `#245FC8` |
| Waiting | `#7A8499` | `#F1F3F6` | `#DCE1E8` | `#596273` |
| Warning | `#D58A16` | `#FFF7E8` | `#F4D9A5` | `#9A6412` |
| Error | `#E14B5A` | `#FFF0F2` | `#F6C5CB` | `#B52E3D` |
| Disabled | `#A5ACB8` | `#F5F6F8` | `#E4E7EC` | `#7A8499` |

## 5. Typography Tokens

Use system fonts first. Do not depend on remote font loading for the desktop app.

```css
--font-ui: Inter, "SF Pro Display", "SF Pro Text", "PingFang SC",
  "Microsoft YaHei", system-ui, sans-serif;
--font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
```

| Role | Size / Line / Weight | Usage |
| --- | --- | --- |
| Page title | `28px / 36px / 650` | Dashboard, Studio, System page title |
| Page subtitle | `14px / 22px / 400` | Title helper text |
| Section title | `18px / 26px / 600` | Main content groups |
| Card title | `14px / 20px / 600` | Panel/card headers |
| Body | `14px / 22px / 400` | Form labels, text |
| Helper | `12px / 18px / 400` | Metadata, descriptions |
| Label | `12px / 16px / 550` | Tags, badges |
| Key number | `24px / 30px / 650` | FPS, latency, GPU usage |
| Large number | `32px / 38px / 650` | Dashboard hero metrics |

Rules:

- Chinese headings should stay at 600 to 650 weight, not heavy bold.
- Letter spacing is `0` for normal text.
- Use monospace only for metrics, tensor shapes, frame IDs, durations, hashes, and code-like values.

## 6. SVG Icon System

All functional icons are SVG by default.

| Property | Rule |
| --- | --- |
| Default viewBox | `0 0 24 24` |
| Default size | `20px` |
| Small size | `16px` |
| Page title | `24px` |
| Card hero icon | `28px` or `32px` |
| Default stroke | `1.7` |
| Emphasis stroke | `1.9` |
| Stroke caps | `round` |
| Stroke joins | `round` |
| Color | `currentColor` |
| Gradients | Not for normal icons |
| Bitmap embedding | Forbidden |

Each icon must remain legible at 16px and should use no more than three visual ideas.

### Icon Categories And Names

| Category | Names |
| --- | --- |
| Navigation | `dashboard`, `studio`, `capture`, `inference`, `models`, `target`, `tracking`, `prediction`, `control`, `devices`, `performance`, `logs`, `system`, `settings`, `help`, `account`, `search`, `command`, `notification` |
| Capture | `camera`, `hdmi-input`, `capture-card`, `video-stream`, `frame`, `fps`, `crop`, `roi`, `resolution`, `aspect-ratio`, `rotate`, `mirror`, `exposure`, `refresh-frame`, `fullscreen`, `preview`, `source-switch`, `signal-input`, `signal-lost`, `timestamp`, `latest-frame` |
| Inference | `ai-model`, `neural-network`, `tensorrt`, `cuda`, `gpu`, `cpu`, `inference-engine`, `model-load`, `model-verify`, `fp16`, `fp32`, `int8`, `onnx`, `engine`, `input-tensor`, `output-tensor`, `batch`, `inference-speed`, `model-cache`, `model-error`, `model-compile`, `model-switch` |
| Tracking | `detection-box`, `target-center`, `candidate-target`, `current-target`, `target-lock`, `target-lost`, `target-filter`, `fov`, `track-trace`, `kalman`, `velocity-vector`, `acceleration`, `future-position`, `confidence`, `track-id`, `occlusion`, `reidentify`, `target-switch`, `prediction-line` |
| Control | `controller`, `pid`, `pd`, `ema`, `smoothing`, `response-curve`, `gain`, `damping`, `error`, `angle`, `pixel-error`, `output`, `velocity-limit`, `acceleration-limit`, `deadzone`, `max-step`, `scheduler`, `command-queue`, `device-send`, `pause-output`, `emergency-stop`, `auto-tune`, `parameter-reset`, `parameter-save` |
| Devices | `jetson`, `usb`, `hid`, `kmbox`, `network`, `lan`, `wifi`, `latency`, `temperature`, `fan`, `memory`, `disk`, `power`, `service`, `daemon`, `plugin`, `driver`, `database`, `backend-api`, `websocket`, `log`, `crash`, `restart`, `update`, `terminal` |
| Device states | `connected`, `disconnected`, `connecting`, `connection-error`, `permission-denied`, `driver-missing` |
| Actions | `start`, `stop`, `pause`, `resume`, `restart`, `refresh`, `save`, `delete`, `edit`, `copy`, `import`, `export`, `upload`, `download`, `expand`, `collapse`, `back`, `forward`, `more`, `filter`, `sort`, `pin`, `unlock`, `lock`, `show`, `hide`, `restore-default`, `undo`, `redo` |
| Status | `check-circle`, `online-dot`, `shield-check`, `play-circle`, `activity-pulse`, `rotating-arc`, `data-stream`, `clock`, `empty-circle`, `triangle-alert`, `thermometer-warning`, `clock-alert`, `error-circle`, `plug-off`, `broken-link`, `x-circle` |

### Required Shape Notes

- `roi`: outer rectangle, inner crop box, four corner handles.
- `latest-frame`: stacked frames, highlighted top frame, small time or pulse mark.
- `gpu`: chip outline, three parallel compute lanes inside.
- `ai-model`: rounded hex/square node frame, neural links in center.
- `target-lock`: rounded target frame, center dot, soft tracking ring, no crosshair.
- `prediction-line`: current point, dashed path, translucent future point.
- `pid`: error waveform into control node into smoothed output curve.
- `ema`: jitter curve plus smoother curve.
- `scheduler`: timeline plus ordered command nodes.

## 7. Navigation Icon States

Selected navigation:

- Brand icon color: `var(--brand-primary)`.
- Background: `var(--brand-soft)`.
- Left selected rail: `2px` width.
- Text: `var(--text-primary)`.

Unselected navigation:

- Icon: `#6F7A8D`.
- Text: `var(--text-secondary)`.
- Background: transparent.

Hover:

- Background: `var(--bg-hover)`.
- Icon/text both move one step darker.

## 8. Status Components

Status must never rely on color alone. Use icon, text, and optional supporting copy.

### Status Dot

Use for compact tables and dense rows.

| Size | Use |
| --- | --- |
| `6px` | Table rows |
| `8px` | Standard compact UI |
| `10px` | Emphasized status |

The visible dot must be paired with readable text, for example `Online`, `TensorRT Ready`, or `Jetson disconnected`.

### Status Badge

Anatomy:

```text
[icon] status label
```

Sizes:

| Size | Height | Padding | Icon |
| --- | --- | --- | --- |
| Small | `24px` | `8px` horizontal | `14px` |
| Default | `28px` | `10px` horizontal | `16px` |
| Large | `32px` | `12px` horizontal | `18px` |

Radius: `999px`.

### Status Card

Use for important runtime blocks:

```text
[status icon]
GPU Runtime
TensorRT ready
48 C · 62%
[optional action]
```

Status cards should include current data when available. They should not be decorative-only.

### Global Status Bar

Use a horizontal set of compact status badges:

```text
Capture Online · TensorRT Ready · Jetson Connected · HID Ready
```

The bar answers one question: "Can the current system run correctly?"

## 9. Feature Illustration System

Illustrations are technical product graphics, not photos and not mascots.

Style:

- Light background.
- Cool blue-gray strokes.
- Local blue-violet highlights.
- Small cyan realtime data-flow accents.
- Translucent glass panels.
- Mild spatial layering.
- No more than three accent colors.

Avoid complex 3D renders, neon, anime/card illustrations, and game-like visuals.

## 10. Illustration Specs

### Startup Flow

Content:

```text
capture device -> video frame -> ROI -> GPU inference -> detections -> target trace -> control output
```

Use six or seven module nodes with flowing connectors. Current stage uses brand blue; completed stages use normal green check; waiting stages use gray; error stage uses red alert.

Recommended file:

```text
src/assets/illustrations/onboarding/startup-flow.svg
```

### Capture

Show input device, frame surface, central ROI, crop handles, timestamp, latest-frame exchange, and GPU ingress. Use abstract software canvas content only.

Recommended file:

```text
src/assets/illustrations/features/capture-pipeline.svg
```

### AI Inference

Show image tensor entering a GPU chip, neural nodes, TensorRT engine box, output detection boxes, and small illustrative metrics like `12.4 ms`, `48 FPS`, `FP16`, `320 x 320`. These values are visual placeholders, not runtime constants.

Recommended file:

```text
src/assets/illustrations/features/inference-runtime.svg
```

### Tracking And Prediction

Show current target point, history trace, velocity direction, predicted position, confidence region, and future dashed path. Current point is brand blue, history is gray-blue, future point is translucent cyan.

Recommended file:

```text
src/assets/illustrations/features/tracking-prediction.svg
```

### Control Algorithm

Show raw error curve, EMA smoothing curve, PD/PID response, output clamp, and convergence. Use two curve layers: jitter input and smooth output.

Recommended file:

```text
src/assets/illustrations/features/control-response.svg
```

### Device Connection

Show capture card, Jetson, GPU, HID device, and desktop client as abstract modules with directional lines and small status labels such as `Connected`, `Online`, `Ready`.

Recommended file:

```text
src/assets/illustrations/devices/device-graph.svg
```

## 11. Empty States

| State | Illustration | Primary action |
| --- | --- | --- |
| No device connected | `empty/no-device.svg` | Connect device |
| No model selected | `empty/no-model.svg` | Select model |
| System not started | `empty/not-started.svg` | Start runtime |
| No detections | `empty/no-detections.svg` | Check input/model |
| No logs | `empty/no-logs.svg` | Refresh |
| No search results | `empty/no-search-results.svg` | Clear search |
| Config incomplete | `empty/config-incomplete.svg` | Complete config |
| Video source unavailable | `empty/video-unavailable.svg` | Select source |
| GPU unavailable | `empty/gpu-unavailable.svg` | Open system check |
| Network disconnected | `empty/network-disconnected.svg` | Retry connection |

Sizes:

- Small: `96 x 96`.
- Standard: `160 x 160`.
- Page-level: `240 x 180`.

An empty state must contain illustration, title, one-sentence explanation, primary action, and optional secondary action.

## 12. Background And Pattern System

Allowed:

- 24px or 32px low-opacity grid.
- Low-opacity dotted field.
- Soft radial light wash.
- Data-flow curve.
- Local scan ring.
- Abstract vision frame.
- Thin trace line.
- Very weak noise texture.
- Translucent glass block.

Do not use large character images or heavy background images in professional operation pages.

Dashboard background:

```text
glacier gray base + weak grid + compact white panels + small realtime signal accents
```

Hero/splash background:

```text
cool white base + pale blue glow left + pale violet glow right + low-opacity vision grid + a few trace curves
```

## 13. Page-Level Usage

### Dashboard

Use device, FPS, GPU, latency, temperature, realtime curve, ROI mini-map, global running status, and subtle data-flow background. The primary question is whether the system is healthy.

### Studio

Use model, parameter, PID, EMA, prediction, control-output icons, grouped parameter panels, curve previews, and help tooltip icons. The primary question is how parameters relate to control behavior.

Parameter groups use subtle grouping backgrounds, not large unrelated colors.

### System

Use Jetson, GPU, USB, network, service, driver, temperature, storage, and status badges. The primary question is whether devices, services, drivers, and hardware are ready.

### Settings

Each settings group must include icon, title, and short explanation. Include restore default, save, import/export, parameter state, and dangerous-zone treatment.

Settings must not be only a long stack of form controls.

## 14. Icon, Text, And Color Rules

Key feature anatomy:

```text
[icon]
Title
Supporting copy
Status or metric
```

Example:

```text
[gpu icon]
GPU Runtime
TensorRT FP16
48 C · 62%
```

Button rules:

- Primary: brand background, white icon, white text.
- Secondary: white background, blue-gray border, dark icon, dark text.
- Danger: pale red or white background, red icon, red text. Solid red only for final confirmation.
- Icon-only buttons require tooltip and `aria-label`.

Card rules:

- Card radius should be `8px` to `12px`, not pill-like.
- Do not nest cards inside cards.
- Data cards show icon, title, number/status, and supporting metadata.
- Dense runtime cards should prefer compact layout over large marketing spacing.

Tag/badge rules:

- Use labels for state and categorical metadata.
- Do not use badge color as the only meaning.
- Numeric data should remain text-first, optionally with icon.

## 15. Motion And Microinteraction

Allowed:

- Status breathing.
- Loading rotation.
- Slow data-flow movement.
- Small icon displacement.
- Card hover lift.
- Smooth number changes.
- Progress fill.
- Flowing connection line.
- Page fade-in.
- Dialog scale/fade.

Durations:

| Motion | Duration |
| --- | --- |
| Fast feedback | `120ms` |
| Standard transition | `180ms` |
| Panel switch | `220ms` |
| Dialog | `240ms` |
| Status breath | `1600ms` to `2200ms` |

Forbidden:

- High-frequency flashing.
- Large bouncing.
- Strong continuous rotation.
- Persistent glow.
- All cards animating together.
- Canvas effects that compete with realtime performance.

All motion must respect `prefers-reduced-motion: reduce`.

## 16. Engineering Directory

Recommended structure:

```text
web/src/
  assets/
    brand/
      logos/
      marks/
    icons/
      navigation/
      capture/
      inference/
      tracking/
      control/
      devices/
      actions/
      status/
    illustrations/
      onboarding/
      empty/
      features/
      devices/
      runtime/
    patterns/
  components/
    visual/
      NovaIcon.tsx
      StatusBadge.tsx
      StatusCard.tsx
      EmptyStateVisual.tsx
  design/
    tokens.css
    iconNames.ts
    statusTokens.ts
```

SVG requirements:

- Do not hardcode `width` or `height` in source icons.
- Use `currentColor`.
- Remove metadata.
- Avoid long uneditable paths when simpler shapes work.
- No embedded bitmaps.
- Avoid heavy filters.
- Support light and dark themes.
- Decorative icons use `aria-hidden="true"`.
- Meaningful icons expose `aria-label`.

## 17. React Icon Component Example

```tsx
import type { SVGProps } from "react";

export type NovaIconName =
  | "gpu"
  | "roi"
  | "latest-frame"
  | "activity-pulse"
  | "triangle-alert"
  | "device-send";

type NovaIconProps = SVGProps<SVGSVGElement> & {
  name: NovaIconName;
  size?: number;
  strokeWidth?: number;
  label?: string;
};

const paths: Record<NovaIconName, JSX.Element> = {
  gpu: (
    <>
      <rect x="5" y="5" width="14" height="14" rx="3" />
      <path d="M9 9h6M9 12h6M9 15h4" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </>
  ),
  roi: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="2.5" />
      <path d="M8 10V8h2M14 8h2v2M16 14v2h-2M10 16H8v-2" />
      <circle cx="8" cy="8" r="1" />
      <circle cx="16" cy="8" r="1" />
      <circle cx="16" cy="16" r="1" />
      <circle cx="8" cy="16" r="1" />
    </>
  ),
  "latest-frame": (
    <>
      <path d="M6 7H4.8A1.8 1.8 0 0 0 3 8.8v8.4A1.8 1.8 0 0 0 4.8 19h8.4A1.8 1.8 0 0 0 15 17.2V16" />
      <rect x="8" y="5" width="13" height="11" rx="2.2" />
      <path d="M16.5 8.5l-2.2 3.1h3.2l-2.1 3" />
    </>
  ),
  "activity-pulse": <path d="M3 12h4l2-5 4 10 2-5h6" />,
  "triangle-alert": (
    <>
      <path d="M12 4l9 16H3L12 4z" />
      <path d="M12 9v4M12 17h.01" />
    </>
  ),
  "device-send": (
    <>
      <rect x="4" y="6" width="8" height="12" rx="2" />
      <path d="M13 9h4.5L16 7.5M13 15h4.5L16 16.5" />
      <path d="M18 9v6" />
    </>
  ),
};

export function NovaIcon({
  name,
  size = 20,
  strokeWidth = 1.7,
  label,
  ...props
}: NovaIconProps) {
  const accessibility = label
    ? { role: "img", "aria-label": label }
    : { "aria-hidden": true };

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...accessibility}
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
```

## 18. React Status Component Example

```tsx
import type { ReactNode } from "react";
import { NovaIcon, type NovaIconName } from "./NovaIcon";

export type NovaStatus =
  | "normal"
  | "running"
  | "waiting"
  | "warning"
  | "error"
  | "disabled";

const statusIcon: Record<NovaStatus, NovaIconName> = {
  normal: "activity-pulse",
  running: "activity-pulse",
  waiting: "latest-frame",
  warning: "triangle-alert",
  error: "triangle-alert",
  disabled: "latest-frame",
};

type StatusBadgeProps = {
  status: NovaStatus;
  label: string;
  detail?: ReactNode;
  size?: "sm" | "md" | "lg";
};

export function StatusBadge({
  status,
  label,
  detail,
  size = "md",
}: StatusBadgeProps) {
  return (
    <span className={`ns-status ns-status-${status} ns-status-${size}`}>
      <NovaIcon name={statusIcon[status]} size={size === "sm" ? 14 : 16} />
      <span className="ns-status-label">{label}</span>
      {detail ? <span className="ns-status-detail">{detail}</span> : null}
    </span>
  );
}
```

## 19. Complete CSS Variables

```css
:root {
  color-scheme: light;

  /* Primitive colors */
  --ns-white: #ffffff;
  --ns-gray-25: #fbfcfe;
  --ns-gray-50: #f8f9fb;
  --ns-gray-100: #f4f6f9;
  --ns-gray-150: #eef2f7;
  --ns-gray-200: #e2e6ec;
  --ns-gray-300: #ccd3dd;
  --ns-gray-400: #aab1bd;
  --ns-gray-500: #818a9b;
  --ns-gray-600: #566074;
  --ns-gray-700: #384155;
  --ns-gray-800: #263044;
  --ns-gray-900: #182033;

  --ns-blue-50: #ebf3ff;
  --ns-blue-100: #c9dcff;
  --ns-blue-500: #3478f6;
  --ns-blue-700: #245fc8;

  --ns-brand-50: #eef1ff;
  --ns-brand-100: #cfd7ff;
  --ns-brand-500: #4f67ec;
  --ns-brand-600: #465fe4;
  --ns-brand-700: #394fcb;

  --ns-cyan-50: #e9f9fb;
  --ns-cyan-100: #bdebf0;
  --ns-cyan-500: #18a6b8;

  --ns-green-50: #eaf8f1;
  --ns-green-100: #bfebd5;
  --ns-green-500: #16a56a;
  --ns-green-700: #117a50;

  --ns-amber-50: #fff7e8;
  --ns-amber-100: #f4d9a5;
  --ns-amber-500: #d58a16;
  --ns-amber-700: #9a6412;

  --ns-red-50: #fff0f2;
  --ns-red-100: #f6c5cb;
  --ns-red-500: #e14b5a;
  --ns-red-700: #b52e3d;

  /* Semantic backgrounds */
  --bg-app: var(--ns-gray-100);
  --bg-surface: var(--ns-white);
  --bg-subtle: var(--ns-gray-50);
  --bg-elevated: rgba(255, 255, 255, 0.88);
  --bg-sidebar: #f7f8fa;
  --bg-hover: #f0f3f8;
  --bg-active: #eaf0ff;

  /* Semantic text */
  --text-primary: var(--ns-gray-900);
  --text-secondary: var(--ns-gray-600);
  --text-tertiary: var(--ns-gray-500);
  --text-disabled: var(--ns-gray-400);
  --text-inverse: var(--ns-white);
  --text-link: #356de8;

  /* Brand and realtime */
  --brand-primary: var(--ns-brand-500);
  --brand-primary-hover: var(--ns-brand-600);
  --brand-primary-active: var(--ns-brand-700);
  --brand-soft: var(--ns-brand-50);
  --brand-border: var(--ns-brand-100);
  --realtime-primary: var(--ns-cyan-500);
  --realtime-soft: var(--ns-cyan-50);
  --realtime-border: var(--ns-cyan-100);

  /* Borders and shadows */
  --border-default: var(--ns-gray-200);
  --border-strong: var(--ns-gray-300);
  --border-focus: #8fa3ff;
  --divider: #e9ecf1;
  --shadow-card: 0 1px 2px rgba(22, 32, 51, 0.04),
    0 8px 24px rgba(22, 32, 51, 0.06);
  --shadow-dialog: 0 18px 50px rgba(25, 35, 55, 0.16);

  /* Typography */
  --font-ui: Inter, "SF Pro Display", "SF Pro Text", "PingFang SC",
    "Microsoft YaHei", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
  --text-page-title: 28px;
  --leading-page-title: 36px;
  --text-section-title: 18px;
  --leading-section-title: 26px;
  --text-body: 14px;
  --leading-body: 22px;
  --text-helper: 12px;
  --leading-helper: 18px;
  --text-label: 12px;
  --leading-label: 16px;
  --text-key-number: 24px;
  --leading-key-number: 30px;
  --text-large-number: 32px;
  --leading-large-number: 38px;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;

  /* Radius */
  --radius-2: 4px;
  --radius-3: 6px;
  --radius-4: 8px;
  --radius-5: 10px;
  --radius-6: 12px;
  --radius-pill: 999px;

  /* Motion */
  --duration-fast: 120ms;
  --duration-normal: 180ms;
  --duration-panel: 220ms;
  --duration-dialog: 240ms;
  --duration-breath: 1800ms;
  --ease-standard: cubic-bezier(0.2, 0, 0, 1);

  /* Component tokens */
  --button-height: 36px;
  --button-height-compact: 30px;
  --button-radius: var(--radius-4);
  --button-primary-bg: var(--brand-primary);
  --button-primary-bg-hover: var(--brand-primary-hover);
  --button-primary-fg: var(--text-inverse);
  --button-secondary-bg: var(--bg-surface);
  --button-secondary-border: var(--border-default);
  --button-secondary-fg: var(--text-primary);

  --card-bg: var(--bg-surface);
  --card-border: var(--border-default);
  --card-radius: var(--radius-4);
  --card-shadow: var(--shadow-card);

  --status-normal-bg: var(--ns-green-50);
  --status-normal-border: var(--ns-green-100);
  --status-normal-fg: var(--ns-green-700);
  --status-normal-icon: var(--ns-green-500);
  --status-running-bg: var(--ns-blue-50);
  --status-running-border: var(--ns-blue-100);
  --status-running-fg: var(--ns-blue-700);
  --status-running-icon: var(--ns-blue-500);
  --status-waiting-bg: #f1f3f6;
  --status-waiting-border: #dce1e8;
  --status-waiting-fg: #596273;
  --status-waiting-icon: #7a8499;
  --status-warning-bg: var(--ns-amber-50);
  --status-warning-border: var(--ns-amber-100);
  --status-warning-fg: var(--ns-amber-700);
  --status-warning-icon: var(--ns-amber-500);
  --status-error-bg: var(--ns-red-50);
  --status-error-border: var(--ns-red-100);
  --status-error-fg: var(--ns-red-700);
  --status-error-icon: var(--ns-red-500);
  --status-disabled-bg: #f5f6f8;
  --status-disabled-border: #e4e7ec;
  --status-disabled-fg: #7a8499;
  --status-disabled-icon: #8a92a0;
}

[data-theme="dark"] {
  color-scheme: dark;
  --bg-app: #111827;
  --bg-surface: #182033;
  --bg-subtle: #202a3e;
  --bg-elevated: rgba(24, 32, 51, 0.88);
  --bg-sidebar: #151d2c;
  --bg-hover: #24304a;
  --bg-active: rgba(85, 111, 246, 0.18);
  --text-primary: #f4f7fb;
  --text-secondary: #c6cfdd;
  --text-tertiary: #94a0b4;
  --text-disabled: #687386;
  --text-link: #9db0ff;
  --border-default: rgba(198, 207, 221, 0.16);
  --border-strong: rgba(198, 207, 221, 0.28);
  --divider: rgba(198, 207, 221, 0.14);
  --card-bg: #182033;
  --card-border: rgba(198, 207, 221, 0.16);
  --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.24),
    0 12px 32px rgba(0, 0, 0, 0.22);
}

@media (prefers-reduced-motion: reduce) {
  :root {
    --duration-fast: 1ms;
    --duration-normal: 1ms;
    --duration-panel: 1ms;
    --duration-dialog: 1ms;
    --duration-breath: 1ms;
  }
}
```

## 20. Status CSS Example

```css
.ns-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--status-border);
  border-radius: var(--radius-pill);
  background: var(--status-bg);
  color: var(--status-fg);
  font-size: var(--text-label);
  line-height: var(--leading-label);
  font-weight: 550;
  white-space: nowrap;
}

.ns-status svg {
  color: var(--status-icon);
  flex: none;
}

.ns-status-sm {
  min-height: 24px;
  padding: 0 8px;
}

.ns-status-md {
  min-height: 28px;
  padding: 0 10px;
}

.ns-status-lg {
  min-height: 32px;
  padding: 0 12px;
}

.ns-status-normal {
  --status-bg: var(--status-normal-bg);
  --status-border: var(--status-normal-border);
  --status-fg: var(--status-normal-fg);
  --status-icon: var(--status-normal-icon);
}

.ns-status-running {
  --status-bg: var(--status-running-bg);
  --status-border: var(--status-running-border);
  --status-fg: var(--status-running-fg);
  --status-icon: var(--status-running-icon);
}

.ns-status-running svg {
  animation: ns-status-breath var(--duration-breath) ease-in-out infinite;
}

@keyframes ns-status-breath {
  0%, 100% { opacity: 0.68; }
  50% { opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
  .ns-status-running svg {
    animation: none;
  }
}
```

## 21. Dark Mode Principles

- Dark mode overrides semantic tokens, not component code.
- Keep brand blue-violet slightly brighter on dark surfaces.
- Keep realtime cyan muted enough to avoid neon.
- Use borders more than shadows for depth.
- Avoid pure black backgrounds; use deep blue-gray.
- Illustrations should switch strokes and panel fills through CSS variables where possible.

## 22. Accessibility Requirements

- Normal text contrast: at least 4.5:1.
- Large text and UI boundaries: at least 3:1.
- Never communicate state only with color.
- Icon-only buttons need `aria-label` and tooltip.
- Decorative SVGs use `aria-hidden="true"`.
- Loading buttons use `aria-busy="true"` when relevant.
- Error messages use `role="alert"` or are tied by `aria-describedby`.
- Keyboard focus must be visible with a 2px focus ring.
- Motion must respect `prefers-reduced-motion`.

## 23. Performance Limits

- Prefer SVG line art and CSS variables over canvas effects for UI decoration.
- Avoid SVG filters in repeated icons and large tables.
- Keep icon components tree-shakeable.
- Do not animate layout-affecting properties in realtime panels.
- Use `transform` and `opacity` only for microinteractions.
- Do not run decorative animation in hidden tabs or inactive panels.
- Realtime charts should update at a UI-safe cadence rather than every backend event when event rate is high.

## 24. Implementation Checklist

- [ ] Create `NovaIcon` and icon-name registry.
- [ ] Move status visuals into `StatusBadge`, `StatusCard`, and global status bar.
- [ ] Replace color-only dots with icon + text status.
- [ ] Expand `web/src/design/tokens.css` into primitive, semantic, and component layers.
- [ ] Add brand mark and logo variants under `web/src/assets/brand/`.
- [ ] Add illustration folders for onboarding, empty states, runtime, features, and devices.
- [ ] Update Dashboard, Studio, System, and Settings to use icon + title + status/metric anatomy.
- [ ] Add reduced-motion handling for all breathing/loading/data-flow animation.
- [ ] Validate contrast for all badge and button states.

## 25. Acceptance Criteria

The v1 visual system is acceptable when:

- Icons share viewBox, stroke, cap, join, sizing, and naming conventions.
- Every critical status has icon, color, text, and optional detail.
- Dashboard answers system health quickly.
- Studio makes model, ROI, inference, prediction, and control relationships visible.
- System makes hardware/service/driver readiness obvious.
- Empty states guide the next action instead of only saying "no data".
- Brand graphics are recognizable without becoming a weapon-like crosshair.
- Runtime pages remain readable under dense data.
- Animations are quiet and disabled under reduced-motion.
- Assets are organized so new modules can add icons and illustrations without changing the component API.
