# NovaSight SVG Icon Assets

Runtime UI icons are rendered through `web/src/components/visual/NovaIcon.tsx` so they can share `currentColor`, `viewBox="0 0 24 24"`, stroke width, accessibility behavior, and tree-shaking.

This folder mirrors the visual-system category contract from `docs/novasight-frontend-visual-system.md` and is reserved for exported source SVGs when a designer supplies hand-tuned artwork. Keep category names stable:

- `navigation/`
- `capture/`
- `inference/`
- `tracking/`
- `control/`
- `devices/`
- `actions/`
- `status/`

Do not use emoji, bitmap embeds, heavy filters, or hardcoded colors in these SVGs.
