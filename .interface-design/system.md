# NovaSight Studio Interface System

## Direction

NovaSight Studio is a realtime vision-control workbench for an operator who wants to start the system, confirm the current state, tune a small number of meaningful parameters, and diagnose failures only when needed.

The interface should feel quiet, dense, precise, and production-grade. It should not feel like a marketing dashboard, a generic admin template, or a developer log viewer.

## Reference Lenses

- Karri Saarinen / Linear: high-density product tools need speed, restraint, and strong hierarchy before decoration.
- Brian Lovin: design quality must survive implementation; controls, states, keyboard behavior, and text fit are part of the design.
- Rauno Freiberg / Vercel: prioritize scannability, layout stability, performance, and information honesty over visual tricks.
- Diana Mounter / GitHub Primer: repeated UI decisions become system rules; avoid one-off styles that drift across themes.

## Information Hierarchy

- Default Studio pages show current status, current FPS/freshness, recent result, and user-operable controls.
- Cumulative counters, protocol receipts, generation numbers, raw trace fields, and backend proof belong in collapsed diagnostics.
- One fact should have one owner. Other pages may reference it as a short status, not repeat full details.
- Control pages should explain the live chain as prediction -> control -> limit -> output. Detailed numbers are evidence, not the main story.

## Visual System

- Density: workbench-tight. Use 10-16px internal spacing for cards and 12px grid gaps unless a section is a major page-level transition.
- Depth: borders-first with subtle surface shifts. Avoid heavy shadows, decorative gradients, and multiple accent colors.
- Radius: keep operational panels at 8-12px. Small controls use smaller radii; do not introduce large soft cards inside dense tools.
- Typography: weight and color carry hierarchy more than size. Values use tabular mono only when they are metrics, IDs, counts, dimensions, or timings.
- Color: dominant cool white / glacier gray surfaces, blue-violet for primary actions, cyan-blue for realtime signals, semantic colors only for actual status.

## Component Patterns

- Browser access gate: use one centered task card with product identity, one access-code field, one primary action, contextual errors, and a quiet connection/status footer. Keep transport, CSRF, IPC, and trust-chain details out of the default view.
- License activation gate: keep caller authentication separate from product licensing. Temporary and formal credentials share one activation form and one submit path; explain the resulting tier after validation instead of presenting separate pre-validation actions.
- Diagnostic disclosure: summary row with title, short reason, and item count; content is collapsed by default and uses the same dense card grid when opened.
- Output gate: primary user-facing safety control. Keep it visible in the parameter page and avoid duplicating it as a separate summary metric.
- Prediction switch: user-facing algorithm switch. Keep it visible before advanced algorithm parameters.
- Configuration pages: the configuration profile, output gate, prediction switch, class configuration, and algorithm entry points are the page body; do not add a duplicate metric strip above them.

## Checks Before Shipping

- Squint check: the page should still reveal the main action and live state when details are visually blurred.
- Duplicate check: if the same runtime fact appears twice on the same page, one instance should become a diagnostic detail or be removed.
- Theme check: new controls must use semantic tokens, not raw colors.
- Runtime honesty check: a displayed value must say whether it is current runtime state, saved config, or diagnostic evidence.
