# NovaSight Historical Plans Archive

This directory is an archive of prior planning, design, and generated reports.
It is useful for historical context, but it is not the active source of project
truth.

Use the current authority in this order:

1. `PROJECT_HEALTH_AUDIT.md` for cleanup, over-design, and health decisions.
2. `docs/ai-programming-team.md` for agent operating rules.
3. `docs/novasight-ai-purpose-audit-questions.md` for evidence-based audits.
4. `docs/novasight-deepstream-code-plan.md` for the current DeepStream/NVMM
   implementation plan.
5. Current production code and verification commands.

Do not treat files under `docs/superpowers/plans/`,
`docs/superpowers/specs/`, or `docs/superpowers/out/` as current requirements
unless a current authority document explicitly promotes that item.

When a historical plan conflicts with current code or current authority docs,
record the conflict in `PROJECT_HEALTH_AUDIT.md` instead of following the
historical plan silently.
