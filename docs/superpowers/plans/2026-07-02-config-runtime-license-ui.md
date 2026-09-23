# Config Runtime License UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep frontend configuration, backend configuration, runtime state, URLs, and local license-key management consistent.

**Architecture:** Backend exposes canonical config metadata and license APIs. Frontend imports endpoint constants from one API module, renders config controls from backend schema, and uses relative URLs/proxy-safe helpers for REST, stream, and WebSocket paths.

**Tech Stack:** FastAPI, dataclass runtime config, React, TypeScript, Vite.

---

## Tasks

- [x] Add backend config schema metadata and apply config changes to runtime, capture, executors, and hardware.
- [x] Add local license-key manager that stores only a hash/status and never returns plaintext keys.
- [x] Centralize frontend URL paths, base URL handling, MJPEG URL generation, and WebSocket URL generation.
- [x] Add Chinese config/license workbench UI with Gemma-style lighter, clearer parameter panels.
- [x] Proxy `/ws` in Vite and verify frontend calls match backend routes.
- [x] Add broad tests for schema, config sync, license storage, and URL helpers.
- [x] Run `pytest -q`, `pnpm --dir web typecheck`, `pnpm --dir web build`, and `git diff --check`.
