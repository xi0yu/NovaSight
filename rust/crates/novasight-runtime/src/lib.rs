//! `novasight-runtime` — daemon lifecycle, RuntimeSupervisor,
//! PipelineRuntime orchestration, and API server wiring.
//!
//! Commit 1 ships the empty crate skeleton. The supervisor loop,
//! Application, ApiServer, and preflight module land in subsequent
//! commits.

#![forbid(unsafe_code)]
