//! Canonical NovaSight daemon executable.
//!
//! The composition code remains shared with the legacy `relink_server`
//! executable so existing installations retain a rollback path without
//! creating a second runtime implementation.

use std::process::ExitCode;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    relink_server::entry().await
}
