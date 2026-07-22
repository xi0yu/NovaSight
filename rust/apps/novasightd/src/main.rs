//! Canonical NovaSight daemon executable.
//!
//! The composition code lives in the `relink-server` library crate, but this
//! is the only server executable. Legacy Python remains the rollback path.

use std::process::ExitCode;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    relink_server::entry().await
}
