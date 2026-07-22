use std::process::ExitCode;

/// Legacy executable name retained for rollback compatibility.
#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    relink_server::entry().await
}
