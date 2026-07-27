use std::time::Duration;

use novasight_runtime::{RuntimeDependencies, RuntimeSupervisor};

#[tokio::test]
async fn cancelling_the_owner_join_future_latches_output_closed() {
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
    runtime.start().await.unwrap();
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

    let owner = tokio::spawn(supervisor.join());
    tokio::task::yield_now().await;
    owner.abort();
    assert!(owner.await.unwrap_err().is_cancelled());

    tokio::time::timeout(Duration::from_secs(5), runtime.wait_for_supervisor_exit())
        .await
        .expect("supervisor survived cancellation of its sole owner");
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
}
