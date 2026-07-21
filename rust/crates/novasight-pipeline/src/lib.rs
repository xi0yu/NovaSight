//! `novasight-pipeline` — capture / inference / control / device
//! orchestration on dedicated `std::thread` workers, with typed
//! `LatestSlot<T>` data flow between stages.
//!
//! Commit 1 ships the empty crate skeleton. The `Worker` trait,
//! `LatestSlot<T>`, Fake pipeline, and the algorithm wiring land in
//! subsequent commits.

#![forbid(unsafe_code)]
