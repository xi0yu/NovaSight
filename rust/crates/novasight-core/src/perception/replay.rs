use std::collections::VecDeque;

use async_trait::async_trait;

use crate::{AppError, DetectionBatch, PerceptionSource};

/// Finite replay source that yields its validated batches exactly once, in order.
#[derive(Clone, Debug, Default)]
pub struct ReplayPerceptionSource {
    batches: VecDeque<DetectionBatch>,
}

impl ReplayPerceptionSource {
    pub fn new(batches: impl IntoIterator<Item = DetectionBatch>) -> Self {
        Self {
            batches: batches.into_iter().collect(),
        }
    }

    pub fn remaining(&self) -> usize {
        self.batches.len()
    }
}

#[async_trait]
impl PerceptionSource for ReplayPerceptionSource {
    async fn next_batch(&mut self) -> Result<Option<DetectionBatch>, AppError> {
        Ok(self.batches.pop_front())
    }
}
