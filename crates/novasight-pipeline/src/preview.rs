use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use novasight_core::RuntimeEpoch;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;

#[derive(Clone, Debug)]
pub struct PreviewFrame {
    pub sequence: u64,
    pub jpeg: Arc<[u8]>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreviewSnapshot {
    #[serde(rename = "preview_enabled")]
    pub enabled: bool,
    #[serde(rename = "preview_running")]
    pub running: bool,
    #[serde(rename = "preview_active")]
    pub active: bool,
    #[serde(rename = "preview_encoder_active")]
    pub encoder_active: bool,
    #[serde(rename = "preview_consumers")]
    pub consumers: usize,
    #[serde(rename = "preview_available")]
    pub available: bool,
    #[serde(rename = "preview_sequence")]
    pub sequence: u64,
    #[serde(rename = "preview_reason")]
    pub reason: String,
    #[serde(rename = "preview_transport")]
    pub transport: String,
}

#[derive(Clone, Debug)]
pub struct PreviewHub {
    inner: Arc<PreviewInner>,
}

#[derive(Debug)]
struct PreviewInner {
    enabled: AtomicBool,
    epoch: AtomicU64,
    running: AtomicBool,
    active: AtomicBool,
    encoder_active: AtomicBool,
    consumers: AtomicUsize,
    sequence: AtomicU64,
    frames: watch::Sender<Option<Arc<PreviewFrame>>>,
}

impl PreviewHub {
    pub fn new(enabled: bool) -> Self {
        let (frames, _) = watch::channel(None);
        Self {
            inner: Arc::new(PreviewInner {
                enabled: AtomicBool::new(enabled),
                epoch: AtomicU64::new(0),
                running: AtomicBool::new(false),
                active: AtomicBool::new(false),
                encoder_active: AtomicBool::new(false),
                consumers: AtomicUsize::new(0),
                sequence: AtomicU64::new(0),
                frames,
            }),
        }
    }

    pub fn begin_epoch(&self, epoch: RuntimeEpoch) {
        self.inner.epoch.store(epoch.0, Ordering::Release);
        self.inner.sequence.store(0, Ordering::Release);
        self.inner.active.store(false, Ordering::Release);
        self.inner.encoder_active.store(false, Ordering::Release);
        self.inner.running.store(true, Ordering::Release);
        self.inner.frames.send_replace(None);
    }

    pub fn configure(&self, enabled: bool) {
        self.inner.enabled.store(enabled, Ordering::Release);
        if !enabled {
            self.inner.active.store(false, Ordering::Release);
            self.inner.encoder_active.store(false, Ordering::Release);
            self.inner.frames.send_replace(None);
        }
    }

    pub fn end_epoch(&self, epoch: RuntimeEpoch) {
        if self.inner.epoch.load(Ordering::Acquire) != epoch.0 {
            return;
        }
        self.inner.running.store(false, Ordering::Release);
        self.inner.active.store(false, Ordering::Release);
        self.inner.encoder_active.store(false, Ordering::Release);
        self.inner.frames.send_replace(None);
    }

    pub fn set_active(&self, active: bool) -> Result<PreviewSnapshot, PreviewError> {
        if !self.inner.enabled.load(Ordering::Acquire) {
            return Err(PreviewError::Disabled);
        }
        if !self.inner.running.load(Ordering::Acquire) {
            return Err(PreviewError::PipelineNotRunning);
        }
        self.inner.active.store(active, Ordering::Release);
        if !active {
            self.inner.encoder_active.store(false, Ordering::Release);
            self.inner.frames.send_replace(None);
        }
        Ok(self.snapshot())
    }

    pub fn subscribe(&self) -> Result<PreviewSubscription, PreviewError> {
        if !self.inner.enabled.load(Ordering::Acquire) {
            return Err(PreviewError::Disabled);
        }
        if !self.inner.running.load(Ordering::Acquire) {
            return Err(PreviewError::PipelineNotRunning);
        }
        if !self.inner.active.load(Ordering::Acquire) {
            return Err(PreviewError::Paused);
        }
        self.inner.consumers.fetch_add(1, Ordering::AcqRel);
        Ok(PreviewSubscription {
            inner: Arc::clone(&self.inner),
            frames: self.inner.frames.subscribe(),
        })
    }

    pub fn encoder_requested(&self, epoch: RuntimeEpoch) -> bool {
        self.inner.enabled.load(Ordering::Acquire)
            && self.inner.epoch.load(Ordering::Acquire) == epoch.0
            && self.inner.running.load(Ordering::Acquire)
            && self.inner.active.load(Ordering::Acquire)
            && self.inner.consumers.load(Ordering::Acquire) > 0
    }

    pub fn set_encoder_active(&self, epoch: RuntimeEpoch, active: bool) {
        if self.inner.epoch.load(Ordering::Acquire) == epoch.0 {
            self.inner.encoder_active.store(active, Ordering::Release);
            if !active {
                self.inner.frames.send_replace(None);
            }
        }
    }

    pub fn publish_jpeg(&self, epoch: RuntimeEpoch, jpeg: Vec<u8>) -> Result<u64, PreviewError> {
        if self.inner.epoch.load(Ordering::Acquire) != epoch.0
            || !self.inner.encoder_active.load(Ordering::Acquire)
        {
            return Err(PreviewError::StaleEpoch);
        }
        if jpeg.len() < 4 || !jpeg.starts_with(&[0xff, 0xd8]) || !jpeg.ends_with(&[0xff, 0xd9]) {
            return Err(PreviewError::InvalidJpeg);
        }
        let sequence = self.inner.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        self.inner.frames.send_replace(Some(Arc::new(PreviewFrame {
            sequence,
            jpeg: jpeg.into(),
        })));
        Ok(sequence)
    }

    pub fn snapshot(&self) -> PreviewSnapshot {
        let enabled = self.inner.enabled.load(Ordering::Acquire);
        let running = self.inner.running.load(Ordering::Acquire);
        let active = self.inner.active.load(Ordering::Acquire);
        let encoder_active = self.inner.encoder_active.load(Ordering::Acquire);
        let consumers = self.inner.consumers.load(Ordering::Acquire);
        let available = self.inner.frames.borrow().is_some();
        let reason = if !enabled {
            "preview consumer is disabled"
        } else if !running {
            "DeepStream pipeline is not running"
        } else if !active {
            "preview paused to preserve inference performance"
        } else if consumers == 0 {
            "preview ready; waiting for a browser viewer"
        } else if !available {
            "waiting for the first hardware JPEG preview frame"
        } else {
            ""
        };
        PreviewSnapshot {
            enabled,
            running,
            active,
            encoder_active,
            consumers,
            available,
            sequence: self.inner.sequence.load(Ordering::Acquire),
            reason: reason.to_owned(),
            transport: "nvmm_nvjpegenc_to_mjpeg_bytes".to_owned(),
        }
    }
}

pub struct PreviewSubscription {
    inner: Arc<PreviewInner>,
    frames: watch::Receiver<Option<Arc<PreviewFrame>>>,
}

impl PreviewSubscription {
    pub async fn next(&mut self) -> Option<Arc<PreviewFrame>> {
        loop {
            if self.frames.changed().await.is_err() {
                return None;
            }
            if let Some(frame) = self.frames.borrow_and_update().clone() {
                return Some(frame);
            }
        }
    }
}

impl Drop for PreviewSubscription {
    fn drop(&mut self) {
        self.inner.consumers.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PreviewError {
    #[error("hardware preview is disabled by configuration")]
    Disabled,
    #[error("DeepStream pipeline is not running")]
    PipelineNotRunning,
    #[error("hardware preview is paused")]
    Paused,
    #[error("preview frame belongs to an inactive runtime epoch")]
    StaleEpoch,
    #[error("hardware preview buffer is not a complete JPEG image")]
    InvalidJpeg,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn preview_only_requests_encoder_with_an_active_consumer() {
        let hub = PreviewHub::new(true);
        let epoch = RuntimeEpoch(7);
        hub.begin_epoch(epoch);
        hub.set_active(true).unwrap();
        assert!(!hub.encoder_requested(epoch));
        let mut subscriber = hub.subscribe().unwrap();
        assert!(hub.encoder_requested(epoch));
        hub.set_encoder_active(epoch, true);
        hub.publish_jpeg(epoch, vec![0xff, 0xd8, 1, 0xff, 0xd9])
            .unwrap();
        assert_eq!(subscriber.next().await.unwrap().sequence, 1);
        drop(subscriber);
        assert!(!hub.encoder_requested(epoch));
    }

    #[test]
    fn stale_epochs_and_invalid_payloads_are_rejected() {
        let hub = PreviewHub::new(true);
        let epoch = RuntimeEpoch(2);
        hub.begin_epoch(epoch);
        hub.set_active(true).unwrap();
        let _subscriber = hub.subscribe().unwrap();
        hub.set_encoder_active(epoch, true);
        assert_eq!(
            hub.publish_jpeg(RuntimeEpoch(1), vec![0xff, 0xd8, 0xff, 0xd9]),
            Err(PreviewError::StaleEpoch)
        );
        assert_eq!(
            hub.publish_jpeg(epoch, vec![0xff, 0xd8, 1]),
            Err(PreviewError::InvalidJpeg)
        );
    }
}
