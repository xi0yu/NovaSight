use std::collections::VecDeque;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};
use novasight_core::RuntimeEpoch;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct CrosshairConfig {
    pub enabled: bool,
    pub use_for_control: bool,
    pub search_size: u32,
    pub sample_frames: usize,
    pub confirm_duration: Duration,
    pub max_age: Duration,
    pub max_offset_px: f64,
    pub min_similarity: f64,
    pub max_step_px: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CrosshairObservation {
    pub status: String,
    pub found: bool,
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
    pub sample_ts_ns: u64,
    pub template_id: String,
    pub point_hits: usize,
    pub point_count: usize,
    pub offset_x: f64,
    pub offset_y: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ControlReference {
    pub x: f64,
    pub y: f64,
    pub source: String,
    pub confidence: f64,
    pub age_ms: f64,
    pub geometry_signature: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CrosshairTemplateSummary {
    pub id: String,
    pub schema_version: u32,
    pub created_ts_ns: u64,
    pub geometry_signature: String,
    pub search_size: u32,
    pub point_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CrosshairSnapshot {
    pub enabled: bool,
    pub use_for_control: bool,
    pub running: bool,
    pub state: String,
    pub observation: Option<CrosshairObservation>,
    pub control_reference_ready: bool,
    pub control_reference_source: String,
    pub control_reference_reason: String,
    pub recent_samples: usize,
    pub required_samples: usize,
    pub processed_frames: u64,
    pub matched_frames: u64,
    pub dropped_frames: u64,
    pub last_error: String,
    pub template: Option<CrosshairTemplateSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Template {
    schema_version: u32,
    id: String,
    created_ts_ns: u64,
    geometry_signature: String,
    search_size: u32,
    points: Vec<TemplatePoint>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct TemplatePoint(i32, i32, u8, u8, u8, u8);

#[derive(Debug)]
struct Sample {
    observed_at: Instant,
    roi_width: u32,
    roi_height: u32,
    image: RgbImage,
}

#[derive(Debug)]
struct State {
    epoch: u64,
    running: bool,
    recent: VecDeque<Sample>,
    template: Option<Template>,
    latest: Option<CrosshairObservation>,
    latest_at: Option<Instant>,
    last_confirmed: Option<CrosshairObservation>,
    last_confirmed_at: Option<Instant>,
    candidate_xy: Option<(f64, f64)>,
    candidate_since: Option<Instant>,
    processed_frames: u64,
    matched_frames: u64,
    dropped_frames: u64,
    last_error: String,
}

#[derive(Clone, Debug)]
pub struct CrosshairHub {
    config: Arc<CrosshairConfig>,
    template_path: Arc<PathBuf>,
    state: Arc<Mutex<State>>,
    mutation: Arc<Mutex<()>>,
}

/// Process-local owner of the currently installed crosshair observer.
///
/// Runtime epoch reloads replace the hub through this shared slot so the
/// perception adapter, control pipeline, and control-plane API all resolve the
/// same instance. Keeping the indirection outside the frame path avoids a
/// configuration lock in crosshair matching and controller evaluation.
#[derive(Clone, Debug, Default)]
pub struct CrosshairHubSlot {
    current: Arc<RwLock<Option<CrosshairHub>>>,
}

impl CrosshairHubSlot {
    pub fn new(current: Option<CrosshairHub>) -> Self {
        Self {
            current: Arc::new(RwLock::new(current)),
        }
    }

    pub fn current(&self) -> Option<CrosshairHub> {
        self.current
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn replace(&self, replacement: Option<CrosshairHub>) -> Option<CrosshairHub> {
        std::mem::replace(
            &mut *self
                .current
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            replacement,
        )
    }
}

impl CrosshairHub {
    pub fn new(config: CrosshairConfig, template_path: impl Into<PathBuf>) -> Self {
        let template_path = template_path.into();
        let (template, last_error) = match load_template(&template_path) {
            Ok(template) => (template, String::new()),
            Err(error) => (None, format!("template load failed: {error}")),
        };
        Self {
            config: Arc::new(config),
            template_path: Arc::new(template_path),
            state: Arc::new(Mutex::new(State {
                epoch: 0,
                running: false,
                recent: VecDeque::new(),
                template,
                latest: None,
                latest_at: None,
                last_confirmed: None,
                last_confirmed_at: None,
                candidate_xy: None,
                candidate_since: None,
                processed_frames: 0,
                matched_frames: 0,
                dropped_frames: 0,
                last_error,
            })),
            mutation: Arc::new(Mutex::new(())),
        }
    }

    pub fn begin_epoch(
        &self,
        epoch: RuntimeEpoch,
        roi_width: u32,
        roi_height: u32,
    ) -> Result<CrosshairEpoch, CrosshairError> {
        if !self.config.enabled {
            return Err(CrosshairError::Disabled);
        }
        let (tx, rx) = sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        {
            let mut state = lock(&self.state);
            state.epoch = epoch.0;
            state.running = true;
            clear_runtime(&mut state);
        }
        let worker_hub = self.clone();
        let worker_stop = Arc::clone(&stop);
        let join = match thread::Builder::new()
            .name("novasight-crosshair".to_owned())
            .spawn(move || worker_hub.observe_frames(epoch, roi_width, roi_height, rx, worker_stop))
        {
            Ok(join) => join,
            Err(error) => {
                lock(&self.state).running = false;
                return Err(CrosshairError::WorkerSpawn(error));
            }
        };
        Ok(CrosshairEpoch {
            publisher: CrosshairFramePublisher {
                epoch,
                tx,
                state: Arc::clone(&self.state),
            },
            stop,
            join: Some(join),
            hub: self.clone(),
        })
    }

    fn observe_frames(
        &self,
        epoch: RuntimeEpoch,
        roi_width: u32,
        roi_height: u32,
        rx: Receiver<Frame>,
        stop: Arc<AtomicBool>,
    ) {
        while !stop.load(Ordering::Acquire) {
            let frame = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(frame) => frame,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            if let Err(error) = self.ingest(epoch, frame, roi_width, roi_height) {
                lock(&self.state).last_error = error.to_string();
            }
        }
    }

    fn ingest(
        &self,
        epoch: RuntimeEpoch,
        frame: Frame,
        roi_width: u32,
        roi_height: u32,
    ) -> Result<(), CrosshairError> {
        let image = image::load_from_memory_with_format(&frame.jpeg, ImageFormat::Jpeg)
            .map_err(CrosshairError::Decode)?
            .into_rgb8();
        if image.width() != self.config.search_size || image.height() != self.config.search_size {
            return Err(CrosshairError::UnexpectedGeometry {
                expected: self.config.search_size,
                width: image.width(),
                height: image.height(),
            });
        }
        let observed_at = Instant::now();
        let mut state = lock(&self.state);
        if state.epoch != epoch.0 || !state.running {
            return Ok(());
        }
        state.processed_frames += 1;
        state.last_error.clear();
        state.recent.push_back(Sample {
            observed_at,
            roi_width,
            roi_height,
            image: image.clone(),
        });
        while state.recent.len() > self.config.sample_frames.max(3) {
            state.recent.pop_front();
        }
        let Some(template) = state.template.clone() else {
            set_empty_observation(
                &mut state,
                "searching",
                frame.sample_ts_ns,
                "template_unavailable",
                observed_at,
            );
            return Ok(());
        };
        if template.geometry_signature != geometry_signature(roi_width, roi_height) {
            set_empty_observation(
                &mut state,
                "searching",
                frame.sample_ts_ns,
                "geometry_changed",
                observed_at,
            );
            return Ok(());
        }
        let hint = state
            .last_confirmed
            .as_ref()
            .map(|item| (item.offset_x.round() as i32, item.offset_y.round() as i32));
        drop(state);
        let matched = match_template(
            &image,
            &template,
            self.config.max_offset_px.round() as i32,
            hint,
        );
        let mut state = lock(&self.state);
        if state.epoch != epoch.0
            || state.template.as_ref().map(|item| item.id.as_str()) != Some(template.id.as_str())
        {
            return Ok(());
        }
        let Some(found) = matched.filter(|item| {
            item.confidence >= self.config.min_similarity && item.uniqueness >= 0.015
        }) else {
            state.candidate_xy = None;
            state.candidate_since = None;
            let status = if state.last_confirmed.is_some() {
                "uncertain"
            } else {
                "searching"
            };
            set_empty_observation(
                &mut state,
                status,
                frame.sample_ts_ns,
                "template_not_matched",
                observed_at,
            );
            return Ok(());
        };
        let mut x = f64::from(roi_width - self.config.search_size) * 0.5
            + f64::from(self.config.search_size) * 0.5
            + f64::from(found.offset_x);
        let mut y = f64::from(roi_height - self.config.search_size) * 0.5
            + f64::from(self.config.search_size) * 0.5
            + f64::from(found.offset_y);
        let candidate = (x, y);
        if state
            .candidate_xy
            .is_none_or(|prior| distance(prior, candidate) > 2.0)
        {
            state.candidate_xy = Some(candidate);
            state.candidate_since = Some(observed_at);
        }
        let confirmed = state
            .candidate_since
            .is_some_and(|since| observed_at.duration_since(since) >= self.config.confirm_duration);
        if confirmed && let Some(previous) = &state.last_confirmed {
            (x, y) = limited_point((previous.x, previous.y), (x, y), self.config.max_step_px);
        }
        let observation = CrosshairObservation {
            status: if confirmed { "confirmed" } else { "candidate" }.to_owned(),
            found: true,
            x,
            y,
            confidence: found.confidence,
            sample_ts_ns: frame.sample_ts_ns,
            template_id: template.id,
            point_hits: found.hits,
            point_count: template.points.len(),
            offset_x: x - f64::from(roi_width) * 0.5,
            offset_y: y - f64::from(roi_height) * 0.5,
            reason: String::new(),
        };
        state.matched_frames += 1;
        if confirmed {
            state.last_confirmed = Some(observation.clone());
            state.last_confirmed_at = Some(observed_at);
        }
        state.latest = Some(observation);
        state.latest_at = Some(observed_at);
        Ok(())
    }

    pub fn learn(&self) -> Result<CrosshairTemplateSummary, CrosshairError> {
        let _mutation = lock(&self.mutation);
        let required = self.config.sample_frames.max(3);
        let (images, roi_width, roi_height) = {
            let state = lock(&self.state);
            if state.recent.len() < required {
                return Err(CrosshairError::InsufficientSamples {
                    required,
                    actual: state.recent.len(),
                });
            }
            let samples: Vec<_> = state.recent.iter().rev().take(required).collect();
            let newest = samples[0];
            if newest.observed_at.elapsed() > self.config.max_age {
                return Err(CrosshairError::StaleSamples);
            }
            if samples.iter().any(|sample| {
                sample.roi_width != newest.roi_width || sample.roi_height != newest.roi_height
            }) {
                return Err(CrosshairError::GeometryChanged);
            }
            (
                samples
                    .iter()
                    .map(|sample| sample.image.clone())
                    .collect::<Vec<_>>(),
                newest.roi_width,
                newest.roi_height,
            )
        };
        let median = median_image(&images);
        let points = learn_points(&median);
        if points.len() < 8 {
            return Err(CrosshairError::NotUnique);
        }
        let template = Template {
            schema_version: 1,
            id: Uuid::new_v4().simple().to_string()[..12].to_owned(),
            created_ts_ns: wall_time_ns(),
            geometry_signature: geometry_signature(roi_width, roi_height),
            search_size: self.config.search_size,
            points,
        };
        let validation = match_template(
            &median,
            &template,
            self.config.max_offset_px.min(20.0).round() as i32,
            None,
        )
        .filter(|item| {
            item.offset_x.abs() <= 1 && item.offset_y.abs() <= 1 && item.uniqueness >= 0.015
        })
        .ok_or(CrosshairError::NotUnique)?;
        let _ = validation;
        save_template(&self.template_path, &template)?;
        let summary = summarize(&template);
        let mut state = lock(&self.state);
        state.template = Some(template);
        clear_tracking(&mut state);
        Ok(summary)
    }

    pub fn clear_template(&self) -> Result<CrosshairSnapshot, CrosshairError> {
        let _mutation = lock(&self.mutation);
        {
            let mut state = lock(&self.state);
            state.template = None;
            clear_runtime(&mut state);
        }
        match fs::remove_file(self.template_path.as_path()) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(CrosshairError::Io(error)),
        }
        Ok(self.snapshot())
    }

    pub fn resolve(
        &self,
        geometric_x: f64,
        geometric_y: f64,
        width: u32,
        height: u32,
    ) -> ControlReference {
        let state = lock(&self.state);
        let geometry = geometry_signature(width, height);
        let reason = fallback_reason(&state, &self.config, &geometry);
        let Some(observation) = state.last_confirmed.as_ref().filter(|_| reason.is_empty()) else {
            return ControlReference {
                x: geometric_x,
                y: geometric_y,
                source: "geometry".to_owned(),
                confidence: 1.0,
                age_ms: 0.0,
                geometry_signature: geometry,
                reason,
            };
        };
        ControlReference {
            x: observation.x,
            y: observation.y,
            source: if state
                .latest
                .as_ref()
                .is_some_and(|item| item.status == "confirmed")
            {
                "vision_verified"
            } else {
                "vision_hold"
            }
            .to_owned(),
            confidence: observation.confidence,
            age_ms: state
                .last_confirmed_at
                .map_or(0.0, |at| at.elapsed().as_secs_f64() * 1000.0),
            geometry_signature: geometry,
            reason: String::new(),
        }
    }

    pub fn snapshot(&self) -> CrosshairSnapshot {
        self.snapshot_locked(&lock(&self.state))
    }

    fn snapshot_locked(&self, state: &State) -> CrosshairSnapshot {
        let geometry = state
            .recent
            .back()
            .map(|sample| geometry_signature(sample.roi_width, sample.roi_height))
            .or_else(|| {
                state
                    .template
                    .as_ref()
                    .map(|item| item.geometry_signature.clone())
            })
            .unwrap_or_default();
        let reason = fallback_reason(state, &self.config, &geometry);
        CrosshairSnapshot {
            enabled: self.config.enabled,
            use_for_control: self.config.use_for_control,
            running: state.running,
            state: state
                .latest
                .as_ref()
                .map_or("idle", |item| item.status.as_str())
                .to_owned(),
            observation: state.latest.clone(),
            control_reference_ready: reason.is_empty(),
            control_reference_source: if reason.is_empty() {
                if state
                    .latest
                    .as_ref()
                    .is_some_and(|item| item.status == "confirmed")
                {
                    "vision_verified"
                } else {
                    "vision_hold"
                }
            } else {
                "geometry"
            }
            .to_owned(),
            control_reference_reason: reason,
            recent_samples: state.recent.len(),
            required_samples: self.config.sample_frames.max(3),
            processed_frames: state.processed_frames,
            matched_frames: state.matched_frames,
            dropped_frames: state.dropped_frames,
            last_error: state.last_error.clone(),
            template: state.template.as_ref().map(summarize),
        }
    }

    pub fn template_preview_png(&self) -> Result<Option<Vec<u8>>, CrosshairError> {
        let template = lock(&self.state).template.clone();
        let Some(template) = template else {
            return Ok(None);
        };
        let mut image = RgbaImage::from_pixel(
            template.search_size,
            template.search_size,
            Rgba([15, 18, 25, 255]),
        );
        let center = template.search_size as i32 / 2;
        for TemplatePoint(dx, dy, ..) in &template.points {
            for oy in -1..=1 {
                for ox in -1..=1 {
                    let x = center + dx + ox;
                    let y = center + dy + oy;
                    if x >= 0
                        && y >= 0
                        && x < template.search_size as i32
                        && y < template.search_size as i32
                    {
                        image.put_pixel(x as u32, y as u32, Rgba([255, 67, 113, 255]));
                    }
                }
            }
        }
        for delta in -4..=4 {
            image.put_pixel(
                (center + delta) as u32,
                center as u32,
                Rgba([255, 255, 255, 180]),
            );
            image.put_pixel(
                center as u32,
                (center + delta) as u32,
                Rgba([255, 255, 255, 180]),
            );
        }
        let mut output = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut output, ImageFormat::Png)
            .map_err(CrosshairError::Encode)?;
        Ok(Some(output.into_inner()))
    }
}

#[derive(Debug)]
struct Frame {
    sample_ts_ns: u64,
    jpeg: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CrosshairFramePublisher {
    epoch: RuntimeEpoch,
    tx: SyncSender<Frame>,
    state: Arc<Mutex<State>>,
}

impl CrosshairFramePublisher {
    pub fn publish_jpeg(&self, sample_ts_ns: u64, jpeg: Vec<u8>) -> Result<(), CrosshairError> {
        if jpeg.len() < 4 || !jpeg.starts_with(&[0xff, 0xd8]) || !jpeg.ends_with(&[0xff, 0xd9]) {
            return Err(CrosshairError::InvalidJpeg);
        }
        match self.tx.try_send(Frame { sample_ts_ns, jpeg }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                lock(&self.state).dropped_frames += 1;
                Err(CrosshairError::Busy)
            }
            Err(TrySendError::Disconnected(_)) => Err(CrosshairError::StaleEpoch(self.epoch.0)),
        }
    }
}

#[derive(Debug)]
pub struct CrosshairEpoch {
    publisher: CrosshairFramePublisher,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    hub: CrosshairHub,
}

impl CrosshairEpoch {
    pub fn publisher(&self) -> CrosshairFramePublisher {
        self.publisher.clone()
    }
}

impl Drop for CrosshairEpoch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        let mut state = lock(&self.hub.state);
        if state.epoch == self.publisher.epoch.0 {
            state.running = false;
            clear_runtime(&mut state);
        }
    }
}

#[derive(Debug, Error)]
pub enum CrosshairError {
    #[error("crosshair observation is disabled")]
    Disabled,
    #[error("crosshair JPEG payload is incomplete")]
    InvalidJpeg,
    #[error("crosshair observer is busy; frame dropped")]
    Busy,
    #[error("crosshair observer epoch {0} is no longer active")]
    StaleEpoch(u64),
    #[error("crosshair JPEG decode failed: {0}")]
    Decode(image::ImageError),
    #[error("crosshair PNG encode failed: {0}")]
    Encode(image::ImageError),
    #[error("crosshair image must be {expected}x{expected}, got {width}x{height}")]
    UnexpectedGeometry {
        expected: u32,
        width: u32,
        height: u32,
    },
    #[error("crosshair learning requires {required} recent samples; only {actual} available")]
    InsufficientSamples { required: usize, actual: usize },
    #[error("crosshair learning samples are stale; wait for live frames")]
    StaleSamples,
    #[error("crosshair learning samples changed ROI geometry")]
    GeometryChanged,
    #[error("crosshair learning could not isolate a unique stable center structure")]
    NotUnique,
    #[error("crosshair observer worker could not start: {0}")]
    WorkerSpawn(std::io::Error),
    #[error("crosshair template I/O failed: {0}")]
    Io(std::io::Error),
    #[error("crosshair template is invalid: {0}")]
    Template(String),
}

#[derive(Clone, Copy)]
struct Match {
    offset_x: i32,
    offset_y: i32,
    confidence: f64,
    hits: usize,
    uniqueness: f64,
}

fn match_template(
    image: &RgbImage,
    template: &Template,
    max_offset: i32,
    hint: Option<(i32, i32)>,
) -> Option<Match> {
    let offsets: Vec<_> = if let Some((hx, hy)) = hint {
        ((hy - 3).max(-max_offset)..=(hy + 3).min(max_offset))
            .flat_map(|y| {
                ((hx - 3).max(-max_offset)..=(hx + 3).min(max_offset)).map(move |x| (x, y))
            })
            .collect()
    } else {
        let step = if max_offset >= 8 { 4 } else { 2 };
        let mut axis: Vec<_> = (-max_offset..=max_offset).step_by(step as usize).collect();
        if axis.last().copied() != Some(max_offset) {
            axis.push(max_offset);
        }
        axis.iter()
            .flat_map(|y| axis.iter().map(move |x| (*x, *y)))
            .collect()
    };
    let mut candidates = score_offsets(image, template, &offsets);
    let coarse = finalize(&candidates, max_offset)?;
    if hint.is_none() {
        let step = if max_offset >= 8 { 4 } else { 2 };
        let fine: Vec<_> = ((coarse.offset_y - step).max(-max_offset)
            ..=(coarse.offset_y + step).min(max_offset))
            .flat_map(|y| {
                ((coarse.offset_x - step).max(-max_offset)
                    ..=(coarse.offset_x + step).min(max_offset))
                    .map(move |x| (x, y))
            })
            .collect();
        candidates.extend(score_offsets(image, template, &fine));
    }
    finalize(&candidates, max_offset)
}

fn score_offsets(image: &RgbImage, template: &Template, offsets: &[(i32, i32)]) -> Vec<Match> {
    let center = image.width() as i32 / 2;
    offsets
        .iter()
        .filter_map(|&(ox, oy)| {
            let mut total = 0.0;
            let mut hits = 0;
            let mut compared = 0;
            for &TemplatePoint(dx, dy, th, ts, tv, tg) in &template.points {
                let x = center + dx + ox;
                let y = center + dy + oy;
                if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
                    continue;
                }
                compared += 1;
                let [r, g, b] = image.get_pixel(x as u32, y as u32).0;
                let (h, s, v) = rgb_to_hsv(r, g, b);
                let gray = gray(r, g, b);
                let hue_difference = u16::from(u8::abs_diff(h, th));
                let hue = hue_difference.min(256 - hue_difference) as u8;
                let gray_score = (1.0 - f64::from(u8::abs_diff(gray, tg)) / 80.0).max(0.0);
                let color_score = (1.0 - f64::from(hue) / 32.0).max(0.0) * 0.45
                    + (1.0 - f64::from(u8::abs_diff(s, ts)) / 110.0).max(0.0) * 0.20
                    + (1.0 - f64::from(u8::abs_diff(v, tv)) / 100.0).max(0.0) * 0.15
                    + gray_score * 0.20;
                let score = if ts >= 50 { color_score } else { gray_score };
                total += score;
                if score >= 0.60 {
                    hits += 1;
                }
            }
            (compared >= 4.max(template.points.len() / 2)).then_some(Match {
                offset_x: ox,
                offset_y: oy,
                confidence: total / compared as f64,
                hits,
                uniqueness: 0.0,
            })
        })
        .collect()
}

fn finalize(candidates: &[Match], max_offset: i32) -> Option<Match> {
    let mut best = *candidates.iter().max_by(|a, b| {
        let adjusted = |item: &Match| {
            item.confidence
                - 0.002
                    * f64::from(item.offset_x * item.offset_x + item.offset_y * item.offset_y)
                        .sqrt()
                    / f64::from(max_offset.max(1))
        };
        adjusted(a).total_cmp(&adjusted(b))
    })?;
    let competing = candidates
        .iter()
        .filter(|item| {
            distance(
                (f64::from(item.offset_x), f64::from(item.offset_y)),
                (f64::from(best.offset_x), f64::from(best.offset_y)),
            ) >= 3.0
        })
        .map(|item| item.confidence)
        .fold(0.0, f64::max);
    best.uniqueness = (best.confidence - competing).max(0.0);
    Some(best)
}

fn median_image(images: &[RgbImage]) -> RgbImage {
    let width = images[0].width();
    let height = images[0].height();
    RgbImage::from_fn(width, height, |x, y| {
        let mut channels = [
            Vec::with_capacity(images.len()),
            Vec::with_capacity(images.len()),
            Vec::with_capacity(images.len()),
        ];
        for image in images {
            let pixel = image.get_pixel(x, y).0;
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
        for values in &mut channels {
            values.sort_unstable();
        }
        let median = |values: &[u8]| {
            let upper = values.len() / 2;
            if values.len().is_multiple_of(2) {
                ((u16::from(values[upper - 1]) + u16::from(values[upper])) / 2) as u8
            } else {
                values[upper]
            }
        };
        Rgb([
            median(&channels[0]),
            median(&channels[1]),
            median(&channels[2]),
        ])
    })
}

fn learn_points(image: &RgbImage) -> Vec<TemplatePoint> {
    let center = image.width() as i32 / 2;
    let radius = 24.min(center / 3 * 2);
    let mut candidates = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let [r, g, b] = image
                .get_pixel((center + dx) as u32, (center + dy) as u32)
                .0;
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let g0 = gray(r, g, b);
            let mut neighbors = [(-3, 0), (3, 0), (0, -3), (0, 3)].map(|(ox, oy)| {
                let p = image
                    .get_pixel(
                        (center + dx + ox).clamp(0, image.width() as i32 - 1) as u32,
                        (center + dy + oy).clamp(0, image.height() as i32 - 1) as u32,
                    )
                    .0;
                gray(p[0], p[1], p[2])
            });
            neighbors.sort_unstable();
            let contrast = u8::abs_diff(g0, neighbors[2]);
            if contrast < 18 && !(s >= 70 && v >= 70) {
                continue;
            }
            let centrality =
                (1.0 - f64::from(dx * dx + dy * dy).sqrt() / f64::from(radius.max(1))).max(0.0);
            candidates.push((
                f64::from(contrast) + f64::from(s) * 0.35 + centrality * 12.0,
                TemplatePoint(dx, dy, h, s, v, g0),
            ));
        }
    }
    let coords: Vec<_> = candidates
        .iter()
        .map(|(_, TemplatePoint(x, y, ..))| (*x, *y))
        .collect();
    candidates.retain(|(_, TemplatePoint(x, y, ..))| {
        coords
            .iter()
            .any(|(cx, cy)| (cx + x).abs() <= 1 && (cy + y).abs() <= 1)
    });
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.into_iter().take(96).map(|(_, p)| p).collect()
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = f64::from(r) / 255.0;
    let gf = f64::from(g) / 255.0;
    let bf = f64::from(b) / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == rf {
        ((gf - bf) / d).rem_euclid(6.0)
    } else if max == gf {
        (bf - rf) / d + 2.0
    } else {
        (rf - gf) / d + 4.0
    };
    (
        ((h / 6.0 * 255.0).round() as i32).rem_euclid(256) as u8,
        if max == 0.0 {
            0
        } else {
            (d / max * 255.0).round() as u8
        },
        (max * 255.0).round() as u8,
    )
}
fn gray(r: u8, g: u8, b: u8) -> u8 {
    (f64::from(r) * 0.299 + f64::from(g) * 0.587 + f64::from(b) * 0.114).round() as u8
}
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}
fn limited_point(previous: (f64, f64), requested: (f64, f64), max_step: f64) -> (f64, f64) {
    let d = distance(previous, requested);
    if d <= max_step || d <= 0.0 {
        return requested;
    }
    let scale = max_step / d;
    (
        previous.0 + (requested.0 - previous.0) * scale,
        previous.1 + (requested.1 - previous.1) * scale,
    )
}
fn geometry_signature(w: u32, h: u32) -> String {
    format!("{w}x{h}")
}
fn wall_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}
fn summarize(t: &Template) -> CrosshairTemplateSummary {
    CrosshairTemplateSummary {
        id: t.id.clone(),
        schema_version: t.schema_version,
        created_ts_ns: t.created_ts_ns,
        geometry_signature: t.geometry_signature.clone(),
        search_size: t.search_size,
        point_count: t.points.len(),
    }
}
fn fallback_reason(state: &State, config: &CrosshairConfig, geometry: &str) -> String {
    if !config.enabled {
        "disabled"
    } else if !config.use_for_control {
        "control_disabled"
    } else if state.template.is_none() {
        "template_unavailable"
    } else if state
        .template
        .as_ref()
        .is_some_and(|t| t.geometry_signature != geometry)
    {
        "geometry_changed"
    } else if state.last_confirmed.is_none() {
        "observation_unconfirmed"
    } else if state
        .last_confirmed_at
        .is_none_or(|at| at.elapsed() > config.max_age)
    {
        "observation_stale"
    } else {
        ""
    }
    .to_owned()
}
fn clear_tracking(state: &mut State) {
    state.latest = None;
    state.latest_at = None;
    state.last_confirmed = None;
    state.last_confirmed_at = None;
    state.candidate_xy = None;
    state.candidate_since = None;
}
fn clear_runtime(state: &mut State) {
    state.recent.clear();
    clear_tracking(state);
    state.last_error.clear();
}
fn set_empty_observation(state: &mut State, status: &str, ts: u64, reason: &str, at: Instant) {
    state.latest = Some(CrosshairObservation {
        status: status.to_owned(),
        found: false,
        x: 0.0,
        y: 0.0,
        confidence: 0.0,
        sample_ts_ns: ts,
        template_id: String::new(),
        point_hits: 0,
        point_count: 0,
        offset_x: 0.0,
        offset_y: 0.0,
        reason: reason.to_owned(),
    });
    state.latest_at = Some(at);
}
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
fn load_template(path: &Path) -> Result<Option<Template>, CrosshairError> {
    match fs::read(path) {
        Ok(bytes) => {
            let template: Template = serde_json::from_slice(&bytes)
                .map_err(|e| CrosshairError::Template(e.to_string()))?;
            validate_template(&template)?;
            Ok(Some(template))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CrosshairError::Io(e)),
    }
}
fn save_template(path: &Path, template: &Template) -> Result<(), CrosshairError> {
    validate_template(template)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(CrosshairError::Io)?;
    }
    let tmp = path.with_extension("tmp");
    let bytes =
        serde_json::to_vec(template).map_err(|e| CrosshairError::Template(e.to_string()))?;
    fs::write(&tmp, bytes).map_err(CrosshairError::Io)?;
    fs::rename(tmp, path).map_err(CrosshairError::Io)
}
fn validate_template(t: &Template) -> Result<(), CrosshairError> {
    let valid_geometry = t
        .geometry_signature
        .split_once('x')
        .and_then(|(width, height)| Some((width.parse::<u32>().ok()?, height.parse::<u32>().ok()?)))
        .is_some_and(|(width, height)| width > 0 && height > 0);
    if t.schema_version != 1
        || t.id.is_empty()
        || t.id.len() > 64
        || !(32..=640).contains(&t.search_size)
        || t.points.len() < 8
        || t.points.len() > 256
        || !valid_geometry
    {
        return Err(CrosshairError::Template(
            "unsupported schema or invalid fields".to_owned(),
        ));
    }
    let radius = t.search_size as i32 / 2;
    if t.points
        .iter()
        .any(|p| p.0.abs() >= radius || p.1.abs() >= radius)
    {
        return Err(CrosshairError::Template(
            "point outside search area".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;

    fn config() -> CrosshairConfig {
        CrosshairConfig {
            enabled: true,
            use_for_control: true,
            search_size: 96,
            sample_frames: 3,
            confirm_duration: Duration::ZERO,
            max_age: Duration::from_secs(1),
            max_offset_px: 20.0,
            min_similarity: 0.60,
            max_step_px: 2.0,
        }
    }
    fn jpeg(offset: i32) -> Vec<u8> {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([20, 20, 20]));
        let c = 48 + offset;
        for d in -12..=12 {
            image.put_pixel((c + d) as u32, 48, Rgb([255, 30, 60]));
            image.put_pixel(c as u32, (48 + d) as u32, Rgb([255, 30, 60]));
        }
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, 95)
            .encode_image(&image)
            .unwrap();
        out
    }
    #[test]
    fn learns_persists_matches_and_resolves_control_reference() {
        let dir = std::env::temp_dir().join(format!("novasight-crosshair-{}", Uuid::new_v4()));
        let path = dir.join("template.json");
        let hub = CrosshairHub::new(config(), &path);
        let epoch = hub.begin_epoch(RuntimeEpoch(1), 640, 640).unwrap();
        for n in 1..=3 {
            loop {
                match epoch.publisher().publish_jpeg(n, jpeg(0)) {
                    Ok(()) => break,
                    Err(CrosshairError::Busy) => thread::sleep(Duration::from_millis(5)),
                    Err(e) => panic!("{e}"),
                }
            }
        }
        for _ in 0..100 {
            if hub.snapshot().recent_samples == 3 {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let learned = hub.learn().unwrap();
        assert!(learned.point_count >= 8);
        epoch.publisher().publish_jpeg(4, jpeg(4)).unwrap();
        for _ in 0..100 {
            if hub.snapshot().state == "confirmed" {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let reference = hub.resolve(320.0, 320.0, 640, 640);
        assert_eq!(reference.source, "vision_verified");
        assert!((reference.x - 324.0).abs() <= 1.0);
        assert!(
            hub.template_preview_png()
                .unwrap()
                .unwrap()
                .starts_with(&[137, 80, 78, 71])
        );
        drop(epoch);
        let reloaded = CrosshairHub::new(config(), &path);
        assert_eq!(reloaded.snapshot().template.unwrap().id, learned.id);
        let _ = fs::remove_dir_all(dir);
    }
}
