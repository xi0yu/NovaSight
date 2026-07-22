//! Phase 2 deterministic tracker and targeting core.
//!
//! The runtime replaces the Python ``RuntimeTracker`` and
//! ``RuntimeTargetSelector`` modules. It consumes admitted detector
//! candidates (including frame-local DeepStream identities), owns temporal
//! association, and returns both the chosen candidate and a stable `TrackId`.
//!
//! The implementation is intentionally small and exhaustive:
//! * no Kalman filter or Hungarian assignment. Bounded deterministic
//!   association retains every admitted candidate needed for switch
//!   hysteresis; temporal identity confidence combines target-height-
//!   normalized center distance and bounding-box IoU.
//! * no unbounded growth. History is bounded by `BoundedHistory` and
//!   `TargetingCore::reset` is the only way to clear it.
//! * lost tracks never produce a control target. After a configurable
//!   number of missed frames the locked track is dropped and the
//!   selection returns `target_object_id = None` until a fresh
//!   candidate re-acquires the lock.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::perception::types::Detection;

/// Maximum number of historical tracking observations retained per
/// track. Past this bound, the oldest entry is dropped.
pub const DEFAULT_HISTORY_LIMIT: usize = 32;

/// Maximum age (in frames) before a non-matched track is marked lost.
pub const DEFAULT_TRACK_MAX_AGE: u64 = 5;

/// Hard admission bound shared with `DetectionBatch`, preventing the
/// association surface from diverging from the perception boundary.
pub const MAX_TRACK_CANDIDATES: usize = crate::perception::types::MAX_DETECTIONS;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LockReason {
    /// First lock acquisition on a fresh frame, preferred class wins.
    PreferredClass,
    /// Previous head (preferred class) left the merit order; the next
    /// best candidate within tolerance takes over.
    FallbackClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrackId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: TrackId,
    pub object_id: u64,
    pub class_id: u32,
    pub state: TrackState,
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    /// Latest detector score, kept separate from temporal identity quality.
    pub confidence: f32,
    /// Association continuity in 0..=1, where one is an exact spatial match.
    pub identity_confidence: f64,
    pub age_frames: u64,
    pub missed_frames: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackerConfig {
    pub history_limit: usize,
    pub track_max_age: u64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            history_limit: DEFAULT_HISTORY_LIMIT,
            track_max_age: DEFAULT_TRACK_MAX_AGE,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundedHistory<T> {
    limit: usize,
    entries: Vec<T>,
}

impl<T> BoundedHistory<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, value: T) {
        if self.entries.len() >= self.limit {
            self.entries.remove(0);
        }
        self.entries.push(value);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.entries.iter()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Association {
    pub track_id: TrackId,
    pub object_id: u64,
    pub center_x: f64,
    pub center_y: f64,
    pub confidence: f32,
}

/// Deterministic nearest-centroid association. Equal distance is broken
/// by ascending `object_id`. Detection objects with duplicate ids are
/// rejected up front.
pub fn associate(tracks: &[Track], detections: &[Detection]) -> Result<Vec<Association>, AppError> {
    if detections.len() > MAX_TRACK_CANDIDATES {
        return Err(AppError::TooManyDetections {
            actual: detections.len(),
            maximum: MAX_TRACK_CANDIDATES,
        });
    }
    let mut sorted_detections = detections.to_vec();
    sorted_detections.sort_by_key(|det| (det.object_id(), det.class_id()));
    for window in sorted_detections.windows(2) {
        if window[0].object_id() == window[1].object_id() {
            return Err(AppError::DuplicateObjectId {
                object_id: window[0].object_id(),
            });
        }
    }
    let mut sorted_tracks = tracks.to_vec();
    sorted_tracks.sort_by_key(|track| track.id.0);

    let mut used = vec![false; sorted_detections.len()];
    let mut out = Vec::with_capacity(sorted_tracks.len().min(sorted_detections.len()));
    for track in &sorted_tracks {
        if track.state == TrackState::Lost {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for (index, detection) in sorted_detections.iter().enumerate() {
            if used[index] {
                continue;
            }
            if detection.class_id() != track.class_id {
                continue;
            }
            let distance = euclidean(
                track.center_x,
                track.center_y,
                detection.center_x(),
                detection.center_y(),
            );
            if !distance.is_finite() {
                continue;
            }
            match best {
                Some((_, current)) if current <= distance => {}
                _ => best = Some((index, distance)),
            }
        }
        if let Some((index, _)) = best {
            used[index] = true;
            let detection = &sorted_detections[index];
            out.push(Association {
                track_id: track.id,
                object_id: detection.object_id(),
                center_x: detection.center_x(),
                center_y: detection.center_y(),
                confidence: detection.confidence(),
            });
        }
    }
    Ok(out)
}

fn euclidean(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetSelection {
    /// Frame-local candidate identity used to find the chosen bounding box.
    pub target_object_id: Option<u64>,
    /// Stable identity allocated and associated by the Rust targeting core.
    pub target_track_id: Option<TrackId>,
    pub target_class_id: Option<u32>,
    pub target_detection_confidence: Option<f32>,
    pub target_identity_confidence: Option<f64>,
    /// Control aim point. Association continues to use the geometric center.
    pub target_aim_x: Option<f64>,
    pub target_aim_y: Option<f64>,
    pub lock_reason: Option<LockReason>,
    pub candidates: usize,
    pub inside_fov: usize,
    pub lost_count: u64,
}

impl TargetSelection {
    pub const fn empty() -> Self {
        Self {
            target_object_id: None,
            target_track_id: None,
            target_class_id: None,
            target_detection_confidence: None,
            target_identity_confidence: None,
            target_aim_x: None,
            target_aim_y: None,
            lock_reason: None,
            candidates: 0,
            inside_fov: 0,
            lost_count: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetingConfig {
    /// Maximum frame-to-frame jump in pixels for a head candidate to
    /// keep the previous lock. Past this window the lock falls back
    /// to the next best class-1 candidate.
    pub debounce_distance_px: f64,
    /// Confidence threshold for admitting a detection into targeting.
    pub min_confidence: f32,
    /// Number of consecutive missed frames before a track is dropped.
    pub track_max_age: u64,
    /// Radial admission gate around the declared observation center.
    pub target_fov_radius_px: f64,
    /// Maximum center displacement measured in target-height units.
    pub tracker_max_match_distance: f64,
    pub tracker_position_cost_weight: f64,
    pub tracker_iou_cost_weight: f64,
    /// Descending class preference. The first two ranks receive the same
    /// 1.0 / 0.5 scores as the Python selector; unlisted classes score zero.
    pub class_priority: Vec<u32>,
    pub selection_class_weight: f64,
    pub selection_distance_weight: f64,
    /// Distance discount applied only to the currently locked TrackId.
    pub sticky_bias: f64,
    pub switch_min_preference_advantage: f64,
    pub switch_min_continuity_score: f64,
    pub switch_delay_ms: f64,
    pub aim_y_ratio: f64,
    pub class_aim_y_ratios: BTreeMap<u32, f64>,
    pub candidate_max_aspect_ratio: f64,
}

impl Default for TargetingConfig {
    fn default() -> Self {
        Self {
            debounce_distance_px: 64.0,
            min_confidence: 0.5,
            track_max_age: DEFAULT_TRACK_MAX_AGE,
            target_fov_radius_px: 180.0,
            tracker_max_match_distance: 1.5,
            tracker_position_cost_weight: 0.75,
            tracker_iou_cost_weight: 0.25,
            class_priority: vec![0, 1],
            selection_class_weight: 0.55,
            selection_distance_weight: 0.40,
            sticky_bias: 0.25,
            switch_min_preference_advantage: 0.08,
            switch_min_continuity_score: 0.70,
            switch_delay_ms: 50.0,
            aim_y_ratio: 0.22,
            class_aim_y_ratios: BTreeMap::new(),
            candidate_max_aspect_ratio: 6.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingSwitch {
    track_id: TrackId,
    started_at_ns: u64,
}

/// Locked target state machine. A new instance starts empty; callers
/// feed it consecutive `select` calls. The state machine owns its
/// history so callers cannot reach into it from the outside.
#[derive(Debug)]
pub struct TargetingCore {
    config: TargetingConfig,
    history: BoundedHistory<Track>,
    tracks: Vec<Track>,
    locked: Option<Track>,
    pending_switch: Option<PendingSwitch>,
    lost_count: u64,
    next_track_id: u64,
}

impl TargetingCore {
    pub fn new(config: TargetingConfig) -> Self {
        Self {
            config,
            history: BoundedHistory::new(DEFAULT_HISTORY_LIMIT),
            tracks: Vec::new(),
            locked: None,
            pending_switch: None,
            lost_count: 0,
            next_track_id: 1,
        }
    }

    pub fn reset(&mut self) {
        self.history = BoundedHistory::new(self.history.limit);
        self.tracks.clear();
        self.locked = None;
        self.pending_switch = None;
        self.lost_count = 0;
        self.next_track_id = 1;
    }

    pub fn history_iter<'a>(&'a self) -> impl Iterator<Item = &'a Track> + 'a {
        self.history.iter()
    }

    pub fn lost_count(&self) -> u64 {
        self.lost_count
    }

    pub fn locked(&self) -> Option<&Track> {
        self.locked.as_ref()
    }

    /// Deterministic compatibility entry point for tests and replay fixtures.
    /// Production callers use [`Self::select_at`] so switch hysteresis is
    /// measured from the admitted capture timestamp.
    pub fn select(
        &mut self,
        detections: &[Detection],
        observation_center: (f64, f64),
    ) -> TargetSelection {
        self.select_at(detections, observation_center, 0)
    }

    /// Production selection path. `captured_at_ns` is the admitted monotonic
    /// capture timestamp and therefore cannot be stretched by worker backlog.
    pub fn select_at(
        &mut self,
        detections: &[Detection],
        observation_center: (f64, f64),
        captured_at_ns: u64,
    ) -> TargetSelection {
        let candidates = detections.len();
        let admissible: Vec<Detection> = detections
            .iter()
            .filter(|det| {
                det.confidence() >= self.config.min_confidence
                    && self.config.class_priority.contains(&det.class_id())
                    && detection_aspect_ratio(det) <= self.config.candidate_max_aspect_ratio
                    && euclidean(
                        det.center_x(),
                        det.center_y(),
                        observation_center.0,
                        observation_center.1,
                    ) <= self.config.target_fov_radius_px
            })
            .cloned()
            .collect();
        if admissible.is_empty() {
            self.pending_switch = None;
            self.miss_locked_target();
            return TargetSelection {
                candidates,
                inside_fov: 0,
                target_object_id: None,
                target_track_id: None,
                target_class_id: None,
                target_detection_confidence: None,
                target_identity_confidence: None,
                target_aim_x: None,
                target_aim_y: None,
                lock_reason: None,
                lost_count: self.lost_count,
            };
        }

        let associations = associate(&self.tracks, &admissible).unwrap_or_default();
        let mut current = Vec::with_capacity(admissible.len());
        for det in &admissible {
            let associated = associations.iter().find_map(|association| {
                let prior = self
                    .tracks
                    .iter()
                    .find(|track| track.id == association.track_id)?;
                if association.object_id != det.object_id() {
                    return None;
                }
                association_identity_confidence(prior, det, &self.config)
                    .map(|identity_confidence| (prior, identity_confidence))
            });
            let id = associated.map_or_else(
                || {
                    let id = TrackId(self.next_track_id);
                    self.next_track_id = self.next_track_id.saturating_add(1);
                    id
                },
                |(track, _)| track.id,
            );
            current.push(Track {
                id,
                object_id: det.object_id(),
                class_id: det.class_id(),
                state: TrackState::Confirmed,
                center_x: det.center_x(),
                center_y: det.center_y(),
                width: f64::from(det.width()),
                height: f64::from(det.height()),
                confidence: det.confidence(),
                identity_confidence: associated.map_or(1.0, |(_, confidence)| confidence),
                age_frames: associated.map_or(1, |(track, _)| track.age_frames.saturating_add(1)),
                missed_frames: 0,
            });
        }

        let locked_index = self
            .locked
            .as_ref()
            .and_then(|locked| current.iter().position(|track| track.id == locked.id));
        let best_index = current
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                target_score(left, self.locked.as_ref(), observation_center, &self.config)
                    .total_cmp(&target_score(
                        right,
                        self.locked.as_ref(),
                        observation_center,
                        &self.config,
                    ))
                    .then_with(|| right.object_id.cmp(&left.object_id))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("admissible detections produce current tracks");

        let chosen_index = match locked_index {
            None => {
                self.pending_switch = None;
                best_index
            }
            Some(index) if index == best_index => {
                self.pending_switch = None;
                best_index
            }
            Some(index) => {
                let locked_score = target_score(
                    &current[index],
                    self.locked.as_ref(),
                    observation_center,
                    &self.config,
                );
                let best = &current[best_index];
                let advantage =
                    target_score(best, self.locked.as_ref(), observation_center, &self.config)
                        - locked_score;
                let continuity = if best.age_frames > 1 {
                    best.identity_confidence
                } else {
                    0.0
                };
                if advantage < self.config.switch_min_preference_advantage
                    || continuity < self.config.switch_min_continuity_score
                {
                    self.pending_switch = None;
                    index
                } else {
                    let started_at_ns = self
                        .pending_switch
                        .filter(|pending| pending.track_id == best.id)
                        .map_or(captured_at_ns, |pending| pending.started_at_ns);
                    let elapsed_ms = captured_at_ns.saturating_sub(started_at_ns) as f64 / 1e6;
                    if elapsed_ms >= self.config.switch_delay_ms {
                        self.pending_switch = None;
                        best_index
                    } else {
                        self.pending_switch = Some(PendingSwitch {
                            track_id: best.id,
                            started_at_ns,
                        });
                        index
                    }
                }
            }
        };
        let track = current[chosen_index].clone();
        let reason = if self.config.class_priority.first().copied() == Some(track.class_id) {
            LockReason::PreferredClass
        } else {
            LockReason::FallbackClass
        };
        let selected_detection = admissible
            .iter()
            .find(|detection| detection.object_id() == track.object_id)
            .expect("selected track belongs to the admitted batch");
        let aim_y_ratio = self
            .config
            .class_aim_y_ratios
            .get(&track.class_id)
            .copied()
            .unwrap_or(self.config.aim_y_ratio);
        let aim_x = selected_detection.center_x();
        let aim_y = f64::from(selected_detection.y())
            + f64::from(selected_detection.height()) * aim_y_ratio;

        self.history.push(track.clone());
        self.tracks = current;
        self.lost_count = 0;
        self.locked = Some(track.clone());
        TargetSelection {
            candidates,
            inside_fov: admissible.len(),
            target_object_id: Some(track.object_id),
            target_track_id: Some(track.id),
            target_class_id: Some(track.class_id),
            target_detection_confidence: Some(track.confidence),
            target_identity_confidence: Some(track.identity_confidence),
            target_aim_x: Some(aim_x),
            target_aim_y: Some(aim_y),
            lock_reason: Some(reason),
            lost_count: self.lost_count,
        }
    }

    fn miss_locked_target(&mut self) {
        self.lost_count = self.lost_count.saturating_add(1);
        for track in &mut self.tracks {
            track.missed_frames = track.missed_frames.saturating_add(1);
            if track.missed_frames > self.config.track_max_age {
                track.state = TrackState::Lost;
            }
        }
        self.tracks.retain(|track| track.state != TrackState::Lost);
        self.locked = None;
    }
}

fn target_score(
    track: &Track,
    locked: Option<&Track>,
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> f64 {
    let class_score = match config
        .class_priority
        .iter()
        .position(|class_id| *class_id == track.class_id)
    {
        Some(0) => 1.0,
        Some(1) => 0.5,
        _ => 0.0,
    };
    let raw_distance = euclidean(
        track.center_x,
        track.center_y,
        observation_center.0,
        observation_center.1,
    );
    let distance = if locked.is_some_and(|locked| {
        locked.id == track.id
            && euclidean(
                locked.center_x,
                locked.center_y,
                track.center_x,
                track.center_y,
            ) <= config.debounce_distance_px
    }) {
        raw_distance * (1.0 - config.sticky_bias.clamp(0.0, 0.9))
    } else {
        raw_distance
    };
    let distance_score = 1.0 - (distance / config.target_fov_radius_px.max(1e-6)).clamp(0.0, 1.0);
    let class_weight = config.selection_class_weight.max(0.0);
    let distance_weight = config.selection_distance_weight.max(0.0);
    let total_weight = class_weight + distance_weight;
    if total_weight <= 0.0 {
        return distance_score;
    }
    (class_weight * class_score + distance_weight * distance_score) / total_weight
}

fn detection_aspect_ratio(detection: &Detection) -> f64 {
    let width = f64::from(detection.width());
    let height = f64::from(detection.height());
    (width / height).max(height / width)
}

fn association_identity_confidence(
    track: &Track,
    detection: &Detection,
    config: &TargetingConfig,
) -> Option<f64> {
    let reference_height = track.height.max(f64::from(detection.height()));
    if !reference_height.is_finite() || reference_height <= 0.0 {
        return None;
    }
    let normalized_distance = euclidean(
        track.center_x,
        track.center_y,
        detection.center_x(),
        detection.center_y(),
    ) / reference_height;
    if !normalized_distance.is_finite() || normalized_distance > config.tracker_max_match_distance {
        return None;
    }
    let position_weight = config.tracker_position_cost_weight.max(0.0);
    let iou_weight = config.tracker_iou_cost_weight.max(0.0);
    let total_weight = position_weight + iou_weight;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let overlap = track_detection_iou(track, detection);
    let cost =
        (position_weight * normalized_distance + iou_weight * (1.0 - overlap)) / total_weight;
    Some((1.0 - cost).clamp(0.0, 1.0))
}

fn track_detection_iou(track: &Track, detection: &Detection) -> f64 {
    let track_left = track.center_x - track.width * 0.5;
    let track_top = track.center_y - track.height * 0.5;
    let track_right = track.center_x + track.width * 0.5;
    let track_bottom = track.center_y + track.height * 0.5;
    let detection_left = f64::from(detection.x());
    let detection_top = f64::from(detection.y());
    let detection_right = detection_left + f64::from(detection.width());
    let detection_bottom = detection_top + f64::from(detection.height());
    let intersection_width =
        (track_right.min(detection_right) - track_left.max(detection_left)).max(0.0);
    let intersection_height =
        (track_bottom.min(detection_bottom) - track_top.max(detection_top)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = track.width * track.height
        + f64::from(detection.width()) * f64::from(detection.height())
        - intersection;
    if union > 0.0 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}
