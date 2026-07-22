//! Phase 2 deterministic tracker and targeting core.
//!
//! The runtime replaces the Python ``RuntimeTracker`` and
//! ``RuntimeTargetSelector`` modules. It consumes admitted detector
//! candidates (including frame-local DeepStream identities), owns temporal
//! association, and returns both the chosen candidate and a stable `TrackId`.
//!
//! The implementation is intentionally small and exhaustive:
//! * no Kalman filter or Hungarian assignment. The bounded nearest-centroid
//!   association is deterministic and sufficient for the current single-lock
//!   control contract.
//! * no unbounded growth. History is bounded by `BoundedHistory` and
//!   `TargetingCore::reset` is the only way to clear it.
//! * lost tracks never produce a control target. After a configurable
//!   number of missed frames the locked track is dropped and the
//!   selection returns `target_object_id = None` until a fresh
//!   candidate re-acquires the lock.

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
    pub confidence: f32,
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
}

impl Default for TargetingConfig {
    fn default() -> Self {
        Self {
            debounce_distance_px: 64.0,
            min_confidence: 0.5,
            track_max_age: DEFAULT_TRACK_MAX_AGE,
        }
    }
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
            lost_count: 0,
            next_track_id: 1,
        }
    }

    pub fn reset(&mut self) {
        self.history = BoundedHistory::new(self.history.limit);
        self.tracks.clear();
        self.locked = None;
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

    /// Run one selection step. Detections outside the minimum
    /// confidence are filtered before association. While a preferred-
    /// class (class 0 / head) candidate is present and within the
    /// debounce window from the prior lock, the lock stays on the
    /// head with `PreferredClass`. Once the head moves further than
    /// the window, the lock falls back to the next best class-1
    /// candidate with `FallbackClass`. This mirrors the Python
    /// `RuntimeTargetSelector` semantics pinned by
    /// `target-switch-loss.jsonl`.
    pub fn select(&mut self, detections: &[Detection]) -> TargetSelection {
        let candidates = detections.len();
        let admissible: Vec<Detection> = detections
            .iter()
            .filter(|det| det.confidence() >= self.config.min_confidence)
            .cloned()
            .collect();
        if admissible.is_empty() {
            self.miss_locked_target();
            return TargetSelection {
                candidates,
                inside_fov: 0,
                target_object_id: None,
                target_track_id: None,
                target_class_id: None,
                lock_reason: None,
                lost_count: self.lost_count,
            };
        }

        let mut class0: Vec<&Detection> = admissible
            .iter()
            .filter(|det| det.class_id() == 0)
            .collect();
        let mut class1: Vec<&Detection> = admissible
            .iter()
            .filter(|det| det.class_id() == 1)
            .collect();
        let associations = associate(&self.tracks, &admissible).unwrap_or_default();
        class0.sort_by(|a, b| {
            distance_to_target(a, self.locked.as_ref())
                .partial_cmp(&distance_to_target(b, self.locked.as_ref()))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object_id().cmp(&b.object_id()))
        });
        class1.sort_by(|a, b| {
            distance_to_target(a, self.locked.as_ref())
                .partial_cmp(&distance_to_target(b, self.locked.as_ref()))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object_id().cmp(&b.object_id()))
        });

        let (chosen, reason) = match (class0.first(), class1.first()) {
            (Some(head), _) if self.locked.is_none() => (Some(*head), LockReason::PreferredClass),
            (Some(head), _) if self.head_within_debounce(head) => {
                (Some(*head), LockReason::PreferredClass)
            }
            (Some(head), None) => (Some(*head), LockReason::PreferredClass),
            (_, Some(body)) => (Some(*body), LockReason::FallbackClass),
            (None, None) => (None, LockReason::FallbackClass),
        };

        let track = match chosen {
            Some(det) => {
                let associated = associations.iter().find_map(|association| {
                    let prior = self
                        .tracks
                        .iter()
                        .find(|track| track.id == association.track_id)?;
                    (association.object_id == det.object_id()
                        && euclidean(
                            prior.center_x,
                            prior.center_y,
                            det.center_x(),
                            det.center_y(),
                        ) <= self.config.debounce_distance_px)
                        .then_some(prior)
                });
                let id = associated.map_or_else(
                    || {
                        let id = TrackId(self.next_track_id);
                        self.next_track_id = self.next_track_id.saturating_add(1);
                        id
                    },
                    |track| track.id,
                );
                Track {
                    id,
                    object_id: det.object_id(),
                    class_id: det.class_id(),
                    state: TrackState::Confirmed,
                    center_x: det.center_x(),
                    center_y: det.center_y(),
                    confidence: det.confidence(),
                    age_frames: associated.map_or(1, |track| track.age_frames.saturating_add(1)),
                    missed_frames: 0,
                }
            }
            None => {
                self.miss_locked_target();
                return TargetSelection {
                    candidates,
                    inside_fov: 0,
                    target_object_id: None,
                    target_track_id: None,
                    target_class_id: None,
                    lock_reason: None,
                    lost_count: self.lost_count,
                };
            }
        };

        self.history.push(track.clone());
        self.tracks.clear();
        self.tracks.push(track.clone());
        self.lost_count = 0;
        self.locked = Some(track.clone());
        TargetSelection {
            candidates,
            inside_fov: admissible.len(),
            target_object_id: Some(track.object_id),
            target_track_id: Some(track.id),
            target_class_id: Some(track.class_id),
            lock_reason: Some(reason),
            lost_count: self.lost_count,
        }
    }

    fn head_within_debounce(&self, head: &Detection) -> bool {
        match self.locked.as_ref() {
            Some(prev) => {
                euclidean(
                    prev.center_x,
                    prev.center_y,
                    head.center_x(),
                    head.center_y(),
                ) <= self.config.debounce_distance_px
            }
            None => true,
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

fn distance_to_target(det: &Detection, target: Option<&Track>) -> f64 {
    match target {
        Some(target) => euclidean(
            target.center_x,
            target.center_y,
            det.center_x(),
            det.center_y(),
        ),
        None => euclidean(0.0, 0.0, det.center_x(), det.center_y()),
    }
}
