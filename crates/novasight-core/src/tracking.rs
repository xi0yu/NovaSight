//! Phase 2 deterministic tracker and targeting core.
//!
//! The runtime replaces the Python ``RuntimeTracker`` and
//! ``RuntimeTargetSelector`` modules. It consumes admitted detector
//! candidates (including frame-local DeepStream identities), owns temporal
//! association, and returns both the chosen candidate and a stable `TrackId`.
//!
//! The implementation is intentionally bounded and exhaustive:
//! * an allocation-free constant-velocity Kalman state predicts association
//!   points and rejects NIS outliers. Mouse control still uses the latest raw
//!   aim point, so identity stability does not add control lag.
//! * rectangular minimum-cost assignment is bounded to 16 tracks and
//!   detections; temporal identity confidence combines predicted distance,
//!   bounding-box IoU, and scale.
//! * no unbounded growth. Active tracks are capped and
//!   `TargetingCore::reset` clears all temporal targeting state.
//! * lost tracks never produce a control target. Production retention uses
//!   monotonic capture time so different inference FPS values get the same
//!   grace period; frame count remains only as a timestamp-free replay fallback.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::perception::types::Detection;

mod kalman;
pub use kalman::KalmanConfig;
use kalman::KalmanState;

/// Timestamp-free replay fallback for expiring a non-matched track.
pub const DEFAULT_TRACK_MAX_AGE: u64 = 5;
pub const DEFAULT_TRACK_MAX_LOST_AGE_MS: f64 = 120.0;
const TRACK_CONFIRM_HITS: u64 = 2;
const IMMEDIATE_CONFIRM_CONFIDENCE: f32 = 0.75;

/// Hard admission bound shared with `DetectionBatch`, preventing the
/// association surface from diverging from the perception boundary.
pub const MAX_TRACK_CANDIDATES: usize = crate::perception::types::MAX_DETECTIONS;
/// Python-compatible bound for association work and retained live candidates.
pub const MAX_ACTIVE_TRACKS: usize = 16;

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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TrackId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Track {
    pub id: TrackId,
    pub object_id: u64,
    pub class_id: u32,
    pub state: TrackState,
    pub center_x: f64,
    pub center_y: f64,
    pub observed_aim_x: f64,
    pub observed_aim_y: f64,
    pub box_x: f64,
    pub box_y: f64,
    pub width: f64,
    pub height: f64,
    /// Latest detector score, kept separate from temporal identity quality.
    pub confidence: f32,
    /// Association continuity in 0..=1, where one is an exact spatial match.
    pub identity_confidence: f64,
    pub last_seen_ns: u64,
    pub age_frames: u64,
    pub hit_count: u64,
    pub confirmed: bool,
    pub missed_frames: u64,
    pub lost_since_ns: Option<u64>,
    #[serde(skip)]
    kalman: KalmanState,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Association {
    pub track_id: TrackId,
    pub object_id: u64,
    pub center_x: f64,
    pub center_y: f64,
    pub confidence: f32,
    pub identity_confidence: f64,
}

#[derive(Clone, Copy, Debug)]
struct AssociationEdge {
    track_index: usize,
    detection_index: usize,
    cost: f64,
    identity_confidence: f64,
}

/// Deterministic bounded minimum-cost association, avoiding TrackId-order
/// greedy capture when multiple same-class targets converge.
fn associate(
    tracks: &[Track],
    detections: &[Detection],
    config: &TargetingConfig,
    captured_at_ns: u64,
) -> Result<Vec<Association>, AppError> {
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

    let mut edges = Vec::with_capacity(sorted_tracks.len() * sorted_detections.len());
    for (track_index, track) in sorted_tracks.iter().enumerate() {
        for (detection_index, detection) in sorted_detections.iter().enumerate() {
            if let Some((cost, identity_confidence)) =
                association_edge(track, detection, config, captured_at_ns)
            {
                edges.push(AssociationEdge {
                    track_index,
                    detection_index,
                    cost,
                    identity_confidence,
                });
            }
        }
    }
    const FORBIDDEN_COST: f64 = 1_000_000.0;
    let mut matrix = vec![vec![FORBIDDEN_COST; sorted_detections.len()]; sorted_tracks.len()];
    for edge in &edges {
        matrix[edge.track_index][edge.detection_index] = edge.cost;
    }
    let mut out = Vec::with_capacity(sorted_tracks.len().min(sorted_detections.len()));
    for (track_index, detection_index) in linear_sum_assignment(&matrix) {
        let Some(edge) = edges.iter().find(|edge| {
            edge.track_index == track_index && edge.detection_index == detection_index
        }) else {
            continue;
        };
        let track = &sorted_tracks[track_index];
        let detection = &sorted_detections[detection_index];
        out.push(Association {
            track_id: track.id,
            object_id: detection.object_id(),
            center_x: detection.center_x(),
            center_y: detection.center_y(),
            confidence: detection.confidence(),
            identity_confidence: edge.identity_confidence,
        });
    }
    out.sort_by_key(|association| association.track_id.0);
    Ok(out)
}

/// Deterministic rectangular minimum-cost assignment in O(n^3). The caller
/// caps both dimensions at [`MAX_ACTIVE_TRACKS`].
fn linear_sum_assignment(costs: &[Vec<f64>]) -> Vec<(usize, usize)> {
    if costs.is_empty() || costs[0].is_empty() {
        return Vec::new();
    }
    let row_count = costs.len();
    let column_count = costs[0].len();
    debug_assert!(costs.iter().all(|row| row.len() == column_count));
    let transposed = row_count > column_count;
    let matrix: Vec<Vec<f64>> = if transposed {
        (0..column_count)
            .map(|column| (0..row_count).map(|row| costs[row][column]).collect())
            .collect()
    } else {
        costs.to_vec()
    };
    let rows = matrix.len();
    let columns = matrix[0].len();
    let mut u = vec![0.0; rows + 1];
    let mut v = vec![0.0; columns + 1];
    let mut p = vec![0_usize; columns + 1];
    let mut way = vec![0_usize; columns + 1];
    for row in 1..=rows {
        p[0] = row;
        let mut min_values = vec![f64::INFINITY; columns + 1];
        let mut used = vec![false; columns + 1];
        let mut column0 = 0;
        loop {
            used[column0] = true;
            let row0 = p[column0];
            let mut delta = f64::INFINITY;
            let mut column1 = 0;
            for column in 1..=columns {
                if used[column] {
                    continue;
                }
                let current = matrix[row0 - 1][column - 1] - u[row0] - v[column];
                if current < min_values[column] {
                    min_values[column] = current;
                    way[column] = column0;
                }
                if min_values[column] < delta {
                    delta = min_values[column];
                    column1 = column;
                }
            }
            for column in 0..=columns {
                if used[column] {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    min_values[column] -= delta;
                }
            }
            column0 = column1;
            if p[column0] == 0 {
                break;
            }
        }
        loop {
            let column1 = way[column0];
            p[column0] = p[column1];
            column0 = column1;
            if column0 == 0 {
                break;
            }
        }
    }
    let mut assignment = Vec::with_capacity(rows);
    for (column, &assigned_row) in p.iter().enumerate().take(columns + 1).skip(1) {
        if assigned_row == 0 {
            continue;
        }
        let row_index = assigned_row - 1;
        let column_index = column - 1;
        assignment.push(if transposed {
            (column_index, row_index)
        } else {
            (row_index, column_index)
        });
    }
    assignment.sort_unstable();
    assignment
}

fn euclidean(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TargetSelection {
    /// Frame-local candidate identity used to find the chosen bounding box.
    pub target_object_id: Option<u64>,
    /// Stable identity allocated and associated by the Rust targeting core.
    pub target_track_id: Option<TrackId>,
    pub target_class_id: Option<u32>,
    pub target_detection_confidence: Option<f32>,
    pub target_identity_confidence: Option<f64>,
    /// Whether the tracker state is safe to feed into motion prediction.
    /// Identity can remain stable while a low-confidence Kalman estimate
    /// deliberately falls back to the latest observed aim point.
    pub target_state_valid: bool,
    /// The selected identity was created or restored on this observation.
    pub target_rebuilt: bool,
    /// Control aim point. Association continues to use the geometric center.
    pub target_aim_x: Option<f64>,
    pub target_aim_y: Option<f64>,
    pub target_box_x: Option<f64>,
    pub target_box_y: Option<f64>,
    pub target_box_width: Option<f64>,
    pub target_box_height: Option<f64>,
    pub lock_reason: Option<LockReason>,
    pub candidates: usize,
    pub inside_fov: usize,
    pub rejected_class_ids: Vec<u32>,
    pub rejected_by_confidence: usize,
    pub rejected_by_class: usize,
    pub rejected_by_aspect_ratio: usize,
    pub rejected_by_fov: usize,
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
            target_state_valid: false,
            target_rebuilt: false,
            target_aim_x: None,
            target_aim_y: None,
            target_box_x: None,
            target_box_y: None,
            target_box_width: None,
            target_box_height: None,
            lock_reason: None,
            candidates: 0,
            inside_fov: 0,
            rejected_class_ids: Vec::new(),
            rejected_by_confidence: 0,
            rejected_by_class: 0,
            rejected_by_aspect_ratio: 0,
            rejected_by_fov: 0,
            lost_count: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetingConfig {
    /// Confidence threshold for admitting a detection into targeting.
    pub min_confidence: f32,
    /// Timestamp-free replay fallback for dropping a lost track. Production
    /// observations use `track_max_lost_age_ms` instead.
    pub track_max_age: u64,
    /// Production wall-clock bound for retaining a lost identity, independent
    /// of capture and inference FPS.
    pub track_max_lost_age_ms: f64,
    /// Radial admission gate around the declared observation center.
    pub target_fov_radius_px: f64,
    /// Maximum center displacement measured in target-height units.
    pub tracker_max_match_distance: f64,
    pub tracker_position_cost_weight: f64,
    pub tracker_iou_cost_weight: f64,
    pub tracker_scale_cost_weight: f64,
    pub tracker_max_size_ratio: f64,
    pub tracker_max_association_dt_ms: f64,
    pub kalman: KalmanConfig,
    /// Descending class preference. Every listed rank participates through a
    /// geometric 1.0, 0.5, 0.25, ... preference curve; unlisted classes score
    /// zero but remain selectable when admitted by `allowed_class_ids`.
    pub class_priority: Vec<u32>,
    /// Optional class admission allowlist. `None` admits every detector class;
    /// an empty set intentionally disables target selection.
    pub allowed_class_ids: Option<BTreeSet<u32>>,
    /// Share of the stateless candidate score assigned to class preference.
    /// Distance receives the complementary `1.0 - selection_class_ratio`.
    pub selection_class_ratio: f64,
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
            min_confidence: 0.5,
            track_max_age: DEFAULT_TRACK_MAX_AGE,
            track_max_lost_age_ms: DEFAULT_TRACK_MAX_LOST_AGE_MS,
            target_fov_radius_px: 180.0,
            tracker_max_match_distance: 1.5,
            tracker_position_cost_weight: 0.75,
            tracker_iou_cost_weight: 0.25,
            tracker_scale_cost_weight: 0.15,
            tracker_max_size_ratio: 2.5,
            tracker_max_association_dt_ms: 150.0,
            kalman: KalmanConfig::default(),
            class_priority: vec![0, 1],
            allowed_class_ids: None,
            selection_class_ratio: 0.35,
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
/// feed it consecutive `select` calls. The state machine owns the live
/// tracks needed for identity continuity and no diagnostic frame history.
#[derive(Debug)]
pub struct TargetingCore {
    config: TargetingConfig,
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
            tracks: Vec::new(),
            locked: None,
            pending_switch: None,
            lost_count: 0,
            next_track_id: 1,
        }
    }

    pub fn set_config(&mut self, config: TargetingConfig) {
        self.config = config;
        self.pending_switch = None;
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.locked = None;
        self.pending_switch = None;
        self.lost_count = 0;
        self.next_track_id = 1;
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
        let mut rejected_class_ids = BTreeSet::new();
        let mut rejected_by_confidence = 0;
        let mut rejected_by_class = 0;
        let mut rejected_by_aspect_ratio = 0;
        let mut rejected_by_fov = 0;
        let mut inside_fov = 0;
        let mut trackable = Vec::with_capacity(detections.len().min(MAX_ACTIVE_TRACKS));
        for detection in detections {
            if detection.confidence() < self.config.min_confidence {
                rejected_by_confidence += 1;
                continue;
            }
            if self
                .config
                .allowed_class_ids
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&detection.class_id()))
            {
                rejected_by_class += 1;
                rejected_class_ids.insert(detection.class_id());
                continue;
            }
            if detection_aspect_ratio(detection) > self.config.candidate_max_aspect_ratio {
                rejected_by_aspect_ratio += 1;
                continue;
            }
            let aim = detection_aim(detection, &self.config);
            let selectable = euclidean(aim.0, aim.1, observation_center.0, observation_center.1)
                <= self.config.target_fov_radius_px;
            if !selectable {
                rejected_by_fov += 1;
            }
            if selectable && inside_fov < MAX_ACTIVE_TRACKS {
                if trackable.len() == MAX_ACTIVE_TRACKS {
                    trackable.pop();
                }
                trackable.insert(inside_fov, detection.clone());
                inside_fov += 1;
            } else if !selectable && trackable.len() < MAX_ACTIVE_TRACKS {
                trackable.push(detection.clone());
            }
        }
        let rejected_class_ids = rejected_class_ids.into_iter().collect::<Vec<_>>();
        if trackable.is_empty() {
            self.pending_switch = None;
            self.miss_locked_target(captured_at_ns);
            return TargetSelection {
                candidates,
                inside_fov: 0,
                target_object_id: None,
                target_track_id: None,
                target_class_id: None,
                target_detection_confidence: None,
                target_identity_confidence: None,
                target_state_valid: false,
                target_rebuilt: false,
                target_aim_x: None,
                target_aim_y: None,
                target_box_x: None,
                target_box_y: None,
                target_box_width: None,
                target_box_height: None,
                lock_reason: None,
                rejected_class_ids,
                rejected_by_confidence,
                rejected_by_class,
                rejected_by_aspect_ratio,
                rejected_by_fov,
                lost_count: self.lost_count,
            };
        }

        for track in &mut self.tracks {
            track.age_frames = track.age_frames.saturating_add(1);
            if track.kalman.predict(
                captured_at_ns,
                self.config.kalman,
                track.identity_confidence,
            ) {
                (track.center_x, track.center_y) = track.kalman.position();
            } else {
                track.center_x = track.box_x + track.width * 0.5;
                track.center_y = track.box_y + track.height * 0.5;
            }
        }

        let associations =
            associate(&self.tracks, &trackable, &self.config, captured_at_ns).unwrap_or_default();
        let mut updated = Vec::with_capacity(trackable.len());
        let mut rebuilt_ids = [None; MAX_ACTIVE_TRACKS];
        for det in &trackable {
            let associated = associations.iter().find_map(|association| {
                let prior = self
                    .tracks
                    .iter()
                    .find(|track| track.id == association.track_id)?;
                if association.object_id != det.object_id() {
                    return None;
                }
                Some((prior.clone(), association.identity_confidence))
            });
            let aim = detection_aim(det, &self.config);
            let association_center = (det.center_x(), det.center_y());
            let track = if let Some((mut prior, identity_confidence)) = associated {
                if prior.state == TrackState::Lost {
                    remember_track_id(&mut rebuilt_ids, prior.id);
                }
                let filtered_valid = prior.kalman.update(
                    association_center.0,
                    association_center.1,
                    captured_at_ns,
                    self.config.kalman,
                    identity_confidence,
                );
                let filtered = prior.kalman.position();
                prior.object_id = det.object_id();
                prior.class_id = det.class_id();
                prior.state =
                    if prior.confirmed || prior.hit_count.saturating_add(1) >= TRACK_CONFIRM_HITS {
                        TrackState::Confirmed
                    } else {
                        TrackState::Tentative
                    };
                prior.confirmed = prior.state == TrackState::Confirmed;
                prior.center_x = if filtered_valid && filtered.0.is_finite() {
                    filtered.0
                } else {
                    association_center.0
                };
                prior.center_y = if filtered_valid && filtered.1.is_finite() {
                    filtered.1
                } else {
                    association_center.1
                };
                prior.observed_aim_x = aim.0;
                prior.observed_aim_y = aim.1;
                prior.box_x = f64::from(det.x());
                prior.box_y = f64::from(det.y());
                prior.width = f64::from(det.width());
                prior.height = f64::from(det.height());
                prior.confidence = det.confidence();
                prior.identity_confidence = identity_confidence;
                prior.last_seen_ns = captured_at_ns;
                prior.hit_count = prior.hit_count.saturating_add(1);
                prior.missed_frames = 0;
                prior.lost_since_ns = None;
                prior
            } else {
                let id = TrackId(self.next_track_id);
                self.next_track_id = self.next_track_id.saturating_add(1);
                let aim_is_selectable =
                    euclidean(aim.0, aim.1, observation_center.0, observation_center.1)
                        <= self.config.target_fov_radius_px;
                let confirmed = inside_fov == 1
                    && aim_is_selectable
                    && det.confidence() >= IMMEDIATE_CONFIRM_CONFIDENCE;
                remember_track_id(&mut rebuilt_ids, id);
                Track {
                    id,
                    object_id: det.object_id(),
                    class_id: det.class_id(),
                    state: if confirmed {
                        TrackState::Confirmed
                    } else {
                        TrackState::Tentative
                    },
                    center_x: association_center.0,
                    center_y: association_center.1,
                    observed_aim_x: aim.0,
                    observed_aim_y: aim.1,
                    box_x: f64::from(det.x()),
                    box_y: f64::from(det.y()),
                    width: f64::from(det.width()),
                    height: f64::from(det.height()),
                    confidence: det.confidence(),
                    identity_confidence: 1.0,
                    last_seen_ns: captured_at_ns,
                    age_frames: 1,
                    hit_count: 1,
                    confirmed,
                    missed_frames: 0,
                    lost_since_ns: None,
                    kalman: KalmanState::new(
                        association_center.0,
                        association_center.1,
                        captured_at_ns,
                        self.config.kalman,
                        1.0,
                    ),
                }
            };
            updated.push(track);
        }

        let mut matched_ids = [None; MAX_ACTIVE_TRACKS];
        for track in &updated {
            remember_track_id(&mut matched_ids, track.id);
        }
        let mut retained = self
            .tracks
            .iter()
            .filter(|track| !has_track_id(&matched_ids, track.id))
            .cloned()
            .filter_map(|mut track| {
                track.state = TrackState::Lost;
                track.missed_frames = track.missed_frames.saturating_add(1);
                track.lost_since_ns.get_or_insert(captured_at_ns);
                track_within_loss_grace(&track, captured_at_ns, &self.config).then_some(track)
            })
            .take(MAX_ACTIVE_TRACKS.saturating_sub(updated.len()))
            .collect::<Vec<_>>();
        let mut current_indices = [0_usize; MAX_ACTIVE_TRACKS];
        let mut current_count = 0;
        for (index, track) in updated.iter().enumerate() {
            if track.state == TrackState::Confirmed
                && euclidean(
                    track.observed_aim_x,
                    track.observed_aim_y,
                    observation_center.0,
                    observation_center.1,
                ) <= self.config.target_fov_radius_px
            {
                current_indices[current_count] = index;
                current_count += 1;
            }
        }
        let current_indices = &current_indices[..current_count];
        self.lost_count = retained.len() as u64;
        let retained_lock = self
            .locked
            .as_ref()
            .and_then(|locked| retained.iter().find(|track| track.id == locked.id).cloned());
        let observed_lock = self
            .locked
            .as_ref()
            .and_then(|locked| updated.iter().find(|track| track.id == locked.id).cloned());
        // A temporary miss is not a switch opportunity. Keep the identity for
        // reacquisition, but emit no target so control stops until a real
        // observation of the same track returns or the grace period expires.
        if retained_lock.is_some() {
            self.pending_switch = None;
            self.tracks = updated;
            self.tracks.append(&mut retained);
            self.locked = retained_lock;
            return TargetSelection {
                candidates,
                inside_fov,
                rejected_class_ids,
                rejected_by_confidence,
                rejected_by_class,
                rejected_by_aspect_ratio,
                rejected_by_fov,
                lost_count: self.lost_count,
                ..TargetSelection::empty()
            };
        }
        // A target observed outside the selection radius remains tracked, but
        // it cannot drive control. This preserves identity for reacquisition
        // without treating the selection FOV as a tracker admission gate.
        if current_indices.is_empty() {
            self.pending_switch = None;
            self.tracks = updated;
            self.tracks.append(&mut retained);
            self.locked = observed_lock;
            return TargetSelection {
                candidates,
                inside_fov,
                rejected_class_ids,
                rejected_by_confidence,
                rejected_by_class,
                rejected_by_aspect_ratio,
                rejected_by_fov,
                lost_count: self.lost_count,
                ..TargetSelection::empty()
            };
        }

        let locked_index = self.locked.as_ref().and_then(|locked| {
            current_indices
                .iter()
                .position(|index| updated[*index].id == locked.id)
        });
        let best_index = current_indices
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                let left = &updated[**left];
                let right = &updated[**right];
                target_score(left, observation_center, &self.config)
                    .total_cmp(&target_score(right, observation_center, &self.config))
                    .then_with(|| right.id.cmp(&left.id))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("admissible detections produce current tracks");

        let chosen_index = match locked_index {
            Some(index) if index == best_index => {
                self.pending_switch = None;
                Some(best_index)
            }
            Some(index) => {
                let best = &updated[current_indices[best_index]];
                if self.challenger_ready(
                    &updated[current_indices[index]],
                    best,
                    observation_center,
                    captured_at_ns,
                ) {
                    Some(best_index)
                } else {
                    Some(index)
                }
            }
            None => match observed_lock.as_ref() {
                Some(locked) => self
                    .challenger_ready(
                        locked,
                        &updated[current_indices[best_index]],
                        observation_center,
                        captured_at_ns,
                    )
                    .then_some(best_index),
                None => {
                    self.pending_switch = None;
                    Some(best_index)
                }
            },
        };
        let Some(chosen_index) = chosen_index else {
            self.tracks = updated;
            self.tracks.append(&mut retained);
            self.locked = observed_lock;
            return TargetSelection {
                candidates,
                inside_fov,
                rejected_class_ids,
                rejected_by_confidence,
                rejected_by_class,
                rejected_by_aspect_ratio,
                rejected_by_fov,
                lost_count: self.lost_count,
                ..TargetSelection::empty()
            };
        };
        let track = updated[current_indices[chosen_index]].clone();
        let reason = if self.config.class_priority.first().copied() == Some(track.class_id) {
            LockReason::PreferredClass
        } else {
            LockReason::FallbackClass
        };
        let selected_detection = trackable
            .iter()
            .find(|detection| detection.object_id() == track.object_id)
            .expect("selected track belongs to the admitted batch");
        let (aim_x, aim_y) = detection_aim(selected_detection, &self.config);

        self.tracks = updated;
        self.tracks.append(&mut retained);
        self.locked = Some(track.clone());
        TargetSelection {
            candidates,
            inside_fov,
            target_object_id: Some(track.object_id),
            target_track_id: Some(track.id),
            target_class_id: Some(track.class_id),
            target_detection_confidence: Some(track.confidence),
            target_identity_confidence: Some(track.identity_confidence),
            target_state_valid: track.kalman.prediction_valid(),
            target_rebuilt: has_track_id(&rebuilt_ids, track.id),
            target_aim_x: Some(aim_x),
            target_aim_y: Some(aim_y),
            target_box_x: Some(f64::from(selected_detection.x())),
            target_box_y: Some(f64::from(selected_detection.y())),
            target_box_width: Some(f64::from(selected_detection.width())),
            target_box_height: Some(f64::from(selected_detection.height())),
            lock_reason: Some(reason),
            rejected_class_ids,
            rejected_by_confidence,
            rejected_by_class,
            rejected_by_aspect_ratio,
            rejected_by_fov,
            lost_count: self.lost_count,
        }
    }

    fn challenger_ready(
        &mut self,
        locked: &Track,
        challenger: &Track,
        observation_center: (f64, f64),
        captured_at_ns: u64,
    ) -> bool {
        let advantage = target_score(challenger, observation_center, &self.config)
            - target_score(locked, observation_center, &self.config);
        let continuity = if challenger.age_frames > 1 {
            challenger.identity_confidence
        } else {
            0.0
        };
        if advantage < self.config.switch_min_preference_advantage
            || continuity < self.config.switch_min_continuity_score
        {
            self.pending_switch = None;
            return false;
        }
        let started_at_ns = self
            .pending_switch
            .filter(|pending| pending.track_id == challenger.id)
            .map_or(captured_at_ns, |pending| pending.started_at_ns);
        let elapsed_ms = captured_at_ns.saturating_sub(started_at_ns) as f64 / 1e6;
        if elapsed_ms >= self.config.switch_delay_ms {
            self.pending_switch = None;
            true
        } else {
            self.pending_switch = Some(PendingSwitch {
                track_id: challenger.id,
                started_at_ns,
            });
            false
        }
    }

    fn miss_locked_target(&mut self, captured_at_ns: u64) {
        for track in &mut self.tracks {
            track.age_frames = track.age_frames.saturating_add(1);
            if track.kalman.predict(
                captured_at_ns,
                self.config.kalman,
                track.identity_confidence,
            ) {
                (track.center_x, track.center_y) = track.kalman.position();
            }
            track.missed_frames = track.missed_frames.saturating_add(1);
            track.state = TrackState::Lost;
            track.lost_since_ns.get_or_insert(captured_at_ns);
        }
        self.tracks
            .retain(|track| track_within_loss_grace(track, captured_at_ns, &self.config));
        self.lost_count = self.tracks.len() as u64;
        self.locked = self
            .locked
            .as_ref()
            .and_then(|locked| self.tracks.iter().find(|track| track.id == locked.id))
            .cloned();
    }
}

fn track_within_loss_grace(track: &Track, captured_at_ns: u64, config: &TargetingConfig) -> bool {
    match track.lost_since_ns {
        Some(lost_since_ns) if captured_at_ns > 0 && lost_since_ns > 0 => {
            let lost_age_ms = captured_at_ns.saturating_sub(lost_since_ns) as f64 / 1e6;
            lost_age_ms <= config.track_max_lost_age_ms
        }
        _ => track.missed_frames <= config.track_max_age,
    }
}

fn target_score(track: &Track, observation_center: (f64, f64), config: &TargetingConfig) -> f64 {
    let mut class_score = 1.0;
    let mut class_matched = false;
    for class_id in &config.class_priority {
        if *class_id == track.class_id {
            class_matched = true;
            break;
        }
        class_score *= 0.5;
    }
    if !class_matched {
        class_score = 0.0;
    }
    let distance = euclidean(
        track.observed_aim_x,
        track.observed_aim_y,
        observation_center.0,
        observation_center.1,
    );
    let distance_score = 1.0 - (distance / config.target_fov_radius_px.max(1e-6)).clamp(0.0, 1.0);
    let class_ratio = config.selection_class_ratio.clamp(0.0, 1.0);
    class_ratio * class_score + (1.0 - class_ratio) * distance_score
}

fn detection_aspect_ratio(detection: &Detection) -> f64 {
    let width = f64::from(detection.width());
    let height = f64::from(detection.height());
    (width / height).max(height / width)
}

fn remember_track_id(slots: &mut [Option<TrackId>; MAX_ACTIVE_TRACKS], track_id: TrackId) {
    if slots.iter().flatten().any(|value| *value == track_id) {
        return;
    }
    if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(track_id);
    }
}

fn has_track_id(slots: &[Option<TrackId>; MAX_ACTIVE_TRACKS], track_id: TrackId) -> bool {
    slots.iter().flatten().any(|value| *value == track_id)
}

fn association_edge(
    track: &Track,
    detection: &Detection,
    config: &TargetingConfig,
    captured_at_ns: u64,
) -> Option<(f64, f64)> {
    if track.class_id != detection.class_id() {
        return None;
    }
    if captured_at_ns > 0 && track.last_seen_ns > 0 {
        let elapsed_ms = captured_at_ns.saturating_sub(track.last_seen_ns) as f64 / 1e6;
        if elapsed_ms > config.tracker_max_association_dt_ms {
            return None;
        }
    }
    let reference_height = track.height.max(f64::from(detection.height()));
    if !reference_height.is_finite() || reference_height <= 0.0 {
        return None;
    }
    // Identity association stays on the bbox geometry. Aim-point ratios are a
    // control/selection preference and must not move a track's physical state.
    let center = (detection.center_x(), detection.center_y());
    let nis = if track.kalman.prediction_valid() {
        track
            .kalman
            .measurement_nis(center.0, center.1, config.kalman)
    } else {
        track.kalman.measurement_nis_from_position(
            center.0,
            center.1,
            track.box_x + track.width * 0.5,
            track.box_y + track.height * 0.5,
            config.kalman,
        )
    };
    if !nis.is_finite() || nis > config.kalman.nis_hard_reject {
        return None;
    }
    let normalized_distance =
        euclidean(track.center_x, track.center_y, center.0, center.1) / reference_height;
    if !normalized_distance.is_finite() || normalized_distance > config.tracker_max_match_distance {
        return None;
    }
    let width_ratio = symmetric_ratio(track.width, f64::from(detection.width()))?;
    let height_ratio = symmetric_ratio(track.height, f64::from(detection.height()))?;
    if width_ratio > config.tracker_max_size_ratio || height_ratio > config.tracker_max_size_ratio {
        return None;
    }
    let position_weight = config.tracker_position_cost_weight.max(0.0);
    let iou_weight = config.tracker_iou_cost_weight.max(0.0);
    let scale_weight = config.tracker_scale_cost_weight.max(0.0);
    let total_weight = position_weight + iou_weight + scale_weight;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let overlap = track_detection_iou(track, detection);
    let scale_cost = (track.width / f64::from(detection.width())).ln().abs()
        + (track.height / f64::from(detection.height())).ln().abs();
    let cost = (position_weight * normalized_distance
        + iou_weight * (1.0 - overlap)
        + scale_weight * scale_cost)
        / total_weight;
    Some((cost, (1.0 - cost).clamp(0.0, 1.0)))
}

fn symmetric_ratio(left: f64, right: f64) -> Option<f64> {
    if !left.is_finite() || !right.is_finite() || left <= 0.0 || right <= 0.0 {
        return None;
    }
    Some((left / right).max(right / left))
}

fn track_detection_iou(track: &Track, detection: &Detection) -> f64 {
    let track_left = track.box_x;
    let track_top = track.box_y;
    let track_right = track.box_x + track.width;
    let track_bottom = track.box_y + track.height;
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

fn detection_aim(detection: &Detection, config: &TargetingConfig) -> (f64, f64) {
    let ratio = config
        .class_aim_y_ratios
        .get(&detection.class_id())
        .copied()
        .unwrap_or(config.aim_y_ratio)
        .clamp(0.0, 1.0);
    (
        detection.center_x(),
        f64::from(detection.y()) + f64::from(detection.height()) * ratio,
    )
}
