use std::cmp::Ordering;

use novasight_core::{AppError, Detection, DetectionBatch, FrameStamp, MAX_DETECTIONS};
use thiserror::Error;

use crate::{ExecutionOutputs, HostTensor, TensorRtError};

const MAX_NMS_CANDIDATES: usize = 300;
const MAX_CLASSES: u32 = 65_536;
const MAX_EXACT_F32_INTEGER: u32 = 1 << 24;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeContract {
    RawYolo {
        output_name: String,
        class_count: u32,
        has_objectness: bool,
    },
    DecodedBoxes6 {
        output_name: String,
        class_count: u32,
    },
}

#[derive(Clone, Debug)]
pub struct DetectionDecoder {
    contract: DecodeContract,
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
    coordinate_width: u32,
    coordinate_height: u32,
}

impl DetectionDecoder {
    pub fn new(
        contract: DecodeContract,
        confidence_threshold: f32,
        nms_threshold: f32,
        max_detections: usize,
        coordinate_width: u32,
        coordinate_height: u32,
    ) -> Result<Self, DecodeError> {
        let (output_name, class_count) = match &contract {
            DecodeContract::RawYolo {
                output_name,
                class_count,
                ..
            }
            | DecodeContract::DecodedBoxes6 {
                output_name,
                class_count,
            } => (output_name, *class_count),
        };
        if output_name.trim().is_empty() {
            return Err(DecodeError::BlankOutputName);
        }
        if class_count == 0 || class_count > MAX_CLASSES {
            return Err(DecodeError::InvalidClassCount(class_count));
        }
        if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
            return Err(DecodeError::InvalidConfidenceThreshold(
                confidence_threshold,
            ));
        }
        if !nms_threshold.is_finite() || !(0.0..=1.0).contains(&nms_threshold) {
            return Err(DecodeError::InvalidNmsThreshold(nms_threshold));
        }
        if max_detections == 0 {
            return Err(DecodeError::InvalidMaxDetections(max_detections));
        }
        if coordinate_width == 0
            || coordinate_height == 0
            || coordinate_width > MAX_EXACT_F32_INTEGER
            || coordinate_height > MAX_EXACT_F32_INTEGER
        {
            return Err(DecodeError::InvalidCoordinateSpace {
                width: coordinate_width,
                height: coordinate_height,
            });
        }
        Ok(Self {
            contract,
            confidence_threshold,
            nms_threshold,
            // Existing manifests default to 300 while the targeting domain
            // admits at most 256 detections. Preserve manifest compatibility
            // while enforcing the stricter downstream ownership boundary.
            max_detections: max_detections.min(MAX_DETECTIONS),
            coordinate_width,
            coordinate_height,
        })
    }

    pub fn decode(&self, outputs: &ExecutionOutputs<'_>) -> Result<DetectionBatch, DecodeError> {
        let candidates = match &self.contract {
            DecodeContract::RawYolo {
                output_name,
                class_count,
                has_objectness,
            } => {
                let tensor = outputs.find(output_name)?;
                let candidates = decode_raw_yolo(
                    tensor,
                    *class_count,
                    *has_objectness,
                    self.confidence_threshold,
                    self.coordinate_width,
                    self.coordinate_height,
                )?;
                class_aware_nms(candidates, self.nms_threshold, self.max_detections)
            }
            DecodeContract::DecodedBoxes6 {
                output_name,
                class_count,
            } => decode_boxes6(
                outputs.find(output_name)?,
                *class_count,
                self.confidence_threshold,
                self.max_detections,
                self.coordinate_width,
                self.coordinate_height,
            )?,
        };
        let detections = candidates
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| {
                Detection::new(
                    index as u64 + 1,
                    candidate.class_id,
                    candidate.x1,
                    candidate.y1,
                    candidate.x2 - candidate.x1,
                    candidate.y2 - candidate.y1,
                    candidate.score,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DetectionBatch::new(
            FrameStamp {
                epoch: outputs.epoch(),
                generation: outputs.generation(),
                captured_at: outputs.captured_at(),
            },
            self.coordinate_width,
            self.coordinate_height,
            detections,
        )?)
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    class_id: u32,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn decode_raw_yolo(
    tensor: HostTensor<'_>,
    class_count: u32,
    has_objectness: bool,
    threshold: f32,
    width: u32,
    height: u32,
) -> Result<Vec<Candidate>, DecodeError> {
    let shape = detection_matrix_shape(tensor.dimensions())?;
    let class_count = usize::try_from(class_count).map_err(|_| DecodeError::ShapeOverflow)?;
    let columns = 4_usize
        .checked_add(class_count)
        .and_then(|value| value.checked_add(if has_objectness { 1 } else { 0 }))
        .ok_or(DecodeError::ShapeOverflow)?;
    let (candidates, channels_first) = if shape[0] == columns && shape[1] == columns {
        return Err(DecodeError::AmbiguousRawYoloShape(shape));
    } else if shape[1] == columns {
        (shape[0], false)
    } else if shape[0] == columns {
        (shape[1], true)
    } else {
        return Err(DecodeError::RawYoloShape {
            shape,
            expected_columns: columns,
        });
    };
    let value = |candidate: usize, channel: usize| {
        let index = if channels_first {
            channel * candidates + candidate
        } else {
            candidate * columns + channel
        };
        tensor.value_f32(index)
    };
    let mut selected = Vec::with_capacity(MAX_NMS_CANDIDATES);
    for candidate_index in 0..candidates {
        let cx = value(candidate_index, 0)?;
        let cy = value(candidate_index, 1)?;
        let box_width = value(candidate_index, 2)?;
        let box_height = value(candidate_index, 3)?;
        if !cx.is_finite()
            || !cy.is_finite()
            || !box_width.is_finite()
            || !box_height.is_finite()
            || box_width <= 0.0
            || box_height <= 0.0
        {
            continue;
        }
        let objectness = if has_objectness {
            value(candidate_index, 4)?
        } else {
            1.0
        };
        if !objectness.is_finite() {
            continue;
        }
        let class_offset = if has_objectness { 5 } else { 4 };
        let mut best_class = 0_u32;
        let mut best_score = f32::NEG_INFINITY;
        for class_id in 0..class_count {
            let class_score = value(candidate_index, class_offset + class_id)?;
            let score = objectness * class_score;
            if score > best_score {
                best_score = score;
                best_class = u32::try_from(class_id).map_err(|_| DecodeError::ShapeOverflow)?;
            }
        }
        if !best_score.is_finite() || best_score < threshold || best_score > 1.0 {
            continue;
        }
        let candidate = clipped_candidate(
            Candidate {
                class_id: best_class,
                score: best_score,
                x1: cx - box_width * 0.5,
                y1: cy - box_height * 0.5,
                x2: cx + box_width * 0.5,
                y2: cy + box_height * 0.5,
            },
            width,
            height,
        );
        if let Some(candidate) = candidate {
            retain_top_candidate(&mut selected, candidate);
        }
    }
    Ok(selected)
}

fn decode_boxes6(
    tensor: HostTensor<'_>,
    class_count: u32,
    threshold: f32,
    max_detections: usize,
    width: u32,
    height: u32,
) -> Result<Vec<Candidate>, DecodeError> {
    let shape = detection_matrix_shape(tensor.dimensions())?;
    if shape[1] != 6 {
        return Err(DecodeError::DecodedBoxesShape(shape));
    }
    let mut selected = Vec::with_capacity(max_detections);
    for index in 0..shape[0] {
        let base = index * 6;
        let mut x1 = tensor.value_f32(base)?;
        let mut y1 = tensor.value_f32(base + 1)?;
        let mut x2 = tensor.value_f32(base + 2)?;
        let mut y2 = tensor.value_f32(base + 3)?;
        let score = tensor.value_f32(base + 4)?;
        let raw_class = tensor.value_f32(base + 5)?;
        if !x1.is_finite()
            || !y1.is_finite()
            || !x2.is_finite()
            || !y2.is_finite()
            || !score.is_finite()
            || !raw_class.is_finite()
            || score < threshold
            || score > 1.0
            || raw_class < 0.0
        {
            continue;
        }
        let rounded_class = raw_class.round();
        if (raw_class - rounded_class).abs() > 1e-3 || rounded_class >= class_count as f32 {
            continue;
        }
        if x1.abs().max(y1.abs()).max(x2.abs()).max(y2.abs()) <= 2.0 {
            x1 *= width as f32;
            x2 *= width as f32;
            y1 *= height as f32;
            y2 *= height as f32;
        }
        if let Some(candidate) = clipped_candidate(
            Candidate {
                class_id: rounded_class as u32,
                score,
                x1,
                y1,
                x2,
                y2,
            },
            width,
            height,
        ) {
            selected.push(candidate);
        }
    }
    selected.sort_by(score_descending);
    selected.truncate(max_detections);
    Ok(selected)
}

fn detection_matrix_shape(dimensions: &[i64]) -> Result<[usize; 2], DecodeError> {
    let dimensions = if dimensions.first() == Some(&1) {
        &dimensions[1..]
    } else {
        dimensions
    };
    let [rows, columns] = dimensions else {
        return Err(DecodeError::DetectionRank(dimensions.to_vec()));
    };
    let rows = usize::try_from(*rows).map_err(|_| DecodeError::ShapeOverflow)?;
    let columns = usize::try_from(*columns).map_err(|_| DecodeError::ShapeOverflow)?;
    if rows == 0 || columns == 0 {
        return Err(DecodeError::DetectionRank(dimensions.to_vec()));
    }
    Ok([rows, columns])
}

fn clipped_candidate(mut candidate: Candidate, width: u32, height: u32) -> Option<Candidate> {
    candidate.x1 = candidate.x1.clamp(0.0, width as f32);
    candidate.y1 = candidate.y1.clamp(0.0, height as f32);
    candidate.x2 = candidate.x2.clamp(0.0, width as f32);
    candidate.y2 = candidate.y2.clamp(0.0, height as f32);
    (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1).then_some(candidate)
}

fn retain_top_candidate(candidates: &mut Vec<Candidate>, candidate: Candidate) {
    if candidates.len() < MAX_NMS_CANDIDATES {
        candidates.push(candidate);
        return;
    }
    if let Some((lowest_index, lowest)) = candidates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| score_ascending(left, right))
        && candidate.score > lowest.score
    {
        candidates[lowest_index] = candidate;
    }
}

fn class_aware_nms(
    mut candidates: Vec<Candidate>,
    threshold: f32,
    max_detections: usize,
) -> Vec<Candidate> {
    candidates.sort_by(score_descending);
    let mut kept: Vec<Candidate> = Vec::with_capacity(max_detections);
    for candidate in candidates {
        if kept.len() == max_detections {
            break;
        }
        if kept.iter().any(|prior| {
            prior.class_id == candidate.class_id
                && intersection_over_union(*prior, candidate) > threshold
        }) {
            continue;
        }
        kept.push(candidate);
    }
    kept
}

fn intersection_over_union(left: Candidate, right: Candidate) -> f32 {
    let intersection_width = (left.x2.min(right.x2) - left.x1.max(right.x1)).max(0.0);
    let intersection_height = (left.y2.min(right.y2) - left.y1.max(right.y1)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left.x2 - left.x1) * (left.y2 - left.y1);
    let right_area = (right.x2 - right.x1) * (right.y2 - right.y1);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn score_descending(left: &Candidate, right: &Candidate) -> Ordering {
    right.score.total_cmp(&left.score)
}

fn score_ascending(left: &Candidate, right: &Candidate) -> Ordering {
    left.score.total_cmp(&right.score)
}

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("decoder output name must not be blank")]
    BlankOutputName,
    #[error("decoder class count must be within 1..={MAX_CLASSES}, got {0}")]
    InvalidClassCount(u32),
    #[error("confidence threshold must be finite within 0..=1, got {0}")]
    InvalidConfidenceThreshold(f32),
    #[error("NMS threshold must be finite within 0..=1, got {0}")]
    InvalidNmsThreshold(f32),
    #[error("max detections must be positive, got {0}")]
    InvalidMaxDetections(usize),
    #[error("coordinate space must be positive, got {width}x{height}")]
    InvalidCoordinateSpace { width: u32, height: u32 },
    #[error("detection tensor must reduce to rank 2, got {0:?}")]
    DetectionRank(Vec<i64>),
    #[error("raw YOLO tensor shape {shape:?} does not contain {expected_columns} columns")]
    RawYoloShape {
        shape: [usize; 2],
        expected_columns: usize,
    },
    #[error("raw YOLO tensor shape {0:?} is ambiguous between [N,C] and [C,N]")]
    AmbiguousRawYoloShape([usize; 2]),
    #[error("decoded boxes tensor must have shape [N,6], got {0:?}")]
    DecodedBoxesShape([usize; 2]),
    #[error("decoder tensor shape overflow")]
    ShapeOverflow,
    #[error(transparent)]
    TensorRt(#[from] TensorRtError),
    #[error(transparent)]
    Domain(#[from] AppError),
}

#[cfg(test)]
mod tests {
    use std::{ffi::c_char, marker::PhantomData};

    use novasight_core::{Generation, MonotonicNanos, RuntimeEpoch};

    use super::*;
    use crate::{MAX_OUTPUTS, NativeHostTensorView, NativeTensorSpec, TensorDtype};

    fn outputs<'data>(data: &'data [f32], dimensions: &[i64]) -> ExecutionOutputs<'data> {
        let mut spec = NativeTensorSpec::empty();
        for (target, source) in spec.name.iter_mut().zip("output0".bytes()) {
            *target = source as c_char;
        }
        spec.rank = dimensions.len() as u32;
        for (target, source) in spec.dimensions.iter_mut().zip(dimensions.iter().copied()) {
            *target = source;
        }
        spec.dtype = 1;
        spec.nbytes = data.len() as u64 * 4;
        let mut views = [NativeHostTensorView::empty(); MAX_OUTPUTS];
        views[0] = NativeHostTensorView {
            host_ptr: data.as_ptr().cast(),
            nbytes: spec.nbytes,
            spec,
        };
        ExecutionOutputs {
            views,
            count: 1,
            epoch: RuntimeEpoch(7),
            generation: Generation(11),
            captured_at: MonotonicNanos(123_456),
            _engine: PhantomData,
        }
    }

    fn raw_decoder(has_objectness: bool, class_count: u32) -> DetectionDecoder {
        DetectionDecoder::new(
            DecodeContract::RawYolo {
                output_name: "output0".to_owned(),
                class_count,
                has_objectness,
            },
            0.25,
            0.5,
            10,
            100,
            100,
        )
        .unwrap()
    }

    #[test]
    fn decoded_boxes_scale_normalized_coordinates_and_preserve_frame_identity() {
        let data = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.0];
        let output = outputs(&data, &[1, 2, 6]);
        let decoder = DetectionDecoder::new(
            DecodeContract::DecodedBoxes6 {
                output_name: "output0".to_owned(),
                class_count: 2,
            },
            0.25,
            0.5,
            10,
            200,
            100,
        )
        .unwrap();

        let batch = decoder.decode(&output).unwrap();
        assert_eq!(batch.stamp().epoch, RuntimeEpoch(7));
        assert_eq!(batch.stamp().generation, Generation(11));
        assert_eq!(batch.stamp().captured_at, MonotonicNanos(123_456));
        assert_eq!(batch.detections().len(), 1);
        let detection = &batch.detections()[0];
        assert_eq!(detection.class_id(), 1);
        assert!((detection.x() - 20.0).abs() < 1e-5);
        assert!((detection.y() - 20.0).abs() < 1e-5);
        assert!((detection.width() - 80.0).abs() < 1e-5);
        assert!((detection.height() - 60.0).abs() < 1e-5);
    }

    #[test]
    fn raw_yolo_channel_first_applies_class_aware_nms() {
        let data = [
            50.0, 52.0, // cx
            50.0, 52.0, // cy
            20.0, 20.0, // width
            20.0, 20.0, // height
            0.9, 0.8, // class 0
            0.1, 0.1, // class 1
        ];
        let batch = raw_decoder(false, 2)
            .decode(&outputs(&data, &[1, 6, 2]))
            .unwrap();
        assert_eq!(batch.detections().len(), 1);
        assert!((batch.detections()[0].confidence() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn raw_yolo_candidate_first_multiplies_objectness() {
        let data = [50.0, 50.0, 20.0, 20.0, 0.5, 0.8, 0.1];
        let batch = raw_decoder(true, 2)
            .decode(&outputs(&data, &[1, 1, 7]))
            .unwrap();
        assert_eq!(batch.detections().len(), 1);
        assert!((batch.detections()[0].confidence() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_overlapping_boxes_from_different_classes() {
        let data = [
            50.0, 50.0, 20.0, 20.0, 0.9, 0.1, // class 0
            50.0, 50.0, 20.0, 20.0, 0.1, 0.8, // class 1
        ];
        let batch = raw_decoder(false, 2)
            .decode(&outputs(&data, &[1, 2, 6]))
            .unwrap();
        assert_eq!(batch.detections().len(), 2);
        assert_eq!(batch.detections()[0].class_id(), 0);
        assert_eq!(batch.detections()[1].class_id(), 1);
    }

    #[test]
    fn ambiguous_raw_yolo_layout_fails_closed() {
        let data = vec![0.0; 36];
        let error = raw_decoder(false, 2)
            .decode(&outputs(&data, &[1, 6, 6]))
            .unwrap_err();
        assert!(matches!(error, DecodeError::AmbiguousRawYoloShape([6, 6])));
    }

    #[test]
    fn decoder_configuration_is_bounded() {
        let decoder = DetectionDecoder::new(
            DecodeContract::DecodedBoxes6 {
                output_name: "output0".to_owned(),
                class_count: 1,
            },
            0.5,
            0.5,
            300,
            100,
            100,
        )
        .unwrap();
        assert_eq!(decoder.max_detections, MAX_DETECTIONS);

        let error = DetectionDecoder::new(
            DecodeContract::DecodedBoxes6 {
                output_name: "output0".to_owned(),
                class_count: MAX_CLASSES + 1,
            },
            0.5,
            0.5,
            1,
            100,
            100,
        )
        .unwrap_err();
        assert!(matches!(error, DecodeError::InvalidClassCount(_)));

        let error = DetectionDecoder::new(
            DecodeContract::DecodedBoxes6 {
                output_name: "output0".to_owned(),
                class_count: 1,
            },
            0.5,
            0.5,
            1,
            MAX_EXACT_F32_INTEGER + 1,
            100,
        )
        .unwrap_err();
        assert!(matches!(error, DecodeError::InvalidCoordinateSpace { .. }));
    }

    #[test]
    fn fixture_dtype_matches_production_output_type() {
        assert_eq!(
            outputs(&[1.0], &[1, 1])
                .iter()
                .next()
                .unwrap()
                .dtype()
                .unwrap(),
            TensorDtype::Float32
        );
    }
}
