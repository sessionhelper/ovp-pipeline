//! Operator chain for post-transcription processing.
//!
//! Operators implement the `Operator` trait and are applied in order
//! to each transcript segment. The crate ships with hallucination
//! detection and scene chunking.

pub mod beat;
pub mod hallucination;
pub mod metatalk;
pub mod scene;
pub mod scene_chunker;

use crate::error::{PipelineError, Result};
use crate::types::{PipelineBeat, PipelineScene, TranscriptSegment};

/// Result of applying an operator to a single segment.
#[derive(Debug, Clone)]
pub enum OperatorResult {
    /// Segment passes this operator.
    Pass,
    /// Segment should be excluded, with a reason.
    Exclude {
        /// Human-readable explanation of why the segment was excluded.
        reason: String,
    },
}

/// Trait for pluggable transcript operators.
///
/// Each operator maintains internal state and processes segments one at a time
/// via `on_segment`. Periodic `sweep` calls allow retroactive analysis
/// (e.g. frequency-based hallucination detection). `finalize` is called
/// once after all segments are processed.
#[async_trait::async_trait]
pub trait Operator: Send + Sync {
    /// Process a single segment. May mutate the segment (e.g. set `chunk_group`)
    /// or return `Exclude` to mark it for removal.
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult;

    /// Periodic sweep for retroactive analysis. Returns the number of
    /// segments retroactively excluded.
    async fn sweep(&mut self) -> Result<u32>;

    /// Called once after all segments have been processed. Perform any
    /// final cleanup or retroactive marking.
    async fn finalize(&mut self) -> Result<()>;

    /// Collect any narrative beats produced by this operator.
    /// Called after all segments have been processed and sweep/finalize have run.
    /// Default implementation returns empty — only the beat operator overrides.
    fn collect_beats(&self) -> Vec<PipelineBeat> { Vec::new() }

    /// Collect any scene groupings produced by this operator.
    /// Called after all segments have been processed and sweep/finalize have run.
    /// Default implementation returns empty — only the scene operator overrides.
    fn collect_scenes(&self) -> Vec<PipelineScene> { Vec::new() }
}

/// Apply all operators to a list of transcript segments.
///
/// Each segment passes through every operator in order. If any operator
/// returns `Exclude`, the segment is marked as excluded with the reason.
/// After all segments, `sweep` and `finalize` are called on each operator.
pub async fn apply_operators(
    mut segments: Vec<TranscriptSegment>,
    operators: &mut [Box<dyn Operator>],
) -> Result<Vec<TranscriptSegment>> {
    // Run each segment through all operators
    for segment in segments.iter_mut() {
        for operator in operators.iter_mut() {
            if segment.excluded {
                break; // Already excluded by a previous operator
            }

            match operator.on_segment(segment).await {
                OperatorResult::Pass => {}
                OperatorResult::Exclude { reason } => {
                    segment.excluded = true;
                    segment.exclude_reason = Some(reason);
                }
            }
        }
    }

    // Run sweep on each operator for retroactive analysis
    for operator in operators.iter_mut() {
        operator
            .sweep()
            .await
            .map_err(|e| PipelineError::Filter(e.to_string()))?;
    }

    // Finalize each operator
    for operator in operators.iter_mut() {
        operator
            .finalize()
            .await
            .map_err(|e| PipelineError::Filter(e.to_string()))?;
    }

    tracing::info!(
        total = segments.len(),
        excluded = segments.iter().filter(|s| s.excluded).count(),
        "operator chain complete"
    );

    Ok(segments)
}

/// Create the default operator chain: hallucination detection + metatalk classification + mechanical scene chunking.
pub fn default_operators() -> Vec<Box<dyn Operator>> {
    vec![
        Box::new(hallucination::HallucinationOperator::new()),
        Box::new(metatalk::MetatalkOperator::new()),
        Box::new(scene_chunker::SceneOperator::new(
            scene_chunker::SceneOperatorConfig::default(),
        )),
    ]
}

/// Create the operator chain with LLM-backed beat detection + scene grouping.
pub fn operators_with_llm_scene(
    beat_config: beat::BeatConfig,
    scene_config: scene::SceneConfig,
) -> Vec<Box<dyn Operator>> {
    vec![
        Box::new(hallucination::HallucinationOperator::new()),
        Box::new(metatalk::MetatalkOperator::new()),
        Box::new(beat::BeatOperator::new(beat_config)),
        Box::new(scene::SceneOperator::new(scene_config)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(text: &str, start: f32, end: f32, speaker: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            segment_index: 0,
            speaker_pseudo_id: speaker.into(),
            start_time: start,
            end_time: end,
            text: text.into(),
            original_text: text.into(),
            confidence: None,
            chunk_group: None,
            beat_id: None,
            talk_type: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[tokio::test]
    async fn apply_operators_passes_clean_segments() {
        let segments = vec![
            make_segment("Hello world", 0.0, 1.0, "a"),
            make_segment("How are you", 1.0, 2.0, "b"),
        ];
        let mut filters = default_operators();
        let result = apply_operators(segments, &mut filters).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(!result[0].excluded);
        assert!(!result[1].excluded);
    }

    #[tokio::test]
    async fn apply_operators_excludes_empty() {
        let segments = vec![
            make_segment("", 0.0, 1.0, "a"),
            make_segment("Good text", 1.0, 2.0, "a"),
        ];
        let mut filters = default_operators();
        let result = apply_operators(segments, &mut filters).await.unwrap();
        assert!(result[0].excluded);
        assert!(!result[1].excluded);
    }

    #[tokio::test]
    async fn apply_operators_assigns_scene_groups() {
        let segments = vec![
            make_segment("Scene one dialog", 0.0, 1.0, "a"),
            make_segment("More scene one", 2.0, 3.0, "a"),
            // 60 second gap — should trigger scene break (default max_silence_gap is 30s)
            make_segment("Scene two dialog", 63.0, 64.0, "a"),
        ];
        let mut filters = default_operators();
        let result = apply_operators(segments, &mut filters).await.unwrap();
        assert_eq!(result[0].chunk_group, Some(0));
        assert_eq!(result[1].chunk_group, Some(0));
        assert_eq!(result[2].chunk_group, Some(1));
    }

    #[tokio::test]
    async fn apply_operators_excluded_segment_skips_later_filters() {
        // An empty segment should be excluded by hallucination filter
        // and never reach the scene chunker (no chunk_group assigned)
        let segments = vec![make_segment("", 0.0, 1.0, "a")];
        let mut filters = default_operators();
        let result = apply_operators(segments, &mut filters).await.unwrap();
        assert!(result[0].excluded);
        assert!(result[0].chunk_group.is_none());
    }

    #[tokio::test]
    async fn sweep_catches_high_frequency_noise() {
        // Generate segments where "Thank you." appears in >3% of all segments
        let mut segments: Vec<TranscriptSegment> = (0..100)
            .map(|i| make_segment(
                &format!("Unique text number {}", i),
                i as f32, (i + 1) as f32, "gm"
            ))
            .collect();
        // Add 5 "Thank you." from one speaker (5% > 3% threshold)
        for i in 0..5 {
            segments.push(make_segment(
                "Thank you.", (100 + i) as f32, (101 + i) as f32, "noisy"
            ));
        }

        let mut filters = default_operators();
        let result = apply_operators(segments, &mut filters).await.unwrap();

        let thank_yous: Vec<_> = result.iter()
            .filter(|s| s.original_text == "Thank you.")
            .collect();
        // At least some should be excluded by sweep
        let excluded_count = thank_yous.iter().filter(|s| s.excluded).count();
        assert!(excluded_count > 0, "sweep should catch high-frequency short text");
    }
}
