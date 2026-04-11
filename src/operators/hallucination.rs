//! Frequency-based hallucination detection.
//!
//! Whisper sometimes hallucinates repeated phrases, noise artifacts,
//! or empty segments. This operator catches them via inline checks
//! (per-segment) and periodic sweeps (retroactive frequency analysis).
//!
//! No hardcoded phrase lists — detection is purely statistical.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::error::Result;
use crate::types::TranscriptSegment;

use super::{OperatorResult, Operator};

/// Frequency-based hallucination detector.
pub struct HallucinationOperator {
    /// Global text frequency counts across all speakers.
    global_counts: HashMap<String, u32>,
    /// Per-speaker text frequency counts.
    per_speaker_counts: HashMap<String, HashMap<String, u32>>,
    /// Texts identified as noise by sweep analysis.
    noise_texts: HashSet<String>,
    /// Total segments seen (for frequency calculations).
    segment_count: u32,
    /// Segment IDs pending retroactive analysis in the next sweep.
    pending: Vec<Uuid>,
    /// Threshold for inline repeated-phrase detection (same text N+ times).
    repeat_threshold: u32,
    /// Threshold for sweep: short text appearing in >X% of segments is noise.
    frequency_threshold: f32,
    /// Threshold for sweep: one speaker producing >X% identical text is noise.
    speaker_dominance_threshold: f32,
}

impl HallucinationOperator {
    /// Create a new hallucination operator with default thresholds.
    pub fn new() -> Self {
        Self {
            global_counts: HashMap::new(),
            per_speaker_counts: HashMap::new(),
            noise_texts: HashSet::new(),
            segment_count: 0,
            pending: Vec::new(),
            repeat_threshold: 5,
            frequency_threshold: 0.03,         // 3% of all segments
            speaker_dominance_threshold: 0.80, // 80% same text from one speaker
        }
    }

    /// Check if text contains no alphabetic characters (pure noise/punctuation).
    fn has_no_letters(text: &str) -> bool {
        !text.chars().any(|c| c.is_alphabetic())
    }

    /// Check if text is a short repeated phrase (e.g. "yeah yeah yeah yeah yeah").
    fn is_repeated_phrase(text: &str) -> bool {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 3 {
            return false;
        }
        // Check if all words are the same
        let first = words[0];
        words.iter().all(|&w| w == first)
    }

    /// Normalize text for frequency comparison: lowercase, trim, collapse whitespace.
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for HallucinationOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Operator for HallucinationOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        self.segment_count += 1;
        let text = segment.text.trim();

        // Inline check: empty text
        if text.is_empty() {
            return OperatorResult::Exclude {
                reason: "empty transcription".into(),
            };
        }

        // Inline check: no alphabetic characters
        if Self::has_no_letters(text) {
            return OperatorResult::Exclude {
                reason: "no alphabetic characters (noise)".into(),
            };
        }

        // Inline check: repeated single word/phrase
        if Self::is_repeated_phrase(text) {
            return OperatorResult::Exclude {
                reason: "repeated phrase pattern".into(),
            };
        }

        let normalized = Self::normalize(text);

        // Inline check: text already identified as noise by a previous sweep
        if self.noise_texts.contains(&normalized) {
            return OperatorResult::Exclude {
                reason: "previously identified as noise by sweep".into(),
            };
        }

        // Update frequency counts
        *self.global_counts.entry(normalized.clone()).or_insert(0) += 1;
        *self
            .per_speaker_counts
            .entry(segment.speaker_pseudo_id.clone())
            .or_default()
            .entry(normalized.clone())
            .or_insert(0) += 1;

        // Inline check: same exact text repeated too many times globally
        if let Some(&count) = self.global_counts.get(&normalized) {
            if count >= self.repeat_threshold {
                self.noise_texts.insert(normalized);
                return OperatorResult::Exclude {
                    reason: format!("text repeated {} times globally", count),
                };
            }
        }

        // Queue for retroactive sweep analysis
        self.pending.push(segment.id);

        OperatorResult::Pass
    }

    async fn sweep(&mut self) -> Result<u32> {
        let mut newly_identified = 0u32;

        // Identify short texts that appear in >frequency_threshold of all segments
        if self.segment_count > 0 {
            let threshold_count =
                (self.segment_count as f32 * self.frequency_threshold).ceil() as u32;

            let new_noise: Vec<String> = self
                .global_counts
                .iter()
                .filter(|(text, &count)| {
                    // Only flag short texts (5 words or fewer) — long repeated
                    // text is more likely legitimate (e.g. game rules recitation)
                    let word_count = text.split_whitespace().count();
                    word_count <= 5 && count >= threshold_count
                })
                .map(|(text, _)| text.clone())
                .filter(|text| !self.noise_texts.contains(text))
                .collect();

            newly_identified += new_noise.len() as u32;
            self.noise_texts.extend(new_noise);
        }

        // Identify per-speaker dominance: one speaker producing >80% identical text
        for (speaker, counts) in &self.per_speaker_counts {
            let speaker_total: u32 = counts.values().sum();
            if speaker_total < 5 {
                continue; // Not enough data to judge
            }

            let dominant: Vec<String> = counts
                .iter()
                .filter(|(_, &count)| {
                    count as f32 / speaker_total as f32 > self.speaker_dominance_threshold
                })
                .map(|(text, _)| text.clone())
                .filter(|text| !self.noise_texts.contains(text))
                .collect();

            if !dominant.is_empty() {
                tracing::info!(
                    speaker = %speaker,
                    dominant_texts = ?dominant,
                    "speaker dominance detected"
                );
                newly_identified += dominant.len() as u32;
                self.noise_texts.extend(dominant);
            }
        }

        // Clear pending buffer — segments already emitted, retroactive
        // exclusion would need to be applied by the caller via segment IDs
        self.pending.clear();

        tracing::debug!(
            noise_texts = self.noise_texts.len(),
            newly_identified,
            "hallucination sweep complete"
        );

        Ok(newly_identified)
    }

    async fn finalize(&mut self) -> Result<()> {
        tracing::info!(
            total_segments = self.segment_count,
            noise_patterns = self.noise_texts.len(),
            "hallucination operator finalized"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            segment_index: 0,
            speaker_pseudo_id: "speaker_a".into(),
            start_time: 0.0,
            end_time: 1.0,
            text: text.into(),
            original_text: text.into(),
            confidence: None,
            beat_id: None,
            chunk_group: None,
            talk_type: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[tokio::test]
    async fn excludes_empty_text() {
        let mut op = HallucinationOperator::new();
        let mut seg = make_segment("");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Exclude { .. }));
    }

    #[tokio::test]
    async fn excludes_no_letters() {
        let mut op = HallucinationOperator::new();
        let mut seg = make_segment("... --- ...");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Exclude { .. }));
    }

    #[tokio::test]
    async fn excludes_repeated_phrase() {
        let mut op = HallucinationOperator::new();
        let mut seg = make_segment("yeah yeah yeah yeah");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Exclude { .. }));
    }

    #[tokio::test]
    async fn passes_normal_text() {
        let mut op = HallucinationOperator::new();
        let mut seg = make_segment("I roll for initiative");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Pass));
    }

    #[tokio::test]
    async fn excludes_after_repeat_threshold() {
        let mut op = HallucinationOperator::new();
        // Feed the same text 5 times (repeat_threshold)
        for _ in 0..5 {
            let mut seg = make_segment("Thank you for watching.");
            op.on_segment(&mut seg).await;
        }
        // The 5th one should trigger exclusion
        let mut seg = make_segment("Thank you for watching.");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Exclude { .. }));
    }
}
