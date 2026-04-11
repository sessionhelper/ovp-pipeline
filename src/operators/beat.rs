//! Beat detection operator.
//!
//! Watches the segment stream and detects narrative beat boundaries using
//! cheap heuristics (silence gaps, GM narration patterns). Periodically
//! calls an LLM to summarize what happened and emit beat assignments.
//! This is the fine-grained layer — the SceneOperator groups beats into
//! scenes.

use crate::error::{PipelineError, Result};
use crate::types::{PipelineBeat, TranscriptSegment};

use super::{Operator, OperatorResult};

use regex::Regex;

/// Configuration for the beat detection operator.
#[derive(Debug, Clone)]
pub struct BeatConfig {
    /// LLM HTTP endpoint (OpenAI-compatible chat completions).
    pub endpoint: String,
    /// Model name to request.
    pub model: String,
    /// How often (in seconds of transcript time) to evaluate for beats.
    pub chunk_seconds: f32,
    /// Silence gap (seconds) that forces an immediate evaluation.
    pub silence_trigger: f32,
    /// GM speaker pseudo_id for narration cue detection.
    pub gm_speaker_id: Option<String>,
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434/v1/chat/completions".into(),
            model: "qwen2.5:7b".into(),
            chunk_seconds: 300.0,
            silence_trigger: 8.0,
            gm_speaker_id: None,
        }
    }
}

/// A detected narrative beat.
#[derive(Debug, Clone)]
pub struct Beat {
    /// Beat ID (0-based, sequential).
    pub id: u32,
    /// Approximate timestamp (seconds) where this beat starts.
    pub timestamp: f32,
    /// Short title for the beat.
    pub title: String,
    /// One-sentence summary.
    pub summary: String,
}

/// GM narration phrases that suggest narrative transitions.
const NARRATION_CUES: &[&str] = &[
    "meanwhile",
    "later that",
    "the next morning",
    "the next day",
    "you arrive at",
    "you wake up",
    "cut to",
    "back at the",
    "some time later",
    "hours pass",
    "when you return",
    "the scene shifts",
    "okay so",
    "alright so",
    "moving on",
];

/// Beat detection operator.
pub struct BeatOperator {
    config: BeatConfig,
    /// All segments seen so far (for context).
    all_segments: Vec<TranscriptSegment>,
    /// Segments since the last beat evaluation.
    pending_segments: Vec<TranscriptSegment>,
    /// Detected beats.
    beats: Vec<Beat>,
    /// Next beat ID to assign.
    next_beat_id: u32,
    /// Timestamp of the last beat evaluation (transcript time, not wall time).
    last_eval_time: Option<f32>,
    /// End time of the last non-excluded segment.
    last_end: Option<f32>,
    /// HTTP client.
    #[cfg(feature = "transcribe")]
    client: reqwest::Client,
}

impl BeatOperator {
    pub fn new(config: BeatConfig) -> Self {
        Self {
            config,
            all_segments: Vec::new(),
            pending_segments: Vec::new(),
            beats: Vec::new(),
            next_beat_id: 0,
            last_eval_time: None,
            last_end: None,
            #[cfg(feature = "transcribe")]
            client: reqwest::Client::new(),
        }
    }

    /// Return a reference to the detected beats.
    pub fn beats(&self) -> &[Beat] {
        &self.beats
    }

    /// Check if we should trigger a beat evaluation based on heuristics.
    fn should_evaluate(&self, segment: &TranscriptSegment) -> bool {
        // No pending segments means nothing to evaluate
        if self.pending_segments.is_empty() {
            return false;
        }

        // Silence gap trigger
        if let Some(last_end) = self.last_end {
            let gap = segment.start_time - last_end;
            if gap >= self.config.silence_trigger {
                return true;
            }
        }

        // Time-based trigger: chunk_seconds since last evaluation
        let eval_start = self.last_eval_time.unwrap_or(0.0);
        let elapsed = segment.start_time - eval_start;
        if elapsed >= self.config.chunk_seconds {
            return true;
        }

        // GM narration cue trigger
        if self.has_narration_cue(segment) {
            return true;
        }

        false
    }

    /// Check if a segment contains GM narration cues.
    fn has_narration_cue(&self, segment: &TranscriptSegment) -> bool {
        if let Some(ref gm_id) = self.config.gm_speaker_id {
            if segment.speaker_pseudo_id == *gm_id {
                let lower = segment.text.to_lowercase();
                for cue in NARRATION_CUES {
                    if lower.contains(cue) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Format pending segments as transcript text for the LLM prompt.
    fn format_pending_transcript(&self) -> String {
        self.pending_segments
            .iter()
            .filter(|s| !s.excluded)
            .map(|s| {
                let minutes = (s.start_time / 60.0) as u32;
                let seconds = (s.start_time % 60.0) as u32;
                format!("[{}:{:02}] {}: {}", minutes, seconds, s.speaker_pseudo_id, s.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Evaluate pending segments for beats via LLM.
    #[cfg(feature = "transcribe")]
    async fn evaluate_beats(&mut self) -> Result<Vec<Beat>> {
        let transcript = self.format_pending_transcript();
        if transcript.trim().is_empty() {
            return Ok(Vec::new());
        }

        let start_time = self
            .pending_segments
            .first()
            .map(|s| s.start_time)
            .unwrap_or(0.0);
        let end_time = self
            .pending_segments
            .last()
            .map(|s| s.end_time)
            .unwrap_or(0.0);

        let start_min = (start_time / 60.0) as u32;
        let start_sec = (start_time % 60.0) as u32;
        let end_min = (end_time / 60.0) as u32;
        let end_sec = (end_time % 60.0) as u32;

        let prompt = format!(
            "TTRPG session transcript, {}:{:02}-{}:{:02}:\n\n{}\n\n\
             List the narrative beats (distinct things that happen). One per line:\n\
             BEAT at M:SS | Title | One sentence summary\n\n\
             If this is all one continuous beat, write:\n\
             BEAT at M:SS | Title | Summary",
            start_min, start_sec, end_min, end_sec, transcript
        );

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        });

        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| PipelineError::Filter(format!("Beat LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(PipelineError::Filter(format!(
                "Beat LLM returned {}: {}",
                status, body_text
            )));
        }

        let resp: serde_json::Value = response
            .json()
            .await
            .map_err(|e| PipelineError::Filter(format!("Beat LLM response parse failed: {}", e)))?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| PipelineError::Filter("Beat LLM response missing content".into()))?;

        let beats = self.parse_beat_response(content);
        Ok(beats)
    }

    /// Stub when transcribe feature is disabled.
    #[cfg(not(feature = "transcribe"))]
    async fn evaluate_beats(&mut self) -> Result<Vec<Beat>> {
        Ok(Vec::new())
    }

    /// Parse the LLM response into beats. Handles markdown fences and
    /// various formatting quirks from small models.
    fn parse_beat_response(&mut self, raw: &str) -> Vec<Beat> {
        let cleaned = strip_code_fences(raw);
        let re = Regex::new(r"(?i)BEAT\s+at\s+(\d+):(\d+)\s*\|\s*(.+?)\s*\|\s*(.+)")
            .expect("invalid beat regex");

        let mut beats = Vec::new();
        for line in cleaned.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(caps) = re.captures(trimmed) {
                let minutes: u32 = caps[1].parse().unwrap_or(0);
                let seconds: u32 = caps[2].parse().unwrap_or(0);
                let timestamp = (minutes * 60 + seconds) as f32;
                let title = caps[3].trim().to_string();
                let summary = caps[4].trim().to_string();

                let beat = Beat {
                    id: self.next_beat_id,
                    timestamp,
                    title,
                    summary,
                };
                self.next_beat_id += 1;
                beats.push(beat);
            }
        }

        if beats.is_empty() {
            tracing::warn!(
                raw_response = raw,
                "beat LLM returned no parseable beats, creating default beat"
            );
            // Create a single default beat covering the whole pending window
            let timestamp = self
                .pending_segments
                .first()
                .map(|s| s.start_time)
                .unwrap_or(0.0);
            let beat = Beat {
                id: self.next_beat_id,
                timestamp,
                title: "Continuation".into(),
                summary: "Session continues".into(),
            };
            self.next_beat_id += 1;
            beats.push(beat);
        }

        beats
    }

    /// Assign beat_id to pending segments based on detected beats.
    /// Each segment is assigned to the beat whose timestamp is closest
    /// but not after the segment's start time.
    fn assign_beats_to_pending(&mut self, new_beats: &[Beat]) {
        if new_beats.is_empty() {
            return;
        }

        for seg in self.pending_segments.iter_mut() {
            if seg.excluded {
                continue;
            }
            // Find the last beat that starts at or before this segment
            let mut assigned = &new_beats[0];
            for beat in new_beats.iter() {
                if beat.timestamp <= seg.start_time {
                    assigned = beat;
                } else {
                    break;
                }
            }
            seg.beat_id = Some(assigned.id);
        }
    }
}

/// Strip markdown code fences from LLM output.
fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.starts_with("```") {
        let inner = trimmed
            .strip_prefix("```markdown")
            .or_else(|| trimmed.strip_prefix("```text"))
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        inner
            .strip_suffix("```")
            .unwrap_or(inner)
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    }
}

#[async_trait::async_trait]
impl Operator for BeatOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        if segment.excluded {
            return OperatorResult::Pass;
        }

        // Check if we should evaluate before adding this segment
        let should_eval = self.should_evaluate(segment);

        if should_eval && !self.pending_segments.is_empty() {
            tracing::debug!(
                time = segment.start_time,
                pending = self.pending_segments.len(),
                "beat evaluation triggered"
            );

            match self.evaluate_beats().await {
                Ok(new_beats) => {
                    tracing::info!(
                        count = new_beats.len(),
                        "beats detected"
                    );

                    // Assign beat IDs to pending segments
                    self.assign_beats_to_pending(&new_beats);

                    // Copy beat assignments back to the all_segments buffer
                    // (the caller sees mutations on the segment refs, but we
                    // also need them on the segments we already stored)
                    for pending in &self.pending_segments {
                        if let Some(stored) = self
                            .all_segments
                            .iter_mut()
                            .find(|s| s.id == pending.id)
                        {
                            stored.beat_id = pending.beat_id;
                        }
                    }

                    self.beats.extend(new_beats);
                    self.last_eval_time = Some(segment.start_time);
                    self.pending_segments.clear();
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "beat evaluation failed, will retry next trigger"
                    );
                }
            }
        }

        // Add segment to pending buffer
        self.pending_segments.push(segment.clone());
        self.all_segments.push(segment.clone());
        self.last_end = Some(segment.end_time);

        // If we have a current beat and no evaluation just happened,
        // assign the current beat to this segment
        if segment.beat_id.is_none() {
            if let Some(last_beat) = self.beats.last() {
                segment.beat_id = Some(last_beat.id);
            }
        }

        OperatorResult::Pass
    }

    fn collect_beats(&self) -> Vec<PipelineBeat> {
        self.beats
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // Estimate end_time from the next beat's timestamp, or fall
                // back to start_time for the last beat.
                let end_time = self
                    .beats
                    .get(i + 1)
                    .map(|next| next.timestamp)
                    .unwrap_or(b.timestamp);
                PipelineBeat {
                    beat_index: b.id,
                    start_time: b.timestamp,
                    end_time,
                    title: b.title.clone(),
                    summary: b.summary.clone(),
                }
            })
            .collect()
    }

    async fn sweep(&mut self) -> Result<u32> {
        Ok(0)
    }

    async fn finalize(&mut self) -> Result<()> {
        // Evaluate any remaining pending segments
        if !self.pending_segments.is_empty() {
            tracing::debug!(
                pending = self.pending_segments.len(),
                "finalizing remaining beats"
            );

            match self.evaluate_beats().await {
                Ok(new_beats) => {
                    self.assign_beats_to_pending(&new_beats);

                    for pending in &self.pending_segments {
                        if let Some(stored) = self
                            .all_segments
                            .iter_mut()
                            .find(|s| s.id == pending.id)
                        {
                            stored.beat_id = pending.beat_id;
                        }
                    }

                    self.beats.extend(new_beats);
                    self.pending_segments.clear();
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "beat finalization failed"
                    );
                    // Assign remaining segments to last beat or create default
                    let beat_id = if let Some(last) = self.beats.last() {
                        last.id
                    } else {
                        let beat = Beat {
                            id: self.next_beat_id,
                            timestamp: self
                                .pending_segments
                                .first()
                                .map(|s| s.start_time)
                                .unwrap_or(0.0),
                            title: "Session".into(),
                            summary: "Session content".into(),
                        };
                        self.next_beat_id += 1;
                        let id = beat.id;
                        self.beats.push(beat);
                        id
                    };
                    for seg in &mut self.pending_segments {
                        seg.beat_id = Some(beat_id);
                    }
                    self.pending_segments.clear();
                }
            }
        }

        tracing::info!(
            beats = self.beats.len(),
            "beat operator finalized"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(start: f32, end: f32, speaker: &str, text: &str) -> TranscriptSegment {
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
            beat_id: None,
            chunk_group: None,
            talk_type: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[test]
    fn silence_gap_triggers_evaluation() {
        let config = BeatConfig {
            silence_trigger: 8.0,
            chunk_seconds: 300.0,
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);
        op.last_end = Some(50.0);
        op.pending_segments
            .push(make_segment(40.0, 50.0, "gm", "Hello"));

        let seg = make_segment(60.0, 65.0, "gm", "You arrive.");
        assert!(op.should_evaluate(&seg));
    }

    #[test]
    fn small_gap_does_not_trigger() {
        let config = BeatConfig {
            silence_trigger: 8.0,
            chunk_seconds: 300.0,
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);
        op.last_end = Some(50.0);
        op.pending_segments
            .push(make_segment(40.0, 50.0, "gm", "Hello"));

        let seg = make_segment(53.0, 58.0, "gm", "Continue.");
        assert!(!op.should_evaluate(&seg));
    }

    #[test]
    fn time_based_trigger() {
        let config = BeatConfig {
            silence_trigger: 8.0,
            chunk_seconds: 300.0,
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);
        op.last_end = Some(300.0);
        op.last_eval_time = Some(0.0);
        op.pending_segments
            .push(make_segment(100.0, 200.0, "gm", "Hello"));

        // 310s since last eval (0.0), exceeds chunk_seconds (300)
        let seg = make_segment(310.0, 315.0, "gm", "Next part.");
        assert!(op.should_evaluate(&seg));
    }

    #[test]
    fn narration_cue_triggers() {
        let config = BeatConfig {
            gm_speaker_id: Some("gm".into()),
            chunk_seconds: 300.0,
            silence_trigger: 8.0,
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);
        op.last_end = Some(50.0);
        op.pending_segments
            .push(make_segment(40.0, 50.0, "gm", "Hello"));

        let seg = make_segment(52.0, 58.0, "gm", "Meanwhile, back at the tavern...");
        assert!(op.should_evaluate(&seg));
    }

    #[test]
    fn narration_cue_ignores_non_gm() {
        let config = BeatConfig {
            gm_speaker_id: Some("gm".into()),
            chunk_seconds: 300.0,
            silence_trigger: 8.0,
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);
        op.last_end = Some(50.0);
        op.pending_segments
            .push(make_segment(40.0, 50.0, "gm", "Hello"));

        let seg = make_segment(52.0, 58.0, "player1", "Meanwhile I go shopping.");
        assert!(!op.should_evaluate(&seg));
    }

    #[test]
    fn parse_beat_response_single() {
        let config = BeatConfig::default();
        let mut op = BeatOperator::new(config);
        op.pending_segments
            .push(make_segment(0.0, 60.0, "gm", "stuff"));

        let raw = "BEAT at 0:00 | Arrival | The party arrives at the dungeon entrance";
        let beats = op.parse_beat_response(raw);

        assert_eq!(beats.len(), 1);
        assert_eq!(beats[0].title, "Arrival");
        assert_eq!(beats[0].timestamp, 0.0);
        assert!(beats[0].summary.contains("dungeon"));
    }

    #[test]
    fn parse_beat_response_multiple() {
        let config = BeatConfig::default();
        let mut op = BeatOperator::new(config);
        op.pending_segments
            .push(make_segment(0.0, 300.0, "gm", "stuff"));

        let raw = "\
BEAT at 0:00 | Arrival | The party arrives at the inn
BEAT at 2:30 | Barkeep | They talk to the barkeep about rumors
BEAT at 4:15 | Fight | A bar fight breaks out";
        let beats = op.parse_beat_response(raw);

        assert_eq!(beats.len(), 3);
        assert_eq!(beats[0].title, "Arrival");
        assert_eq!(beats[0].timestamp, 0.0);
        assert_eq!(beats[1].title, "Barkeep");
        assert_eq!(beats[1].timestamp, 150.0);
        assert_eq!(beats[2].title, "Fight");
        assert_eq!(beats[2].timestamp, 255.0);
    }

    #[test]
    fn parse_beat_response_with_code_fences() {
        let config = BeatConfig::default();
        let mut op = BeatOperator::new(config);
        op.pending_segments
            .push(make_segment(0.0, 60.0, "gm", "stuff"));

        let raw = "```\nBEAT at 1:00 | Combat | The goblins attack\n```";
        let beats = op.parse_beat_response(raw);

        assert_eq!(beats.len(), 1);
        assert_eq!(beats[0].title, "Combat");
    }

    #[test]
    fn parse_beat_response_fallback_on_garbage() {
        let config = BeatConfig::default();
        let mut op = BeatOperator::new(config);
        op.pending_segments
            .push(make_segment(0.0, 60.0, "gm", "stuff"));

        let raw = "I don't understand the format you want";
        let beats = op.parse_beat_response(raw);

        // Should get a default "Continuation" beat
        assert_eq!(beats.len(), 1);
        assert_eq!(beats[0].title, "Continuation");
    }

    #[test]
    fn strip_code_fences_works() {
        assert_eq!(strip_code_fences("hello"), "hello");
        assert_eq!(strip_code_fences("```\nhello\n```"), "hello");
        assert_eq!(strip_code_fences("```text\nhello\n```"), "hello");
        assert_eq!(
            strip_code_fences("```markdown\nhello\n```"),
            "hello"
        );
    }

    #[test]
    fn empty_pending_does_not_trigger() {
        let config = BeatConfig::default();
        let op = BeatOperator::new(config);

        let seg = make_segment(500.0, 505.0, "gm", "Stuff");
        assert!(!op.should_evaluate(&seg));
    }

    #[tokio::test]
    async fn segments_get_beat_ids_on_finalize() {
        // Without a real LLM, finalize falls back to a default beat
        let config = BeatConfig {
            endpoint: "http://localhost:99999/nope".into(),
            ..Default::default()
        };
        let mut op = BeatOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0, "gm", "You enter the room.");
        let mut s2 = make_segment(6.0, 10.0, "player", "I look around.");

        op.on_segment(&mut s1).await;
        op.on_segment(&mut s2).await;
        let _ = op.finalize().await;

        // After finalize, the operator should have at least one beat
        assert!(!op.beats().is_empty());
    }
}
