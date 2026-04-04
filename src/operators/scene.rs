//! Context-aware scene segmentation via LLM.
//!
//! Watches the transcript stream for scene-break signals (silence gaps,
//! speaker pattern shifts, GM narration cues). When a trigger fires,
//! sends the recent context to an LLM endpoint for evaluation.
//! The LLM decides: is this a scene break? If yes, names and summarizes it.
//!
//! No mechanical fallback — if the LLM says it's one scene, it's one scene.
//! A 3-hour session with one continuous conversation is one scene.

use crate::error::{PipelineError, Result};
use crate::types::TranscriptSegment;

use super::{Operator, OperatorResult};

use std::collections::HashSet;

/// Configuration for the context-aware scene operator.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// LLM HTTP endpoint (OpenAI-compatible chat completions).
    pub endpoint: String,
    /// Model name to request.
    pub model: String,
    /// Minimum silence gap (seconds) to trigger LLM evaluation.
    pub silence_trigger: f32,
    /// Number of recent segments to include as context for the LLM.
    pub context_window: usize,
    /// GM speaker pseudo_id — used for narration cue detection.
    pub gm_speaker_id: Option<String>,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434/v1/chat/completions".into(),
            model: "llama3".into(),
            silence_trigger: 10.0,
            context_window: 30,
            gm_speaker_id: None,
        }
    }
}

/// LLM response for a scene break evaluation.
#[derive(Debug, Clone, serde::Deserialize)]
struct SceneEvaluation {
    is_scene_break: bool,
    scene_title: Option<String>,
    scene_summary: Option<String>,
}

/// GM narration phrases that suggest a scene transition.
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

/// Context-aware scene operator.
pub struct SceneOperator {
    config: SceneConfig,
    /// Buffer of recent segments for LLM context.
    buffer: Vec<TranscriptSegment>,
    /// Current scene index.
    current_scene: u32,
    /// End time of the last non-excluded segment.
    last_end: Option<f32>,
    /// Speakers active in the current scene.
    active_speakers: HashSet<String>,
    /// Speakers active in a sliding window (last N segments).
    recent_speakers: Vec<String>,
    /// Total scenes created.
    scenes_created: u32,
    /// HTTP client (reused across calls).
    #[cfg(feature = "transcribe")]
    client: reqwest::Client,
}

impl SceneOperator {
    pub fn new(config: SceneConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            current_scene: 0,
            last_end: None,
            active_speakers: HashSet::new(),
            recent_speakers: Vec::new(),
            scenes_created: 0,
            #[cfg(feature = "transcribe")]
            client: reqwest::Client::new(),
        }
    }

    /// Check cheap heuristics for scene-break signals.
    fn check_triggers(&self, segment: &TranscriptSegment) -> Vec<String> {
        let mut triggers = Vec::new();

        // 1. Silence gap
        if let Some(last_end) = self.last_end {
            let gap = segment.start_time - last_end;
            if gap >= self.config.silence_trigger {
                triggers.push(format!("silence_gap:{:.0}s", gap));
            }
        }

        // 2. Speaker pattern shift — new speaker not in the recent window
        if self.buffer.len() > 5 {
            let recent: HashSet<&str> = self.buffer[self.buffer.len().saturating_sub(10)..]
                .iter()
                .map(|s| s.speaker_pseudo_id.as_str())
                .collect();
            if !recent.contains(segment.speaker_pseudo_id.as_str())
                && self.active_speakers.len() > 1
            {
                triggers.push("speaker_shift".into());
            }
        }

        // 3. GM narration cues
        if let Some(ref gm_id) = self.config.gm_speaker_id {
            if segment.speaker_pseudo_id == *gm_id {
                let lower = segment.text.to_lowercase();
                for cue in NARRATION_CUES {
                    if lower.contains(cue) {
                        triggers.push(format!("narration_cue:{}", cue));
                        break;
                    }
                }
            }
        }

        triggers
    }

    /// Format the buffer as context for the LLM prompt.
    fn format_context(&self) -> String {
        let start = self.buffer.len().saturating_sub(self.config.context_window);
        self.buffer[start..]
            .iter()
            .map(|s| {
                format!(
                    "[{:.0}s] {}: {}",
                    s.start_time, s.speaker_pseudo_id, s.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Call the LLM to evaluate whether this is a scene break.
    #[cfg(feature = "transcribe")]
    async fn evaluate_scene_break(
        &self,
        triggers: &[String],
        segment: &TranscriptSegment,
    ) -> Result<SceneEvaluation> {
        let context = self.format_context();
        let trigger_desc = triggers.join(", ");

        let prompt = format!(
            r#"You are analyzing a TTRPG session transcript for scene boundaries.

RECENT TRANSCRIPT:
{}

NEXT LINE:
[{:.0}s] {}: {}

TRIGGER SIGNALS: {}

A "scene" is a narrative unit — a location, a conversation, an encounter. Scene breaks happen when the story moves to a new location, time skips forward, new characters enter, or the GM transitions to a new situation.

Rules:
- If the conversation is continuing on the same topic with the same people, this is NOT a scene break.
- A pause where the GM is reading notes or players are discussing mechanics is NOT a scene break.
- Only mark a scene break if the narrative is genuinely shifting.

Respond with ONLY valid JSON, no other text:
{{"is_scene_break": true/false, "scene_title": "short title if break", "scene_summary": "one sentence summary if break"}}"#,
            context,
            segment.start_time,
            segment.speaker_pseudo_id,
            segment.text,
            trigger_desc
        );

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150,
        });

        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| PipelineError::Filter(format!("LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PipelineError::Filter(format!(
                "LLM returned {}: {}",
                status, body
            )));
        }

        let resp: serde_json::Value = response
            .json()
            .await
            .map_err(|e| PipelineError::Filter(format!("LLM response parse failed: {}", e)))?;

        // Extract the assistant's message content
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                PipelineError::Filter("LLM response missing content".into())
            })?;

        // Parse the JSON from the LLM's response
        let eval: SceneEvaluation = serde_json::from_str(content.trim()).map_err(|e| {
            tracing::warn!(
                raw_content = content,
                error = %e,
                "LLM returned non-JSON, treating as no scene break"
            );
            PipelineError::Filter(format!("LLM JSON parse failed: {}", e))
        })?;

        Ok(eval)
    }

    /// Stub when transcribe feature is disabled.
    #[cfg(not(feature = "transcribe"))]
    async fn evaluate_scene_break(
        &self,
        _triggers: &[String],
        _segment: &TranscriptSegment,
    ) -> Result<SceneEvaluation> {
        Ok(SceneEvaluation {
            is_scene_break: false,
            scene_title: None,
            scene_summary: None,
        })
    }
}

#[async_trait::async_trait]
impl Operator for SceneOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        if segment.excluded {
            return OperatorResult::Pass;
        }

        // Initialize first scene
        if self.scenes_created == 0 {
            self.scenes_created = 1;
        }

        // Check cheap triggers
        let triggers = self.check_triggers(segment);

        if !triggers.is_empty() {
            tracing::debug!(
                triggers = ?triggers,
                time = segment.start_time,
                "scene break triggers fired, evaluating"
            );

            // Call LLM to decide
            match self.evaluate_scene_break(&triggers, segment).await {
                Ok(eval) => {
                    if eval.is_scene_break {
                        self.current_scene += 1;
                        self.scenes_created += 1;
                        self.active_speakers.clear();

                        tracing::info!(
                            scene = self.current_scene,
                            title = eval.scene_title.as_deref().unwrap_or("untitled"),
                            summary = eval.scene_summary.as_deref().unwrap_or(""),
                            time = segment.start_time,
                            triggers = ?triggers,
                            "scene break detected"
                        );
                    } else {
                        tracing::debug!(
                            time = segment.start_time,
                            triggers = ?triggers,
                            "LLM says no scene break"
                        );
                    }
                }
                Err(e) => {
                    // LLM failure is not fatal — log and continue without splitting
                    tracing::warn!(
                        error = %e,
                        time = segment.start_time,
                        "LLM evaluation failed, skipping scene break check"
                    );
                }
            }
        }

        // Assign scene group
        segment.chunk_group = Some(self.current_scene);

        // Update state
        self.last_end = Some(segment.end_time);
        self.active_speakers
            .insert(segment.speaker_pseudo_id.clone());
        self.recent_speakers
            .push(segment.speaker_pseudo_id.clone());
        self.buffer.push(segment.clone());

        // Trim buffer to 2x context window
        let max_buf = self.config.context_window * 2;
        if self.buffer.len() > max_buf {
            self.buffer.drain(..self.buffer.len() - max_buf);
        }

        OperatorResult::Pass
    }

    async fn sweep(&mut self) -> Result<u32> {
        Ok(0)
    }

    async fn finalize(&mut self) -> Result<()> {
        tracing::info!(
            scenes = self.scenes_created,
            buffer_size = self.buffer.len(),
            "context-aware scene operator finalized"
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
            chunk_group: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[test]
    fn silence_trigger_fires() {
        let config = SceneConfig {
            silence_trigger: 10.0,
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);
        op.last_end = Some(50.0);

        let seg = make_segment(65.0, 70.0, "gm", "You arrive at the inn.");
        let triggers = op.check_triggers(&seg);
        assert!(!triggers.is_empty());
        assert!(triggers[0].starts_with("silence_gap"));
    }

    #[test]
    fn no_trigger_on_small_gap() {
        let config = SceneConfig {
            silence_trigger: 10.0,
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);
        op.last_end = Some(50.0);

        let seg = make_segment(55.0, 60.0, "gm", "And then...");
        let triggers = op.check_triggers(&seg);
        assert!(triggers.is_empty());
    }

    #[test]
    fn narration_cue_trigger_fires() {
        let config = SceneConfig {
            gm_speaker_id: Some("gm".into()),
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);
        op.last_end = Some(50.0);

        let seg = make_segment(52.0, 58.0, "gm", "Meanwhile, back at the cottage...");
        let triggers = op.check_triggers(&seg);
        assert!(triggers.iter().any(|t| t.starts_with("narration_cue")));
    }

    #[test]
    fn narration_cue_ignores_non_gm() {
        let config = SceneConfig {
            gm_speaker_id: Some("gm".into()),
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);
        op.last_end = Some(50.0);

        let seg = make_segment(52.0, 58.0, "player", "Meanwhile I'll go to the shop.");
        let triggers = op.check_triggers(&seg);
        assert!(!triggers.iter().any(|t| t.starts_with("narration_cue")));
    }

    #[tokio::test]
    async fn segments_get_scene_groups() {
        // Without an LLM endpoint, triggers fire but evaluation fails gracefully
        let config = SceneConfig {
            silence_trigger: 10.0,
            endpoint: "http://localhost:99999/v1/chat/completions".into(),
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0, "gm", "You enter the room.");
        let mut s2 = make_segment(6.0, 10.0, "player", "I look around.");
        // Big gap but LLM unreachable — should stay same scene
        let mut s3 = make_segment(25.0, 30.0, "gm", "The room is empty.");

        op.on_segment(&mut s1).await;
        op.on_segment(&mut s2).await;
        op.on_segment(&mut s3).await;

        // All same scene because LLM call fails gracefully
        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        assert_eq!(s3.chunk_group, Some(0));
    }

    #[test]
    fn context_formatting() {
        let config = SceneConfig::default();
        let mut op = SceneOperator::new(config);
        op.buffer.push(make_segment(0.0, 5.0, "gm", "Hello."));
        op.buffer.push(make_segment(5.0, 10.0, "player", "Hi."));

        let ctx = op.format_context();
        assert!(ctx.contains("gm: Hello."));
        assert!(ctx.contains("player: Hi."));
    }
}
