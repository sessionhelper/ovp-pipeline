//! Scene grouping operator via LLM.
//!
//! Watches beat assignments from the BeatOperator and groups beats into
//! higher-level scenes. Every time a new beat appears, re-evaluates the
//! full beat list via the LLM to produce scene groupings.
//!
//! This operator does NOT detect beats — it only groups them. The
//! BeatOperator handles fine-grained beat detection.

use crate::error::{PipelineError, Result};
use crate::types::{PipelineScene, TranscriptSegment};

use super::{Operator, OperatorResult};

use regex::Regex;
use std::collections::HashMap;

/// Configuration for the scene grouping operator.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// LLM HTTP endpoint (OpenAI-compatible chat completions).
    pub endpoint: String,
    /// Model name to request.
    pub model: String,
    /// Minimum silence gap — kept for backward compat, not used directly.
    pub silence_trigger: f32,
    /// Context window size — kept for backward compat, not used directly.
    pub context_window: usize,
    /// GM speaker pseudo_id.
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

/// An internal beat record for scene grouping.
#[derive(Debug, Clone)]
struct BeatRecord {
    beat_id: u32,
    /// Approximate timestamp.
    timestamp: f32,
    /// Segments that belong to this beat.
    segment_count: u32,
}

/// A scene grouping result from the LLM.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SceneGroup {
    /// Scene title.
    title: String,
    /// Summary.
    summary: String,
    /// Beat IDs included in this scene (inclusive range).
    beat_start: u32,
    beat_end: u32,
}

/// Scene grouping operator.
pub struct SceneOperator {
    config: SceneConfig,
    /// Known beats, ordered by ID.
    known_beats: Vec<BeatRecord>,
    /// Mapping from beat_id to scene_id (0-based).
    beat_to_scene: HashMap<u32, u32>,
    /// Current scene groupings.
    scene_groups: Vec<SceneGroup>,
    /// Total scenes created.
    scenes_created: u32,
    /// HTTP client.
    #[cfg(feature = "transcribe")]
    client: reqwest::Client,
}

impl SceneOperator {
    pub fn new(config: SceneConfig) -> Self {
        Self {
            config,
            known_beats: Vec::new(),
            beat_to_scene: HashMap::new(),
            scene_groups: Vec::new(),
            scenes_created: 0,
            #[cfg(feature = "transcribe")]
            client: reqwest::Client::new(),
        }
    }

    /// Return number of scenes detected.
    pub fn scenes_detected(&self) -> u32 {
        self.scenes_created
    }

    /// Check if a beat_id is new (not yet tracked).
    fn is_new_beat(&self, beat_id: u32) -> bool {
        !self.known_beats.iter().any(|b| b.beat_id == beat_id)
    }

    /// Register a new beat.
    fn register_beat(&mut self, beat_id: u32, timestamp: f32) {
        if self.is_new_beat(beat_id) {
            self.known_beats.push(BeatRecord {
                beat_id,
                timestamp,
                segment_count: 1,
            });
        } else {
            // Increment segment count
            if let Some(b) = self.known_beats.iter_mut().find(|b| b.beat_id == beat_id) {
                b.segment_count += 1;
            }
        }
    }

    /// Format the beat list for the LLM prompt.
    fn format_beat_list(&self) -> String {
        self.known_beats
            .iter()
            .map(|b| {
                let minutes = (b.timestamp / 60.0) as u32;
                let seconds = (b.timestamp % 60.0) as u32;
                format!(
                    "Beat {} at {}:{:02} ({} segments)",
                    b.beat_id, minutes, seconds, b.segment_count
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Call the LLM to group beats into scenes.
    #[cfg(feature = "transcribe")]
    async fn evaluate_scene_grouping(&mut self) -> Result<Vec<SceneGroup>> {
        if self.known_beats.is_empty() {
            return Ok(Vec::new());
        }

        let beat_list = self.format_beat_list();

        let prompt = format!(
            "TTRPG session beats:\n\n{}\n\n\
             Group these beats into scenes. A scene is a continuous location + situation.\n\
             One per line:\n\
             SCENE: Title | beats N-M | One sentence summary\n\n\
             If only one scene so far:\n\
             SCENE: Title | beats 0-0 | Summary",
            beat_list
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
            .map_err(|e| PipelineError::Filter(format!("Scene LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(PipelineError::Filter(format!(
                "Scene LLM returned {}: {}",
                status, body_text
            )));
        }

        let resp: serde_json::Value = response
            .json()
            .await
            .map_err(|e| {
                PipelineError::Filter(format!("Scene LLM response parse failed: {}", e))
            })?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| PipelineError::Filter("Scene LLM response missing content".into()))?;

        let groups = Self::parse_scene_response(content, &self.known_beats);
        Ok(groups)
    }

    /// Stub when transcribe feature is disabled.
    #[cfg(not(feature = "transcribe"))]
    async fn evaluate_scene_grouping(&mut self) -> Result<Vec<SceneGroup>> {
        Ok(Vec::new())
    }

    /// Parse the LLM scene grouping response.
    fn parse_scene_response(raw: &str, known_beats: &[BeatRecord]) -> Vec<SceneGroup> {
        let cleaned = strip_code_fences(raw);
        let re = Regex::new(r"(?i)SCENE:\s*(.+?)\s*\|\s*beats?\s+(\d+)\s*-\s*(\d+)\s*\|\s*(.+)")
            .expect("invalid scene regex");

        let mut groups = Vec::new();
        for line in cleaned.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(caps) = re.captures(trimmed) {
                let title = caps[1].trim().to_string();
                let beat_start: u32 = caps[2].parse().unwrap_or(0);
                let beat_end: u32 = caps[3].parse().unwrap_or(0);
                let summary = caps[4].trim().to_string();

                groups.push(SceneGroup {
                    title,
                    summary,
                    beat_start,
                    beat_end,
                });
            }
        }

        if groups.is_empty() && !known_beats.is_empty() {
            tracing::warn!(
                raw_response = raw,
                "scene LLM returned no parseable scenes, creating default scene"
            );
            let min_beat = known_beats.first().map(|b| b.beat_id).unwrap_or(0);
            let max_beat = known_beats.last().map(|b| b.beat_id).unwrap_or(0);
            groups.push(SceneGroup {
                title: "Session".into(),
                summary: "Ongoing session".into(),
                beat_start: min_beat,
                beat_end: max_beat,
            });
        }

        groups
    }

    /// Rebuild the beat_to_scene mapping from current scene groups.
    fn rebuild_mapping(&mut self) {
        self.beat_to_scene.clear();
        for (scene_idx, group) in self.scene_groups.iter().enumerate() {
            for beat_id in group.beat_start..=group.beat_end {
                self.beat_to_scene.insert(beat_id, scene_idx as u32);
            }
        }
        self.scenes_created = self.scene_groups.len() as u32;
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
impl Operator for SceneOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        if segment.excluded {
            return OperatorResult::Pass;
        }

        // If this segment has a beat_id, register it
        if let Some(beat_id) = segment.beat_id {
            let is_new = self.is_new_beat(beat_id);
            self.register_beat(beat_id, segment.start_time);

            if is_new && self.known_beats.len() > 1 {
                // New beat appeared — re-evaluate scene groupings
                tracing::debug!(
                    beat_id = beat_id,
                    total_beats = self.known_beats.len(),
                    "new beat detected, re-evaluating scene groupings"
                );

                match self.evaluate_scene_grouping().await {
                    Ok(groups) => {
                        if !groups.is_empty() {
                            tracing::info!(
                                scenes = groups.len(),
                                "scene groupings updated"
                            );
                            self.scene_groups = groups;
                            self.rebuild_mapping();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "scene grouping evaluation failed"
                        );
                    }
                }
            } else if is_new && self.known_beats.len() == 1 {
                // First beat — create initial scene
                self.scene_groups = vec![SceneGroup {
                    title: "Opening".into(),
                    summary: "Session begins".into(),
                    beat_start: beat_id,
                    beat_end: beat_id,
                }];
                self.rebuild_mapping();
            }

            // Assign chunk_group based on the mapping
            if let Some(&scene_id) = self.beat_to_scene.get(&beat_id) {
                segment.chunk_group = Some(scene_id);
            } else {
                // Beat not yet mapped — assign to the last known scene
                let last_scene = if self.scenes_created > 0 {
                    self.scenes_created - 1
                } else {
                    0
                };
                segment.chunk_group = Some(last_scene);
            }
        } else {
            // No beat_id — assign to the last known scene
            let last_scene = if self.scenes_created > 0 {
                self.scenes_created - 1
            } else {
                0
            };
            segment.chunk_group = Some(last_scene);
        }

        OperatorResult::Pass
    }

    fn collect_scenes(&self) -> Vec<PipelineScene> {
        self.scene_groups
            .iter()
            .enumerate()
            .map(|(i, group)| {
                // Derive scene time range from known beat timestamps.
                let start_time = self
                    .known_beats
                    .iter()
                    .find(|b| b.beat_id == group.beat_start)
                    .map(|b| b.timestamp)
                    .unwrap_or(0.0);
                let end_time = self
                    .known_beats
                    .iter()
                    .find(|b| b.beat_id == group.beat_end)
                    .map(|b| b.timestamp)
                    .unwrap_or(start_time);
                PipelineScene {
                    scene_index: i as u32,
                    start_time,
                    end_time,
                    title: group.title.clone(),
                    summary: group.summary.clone(),
                    beat_start: group.beat_start,
                    beat_end: group.beat_end,
                }
            })
            .collect()
    }

    async fn sweep(&mut self) -> Result<u32> {
        Ok(0)
    }

    async fn finalize(&mut self) -> Result<()> {
        // Final scene grouping evaluation
        if self.known_beats.len() > 1 {
            match self.evaluate_scene_grouping().await {
                Ok(groups) => {
                    if !groups.is_empty() {
                        self.scene_groups = groups;
                        self.rebuild_mapping();
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "final scene grouping evaluation failed"
                    );
                }
            }
        }

        tracing::info!(
            scenes = self.scenes_created,
            beats = self.known_beats.len(),
            "scene grouping operator finalized"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(
        start: f32,
        end: f32,
        speaker: &str,
        text: &str,
        beat_id: Option<u32>,
    ) -> TranscriptSegment {
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
            beat_id,
            chunk_group: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[test]
    fn parse_scene_response_single() {
        let beats = vec![BeatRecord {
            beat_id: 0,
            timestamp: 0.0,
            segment_count: 5,
        }];
        let raw = "SCENE: Tavern Arrival | beats 0-0 | The party arrives at the tavern";
        let groups = SceneOperator::parse_scene_response(raw, &beats);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].title, "Tavern Arrival");
        assert_eq!(groups[0].beat_start, 0);
        assert_eq!(groups[0].beat_end, 0);
    }

    #[test]
    fn parse_scene_response_multiple() {
        let beats = vec![
            BeatRecord { beat_id: 0, timestamp: 0.0, segment_count: 5 },
            BeatRecord { beat_id: 1, timestamp: 150.0, segment_count: 3 },
            BeatRecord { beat_id: 2, timestamp: 300.0, segment_count: 4 },
            BeatRecord { beat_id: 3, timestamp: 500.0, segment_count: 6 },
        ];
        let raw = "\
SCENE: Tavern | beats 0-1 | Party gathers info at the tavern
SCENE: Dungeon Entrance | beats 2-3 | Party travels to and enters the dungeon";
        let groups = SceneOperator::parse_scene_response(raw, &beats);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].beat_start, 0);
        assert_eq!(groups[0].beat_end, 1);
        assert_eq!(groups[1].beat_start, 2);
        assert_eq!(groups[1].beat_end, 3);
    }

    #[test]
    fn parse_scene_response_with_code_fences() {
        let beats = vec![BeatRecord {
            beat_id: 0,
            timestamp: 0.0,
            segment_count: 5,
        }];
        let raw = "```\nSCENE: Combat | beats 0-0 | Goblins attack\n```";
        let groups = SceneOperator::parse_scene_response(raw, &beats);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].title, "Combat");
    }

    #[test]
    fn parse_scene_response_fallback_on_garbage() {
        let beats = vec![BeatRecord {
            beat_id: 0,
            timestamp: 0.0,
            segment_count: 5,
        }];
        let raw = "This is nonsense output from the LLM";
        let groups = SceneOperator::parse_scene_response(raw, &beats);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].title, "Session");
    }

    #[tokio::test]
    async fn segments_with_beats_get_scene_groups() {
        let config = SceneConfig {
            endpoint: "http://localhost:99999/nope".into(),
            ..Default::default()
        };
        let mut op = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0, "gm", "You enter the room.", Some(0));
        let mut s2 = make_segment(6.0, 10.0, "player", "I look around.", Some(0));
        let mut s3 = make_segment(300.0, 305.0, "gm", "Later, at the inn.", Some(1));

        op.on_segment(&mut s1).await;
        op.on_segment(&mut s2).await;
        op.on_segment(&mut s3).await;

        // s1 and s2 belong to beat 0 -> scene 0
        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        // s3 has beat 1, LLM fails, falls back to last scene
        assert!(s3.chunk_group.is_some());
    }

    #[tokio::test]
    async fn segments_without_beats_get_default_scene() {
        let config = SceneConfig::default();
        let mut op = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0, "gm", "Hello.", None);
        op.on_segment(&mut s1).await;

        assert_eq!(s1.chunk_group, Some(0));
    }

    #[test]
    fn beat_registration() {
        let config = SceneConfig::default();
        let mut op = SceneOperator::new(config);

        op.register_beat(0, 0.0);
        op.register_beat(0, 5.0); // duplicate
        op.register_beat(1, 150.0);

        assert_eq!(op.known_beats.len(), 2);
        assert_eq!(op.known_beats[0].segment_count, 2);
        assert_eq!(op.known_beats[1].segment_count, 1);
    }

    #[test]
    fn strip_code_fences_works() {
        assert_eq!(strip_code_fences("hello"), "hello");
        assert_eq!(strip_code_fences("```\nhello\n```"), "hello");
        assert_eq!(strip_code_fences("```text\nhello\n```"), "hello");
    }
}
