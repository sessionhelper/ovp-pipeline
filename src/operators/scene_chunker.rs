//! Scene operator: assigns `chunk_group` to segments based on silence
//! gaps and duration limits.
//!
//! A new scene starts when either:
//! - The silence gap between consecutive segments exceeds `max_silence_gap`
//! - The current scene's duration exceeds `max_chunk_duration`

use crate::error::Result;
use crate::types::{PipelineScene, TranscriptSegment};

use super::{OperatorResult, Operator};

/// Configuration for scene boundary detection.
#[derive(Debug, Clone)]
pub struct SceneOperatorConfig {
    /// Maximum silence gap in seconds before starting a new scene.
    pub max_silence_gap: f32,
    /// Maximum duration of a single scene in seconds.
    pub max_chunk_duration: f32,
}

impl Default for SceneOperatorConfig {
    fn default() -> Self {
        Self {
            max_silence_gap: 30.0,
            max_chunk_duration: 600.0,
        }
    }
}

/// Scene operator. Assigns sequential `chunk_group` numbers to
/// transcript segments for UI organization.
pub struct SceneOperator {
    config: SceneOperatorConfig,
    /// Current scene index (0-based).
    current_group: u32,
    /// Start time of the current scene.
    scene_start: Option<f32>,
    /// End time of the most recent segment in the current scene.
    last_end: Option<f32>,
    /// Total number of scenes created.
    scenes_created: u32,
    /// Completed scene boundaries for `collect_scenes()`.
    completed_scenes: Vec<SceneBoundary>,
}

/// Internal record of a scene's time boundaries.
#[derive(Debug, Clone)]
struct SceneBoundary {
    index: u32,
    start_time: f32,
    end_time: f32,
}

impl SceneOperator {
    /// Create a new scene operator with the given configuration.
    pub fn new(config: SceneOperatorConfig) -> Self {
        Self {
            config,
            current_group: 0,
            scene_start: None,
            last_end: None,
            scenes_created: 0,
            completed_scenes: Vec::new(),
        }
    }

    /// Check whether we should start a new scene based on the gap since
    /// the last segment and the current scene's duration.
    fn should_split(&self, segment_start: f32) -> bool {
        let Some(last_end) = self.last_end else {
            return false; // First segment, no split needed
        };
        let Some(scene_start) = self.scene_start else {
            return false;
        };

        // Split on silence gap
        let gap = segment_start - last_end;
        if gap >= self.config.max_silence_gap {
            return true;
        }

        // Split on scene duration
        let scene_duration = segment_start - scene_start;
        if scene_duration >= self.config.max_chunk_duration {
            return true;
        }

        false
    }
}

#[async_trait::async_trait]
impl Operator for SceneOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        // Skip excluded segments — they don't affect scene boundaries
        if segment.excluded {
            return OperatorResult::Pass;
        }

        if self.scene_start.is_none() {
            // First segment starts the first scene
            self.scene_start = Some(segment.start_time);
            self.scenes_created = 1;
        } else if self.should_split(segment.start_time) {
            // Close the current scene before starting a new one
            if let (Some(start), Some(end)) = (self.scene_start, self.last_end) {
                self.completed_scenes.push(SceneBoundary {
                    index: self.current_group,
                    start_time: start,
                    end_time: end,
                });
            }

            // Start a new scene
            self.current_group += 1;
            self.scene_start = Some(segment.start_time);
            self.scenes_created += 1;

            tracing::debug!(
                group = self.current_group,
                start = segment.start_time,
                "new scene boundary"
            );
        }

        segment.chunk_group = Some(self.current_group);
        self.last_end = Some(segment.end_time);

        OperatorResult::Pass
    }

    async fn sweep(&mut self) -> Result<u32> {
        // Scene operator doesn't do retroactive analysis
        Ok(0)
    }

    fn collect_scenes(&self) -> Vec<PipelineScene> {
        self.completed_scenes
            .iter()
            .map(|sb| PipelineScene {
                scene_index: sb.index,
                start_time: sb.start_time,
                end_time: sb.end_time,
                title: format!("Scene {}", sb.index + 1),
                summary: String::new(),
                beat_start: 0,
                beat_end: 0,
            })
            .collect()
    }

    async fn finalize(&mut self) -> Result<()> {
        // Close the last open scene so it appears in collect_scenes()
        if let (Some(start), Some(end)) = (self.scene_start, self.last_end) {
            self.completed_scenes.push(SceneBoundary {
                index: self.current_group,
                start_time: start,
                end_time: end,
            });
        }

        tracing::info!(
            scenes = self.scenes_created,
            "scene operator finalized"
        );
        Ok(())
    }
}

impl SceneOperator {
    /// Return the total number of scenes detected so far.
    pub fn scenes_detected(&self) -> u32 {
        self.scenes_created
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(start: f32, end: f32) -> TranscriptSegment {
        TranscriptSegment {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            segment_index: 0,
            speaker_pseudo_id: "speaker_a".into(),
            start_time: start,
            end_time: end,
            text: "test".into(),
            original_text: "test".into(),
            confidence: None,
            beat_id: None,
            chunk_group: None,
            talk_type: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    #[tokio::test]
    async fn assigns_same_group_for_close_segments() {
        let mut chunker = SceneOperator::new(SceneOperatorConfig::default());

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(6.0, 10.0);
        let mut s3 = make_segment(11.0, 15.0);

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.on_segment(&mut s3).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        assert_eq!(s3.chunk_group, Some(0));
    }

    #[tokio::test]
    async fn splits_on_silence_gap() {
        let config = SceneOperatorConfig {
            max_silence_gap: 10.0,
            max_chunk_duration: 600.0,
        };
        let mut chunker = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(20.0, 25.0); // 15s gap > 10s threshold

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(1));
        assert_eq!(chunker.scenes_detected(), 2);
    }

    #[tokio::test]
    async fn splits_on_duration_limit() {
        let config = SceneOperatorConfig {
            max_silence_gap: 30.0,
            max_chunk_duration: 100.0,
        };
        let mut chunker = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 50.0);
        let mut s2 = make_segment(51.0, 90.0);
        let mut s3 = make_segment(101.0, 110.0); // scene duration > 100s

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.on_segment(&mut s3).await;

        assert_eq!(s1.chunk_group, Some(0));
        assert_eq!(s2.chunk_group, Some(0));
        assert_eq!(s3.chunk_group, Some(1));
    }

    #[tokio::test]
    async fn collect_scenes_returns_all_scenes_after_finalize() {
        let config = SceneOperatorConfig {
            max_silence_gap: 10.0,
            max_chunk_duration: 600.0,
        };
        let mut chunker = SceneOperator::new(config);

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(6.0, 10.0);
        // 15s gap triggers scene split
        let mut s3 = make_segment(25.0, 30.0);
        let mut s4 = make_segment(31.0, 35.0);

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.on_segment(&mut s3).await;
        chunker.on_segment(&mut s4).await;
        chunker.finalize().await.unwrap();

        let scenes = chunker.collect_scenes();
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0].scene_index, 0);
        assert_eq!(scenes[0].start_time, 0.0);
        assert_eq!(scenes[0].end_time, 10.0);
        assert_eq!(scenes[0].title, "Scene 1");
        assert_eq!(scenes[1].scene_index, 1);
        assert_eq!(scenes[1].start_time, 25.0);
        assert_eq!(scenes[1].end_time, 35.0);
        assert_eq!(scenes[1].title, "Scene 2");
    }

    #[tokio::test]
    async fn collect_scenes_single_scene_after_finalize() {
        let mut chunker = SceneOperator::new(SceneOperatorConfig::default());

        let mut s1 = make_segment(0.0, 5.0);
        let mut s2 = make_segment(6.0, 10.0);

        chunker.on_segment(&mut s1).await;
        chunker.on_segment(&mut s2).await;
        chunker.finalize().await.unwrap();

        let scenes = chunker.collect_scenes();
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].scene_index, 0);
        assert_eq!(scenes[0].start_time, 0.0);
        assert_eq!(scenes[0].end_time, 10.0);
    }

    #[tokio::test]
    async fn collect_scenes_empty_when_no_segments() {
        let mut chunker = SceneOperator::new(SceneOperatorConfig::default());
        chunker.finalize().await.unwrap();
        assert_eq!(chunker.collect_scenes().len(), 0);
    }
}
