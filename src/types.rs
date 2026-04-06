//! Core pipeline types.
//!
//! Data structures that flow between pipeline stages. Each stage takes
//! typed input and returns typed output — no shared mutable state.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single speaker's mono PCM audio stream.
///
/// The caller is responsible for byte decoding, stereo downmix, and
/// any format conversion before constructing this. The pipeline
/// receives mono f32 samples and a sample rate — nothing else.
#[derive(Debug, Clone)]
pub struct SpeakerTrack {
    /// Opaque speaker identifier.
    pub pseudo_id: String,
    /// Mono f32 audio samples in [-1.0, 1.0] range.
    pub samples: Vec<f32>,
    /// Sample rate of these samples (e.g. 48000 from Discord).
    pub sample_rate: u32,
}

/// Input to the pipeline: a session's worth of speaker tracks.
#[derive(Debug, Clone)]
pub struct SessionInput {
    /// Unique session identifier.
    pub session_id: Uuid,
    /// Per-speaker mono PCM streams to process.
    pub tracks: Vec<SpeakerTrack>,
}

/// A contiguous region of detected speech within a speaker's audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRegion {
    /// Speaker who produced this speech.
    pub speaker: String,
    /// Start time in seconds (absolute within the session).
    pub start: f32,
    /// End time in seconds (absolute within the session).
    pub end: f32,
}

/// An extracted audio chunk ready for transcription.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Speaker who produced this audio.
    pub speaker: String,
    /// Mono f32 samples at 16kHz.
    pub samples: Vec<f32>,
    /// Sample rate (always 16000).
    pub sample_rate: u32,
    /// Absolute start time within the session, in seconds.
    pub original_start: f32,
    /// Absolute end time within the session, in seconds.
    pub original_end: f32,
}

// SpeakerTrack is used throughout the pipeline where SpeakerSamples
// was previously used. This alias keeps internal code readable.
pub type SpeakerSamples = SpeakerTrack;

/// A single transcript segment produced by Whisper and processed by operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Unique segment identifier.
    pub id: Uuid,
    /// Session this segment belongs to.
    pub session_id: Uuid,
    /// Ordering index within the session.
    pub segment_index: u32,
    /// Speaker who said this.
    pub speaker_pseudo_id: String,
    /// Absolute start time in seconds.
    pub start_time: f32,
    /// Absolute end time in seconds.
    pub end_time: f32,
    /// Text after operator processing. May differ from `original_text`.
    pub text: String,
    /// Immutable Whisper output, preserved for audit/debugging.
    pub original_text: String,
    /// Whisper confidence score, if available.
    pub confidence: Option<f32>,
    /// Beat identifier assigned by the beat operator.
    #[serde(default)]
    pub beat_id: Option<u32>,
    /// Scene/chunk group assigned by the scene operator.
    pub chunk_group: Option<u32>,
    /// Whether this segment was excluded by an operator.
    pub excluded: bool,
    /// Reason for exclusion, if excluded.
    pub exclude_reason: Option<String>,
}

/// A narrative beat detected by the BeatOperator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineBeat {
    /// Beat index (0-based, sequential).
    pub beat_index: u32,
    /// Approximate start time in seconds.
    pub start_time: f32,
    /// End time in seconds. Set to start_time when no better estimate is available.
    pub end_time: f32,
    /// Short title for the beat.
    pub title: String,
    /// One-sentence summary.
    pub summary: String,
}

/// A scene grouping produced by the SceneOperator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineScene {
    /// Scene index (0-based, sequential).
    pub scene_index: u32,
    /// Start time in seconds (from the earliest beat in this scene).
    pub start_time: f32,
    /// End time in seconds (from the latest beat in this scene).
    pub end_time: f32,
    /// Scene title.
    pub title: String,
    /// One-sentence summary.
    pub summary: String,
    /// First beat index included in this scene (inclusive).
    pub beat_start: u32,
    /// Last beat index included in this scene (inclusive).
    pub beat_end: u32,
}

/// Final output of the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// All transcript segments (including excluded ones).
    pub segments: Vec<TranscriptSegment>,
    /// Narrative beats detected by the beat operator.
    pub beats: Vec<PipelineBeat>,
    /// Scene groupings produced by the scene operator.
    pub scenes: Vec<PipelineScene>,
    /// Number of segments that passed operators.
    pub segments_produced: u32,
    /// Number of segments excluded by operators.
    pub segments_excluded: u32,
    /// Number of scenes detected by the scene chunker.
    pub scenes_detected: u32,
    /// Total audio duration processed, in seconds.
    pub duration_processed: f32,
}
