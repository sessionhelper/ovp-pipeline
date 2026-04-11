//! chronicle-pipeline: audio transcription pipeline for TTRPG voice sessions.
//!
//! Pure audio processing library. Takes raw PCM audio bytes per speaker,
//! runs VAD, sends audio to an external Whisper HTTP endpoint for
//! transcription, applies operators (hallucination detection + scene
//! chunking), and returns structured transcript segments.
//!
//! **No I/O except HTTP to Whisper.** No S3, no Postgres, no file system.
//! Caller provides audio bytes, crate returns objects.

pub mod ad;
pub mod audio;
pub mod error;
pub mod operators;
pub mod pipeline;
pub mod streaming;
pub mod transcribe;
pub mod types;
pub mod vad;

// Re-export the main public API at the crate root for convenience.
pub use error::{PipelineError, Result};
pub use operators::{default_operators, operators_with_llm_scene, OperatorResult, Operator};
pub use pipeline::{process_session, PipelineConfig};
pub use streaming::{StreamingConfig, StreamingPipeline};
pub use transcribe::TranscriberConfig;
pub use types::*;
pub use vad::{VadConfig, VadSession};
