//! ovp-pipeline: audio transcription pipeline for TTRPG voice sessions.
//!
//! Pure audio processing library. Takes raw PCM audio bytes per speaker,
//! runs VAD, sends audio to an external Whisper HTTP endpoint for
//! transcription, applies filters (hallucination detection + scene
//! chunking), and returns structured transcript segments.
//!
//! **No I/O except HTTP to Whisper.** No S3, no Postgres, no file system.
//! Caller provides audio bytes, crate returns objects.

pub mod ad;
pub mod audio;
pub mod error;
pub mod filters;
pub mod pipeline;
pub mod transcribe;
pub mod types;
pub mod vad;

// Re-export the main public API at the crate root for convenience.
pub use error::{PipelineError, Result};
pub use filters::{default_filters, FilterResult, StreamFilter};
pub use pipeline::{process_session, PipelineConfig};
pub use transcribe::TranscriberConfig;
pub use types::*;
pub use vad::VadConfig;
