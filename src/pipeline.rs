//! Pipeline orchestration.
//!
//! Chains streaming stages: resample → RMS audio detection →
//! Silero VAD → Whisper transcription → operators. Input is mono f32
//! samples per speaker — byte decoding and downmix are the caller's
//! responsibility.

use std::time::Instant;

use crate::ad::{self, RmsConfig};
use crate::audio::resample;
use crate::error::Result;
use crate::operators::{self, Operator};
use crate::transcribe::{self, TranscriberConfig};
use crate::types::{PipelineResult, SessionInput};
use crate::vad::{self, VadConfig};

/// Top-level pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// RMS audio detection (tier 1 — silence gate).
    pub rms: RmsConfig,
    /// VAD configuration (tier 2 — speech vs non-speech).
    pub vad: VadConfig,
    /// Whisper transcription configuration.
    pub whisper: TranscriberConfig,
    /// Minimum speech region duration in seconds. Regions shorter
    /// than this are dropped before transcription — the "pop filter"
    /// that catches blips, breaths, and mic noise pre-Whisper.
    pub min_chunk_duration: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            rms: RmsConfig::default(),
            vad: VadConfig::default(),
            whisper: TranscriberConfig {
                endpoint: "http://localhost:8080/v1/audio/transcriptions".into(),
                model: "large-v3-turbo".into(),
                language: Some("en".into()),
                initial_prompt: Some(
                    "TTRPG session dialogue. Multiple speakers discussing \
                     combat, exploration, and roleplay."
                        .into(),
                ),
                beam_size: 5,
                temperature: vec![0.0, 0.2, 0.4],
                hallucination_logprob_threshold: -0.85,
                hallucination_no_speech_threshold: 0.5,
                hallucination_compression_ratio: 1.8,
            },
            min_chunk_duration: 1.0,
        }
    }
}

/// Process a completed recording session end-to-end.
///
/// Audio flows through: resample (to 16kHz) → RMS audio detection
/// (silence removed) → Silero VAD (non-speech removed) → Whisper
/// (speech → text) → operators (hallucination detection, scene chunking).
/// Input is already mono f32 samples — caller handles byte decoding.
pub async fn process_session(
    config: &PipelineConfig,
    input: SessionInput,
    operators: &mut [Box<dyn Operator>],
) -> Result<PipelineResult> {
    let pipeline_start = Instant::now();
    let session_id = input.session_id;
    let speaker_count = input.tracks.len();

    let input_duration: f32 = input.tracks.iter()
        .map(|t| t.samples.len() as f32 / t.sample_rate as f32)
        .sum();

    tracing::info!(
        session_id = %session_id,
        speakers = speaker_count,
        input_duration_secs = input_duration,
        "pipeline started"
    );

    // --- Resample ---
    let resample_start = Instant::now();
    let input_rate = input.tracks.first().map(|t| t.sample_rate).unwrap_or(48000);
    let resampled = resample::resample(&input.tracks, input_rate, 16000)?;
    let resample_ms = resample_start.elapsed().as_millis();

    let resampled_samples: usize = resampled.iter().map(|s| s.samples.len()).sum();
    tracing::info!(
        stage = "resample",
        from_rate = input_rate,
        to_rate = 16000,
        output_samples = resampled_samples,
        duration_ms = resample_ms,
        "resample complete"
    );

    let duration_processed: f32 = resampled
        .iter()
        .map(|s| s.samples.len() as f32 / s.sample_rate as f32)
        .sum();

    // --- RMS Audio Detection ---
    let rms_start = Instant::now();
    let audio_segments = ad::detect_audio_all(&config.rms, &resampled);
    let rms_ms = rms_start.elapsed().as_millis();

    let rms_active_duration: f32 = audio_segments.iter()
        .map(|s| s.end_time - s.start_time)
        .sum();
    let silence_ratio = if duration_processed > 0.0 {
        1.0 - (rms_active_duration / duration_processed)
    } else {
        0.0
    };

    tracing::info!(
        stage = "rms",
        input_duration_secs = duration_processed,
        active_duration_secs = rms_active_duration,
        segments = audio_segments.len(),
        silence_removed_pct = format_args!("{:.1}", silence_ratio * 100.0),
        duration_ms = rms_ms,
        "RMS audio detection complete"
    );

    // Per-speaker RMS breakdown
    for speaker in &resampled {
        let speaker_segments: Vec<_> = audio_segments.iter()
            .filter(|s| s.speaker == speaker.pseudo_id)
            .collect();
        let speaker_active: f32 = speaker_segments.iter()
            .map(|s| s.end_time - s.start_time)
            .sum();
        let speaker_duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;

        tracing::debug!(
            stage = "rms",
            speaker = %speaker.pseudo_id,
            duration_secs = speaker_duration,
            active_secs = speaker_active,
            segments = speaker_segments.len(),
            silence_pct = format_args!("{:.1}", (1.0 - speaker_active / speaker_duration) * 100.0),
            "per-speaker RMS"
        );
    }

    // --- Silero VAD ---
    let vad_start = Instant::now();
    let voice_chunks =
        vad::detect_speech_from_segments(&config.vad, &audio_segments).await?;
    let vad_ms = vad_start.elapsed().as_millis();

    let vad_speech_duration: f32 = voice_chunks.iter()
        .map(|c| c.original_end - c.original_start)
        .sum();

    tracing::info!(
        stage = "vad",
        input_segments = audio_segments.len(),
        output_chunks = voice_chunks.len(),
        speech_duration_secs = vad_speech_duration,
        reduction_from_rms_pct = format_args!("{:.1}",
            if rms_active_duration > 0.0 {
                (1.0 - vad_speech_duration / rms_active_duration) * 100.0
            } else { 0.0 }
        ),
        duration_ms = vad_ms,
        "VAD complete"
    );

    tracing::info!(
        speakers = speaker_count,
        input_duration_secs = duration_processed,
        rms_segments = audio_segments.len(),
        voice_chunks = voice_chunks.len(),
        speech_duration_secs = vad_speech_duration,
        total_detection_ms = rms_ms + vad_ms,
        "audio detection complete, sending speech to Whisper"
    );

    // --- Whisper Transcription ---
    let whisper_start = Instant::now();
    let segments =
        transcribe::transcribe(&config.whisper, &voice_chunks, session_id).await?;
    let whisper_ms = whisper_start.elapsed().as_millis();

    let avg_confidence: f32 = {
        let confs: Vec<f32> = segments.iter()
            .filter_map(|s| s.confidence)
            .collect();
        if confs.is_empty() { 0.0 } else { confs.iter().sum::<f32>() / confs.len() as f32 }
    };

    tracing::info!(
        stage = "whisper",
        input_chunks = voice_chunks.len(),
        output_segments = segments.len(),
        avg_confidence = format_args!("{:.3}", avg_confidence),
        http_requests = voice_chunks.len(),
        duration_ms = whisper_ms,
        ms_per_request = if voice_chunks.is_empty() { 0 } else { whisper_ms as usize / voice_chunks.len() },
        "transcription complete"
    );

    // --- Operator Chain ---
    let operator_start = Instant::now();
    let pre_operator_count = segments.len();
    let processed = operators::apply_operators(segments, operators).await?;
    let operator_ms = operator_start.elapsed().as_millis();

    let segments_produced = processed.iter().filter(|s| !s.excluded).count() as u32;
    let segments_excluded = processed.iter().filter(|s| s.excluded).count() as u32;
    let scenes_detected = processed
        .iter()
        .filter(|s| !s.excluded)
        .filter_map(|s| s.chunk_group)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);

    // Collect structural output from operators
    let mut all_beats = Vec::new();
    let mut all_scenes = Vec::new();
    for op in operators.iter() {
        all_beats.extend(op.collect_beats());
        all_scenes.extend(op.collect_scenes());
    }

    tracing::info!(
        stage = "operators",
        input_segments = pre_operator_count,
        kept = segments_produced,
        excluded = segments_excluded,
        scenes = scenes_detected,
        beats = all_beats.len(),
        duration_ms = operator_ms,
        "operator chain complete"
    );

    // --- Pipeline Summary ---
    let pipeline_ms = pipeline_start.elapsed().as_millis();

    tracing::info!(
        session_id = %session_id,
        speakers = speaker_count,
        input_duration_secs = input_duration,
        segments_produced,
        segments_excluded,
        scenes_detected,
        duration_processed = duration_processed,
        pipeline_ms = pipeline_ms,
        resample_ms = resample_ms,
        rms_ms = rms_ms,
        vad_ms = vad_ms,
        whisper_ms = whisper_ms,
        operator_ms = operator_ms,
        whisper_pct = format_args!("{:.0}", whisper_ms as f64 / pipeline_ms as f64 * 100.0),
        realtime_factor = format_args!("{:.1}x", input_duration / (pipeline_ms as f32 / 1000.0)),
        "pipeline complete"
    );

    Ok(PipelineResult {
        segments: processed,
        beats: all_beats,
        scenes: all_scenes,
        segments_produced,
        segments_excluded,
        scenes_detected,
        duration_processed,
    })
}
