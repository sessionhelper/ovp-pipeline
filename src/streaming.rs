//! Streaming (incremental) pipeline for processing audio chunks as they arrive.
//!
//! Unlike [`pipeline::process_session`] which processes a complete session
//! end-to-end, `StreamingPipeline` accepts audio one chunk at a time per
//! speaker and produces transcript segments incrementally. This enables
//! real-time progressive transcription during an active recording session.
//!
//! # Design
//!
//! Each call to [`StreamingPipeline::feed_chunk`] runs the full
//! resample -> RMS -> VAD -> Whisper chain on the new audio. Speaker audio
//! is tracked independently — a chunk for speaker A does not block or
//! affect processing of speaker B.
//!
//! The `finalize` method flushes any remaining VAD state and runs the
//! operator chain (hallucination detection, beat/scene detection) over
//! all accumulated segments. Operators are intentionally deferred to
//! finalize because they need cross-segment context (e.g. the scene
//! chunker looks at the full conversation arc).
//!
//! # Chunk ordering
//!
//! Callers are responsible for feeding chunks in per-speaker sequence
//! order. The pipeline tracks the cumulative time offset per speaker
//! so that absolute timestamps are correct across chunks.

use std::collections::HashMap;
use std::time::Instant;

use uuid::Uuid;

use crate::ad::{self, RmsConfig};
use crate::audio::resample;
use crate::error::{PipelineError, Result};
use crate::operators::{self, Operator};
use crate::transcribe::{self, TranscriberConfig};
use crate::types::{AudioChunk, PipelineResult, SpeakerSamples, TranscriptSegment};
use crate::vad::{VadConfig, VadSession};

/// Per-speaker state tracked across streaming chunks.
struct SpeakerState {
    /// Persistent VAD session carrying LSTM state between chunks.
    /// `Option` so we can temporarily take it out for `spawn_blocking`
    /// without unsafe code. Always `Some` outside of `feed_chunk`.
    vad: Option<VadSession>,
    /// Cumulative time offset in seconds. Each chunk advances this by
    /// its duration so subsequent chunks get correct absolute timestamps.
    time_offset: f32,
    /// Total samples fed so far (at the input sample rate).
    total_samples: usize,
    /// Accumulated resampled (16 kHz) samples across all chunks for this
    /// speaker. VAD regions reference absolute time, so we need to keep
    /// the full track to extract audio for Whisper.
    resampled_samples: Vec<f32>,
}

/// Configuration for the streaming pipeline. Mirrors [`crate::PipelineConfig`]
/// but is constructed independently so callers can configure streaming
/// without importing the batch pipeline types.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub rms: RmsConfig,
    pub vad: VadConfig,
    pub whisper: TranscriberConfig,
    /// Input sample rate (typically 48000 from Discord).
    pub input_sample_rate: u32,
    /// Session ID for segment metadata.
    pub session_id: Uuid,
}

/// Incremental pipeline that processes audio chunks as they arrive.
///
/// Create one per active recording session. Feed chunks via
/// [`feed_chunk`](Self::feed_chunk), then call
/// [`finalize`](Self::finalize) when recording ends.
///
/// Each speaker gets a persistent [`VadSession`] so the Silero LSTM
/// state carries across chunk boundaries. The model is loaded once per
/// speaker (not per chunk), and speech regions are emitted as soon as
/// a sufficient silence gap is detected.
pub struct StreamingPipeline {
    config: StreamingConfig,
    /// Per-speaker accumulated state (including persistent VAD session).
    speakers: HashMap<String, SpeakerState>,
    /// All segments produced so far (pre-operator).
    segments: Vec<TranscriptSegment>,
    /// Running segment index counter across all speakers.
    next_segment_index: u32,
    /// Minimum speech region duration for Whisper (the "pop filter").
    min_chunk_duration: f32,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline for a session.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            speakers: HashMap::new(),
            segments: Vec::new(),
            next_segment_index: 0,
            min_chunk_duration: 1.0,
        }
    }

    /// Override the minimum chunk duration filter (default 0.8 s).
    pub fn with_min_chunk_duration(mut self, dur: f32) -> Self {
        self.min_chunk_duration = dur;
        self
    }

    /// Feed one audio chunk for a speaker and return any new transcript
    /// segments produced.
    ///
    /// `speaker` is the pseudo_id. `samples` must be mono f32 at the
    /// input sample rate configured in [`StreamingConfig`]. Returns an
    /// empty vec if the chunk contained no speech.
    ///
    /// VAD state (Silero LSTM hidden/cell) persists between calls for the
    /// same speaker, so the model sees a continuous audio stream even when
    /// the audio arrives in 2 MB chunks.
    pub async fn feed_chunk(
        &mut self,
        speaker: &str,
        samples: Vec<f32>,
    ) -> Result<Vec<TranscriptSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_start = Instant::now();
        let input_rate = self.config.input_sample_rate;
        let chunk_duration = samples.len() as f32 / input_rate as f32;

        // Get or create speaker state (including VadSession).
        if !self.speakers.contains_key(speaker) {
            let vad = VadSession::new(&self.config.vad, speaker)?;
            self.speakers.insert(
                speaker.to_string(),
                SpeakerState {
                    vad: Some(vad),
                    time_offset: 0.0,
                    total_samples: 0,
                    resampled_samples: Vec::new(),
                },
            );
        }

        let time_offset = self.speakers[speaker].time_offset;

        tracing::debug!(
            speaker,
            chunk_samples = samples.len(),
            chunk_duration_secs = format_args!("{:.2}", chunk_duration),
            time_offset_secs = format_args!("{:.2}", time_offset),
            "streaming: processing chunk"
        );

        // --- Resample to 16 kHz ---
        let input_track = SpeakerSamples {
            pseudo_id: speaker.to_string(),
            samples,
            sample_rate: input_rate,
        };
        let resampled = resample::resample(&[input_track], input_rate, 16000)?;
        let resampled_track = &resampled[0];

        // --- RMS audio detection ---
        let audio_segments = ad::detect_audio(&self.config.rms, resampled_track);

        if audio_segments.is_empty() {
            tracing::debug!(speaker, "streaming: chunk was silence, no speech detected");
            let state = self.speakers.get_mut(speaker).unwrap();
            state.resampled_samples.extend_from_slice(&resampled_track.samples);
            state.time_offset += chunk_duration;
            state.total_samples += (chunk_duration * input_rate as f32) as usize;
            return Ok(Vec::new());
        }

        // Concatenate RMS-active samples for the stateful VAD.
        let mut active_samples = Vec::new();
        for seg in &audio_segments {
            active_samples.extend_from_slice(&seg.samples);
        }

        // Accumulate the full resampled track (needed for Whisper audio extraction).
        let state = self.speakers.get_mut(speaker).unwrap();
        state.resampled_samples.extend_from_slice(&resampled_track.samples);

        // --- Stateful VAD (runs on blocking thread for ort safety) ---
        // Take VadSession out of the Option, send to blocking thread, put back.
        let mut vad_session = state.vad.take().expect("VadSession missing from speaker state");

        let (vad_back, regions) = tokio::task::spawn_blocking(move || {
            let regions = vad_session.feed(&active_samples, 16000);
            (vad_session, regions)
        })
        .await
        .map_err(|e| PipelineError::Vad(format!("spawn_blocking: {e}")))?;

        let regions = regions?;

        // Put the VadSession back.
        let state = self.speakers.get_mut(speaker).unwrap();
        state.vad = Some(vad_back);

        if regions.is_empty() {
            tracing::debug!(speaker, "streaming: chunk had audio but no speech after VAD");
            state.time_offset += chunk_duration;
            state.total_samples += (chunk_duration * input_rate as f32) as usize;
            return Ok(Vec::new());
        }

        // --- Build AudioChunks from completed speech regions ---
        // VAD regions have absolute timestamps (across all chunks fed so far).
        // Pull the audio from the accumulated resampled buffer.
        let mut voice_chunks = Vec::new();
        for region in &regions {
            let start_sample = (region.start * 16000.0) as usize;
            let end_sample = (region.end * 16000.0) as usize;

            let src = &state.resampled_samples;
            let start = start_sample.min(src.len());
            let end = end_sample.min(src.len());

            if start >= end {
                continue;
            }

            let region_dur = (end - start) as f32 / 16000.0;
            if region_dur < self.min_chunk_duration {
                continue;
            }

            voice_chunks.push(AudioChunk {
                speaker: speaker.to_string(),
                samples: src[start..end].to_vec(),
                sample_rate: 16000,
                original_start: region.start,
                original_end: region.end,
            });
        }

        if voice_chunks.is_empty() {
            state.time_offset += chunk_duration;
            state.total_samples += (chunk_duration * input_rate as f32) as usize;
            return Ok(Vec::new());
        }

        // --- Whisper transcription ---
        let new_segments = transcribe::transcribe(
            &self.config.whisper,
            &voice_chunks,
            self.config.session_id,
        )
        .await?;

        // Re-index segments with our running counter
        let mut result = Vec::with_capacity(new_segments.len());
        for mut seg in new_segments {
            seg.segment_index = self.next_segment_index;
            self.next_segment_index += 1;
            result.push(seg);
        }

        let chunk_ms = chunk_start.elapsed().as_millis();
        tracing::info!(
            speaker,
            new_segments = result.len(),
            vad_regions = regions.len(),
            voice_chunks = voice_chunks.len(),
            chunk_duration_secs = format_args!("{:.2}", chunk_duration),
            processing_ms = chunk_ms,
            "streaming: chunk processed"
        );

        // Accumulate for finalize
        self.segments.extend(result.clone());

        // Advance speaker time offset
        let state = self.speakers.get_mut(speaker).unwrap();
        state.time_offset += chunk_duration;
        state.total_samples += (chunk_duration * input_rate as f32) as usize;

        Ok(result)
    }

    /// How many segments have been produced so far.
    pub fn segment_count(&self) -> u32 {
        self.next_segment_index
    }

    /// Finalize the session: flush remaining VAD state, run the operator
    /// chain over all accumulated segments, and return the complete result.
    ///
    /// After this call, the pipeline is consumed and should not be reused.
    pub async fn finalize(
        mut self,
        operators: &mut [Box<dyn Operator>],
    ) -> Result<PipelineResult> {
        let finalize_start = Instant::now();

        // Flush each speaker's VAD session for trailing speech regions.
        let speaker_keys: Vec<String> = self.speakers.keys().cloned().collect();

        for speaker_key in &speaker_keys {
            let state = self.speakers.get_mut(speaker_key).unwrap();
            let trailing = match state.vad.as_mut() {
                Some(vad) => vad.flush(),
                None => Vec::new(),
            };

            if !trailing.is_empty() {
                // Build AudioChunks from trailing regions and transcribe.
                let mut voice_chunks = Vec::new();
                for region in &trailing {
                    let start_sample = (region.start * 16000.0) as usize;
                    let end_sample = (region.end * 16000.0) as usize;

                    let src = &state.resampled_samples;
                    let start = start_sample.min(src.len());
                    let end = end_sample.min(src.len());

                    if start >= end {
                        continue;
                    }

                    let region_dur = (end - start) as f32 / 16000.0;
                    if region_dur < self.min_chunk_duration {
                        continue;
                    }

                    voice_chunks.push(AudioChunk {
                        speaker: speaker_key.clone(),
                        samples: src[start..end].to_vec(),
                        sample_rate: 16000,
                        original_start: region.start,
                        original_end: region.end,
                    });
                }

                if !voice_chunks.is_empty() {
                    let new_segments = transcribe::transcribe(
                        &self.config.whisper,
                        &voice_chunks,
                        self.config.session_id,
                    )
                    .await?;

                    for mut seg in new_segments {
                        seg.segment_index = self.next_segment_index;
                        self.next_segment_index += 1;
                        self.segments.push(seg);
                    }
                }

                tracing::info!(
                    speaker = %speaker_key,
                    trailing_regions = trailing.len(),
                    "streaming: VAD flush produced trailing regions"
                );
            }
        }

        let pre_operator_count = self.segments.len();

        tracing::info!(
            session_id = %self.config.session_id,
            total_segments = pre_operator_count,
            speakers = self.speakers.len(),
            "streaming: finalizing session"
        );

        // Sort segments by start_time for operator chain (they may be
        // interleaved across speakers)
        let mut segments = self.segments;
        segments.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-index after sorting so segment_index matches chronological order
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.segment_index = i as u32;
        }

        // Run operator chain
        let processed = operators::apply_operators(segments, operators).await?;

        // Collect structural output from operators
        let mut all_beats = Vec::new();
        let mut all_scenes = Vec::new();
        for op in operators.iter() {
            all_beats.extend(op.collect_beats());
            all_scenes.extend(op.collect_scenes());
        }

        let segments_produced = processed.iter().filter(|s| !s.excluded).count() as u32;
        let segments_excluded = processed.iter().filter(|s| s.excluded).count() as u32;
        let scenes_detected = processed
            .iter()
            .filter(|s| !s.excluded)
            .filter_map(|s| s.chunk_group)
            .max()
            .map(|max| max + 1)
            .unwrap_or(0);

        let duration_processed: f32 = self
            .speakers
            .values()
            .map(|s| s.time_offset)
            .sum();

        let finalize_ms = finalize_start.elapsed().as_millis();

        tracing::info!(
            session_id = %self.config.session_id,
            segments_produced,
            segments_excluded,
            scenes_detected,
            beats = all_beats.len(),
            finalize_ms,
            "streaming: finalize complete"
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
}
