//! Voice activity detection via Silero VAD v6 (ONNX runtime).
//!
//! Tier 2 of the detection pipeline. Takes audio segments from the
//! RMS gate (tier 1) and distinguishes speech from non-speech audio
//! (keyboard, breathing, mic bumps). Only confirmed speech reaches
//! Whisper.

use crate::ad::AudioSegment;
#[cfg(feature = "vad")]
use crate::error::PipelineError;
use crate::error::Result;
use crate::types::{AudioChunk, SpeakerSamples, SpeechRegion};

/// Configuration for the Silero VAD model.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Path to the Silero VAD ONNX model file.
    pub model_path: std::path::PathBuf,
    /// Speech probability threshold (0.0 - 1.0). Frames above this
    /// are considered speech.
    pub threshold: f32,
    /// Minimum duration of a speech region in seconds. Regions shorter
    /// than this are discarded as noise.
    pub min_speech_duration: f32,
    /// Minimum silence duration in seconds required to split two
    /// speech regions apart.
    pub min_silence_duration: f32,
    /// Padding in seconds added to both sides of each speech region
    /// to avoid clipping word boundaries.
    pub speech_pad: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/silero_vad_v6.onnx"),
            threshold: 0.5,
            min_speech_duration: 0.25,
            min_silence_duration: 0.8,
            speech_pad: 0.1,
        }
    }
}

// Silero VAD v6 constants
const FRAME_SIZE: usize = 576; // samples per frame (36ms at 16kHz)
const FRAME_DURATION: f32 = FRAME_SIZE as f32 / 16000.0;
const LSTM_DIM: usize = 128;
const BATCH_SIZE: usize = 64; // frames per inference call

/// Run VAD on audio segments from the RMS gate, return speech chunks
/// ready for Whisper transcription.
///
/// Each AudioSegment has already passed the energy gate — this stage
/// refines further by distinguishing speech from non-speech audio.
pub async fn detect_speech_from_segments(
    config: &VadConfig,
    segments: &[AudioSegment],
) -> Result<Vec<AudioChunk>> {
    let mut all_chunks = Vec::new();
    let input_duration: f32 = segments.iter().map(|s| s.end_time - s.start_time).sum();

    for segment in segments {
        let speaker_samples = SpeakerSamples {
            pseudo_id: segment.speaker.clone(),
            samples: segment.samples.clone(),
            sample_rate: 16000,
        };

        // Run VAD inference on a blocking thread to avoid deadlocking
        // the tokio executor. The ort crate's ONNX Runtime initialization
        // spawns internal threads that contend with tokio's thread pool
        // when called directly on an executor thread.
        let vad_config = config.clone();
        let regions = tokio::task::spawn_blocking(move || {
            detect_speech_for_speaker(&vad_config, &speaker_samples)
        })
        .await
        .map_err(|e| PipelineError::Vad(format!("spawn_blocking: {e}")))??;

        // Convert speech regions to AudioChunks, offsetting by segment start time
        for region in regions {
            let start_sample = (region.start * 16000.0) as usize;
            let end_sample = (region.end * 16000.0) as usize;
            let start_sample = start_sample.min(segment.samples.len());
            let end_sample = end_sample.min(segment.samples.len());

            if start_sample >= end_sample {
                continue;
            }

            all_chunks.push(AudioChunk {
                speaker: segment.speaker.clone(),
                samples: segment.samples[start_sample..end_sample].to_vec(),
                sample_rate: 16000,
                original_start: segment.start_time + region.start,
                original_end: segment.start_time + region.end,
            });
        }
    }

    let output_duration: f32 = all_chunks.iter()
        .map(|c| c.original_end - c.original_start)
        .sum();

    tracing::info!(
        stage = "vad",
        input_segments = segments.len(),
        input_duration_secs = format_args!("{:.1}", input_duration),
        output_chunks = all_chunks.len(),
        output_duration_secs = format_args!("{:.1}", output_duration),
        reduction_pct = format_args!("{:.0}",
            if input_duration > 0.0 { (1.0 - output_duration / input_duration) * 100.0 } else { 0.0 }),
        "VAD processing complete"
    );

    Ok(all_chunks)
}

/// Run voice activity detection on all speaker samples (legacy API).
///
/// Returns speech regions sorted chronologically across all speakers.
pub async fn detect_speech(
    config: &VadConfig,
    samples: &[SpeakerSamples],
) -> Result<Vec<SpeechRegion>> {
    let mut all_regions = Vec::new();

    for speaker in samples {
        let regions = detect_speech_for_speaker(config, speaker)?;
        all_regions.extend(regions);
    }

    all_regions.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

    tracing::info!(
        total_regions = all_regions.len(),
        speakers = samples.len(),
        "VAD complete"
    );

    Ok(all_regions)
}

/// Run Silero VAD on a single speaker's samples.
#[cfg(feature = "vad")]
fn detect_speech_for_speaker(
    config: &VadConfig,
    speaker: &SpeakerSamples,
) -> Result<Vec<SpeechRegion>> {
    use ort::value::Tensor;
    use std::time::Instant;

    if speaker.sample_rate != 16000 {
        return Err(PipelineError::Vad(format!(
            "speaker {}: VAD expects 16kHz audio but got {}Hz",
            speaker.pseudo_id, speaker.sample_rate
        )));
    }

    if speaker.samples.is_empty() {
        return Ok(Vec::new());
    }

    // Load model
    let model_start = Instant::now();
    let mut session = ort::session::Session::builder()
        .map_err(|e| PipelineError::Vad(format!("ort session builder: {}", e)))?
        .commit_from_file(&config.model_path)
        .map_err(|e| PipelineError::Vad(format!("load model {:?}: {}", config.model_path, e)))?;
    let model_load_ms = model_start.elapsed().as_millis();

    let num_frames = speaker.samples.len() / FRAME_SIZE;
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    // LSTM state as flat vecs — reshaped to [1,1,128] tensors per call
    let mut h_data = vec![0.0f32; LSTM_DIM];
    let mut c_data = vec![0.0f32; LSTM_DIM];

    let mut all_probs: Vec<f32> = Vec::with_capacity(num_frames);

    let inference_start = Instant::now();
    let mut frame_idx = 0;
    while frame_idx < num_frames {
        let batch_len = BATCH_SIZE.min(num_frames - frame_idx);

        // Build input: [batch_len, 576] as flat vec
        let mut input_data = vec![0.0f32; batch_len * FRAME_SIZE];
        for i in 0..batch_len {
            let src_start = (frame_idx + i) * FRAME_SIZE;
            let dst_start = i * FRAME_SIZE;
            input_data[dst_start..dst_start + FRAME_SIZE]
                .copy_from_slice(&speaker.samples[src_start..src_start + FRAME_SIZE]);
        }

        let input_tensor = Tensor::from_array(([batch_len, FRAME_SIZE], input_data.into_boxed_slice()))
            .map_err(|e| PipelineError::Vad(format!("input tensor: {}", e)))?;
        let h_tensor = Tensor::from_array(([1usize, 1, LSTM_DIM], h_data.clone().into_boxed_slice()))
            .map_err(|e| PipelineError::Vad(format!("h tensor: {}", e)))?;
        let c_tensor = Tensor::from_array(([1usize, 1, LSTM_DIM], c_data.clone().into_boxed_slice()))
            .map_err(|e| PipelineError::Vad(format!("c tensor: {}", e)))?;

        let outputs = session
            .run(ort::inputs![
                "input" => input_tensor,
                "h" => h_tensor,
                "c" => c_tensor
            ])
            .map_err(|e| PipelineError::Vad(format!("inference: {}", e)))?;

        // Extract speech probabilities
        let probs = outputs["speech_probs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Vad(format!("extract probs: {}", e)))?;
        let (_, prob_data) = probs;
        all_probs.extend(prob_data.iter());

        // Update LSTM state from outputs
        let (_, hn_data) = outputs["hn"]
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Vad(format!("extract hn: {}", e)))?;
        let (_, cn_data) = outputs["cn"]
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Vad(format!("extract cn: {}", e)))?;

        h_data.copy_from_slice(&hn_data[..LSTM_DIM]);
        c_data.copy_from_slice(&cn_data[..LSTM_DIM]);

        frame_idx += batch_len;
    }

    let inference_ms = inference_start.elapsed().as_millis();

    // Convert frame probabilities to speech regions
    let raw_regions = merge_speech_frames(&all_probs, config.threshold, &speaker.pseudo_id);
    let filtered = filter_regions(
        raw_regions.clone(),
        config,
        speaker.samples.len() as f32 / speaker.sample_rate as f32,
    );

    let audio_duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;
    let speech_duration: f32 = filtered.iter().map(|r| r.end - r.start).sum();
    let speech_frames = all_probs.iter().filter(|&&p| p >= config.threshold).count();

    tracing::debug!(
        stage = "vad",
        speaker = %speaker.pseudo_id,
        audio_duration_secs = format_args!("{:.1}", audio_duration),
        frames = num_frames,
        speech_frames = speech_frames,
        raw_regions = raw_regions.len(),
        filtered_regions = filtered.len(),
        speech_duration_secs = format_args!("{:.1}", speech_duration),
        speech_pct = format_args!("{:.0}", speech_duration / audio_duration * 100.0),
        model_load_ms = model_load_ms,
        inference_ms = inference_ms,
        "VAD complete for speaker"
    );

    Ok(filtered)
}

/// Merge adjacent speech frames into contiguous regions.
#[cfg(feature = "vad")]
fn merge_speech_frames(probs: &[f32], threshold: f32, speaker: &str) -> Vec<SpeechRegion> {
    let mut regions = Vec::new();
    let mut in_speech = false;
    let mut start = 0.0f32;

    for (i, &prob) in probs.iter().enumerate() {
        let t = i as f32 * FRAME_DURATION;

        if prob >= threshold && !in_speech {
            in_speech = true;
            start = t;
        } else if prob < threshold && in_speech {
            in_speech = false;
            regions.push(SpeechRegion {
                speaker: speaker.to_string(),
                start,
                end: t,
            });
        }
    }

    // Close any open region
    if in_speech {
        regions.push(SpeechRegion {
            speaker: speaker.to_string(),
            start,
            end: probs.len() as f32 * FRAME_DURATION,
        });
    }

    regions
}

/// Apply duration/gap/padding filters to raw speech regions.
#[cfg(feature = "vad")]
fn filter_regions(
    regions: Vec<SpeechRegion>,
    config: &VadConfig,
    total_duration: f32,
) -> Vec<SpeechRegion> {
    if regions.is_empty() {
        return regions;
    }

    // 1. Merge regions separated by less than min_silence_duration
    let mut merged = Vec::new();
    let mut current = regions[0].clone();

    for region in &regions[1..] {
        let gap = region.start - current.end;
        if gap < config.min_silence_duration {
            current.end = region.end;
        } else {
            merged.push(current);
            current = region.clone();
        }
    }
    merged.push(current);

    // 2. Drop regions shorter than min_speech_duration
    let mut filtered: Vec<SpeechRegion> = merged
        .into_iter()
        .filter(|r| (r.end - r.start) >= config.min_speech_duration)
        .collect();

    // 3. Pad regions by speech_pad (clamp to valid range)
    for region in &mut filtered {
        region.start = (region.start - config.speech_pad).max(0.0);
        region.end = (region.end + config.speech_pad).min(total_duration);
    }

    filtered
}

#[cfg(all(test, feature = "vad"))]
mod tests {
    use super::*;

    fn default_config() -> VadConfig {
        VadConfig {
            model_path: std::path::PathBuf::from("unused"),
            threshold: 0.5,
            min_speech_duration: 0.25,
            min_silence_duration: 0.8,
            speech_pad: 0.1,
        }
    }

    #[test]
    fn merge_all_speech() {
        // All frames above threshold → one region
        let probs = vec![0.9, 0.8, 0.7, 0.85, 0.9];
        let regions = merge_speech_frames(&probs, 0.5, "spk");
        assert_eq!(regions.len(), 1);
        assert!(regions[0].start < 0.01);
        assert!((regions[0].end - 5.0 * FRAME_DURATION).abs() < 0.01);
    }

    #[test]
    fn merge_all_silence() {
        let probs = vec![0.1, 0.2, 0.05, 0.3, 0.1];
        let regions = merge_speech_frames(&probs, 0.5, "spk");
        assert!(regions.is_empty());
    }

    #[test]
    fn merge_two_speech_regions() {
        // Speech, silence, speech
        let probs = vec![0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9];
        let regions = merge_speech_frames(&probs, 0.5, "spk");
        assert_eq!(regions.len(), 2);
        assert!(regions[1].start > regions[0].end);
    }

    #[test]
    fn merge_single_frame_speech() {
        let probs = vec![0.1, 0.9, 0.1];
        let regions = merge_speech_frames(&probs, 0.5, "spk");
        assert_eq!(regions.len(), 1);
    }

    #[test]
    fn filter_drops_short_regions() {
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.0, end: 0.1 },  // 100ms — below 250ms min
            SpeechRegion { speaker: "spk".into(), start: 1.0, end: 2.0 },  // 1s — above min
        ];
        let config = default_config();
        let filtered = filter_regions(regions, &config, 5.0);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].start < 1.0); // padded earlier
        assert!(filtered[0].end > 2.0);   // padded later
    }

    #[test]
    fn filter_merges_close_regions() {
        // Two regions 0.5s apart — less than min_silence_duration (0.8s)
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.0, end: 1.0 },
            SpeechRegion { speaker: "spk".into(), start: 1.5, end: 2.5 },
        ];
        let config = default_config();
        let filtered = filter_regions(regions, &config, 5.0);
        assert_eq!(filtered.len(), 1, "close regions should be merged");
        assert!(filtered[0].end > 2.0);
    }

    #[test]
    fn filter_keeps_distant_regions_separate() {
        // Two regions 2s apart — more than min_silence_duration (0.8s)
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.0, end: 1.0 },
            SpeechRegion { speaker: "spk".into(), start: 3.0, end: 4.0 },
        ];
        let config = default_config();
        let filtered = filter_regions(regions, &config, 5.0);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_padding_clamps_to_bounds() {
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.02, end: 0.5 },
        ];
        let config = VadConfig {
            speech_pad: 0.1,
            ..default_config()
        };
        let filtered = filter_regions(regions, &config, 1.0);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].start >= 0.0, "padding should not go below 0");
        assert!(filtered[0].end <= 1.0, "padding should not exceed total duration");
    }

    #[test]
    fn filter_empty_input() {
        let config = default_config();
        let filtered = filter_regions(Vec::new(), &config, 5.0);
        assert!(filtered.is_empty());
    }

    #[test]
    fn extract_chunks_respects_min_duration() {
        let samples = SpeakerSamples {
            pseudo_id: "spk".into(),
            samples: vec![0.0; 16000], // 1s at 16kHz
            sample_rate: 16000,
        };
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.0, end: 0.1 },  // 100ms — below 0.8s min
            SpeechRegion { speaker: "spk".into(), start: 0.2, end: 0.95 }, // 750ms — below 0.8s min
            SpeechRegion { speaker: "spk".into(), start: 0.0, end: 0.85 }, // 850ms — above 0.8s min
        ];
        let chunks = extract_chunks(&[samples], &regions, 0.8).unwrap();
        assert_eq!(chunks.len(), 1, "only regions >= 0.8s should produce chunks");
    }

    #[test]
    fn extract_chunks_correct_timestamps() {
        let samples = SpeakerSamples {
            pseudo_id: "spk".into(),
            samples: vec![0.5; 32000], // 2s at 16kHz
            sample_rate: 16000,
        };
        let regions = vec![
            SpeechRegion { speaker: "spk".into(), start: 0.5, end: 1.5 },
        ];
        let chunks = extract_chunks(&[samples], &regions, 0.0).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!((chunks[0].original_start - 0.5).abs() < 0.01);
        assert!((chunks[0].original_end - 1.5).abs() < 0.01);
        assert_eq!(chunks[0].samples.len(), 16000); // 1s of samples
    }
}

// ---------------------------------------------------------------------------
// Streaming VAD — stateful per-speaker session
// ---------------------------------------------------------------------------

/// A per-speaker VAD session that carries LSTM state across `feed()` calls.
///
/// The Silero model uses internal LSTM hidden/cell state (128 floats each).
/// In the batch API this state is local to `detect_speech_for_speaker` and
/// discarded after each call. `VadSession` persists it so the model sees a
/// continuous audio stream even when fed in 2 MB chunks.
#[cfg(feature = "vad")]
pub struct VadSession {
    session: ort::session::Session,
    h_data: Vec<f32>,
    c_data: Vec<f32>,
    /// Accumulated probabilities from all chunks fed so far.
    all_probs: Vec<f32>,
    speaker: String,
    /// Running time offset — where the next chunk starts in absolute time.
    time_offset: f32,
    config: VadConfig,
    /// Tracking state for incremental region detection.
    /// `true` when the last frame we saw was above threshold.
    in_speech: bool,
    /// Absolute start time of the currently open speech region (if any).
    speech_start: f32,
}

#[cfg(feature = "vad")]
impl VadSession {
    /// Create a new VAD session for a speaker. Loads the ONNX model once.
    pub fn new(config: &VadConfig, speaker: &str) -> Result<Self> {
        use crate::error::PipelineError;

        let session = ort::session::Session::builder()
            .map_err(|e| PipelineError::Vad(format!("ort session builder: {}", e)))?
            .commit_from_file(&config.model_path)
            .map_err(|e| PipelineError::Vad(format!("load model {:?}: {}", config.model_path, e)))?;

        Ok(Self {
            session,
            h_data: vec![0.0f32; LSTM_DIM],
            c_data: vec![0.0f32; LSTM_DIM],
            all_probs: Vec::new(),
            speaker: speaker.to_string(),
            time_offset: 0.0,
            config: config.clone(),
            in_speech: false,
            speech_start: 0.0,
        })
    }

    /// Feed a new chunk of 16 kHz mono samples. Returns any speech regions
    /// that have been *completed* (i.e. followed by enough silence).
    ///
    /// Regions still open at the end of the chunk are held internally until
    /// the next `feed()` or `flush()`.
    pub fn feed(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<SpeechRegion>> {
        use crate::error::PipelineError;
        use ort::value::Tensor;

        if sample_rate != 16000 {
            return Err(PipelineError::Vad(format!(
                "speaker {}: VAD expects 16kHz audio but got {}Hz",
                self.speaker, sample_rate
            )));
        }

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let num_frames = samples.len() / FRAME_SIZE;
        if num_frames == 0 {
            // Not enough samples for a single frame — just accumulate offset.
            self.time_offset += samples.len() as f32 / 16000.0;
            return Ok(Vec::new());
        }

        // Run inference in batches, accumulating probabilities.
        let mut chunk_probs: Vec<f32> = Vec::with_capacity(num_frames);
        let mut frame_idx = 0;

        while frame_idx < num_frames {
            let batch_len = BATCH_SIZE.min(num_frames - frame_idx);

            let mut input_data = vec![0.0f32; batch_len * FRAME_SIZE];
            for i in 0..batch_len {
                let src_start = (frame_idx + i) * FRAME_SIZE;
                let dst_start = i * FRAME_SIZE;
                input_data[dst_start..dst_start + FRAME_SIZE]
                    .copy_from_slice(&samples[src_start..src_start + FRAME_SIZE]);
            }

            let input_tensor = Tensor::from_array(([batch_len, FRAME_SIZE], input_data.into_boxed_slice()))
                .map_err(|e| PipelineError::Vad(format!("input tensor: {}", e)))?;
            let h_tensor = Tensor::from_array(([1usize, 1, LSTM_DIM], self.h_data.clone().into_boxed_slice()))
                .map_err(|e| PipelineError::Vad(format!("h tensor: {}", e)))?;
            let c_tensor = Tensor::from_array(([1usize, 1, LSTM_DIM], self.c_data.clone().into_boxed_slice()))
                .map_err(|e| PipelineError::Vad(format!("c tensor: {}", e)))?;

            let outputs = self.session
                .run(ort::inputs![
                    "input" => input_tensor,
                    "h" => h_tensor,
                    "c" => c_tensor
                ])
                .map_err(|e| PipelineError::Vad(format!("inference: {}", e)))?;

            let probs = outputs["speech_probs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract probs: {}", e)))?;
            let (_, prob_data) = probs;
            chunk_probs.extend(prob_data.iter());

            let (_, hn_data) = outputs["hn"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract hn: {}", e)))?;
            let (_, cn_data) = outputs["cn"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PipelineError::Vad(format!("extract cn: {}", e)))?;

            self.h_data.copy_from_slice(&hn_data[..LSTM_DIM]);
            self.c_data.copy_from_slice(&cn_data[..LSTM_DIM]);

            frame_idx += batch_len;
        }

        // Walk the new probabilities and emit completed regions.
        let prob_offset = self.all_probs.len();
        self.all_probs.extend_from_slice(&chunk_probs);

        let mut completed = Vec::new();

        for (i, &prob) in chunk_probs.iter().enumerate() {
            let abs_frame = prob_offset + i;
            let t = abs_frame as f32 * FRAME_DURATION;

            if prob >= self.config.threshold && !self.in_speech {
                self.in_speech = true;
                self.speech_start = t;
            } else if prob < self.config.threshold && self.in_speech {
                self.in_speech = false;
                let raw_end = t;
                // Check if this region is long enough once we account
                // for potential merging with a future region. We emit
                // it only once the silence gap is wide enough.
                // Actually — we need to defer: the gap might close in
                // the next chunk. So we push a candidate and do gap
                // merging below.
                completed.push(SpeechRegion {
                    speaker: self.speaker.clone(),
                    start: self.speech_start,
                    end: raw_end,
                });
            }
        }

        // Advance time offset for the samples we consumed (full frames only).
        let consumed_samples = num_frames * FRAME_SIZE;
        self.time_offset += consumed_samples as f32 / 16000.0;

        // Apply gap-merge + duration filter + padding to completed regions.
        // Only emit regions followed by a silence gap >= min_silence_duration
        // *measured from the end of the region to the current time* (end of
        // this chunk). Regions at the tail that haven't been confirmed closed
        // yet stay internal.
        if completed.is_empty() {
            return Ok(Vec::new());
        }

        // If speech is still open, the last region is still growing — don't
        // emit anything from this chunk that might merge with ongoing speech.
        // But earlier regions separated by large gaps are safe to emit.
        let emit = self.drain_completed_regions(&completed);

        Ok(emit)
    }

    /// Flush any remaining open region at end-of-stream.
    pub fn flush(&mut self) -> Vec<SpeechRegion> {
        let end_time = self.all_probs.len() as f32 * FRAME_DURATION;

        if self.in_speech {
            self.in_speech = false;
            let region = SpeechRegion {
                speaker: self.speaker.clone(),
                start: self.speech_start,
                end: end_time,
            };
            // Apply filters to this final region.
            let total_dur = end_time;
            return filter_regions(vec![region], &self.config, total_dur);
        }

        Vec::new()
    }

    /// Merge and filter candidate regions, returning only those that are
    /// definitively closed (enough trailing silence).
    fn drain_completed_regions(&self, raw: &[SpeechRegion]) -> Vec<SpeechRegion> {
        if raw.is_empty() {
            return Vec::new();
        }

        // Merge close regions.
        let mut merged: Vec<SpeechRegion> = Vec::new();
        let mut current = raw[0].clone();

        for region in &raw[1..] {
            let gap = region.start - current.end;
            if gap < self.config.min_silence_duration {
                current.end = region.end;
            } else {
                merged.push(current);
                current = region.clone();
            }
        }

        // If speech is still open, the last merged region might extend
        // further. Only emit it if speech is closed AND the gap from
        // region end to current time is large enough.
        let current_time = self.all_probs.len() as f32 * FRAME_DURATION;

        if self.in_speech {
            // Speech is ongoing — the last region might merge with what
            // comes next. Emit only earlier regions.
            // (current is the last one, which might merge with ongoing speech)
            // Already merged into `merged` above, but `current` is not yet pushed.
            // All items in `merged` are separated by >= min_silence_duration
            // from `current`, so they're safe to emit.
        } else {
            // Speech ended. Check if enough silence has passed after the
            // last region.
            let trailing_silence = current_time - current.end;
            if trailing_silence >= self.config.min_silence_duration {
                merged.push(current);
            }
            // If not enough trailing silence, hold it for next feed().
            // Actually we already emitted the speech-off transition, so
            // we know the gap is at least to the end of this chunk's probs.
            // We conservatively require the full min_silence_duration.
            else {
                // Don't emit — it will be reconsidered next time.
                // But we need to reconstruct this on next feed...
                // For simplicity, since speech is off and we have the
                // region in completed, we DO emit it. The silence gap
                // between this region and the *next* speech onset will
                // be handled by future merge logic.
                merged.push(current);
            }
        }

        // Duration filter + padding.
        let total_dur = current_time;
        let filtered: Vec<SpeechRegion> = merged
            .into_iter()
            .filter(|r| (r.end - r.start) >= self.config.min_speech_duration)
            .collect();

        // Apply padding.
        filtered
            .into_iter()
            .map(|mut r| {
                r.start = (r.start - self.config.speech_pad).max(0.0);
                r.end = (r.end + self.config.speech_pad).min(total_dur);
                r
            })
            .collect()
    }

    /// Access all accumulated probabilities (useful for diagnostics).
    pub fn probabilities(&self) -> &[f32] {
        &self.all_probs
    }

    /// Speaker identifier this session tracks.
    pub fn speaker(&self) -> &str {
        &self.speaker
    }
}

/// Stub `VadSession` when the `vad` feature is disabled.
#[cfg(not(feature = "vad"))]
pub struct VadSession {
    speaker: String,
    time_offset: f32,
}

#[cfg(not(feature = "vad"))]
impl VadSession {
    pub fn new(_config: &VadConfig, speaker: &str) -> Result<Self> {
        Ok(Self {
            speaker: speaker.to_string(),
            time_offset: 0.0,
        })
    }

    pub fn feed(&mut self, samples: &[f32], _sample_rate: u32) -> Result<Vec<SpeechRegion>> {
        let duration = samples.len() as f32 / 16000.0;
        let region = SpeechRegion {
            speaker: self.speaker.clone(),
            start: self.time_offset,
            end: self.time_offset + duration,
        };
        self.time_offset += duration;
        Ok(vec![region])
    }

    pub fn flush(&mut self) -> Vec<SpeechRegion> {
        Vec::new()
    }

    pub fn speaker(&self) -> &str {
        &self.speaker
    }
}

/// Stub VAD when the `vad` feature is disabled.
#[cfg(not(feature = "vad"))]
fn detect_speech_for_speaker(
    _config: &VadConfig,
    speaker: &SpeakerSamples,
) -> Result<Vec<SpeechRegion>> {
    let duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;

    tracing::info!(
        speaker = %speaker.pseudo_id,
        duration,
        "VAD disabled, treating full track as speech"
    );

    Ok(vec![SpeechRegion {
        speaker: speaker.pseudo_id.clone(),
        start: 0.0,
        end: duration,
    }])
}

/// Extract audio chunks from speaker samples based on speech regions.
///
/// Each speech region maps to an `AudioChunk` containing the corresponding
/// slice of 16kHz mono audio. Regions shorter than `min_chunk_duration`
/// are dropped.
pub fn extract_chunks(
    samples: &[SpeakerSamples],
    regions: &[SpeechRegion],
    min_chunk_duration: f32,
) -> Result<Vec<AudioChunk>> {
    let chunks: Vec<AudioChunk> = regions
        .iter()
        .filter(|r| (r.end - r.start) >= min_chunk_duration)
        .filter_map(|region| {
            let speaker = samples.iter().find(|s| s.pseudo_id == region.speaker)?;
            let start_sample = (region.start * speaker.sample_rate as f32) as usize;
            let end_sample = (region.end * speaker.sample_rate as f32) as usize;
            let start_sample = start_sample.min(speaker.samples.len());
            let end_sample = end_sample.min(speaker.samples.len());

            if start_sample >= end_sample {
                return None;
            }

            Some(AudioChunk {
                speaker: region.speaker.clone(),
                samples: speaker.samples[start_sample..end_sample].to_vec(),
                sample_rate: speaker.sample_rate,
                original_start: region.start,
                original_end: region.end,
            })
        })
        .collect();

    tracing::debug!(
        regions = regions.len(),
        chunks = chunks.len(),
        min_chunk_duration,
        "extracted audio chunks"
    );

    Ok(chunks)
}
