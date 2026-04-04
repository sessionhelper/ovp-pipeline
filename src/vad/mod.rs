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

    for segment in segments {
        let speaker_samples = SpeakerSamples {
            pseudo_id: segment.speaker.clone(),
            samples: segment.samples.clone(),
            sample_rate: 16000,
        };

        let regions = detect_speech_for_speaker(config, &speaker_samples)?;

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

    tracing::info!(
        input_segments = segments.len(),
        output_chunks = all_chunks.len(),
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
    let mut session = ort::session::Session::builder()
        .map_err(|e| PipelineError::Vad(format!("ort session builder: {}", e)))?
        .commit_from_file(&config.model_path)
        .map_err(|e| PipelineError::Vad(format!("load model {:?}: {}", config.model_path, e)))?;

    let num_frames = speaker.samples.len() / FRAME_SIZE;
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    // LSTM state as flat vecs — reshaped to [1,1,128] tensors per call
    let mut h_data = vec![0.0f32; LSTM_DIM];
    let mut c_data = vec![0.0f32; LSTM_DIM];

    let mut all_probs: Vec<f32> = Vec::with_capacity(num_frames);

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

    // Convert frame probabilities to speech regions
    let raw_regions = merge_speech_frames(&all_probs, config.threshold, &speaker.pseudo_id);
    let filtered = filter_regions(
        raw_regions,
        config,
        speaker.samples.len() as f32 / speaker.sample_rate as f32,
    );

    tracing::debug!(
        speaker = %speaker.pseudo_id,
        frames = num_frames,
        raw_regions = all_probs.iter().filter(|&&p| p >= config.threshold).count(),
        speech_regions = filtered.len(),
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
