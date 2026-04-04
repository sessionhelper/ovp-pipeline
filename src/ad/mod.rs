//! Audio detection via RMS energy thresholding.
//!
//! Tier 1 of the two-tier detection pipeline. Cheap energy gate that
//! removes silence before it reaches the more expensive Silero VAD.
//! Operates on continuous 16kHz mono samples — chunk boundaries are
//! already gone by this point.

use crate::types::SpeakerSamples;

/// Configuration for RMS-based audio detection.
#[derive(Debug, Clone)]
pub struct RmsConfig {
    /// RMS energy threshold. Frames above this are "active audio."
    pub threshold: f32,
    /// Samples per RMS window (480 = 30ms at 16kHz).
    pub frame_size: usize,
    /// Seconds of silence before closing an active segment.
    pub silence_gap: f32,
    /// Drop segments shorter than this (seconds).
    pub min_segment_duration: f32,
}

impl Default for RmsConfig {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            frame_size: 480,
            silence_gap: 0.5,
            min_segment_duration: 0.1,
        }
    }
}

/// A region of detected audio energy for one speaker.
#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub speaker: String,
    pub samples: Vec<f32>,
    pub start_time: f32,
    pub end_time: f32,
}

/// Run RMS audio detection on one speaker's continuous sample stream.
///
/// Walks samples in frame-sized windows, computes RMS per window.
/// When energy rises above threshold, starts accumulating. When energy
/// stays below threshold for `silence_gap` seconds, emits the segment.
pub fn detect_audio(config: &RmsConfig, speaker: &SpeakerSamples) -> Vec<AudioSegment> {
    let sr = speaker.sample_rate as f32;
    let silence_frames = (config.silence_gap * sr / config.frame_size as f32).ceil() as usize;

    let mut segments = Vec::new();
    let mut active = false;
    let mut segment_start_sample: usize = 0;
    let mut silent_count: usize = 0;
    // Track where the last voiced frame ended so we don't include
    // trailing silence in the segment.
    let mut last_active_end: usize = 0;

    let samples = &speaker.samples;
    let mut pos = 0;

    while pos + config.frame_size <= samples.len() {
        let frame = &samples[pos..pos + config.frame_size];
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();

        if rms >= config.threshold {
            if !active {
                // Transition: silent → active
                active = true;
                segment_start_sample = pos;
            }
            silent_count = 0;
            last_active_end = pos + config.frame_size;
        } else if active {
            silent_count += 1;
            if silent_count >= silence_frames {
                // Enough silence — close the segment
                let start_time = segment_start_sample as f32 / sr;
                let end_time = last_active_end as f32 / sr;
                let duration = end_time - start_time;

                if duration >= config.min_segment_duration {
                    segments.push(AudioSegment {
                        speaker: speaker.pseudo_id.clone(),
                        samples: samples[segment_start_sample..last_active_end].to_vec(),
                        start_time,
                        end_time,
                    });
                }

                active = false;
                silent_count = 0;
            }
        }

        pos += config.frame_size;
    }

    // Flush any open segment at the end of the stream
    if active {
        let start_time = segment_start_sample as f32 / sr;
        let end_time = last_active_end as f32 / sr;
        let duration = end_time - start_time;

        if duration >= config.min_segment_duration {
            segments.push(AudioSegment {
                speaker: speaker.pseudo_id.clone(),
                samples: samples[segment_start_sample..last_active_end].to_vec(),
                start_time,
                end_time,
            });
        }
    }

    tracing::debug!(
        speaker = %speaker.pseudo_id,
        total_duration = samples.len() as f32 / sr,
        segments = segments.len(),
        "RMS audio detection complete"
    );

    segments
}

/// Run RMS detection on all speakers, return segments sorted by start time.
pub fn detect_audio_all(
    config: &RmsConfig,
    speakers: &[SpeakerSamples],
) -> Vec<AudioSegment> {
    let mut all: Vec<AudioSegment> = speakers
        .iter()
        .flat_map(|s| detect_audio(config, s))
        .collect();

    all.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_duration: f32 = all.iter().map(|s| s.end_time - s.start_time).sum();
    tracing::info!(
        speakers = speakers.len(),
        segments = all.len(),
        total_audio_duration = total_duration,
        "RMS detection complete for all speakers"
    );

    all
}
