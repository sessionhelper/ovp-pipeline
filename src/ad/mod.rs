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

// Helper: generate a sine wave at given frequency and amplitude
#[cfg(test)]
fn gen_sine(sample_rate: u32, duration_secs: f32, freq: f32, amplitude: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

#[cfg(test)]
fn gen_silence(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    vec![0.0; (sample_rate as f32 * duration_secs) as usize]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeakerTrack;

    fn default_config() -> RmsConfig {
        RmsConfig {
            threshold: 0.01,
            frame_size: 480,
            silence_gap: 0.3,
            min_segment_duration: 0.1,
        }
    }

    fn make_track(pseudo_id: &str, samples: Vec<f32>) -> SpeakerTrack {
        SpeakerTrack {
            pseudo_id: pseudo_id.into(),
            samples,
            sample_rate: 16000,
        }
    }

    #[test]
    fn silence_produces_no_segments() {
        let config = default_config();
        let track = make_track("silent", gen_silence(16000, 2.0));
        let segments = detect_audio(&config, &track);
        assert!(segments.is_empty());
    }

    #[test]
    fn constant_tone_produces_one_segment() {
        let config = default_config();
        let track = make_track("tone", gen_sine(16000, 1.0, 440.0, 0.5));
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].start_time < 0.01);
        assert!((segments[0].end_time - 1.0).abs() < 0.05);
    }

    #[test]
    fn tone_silence_tone_produces_two_segments() {
        let config = default_config();
        let mut samples = gen_sine(16000, 0.5, 440.0, 0.5);
        samples.extend(gen_silence(16000, 1.0));
        samples.extend(gen_sine(16000, 0.5, 440.0, 0.5));
        let track = make_track("two_bursts", samples);
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 2, "expected two segments, got {:?}",
            segments.iter().map(|s| (s.start_time, s.end_time)).collect::<Vec<_>>());
    }

    #[test]
    fn short_blip_dropped_by_min_duration() {
        let config = RmsConfig {
            min_segment_duration: 0.2,
            ..default_config()
        };
        // 50ms blip — shorter than 200ms min
        let mut samples = gen_silence(16000, 0.5);
        samples.extend(gen_sine(16000, 0.05, 440.0, 0.5));
        samples.extend(gen_silence(16000, 0.5));
        let track = make_track("blip", samples);
        let segments = detect_audio(&config, &track);
        assert!(segments.is_empty(), "short blip should be dropped");
    }

    #[test]
    fn brief_silence_gap_does_not_split() {
        // silence_gap is 0.3s, insert 0.1s silence — should NOT split
        let config = default_config();
        let mut samples = gen_sine(16000, 0.5, 440.0, 0.5);
        samples.extend(gen_silence(16000, 0.1));
        samples.extend(gen_sine(16000, 0.5, 440.0, 0.5));
        let track = make_track("short_gap", samples);
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 1, "brief gap should not split the segment");
    }

    #[test]
    fn segment_timestamps_are_correct() {
        let config = default_config();
        // 1s silence, 0.5s tone, 1s silence
        let mut samples = gen_silence(16000, 1.0);
        samples.extend(gen_sine(16000, 0.5, 440.0, 0.5));
        samples.extend(gen_silence(16000, 1.0));
        let track = make_track("offset", samples);
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 1);
        // Tone starts at 1.0s
        assert!((segments[0].start_time - 1.0).abs() < 0.05,
            "start should be ~1.0s, got {}", segments[0].start_time);
        // Tone ends at 1.5s
        assert!((segments[0].end_time - 1.5).abs() < 0.05,
            "end should be ~1.5s, got {}", segments[0].end_time);
    }

    #[test]
    fn segment_samples_match_duration() {
        let config = default_config();
        let track = make_track("match", gen_sine(16000, 1.0, 440.0, 0.5));
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 1);
        let expected_samples = ((segments[0].end_time - segments[0].start_time) * 16000.0) as usize;
        let actual = segments[0].samples.len();
        // Allow 1 frame of tolerance
        assert!((actual as i64 - expected_samples as i64).unsigned_abs() <= config.frame_size as u64,
            "sample count {} should be close to {}", actual, expected_samples);
    }

    #[test]
    fn below_threshold_audio_is_silent() {
        let config = default_config(); // threshold 0.01
        // Very quiet tone — amplitude below threshold
        let track = make_track("quiet", gen_sine(16000, 1.0, 440.0, 0.005));
        let segments = detect_audio(&config, &track);
        assert!(segments.is_empty(), "audio below threshold should produce no segments");
    }

    #[test]
    fn multiple_speakers_sorted_by_time() {
        let config = default_config();
        // Speaker A: tone at 0.0s
        let a = make_track("a", gen_sine(16000, 0.5, 440.0, 0.5));
        // Speaker B: silence then tone at 1.0s
        let mut b_samples = gen_silence(16000, 1.0);
        b_samples.extend(gen_sine(16000, 0.5, 440.0, 0.5));
        let b = make_track("b", b_samples);

        let segments = detect_audio_all(&config, &[a, b]);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "a");
        assert_eq!(segments[1].speaker, "b");
        assert!(segments[0].start_time < segments[1].start_time);
    }

    #[test]
    fn flush_emits_trailing_segment() {
        let config = default_config();
        // Tone at the end with no trailing silence
        let mut samples = gen_silence(16000, 0.5);
        samples.extend(gen_sine(16000, 0.5, 440.0, 0.5));
        let track = make_track("trailing", samples);
        let segments = detect_audio(&config, &track);
        assert_eq!(segments.len(), 1, "trailing audio should be flushed as a segment");
    }
}
