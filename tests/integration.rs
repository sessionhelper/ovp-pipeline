//! Integration tests against real TTRPG audio data.
//!
//! These tests require:
//! - test_data/scene_01/ with PCM chunks (gitignored, not committed)
//! - models/silero_vad_v6.onnx
//! - ORT_DYLIB_PATH set to an ONNX Runtime shared library
//!
//! Tests skip gracefully if any of these are missing.

use std::fs;
use std::path::Path;

use ovp_pipeline::ad::{detect_audio, detect_audio_all, RmsConfig};
use ovp_pipeline::vad::VadConfig;
use ovp_pipeline::audio::resample;

const SCENE_DIR: &str = "test_data/scene_01";
const VAD_MODEL: &str = "models/silero_vad_v6.onnx";

// Speaker pseudo IDs from test recording
const SPEAKER_A: &str = "102597e0ee793a7b";
const GM: &str = "2f09cc7c1965a203";
const SPEAKER_C: &str = "787a9547c27cd7f8";
const SPEAKER_D: &str = "d0143e57ce524cdc";

/// Load a speaker's PCM chunks from test_data, decode to mono f32.
fn load_speaker(pseudo_id: &str) -> Option<ovp_pipeline::SpeakerTrack> {
    let audio_dir = Path::new(SCENE_DIR).join("audio").join(pseudo_id);
    if !audio_dir.exists() {
        return None;
    }

    let mut chunk_paths: Vec<_> = fs::read_dir(&audio_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "pcm").unwrap_or(false))
        .collect();
    chunk_paths.sort();

    let mut raw_bytes = Vec::new();
    for path in &chunk_paths {
        raw_bytes.extend_from_slice(&fs::read(path).ok()?);
    }

    // Decode s16le stereo → mono f32
    let all_samples: Vec<f32> = raw_bytes
        .chunks_exact(2)
        .map(|pair| {
            let sample = i16::from_le_bytes([pair[0], pair[1]]);
            sample as f32 / i16::MAX as f32
        })
        .collect();
    let mono: Vec<f32> = all_samples
        .chunks_exact(2)
        .map(|frame| (frame[0] + frame[1]) / 2.0)
        .collect();

    Some(ovp_pipeline::SpeakerTrack {
        pseudo_id: pseudo_id.into(),
        samples: mono,
        sample_rate: 48000,
    })
}

/// Load all 4 speakers, return None if test data isn't available.
fn load_all_speakers() -> Option<Vec<ovp_pipeline::SpeakerTrack>> {
    let tracks: Vec<_> = [SPEAKER_A, GM, SPEAKER_C, SPEAKER_D]
        .iter()
        .filter_map(|id| load_speaker(id))
        .collect();
    if tracks.len() == 4 { Some(tracks) } else { None }
}

fn has_test_data() -> bool {
    Path::new(SCENE_DIR).join("audio").exists()
}

fn has_vad_model() -> bool {
    Path::new(VAD_MODEL).exists()
}

// ============================================================
// RMS Audio Detection
// ============================================================

#[test]
fn rms_reduces_total_audio_duration() {
    if !has_test_data() { eprintln!("SKIP: no test data"); return; }
    let tracks = load_all_speakers().unwrap();
    let resampled = resample::resample(&tracks, 48000, 16000).unwrap();

    let total_input: f32 = resampled.iter()
        .map(|s| s.samples.len() as f32 / s.sample_rate as f32)
        .sum();

    let config = RmsConfig::default();
    let segments = detect_audio_all(&config, &resampled);

    let total_active: f32 = segments.iter()
        .map(|s| s.end_time - s.start_time)
        .sum();

    // RMS should remove significant silence — at least 50% reduction
    assert!(total_active < total_input * 0.5,
        "RMS should reduce audio by >50%: {:.0}s input → {:.0}s active",
        total_input, total_active);
}

#[test]
fn rms_hot_mic_speaker_mostly_silent() {
    if !has_test_data() { eprintln!("SKIP: no test data"); return; }
    let track = load_speaker(SPEAKER_C).unwrap();
    let resampled = resample::resample(&[track], 48000, 16000).unwrap();

    let total_duration = resampled[0].samples.len() as f32 / 16000.0;
    let config = RmsConfig::default();
    let segments = detect_audio(&config, &resampled[0]);

    let active_duration: f32 = segments.iter()
        .map(|s| s.end_time - s.start_time)
        .sum();

    // Speaker C is a hot-mic track — mostly silence with brief speech.
    // Active audio should be well under 30% of total.
    assert!(active_duration < total_duration * 0.30,
        "hot-mic speaker should be <30% active: {:.0}s total, {:.0}s active ({:.0}%)",
        total_duration, active_duration, active_duration / total_duration * 100.0);
}

#[test]
fn rms_narrator_mostly_active() {
    if !has_test_data() { eprintln!("SKIP: no test data"); return; }
    let track = load_speaker(GM).unwrap();
    let resampled = resample::resample(&[track], 48000, 16000).unwrap();

    let config = RmsConfig::default();
    let segments = detect_audio(&config, &resampled[0]);

    // GM talks a lot in this scene — should have many segments
    assert!(segments.len() > 10,
        "GM should have many audio segments, got {}", segments.len());
}

#[test]
fn rms_all_speakers_have_some_audio() {
    if !has_test_data() { eprintln!("SKIP: no test data"); return; }
    let tracks = load_all_speakers().unwrap();
    let resampled = resample::resample(&tracks, 48000, 16000).unwrap();
    let config = RmsConfig::default();

    for speaker in &resampled {
        let segments = detect_audio(&config, speaker);
        assert!(!segments.is_empty(),
            "speaker {} should have at least one audio segment", speaker.pseudo_id);
    }
}

#[test]
fn rms_timestamps_within_bounds() {
    if !has_test_data() { eprintln!("SKIP: no test data"); return; }
    let tracks = load_all_speakers().unwrap();
    let resampled = resample::resample(&tracks, 48000, 16000).unwrap();
    let config = RmsConfig::default();

    for speaker in &resampled {
        let total_duration = speaker.samples.len() as f32 / speaker.sample_rate as f32;
        let segments = detect_audio(&config, speaker);

        for seg in &segments {
            assert!(seg.start_time >= 0.0,
                "speaker {} has negative start_time: {}", speaker.pseudo_id, seg.start_time);
            assert!(seg.end_time <= total_duration + 0.1,
                "speaker {} end_time {} exceeds duration {}",
                speaker.pseudo_id, seg.end_time, total_duration);
            assert!(seg.start_time < seg.end_time,
                "speaker {} has inverted times: {} >= {}",
                speaker.pseudo_id, seg.start_time, seg.end_time);
        }
    }
}

// ============================================================
// Silero VAD
// ============================================================

#[tokio::test]
async fn vad_further_reduces_hot_mic() {
    if !has_test_data() || !has_vad_model() { eprintln!("SKIP: missing test data or VAD model"); return; }
    let track = load_speaker(SPEAKER_C).unwrap();
    let resampled = resample::resample(&[track], 48000, 16000).unwrap();

    let rms_config = RmsConfig::default();
    let rms_segments = detect_audio(&rms_config, &resampled[0]);
    let rms_count = rms_segments.len();

    let vad_config = VadConfig {
        model_path: VAD_MODEL.into(),
        ..VadConfig::default()
    };
    let vad_chunks = ovp_pipeline::vad::detect_speech_from_segments(&vad_config, &rms_segments)
        .await
        .unwrap();

    // VAD should produce fewer chunks than RMS segments
    // (filters out non-speech audio that passed the energy gate)
    assert!(vad_chunks.len() <= rms_count,
        "VAD should not produce more chunks than RMS segments: {} VAD > {} RMS",
        vad_chunks.len(), rms_count);

    // Speaker C has ~7-8 real speech moments in this scene
    assert!(vad_chunks.len() < 20,
        "hot-mic speaker should have few VAD chunks (ground truth ~7-8), got {}", vad_chunks.len());
}

#[tokio::test]
async fn vad_preserves_narrator_speech() {
    if !has_test_data() || !has_vad_model() { eprintln!("SKIP: missing test data or VAD model"); return; }
    let track = load_speaker(GM).unwrap();
    let resampled = resample::resample(&[track], 48000, 16000).unwrap();

    let rms_config = RmsConfig::default();
    let rms_segments = detect_audio(&rms_config, &resampled[0]);

    let vad_config = VadConfig {
        model_path: VAD_MODEL.into(),
        ..VadConfig::default()
    };
    let vad_chunks = ovp_pipeline::vad::detect_speech_from_segments(&vad_config, &rms_segments)
        .await
        .unwrap();

    // GM talks extensively — VAD should preserve most of the speech
    assert!(vad_chunks.len() > 5,
        "GM should have many VAD chunks, got {}", vad_chunks.len());

    let total_speech: f32 = vad_chunks.iter()
        .map(|c| c.original_end - c.original_start)
        .sum();
    assert!(total_speech > 30.0,
        "GM should have >30s of detected speech, got {:.0}s", total_speech);
}

#[tokio::test]
async fn vad_chunks_have_valid_timestamps() {
    if !has_test_data() || !has_vad_model() { eprintln!("SKIP: missing test data or VAD model"); return; }
    let tracks = load_all_speakers().unwrap();
    let resampled = resample::resample(&tracks, 48000, 16000).unwrap();

    let rms_config = RmsConfig::default();
    let all_segments = detect_audio_all(&rms_config, &resampled);

    let vad_config = VadConfig {
        model_path: VAD_MODEL.into(),
        ..VadConfig::default()
    };
    let chunks = ovp_pipeline::vad::detect_speech_from_segments(&vad_config, &all_segments)
        .await
        .unwrap();

    for chunk in &chunks {
        assert!(chunk.original_start >= 0.0,
            "negative start: {}", chunk.original_start);
        assert!(chunk.original_start < chunk.original_end,
            "inverted: {} >= {}", chunk.original_start, chunk.original_end);
        assert!(!chunk.samples.is_empty(),
            "empty samples in chunk at {:.1}s", chunk.original_start);
        assert_eq!(chunk.sample_rate, 16000);
    }
}

// ============================================================
// Full Pipeline (requires Whisper endpoint)
// ============================================================

#[tokio::test]
async fn full_pipeline_scene_01() {
    if !has_test_data() || !has_vad_model() { eprintln!("SKIP: missing test data or VAD model"); return; }

    // Check if Whisper is running
    let whisper_url = std::env::var("WHISPER_URL")
        .unwrap_or_else(|_| "http://localhost:8300/v1/audio/transcriptions".into());
    let client = reqwest::Client::new();
    if client.get(whisper_url.replace("/audio/transcriptions", "/models"))
        .send().await.is_err()
    {
        eprintln!("SKIP: Whisper endpoint not available at {}", whisper_url);
        return;
    }

    let tracks = load_all_speakers().unwrap();
    let session_id = uuid::Uuid::new_v4();

    let config = ovp_pipeline::PipelineConfig {
        rms: RmsConfig::default(),
        vad: VadConfig {
            model_path: VAD_MODEL.into(),
            ..VadConfig::default()
        },
        whisper: ovp_pipeline::TranscriberConfig {
            endpoint: whisper_url,
            model: "deepdml/faster-whisper-large-v3-turbo-ct2".into(),
            language: Some("en".into()),
        },
        min_chunk_duration: 0.8,
    };

    let input = ovp_pipeline::SessionInput { session_id, tracks };
    let mut filters = ovp_pipeline::default_operators();

    let result = ovp_pipeline::process_session(&config, input, &mut filters)
        .await
        .unwrap();

    // Basic sanity
    assert!(result.segments_produced > 0, "should produce segments");
    assert!(result.duration_processed > 100.0, "should process >100s of audio");

    let kept: Vec<_> = result.segments.iter().filter(|s| !s.excluded).collect();

    // Ground truth: ~111 kept segments total
    assert!(kept.len() > 50 && kept.len() < 200,
        "expected 50-200 kept segments, got {}", kept.len());

    // Speaker C (hot-mic): ground truth ~7-8 real speech segments
    let ama_kept: Vec<_> = kept.iter()
        .filter(|s| s.speaker_pseudo_id == SPEAKER_C)
        .collect();
    assert!(ama_kept.len() >= 4 && ama_kept.len() <= 15,
        "hot-mic speaker should have 4-15 segments (ground truth ~7-8), got {}", ama_kept.len());

    // GM should have the most segments (narrator)
    let gm_kept = kept.iter().filter(|s| s.speaker_pseudo_id == GM).count();
    let max_other = [SPEAKER_A, SPEAKER_C, SPEAKER_D].iter()
        .map(|id| kept.iter().filter(|s| s.speaker_pseudo_id == *id).count())
        .max()
        .unwrap_or(0);
    assert!(gm_kept >= max_other,
        "GM should have the most segments: GM={}, max_other={}", gm_kept, max_other);

    // All segments should have non-empty text
    for seg in &kept {
        assert!(!seg.text.trim().is_empty(),
            "segment at {:.1}s has empty text", seg.start_time);
    }

    // Timestamps should be monotonically non-decreasing per speaker
    for speaker_id in &[SPEAKER_A, GM, SPEAKER_C, SPEAKER_D] {
        let speaker_segs: Vec<_> = kept.iter()
            .filter(|s| s.speaker_pseudo_id == *speaker_id)
            .collect();
        for window in speaker_segs.windows(2) {
            assert!(window[0].start_time <= window[1].start_time,
                "speaker {} has non-monotonic timestamps: {:.1} > {:.1}",
                speaker_id, window[0].start_time, window[1].start_time);
        }
    }

    // With RMS+VAD, hallucination filter should exclude very few (or zero)
    assert!(result.segments_excluded < 10,
        "with VAD, few segments should be excluded by filters, got {}", result.segments_excluded);
}
