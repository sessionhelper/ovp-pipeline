//! Whisper transcription via an external HTTP endpoint.
//!
//! Sends audio chunks to an OpenAI-compatible Whisper API and parses
//! the response into `TranscriptSegment`s with absolute timestamps.

use crate::error::{PipelineError, Result};
use crate::types::{AudioChunk, TranscriptSegment};
use uuid::Uuid;

/// Configuration for the external Whisper transcription endpoint.
#[derive(Debug, Clone)]
pub struct TranscriberConfig {
    /// Full URL of the transcription endpoint.
    /// e.g. "http://localhost:8080/v1/audio/transcriptions"
    pub endpoint: String,
    /// Whisper model name to request. e.g. "large-v3-turbo"
    pub model: String,
    /// Language hint for Whisper. `None` for auto-detect.
    pub language: Option<String>,
    /// Initial prompt to condition Whisper's language model.
    /// Anchors toward expected domain vocabulary.
    pub initial_prompt: Option<String>,
    /// Beam size for search. Higher = more accurate but slower.
    /// Default 5. Set to 1 for greedy decoding.
    pub beam_size: u32,
    /// Temperature for fallback sampling. Whisper tries each in order,
    /// falling back if compression_ratio exceeds the threshold.
    pub temperature: Vec<f32>,
    /// Segments with avg_logprob below this are likely hallucinations.
    pub hallucination_logprob_threshold: f32,
    /// Segments with no_speech_prob above this are likely silence.
    pub hallucination_no_speech_threshold: f32,
    /// Segments with compression_ratio above this are likely repetitive
    /// hallucinations ("Thank you thank you thank you...").
    pub hallucination_compression_ratio: f32,
}

/// Transcribe all audio chunks via the external Whisper endpoint.
///
/// Each chunk is sent as a separate HTTP request. Timestamps from Whisper
/// are mapped to absolute session time using `chunk.original_start`.
pub async fn transcribe(
    config: &TranscriberConfig,
    chunks: &[AudioChunk],
    session_id: Uuid,
) -> Result<Vec<TranscriptSegment>> {
    let mut all_segments = Vec::new();
    let mut segment_index = 0u32;
    let mut total_audio_secs = 0.0f32;
    let mut min_request_ms = u128::MAX;
    let mut max_request_ms = 0u128;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_duration = chunk.samples.len() as f32 / chunk.sample_rate as f32;
        total_audio_secs += chunk_duration;

        let start = std::time::Instant::now();
        let segments = transcribe_chunk(config, chunk, session_id, &mut segment_index).await?;
        let request_ms = start.elapsed().as_millis();

        min_request_ms = min_request_ms.min(request_ms);
        max_request_ms = max_request_ms.max(request_ms);

        tracing::debug!(
            stage = "whisper",
            chunk = i,
            speaker = %chunk.speaker,
            chunk_duration_secs = format_args!("{:.1}", chunk_duration),
            offset_secs = format_args!("{:.1}", chunk.original_start),
            segments_returned = segments.len(),
            request_ms = request_ms,
            "chunk transcribed"
        );

        all_segments.extend(segments);
    }

    if min_request_ms == u128::MAX { min_request_ms = 0; }

    tracing::info!(
        total_segments = all_segments.len(),
        chunks = chunks.len(),
        total_audio_secs = format_args!("{:.1}", total_audio_secs),
        min_request_ms = min_request_ms,
        max_request_ms = max_request_ms,
        "transcription complete"
    );

    Ok(all_segments)
}

/// Transcribe a single audio chunk.
#[cfg(feature = "transcribe")]
async fn transcribe_chunk(
    config: &TranscriberConfig,
    chunk: &AudioChunk,
    session_id: Uuid,
    segment_index: &mut u32,
) -> Result<Vec<TranscriptSegment>> {
    let encode_start = std::time::Instant::now();
    let wav_bytes = encode_wav(chunk)?;
    let encode_ms = encode_start.elapsed().as_millis();

    let client = reqwest::Client::new();

    let mut form = reqwest::multipart::Form::new()
        .text("model", config.model.clone())
        .text("response_format", "verbose_json")
        .text("beam_size", config.beam_size.to_string())
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav_bytes.clone())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| PipelineError::Transcribe(e.to_string()))?,
        );

    if let Some(ref lang) = config.language {
        form = form.text("language", lang.clone());
    }
    if let Some(ref prompt) = config.initial_prompt {
        form = form.text("initial_prompt", prompt.clone());
    }
    if !config.temperature.is_empty() {
        // The OpenAI-compatible endpoint accepts a single float. When
        // multiple fallback temperatures are configured, send only the
        // first one. Whisper internally falls back to higher temperatures
        // if the compression ratio is poor; the API does not expose the
        // fallback list directly.
        form = form.text("temperature", format!("{:.1}", config.temperature[0]));
    }

    let send_start = std::time::Instant::now();
    let response = client
        .post(&config.endpoint)
        .multipart(form)
        .send()
        .await
        .map_err(|e| PipelineError::Transcribe(format!("HTTP request failed: {}", e)))?;
    let ttfb_ms = send_start.elapsed().as_millis(); // time to first byte — includes inference

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable>".into());
        return Err(PipelineError::Transcribe(format!(
            "Whisper endpoint returned {}: {}",
            status, body
        )));
    }

    let parse_start = std::time::Instant::now();
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| PipelineError::Transcribe(format!("failed to parse response: {}", e)))?;
    let parse_ms = parse_start.elapsed().as_millis();

    // Parse OpenAI-compatible verbose_json response
    let whisper_segments = body["segments"]
        .as_array()
        .ok_or_else(|| PipelineError::Transcribe("response missing 'segments' array".into()))?;

    let segments: Vec<TranscriptSegment> = whisper_segments
        .iter()
        .filter_map(|seg| {
            let text = seg["text"].as_str()?.trim().to_string();
            if text.is_empty() {
                return None;
            }
            let start = seg["start"].as_f64()? as f32;
            let end = seg["end"].as_f64()? as f32;

            let avg_logprob = seg["avg_logprob"].as_f64().unwrap_or(-1.0) as f32;
            let no_speech_prob = seg["no_speech_prob"].as_f64().unwrap_or(0.0) as f32;
            let compression_ratio = seg["compression_ratio"].as_f64().unwrap_or(1.0) as f32;

            // Filter likely hallucinations
            if avg_logprob < config.hallucination_logprob_threshold {
                tracing::debug!(
                    speaker = %chunk.speaker,
                    text = %text,
                    avg_logprob,
                    "hallucination_filtered: low logprob"
                );
                return None;
            }
            if no_speech_prob > config.hallucination_no_speech_threshold {
                tracing::debug!(
                    speaker = %chunk.speaker,
                    text = %text,
                    no_speech_prob,
                    "hallucination_filtered: high no_speech_prob"
                );
                return None;
            }
            if compression_ratio > config.hallucination_compression_ratio {
                tracing::debug!(
                    speaker = %chunk.speaker,
                    text = %text,
                    compression_ratio,
                    "hallucination_filtered: high compression_ratio"
                );
                return None;
            }

            // Map Whisper-relative timestamps to absolute session time
            let absolute_start = chunk.original_start + start;
            let absolute_end = chunk.original_start + end;

            let idx = *segment_index;
            *segment_index += 1;

            Some(TranscriptSegment {
                id: Uuid::new_v4(),
                session_id,
                segment_index: idx,
                speaker_pseudo_id: chunk.speaker.clone(),
                start_time: absolute_start,
                end_time: absolute_end,
                text: text.clone(),
                original_text: text,
                confidence: Some(avg_logprob),
                beat_id: None,
                chunk_group: None,
                talk_type: None,
                excluded: false,
                exclude_reason: None,
            })
        })
        .collect();

    tracing::trace!(
        stage = "whisper",
        speaker = %chunk.speaker,
        encode_ms = encode_ms,
        ttfb_ms = ttfb_ms,
        parse_ms = parse_ms,
        payload_bytes = wav_bytes.len(),
        segments = segments.len(),
        "chunk request breakdown"
    );

    Ok(segments)
}

/// Encode f32 mono samples as a WAV byte buffer for the HTTP upload.
#[cfg(feature = "transcribe")]
fn encode_wav(chunk: &AudioChunk) -> Result<Vec<u8>> {
    let sample_rate = chunk.sample_rate;
    let num_samples = chunk.samples.len() as u32;
    let bits_per_sample: u16 = 16;
    let num_channels: u16 = 1;
    let byte_rate = sample_rate * (bits_per_sample as u32 / 8) * num_channels as u32;
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples * (bits_per_sample as u32 / 8);

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 [-1.0, 1.0] to s16le
    for &sample in &chunk.samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let s16 = (clamped * i16::MAX as f32) as i16;
        buf.extend_from_slice(&s16.to_le_bytes());
    }

    Ok(buf)
}

/// Stub transcription when the `transcribe` feature is disabled.
#[cfg(not(feature = "transcribe"))]
async fn transcribe_chunk(
    _config: &TranscriberConfig,
    _chunk: &AudioChunk,
    _session_id: Uuid,
    _segment_index: &mut u32,
) -> Result<Vec<TranscriptSegment>> {
    Err(PipelineError::Transcribe(
        "transcription requires the `transcribe` feature".into(),
    ))
}
