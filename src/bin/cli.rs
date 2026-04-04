//! CLI scaffold for testing the ovp-pipeline against real audio data.
//!
//! Two subcommands:
//! - `ingest`: Convert Craig multi-track FLAC recordings to the raw PCM
//!   chunk layout the bot writes to S3.
//! - `run`: Load prepared chunks and run the full transcription pipeline.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use clap::{Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};
use symphonia_core::audio::SampleBuffer;
use symphonia_core::codecs::{Decoder, DecoderOptions};
use symphonia_core::formats::FormatOptions;
use symphonia_core::io::MediaSourceStream;
use symphonia_core::meta::MetadataOptions;
use symphonia_core::probe::Hint;
use uuid::Uuid;

use ovp_pipeline::{
    default_filters, process_session, PipelineConfig, SessionInput, SpeakerTrack, TranscriberConfig,
    VadConfig,
};

/// Pipeline test scaffold for TTRPG audio transcription.
#[derive(Parser)]
#[command(name = "ovp-cli")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert Craig FLAC recording to raw PCM chunk layout.
    Ingest {
        /// Path to Craig extracted directory (contains .flac files + info.txt).
        #[arg(long)]
        input_dir: PathBuf,
        /// Output directory for the prepared session data.
        #[arg(long)]
        output_dir: PathBuf,
        /// Skip this many seconds from the start of each track.
        #[arg(long, default_value = "0")]
        start: f32,
        /// Limit audio to this many seconds (after start offset).
        #[arg(long)]
        duration: Option<f32>,
    },
    /// Run the transcription pipeline against prepared session data.
    Run {
        /// Path to a prepared session directory (from `ingest`).
        #[arg(long)]
        session_dir: PathBuf,
        /// Whisper HTTP endpoint URL.
        #[arg(long, default_value = "http://localhost:8300/v1/audio/transcriptions")]
        whisper_url: String,
        /// Whisper model name.
        #[arg(long, default_value = "large-v3-turbo")]
        model: String,
        /// Path to Silero VAD ONNX model.
        #[arg(long, default_value = "models/silero_vad_v6.onnx")]
        vad_model: PathBuf,
        /// Output JSON file (default: stdout).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

// --- Craig info.txt parsing ---

struct CraigInfo {
    guild_name: String,
    guild_id: u64,
    channel_name: String,
    start_time: String,
    tracks: Vec<TrackInfo>,
}

struct TrackInfo {
    username: String,
    user_id: u64,
}

fn parse_info_txt(path: &Path) -> CraigInfo {
    let content = fs::read_to_string(path).expect("Failed to read info.txt");
    let mut guild_name = String::new();
    let mut guild_id = 0u64;
    let mut channel_name = String::new();
    let mut start_time = String::new();
    let mut tracks = Vec::new();
    let mut in_tracks = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Guild:") {
            // "Guild:\t\tDark Heresy 2e TTRPG (557647168052264962)"
            let val = trimmed.trim_start_matches("Guild:").trim();
            if let Some(paren_start) = val.rfind('(') {
                guild_name = val[..paren_start].trim().to_string();
                guild_id = val[paren_start + 1..val.len() - 1]
                    .parse()
                    .expect("Invalid guild_id");
            }
        } else if trimmed.starts_with("Channel:") {
            let val = trimmed.trim_start_matches("Channel:").trim();
            if let Some(paren_start) = val.rfind('(') {
                channel_name = val[..paren_start].trim().to_string();
            } else {
                channel_name = val.to_string();
            }
        } else if trimmed.starts_with("Start time:") {
            start_time = trimmed
                .trim_start_matches("Start time:")
                .trim()
                .to_string();
        } else if trimmed == "Tracks:" {
            in_tracks = true;
        } else if in_tracks && !trimmed.is_empty() {
            // "\tfenixwhitewood#0 (529123741276438528)"
            if let Some(paren_start) = trimmed.rfind('(') {
                let name_part = trimmed[..paren_start].trim();
                let username = name_part.split('#').next().unwrap_or(name_part).to_string();
                let user_id: u64 = trimmed[paren_start + 1..trimmed.len() - 1]
                    .parse()
                    .expect("Invalid user_id in tracks");
                tracks.push(TrackInfo { username, user_id });
            }
        }
    }

    CraigInfo {
        guild_name,
        guild_id,
        channel_name,
        start_time,
        tracks,
    }
}

// --- Pseudonymization (matches bot's SHA256 derivation) ---

fn pseudonymize(user_id: u64) -> String {
    let hash = Sha256::digest(user_id.to_string());
    hex::encode(&hash[..8])
}

// --- FLAC decoding via symphonia ---

/// Decode a FLAC file to raw s16le PCM bytes (preserving stereo).
/// Skips `skip_samples` frames from the start, then takes up to `max_samples` frames.
/// Returns the raw bytes and the number of frames written.
fn decode_flac(path: &Path, skip_samples: usize, max_samples: Option<usize>) -> (Vec<u8>, usize) {
    let file = std::fs::File::open(path).expect("Failed to open FLAC file");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    // symphonia's default Probe has no codecs registered — build one
    // with the FLAC reader explicitly.
    let mut probe = symphonia_core::probe::Probe::default();
    probe.register_all::<symphonia_bundle_flac::FlacReader>();

    let probed = probe
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .expect("Failed to probe FLAC file");

    let mut format = probed.format;
    let track = format.default_track().expect("No default track").clone();
    let mut decoder = symphonia_bundle_flac::FlacDecoder::try_new(&track.codec_params, &DecoderOptions::default())
        .expect("Failed to create FLAC decoder");

    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2);

    let mut pcm_bytes: Vec<u8> = Vec::new();
    let mut frames_read = 0usize;
    let mut frames_written = 0usize;
    let max_frames = max_samples.unwrap_or(usize::MAX);

    loop {
        if frames_written >= max_frames {
            break;
        }

        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia_core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => panic!("Error reading packet: {}", e),
        };

        if packet.track_id() != track.id {
            continue;
        }

        let decoded = decoder.decode(&packet).expect("Failed to decode packet");
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        // Skip frames until we've passed the start offset
        if frames_read + num_frames <= skip_samples {
            frames_read += num_frames;
            continue;
        }

        let mut sample_buf = SampleBuffer::<i16>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Calculate which portion of this packet to use
        let skip_in_packet = if frames_read < skip_samples {
            skip_samples - frames_read
        } else {
            0
        };
        let available = num_frames - skip_in_packet;
        let frames_to_take = (max_frames - frames_written).min(available);

        let start_sample = skip_in_packet * channels;
        let end_sample = start_sample + frames_to_take * channels;
        for &s in &samples[start_sample..end_sample] {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }

        frames_read += num_frames;
        frames_written += frames_to_take;
    }

    (pcm_bytes, frames_written)
}

// --- Metadata types for JSON output ---

#[derive(Serialize)]
struct SessionMeta {
    session_id: String,
    guild_id: u64,
    guild_name: String,
    channel_name: String,
    started_at: String,
    created_at: String,
    audio_format: AudioFormat,
    participants: Vec<ParticipantMeta>,
}

#[derive(Serialize)]
struct AudioFormat {
    sample_rate: u32,
    channels: u16,
    bit_depth: u16,
    encoding: String,
    chunk_size_bytes: usize,
}

#[derive(Serialize)]
struct ParticipantMeta {
    pseudo_id: String,
    track_index: usize,
}

#[derive(Serialize)]
struct ConsentRecord {
    session_id: String,
    participants: Vec<ConsentEntry>,
}

#[derive(Serialize)]
struct ConsentEntry {
    pseudo_id: String,
    consent_scope: String,
    no_llm_training: bool,
    no_public_release: bool,
}

// --- Ingest command ---

const CHUNK_SIZE: usize = 5 * 1024 * 1024; // 5MB, matches bot

fn cmd_ingest(input_dir: &Path, output_dir: &Path, start: f32, duration: Option<f32>) {
    let info = parse_info_txt(&input_dir.join("info.txt"));

    let session_id = Uuid::new_v4();
    println!("Session ID: {}", session_id);
    println!(
        "Guild: {} ({})",
        info.guild_name, info.guild_id
    );
    println!("Tracks: {}", info.tracks.len());

    let skip_samples = (start * 48000.0) as usize;
    let max_samples = duration.map(|d| (d * 48000.0) as usize);
    if start > 0.0 {
        println!("Start offset: {:.0}s", start);
    }
    if let Some(d) = duration {
        println!("Duration limit: {:.0}s", d);
    }

    fs::create_dir_all(output_dir).expect("Failed to create output dir");

    let mut participants = Vec::new();

    for (i, track) in info.tracks.iter().enumerate() {
        let pseudo_id = pseudonymize(track.user_id);
        println!(
            "\n[{}/{}] {} → {}",
            i + 1,
            info.tracks.len(),
            track.username,
            pseudo_id
        );

        // Find the matching FLAC file: "{N}-{username}.flac"
        let flac_name = format!("{}-{}.flac", i + 1, track.username);
        let flac_path = input_dir.join(&flac_name);
        if !flac_path.exists() {
            eprintln!("  Warning: {} not found, skipping", flac_name);
            continue;
        }

        // Decode FLAC to raw PCM
        print!("  Decoding FLAC...");
        std::io::stdout().flush().unwrap();
        let (pcm_bytes, frames) = decode_flac(&flac_path, skip_samples, max_samples);
        let duration_secs = frames as f32 / 48000.0;
        println!(
            " {:.1}s, {:.1} MB",
            duration_secs,
            pcm_bytes.len() as f64 / 1_048_576.0
        );

        // Split into chunks and write
        let audio_dir = output_dir.join("audio").join(&pseudo_id);
        fs::create_dir_all(&audio_dir).expect("Failed to create audio dir");

        let num_chunks = (pcm_bytes.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for (seq, chunk_data) in pcm_bytes.chunks(CHUNK_SIZE).enumerate() {
            let chunk_path = audio_dir.join(format!("chunk_{:04}.pcm", seq));
            fs::write(&chunk_path, chunk_data).expect("Failed to write chunk");
        }
        println!("  Wrote {} chunks to audio/{}/", num_chunks, pseudo_id);

        participants.push((pseudo_id, i));
    }

    // Write meta.json
    let meta = SessionMeta {
        session_id: session_id.to_string(),
        guild_id: info.guild_id,
        guild_name: info.guild_name,
        channel_name: info.channel_name,
        started_at: info.start_time.clone(),
        created_at: Utc::now().to_rfc3339(),
        audio_format: AudioFormat {
            sample_rate: 48000,
            channels: 2,
            bit_depth: 16,
            encoding: "s16le".to_string(),
            chunk_size_bytes: CHUNK_SIZE,
        },
        participants: participants
            .iter()
            .map(|(pid, idx)| ParticipantMeta {
                pseudo_id: pid.clone(),
                track_index: *idx,
            })
            .collect(),
    };
    let meta_path = output_dir.join("meta.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .expect("Failed to write meta.json");
    println!("\nWrote meta.json");

    // Write consent.json — real player audio, restricted
    let consent = ConsentRecord {
        session_id: session_id.to_string(),
        participants: participants
            .iter()
            .map(|(pid, _)| ConsentEntry {
                pseudo_id: pid.clone(),
                consent_scope: "full".to_string(),
                no_llm_training: true,
                no_public_release: true,
            })
            .collect(),
    };
    let consent_path = output_dir.join("consent.json");
    fs::write(
        &consent_path,
        serde_json::to_string_pretty(&consent).unwrap(),
    )
    .expect("Failed to write consent.json");
    println!("Wrote consent.json (no_llm_training: true, no_public_release: true)");

    println!("\nDone. Session prepared at {}", output_dir.display());
}

// --- Run command ---

fn cmd_run(session_dir: &Path, whisper_url: &str, model: &str, vad_model: &Path, output: Option<&Path>) {
    // Read meta.json
    let meta_path = session_dir.join("meta.json");
    let meta_str = fs::read_to_string(&meta_path).expect("Failed to read meta.json");
    let meta: serde_json::Value =
        serde_json::from_str(&meta_str).expect("Failed to parse meta.json");

    let session_id: Uuid = meta["session_id"]
        .as_str()
        .expect("No session_id in meta.json")
        .parse()
        .expect("Invalid session_id UUID");

    println!("Session: {}", session_id);

    // Load speaker tracks from audio/ subdirectories
    let audio_dir = session_dir.join("audio");
    let mut tracks = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(&audio_dir)
        .expect("No audio/ directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let pseudo_id = entry.file_name().to_string_lossy().to_string();

        // Read and concatenate PCM chunks, then decode to mono f32.
        // The pipeline takes mono f32 samples — byte decoding and
        // stereo downmix happen here in the caller, not in the library.
        let mut chunk_paths: Vec<PathBuf> = fs::read_dir(entry.path())
            .expect("Failed to read speaker dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "pcm").unwrap_or(false))
            .collect();
        chunk_paths.sort();

        let mut raw_bytes = Vec::new();
        for chunk_path in &chunk_paths {
            let data = fs::read(chunk_path).expect("Failed to read chunk");
            raw_bytes.extend_from_slice(&data);
        }

        // Decode s16le stereo → mono f32
        let channels = 2usize;
        let all_samples: Vec<f32> = raw_bytes
            .chunks_exact(2)
            .map(|pair| {
                let sample = i16::from_le_bytes([pair[0], pair[1]]);
                sample as f32 / i16::MAX as f32
            })
            .collect();
        let mono_samples: Vec<f32> = all_samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();

        let duration_secs = mono_samples.len() as f32 / 48000.0;
        println!(
            "  Speaker {}: {} chunks, {:.1}s, {:.1} MB raw",
            pseudo_id,
            chunk_paths.len(),
            duration_secs,
            raw_bytes.len() as f64 / 1_048_576.0
        );

        tracks.push(SpeakerTrack {
            pseudo_id,
            samples: mono_samples,
            sample_rate: 48000,
        });
    }

    println!("\nRunning pipeline...");
    println!("  Whisper: {}", whisper_url);
    println!("  Model: {}", model);

    println!("  VAD model: {}", vad_model.display());

    let config = PipelineConfig {
        rms: ovp_pipeline::ad::RmsConfig::default(),
        vad: VadConfig {
            model_path: vad_model.to_path_buf(),
            ..VadConfig::default()
        },
        whisper: TranscriberConfig {
            endpoint: whisper_url.to_string(),
            model: model.to_string(),
            language: Some("en".to_string()),
        },
        min_chunk_duration: 0.8,
    };

    let input = SessionInput {
        session_id,
        tracks,
    };

    let mut filters = default_filters();

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let result = rt.block_on(async { process_session(&config, input, &mut filters).await });

    match result {
        Ok(result) => {
            println!("\nPipeline complete:");
            println!("  Segments produced: {}", result.segments_produced);
            println!("  Segments excluded: {}", result.segments_excluded);
            println!("  Scenes detected:  {}", result.scenes_detected);
            println!("  Duration processed: {:.1}s", result.duration_processed);

            let json = serde_json::to_string_pretty(&result).expect("Failed to serialize result");

            if let Some(out_path) = output {
                fs::write(out_path, &json).expect("Failed to write output");
                println!("\nResults written to {}", out_path.display());
            } else {
                println!("\n{}", json);
            }
        }
        Err(e) => {
            eprintln!("\nPipeline error: {}", e);
            std::process::exit(1);
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Ingest {
            input_dir,
            output_dir,
            start,
            duration,
        } => cmd_ingest(&input_dir, &output_dir, start, duration),
        Command::Run {
            session_dir,
            whisper_url,
            model,
            vad_model,
            output,
        } => cmd_run(&session_dir, &whisper_url, &model, &vad_model, output.as_deref()),
    }
}
