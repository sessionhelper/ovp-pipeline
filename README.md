# chronicle-pipeline

Pure Rust library for turning multi-speaker voice audio into structured,
enriched transcripts. Used as a library dependency by `chronicle-worker`
(the orchestrator) and as a standalone CLI binary (`chronicle-cli`) for
local testing against real audio.

Input: per-speaker mono f32 PCM sample streams.
Output: transcript segments with speaker attribution, scene boundaries,
narrative beats, and hallucination-filtered content.

## Pipeline stages

```
Input PCM (per-speaker)
       │
       ▼
 ┌──────────┐   ┌────────┐   ┌──────────┐   ┌──────────────┐
 │  Resample│──►│  RMS   │──►│   VAD    │──►│   Whisper    │
 │ (if !48k)│   │ filter │   │(stateful)│   │(transcribe)  │
 └──────────┘   └────────┘   └──────────┘   └──────┬───────┘
                                                   │ segments
                                                   ▼
                        ┌──────────────────────────────────────┐
                        │  Operator chain (stateful, in order) │
                        │                                      │
                        │  HallucinationFilter                 │
                        │  MetatalkOperator  (IC/OOC)          │
                        │  SceneOperator     (silence chunker) │
                        │  BeatOperator      (optional LLM)    │
                        └──────────────┬───────────────────────┘
                                       │
                                       ▼
                              PipelineResult
                              ├── segments
                              ├── scenes
                              └── beats
```

See [`docs/architecture.md`](docs/architecture.md) for the full stage-by-stage
breakdown, configuration surfaces, and streaming vs batch modes.

## Modes

**Batch mode** — `process_session` takes the full audio for a session
and produces a final `PipelineResult` in one call. Used for offline
reprocessing and test harnesses.

**Streaming mode** — `StreamingPipeline::feed_chunk` incrementally
consumes per-speaker chunks as they arrive and emits segments as they're
ready, keeping stateful VadSession state across chunk boundaries so
utterances that straddle a chunk boundary aren't split. This is what
`chronicle-worker` uses for real-time session processing.

## Quick start

```bash
# Build the library
cargo build --release

# Build the CLI binary for local testing
cargo build --release --features cli
./target/release/chronicle-cli --help
```

The CLI handles Craig multi-track FLAC conversion and direct PCM input
for development. It's **not** part of the library API — it's a test
scaffold for operating on recorded captures.

## Features

- `vad` — include Silero VAD (default for worker use)
- `transcribe` — include Whisper integration (default for worker use)
- `cli` — build the `chronicle-cli` binary (dev only)

Feature selection matters for worker builds because Whisper pulls in
heavy dependencies.

## Dependencies

- **Silero VAD** — ONNX model baked into the build (`models/silero_vad_v6.onnx`, ~1.2 MB)
- **Whisper** — via a separate whisper HTTP service (not embedded); configured via
  `WhisperConfig`
- **Optional LLM** — for beat detection, also via HTTP

No I/O happens in the library beyond HTTP calls to Whisper and LLM
endpoints. Storage, database, and network event handling all live in
`chronicle-worker` / `chronicle-data-api`.

## Configuration surfaces

- `PipelineConfig` — global settings (sample rate, VAD thresholds, whisper config, operator chain)
- `VadConfig` — speech detection thresholds
- `RmsConfig` — audio activity detection thresholds
- `TranscriberConfig` — Whisper model, temperature, logprob threshold, initial prompt
- `BeatConfig`, `SceneConfig`, `MetatalkConfig` — per-operator tuning

See [`docs/architecture.md`](docs/architecture.md) for the full option catalog.

## Tests

```bash
cargo test
```

Integration tests exercise the pipeline end-to-end against the synthetic
captures in `test-data/` (gitignored, staged separately). See
`tests/integration.rs`.

## Related

- [`chronicle-worker`](https://github.com/sessionhelper/chronicle-worker) — the service that invokes this library on real sessions
- [`sessionhelper-hub/ARCHITECTURE.md`](https://github.com/sessionhelper/sessionhelper-hub/blob/main/ARCHITECTURE.md) — cross-service data flow
- [`sessionhelper-hub/SPEC.md`](https://github.com/sessionhelper/sessionhelper-hub/blob/main/SPEC.md) — OVP program spec (this pipeline delivers G3 and feeds G4)
