# chronicle-pipeline

Pure Rust library for turning multi-speaker TTRPG voice audio into
structured, enriched transcripts. Used as a library dependency by
`chronicle-worker` (the orchestrator that drives real sessions) and as a
standalone binary (`chronicle-cli`) for testing against recorded
captures locally.

Input: per-speaker `f32` PCM sample streams at any sample rate.
Output: `PipelineResult` with transcript segments (speaker-attributed,
timestamped, hallucination-filtered), scene boundaries, and narrative
beats.

## Pipeline layers

The pipeline runs in two conceptual layers, each composed of multiple
stateful stages. Both the batch entry point (`process_session` in
`src/pipeline.rs`) and the streaming entry point (`StreamingPipeline::feed_chunk`
in `src/streaming.rs`) run the same stages in the same order.

```
                  per-speaker f32 PCM samples (any rate)
                                  │
                ┌─────────────────┼─────────────────┐
                │   Audio layer   │                 │
                │                 ▼                 │
                │        Resample to 16 kHz         │   stateless
                │         (rubato, gated by         │
                │          `transcribe`)            │
                │                 │                 │
                │                 ▼                 │
                │      RMS energy gate              │   stateless
                │   (carves AudioSegments out of    │
                │     silent regions)               │
                │                 │                 │
                │                 ▼                 │
                │    Silero VAD (v6 ONNX, gated     │   STATEFUL
                │    by `vad`)                      │   (LSTM h/c state
                │    AudioSegment → AudioChunk      │    in VadSession,
                │                                   │    carries across
                │                                   │    feed_chunk calls)
                └─────────────────┼─────────────────┘
                                  │
                                  ▼
                          Whisper HTTP                 STATEFUL
                       (gated by `transcribe`)         (hallucination /
                   AudioChunk → TranscriptSegment       logprob / compression
                                                        thresholds carry)
                                  │
                ┌─────────────────┼─────────────────┐
                │ Enrichment layer│                 │
                │                 ▼                 │
                │   HallucinationOperator           │   STATEFUL
                │   (frequency counters; marks      │
                │    repeats and known whisper      │
                │    hallucinations excluded)       │
                │                 │                 │
                │                 ▼                 │
                │   MetatalkOperator                │   stateless
                │   (adds talk_type: ic / ooc)      │
                │                 │                 │
                │                 ▼                 │
                │   SceneOperator (mechanical)      │   STATEFUL
                │   (silence-gap chunker, adds      │
                │    chunk_group + scene metadata)  │
                │                 │                 │
                │                 ▼                 │
                │   BeatOperator (optional, LLM)    │   STATEFUL
                │   (narrative beat extraction via  │   (LLM context)
                │    LLM endpoint)                  │
                │                 │                 │
                │                 ▼                 │
                │   SceneOperator (optional, LLM)   │   STATEFUL
                │   (LLM-based scene grouping)      │   (LLM context)
                └─────────────────┼─────────────────┘
                                  │
                                  ▼
                          PipelineResult
                          ├── segments
                          ├── scenes
                          └── beats
```

**Audio layer (stages 1-3)** turns raw PCM into speech regions. Resample
normalizes to 16 kHz (Whisper's native input rate), RMS gates out silent
regions, VAD classifies the remainder as speech or non-speech using
Silero's ONNX model. The VAD stage is stateful — the LSTM h/c state is
carried across `feed_chunk` calls via a per-speaker `VadSession` so
utterances that straddle a chunk boundary aren't split.

**Transcription (stage 4)** turns speech regions into `TranscriptSegment`s
via a Whisper HTTP service (endpoint and model configured via
`TranscriberConfig`). Whisper itself is stateful inside the pipeline —
hallucination / logprob / compression thresholds carry across calls so
patterns across a session affect the filter.

**Enrichment layer (stages 5+)** operates on `TranscriptSegment`s via the
`Operator` trait defined in [`src/operators/mod.rs`](src/operators/mod.rs).
Every operator receives `&mut TranscriptSegment` and can edit metadata,
flag for exclusion, or attach grouping markers — **operators cannot see
pre-transcription audio**. The mechanical `SceneOperator` and
`HallucinationOperator` are the defaults, with `BeatOperator` and an
LLM-based `SceneOperator` as optional LLM-driven variants.

See [`docs/architecture.md`](docs/architecture.md) for the full
stage-by-stage breakdown, configuration surfaces, and detailed streaming
vs batch semantics.

## The `Operator` trait

Defined at `src/operators/mod.rs`. Stages in the enrichment layer
implement it — the audio-layer stages (Resample, RMS, VAD, Whisper) do
**not**. The trait method is roughly:

```rust
fn on_segment(&mut self, segment: &mut TranscriptSegment);
```

Operators are stateful across segments (can hold counters, context, or
an LLM conversation scratch) and run in the order configured in
`PipelineConfig`. The current default order is:
`HallucinationOperator → MetatalkOperator → SceneOperator`,
with optional LLM `BeatOperator` and LLM `SceneOperator` appended when
an LLM endpoint is configured.

## Modes

**Batch mode** — `process_session` (in `src/pipeline.rs`) takes the full
per-speaker audio for a session and produces a final `PipelineResult` in
one call. Used for offline reprocessing, test harnesses, and the
`chronicle-cli` binary.

**Streaming mode** — `StreamingPipeline::feed_chunk` (in `src/streaming.rs`)
incrementally consumes per-speaker chunks as they arrive and emits
segments as they're ready, keeping stateful `VadSession` state across
chunk boundaries. `finalize()` drains the pipeline and runs operators
over the complete transcript so enrichment stages can see the full
conversation arc. This is the mode `chronicle-worker` uses for real-time
session processing.

## Quick start

```bash
# Build the library (default features: vad + transcribe)
cargo build --release

# Build the CLI binary for local testing against recorded audio
cargo build --release --features cli
./target/release/chronicle-cli --help
```

The CLI handles Craig multi-track FLAC conversion (via
`symphonia-bundle-flac`) and direct PCM input for development. It's
**not** part of the library API — it's a test scaffold for operating on
recorded captures.

## Feature flags

```toml
[features]
default = ["vad", "transcribe"]
vad = ["dep:ort", "dep:ndarray"]           # Silero VAD (ONNX runtime)
transcribe = ["dep:reqwest", "dep:rubato"] # Whisper HTTP + audio resampler
opus = []                                   # Opus support hooks
cli = [...]                                 # chronicle-cli binary (FLAC, clap, tracing-subscriber)
full = ["vad", "transcribe", "opus"]        # everything
```

Note: the `transcribe` feature gates both Whisper transcription AND the
Rubato resampler used in the audio layer, because both sit on the same
`reqwest` + `rubato` dependency set. Disabling `transcribe` drops the
Resample stage and stage 4 together — the pipeline becomes a PCM → VAD
pass-through with no transcripts or enrichment.

LLM-driven operators (`BeatOperator`, LLM `SceneOperator`) also live
behind the `transcribe` feature because they share the `reqwest` HTTP
client. If no LLM endpoint is configured in `PipelineConfig`, they are
not instantiated regardless of build features.

## Configuration surfaces

- `PipelineConfig` — global settings: target sample rate, operator order, whisper + LLM endpoints
- `RmsConfig` — RMS energy gate thresholds
- `VadConfig` — Silero speech detection thresholds and chunk sizing
- `TranscriberConfig` — Whisper endpoint, model, temperature, logprob threshold, compression ratio, no-speech probability, initial prompt
- `BeatConfig`, `SceneConfig`, `MetatalkConfig` — per-operator tuning

See [`docs/architecture.md`](docs/architecture.md) for the full option
catalog.

## Dependencies

- **Silero VAD v6** — ONNX model baked into the build at
  `models/silero_vad_v6.onnx` (~1.2 MB), loaded via `ort`
- **Whisper** — external HTTP service (not embedded), called via `reqwest`
- **Rubato** — audio resampling to 16 kHz
- **Optional LLM** — external HTTP endpoint for beat detection and LLM
  scene grouping, also via `reqwest`

No I/O happens in the library beyond HTTP calls to Whisper and the
optional LLM. Storage, database, and network event handling all live in
`chronicle-worker` and `chronicle-data-api`.

## Tests

```bash
cargo test
```

Integration tests in `tests/integration.rs` exercise the pipeline
end-to-end against synthetic captures. Regression corpus for the audio
layer lives in `test-data/` (gitignored, staged separately).

## Related

- [`chronicle-worker`](https://github.com/sessionhelper/chronicle-worker) — the event-driven orchestrator that invokes this library on real sessions
- [`chronicle-data-api`](https://github.com/sessionhelper/chronicle-data-api) — stores the transcripts this pipeline produces
- [`sessionhelper-hub/ARCHITECTURE.md`](https://github.com/sessionhelper/sessionhelper-hub/blob/main/ARCHITECTURE.md) — cross-service data flow
- [`sessionhelper-hub/SPEC.md`](https://github.com/sessionhelper/sessionhelper-hub/blob/main/SPEC.md) — OVP program spec (this pipeline delivers G3 and feeds G4)
