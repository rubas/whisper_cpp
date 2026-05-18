# whisper_cpp

`whisper_cpp` is an Elixir library for running OpenAI Whisper speech-to-text
models inside the BEAM via [whisper.cpp](https://github.com/ggerganov/whisper.cpp).
It calls whisper.cpp's C API through a Rustler NIF (using the
[`whisper-rs`](https://github.com/tazz4843/whisper-rs) crate), so Elixir code can
transcribe WAV files or f32 PCM buffers without spawning a `whisper-cli`
subprocess, writing temporary files, or shipping Python.

Diarization-driven workflows are first-class: decode a master WAV once, then
run many short transcribe calls over per-turn slices with absolute timestamps
preserved across the original audio timeline.

## Installation

```elixir
def deps do
  [{:whisper_cpp, "~> 0.1.0"}]
end
```

Installation downloads a precompiled NIF artefact matching your target triple
from the project's GitHub releases. No Rust toolchain or CMake is needed on
the consumer side.

### Source builds

Set `WHISPER_CPP_BUILD=1` in your environment (or
`config :rustler_precompiled, :force_build, whisper_cpp: true`) to compile
from source. Base requirements:

- Rust toolchain (`rustup`, stable >= 1.85)
- `cmake`, a C++17 compiler, `make`
- `pkg-config`, `libclang` (for `bindgen`)

Backend-specific SDKs are only needed when you opt in to that backend:

| Cargo feature  | Backend                              | Extra SDK at build time                      |
| -------------- | ------------------------------------ | -------------------------------------------- |
| _(none)_       | Pure CPU (SIMD: AVX2 / NEON)         | -                                            |
| `cuda`         | NVIDIA GPU via CUDA                  | CUDA toolkit 12+                             |
| `hipblas`      | AMD GPU via ROCm hipBLAS             | ROCm 6.x, `hipblas-dev`, `rocblas-dev`       |
| `vulkan`       | Cross-vendor GPU via Vulkan          | Vulkan loader + headers                      |
| `metal`        | Apple Silicon GPU                    | Xcode CLT                                    |
| `coreml`       | Apple Neural Engine (encoder)        | Xcode + Core ML tools                        |
| `intel-sycl`   | Intel Arc / Xe via oneAPI SYCL       | Intel oneAPI Base Toolkit                    |
| `openblas`     | CPU SGEMM acceleration               | `libopenblas-dev`                            |
| `openmp`       | OpenMP CPU multi-threading           | `libgomp` / `libomp`                         |

## Backends

The published Hex package ships precompiled artefacts. `rustler_precompiled`
picks the right one at install time based on the target triple plus the
optional `WHISPER_CPP_VARIANT` environment variable.

| Target triple                 | Default backend | Variant artefacts             | Selection                       |
| ----------------------------- | --------------- | ----------------------------- | ------------------------------- |
| `aarch64-apple-darwin`        | Metal           | -                             | auto                            |
| `x86_64-unknown-linux-gnu`    | CPU             | `cuda`, `hipblas`             | `WHISPER_CPP_VARIANT=<name>`    |
| `aarch64-unknown-linux-gnu`   | CPU             | `cuda`                        | `WHISPER_CPP_VARIANT=<name>`    |

For example, to pull the CUDA build on an x86_64 Linux host:

```bash
WHISPER_CPP_VARIANT=cuda mix deps.compile whisper_cpp
```

For the AMD ROCm build:

```bash
WHISPER_CPP_VARIANT=hipblas mix deps.compile whisper_cpp
```

x86_64 macOS and Windows are not shipped as precompiled binaries. Other
backends (`vulkan`, `coreml`, `intel-sycl`, `openblas`, `openmp`) are not
in the precompiled matrix; build them from source.

### Build from source with a custom backend

```bash
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=cuda       mix compile  # NVIDIA CUDA
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=hipblas    mix compile  # AMD ROCm
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=vulkan     mix compile  # Cross-vendor Vulkan
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=metal      mix compile  # Apple Silicon GPU
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=coreml     mix compile  # Apple Neural Engine
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=intel-sycl mix compile  # Intel Arc / Xe
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=openblas   mix compile  # CPU + OpenBLAS
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=openmp     mix compile  # CPU + OpenMP
WHISPER_CPP_BUILD=1                                 mix compile  # Pure CPU
```

Pick **one** acceleration backend per build. The features are mutually
exclusive at runtime - the chosen backend is baked into the compiled
artefact. The `Taskfile.yml` ships `task build:cpu`, `task build:cuda`,
and `task build:hipblas` shortcuts.

### Runtime device selection

```elixir
WhisperCpp.available_devices()
#=> {:ok, %{backends: [:cpu, :cuda], gpu_supported: true, gpu_devices: 1}}

{:ok, model} =
  WhisperCpp.load_model("models/ggml-large-v3.bin",
    device: :auto     # :cpu | :cuda | :hipblas | :vulkan | :metal | :coreml | :intel_sycl | :auto
  )
```

`:auto` picks the GPU backend if the artefact has one compiled in;
otherwise CPU. Explicitly requesting a backend that was not compiled
returns `{:error, %WhisperCpp.Error{reason: :invalid_request}}`.

## Models

Download official whisper.cpp checkpoints from
<https://huggingface.co/ggerganov/whisper.cpp>:

```bash
mkdir -p models
curl -fLo models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

Both legacy `.bin` (GGML) and `.gguf` files are supported.

## Usage

```elixir
{:ok, model} = WhisperCpp.load_model("models/ggml-large-v3.bin")

{:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
  WhisperCpp.transcribe(model, "jfk.wav", language: "en")

IO.puts(text)
# => "And so, my fellow Americans, ask not what your country can do for you ..."

for s <- segs do
  IO.puts("[#{s.start}-#{s.end}] (no_speech=#{Float.round(s.no_speech_prob, 3)}) #{s.text}")
end
```

`%WhisperCpp.Segment{}` carries absolute `:start` / `:end` seconds,
`:no_speech_prob`, `:avg_logprob`, the underlying text token IDs, and
(when `:word_timestamps` is on) a list of `%WhisperCpp.Word{}` with
per-word timing.

### Audio contract

whisper.cpp expects **mono `f32` PCM samples at 16 kHz**, normalised to
the `-1.0..1.0` range. `transcribe/3` accepts:

- a `.wav` path (16 kHz mono or stereo, 16/32-bit PCM or 32-bit float),
  decoded by the built-in `WhisperCpp.Wav` module;
- `{:pcm_f32, binary}` containing little-endian f32 samples.

Non-`.wav` paths and bare binaries are rejected up front - a typo'd path
used to silently turn into garbage PCM; now it returns a clear
`:invalid_request` error.

### Diarization-driven workflows

Decode the master WAV once and transcribe each diarization turn without
re-decoding or shelling out to a subprocess per turn:

```elixir
{:ok, samples} = WhisperCpp.Wav.read_file("call.wav")

results =
  for {start_s, end_s, speaker} <- diar_turns do
    {:ok, t} =
      WhisperCpp.transcribe_slice(model, samples, {start_s, end_s},
        language: "en",
        word_timestamps: true,
        n_threads: 4
      )

    {speaker, t}
  end
```

`transcribe_slice/4` slices the PCM, runs whisper.cpp on the slice, and
rewrites the segment/word times back into the absolute timeline of the
original audio. Slices shorter than 0.3 s return an empty transcription
(whisper.cpp pads short inputs and hallucinates into the padding).

### Decoding biases

```elixir
WhisperCpp.transcribe(model, "talk.wav",
  language: "en",
  initial_prompt: "Discussion of whisper.cpp, BEAM, and Whisper internals."
)
```

## Options

`transcribe/3` and `transcribe_slice/4` accept any subset of:

| Option                          | Type              | Notes                                                  |
| ------------------------------- | ----------------- | ------------------------------------------------------ |
| `:language`                     | `String.t \| nil` | ISO code (`"en"`). `nil` auto-detects on multilingual. |
| `:translate`                    | `boolean`         | Translate to English instead of transcribing.          |
| `:initial_prompt`               | `String.t \| nil` | Free-text context for decoder biasing.                 |
| `:word_timestamps`              | `boolean`         | Attach per-word timing.                                |
| `:beam_size`                    | `pos_integer`     | Beam-search width. Default `5`.                        |
| `:best_of`                      | `pos_integer`     | Greedy candidates when `beam_size <= 1`.               |
| `:temperature`                  | `float`           | Sampling temperature (`0.0` = greedy/beam).            |
| `:n_threads`                    | `pos_integer`     | Intra-op threads. Default `4`.                         |
| `:n_max_text_ctx`               | `non_neg_integer` | Cap decoder context tokens.                            |
| `:offset_ms`, `:duration_ms`    | `non_neg_integer` | Clip the audio window.                                 |
| `:no_speech_thold`              | `float`           | Silence detection threshold.                           |
| `:logprob_thold`                | `float`           | Reject segments below this `avg_logprob`.              |
| `:suppress_blank`               | `boolean`         | Suppress the initial blank token.                      |
| `:suppress_non_speech_tokens`   | `boolean`         | Suppress music/noise tokens.                           |
| `:single_segment`               | `boolean`         | Force a single segment for the whole audio.            |
| `:print_progress`               | `boolean`         | whisper.cpp progress to stderr.                        |

Unknown option keys and out-of-range values return
`{:error, %WhisperCpp.Error{reason: :invalid_request}}` before reaching
the NIF.

## Errors

All failures return `{:error, %WhisperCpp.Error{}}`. `reason` is one of
`:invalid_request`, `:load_error`, `:inference_error`, `:runtime_error`,
`:nif_panic`, or `:native_error`. The struct also implements `Exception`,
so `raise/1` works.

## Testing

Unit tests run with no external dependencies:

```bash
mix test
```

The end-to-end transcription test downloads `ggml-tiny.en.bin` (~75 MB)
and the JFK sample on first run, caches them under `test/fixtures/`, and
runs a real whisper.cpp inference:

```bash
mix test --include integration
```

Set `WHISPER_CPP_REFRESH=1` to redownload.

## License

MIT. whisper.cpp itself is MIT-licensed. The bundled `whisper-rs` crate
vendors whisper.cpp under `sys/whisper.cpp/` and links it statically.
