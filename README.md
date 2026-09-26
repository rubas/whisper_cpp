# whisper_cpp

Elixir bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) speech-to-text. A Rustler NIF on the
[`whisper-rs`](https://codeberg.org/tazz4843/whisper-rs) crate runs whisper.cpp in the BEAM process. You load a model,
pass it 16 kHz mono f32 PCM, and get back structured segments. No subprocess, no Python, no temporary files.

## Installation

```elixir
def deps do
  [{:whisper_cpp, "~> 0.5.0"}]
end
```

Mix downloads a precompiled NIF for your target from the GitHub releases, so you need no Rust toolchain. The package
needs Elixir 1.19 or newer.

## Usage

```elixir
{:ok, model} = WhisperCpp.load_model("models/ggml-large-v3.bin")

# Decode the audio first (ffmpeg, bumblebee, ...) into 16 kHz mono f32 PCM:
#   ffmpeg -i jfk.wav -f f32le -ac 1 -ar 16000 jfk.pcm
pcm = File.read!("jfk.pcm")

{:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
  WhisperCpp.transcribe(model, {:pcm_f32, pcm}, language: "en")

IO.puts(text)
for s <- segs, do: IO.puts("[#{s.start}-#{s.end}] #{s.text}")
```

Audio is always `{:pcm_f32, binary}`: little-endian f32 samples, mono, 16 kHz, in the range `[-1.0, 1.0]`. The library
does not decode WAV, MP3, or other formats; decode them before the call. `transcribe_slice/4` transcribes a
`[start_s, end_s)` window of a larger PCM buffer and returns times on the timeline of the full buffer.

The built-in silero voice activity detection removes silence before the encoder. Pass `vad_model_path:` with
`ggml-silero-v5.1.2.bin` (about 0.85 MB, from [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad)).
The timestamps stay on the original timeline.

[The docs](https://hexdocs.pm/whisper_cpp) list all options (`:translate`, `:initial_prompt`, `:word_timestamps`,
`:beam_size`, `:n_threads`, VAD tuning, cancellation, progress messages, and more) and the errors.

## Backends

Each build has one accelerator. Every build also runs on the CPU, except `coreml`. The precompiled package has a CPU
build for each target, `cuda` and `hipblas` variants for Linux, and Metal on Apple Silicon. `WHISPER_CPP_VARIANT`
selects a variant:

```bash
WHISPER_CPP_VARIANT=cuda mix deps.compile whisper_cpp
```

| Variant   | Targets                                                  |
| --------- | -------------------------------------------------------- |
| `cuda`    | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` |
| `hipblas` | `x86_64-unknown-linux-gnu`                               |

The precompiled NIFs, the GPU variants included, need this CPU:

| Target                      | Minimum CPU                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `x86_64-unknown-linux-gnu`  | AVX2, FMA, F16C, BMI2 (Intel Haswell, AMD Zen, or newer)            |
| `aarch64-unknown-linux-gnu` | ARMv8.2-A with dotprod and fp16 (Neoverse N1, Cortex-A76, or newer) |
| `aarch64-apple-darwin`      | Apple M1 or newer                                                   |

On an older CPU, build from source with `WHISPER_CPP_BUILD=1`. A source build tunes ggml for the CPU it runs on. This
baseline applies from 0.5.0. Up to 0.4.1, the `aarch64-unknown-linux-gnu` NIFs need SVE and i8mm, and the 0.3.1
`x86_64-unknown-linux-gnu` NIF needs AVX-512.

A source build can use any `whisper-rs` backend: `cuda`, `hipblas`, `vulkan`, `metal`, `coreml`, `intel-sycl`,
`openblas`, or `openmp`.

```bash
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=cuda mix deps.compile whisper_cpp
```

A source build needs Rust 1.98 or newer, `cmake`, a C++17 compiler, and the SDK of the backend (CUDA toolkit, ROCm,
Vulkan SDK, ...).

A `coreml` build uses the Core ML encoder whenever the model's `-encoder.mlmodelc` exists, and you cannot turn it off
per model. It returns `:invalid_request` for `device: :cpu` and `use_gpu: false`. For CPU-only inference, build without
`coreml`.

## Development

`task check` runs the format check, compile, lint, the Elixir and Rust unit tests, and `zizmor` on the workflows.
`task test:integration` downloads `ggml-tiny.en` and `ggml-tiny` (about 75 MB each) and runs real inference.

## License

MIT. whisper.cpp is MIT. `whisper-rs` is public domain (Unlicense); it vendors whisper.cpp and links it statically.
