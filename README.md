# whisper_cpp

A thin Elixir wrapper around [`whisper-rs`](https://codeberg.org/tazz4843/whisper-rs),
the Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp).
It exposes whisper.cpp speech-to-text to the BEAM through a Rustler NIF: load a
model, hand it 16 kHz mono f32 PCM, get structured segments back. No subprocess,
no Python, no temporary files.

## Installation

```elixir
def deps do
  [{:whisper_cpp, "~> 0.1.0"}]
end
```

Installation downloads a precompiled NIF for your target from the project's
GitHub releases - no Rust toolchain needed. Requires Elixir 1.19+.

## Usage

```elixir
{:ok, model} = WhisperCpp.load_model("models/ggml-large-v3.bin")

# Decode upstream (ffmpeg, bumblebee, ...) into 16 kHz mono f32 PCM:
#   ffmpeg -i jfk.wav -f f32le -ac 1 -ar 16000 jfk.pcm
pcm = File.read!("jfk.pcm")

{:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
  WhisperCpp.transcribe(model, {:pcm_f32, pcm}, language: "en")

IO.puts(text)
for s <- segs, do: IO.puts("[#{s.start}-#{s.end}] #{s.text}")
```

Audio is always `{:pcm_f32, binary}` - little-endian f32 samples, mono, 16 kHz,
normalised to `[-1.0, 1.0]`. The library does **not** decode WAV/MP3/etc;
decode upstream. `transcribe_slice/4` runs a `[start_s, end_s)` window of a
master PCM buffer and shifts the returned times back into the source timeline.

See [the docs](https://hexdocs.pm/whisper_cpp) for the full option list
(`:translate`, `:initial_prompt`, `:word_timestamps`, `:beam_size`,
`:n_threads`, cancellation, progress messages, ...) and error handling.

## Backends

CPU is always available. Pick one accelerator per build; the precompiled Hex
package ships CPU plus `cuda` / `hipblas` variants for Linux and Metal on Apple
Silicon, selected via `WHISPER_CPP_VARIANT`:

```bash
WHISPER_CPP_VARIANT=cuda mix deps.compile whisper_cpp
```

To build from source with any whisper-rs backend (`cuda`, `hipblas`, `vulkan`,
`metal`, `coreml`, `intel-sycl`, `openblas`, `openmp`):

```bash
WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=cuda mix compile
```

Source builds need a Rust toolchain, `cmake`, a C++17 compiler, and the
backend's own SDK (CUDA toolkit, ROCm, Vulkan SDK, ...).

## Testing

```bash
mix test                  # unit tests, no downloads
mix test --include integration  # downloads ggml-tiny.en + JFK sample, real inference
```

## License

MIT. whisper.cpp is MIT-licensed; `whisper-rs` is public domain (Unlicense)
and vendors whisper.cpp, linking it statically.
