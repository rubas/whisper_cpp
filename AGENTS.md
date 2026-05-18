# AGENTS.md

Guidance for AI agents working on `whisper_cpp`.

## What this library is

Elixir bindings for whisper.cpp over a Rustler NIF. The Rust side links
whisper.cpp through the `whisper-rs` crate; the Elixir side wraps it
with a typed, validated API plus a WAV decoder and PCM slicing helpers.
There is no `whisper-cli` subprocess, no Python interop, no temporary
WAV files.

## Source-of-truth files

- `lib/whisper_cpp.ex` - public API (`load_model`, `transcribe`,
  `transcribe_slice`, `available_devices`).
- `lib/whisper_cpp/native.ex` - Rustler NIF stubs and
  `rustler_precompiled` targets / variants.
- `native/whisper_cpp_native/Cargo.toml` - cargo feature matrix
  (`cuda`, `hipblas`, `vulkan`, `metal`, `coreml`, `openblas`).
- `native/whisper_cpp_native/src/lib.rs` - NIF entry points; every
  function returns `{:ok, _}` or `{:error, %{type, message, details}}`.
- `native/whisper_cpp_native/src/transcribe.rs` - per-call decoding
  glue around `whisper-rs`.

## Public API shape

```
WhisperCpp.load_model(path, opts) -> {:ok, Model.t()} | {:error, Error.t()}
WhisperCpp.transcribe(Model.t(), audio, opts) -> {:ok, Transcription.t()} | {:error, _}
WhisperCpp.transcribe_slice(Model.t(), pcm_binary, {start_s, end_s}, opts) -> {:ok, Transcription.t()} | {:error, _}
WhisperCpp.available_devices() -> {:ok, %{backends, gpu_supported, gpu_devices}} | {:error, _}
```

Audio is either a `.wav` path (decoded by `WhisperCpp.Wav`) or
`{:pcm_f32, binary}` (little-endian f32 mono at 16 kHz). Bare binaries
are rejected.

## Backends

Pick **one** accelerator per build via `WHISPER_CPP_FEATURES`. CPU is
always available. The `rustler_precompiled` variants map to GPU
backends so users select them at install time via `WHISPER_CPP_VARIANT`.

| Feature      | Selected by                       | SDK requirement at build time   |
| ------------ | --------------------------------- | ------------------------------- |
| `cuda`       | `WHISPER_CPP_VARIANT=cuda`        | CUDA toolkit 12+                |
| `hipblas`    | `WHISPER_CPP_VARIANT=hipblas`     | ROCm 6.x, `hipblas-dev`         |
| `vulkan`     | source build                      | Vulkan loader / headers         |
| `metal`      | default on `aarch64-apple-darwin` | Xcode CLT                       |
| `coreml`     | source build                      | Xcode + Core ML tools           |
| `intel-sycl` | source build                      | Intel oneAPI Base Toolkit       |
| `openblas`   | source build                      | `libopenblas-dev`               |
| `openmp`     | source build                      | `libgomp` / `libomp`            |

## Conventions

- Every public function gets a `@spec`. Public modules get a
  `@moduledoc`. Tests file under `test/whisper_cpp/<module>_test.exs`.
- Integration tests are tagged `:integration` and excluded by default;
  they download fixtures on first run.
- Cross-language errors are categorised on the Rust side via
  `errors::{invalid_request, load_error, inference_error, runtime_error}`
  and decoded back to atoms on the Elixir side in `WhisperCpp.Error`.
- No silent fallbacks: a typo'd audio path returns an error, not garbage
  PCM. An unknown option returns an error, not a default.
- Stick to whisper.cpp's published 16 kHz contract; the WAV decoder
  rejects other sample rates and tells the caller to resample upstream.

## Local commands

```bash
task setup           # install elixir deps
task compile         # compile (builds NIF; first source build is slow)
task test            # fast unit tests
task test:integration  # downloads model + audio
task fmt:check       # mix format + cargo fmt verify
task lint            # credo --strict + clippy -D warnings
task check           # full local gate
```

## Hex publish flow

1. Bump `@version` in `mix.exs` and add a `CHANGELOG.md` entry.
2. Push to `main`. The `release.yml` workflow detects the version bump,
   builds NIF tarballs for every target/variant, creates the tag and
   GitHub release, and commits an updated
   `checksum-Elixir.WhisperCpp.Native.exs`.
3. Pull `main` locally and run `mix hex.publish` from a clean tree.
