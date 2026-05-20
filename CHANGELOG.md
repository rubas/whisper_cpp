# Changelog

All notable changes to `whisper_cpp` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-20

### Added
- `WhisperCpp.load_model/2`: GGML/GGUF model loading with `:cpu`, `:cuda`,
  `:hipblas`, `:vulkan`, `:metal`, `:coreml`, `:intel_sycl`, and `:auto`
  device selection.
- `WhisperCpp.transcribe/3`: full whisper.cpp transcription on
  `{:pcm_f32, binary}` buffers (little-endian f32 mono at 16 kHz) with
  segment, token, and optional per-word output.
- `WhisperCpp.transcribe_slice/4`: time-shifted per-slice transcription that
  reuses one decoded PCM buffer.
- `WhisperCpp.AbortHandle`: cooperative cancellation. Pass an `%AbortHandle{}`
  via `:abort_handle` and call `AbortHandle.abort/1` from another process to
  stop in-flight inference; the partial transcription produced before the
  abort is returned.
- `:progress_pid` transcribe option: receive `{:whisper_progress, pct}`
  messages as work advances; duplicate percentages are coalesced.
- `:word_timestamps` option for per-word timing.
- `WhisperCpp.available_devices/0`: backend introspection for the loaded NIF
  artefact.
- `WhisperCpp.Pcm`: PCM slicing helpers. Audio file decoding is intentionally
  out of scope; callers decode upstream (ffmpeg, Bumblebee, ...) and share one
  decoded PCM buffer across stages.
- Rustler NIF built on `whisper-rs`, with cargo features for `cuda`,
  `hipblas`, `vulkan`, `metal`, `coreml`, `intel-sycl`, `openblas`, and
  `openmp`. Inference does not serialise across processes sharing one loaded
  model.
- Precompiled NIF artefacts via `rustler_precompiled` for x86_64 / aarch64
  Linux (CPU, CUDA, hipBLAS variants) and aarch64 macOS (Metal).
