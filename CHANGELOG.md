# Changelog

All notable changes to `whisper_cpp` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-18

### Added
- Initial release.
- `WhisperCpp.load_model/2`: GGML/GGUF model loading with
  `:cpu`, `:cuda`, `:hipblas`, `:vulkan`, `:metal`, `:coreml`,
  `:intel_sycl`, `:auto` device selection.
- `WhisperCpp.transcribe/3`: full whisper.cpp transcription on `.wav`
  paths and `{:pcm_f32, binary}` buffers with segment + token output.
- `WhisperCpp.transcribe_slice/4`: time-shifted per-slice transcription
  for diarization-driven workflows.
- `WhisperCpp.Wav` / `WhisperCpp.Pcm`: WAV decoder and PCM slicing
  helpers.
- `WhisperCpp.available_devices/0`: backend introspection for the
  loaded NIF artefact.
- `:word_timestamps` option for per-word timing.
- Rustler NIF using `whisper-rs` 0.16 with cargo features for `cuda`,
  `hipblas`, `vulkan`, `metal`, `coreml`, `intel-sycl`, `openblas`,
  `openmp`.
- Precompiled NIF artefacts via `rustler_precompiled` for x86_64 / aarch64
  Linux (CPU, CUDA, hipBLAS variants) and aarch64 macOS (Metal).
- `Taskfile.yml`, strict Credo configuration, CI / integration / release
  GitHub workflows.
