# Changelog

All notable changes to `whisper_cpp` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-19

### Fixed
- `--hipblas` release artefact now actually carries the gfx1100-1201
  device kernels declared in v0.1.1. The release workflow had been
  reusing `actions/cache` across version bumps, restoring a stale
  `native/whisper_cpp_native/target/` tree whose cmake build state
  predated the `AMDGPU_TARGETS` env vars. The new HIP arch list was
  visible to cmake configure but the cached HIP object files never
  recompiled, so the published `.so` shipped with the old arch list
  baked in and gfx1200/gfx1201 GPUs still crashed at the first kernel
  launch. The cache now only covers the cargo registry; the build
  tree is rebuilt from scratch each release, and a post-build check
  fails the workflow if `.hip_fatbin` is suspiciously small.

## [0.1.1] - 2026-05-19

### Fixed
- Precompiled `--hipblas` NIF now ships device code for the full
  RDNA 3 / RDNA 4 consumer line (`gfx1100`, `gfx1101`, `gfx1102`,
  `gfx1103`, `gfx1200`, `gfx1201`). Previously the release build
  picked ggml's default arch list, which omitted gfx1200 / gfx1201,
  so RX 9000-series cards loaded the model and detected the GPU but
  crashed on the first kernel launch with a missing device kernel.

## [0.1.0] - 2026-05-18

### Added
- Initial release.
- `WhisperCpp.load_model/2`: GGML/GGUF model loading with
  `:cpu`, `:cuda`, `:hipblas`, `:vulkan`, `:metal`, `:coreml`,
  `:intel_sycl`, `:auto` device selection.
- `WhisperCpp.transcribe/3`: full whisper.cpp transcription on
  `{:pcm_f32, binary}` buffers with segment + token output.
- `WhisperCpp.transcribe_slice/4`: time-shifted per-slice transcription.
- `WhisperCpp.AbortHandle`: cooperative cancellation. Pass an
  `%AbortHandle{}` via `:abort_handle` and call `AbortHandle.abort/1`
  from another process to stop in-flight inference.
- `:progress_pid` transcribe option: receive `{:whisper_progress, pct}`
  messages as work advances; duplicate percentages are coalesced.
- `WhisperCpp.Pcm`: PCM slicing helpers. `transcribe/3` accepts only
  `{:pcm_f32, binary}` (little-endian f32 mono at 16 kHz); callers
  decode audio files upstream.
- `WhisperCpp.available_devices/0`: backend introspection for the
  loaded NIF artefact.
- `:word_timestamps` option for per-word timing.
- Rustler NIF using `whisper-rs` 0.16 with cargo features for `cuda`,
  `hipblas`, `vulkan`, `metal`, `coreml`, `intel-sycl`, `openblas`,
  `openmp`. Inference no longer serialises across processes sharing one
  loaded model.
- Precompiled NIF artefacts via `rustler_precompiled` for x86_64 / aarch64
  Linux (CPU, CUDA, hipBLAS variants) and aarch64 macOS (Metal).
- `Taskfile.yml`, strict Credo configuration, CI / integration / release
  GitHub workflows on Elixir 1.20-rc / OTP 29.
