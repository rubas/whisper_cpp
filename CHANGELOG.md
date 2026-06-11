# Changelog

All notable changes to `whisper_cpp` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-11

### Added
- Built-in voice activity detection: pass `:vad_model_path` (a silero GGML
  model from `huggingface.co/ggml-org/whisper-vad`) to strip silence before
  the encoder, with `:vad_threshold`, `:vad_min_speech_ms`,
  `:vad_min_silence_ms`, and `:vad_speech_pad_ms` tuning options. Audio
  with no detected speech returns an empty transcription. The NIF runs the
  VAD itself and remaps all timestamps back to the original timeline -
  whisper.cpp's own VAD hook is dead code on the state-based API whisper-rs
  uses.

### Changed
- Native whisper.cpp/GGML logging is filtered to warnings and errors;
  the dozens of info lines per model load no longer reach stderr.
  `WHISPER_CPP_NATIVE_LOG` accepts `none`, `error`, `warn` (default),
  `info`, and `debug`. VAD contexts stay per-call: loading the silero
  model costs about a millisecond, and a shared context would serialise
  detection across concurrent transcribes.
- Integer options are bounded to `u32` and the VAD millisecond knobs to
  two minutes, returning `:invalid_request` instead of raising or
  overflowing inside the detector. Option validation now also runs
  before the sub-0.3 s `transcribe_slice` short-circuit, and an abort
  raised during the VAD pass is honoured before the encoder starts.
- `:duration_ms` must be at least 1. `0` previously meant "whole audio"
  without VAD but "empty window" with it; the ambiguity is rejected as
  `:invalid_request`.
- Passing `:vad_threshold`, `:vad_min_speech_ms`, `:vad_min_silence_ms`,
  or `:vad_speech_pad_ms` without `:vad_model_path` returns
  `:invalid_request` instead of being silently ignored.
- Buffers above `i32::MAX` samples (about 37 hours) are rejected instead
  of silently truncating at the FFI boundary.

### Fixed
- Multi-segment transcriptions no longer contain doubled spaces in
  `Transcription.text` (whisper segments carry their own leading space;
  the join added another). Space-free scripts no longer gain spurious
  spaces.
- `:temperature` is validated to `0.0..1.0` (above 1.0 whisper.cpp's
  retry ladder is empty and the decoder state undefined), `:n_threads`
  to GGML's 512-thread abort threshold, and `:beam_size`/`:best_of` to
  whisper.cpp's 8-decoder limit - all returning `:invalid_request`
  instead of native crashes or opaque inference errors.
- `:best_of` defaults to 5, matching whisper.cpp, and now also applies
  to temperature-fallback passes in beam-search mode.
- Sub-0.3 s `transcribe_slice` windows validate options, buffer bounds,
  and alignment before returning the documented empty transcription, and
  a window of exactly 0.3 s transcribes instead of being dropped by
  float subtraction error. The empty result keeps the pinned language.
- `translate: true` on English-only models returns `:invalid_request`
  instead of being silently ignored; `use_gpu: false` wins over a
  conflicting `:device`; invalid UTF-8 string options and non-keyword
  option lists return `:invalid_request` instead of raising.
- Native error messages no longer leak the internal "kind=..." routing
  tag; results with no decoded segments echo the requested language
  instead of fabricating "en"; progress percentages are clamped to the
  documented 0..100.
- `Pcm.slice/4` rounds sample positions instead of truncating, so
  millisecond-precise windows keep their last sample.
- Builds with two GPU features fail at compile time instead of silently
  picking one; unknown `WHISPER_CPP_VARIANT` values fail the build
  instead of falling back to the CPU artefact.
- `:abort_handle` and `:progress_pid` callbacks no longer leak memory per
  call: the vendored whisper-rs (branch `vendor/whisper-rs-0.16.0-patched`)
  fixes the abort-trampoline type confusion and the callback closure leak
  at the source (upstream issues 277/271, fix PR 278), replacing the
  downstream pre-boxing and sentinel workarounds. The progress sender
  thread now exits via natural channel close. The same vendor patch stops
  `set_language`, `set_initial_prompt`, and the VAD path from leaking one
  `CString` per call.

## [0.3.1] - 2026-06-11

### Changed
- Vendored whisper.cpp 1.8.3 -> 1.8.6. whisper-rs has no release vendoring
  anything newer, so `whisper-rs-sys` is patched via `[patch.crates-io]` to
  this repo's `vendor/whisper-rs-sys-1.8.6` branch - the published
  whisper-rs-sys 0.15.0 with only its whisper.cpp submodule bumped. The
  patch applies to source builds and the precompiled NIF artefacts alike,
  and is dropped as soon as upstream re-vendors (see issue #18).
- CI: `sccache-action` v0.0.9 -> v0.0.10 (Node 24; GitHub retires the
  Node 20 runtime on 2026-06-16).

## [0.3.0] - 2026-06-11

### Changed
- rustler 0.37 → 0.38 (Rust crate and optional Hex package). Additive
  upstream release; no NIF API changes needed. The vendored whisper.cpp
  stays at 1.8.3 until whisper-rs ships a release vendoring something
  newer - upstream's latest (0.16.0, 2026-03-12) predates whisper.cpp
  1.8.4.
- `language: nil` now actually auto-detects on multilingual models, as the
  docs always claimed. Previously `nil` silently fell through to
  whisper.cpp's forced-`"en"` default, decoding non-English audio as
  English. English-only models resolve `nil`/`"auto"` to `"en"`.
- `:language` is validated against whisper.cpp's language table. Unknown
  codes - including BCP 47 tags such as `"de-CH"` - return
  `:invalid_request` instead of silently corrupting the decoder prompt
  with an invalid language token. Passing a non-English language to an
  English-only model is rejected the same way instead of being silently
  ignored.
- The `:beam_size` and `:best_of` docs state the real defaults: greedy
  decoding with `best_of: 1`. The docs previously claimed a beam-search
  default of 5 that no code path produced.

### Fixed
- `:abort_handle` cancellation works now. The abort callback is passed to
  whisper-rs as a boxed trait object so the trampoline polls the real flag;
  the bare closure was reinterpreted memory (out-of-bounds reads) and the
  flag was never consulted, so cancellation silently did nothing.
- `:progress_pid` no longer leaks one OS thread per call. The progress
  sender thread is shut down explicitly after inference; the previous
  design waited for a channel close that whisper-rs's leaked callback
  closure could never trigger.
- `:word_timestamps` no longer corrupts multibyte UTF-8. Token bytes are
  accumulated per word and converted once, so characters split across BPE
  tokens (umlauts and most non-Latin scripts) survive instead of turning
  into replacement characters.
- Dropping the last reference to a loaded model frees the whisper context
  on a detached thread instead of the garbage-collecting BEAM scheduler,
  which a multi-gigabyte free would stall.
- `{:pcm_f32, _}` buffers containing NaN or infinity samples are rejected
  with `:invalid_request` instead of being fed to inference.

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
