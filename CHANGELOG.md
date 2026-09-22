# Changelog

All notable changes to `whisper_cpp` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-09-23

Several changes are caller-visible. Items that need an action from the
caller say so.

### Changed
- The precompiled NIFs have a fixed CPU baseline (#44). Before, each
  artefact was tuned for the CPU of the release runner. x86_64 Linux needs
  AVX2, FMA, F16C, and BMI2. aarch64 Linux needs ARMv8.2-A with dotprod and
  fp16. Apple Silicon needs an M1 or newer. On an older CPU, build from
  source with `WHISPER_CPP_BUILD=1`.
- The `cuda` variants carry kernels for the ggml release GPU list (`sm_50`
  to `sm_90`), not only `sm_52` PTX. Modern GPUs get faster kernels. The
  tarballs are larger.
- Vendored whisper.cpp 1.8.6 -> 1.9.4 (#47), through the same vendor branch
  of this repo. whisper-rs stays at the patched 0.16.0 (issue #26).
- `%Transcription{language: ...}` is always the ISO code (#52). A full name
  such as `"german"` now reports `"de"` on an empty result too (no speech,
  abort, no segment, or a `transcribe_slice/4` window under 0.3 s). Compare
  against the code, not the name. On a `transcribe_slice/4` window under
  0.3 s, `nil` and `"auto"` now report `"en"` on an English-only model and
  `""` on a multilingual model (was `""` and `"auto"`).
- `WHISPER_CPP_VARIANT` with a variant that the target does not publish
  fails the compile (#49), for example `hipblas` on aarch64 Linux or any
  variant on macOS. Before, it installed the CPU or Metal artefact without a
  message. Unset the variable or pick a published variant. A source build
  ignores the variant, as before.
- A change to `WHISPER_CPP_FEATURES`, `WHISPER_CPP_VARIANT`, or
  `WHISPER_CPP_BUILD` recompiles the NIF wrapper without `--force` (#46).
  To switch the backend of a Hex dependency, run
  `mix deps.compile whisper_cpp`.
- A `coreml` source build rejects `device: :cpu` and `use_gpu: false` with
  `:invalid_request`, and `available_devices/0` does not list `:cpu` there
  (#50). That build uses the Core ML encoder whenever the model's
  `-encoder.mlmodelc` loads, and a caller cannot turn it off per model.
- `:word_timestamps` returns one word per token group for Chinese, Japanese,
  Thai, Lao, Burmese, and Cantonese, not one word per segment (#51).
  Trailing punctuation joins the word before it.
- `WhisperCpp.Pcm.slice/4` rounds the end time, not the duration (#53). A
  window that ends at the buffer end no longer fails with "past the end".
  The slice can be one sample shorter than `round(duration_s * sample_rate)`.
- `transcribe_slice/4` reports every window past the buffer end as
  "requested window extends past the end of the buffer", with `start_s`,
  `end_s`, and `buffer_duration_s` in the details.
- `load_model/2` reports an empty or blank path as "path must be a
  non-empty UTF-8 string" (#54). Before, the message was "path must be a
  non-empty string".
- Source builds need Rust 1.98 or later (#60). The crate declares
  `rust-version = "1.98"`. Precompiled installs do not change.
- CI uses Elixir 1.20.4 (fixes CVE-2026-75758), OTP 29.1.1, and Rust 1.98.1.
  `mix.exs` keeps `elixir: "~> 1.19"` as the minimum version.
- Release CI builds a manual dispatch from the tag commit and never replaces
  a published tarball (#43). A push to `main` releases when the tag for the
  `mix.exs` version is missing, so a dropped run no longer loses a release.
  CI jobs have timeouts (#56).
- The NIF release profile uses 16 codegen units (was 1) and keeps thin LTO.
- Cargo refresh of transitive crates. `cfg-if` 1.0.5 and `smallvec` 1.16.1
  are in the NIF. The rest are build only. Dev only: ex_doc 0.40.3 -> 0.40.4,
  ex_slop 0.4.4 -> 0.4.5.

### Fixed
- The aarch64 Linux artefacts of 0.2.0 to 0.4.1 contain SVE and i8mm code
  and can die with SIGILL on Graviton2, Ampere Altra, Raspberry Pi 5, or
  Jetson Orin (#44). The 0.5.0 artefacts do not.
- whisper.cpp 1.9.4 fixes a heap out-of-bounds read when `transcribe/3` gets
  1 to 200 samples (ggml-org/whisper.cpp#3956), a stack buffer overflow when
  `load_model/2` reads a malformed model file (ggml-org/whisper.cpp#3957),
  and a C++ exception during a model load that could escape into the NIF
  (ggml-org/whisper.cpp#3831).
- `transcribe/3` on a multilingual model returns an empty transcription
  (`language: ""`) for 1 to 40 samples when it detects the language (#66).
  Before, it returned `:inference_error`.
- Native error messages no longer start with the internal `kind=<reason>: `
  tag (#48). The 0.4.0 fix for this did not work.
- These inputs return `:invalid_request` and no longer raise (#53, #54):
  - a time too large to convert to samples in `Pcm.slice/4` and
    `transcribe_slice/4` (was `ArithmeticError`);
  - an integer `:no_speech_thold` or `:logprob_thold` outside the i64 range
    (was `ErlangError`);
  - a `:progress_pid` on another node (was `ArgumentError`). The pid must be
    local;
  - a model path that is not valid UTF-8 in `load_model/2` (was
    `ArgumentError`).
- `transcribe_slice/4` windows under 0.3 s check the same things as longer
  windows (#55). A NaN or infinite sample in the window returns
  `:invalid_request` with the native details, and a window that rounds to
  zero samples returns `:invalid_request`. Before, both returned an empty
  success.
- `task build:*` sets the backend variables on its command line, so an
  exported `WHISPER_CPP_FEATURES` no longer changes the backend of the task.

## [0.4.1] - 2026-08-28

### Changed
- Dependency refresh. `libc` 0.2.188 -> 0.2.189 is the only runtime crate in
  the NIF that moved; the rest are build-only (`cc` 1.3.0 -> 1.4.4,
  `clang-sys` 1.8.1 -> 1.9.1, `aho-corasick`, `either`, `find-msvc-tools`,
  `log`, `regex-automata`). No API change. Every direct dependency, Hex and
  cargo, was already at its latest release; whisper-rs stays at the vendored
  0.16.0 patch (issue #26).
- Dev only: ex_slop 0.4.3 -> 0.4.4.
- `decode_pcm_f32` reads samples through `slice::as_chunks` instead of
  `chunks_exact`, which drops the manual byte indexing and clears the
  `clippy::chunks_exact_to_as_chunks` lint on newer toolchains. Same
  behaviour; the multiple-of-4 guard still rejects a misaligned buffer
  before any sample is read.

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
