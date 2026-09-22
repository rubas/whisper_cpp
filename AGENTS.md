# AGENTS.md

## Goal

Elixir bindings for whisper.cpp. A Rustler NIF links whisper.cpp through the
`whisper-rs` crate; the Elixir side adds typed options, validation, and PCM
slicing. The Hex package ships precompiled NIF artefacts, so most users never
build Rust.

Audio decoding stays out of the library. `transcribe/3` takes
`{:pcm_f32, binary}` only (little-endian f32, mono, 16 kHz); a file path or a
bare binary returns `:invalid_request`. Callers decode upstream (ffmpeg,
bumblebee) and share one decoded buffer across a pipeline.

## Gates

`task check` is the gate: format check, compile, credo, Elixir tests, Rust
tests, zizmor. `ci.yml` runs the same target on pushes to `main` and on pull
requests, so a push to a branch with no open pull request runs nothing.
`task --list` shows the rest.

- `task test:integration` downloads `ggml-tiny.en` (~75 MB) and runs real
  inference. `integration.yml` runs it weekly and on manual dispatch, never per
  pull request. Run it locally when you touch the NIF boundary.
- `security.yml` audits Hex and cargo dependencies nightly and files an issue on
  a finding.
- `release.yml` builds the six-entry NIF matrix. One backend already takes
  minutes to build from source, so leave the matrix to CI.

## Layout

- `lib/whisper_cpp/native.ex` holds the NIF stubs plus the
  `rustler_precompiled` targets, variants, and `nif_versions`.
- `native/whisper_cpp_native/src/lib.rs` holds the NIF entry points and
  resources; `transcribe.rs`, `vad.rs`, `errors.rs`, and `native_log.rs` hold
  the rest.
- `checksum-Elixir.WhisperCpp.Native.exs` is tracked on purpose, see Release.

## House decisions

- `whisper-rs` and `whisper-rs-sys` resolve through a `[patch.crates-io]` pin to
  a vendor branch of this repo, not from crates.io. That branch adds the
  callback and CString-leak fixes and moves the whisper.cpp submodule to v1.9.4.
  A `whisper-rs` version bump means re-checking the patch, see issue #26.
- One accelerator per build. `WHISPER_CPP_FEATURES` picks the cargo feature and
  `WHISPER_CPP_BUILD=1` forces a source build. Precompiled variants exist only
  for `cuda` (x86_64 and aarch64 Linux) and `hipblas` (x86_64 Linux); users
  select one with `WHISPER_CPP_VARIANT`. The darwin artefact is built with
  `metal`. The other features in `Cargo.toml` (`vulkan`, `coreml`,
  `intel-sycl`, `openblas`, `openmp`) are source build only.
- Errors cross the boundary as `{:error, %{type, message, details}}`.
  `errors.rs` sets the type and `WhisperCpp.Error` maps it to a reason atom; an
  unrecognised type becomes `:native_error`. A new type needs both sides.
- No silent fallback. An unknown option, a device the artefact does not carry,
  and an audio shape the library rejects all return an error, never a default.
- Credo runs `--strict` with the ExSlop and ExDNA checks. `Readability.Specs` is
  on, so a public function without a `@spec` fails the gate.
  `Readability.ModuleDoc` is off, so a `@moduledoc` on every public module is a
  house rule that no check catches. Write it anyway.

## Pitfalls

- `release.yml` builds with `GGML_NATIVE=OFF` and `GGML_CPU_ARM_ARCH` from the
  matrix. Drop them and ggml builds for the runner CPU, so the artefact can die
  with SIGILL on an older CPU. The release job fails when the ggml CPU flags
  contain `native`. Local source builds keep the native tuning.
- `release.yml` parses `nif_versions:` out of `lib/whisper_cpp/native.ex` with
  `sed`. Reformat that line and the release job fails.
- A new precompiled variant needs an entry in both the `variants` map of
  `native.ex` and the `release.yml` build matrix, and each miss breaks
  differently. Without the `native.ex` entry the artefact name keeps no variant
  suffix, so the install picks the CPU artefact without a warning. Without the
  matrix entry the suffix is there but no such tarball was published, so the
  install fails on the download.
- The ROCm build needs the `GPU_TARGETS` arch list in `release.yml`. Without
  gfx1200 and gfx1201 the artefact loads and reports the GPU, then dies on the
  first kernel launch. The release job checks the `.hip_fatbin` size to catch
  this before publishing.

## Release

1. Bump `@version` in `mix.exs`, add the `CHANGELOG.md` entry, push to `main`.
2. On every push to `main`, `release.yml` releases when no tag exists for the
   `mix.exs` version. It builds a tarball per target and variant, creates the
   tag, and uploads the tarballs plus `SHA256SUMS`. A run that is dropped
   before it creates the tag does not lose the release, because the next push
   retries it. Once the tag exists, only a manual dispatch rebuilds it. The
   dispatch builds the tag's commit and fails when the tag is missing or its
   `mix.exs` version differs.
3. Regenerate the checksum file from the published assets, then commit and push
   it. The checksum for each tag stays reproducible from the repo:

   ```bash
   mix rustler_precompiled.download WhisperCpp.Native --all --no-config --ignore-unavailable --print
   ```

4. Run `mix hex.publish` from a clean tree. `mix.exs` ships `checksum-*.exs`
   inside the Hex tarball.
