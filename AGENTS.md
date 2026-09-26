# whisper_cpp

The README describes the library, its backends, and the CPU baseline. The Hex package ships precompiled NIFs, so most
users never build Rust.

## Checks

- `ci.yml` runs `task check` on pushes to `main` and on pull requests. A push to a branch without a pull request runs
  nothing.
- `task test:integration` runs real inference. `integration.yml` runs it weekly and on manual dispatch, never on a pull
  request. Run it locally when you change the NIF boundary.
- `release.yml` builds six NIFs (target and variant). One backend takes minutes to build from source, so leave the
  matrix to CI.

## Rules

- Audio decoding stays out of the library. `transcribe/3` takes only `{:pcm_f32, binary}`; a file path or a bare binary
  returns `:invalid_request`. Callers decode first and share one decoded buffer across a pipeline.
- No silent fallback. An unknown option, a device the NIF does not have, and an audio shape the library rejects all
  return an error, never a default.
- Errors cross the NIF boundary as `{:error, %{type, message, details}}`. `errors.rs` sets the type, and
  `WhisperCpp.Error` maps it to a reason atom. An unknown type becomes `:native_error`, so a new type needs both sides.
- Credo enables `Readability.Specs`, so a public function without a `@spec` fails. `Readability.ModuleDoc` is off, but
  every public module still gets a `@moduledoc`; no check catches a missing one.
- `whisper-rs` and `whisper-rs-sys` come from a `[patch.crates-io]` pin to the `vendor/whisper-rs-0.16.0-patched`
  branch of this repo. It adds the callback and CString-leak fixes and moves whisper.cpp to v1.9.4. On a `whisper-rs`
  version bump, check the patch again (issue #26).
- `WHISPER_CPP_FEATURES` picks the cargo feature, and `WHISPER_CPP_BUILD=1` forces a source build. A change to either,
  or to `WHISPER_CPP_VARIANT`, recompiles the NIF wrapper. The `build:*` tasks set them only for their own run, so a
  later `task test` uses the precompiled NIF again. Test a backend with the same variables, for example
  `WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=cuda task test:integration`.

## Pitfalls

- `release.yml` builds with `GGML_NATIVE=OFF` and the `GGML_CPU_ARM_ARCH` of the matrix. Without them ggml builds for
  the runner CPU, and the NIF can die with SIGILL on an older CPU. The release job fails when `CMakeCache.txt` does not
  have `GGML_NATIVE:BOOL=OFF`. Local source builds keep the native tuning.
- `release.yml` reads `nif_versions:` from `lib/whisper_cpp/native.ex` with `sed`. If you reformat that line, the
  release job fails.
- A new precompiled variant needs an entry in the `@variants` map of `native.ex` and in the `release.yml` build
  matrix. Without the `native.ex` entry, the compile fails because the variant is not published for the target.
  Without the matrix entry, the install fails on the download because no tarball exists.
- The ROCm build needs gfx1200 and gfx1201 in the `GPU_TARGETS` list in `release.yml`. Without them the NIF loads and
  reports the GPU, then dies on the first kernel launch. The release job checks the size of the `.hip_fatbin` section
  to catch this before it publishes.

## Release

1. Bump `@version` in `mix.exs`, add the `CHANGELOG.md` entry, and push to `main`.
2. On each push to `main`, `release.yml` releases when no tag exists for the `mix.exs` version. It builds a tarball per
   target and variant, creates the tag, and uploads the tarballs and `SHA256SUMS`. If a run stops before it creates the
   tag, the next push retries the release.
3. After the tag exists, a manual dispatch builds the tag's commit. It fails when the tag is missing or its `mix.exs`
   version differs, and it uploads only the assets the release does not have yet. It never replaces a published
   tarball, because a new tarball breaks the checksum file in the Hex package.
4. Regenerate `checksum-Elixir.WhisperCpp.Native.exs` from the published assets, then commit and push it. This keeps
   the checksum of each tag reproducible from the repo:

   ```bash
   mix rustler_precompiled.download WhisperCpp.Native --all --no-config --ignore-unavailable --print
   ```

5. Run `mix hex.publish` from a clean tree. `mix.exs` puts `checksum-*.exs` in the Hex tarball.
