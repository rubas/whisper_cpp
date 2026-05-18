# whisper_cpp usage rules

For agents and humans writing code against `whisper_cpp`. These rules are
shipped with the Hex package so downstream consumers can opt in to a
consistent set of conventions.

## Loading models

- Pass a path to a `.bin` or `.gguf` whisper.cpp checkpoint to
  `WhisperCpp.load_model/2`. Download checkpoints from
  <https://huggingface.co/ggerganov/whisper.cpp>.
- Cache the `%WhisperCpp.Model{}` for the process lifetime; loading is
  expensive and the underlying NIF resource is thread-safe.
- Prefer `device: :auto` (the default). Explicit device selection that
  does not match the installed NIF artefact returns `:invalid_request`.

## Audio input

- whisper.cpp requires **16 kHz mono f32 PCM**. The built-in
  `WhisperCpp.Wav` decoder handles 16 kHz WAV files; other sample
  rates are rejected.
- For non-WAV sources (mp3, ogg, etc.), decode upstream (e.g. via
  ffmpeg) and pass `{:pcm_f32, binary}`.
- Bare binaries are rejected on purpose. A typo'd path used to turn
  into garbage PCM; we surface the bug instead.

## Per-turn / diarization workflows

- Decode the master WAV once via `WhisperCpp.Wav.read_file/1`.
- Use `WhisperCpp.transcribe_slice/4` for per-turn transcription -
  it handles the byte math, runs whisper.cpp on the slice, and
  shifts segment/word times back into the absolute timeline.
- Slices shorter than 0.3 s return an empty transcription. whisper.cpp
  pads short inputs and hallucinates into the padding; do not pass
  unfiltered VAD output.

## Options and errors

- Pass options as keyword lists. Unknown keys and out-of-range values
  fail with `{:error, %WhisperCpp.Error{reason: :invalid_request}}`
  before reaching the NIF - rely on this for input validation.
- Match `%WhisperCpp.Error{}` (or its `:reason` field) rather than
  inspecting message strings.

## Performance

- `:n_threads` defaults to 4. On dedicated nodes, set it to the number
  of physical cores.
- Word timestamps add one DTW pass; enable `:word_timestamps` only when
  you need them.
- For latency-sensitive workloads, prefer `:single_segment` on short
  clips to skip the segment-split pass.
- Beam search (`:beam_size > 1`) is roughly 2-3x slower than greedy and
  worth it for the lowest WER on long-form audio; for per-turn
  diarization slices, greedy is usually fine.
