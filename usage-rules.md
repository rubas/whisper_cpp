# whisper_cpp usage rules

For agents and humans writing code against `whisper_cpp`. These rules are
shipped with the Hex package so downstream consumers can opt in to a
consistent set of conventions.

## Loading models

- Pass a path to a `.bin` or `.gguf` whisper.cpp checkpoint to
  `WhisperCpp.load_model/2`. Download checkpoints from
  <https://huggingface.co/ggerganov/whisper.cpp>.
- Cache the `%WhisperCpp.Model{}` for the process lifetime; loading is
  expensive and the underlying NIF resource is safe to share across
  BEAM processes - concurrent `transcribe/3` calls do not serialise.
- Prefer `device: :auto` (the default). Explicit device selection that
  does not match the installed NIF artefact returns `:invalid_request`.

## Audio input

- `transcribe/3` accepts exactly one shape: `{:pcm_f32, binary()}`,
  where the binary is little-endian IEEE-754 `f32` samples, mono,
  16 kHz, normalised to `[-1.0, 1.0]`.
- This library does **not** decode audio file formats. Decode WAV,
  MP3, FLAC, M4A, Opus, etc. upstream and hand the PCM in. Standard
  recipe with ffmpeg:

  ```bash
  ffmpeg -i input.mp3 -f f32le -ac 1 -ar 16000 input.pcm
  ```

  In Elixir: `pcm = File.read!("input.pcm")`, then
  `WhisperCpp.transcribe(model, {:pcm_f32, pcm}, ...)`.

- Bare binaries (without the `{:pcm_f32, _}` wrapper) and file paths
  are rejected with `:invalid_request`. A typo'd path used to turn
  into garbage PCM; the wrapper surfaces the bug instead.

## Per-turn / diarization workflows

- Decode the source file once upstream into a master PCM buffer.
- Use `WhisperCpp.transcribe_slice/4` for per-turn transcription -
  it handles the byte math, runs whisper.cpp on the slice, and
  shifts segment/word times back into the absolute timeline.
- Slices shorter than 0.3 s return an empty transcription. whisper.cpp
  pads short inputs and hallucinates into the padding; do not pass
  unfiltered VAD output.

## Cancellation and progress

- For cancellable transcribes, mint a `%WhisperCpp.AbortHandle{}` via
  `WhisperCpp.AbortHandle.new/0` and pass it via `:abort_handle`.
  Signal cancellation from another process with
  `WhisperCpp.AbortHandle.abort/1`. The call returns
  `{:ok, partial_transcription}` with whatever segments completed
  before whisper.cpp's next abort poll.
- For progress, pass `:progress_pid` (commonly `self()` inside a
  `Task`). The pid receives `{:whisper_progress, percent}` messages
  (0..100) as work advances; duplicate percentages are coalesced.
- Both hooks are zero-cost when omitted.

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
- A single loaded model handle is safe to share: parallel transcribe
  calls do not serialise on the context lock, so saturating a GPU or
  multi-core CPU from many BEAM processes is the expected pattern.
