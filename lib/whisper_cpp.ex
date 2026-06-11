defmodule WhisperCpp do
  @moduledoc """
  Native Elixir bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

  A thin wrapper around the
  [`whisper-rs`](https://codeberg.org/tazz4843/whisper-rs) crate, calling
  whisper.cpp's C API through a Rustler NIF. No `whisper-cli` subprocess,
  no Python, no temporary files. Structured per-segment results,
  `:initial_prompt` biasing, word-level timestamps, built-in silero
  voice activity detection, and CUDA / ROCm (hipBLAS) / Metal / CPU
  backends.

  ## Quickstart

      {:ok, model} = WhisperCpp.load_model("models/ggml-large-v3.bin")

      {:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
        WhisperCpp.transcribe(model, {:pcm_f32, samples}, language: "en")

      IO.puts(text)
      for s <- segs, do: IO.puts("[\#{s.start}-\#{s.end}] \#{s.text}")

  ## Audio contract

  `transcribe/3` accepts exactly one input shape:

      {:pcm_f32, binary()}

  where `binary` is little-endian IEEE-754 `f32` samples, mono, 16 kHz,
  normalised to `[-1.0, 1.0]`. Decode audio file formats (WAV, MP3,
  FLAC, M4A, Opus, ...) upstream with ffmpeg or similar:

      ffmpeg -i input.mp3 -f f32le -ac 1 -ar 16000 - | …

  Use `transcribe_slice/4` to transcribe a `[start_s, end_s)` window of an
  already-decoded master PCM buffer; the returned segment / word times
  are shifted back into the original audio timeline.
  """

  alias WhisperCpp.AbortHandle
  alias WhisperCpp.Error
  alias WhisperCpp.Model
  alias WhisperCpp.Native
  alias WhisperCpp.Pcm
  alias WhisperCpp.Segment
  alias WhisperCpp.Transcription
  alias WhisperCpp.Word

  @typedoc "Audio input accepted by `transcribe/3`."
  @type audio :: {:pcm_f32, binary()}

  @target_sample_rate 16_000

  @typedoc "Options accepted by `transcribe/3` / `transcribe_slice/4`."
  @type transcribe_opt ::
          {:language, String.t() | nil}
          | {:translate, boolean()}
          | {:initial_prompt, String.t() | nil}
          | {:word_timestamps, boolean()}
          | {:beam_size, pos_integer()}
          | {:best_of, pos_integer()}
          | {:temperature, float()}
          | {:n_threads, pos_integer()}
          | {:n_max_text_ctx, non_neg_integer()}
          | {:offset_ms, non_neg_integer()}
          | {:duration_ms, pos_integer()}
          | {:no_speech_thold, float()}
          | {:logprob_thold, float()}
          | {:suppress_blank, boolean()}
          | {:suppress_non_speech_tokens, boolean()}
          | {:single_segment, boolean()}
          | {:print_progress, boolean()}
          | {:vad_model_path, String.t() | nil}
          | {:vad_threshold, float()}
          | {:vad_min_speech_ms, pos_integer()}
          | {:vad_min_silence_ms, pos_integer()}
          | {:vad_speech_pad_ms, non_neg_integer()}
          | {:abort_handle, AbortHandle.t() | nil}
          | {:progress_pid, pid() | nil}

  @typedoc "Options accepted by `load_model/2`."
  @type load_opt :: {:device, Model.device() | :auto} | {:use_gpu, boolean()}

  @devices [:cpu, :cuda, :hipblas, :vulkan, :metal, :coreml, :intel_sycl, :auto]

  @doc """
  Reports the runtime backends compiled into this NIF artefact.

  Returns `{:ok, %{backends: [...], gpu_supported: bool}}`. The
  `backends` list reflects compile-time cargo features (e.g.
  `[:cpu, :cuda]` on a `WHISPER_CPP_VARIANT=cuda` build).

  Build a source artefact with GPU support via:

      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=cuda       mix compile  # NVIDIA
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=hipblas    mix compile  # AMD ROCm
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=vulkan     mix compile  # cross-vendor
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=metal      mix compile  # Apple Silicon
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=coreml     mix compile  # Apple ANE
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=intel-sycl mix compile  # Intel Arc/Xe
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=openblas   mix compile  # CPU + OpenBLAS
      WHISPER_CPP_BUILD=1 WHISPER_CPP_FEATURES=openmp     mix compile  # CPU + OpenMP

  Pick one accelerator per build; the backend is baked into the artefact.
  """
  @spec available_devices() ::
          {:ok, %{backends: [atom()], gpu_supported: boolean()}}
          | {:error, Error.t()}
  def available_devices do
    case Native.available_devices() do
      {:ok, info} ->
        {:ok,
         %{
           backends: Enum.map(info.backends, &String.to_existing_atom/1),
           gpu_supported: info.gpu_supported
         }}

      {:error, payload} ->
        {:error, Error.from_native(payload)}
    end
  end

  @doc """
  Loads a GGUF or GGML whisper.cpp model file.

  Pass a path to a `.bin` (legacy GGML) or `.gguf` file. Download official
  weights from <https://huggingface.co/ggerganov/whisper.cpp>.

  ## Options

  - `:device` - one of `:cpu`, `:cuda`, `:hipblas`, `:vulkan`, `:metal`,
    `:coreml`, `:intel_sycl`, or `:auto` (default). `:auto` picks the GPU
    backend when the artefact was built with one; otherwise CPU.
    Requesting a backend that was not compiled in returns
    `{:error, %WhisperCpp.Error{reason: :invalid_request}}`.
  - `:use_gpu` - shortcut: `false` forces `device: :cpu`. Default `true`.
  """
  @spec load_model(Path.t(), [load_opt()]) :: {:ok, Model.t()} | {:error, Error.t()}
  def load_model(path, opts \\ [])

  def load_model(path, opts) when is_binary(path) and is_list(opts) do
    with :ok <- validate_non_empty_string(path, :path),
         :ok <- validate_options(opts, load_validators()) do
      do_load_model(path, opts)
    end
  end

  def load_model(_path, _opts) do
    {:error, Error.new(:invalid_request, "path must be a string and opts a keyword list")}
  end

  defp do_load_model(path, opts) do
    with {:ok, ref} <- native_call(Native.load_model(path, build_load_opts(opts))),
         {:ok, info} <- native_call(Native.model_info(ref)),
         {:ok, device} <- decode_device(info.device) do
      {:ok,
       %Model{
         ref: ref,
         path: path,
         sampling_rate: info.sampling_rate,
         multilingual: info.multilingual,
         n_vocab: info.n_vocab,
         device: device
       }}
    end
  end

  @doc """
  Transcribes `audio` using `model`.

  Returns `{:ok, %WhisperCpp.Transcription{}}` whose `:segments` carry
  absolute start/end times, `no_speech_prob`, `avg_logprob`, the
  underlying text tokens, and (when `:word_timestamps` is set) per-word
  timing.

  ## Options

  - `:language` - ISO 639-1 code (`"de"`), a full language name whisper.cpp
    knows (`"german"`), or `"auto"`. `nil` (default) auto-detects on
    multilingual models; English-only models resolve `nil`/`"auto"` to
    `"en"` and reject any other language. Unknown codes - including BCP 47
    tags such as `"de-CH"` - return `{:error, %Error{reason: :invalid_request}}`.
  - `:translate` - translate to English instead of transcribing.
  - `:initial_prompt` - free-text context prepended via `<|startofprev|>`
    to bias decoding (max ~224 tokens).
  - `:word_timestamps` - attach per-word timing. Default `false`.
  - `:beam_size` - beam-search width; `2` or higher enables beam search,
    up to whisper.cpp's limit of `8`. Default: greedy decoding.
  - `:best_of` - sampling candidates on temperature-fallback passes (used
    in greedy and beam mode), `1..8`. Default `5`, matching whisper.cpp.
  - `:temperature` - start of whisper.cpp's retry ladder, `0.0..1.0`
    (`0.0` = deterministic first pass).
  - `:n_threads` - intra-op threads, up to `512`. Default: whisper.cpp
    picks `min(4, cores)`.
  - `:n_max_text_ctx` - cap decoder context tokens.
  - `:offset_ms`, `:duration_ms` - clip the audio window; `:duration_ms`
    must be at least 1.
  - `:no_speech_thold` - silence detection threshold. Default `0.6`.
  - `:logprob_thold` - reject segments with `avg_logprob` below this.
  - `:suppress_blank`, `:suppress_non_speech_tokens` - decoder suppressions.
  - `:single_segment` - force a single segment per 30 s whisper window
    (audio longer than one window still yields several segments).
  - `:print_progress` - whisper.cpp progress to stderr.
  - `:vad_model_path` - path to a silero VAD model in GGML format (download
    `ggml-silero-v5.1.2.bin` from
    <https://huggingface.co/ggml-org/whisper-vad>). When set, whisper.cpp
    detects speech first, strips silence before the encoder, and remaps all
    timestamps back to the original timeline. Audio that contains no speech
    returns `{:ok, %Transcription{text: "", segments: []}}`. A path that is
    not a regular file returns `{:error, %Error{reason: :invalid_request}}`.
  - `:vad_threshold` - silero speech probability threshold in `0.0..1.0`;
    frames above it count as speech. Default `0.5`.
  - `:vad_min_speech_ms` - discard detected speech segments shorter than
    this. Default `250`.
  - `:vad_min_silence_ms` - a silence gap must be at least this long to end
    a speech segment. Default `100`.
  - `:vad_speech_pad_ms` - padding added before and after each detected
    speech segment to avoid clipping. Default `30`.
  - `:abort_handle` - `%WhisperCpp.AbortHandle{}` whose `abort/1` cancels
    in-flight inference. The call returns `{:ok, partial_transcription}`
    with whatever segments completed before the abort took effect. The
    VAD pass itself is not interruptible; the flag is honoured right
    after it, before the encoder starts.
  - `:progress_pid` - pid that receives `{:whisper_progress, percent}`
    messages (0..100) as work advances; duplicate percentages are
    coalesced. Messages already in flight can arrive after the call
    returns.
  """
  @spec transcribe(Model.t(), audio(), [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe(model, audio, opts \\ [])

  def transcribe(%Model{} = model, audio, opts) when is_list(opts) do
    with :ok <- validate_options(opts, transcribe_validators()),
         :ok <- validate_vad_options(opts),
         {:ok, samples} <- resolve_audio(audio) do
      do_transcribe(model, samples, opts, 0.0)
    end
  end

  def transcribe(_model, _audio, _opts) do
    {:error, Error.new(:invalid_request, "expected a %WhisperCpp.Model{} and a keyword list")}
  end

  @doc """
  Transcribes a `[start_s, end_s)` slice of `samples` and shifts the
  returned segment/word timestamps to absolute seconds in the original
  audio.

  Slices the f32 PCM buffer, runs whisper.cpp on the slice, and rewrites
  local segment times back into the absolute timeline. Returns
  `{:ok, %Transcription{}}` with absolute timings, or
  `{:error, Error.t()}`. Slices shorter than 0.3 s return an empty
  transcription (whisper.cpp pads short inputs and hallucinates into the
  padding).
  """
  @spec transcribe_slice(Model.t(), binary(), {number(), number()}, [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe_slice(model, samples, range, opts \\ [])

  def transcribe_slice(%Model{} = model, samples, {start_s, end_s}, opts)
      when is_binary(samples) and is_number(start_s) and is_number(end_s) and is_list(opts) do
    with :ok <- validate_options(opts, transcribe_validators()),
         :ok <- validate_vad_options(opts),
         :ok <- validate_slice_range(start_s, end_s),
         {:ok, slice} <- Pcm.slice(samples, sample_rate(), start_s, end_s - start_s),
         {:ok, transcription} <- do_transcribe(model, slice, opts, start_s * 1.0) do
      {:ok, transcription}
    else
      {:short, _} -> short_slice_result(model, samples, start_s, end_s, opts)
      err -> err
    end
  end

  def transcribe_slice(_model, _samples, _range, _opts) do
    {:error, Error.new(:invalid_request, "expected a %Model{}, an f32 PCM binary, and a {start_s, end_s} tuple")}
  end

  defp validate_slice_range(start_s, _end_s) when start_s < 0,
    do: {:error, Error.new(:invalid_request, "start_s must be >= 0", %{start_s: start_s})}

  defp validate_slice_range(start_s, end_s) when end_s <= start_s,
    do:
      {:error,
       Error.new(:invalid_request, "end_s must be greater than start_s", %{
         start_s: start_s,
         end_s: end_s
       })}

  # Strictly-below comparison with an epsilon: a window of exactly the
  # documented 0.3 s minimum must transcribe even when float subtraction
  # lands a hair under (2.3 - 2.0 == 0.2999...).
  defp validate_slice_range(start_s, end_s) when end_s - start_s < 0.3 - 1.0e-9,
    do: {:short, end_s - start_s}

  defp validate_slice_range(_start_s, _end_s), do: :ok

  # Sub-0.3 s windows return an empty transcription, but only after the
  # same buffer checks a full slice would run - an out-of-bounds or
  # malformed request is a caller bug regardless of window size.
  defp short_slice_result(model, samples, start_s, end_s, opts) do
    cond do
      rem(byte_size(samples), 4) != 0 ->
        {:error,
         Error.new(:invalid_request, "samples binary length must be a multiple of 4 (f32)", %{
           byte_size: byte_size(samples)
         })}

      end_s > Pcm.duration_s(samples, sample_rate()) ->
        {:error,
         Error.new(:invalid_request, "requested window extends past the end of the buffer", %{
           start_s: start_s,
           end_s: end_s,
           buffer_duration_s: Pcm.duration_s(samples, sample_rate())
         })}

      true ->
        with :ok <- validate_request_semantics(model, opts) do
          {:ok, empty_transcription(start_s, end_s, Keyword.get(opts, :language))}
        end
    end
  end

  # Mirrors the native request checks (`resolve_language` and friends in
  # transcribe.rs, which stay authoritative for full runs) so semantics
  # do not depend on slice length: a request the native path rejects
  # must not succeed just because the window is under 0.3 s.
  defp validate_request_semantics(%Model{multilingual: multilingual}, opts) do
    with :ok <- check_language(Keyword.get(opts, :language), multilingual),
         :ok <- check_translate(Keyword.get(opts, :translate, false), multilingual),
         :ok <- check_prompt(Keyword.get(opts, :initial_prompt)) do
      check_vad_path(Keyword.get(opts, :vad_model_path))
    end
  end

  defp check_language(language, _multilingual) when language in [nil, "auto"], do: :ok

  defp check_language(language, multilingual) do
    cond do
      not Native.known_language?(language) ->
        {:error,
         Error.new(
           :invalid_request,
           "unknown language #{inspect(language)}; pass an ISO 639-1 code whisper.cpp " <>
             "supports (e.g. \"de\"), a full language name (\"german\"), or \"auto\""
         )}

      not multilingual and language not in ["en", "english"] ->
        {:error,
         Error.new(
           :invalid_request,
           "model is English-only; language #{inspect(language)} is unavailable " <>
             "(use \"en\", \"auto\", or omit the option)"
         )}

      true ->
        :ok
    end
  end

  defp check_translate(true, false = _multilingual) do
    {:error, Error.new(:invalid_request, "model is English-only; translate has nothing to translate from")}
  end

  defp check_translate(_translate, _multilingual), do: :ok

  defp check_prompt(prompt) when is_binary(prompt) do
    if String.contains?(prompt, <<0>>) do
      {:error, Error.new(:invalid_request, "initial_prompt must not contain NUL bytes")}
    else
      :ok
    end
  end

  defp check_prompt(_prompt), do: :ok

  defp check_vad_path(path) when is_binary(path) do
    if File.regular?(path) do
      :ok
    else
      {:error, Error.new(:invalid_request, "vad_model_path is not a regular file: #{inspect(path)}")}
    end
  end

  defp check_vad_path(_path), do: :ok

  defp empty_transcription(start_s, end_s, language) do
    %Transcription{
      text: "",
      segments: [],
      language: language || "",
      duration_s: (end_s - start_s) * 1.0
    }
  end

  defp do_transcribe(%Model{ref: ref}, samples, opts, offset_s) do
    abort_ref =
      case Keyword.get(opts, :abort_handle) do
        %AbortHandle{ref: ref} -> ref
        nil -> nil
      end

    progress_pid = Keyword.get(opts, :progress_pid)

    case Native.transcribe(ref, samples, build_transcribe_opts(opts), abort_ref, progress_pid) do
      {:ok, payload} -> {:ok, build_transcription(payload, offset_s)}
      {:error, payload} -> {:error, Error.from_native(payload)}
    end
  end

  defp resolve_audio({:pcm_f32, samples}) when is_binary(samples) do
    cond do
      byte_size(samples) == 0 ->
        {:error, Error.new(:invalid_request, "PCM binary is empty")}

      rem(byte_size(samples), 4) != 0 ->
        {:error,
         Error.new(:invalid_request, "PCM binary length must be a multiple of 4 (f32)", %{
           byte_size: byte_size(samples)
         })}

      true ->
        {:ok, samples}
    end
  end

  defp resolve_audio(_) do
    {:error,
     Error.new(
       :invalid_request,
       "audio must be {:pcm_f32, binary} (little-endian f32 mono at 16 kHz); " <>
         "decode files upstream"
     )}
  end

  defp sample_rate, do: @target_sample_rate

  defp native_call({:ok, _} = ok), do: ok
  defp native_call({:error, payload}), do: {:error, Error.from_native(payload)}

  defp build_load_opts(opts) do
    device =
      case {Keyword.get(opts, :device), Keyword.get(opts, :use_gpu, true)} do
        {nil, true} -> "auto"
        {nil, false} -> "cpu"
        # use_gpu: false wins, as the docs promise.
        {device, false} when device not in [:cpu, nil] -> "cpu"
        {device, _} -> Atom.to_string(device)
      end

    %{device: device}
  end

  @device_atoms Map.new(@devices, fn a -> {Atom.to_string(a), a} end)

  defp decode_device(label) when is_binary(label) do
    case Map.fetch(@device_atoms, label) do
      {:ok, atom} ->
        {:ok, atom}

      :error ->
        {:error, Error.new(:runtime_error, "NIF reported unknown device", %{device: label})}
    end
  end

  @doc false
  @spec build_transcription(map(), float()) :: Transcription.t()
  def build_transcription(%{language: language, duration_s: duration_s, segments: raw_segments}, offset_s) do
    segments = Enum.map(raw_segments, &build_segment(&1, offset_s))

    # whisper.cpp segment text starts with its own leading space, so a
    # plain concatenation reproduces canonical whisper output (and adds
    # no spurious spaces to space-free scripts).
    text =
      segments
      |> Enum.map_join("", & &1.text)
      |> String.trim()

    %Transcription{
      text: text,
      segments: segments,
      language: language,
      duration_s: duration_s
    }
  end

  @doc false
  @spec build_segment(map(), float()) :: Segment.t()
  def build_segment(
        %{
          text: text,
          start: start,
          end: end_s,
          no_speech_prob: no_speech_prob,
          avg_logprob: avg_logprob,
          tokens: tokens,
          words: words
        },
        offset_s
      ) do
    %Segment{
      text: text,
      start: start + offset_s,
      end: end_s + offset_s,
      no_speech_prob: no_speech_prob,
      avg_logprob: avg_logprob,
      tokens: tokens,
      words: words && Enum.map(words, &build_word(&1, offset_s))
    }
  end

  @doc false
  @spec build_word(map(), float()) :: Word.t()
  def build_word(%{text: text, start: start, end: end_s, probability: probability}, offset_s) do
    %Word{text: text, start: start + offset_s, end: end_s + offset_s, probability: probability}
  end

  defp build_transcribe_opts(opts) do
    %{
      language: Keyword.get(opts, :language),
      translate: Keyword.get(opts, :translate),
      initial_prompt: Keyword.get(opts, :initial_prompt),
      word_timestamps: Keyword.get(opts, :word_timestamps),
      beam_size: Keyword.get(opts, :beam_size),
      best_of: Keyword.get(opts, :best_of),
      temperature: Keyword.get(opts, :temperature),
      n_threads: Keyword.get(opts, :n_threads),
      n_max_text_ctx: Keyword.get(opts, :n_max_text_ctx),
      offset_ms: Keyword.get(opts, :offset_ms),
      duration_ms: Keyword.get(opts, :duration_ms),
      no_speech_thold: Keyword.get(opts, :no_speech_thold),
      logprob_thold: Keyword.get(opts, :logprob_thold),
      suppress_blank: Keyword.get(opts, :suppress_blank),
      suppress_non_speech_tokens: Keyword.get(opts, :suppress_non_speech_tokens),
      single_segment: Keyword.get(opts, :single_segment),
      print_progress: Keyword.get(opts, :print_progress),
      vad_model_path: Keyword.get(opts, :vad_model_path),
      vad_threshold: Keyword.get(opts, :vad_threshold),
      vad_min_speech_ms: Keyword.get(opts, :vad_min_speech_ms),
      vad_min_silence_ms: Keyword.get(opts, :vad_min_silence_ms),
      vad_speech_pad_ms: Keyword.get(opts, :vad_speech_pad_ms)
    }
  end

  @spec validate_non_empty_string(String.t(), atom()) :: :ok | {:error, Error.t()}
  defp validate_non_empty_string(value, name) do
    if String.trim(value) == "" do
      {:error, Error.new(:invalid_request, "#{name} must be a non-empty string")}
    else
      :ok
    end
  end

  defp load_validators do
    %{device: &(&1 in @devices), use_gpu: &is_boolean/1}
  end

  # GGML aborts the process past GGML_MAX_N_THREADS.
  @ggml_max_threads 512

  defp transcribe_validators do
    Map.merge(decoding_validators(), vad_validators())
  end

  defp decoding_validators do
    %{
      language: &valid_optional_string?/1,
      translate: &is_boolean/1,
      initial_prompt: &valid_optional_string?/1,
      word_timestamps: &is_boolean/1,
      beam_size: &decoder_count?/1,
      best_of: &decoder_count?/1,
      temperature: &temperature?/1,
      n_threads: &(positive_integer?(&1) and &1 <= @ggml_max_threads),
      n_max_text_ctx: &non_neg_integer?/1,
      offset_ms: &non_neg_integer?/1,
      duration_ms: &positive_integer?/1,
      no_speech_thold: &number?/1,
      logprob_thold: &number?/1,
      suppress_blank: &is_boolean/1,
      suppress_non_speech_tokens: &is_boolean/1,
      single_segment: &is_boolean/1,
      print_progress: &is_boolean/1,
      abort_handle: &valid_abort_handle?/1,
      progress_pid: &valid_optional_pid?/1
    }
  end

  # The vad_* tuning options only act when a VAD model is set; a silent
  # no-op would hide caller bugs.
  defp validate_vad_options(opts) do
    tuning =
      for {key, _} <- opts,
          key in [:vad_threshold, :vad_min_speech_ms, :vad_min_silence_ms, :vad_speech_pad_ms],
          do: key

    if tuning == [] or Keyword.get(opts, :vad_model_path) != nil do
      :ok
    else
      {:error,
       Error.new(
         :invalid_request,
         "#{inspect(tuning)} have no effect without :vad_model_path"
       )}
    end
  end

  # whisper.cpp converts the millisecond knobs to sample counts in a C
  # int; values past ~134 s would overflow inside the detector. Two
  # minutes is far beyond any useful setting.
  @vad_ms_max 120_000

  defp vad_validators do
    %{
      vad_model_path: &valid_optional_string?/1,
      vad_threshold: &probability?/1,
      vad_min_speech_ms: &(positive_integer?(&1) and &1 <= @vad_ms_max),
      vad_min_silence_ms: &(positive_integer?(&1) and &1 <= @vad_ms_max),
      vad_speech_pad_ms: &(non_neg_integer?(&1) and &1 <= @vad_ms_max)
    }
  end

  defp valid_abort_handle?(nil), do: true
  defp valid_abort_handle?(%AbortHandle{}), do: true
  defp valid_abort_handle?(_), do: false

  defp valid_optional_pid?(nil), do: true
  defp valid_optional_pid?(pid) when is_pid(pid), do: true
  defp valid_optional_pid?(_), do: false

  @spec validate_options(keyword(), map()) :: :ok | {:error, Error.t()}
  defp validate_options(opts, validators) do
    Enum.reduce_while(opts, :ok, fn pair, :ok -> check_option(pair, validators) end)
  end

  defp check_option({key, value}, validators) when is_atom(key) do
    case Map.fetch(validators, key) do
      :error ->
        {:halt, {:error, Error.new(:invalid_request, "unknown option #{inspect(key)}")}}

      {:ok, validator} ->
        if validator.(value) do
          {:cont, :ok}
        else
          {:halt, {:error, Error.new(:invalid_request, "invalid value for option #{inspect(key)}: #{inspect(value)}")}}
        end
    end
  end

  defp check_option(other, _validators) do
    {:halt, {:error, Error.new(:invalid_request, "options must be a keyword list; got element #{inspect(other)}")}}
  end

  defp valid_optional_string?(nil), do: true

  defp valid_optional_string?(value) when is_binary(value),
    do: String.valid?(value) and String.trim(value) != ""

  defp valid_optional_string?(_), do: false

  defp probability?(v), do: number?(v) and v >= 0 and v <= 1

  # whisper.cpp builds its retry ladder from the start temperature in
  # +0.2 steps up to 1.0; a start above 1.0 yields an empty ladder and
  # undefined decoder state.
  defp temperature?(v), do: number?(v) and v >= 0 and v <= 1

  # whisper.cpp allocates WHISPER_MAX_DECODERS (8) decoder slots.
  defp decoder_count?(v), do: is_integer(v) and v in 1..8

  # The NIF decodes integer options as u32; larger values would raise a
  # decode ArgumentError instead of returning :invalid_request.
  @u32_max 4_294_967_295

  defp positive_integer?(v), do: is_integer(v) and v > 0 and v <= @u32_max
  defp non_neg_integer?(v), do: is_integer(v) and v >= 0 and v <= @u32_max
  # Floats cross the NIF as f32; values outside its range fail decode
  # with a raise instead of an error tuple.
  @f32_max 3.402_823_5e38

  defp number?(v), do: (is_integer(v) or is_float(v)) and abs(v) <= @f32_max
end
