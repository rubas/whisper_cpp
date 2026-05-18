defmodule WhisperCpp do
  @moduledoc """
  Native Elixir bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

  Calls into whisper.cpp's C API through the `whisper-rs` Rust crate via a
  Rustler NIF. No `whisper-cli` subprocess, no Python, no temporary WAV
  files. Structured per-segment results, `:initial_prompt` biasing,
  word-level timestamps, and CUDA / ROCm (hipBLAS) / CPU backends are all
  first-class.

  Diarization-driven workflows are first-class via `transcribe_slice/4`:
  decode WAV once, then run many short transcribe calls over per-turn
  slices with absolute timestamps preserved across the original audio
  timeline.

  ## Quickstart

      {:ok, model} = WhisperCpp.load_model("models/ggml-large-v3.bin")

      {:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
        WhisperCpp.transcribe(model, "jfk.wav", language: "en")

      IO.puts(text)
      for s <- segs, do: IO.puts("[\#{s.start}-\#{s.end}] \#{s.text}")

  ## Audio contract

  whisper.cpp expects **mono `f32` PCM samples at 16 kHz**, normalised to
  the `-1.0..1.0` range. `transcribe/3` accepts:

  - a `.wav` path (16 kHz mono or stereo, 16/32-bit PCM or 32-bit float),
    decoded by the built-in `WhisperCpp.Wav` module;
  - a `{:pcm_f32, binary}` tuple containing little-endian f32 samples.

  Raw bare-binary input is rejected on purpose: a typo'd path used to
  silently turn into garbage PCM. Use `{:pcm_f32, binary}` for in-memory
  buffers.

  ## Diarization-driven workflows

  Decode the master WAV once, then transcribe each diarization turn:

      {:ok, samples} = WhisperCpp.Wav.read_file("call.wav")

      for {start_s, end_s, _spk} <- turns do
        {:ok, slice} =
          WhisperCpp.Pcm.slice(samples, 16_000, start_s, end_s - start_s)

        {:ok, t} =
          WhisperCpp.transcribe(model, {:pcm_f32, slice},
            language: "en", word_timestamps: true)

        # `t.segments` carry per-segment times relative to the slice; add
        # `start_s` for absolute timings.
      end

  See `WhisperCpp.transcribe_slice/4` for a wrapper that does the slice
  math and timestamp shift for you.
  """

  alias WhisperCpp.AbortHandle
  alias WhisperCpp.Error
  alias WhisperCpp.Model
  alias WhisperCpp.Native
  alias WhisperCpp.Pcm
  alias WhisperCpp.Segment
  alias WhisperCpp.Transcription
  alias WhisperCpp.Wav
  alias WhisperCpp.Word

  @typedoc "Audio sources accepted by `transcribe/3` and `transcribe_slice/4`."
  @type audio :: Path.t() | {:pcm_f32, binary()}

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
          | {:duration_ms, non_neg_integer()}
          | {:no_speech_thold, float()}
          | {:logprob_thold, float()}
          | {:suppress_blank, boolean()}
          | {:suppress_non_speech_tokens, boolean()}
          | {:single_segment, boolean()}
          | {:print_progress, boolean()}
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

  - `:language` - ISO code (`"en"`). `nil` (default) auto-detects on
    multilingual models; auto-detect on monolingual models always returns
    `"en"`.
  - `:translate` - translate to English instead of transcribing.
  - `:initial_prompt` - free-text context prepended via `<|startofprev|>`
    to bias decoding (max ~224 tokens).
  - `:word_timestamps` - attach per-word timing. Default `false`.
  - `:beam_size` - beam-search width. Default `5`.
  - `:best_of` - greedy candidates kept when `beam_size <= 1`.
  - `:temperature` - sampling temperature (`0.0` = greedy/beam).
  - `:n_threads` - intra-op threads. Default `4`.
  - `:n_max_text_ctx` - cap decoder context tokens.
  - `:offset_ms`, `:duration_ms` - clip the audio window.
  - `:no_speech_thold` - silence detection threshold. Default `0.6`.
  - `:logprob_thold` - reject segments with `avg_logprob` below this.
  - `:suppress_blank`, `:suppress_non_speech_tokens` - decoder suppressions.
  - `:single_segment` - force a single segment for the whole audio.
  - `:print_progress` - whisper.cpp progress to stderr.
  - `:abort_handle` - `%WhisperCpp.AbortHandle{}` whose `abort/1` cancels
    in-flight inference. The call returns `{:ok, partial_transcription}`
    with whatever segments completed before the abort took effect.
  - `:progress_pid` - pid that receives `{:whisper_progress, percent}`
    messages (0..100) as work advances; duplicate percentages are
    coalesced.
  """
  @spec transcribe(Model.t(), audio(), [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe(model, audio, opts \\ [])

  def transcribe(%Model{} = model, audio, opts) when is_list(opts) do
    with :ok <- validate_options(opts, transcribe_validators()),
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

  The caller decodes the master WAV once; this function slices the f32
  PCM, runs whisper.cpp on the slice, and rewrites local segment times
  back into the global timeline so the output integrates with diarization
  spans without extra bookkeeping.

  Returns `{:ok, %Transcription{}}` with absolute timings, or
  `{:error, Error.t()}`. Slices shorter than 0.3 s return an empty
  transcription (whisper.cpp pads short inputs and hallucinates into the
  padding).
  """
  @spec transcribe_slice(Model.t(), binary(), {number(), number()}, [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe_slice(model, samples, range, opts \\ [])

  def transcribe_slice(%Model{} = model, samples, {start_s, end_s}, opts)
      when is_binary(samples) and is_number(start_s) and is_number(end_s) and is_list(opts) do
    with :ok <- validate_slice_range(start_s, end_s),
         :ok <- validate_options(opts, transcribe_validators()),
         {:ok, slice} <- Pcm.slice(samples, sample_rate(), start_s, end_s - start_s),
         {:ok, transcription} <- do_transcribe(model, slice, opts, start_s * 1.0) do
      {:ok, transcription}
    else
      {:short, _} -> {:ok, empty_transcription(start_s, end_s)}
      err -> err
    end
  end

  def transcribe_slice(_model, _samples, _range, _opts) do
    {:error, Error.new(:invalid_request, "expected a %Model{}, an f32 PCM binary, and a {start_s, end_s} tuple")}
  end

  # Order matters: validate the range itself (non-negative start, end
  # strictly greater than start) before the short-slice fast path so an
  # inverted or negative-duration tuple does not silently return an empty
  # transcription.
  defp validate_slice_range(start_s, _end_s) when start_s < 0,
    do: {:error, Error.new(:invalid_request, "start_s must be >= 0", %{start_s: start_s})}

  defp validate_slice_range(start_s, end_s) when end_s <= start_s,
    do:
      {:error,
       Error.new(:invalid_request, "end_s must be greater than start_s", %{
         start_s: start_s,
         end_s: end_s
       })}

  defp validate_slice_range(start_s, end_s) when end_s - start_s < 0.3, do: {:short, end_s - start_s}

  defp validate_slice_range(_start_s, _end_s), do: :ok

  defp empty_transcription(start_s, end_s) do
    %Transcription{text: "", segments: [], language: "", duration_s: (end_s - start_s) * 1.0}
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

  defp resolve_audio(path) when is_binary(path) do
    cond do
      not File.regular?(path) ->
        {:error, Error.new(:invalid_request, "audio path does not exist or is not a regular file", %{path: path})}

      String.ends_with?(path, ".wav") ->
        case Wav.read_file(path) do
          {:ok, <<>>} -> {:error, Error.new(:invalid_request, "WAV decoded to zero samples", %{path: path})}
          other -> other
        end

      true ->
        {:error,
         Error.new(
           :invalid_request,
           "only .wav paths are accepted; resample/decode upstream or pass {:pcm_f32, binary}",
           %{path: path}
         )}
    end
  end

  defp resolve_audio(_) do
    {:error, Error.new(:invalid_request, "audio must be a .wav path or {:pcm_f32, binary}")}
  end

  defp sample_rate, do: Wav.target_rate()

  defp native_call({:ok, _} = ok), do: ok
  defp native_call({:error, payload}), do: {:error, Error.from_native(payload)}

  defp build_load_opts(opts) do
    device =
      case Keyword.get(opts, :device) do
        nil -> if Keyword.get(opts, :use_gpu, true), do: "auto", else: "cpu"
        atom when is_atom(atom) -> Atom.to_string(atom)
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

    text =
      segments
      |> Enum.map_join(" ", & &1.text)
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
      print_progress: Keyword.get(opts, :print_progress)
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

  defp transcribe_validators do
    %{
      language: &valid_optional_string?/1,
      translate: &is_boolean/1,
      initial_prompt: &valid_optional_string?/1,
      word_timestamps: &is_boolean/1,
      beam_size: &positive_integer?/1,
      best_of: &positive_integer?/1,
      temperature: &non_neg_number?/1,
      n_threads: &positive_integer?/1,
      n_max_text_ctx: &non_neg_integer?/1,
      offset_ms: &non_neg_integer?/1,
      duration_ms: &non_neg_integer?/1,
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

  defp check_option({key, value}, validators) do
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

  defp valid_optional_string?(nil), do: true
  defp valid_optional_string?(value) when is_binary(value), do: String.trim(value) != ""
  defp valid_optional_string?(_), do: false

  defp positive_integer?(v), do: is_integer(v) and v > 0
  defp non_neg_integer?(v), do: is_integer(v) and v >= 0
  defp number?(v), do: is_integer(v) or is_float(v)
  defp non_neg_number?(v), do: number?(v) and v >= 0
end
