defmodule WhisperCpp.Native do
  @moduledoc """
  Low-level Rustler bindings to whisper.cpp via the `whisper-rs` crate.

  This module is private to the library. Use `WhisperCpp` for the public
  API. Stub names must match the Rust NIF symbols verbatim (Rustler
  verifies arity at module load time); user-friendly wrappers live below
  them.
  """

  alias WhisperCpp.Native.BuildEnv

  @cargo_features_env BuildEnv.get("WHISPER_CPP_FEATURES") || ""
  @cargo_features_raw Application.compile_env(:whisper_cpp, :cargo_features, @cargo_features_env)
  @cargo_features String.split(@cargo_features_raw, ~r/[,\s]+/, trim: true)

  @version Mix.Project.config()[:version]

  # Precompiled variants per target; release.yml builds the same matrix.
  @variants %{
    "x86_64-unknown-linux-gnu" => [:cuda, :hipblas],
    "aarch64-unknown-linux-gnu" => [:cuda]
  }
  @variant BuildEnv.get("WHISPER_CPP_VARIANT")
  @force_build BuildEnv.get("WHISPER_CPP_BUILD") in ["1", "true"] or
                 Application.compile_env(:rustler_precompiled, [:force_build, :whisper_cpp], false)

  # rustler_precompiled picks the default artefact when no variant of the
  # target matches, so check the request against the resolved target. A
  # source build ignores variants.
  with false <- @force_build,
       variant when is_binary(variant) <- @variant,
       {:ok, "nif-" <> nif_target} <- RustlerPrecompiled.target(),
       [_nif_version, target] = String.split(nif_target, "-", parts: 2),
       published = @variants |> Map.get(target, []) |> Enum.map(&Atom.to_string/1),
       false <- variant in published do
    raise CompileError,
      description:
        "WHISPER_CPP_VARIANT #{inspect(variant)} is not published for #{target}; expected one of #{inspect(published)}"
  end

  use RustlerPrecompiled,
    otp_app: :whisper_cpp,
    crate: "whisper_cpp_native",
    base_url: "https://github.com/rubas/whisper_cpp/releases/download/v#{@version}",
    version: @version,
    force_build: @force_build,
    nif_versions: ["2.17"],
    targets: ~w(
      aarch64-apple-darwin
      x86_64-unknown-linux-gnu
      aarch64-unknown-linux-gnu
    ),
    variants:
      Map.new(@variants, fn {target, names} ->
        {target, for(name <- names, do: {name, fn -> Atom.to_string(name) == @variant end})}
      end),
    features: @cargo_features

  @doc """
  Resolves a requested language the way `transcribe/5` does: the ISO
  code, or `""` when a multilingual model auto-detects.
  """
  @spec resolve_language(String.t() | nil, boolean()) :: {:ok, String.t()} | {:error, map()}
  def resolve_language(language, multilingual), do: nif_resolve_language(language, multilingual)

  @doc "Reports the active runtime backends compiled into this NIF artefact."
  @spec available_devices() :: {:ok, map()} | {:error, map()}
  def available_devices, do: nif_available_devices()

  @doc "Loads a GGUF/GGML whisper.cpp model file."
  @spec load_model(String.t(), map()) :: {:ok, reference()} | {:error, map()}
  def load_model(path, opts), do: nif_load_model(path, opts)

  @doc "Returns loaded-model metadata."
  @spec model_info(reference()) :: {:ok, map()} | {:error, map()}
  def model_info(model), do: nif_model_info(model)

  @doc """
  Runs whisper.cpp on a buffer of PCM samples.

  `samples_bin` is a binary of little-endian `f32` mono samples at 16 kHz.
  `abort_handle` is either `nil` or an opaque resource minted by
  `new_abort_handle/0`; signalling it from another process cancels
  in-flight inference. `progress_pid` is `nil` or a local pid that receives
  `{:whisper_progress, percent}` messages as work advances.
  """
  @spec transcribe(reference(), binary(), map(), reference() | nil, pid() | nil) ::
          {:ok, map()} | {:error, map()}
  def transcribe(model, samples_bin, opts, abort_handle, progress_pid),
    do: nif_transcribe(model, samples_bin, opts, abort_handle, progress_pid)

  @doc "Mints a fresh cooperative-cancellation handle."
  @spec new_abort_handle() :: reference()
  def new_abort_handle, do: nif_new_abort_handle()

  @doc "Signals an abort handle; in-flight transcribes observing it return early."
  @spec abort_handle_signal(reference()) :: :ok
  def abort_handle_signal(handle), do: nif_abort_handle_signal(handle)

  @doc "Returns `true` once an abort handle has been signalled."
  @spec abort_handle_aborted?(reference()) :: boolean()
  def abort_handle_aborted?(handle), do: nif_abort_handle_aborted(handle)

  defp nif_resolve_language(_language, _multilingual), do: :erlang.nif_error(:nif_not_loaded)

  defp nif_available_devices, do: :erlang.nif_error(:nif_not_loaded)
  defp nif_load_model(_path, _opts), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_model_info(_model), do: :erlang.nif_error(:nif_not_loaded)

  defp nif_transcribe(_model, _samples_bin, _opts, _abort_handle, _progress_pid),
    do: :erlang.nif_error(:nif_not_loaded)

  defp nif_new_abort_handle, do: :erlang.nif_error(:nif_not_loaded)
  defp nif_abort_handle_signal(_handle), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_abort_handle_aborted(_handle), do: :erlang.nif_error(:nif_not_loaded)
end
