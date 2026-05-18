defmodule WhisperCpp.Native do
  @moduledoc """
  Low-level Rustler bindings to whisper.cpp via the `whisper-rs` crate.

  This module is private to the library. Use `WhisperCpp` for the public
  API. Stub names must match the Rust NIF symbols verbatim (Rustler
  verifies arity at module load time); user-friendly wrappers live below
  them.
  """

  @cargo_features_env System.get_env("WHISPER_CPP_FEATURES", "")
  @cargo_features_raw Application.compile_env(:whisper_cpp, :cargo_features, @cargo_features_env)
  @cargo_features String.split(@cargo_features_raw, ~r/[,\s]+/, trim: true)

  @version Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :whisper_cpp,
    crate: "whisper_cpp_native",
    base_url: "https://github.com/rubas/whisper_cpp/releases/download/v#{@version}",
    version: @version,
    # Opt-in source build; toggled by env or umbrella config.
    force_build:
      System.get_env("WHISPER_CPP_BUILD") in ["1", "true"] or
        Application.compile_env(:rustler_precompiled, [:force_build, :whisper_cpp], false),
    nif_versions: ["2.17"],
    targets: ~w(
      aarch64-apple-darwin
      x86_64-unknown-linux-gnu
      aarch64-unknown-linux-gnu
    ),
    # Variant matrix: every target ships a default CPU artefact; x86_64 and
    # aarch64 Linux also ship `--cuda` and `--hipblas` variants for GPU
    # acceleration. Selected at install time via WHISPER_CPP_VARIANT.
    variants: %{
      "x86_64-unknown-linux-gnu" => [
        cuda: fn -> System.get_env("WHISPER_CPP_VARIANT") == "cuda" end,
        hipblas: fn -> System.get_env("WHISPER_CPP_VARIANT") == "hipblas" end
      ],
      "aarch64-unknown-linux-gnu" => [
        cuda: fn -> System.get_env("WHISPER_CPP_VARIANT") == "cuda" end
      ]
    },
    features: @cargo_features

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
  Returns a structured transcription map.
  """
  @spec transcribe(reference(), binary(), map()) :: {:ok, map()} | {:error, map()}
  def transcribe(model, samples_bin, opts), do: nif_transcribe(model, samples_bin, opts)

  defp nif_available_devices, do: :erlang.nif_error(:nif_not_loaded)
  defp nif_load_model(_path, _opts), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_model_info(_model), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_transcribe(_model, _samples_bin, _opts), do: :erlang.nif_error(:nif_not_loaded)
end
