defmodule WhisperCpp.Test.Fixtures do
  @moduledoc """
  Test fixture helpers.

  The integration suite downloads the `ggml-tiny.en` model (~75 MB) on
  first run and caches it under `test/fixtures/`. Set
  `WHISPER_CPP_REFRESH=1` to force a re-download.

  Audio is shipped as a pre-converted PCM fixture (`jfk.f32le.16k.pcm`
  alongside this file) so tests need neither ffmpeg nor the JFK WAV
  download. The PCM is little-endian f32 mono at 16 kHz, matching the
  contract `WhisperCpp.transcribe/3` expects.
  """

  require Logger

  @model_url "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"

  @spec fixtures_dir() :: Path.t()
  def fixtures_dir do
    Path.join([File.cwd!(), "test", "fixtures"])
  end

  @spec model_path() :: Path.t()
  def model_path, do: Path.join(fixtures_dir(), "ggml-tiny.en.bin")

  @spec pcm_path() :: Path.t()
  def pcm_path do
    Path.join([File.cwd!(), "test", "support", "jfk.f32le.16k.pcm"])
  end

  @spec ensure_model!() :: Path.t()
  def ensure_model!, do: ensure_file!(model_path(), @model_url)

  @doc """
  Returns the JFK sample as a binary of little-endian f32 mono samples
  at 16 kHz. The file is committed at `test/support/jfk.f32le.16k.pcm`,
  so this is a plain `File.read!` — no network, no ffmpeg.
  """
  @spec pcm!() :: binary()
  def pcm!, do: File.read!(pcm_path())

  defp ensure_file!(path, url) do
    path |> Path.dirname() |> File.mkdir_p!()

    if System.get_env("WHISPER_CPP_REFRESH") == "1" do
      _ = File.rm(path)
    end

    if File.exists?(path) and File.stat!(path).size > 0 do
      path
    else
      download!(url, path)
      path
    end
  end

  defp download!(url, dest) do
    Logger.info("downloading fixture #{url} -> #{dest}")
    tmp = dest <> ".part"

    # Test-only fetch from a hardcoded literal URL. The `env: []` clears
    # the parent's environment so sensitive variables (auth tokens,
    # creds) cannot leak into the curl subprocess.
    {_, 0} =
      System.cmd("curl", ["-fLsS", "--retry", "3", "-o", tmp, url],
        stderr_to_stdout: true,
        env: []
      )

    File.rename!(tmp, dest)
  end
end
