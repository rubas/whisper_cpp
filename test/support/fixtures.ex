defmodule WhisperCpp.Test.Fixtures do
  @moduledoc """
  Test fixture helpers.

  The integration suite downloads the `ggml-tiny.en` model (~75 MB) and
  the `jfk.wav` sample from the whisper.cpp release server, caches them
  under `test/fixtures/`, and reuses them across runs. Set
  `WHISPER_CPP_REFRESH=1` to force re-download.
  """

  @model_url "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
  @audio_url "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"

  @spec fixtures_dir() :: Path.t()
  def fixtures_dir do
    Path.join([File.cwd!(), "test", "fixtures"])
  end

  @spec model_path() :: Path.t()
  def model_path, do: Path.join(fixtures_dir(), "ggml-tiny.en.bin")

  @spec audio_path() :: Path.t()
  def audio_path, do: Path.join(fixtures_dir(), "jfk.wav")

  @spec ensure_model!() :: Path.t()
  def ensure_model!, do: ensure_file!(model_path(), @model_url)

  @spec ensure_audio!() :: Path.t()
  def ensure_audio!, do: ensure_file!(audio_path(), @audio_url)

  @spec wav_fixture_bytes() :: binary()
  def wav_fixture_bytes do
    # 16 kHz mono 16-bit PCM, 0.5 s of silence; small enough to embed in
    # tests without slowing them down.
    n_samples = 8_000
    data = <<0::size(n_samples * 16)>>

    fmt_chunk = <<
      0x10::little-32,
      1::little-16,
      1::little-16,
      16_000::little-32,
      32_000::little-32,
      2::little-16,
      16::little-16
    >>

    data_chunk = <<byte_size(data)::little-32, data::binary>>
    fmt = <<"fmt ", fmt_chunk::binary>>
    data_chunk_with_tag = <<"data", data_chunk::binary>>

    body = <<"WAVE", fmt::binary, data_chunk_with_tag::binary>>
    <<"RIFF", byte_size(body)::little-32, body::binary>>
  end

  defp ensure_file!(path, url) do
    File.mkdir_p!(Path.dirname(path))

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
    IO.puts(:stderr, "[fixtures] downloading #{url} -> #{dest}")
    tmp = dest <> ".part"

    {_, 0} =
      System.cmd("curl", ["-fLsS", "--retry", "3", "-o", tmp, url], stderr_to_stdout: true)

    File.rename!(tmp, dest)
  end
end
