defmodule WhisperCpp.WavTest do
  @moduledoc """
  Tests for `WhisperCpp.Wav` - RIFF/WAVE parsing, format rejection for
  unsupported sample rates / channel counts / bit depths, and disk I/O
  failure paths.
  """

  use ExUnit.Case, async: true

  alias WhisperCpp.Error
  alias WhisperCpp.Test.Fixtures
  alias WhisperCpp.Wav

  describe "decode/1" do
    test "decodes a 16 kHz mono 16-bit PCM WAV" do
      bytes = Fixtures.wav_fixture_bytes()
      assert {:ok, pcm} = Wav.decode(bytes)
      # 8_000 samples * 4 bytes
      assert byte_size(pcm) == 32_000
    end

    test "rejects a non-RIFF blob" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} = Wav.decode(<<"NOPE">>)
      assert msg =~ "RIFF"
    end

    test "rejects unsupported sample rates" do
      bytes = build_wav(sample_rate: 44_100)
      assert {:error, %Error{reason: :invalid_request, message: msg}} = Wav.decode(bytes)
      assert msg =~ "16 kHz"
    end

    test "rejects unsupported channel counts" do
      bytes = build_wav(channels: 5)
      assert {:error, %Error{reason: :invalid_request, message: msg}} = Wav.decode(bytes)
      assert msg =~ "channel"
    end

    test "rejects unsupported bits per sample" do
      bytes = build_wav(bits: 24)
      assert {:error, %Error{reason: :invalid_request, message: msg}} = Wav.decode(bytes)
      assert msg =~ "bits"
    end
  end

  describe "target_rate/0" do
    test "returns 16 kHz" do
      assert Wav.target_rate() == 16_000
    end
  end

  describe "read_file/1" do
    test "rejects a missing file" do
      assert {:error, %Error{reason: :invalid_request}} =
               Wav.read_file("/nonexistent/audio.wav")
    end

    test "decodes a written WAV from disk" do
      tmp = Path.join(System.tmp_dir!(), "whisper_cpp_wav_test_#{:rand.uniform(1_000_000)}.wav")
      File.write!(tmp, Fixtures.wav_fixture_bytes())
      on_exit(fn -> File.rm(tmp) end)

      assert {:ok, pcm} = Wav.read_file(tmp)
      assert byte_size(pcm) == 32_000
    end
  end

  defp build_wav(opts) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)
    channels = Keyword.get(opts, :channels, 1)
    bits = Keyword.get(opts, :bits, 16)
    block_align = div(channels * bits, 8)
    byte_rate = sample_rate * block_align

    fmt_body = <<
      1::little-16,
      channels::little-16,
      sample_rate::little-32,
      byte_rate::little-32,
      block_align::little-16,
      bits::little-16
    >>

    fmt = <<"fmt ", byte_size(fmt_body)::little-32, fmt_body::binary>>
    data = <<"data", 0::little-32>>
    body = <<"WAVE", fmt::binary, data::binary>>
    <<"RIFF", byte_size(body)::little-32, body::binary>>
  end
end
