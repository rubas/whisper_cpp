defmodule WhisperCpp.Wav do
  @moduledoc """
  Minimal RIFF/WAVE decoder for the formats whisper.cpp consumes directly.

  Accepts 16 kHz audio in any of: mono / stereo 16-bit PCM, mono / stereo
  32-bit PCM, mono / stereo 32-bit float. Stereo is downmixed by
  averaging channels. Sample rates other than 16 kHz are rejected;
  resample upstream (e.g. `ffmpeg -ar 16000 -ac 1`).

  Decoding runs in the NIF (dirty-CPU scheduled) so hour-long inputs do
  not block a BEAM scheduler with per-sample arithmetic. Returns samples
  as a binary of little-endian `f32` values in `[-1.0, 1.0]`, ready to
  feed into `WhisperCpp.transcribe/3`.
  """

  alias WhisperCpp.Error
  alias WhisperCpp.Native

  @target_rate 16_000

  @doc """
  The sample rate this decoder produces (always 16 kHz). Exposed so the
  rest of the library can assert that audio matches without reaching
  into the module attribute.
  """
  @spec target_rate() :: pos_integer()
  def target_rate, do: @target_rate

  # 256 MiB cap on `read_file/1` - refuses a typo'd path or huge file
  # before slurping it into the BEAM heap.
  @max_bytes 268_435_456

  @spec read_file(Path.t()) :: {:ok, binary()} | {:error, Error.t()}
  def read_file(path) do
    with {:ok, bytes} <- read_bytes(path) do
      decode(bytes)
    end
  end

  @doc """
  Decodes the bytes of a WAV file into little-endian f32 PCM at 16 kHz mono.
  """
  @spec decode(binary()) :: {:ok, binary()} | {:error, Error.t()}
  def decode(bytes) when is_binary(bytes) do
    case Native.decode_wav(bytes) do
      {:ok, pcm} -> {:ok, pcm}
      {:error, payload} -> {:error, Error.from_native(payload)}
    end
  end

  defp read_bytes(path) do
    case File.stat(path) do
      {:ok, %File.Stat{size: size}} when size > @max_bytes ->
        {:error,
         Error.new(:invalid_request, "WAV file exceeds the in-memory size cap", %{
           path: path,
           size: size,
           max_bytes: @max_bytes
         })}

      {:ok, _stat} ->
        case File.read(path) do
          {:ok, bytes} ->
            {:ok, bytes}

          {:error, posix} ->
            {:error, Error.new(:invalid_request, "cannot read WAV", %{posix: posix, path: path})}
        end

      {:error, posix} ->
        {:error, Error.new(:invalid_request, "cannot stat WAV", %{posix: posix, path: path})}
    end
  end
end
