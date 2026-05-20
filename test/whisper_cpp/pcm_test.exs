defmodule WhisperCpp.PcmTest do
  @moduledoc """
  Tests for `WhisperCpp.Pcm` - byte-math, alignment guards, and bounds
  rejection on f32 PCM slicing.
  """

  use ExUnit.Case, async: true

  alias WhisperCpp.Error
  alias WhisperCpp.Pcm

  defp samples(seconds), do: :binary.copy(<<1.0::little-float-32>>, 16_000 * seconds)

  describe "slice/4" do
    test "returns the requested byte window" do
      buffer = samples(3)
      assert {:ok, slice} = Pcm.slice(buffer, 16_000, 1.0, 1.0)
      assert byte_size(slice) == 16_000 * 4
    end

    test "rejects negative start_s" do
      one_second = samples(1)

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               Pcm.slice(one_second, 16_000, -0.1, 0.5)

      assert msg =~ ">= 0"
    end

    test "rejects non-positive duration" do
      one_second = samples(1)
      assert {:error, %Error{reason: :invalid_request}} = Pcm.slice(one_second, 16_000, 0.0, 0)
      assert {:error, %Error{reason: :invalid_request}} = Pcm.slice(one_second, 16_000, 0.0, -1.0)
    end

    test "rejects misaligned binary length" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               Pcm.slice(<<1, 2, 3>>, 16_000, 0.0, 0.001)

      assert msg =~ "multiple of 4"
    end

    test "rejects request past buffer end" do
      one_second = samples(1)

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               Pcm.slice(one_second, 16_000, 0.0, 2.0)

      assert msg =~ "past the end"
    end

    test "rejects start past buffer end" do
      one_second = samples(1)
      assert {:error, %Error{reason: :invalid_request}} = Pcm.slice(one_second, 16_000, 5.0, 0.1)
    end

    test "rejects bad argument shapes" do
      assert {:error, %Error{reason: :invalid_request}} = Pcm.slice("not-binary", 16_000, 0, 0.1)
      assert {:error, %Error{reason: :invalid_request}} = Pcm.slice(<<>>, 0, 0, 0.1)
    end
  end

  describe "duration_s/2" do
    test "computes seconds from byte size" do
      two_seconds = samples(2)
      assert Pcm.duration_s(two_seconds, 16_000) == 2.0
      assert Pcm.duration_s(<<>>, 16_000) == 0.0
    end
  end
end
