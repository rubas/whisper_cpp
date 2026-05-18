defmodule WhisperCpp.Word do
  @moduledoc """
  One word produced by `:word_timestamps`.

  Times are absolute seconds within the input audio. `probability` is the
  per-token acoustic probability reported by whisper.cpp for the most
  likely token in this word.
  """

  @type t :: %__MODULE__{
          text: String.t(),
          start: float(),
          end: float(),
          probability: float()
        }

  @enforce_keys [:text, :start, :end, :probability]
  defstruct [:text, :start, :end, :probability]
end
