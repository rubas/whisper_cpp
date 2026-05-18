defmodule WhisperCpp.Word do
  @moduledoc """
  One word produced by `:word_timestamps`.

  Times are absolute seconds within the input audio. `probability` is the
  minimum per-token acoustic probability across the tokens that make up
  this word (matching faster-whisper's per-word confidence reduction);
  filter at e.g. `probability < 0.3` to flag low-confidence words.
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
