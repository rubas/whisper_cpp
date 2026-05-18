defmodule WhisperCpp.Segment do
  @moduledoc """
  One segment of a whisper.cpp transcription.

  Times are absolute seconds within the input audio. `tokens` is the raw
  text-token ID list (timestamp tokens stripped). `no_speech_prob` is
  whisper.cpp's `no_speech` probability for the segment. `avg_logprob` is
  the segment's average token log probability - filter at e.g.
  `avg_logprob < -1.0` to reject low-confidence hallucinations.
  `words` is `nil` unless `:word_timestamps` was set on the transcribe
  call; when present it carries one `%WhisperCpp.Word{}` per Whisper
  token-word with its own time span.
  """

  alias WhisperCpp.Word

  @type t :: %__MODULE__{
          text: String.t(),
          start: float(),
          end: float(),
          no_speech_prob: float(),
          avg_logprob: float(),
          tokens: [non_neg_integer()],
          words: [Word.t()] | nil
        }

  @enforce_keys [:text, :start, :end, :no_speech_prob, :avg_logprob, :tokens]
  defstruct [:text, :start, :end, :no_speech_prob, :avg_logprob, :tokens, words: nil]
end
