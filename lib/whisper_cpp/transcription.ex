defmodule WhisperCpp.Transcription do
  @moduledoc """
  Result of a `WhisperCpp.transcribe/3` call.

  `text` is the concatenated, whitespace-trimmed transcript across every
  segment. `segments` is the structured per-segment decomposition produced
  by whisper.cpp, with absolute start/end times in seconds. `language` is
  the resolved ISO code (auto-detected when not pinned). `duration_s` is
  the input audio length, useful for VAD/diarization pipelines that hand
  short splices in.
  """

  alias WhisperCpp.Segment

  @type t :: %__MODULE__{
          text: String.t(),
          segments: [Segment.t()],
          language: String.t(),
          duration_s: float()
        }

  @enforce_keys [:text, :segments, :language, :duration_s]
  defstruct [:text, :segments, :language, :duration_s]
end
