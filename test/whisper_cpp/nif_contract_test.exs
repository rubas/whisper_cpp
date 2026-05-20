defmodule WhisperCpp.NifContractTest do
  @moduledoc """
  Static contract tests that pin the atom-keyed map shape the NIF returns
  to the struct fields the Elixir side reads. Runs without a loaded model
  so the contract is checked on every CI lane, not only the integration
  lane.
  """

  use ExUnit.Case, async: true

  test "transcription payload shape maps to %Transcription{}" do
    payload = %{
      language: "en",
      duration_s: 2.5,
      segments: [
        %{
          text: "ok",
          start: 0.0,
          end: 1.2,
          no_speech_prob: 0.05,
          avg_logprob: -0.3,
          tokens: [10, 11],
          words: nil
        }
      ]
    }

    t = WhisperCpp.build_transcription(payload, 0.0)
    assert %WhisperCpp.Transcription{language: "en", duration_s: 2.5} = t
    assert [%WhisperCpp.Segment{text: "ok", tokens: [10, 11]}] = t.segments
  end

  test "word payload shape maps to %Word{}" do
    payload = %{
      language: "en",
      duration_s: 1.0,
      segments: [
        %{
          text: "hi",
          start: 0.0,
          end: 0.5,
          no_speech_prob: 0.0,
          avg_logprob: -0.2,
          tokens: [1],
          words: [%{text: "hi", start: 0.0, end: 0.5, probability: 0.9}]
        }
      ]
    }

    t = WhisperCpp.build_transcription(payload, 0.0)
    [seg] = t.segments
    assert [%WhisperCpp.Word{text: "hi", probability: 0.9}] = seg.words
  end
end
