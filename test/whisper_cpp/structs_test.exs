defmodule WhisperCpp.StructsTest do
  @moduledoc """
  Tests for default field values on result structs that aren't enforced
  by `@enforce_keys`. Drift here would silently break consumer pattern
  matches; the rest of the struct shape is covered by `nif_contract_test`.
  """

  use ExUnit.Case, async: true

  test "Segment.words defaults to nil when not provided" do
    seg = %WhisperCpp.Segment{
      text: "hi",
      start: 0.0,
      end: 0.5,
      no_speech_prob: 0.1,
      avg_logprob: -0.3,
      tokens: [1, 2]
    }

    assert seg.words == nil
  end
end
