defmodule WhisperCpp.IntegrationTest do
  @moduledoc """
  End-to-end transcription test. Downloads `ggml-tiny.en.bin` (~75 MB) and
  the JFK sample on first run, caches them under `test/fixtures/`, and
  runs a real whisper.cpp inference.

  Tagged `:integration` so it is excluded from `mix test` by default.
  Run with `mix test --include integration`.
  """

  use ExUnit.Case, async: false

  alias WhisperCpp.Test.Fixtures

  @moduletag :integration
  @moduletag timeout: :timer.minutes(5)

  setup_all do
    {:ok, model_path: Fixtures.ensure_model!(), audio_path: Fixtures.ensure_audio!()}
  end

  test "available_devices reports a backend list" do
    assert {:ok, %{backends: backends}} = WhisperCpp.available_devices()
    assert :cpu in backends
  end

  test "transcribes JFK with the tiny.en model", %{model_path: model, audio_path: audio} do
    {:ok, model_ref} = WhisperCpp.load_model(model, device: :auto)

    {:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
      WhisperCpp.transcribe(model_ref, audio, language: "en", n_threads: 4)

    assert text =~ "country"
    assert text =~ "ask"
    refute segs == []
    [first | _] = segs
    assert first.start >= 0.0
    assert first.end > first.start
  end

  test "transcribe_slice shifts timestamps to absolute seconds", %{model_path: model, audio_path: audio} do
    {:ok, model_ref} = WhisperCpp.load_model(model)
    {:ok, samples} = WhisperCpp.Wav.read_file(audio)

    {:ok, %WhisperCpp.Transcription{segments: segs}} =
      WhisperCpp.transcribe_slice(model_ref, samples, {2.0, 6.0},
        language: "en",
        n_threads: 4
      )

    refute segs == []
    [first | _] = segs
    assert first.start >= 2.0
  end

  test "word_timestamps attaches per-word timings", %{model_path: model, audio_path: audio} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    {:ok, %WhisperCpp.Transcription{segments: segs}} =
      WhisperCpp.transcribe(model_ref, audio, language: "en", word_timestamps: true, n_threads: 4)

    [first | _] = segs
    assert is_list(first.words)
    assert Enum.all?(first.words, &match?(%WhisperCpp.Word{}, &1))
  end
end
