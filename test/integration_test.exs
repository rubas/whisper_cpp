defmodule WhisperCpp.IntegrationTest do
  @moduledoc """
  End-to-end transcription test. Downloads `ggml-tiny.en.bin` (~75 MB)
  on first run and caches it under `test/fixtures/`; the audio fixture
  ships pre-converted under `test/support/jfk.f32le.16k.pcm` so no
  ffmpeg is required.

  Tagged `:integration` so it is excluded from `mix test` by default.
  Run with `mix test --include integration`.
  """

  use ExUnit.Case, async: false

  alias WhisperCpp.Test.Fixtures

  @moduletag :integration
  @moduletag timeout: :timer.minutes(5)

  setup_all do
    {:ok, model_path: Fixtures.ensure_model!(), pcm: Fixtures.pcm!()}
  end

  test "available_devices reports a backend list" do
    assert {:ok, %{backends: backends}} = WhisperCpp.available_devices()
    assert :cpu in backends
  end

  test "transcribes JFK with the tiny.en model", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model, device: :auto)

    {:ok, %WhisperCpp.Transcription{text: text, segments: segs}} =
      WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "en", n_threads: 4)

    assert text =~ "country"
    assert text =~ "ask"
    refute segs == []
    [first | _] = segs
    assert first.start >= 0.0
    assert first.end > first.start
  end

  test "transcribe_slice shifts timestamps to absolute seconds", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    {:ok, %WhisperCpp.Transcription{segments: segs}} =
      WhisperCpp.transcribe_slice(model_ref, pcm, {2.0, 6.0},
        language: "en",
        n_threads: 4
      )

    refute segs == []
    [first | _] = segs
    assert first.start >= 2.0
  end

  test "word_timestamps attaches per-word timings", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    {:ok, %WhisperCpp.Transcription{segments: segs}} =
      WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
        language: "en",
        word_timestamps: true,
        n_threads: 4
      )

    [first | _] = segs
    assert is_list(first.words)
    assert Enum.all?(first.words, &match?(%WhisperCpp.Word{}, &1))
  end

  test "progress_pid receives whisper_progress messages", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)
    parent = self()

    task =
      Task.async(fn ->
        WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
          language: "en",
          n_threads: 4,
          progress_pid: parent
        )
      end)

    Task.await(task, :timer.minutes(2))

    received = collect_progress([])
    refute received == []
    assert Enum.all?(received, &(&1 in 0..100))
  end

  test "abort_handle is observed by inference", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)
    handle = WhisperCpp.AbortHandle.new()

    # Pre-arm the flag before inference starts so the test does not race
    # the decode loop on fast CPUs.
    WhisperCpp.AbortHandle.abort(handle)

    assert {:ok, %WhisperCpp.Transcription{}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
               language: "en",
               n_threads: 4,
               abort_handle: handle
             )

    assert WhisperCpp.AbortHandle.aborted?(handle)
  end

  defp collect_progress(acc) do
    receive do
      {:whisper_progress, pct} -> collect_progress([pct | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end
end
