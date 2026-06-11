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
    # consecutive duplicates are coalesced at the source
    assert received == Enum.dedup(received)
  end

  test "abort_handle is observed by inference", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)
    handle = WhisperCpp.AbortHandle.new()

    # Pre-arm the flag before inference starts so the test does not race
    # the decode loop on fast CPUs. whisper.cpp polls the abort callback
    # before the first encoder step, so a pre-armed flag must cancel the
    # run before any segment is decoded: an inert abort callback (the
    # whisper-rs 0.16.0 trampoline type confusion) returns the full
    # transcription here instead.
    WhisperCpp.AbortHandle.abort(handle)

    assert {:ok, %WhisperCpp.Transcription{text: "", segments: []}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
               language: "en",
               n_threads: 4,
               abort_handle: handle
             )

    assert WhisperCpp.AbortHandle.aborted?(handle)
  end

  if match?({:unix, :linux}, :os.type()) do
    test "progress sender threads exit after each call", %{model_path: model, pcm: pcm} do
      {:ok, model_ref} = WhisperCpp.load_model(model)
      parent = self()

      transcribe = fn ->
        {:ok, _} =
          WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
            language: "en",
            n_threads: 4,
            progress_pid: parent
          )
      end

      # Warm-up so lazily started BEAM/ggml thread pools do not count
      # against the baseline.
      transcribe.()
      Process.sleep(200)
      baseline = os_thread_count()

      for _ <- 1..5, do: transcribe.()
      Process.sleep(200)

      growth = os_thread_count() - baseline
      assert growth < 5, "leaked #{growth} OS threads across 5 progress_pid transcriptions"

      collect_progress([])
    end
  end

  test "language is validated against whisper.cpp's table", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    # BCP 47 tags and unknown codes are rejected instead of silently
    # corrupting the decoder prompt.
    assert {:error, %WhisperCpp.Error{reason: :invalid_request} = error} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "de-CH")

    assert error.message =~ "unknown language"

    # Embedded NUL bytes stay on the :invalid_request path; unchecked
    # they panic inside whisper-rs's CString conversion (:nif_panic).
    assert {:error, %WhisperCpp.Error{reason: :invalid_request}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "en" <> <<0>>)

    assert {:error, %WhisperCpp.Error{reason: :invalid_request}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
               language: "en",
               initial_prompt: "context" <> <<0>>
             )

    # tiny.en is English-only: other languages are rejected, not ignored.
    assert {:error, %WhisperCpp.Error{reason: :invalid_request} = error} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "de")

    assert error.message =~ "English-only"

    # nil and "auto" resolve to "en" on English-only models.
    assert {:ok, %WhisperCpp.Transcription{language: "en", segments: [_ | _]}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, n_threads: 4)

    assert {:ok, %WhisperCpp.Transcription{language: "en"}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "auto", n_threads: 4)
  end

  describe "built-in VAD" do
    setup do
      {:ok, vad_path: Fixtures.ensure_vad_model!()}
    end

    test "gates transcription on detected speech", %{model_path: model, pcm: pcm, vad_path: vad} do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      assert {:ok, %WhisperCpp.Transcription{text: text, segments: [_ | _]}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad
               )

      assert text =~ "country"
    end

    test "pure silence returns an empty transcription", %{model_path: model, vad_path: vad} do
      {:ok, model_ref} = WhisperCpp.load_model(model)
      silence = <<0::size(48_000 * 4)-unit(8)>>

      assert {:ok, %WhisperCpp.Transcription{text: "", segments: []}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, silence},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad,
                 vad_threshold: 0.9
               )
    end

    test "word timestamps stay within segment bounds", %{model_path: model, pcm: pcm, vad_path: vad} do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      assert {:ok, %WhisperCpp.Transcription{segments: [_ | _] = segs}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad,
                 word_timestamps: true
               )

      for seg <- segs, words = seg.words || [], word <- words do
        assert word.start >= seg.start - 0.2
        assert word.end <= seg.end + 0.2
      end
    end

    test "transcribe_slice keeps absolute timestamps", %{model_path: model, pcm: pcm, vad_path: vad} do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      assert {:ok, %WhisperCpp.Transcription{segments: [first | _]}} =
               WhisperCpp.transcribe_slice(model_ref, pcm, {2.0, 8.0},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad
               )

      assert first.start >= 2.0
      assert first.end <= 8.5
    end

    test "tuning options reach the detector", %{model_path: model, pcm: pcm, vad_path: vad} do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      assert {:ok, %WhisperCpp.Transcription{segments: [_ | _], text: text}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad,
                 vad_threshold: 0.3,
                 vad_min_speech_ms: 100,
                 vad_min_silence_ms: 50,
                 vad_speech_pad_ms: 60
               )

      assert text =~ "country"
    end

    test "offset and duration window the original timeline", %{
      model_path: model,
      pcm: pcm,
      vad_path: vad
    } do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      # The JFK clip is ~11 s. Windowing to 2..8 s must yield absolute
      # timestamps inside that window even though VAD compresses the
      # audio whisper actually sees.
      assert {:ok, %WhisperCpp.Transcription{segments: [first | _] = segs}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad,
                 offset_ms: 2_000,
                 duration_ms: 6_000
               )

      assert first.start >= 2.0
      assert List.last(segs).end <= 8.5
    end

    test "multi-span audio is stitched and remapped to the original timeline", %{
      model_path: model,
      pcm: pcm,
      vad_path: vad
    } do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      # Two speech spans separated by 2 s of silence: VAD must produce
      # at least two spans, and the second span's words must land after
      # the silence gap on the original timeline (jfk is ~11 s).
      silence = <<0::size(16_000 * 2 * 4)-unit(8)>>
      doubled = pcm <> silence <> pcm

      assert {:ok, %WhisperCpp.Transcription{segments: [_ | _] = segs, text: text}} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, doubled},
                 language: "en",
                 n_threads: 4,
                 vad_model_path: vad
               )

      assert text =~ "country"
      assert List.last(segs).end > 13.0
      assert List.last(segs).end < 25.0
    end

    test "a missing VAD model file is rejected", %{model_path: model, pcm: pcm} do
      {:ok, model_ref} = WhisperCpp.load_model(model)

      assert {:error, %WhisperCpp.Error{reason: :invalid_request} = error} =
               WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, vad_model_path: "/nonexistent/vad.bin")

      assert error.message =~ "vad_model_path"
    end
  end

  test "beam search decodes correctly", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    assert {:ok, %WhisperCpp.Transcription{text: text}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm},
               language: "en",
               n_threads: 4,
               beam_size: 2
             )

    assert text =~ "country"
  end

  test "translate is rejected on English-only models", %{model_path: model, pcm: pcm} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    assert {:error, %WhisperCpp.Error{reason: :invalid_request} = error} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, translate: true)

    assert error.message =~ "English-only"
  end

  test "non-finite PCM samples are rejected", %{model_path: model} do
    {:ok, model_ref} = WhisperCpp.load_model(model)
    pcm = <<0.5::little-float-32, 0x7FC0_0000::little-32, 0.5::little-float-32>>

    assert {:error, %WhisperCpp.Error{reason: :invalid_request} = error} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, pcm}, language: "en")

    assert error.message =~ "non-finite"
  end

  test "sub-window audio returns a clean result instead of aborting the node", %{model_path: model} do
    {:ok, model_ref} = WhisperCpp.load_model(model)

    # 0.1 s of silence: shorter than whisper.cpp's 30 s window. whisper.cpp
    # pads short inputs, so this must come back as {:ok, _}, never a
    # SIGABRT inside ggml that would take the BEAM node down.
    short_pcm = <<0::size(1_600 * 4)-unit(8)>>

    assert {:ok, %WhisperCpp.Transcription{}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, short_pcm}, language: "en", n_threads: 4)

    assert {:error, %WhisperCpp.Error{reason: :invalid_request}} =
             WhisperCpp.transcribe(model_ref, {:pcm_f32, ""}, language: "en")
  end

  defp collect_progress(acc) do
    receive do
      {:whisper_progress, pct} -> collect_progress([pct | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  defp os_thread_count do
    [_, count] = Regex.run(~r/^Threads:\s*(\d+)$/m, File.read!("/proc/self/status"))
    String.to_integer(count)
  end
end
