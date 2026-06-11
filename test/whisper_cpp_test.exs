defmodule WhisperCppTest do
  @moduledoc """
  Tests for the public `WhisperCpp` API: argument validation paths,
  transcription / segment / word payload mapping, and the
  `transcribe_slice/4` range-validation contract. Covers everything that
  can be checked without a loaded NIF; end-to-end behaviour lives in
  `test/integration_test.exs`.
  """

  use ExUnit.Case, async: true

  alias WhisperCpp.Error

  doctest WhisperCpp

  describe "load_model/2 validation" do
    test "rejects non-string paths" do
      assert {:error, %Error{reason: :invalid_request}} = WhisperCpp.load_model(123)
    end

    test "rejects empty paths" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.load_model("   ")

      assert msg =~ "non-empty"
    end

    test "rejects unknown options" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.load_model("/tmp/whatever.bin", made_up: true)

      assert msg =~ "unknown option"
    end

    test "rejects invalid device" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.load_model("/tmp/whatever.bin", device: :tpu)
    end
  end

  describe "transcribe/3 validation" do
    test "rejects non-model first argument" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe(%{not: :a_model}, "/tmp/audio.wav")
    end

    test "rejects audio with wrong byte alignment" do
      fake_model = %WhisperCpp.Model{
        ref: make_ref(),
        path: "fake",
        sampling_rate: 16_000,
        multilingual: false,
        n_vocab: 51_864,
        device: :cpu
      }

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(fake_model, {:pcm_f32, <<1, 2, 3>>})

      assert msg =~ "multiple of 4"
    end

    test "rejects empty PCM" do
      fake_model = %WhisperCpp.Model{
        ref: make_ref(),
        path: "fake",
        sampling_rate: 16_000,
        multilingual: false,
        n_vocab: 51_864,
        device: :cpu
      }

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(fake_model, {:pcm_f32, <<>>})

      assert msg =~ "empty"
    end

    test "rejects unknown options" do
      fake_model = %WhisperCpp.Model{
        ref: make_ref(),
        path: "fake",
        sampling_rate: 16_000,
        multilingual: false,
        n_vocab: 51_864,
        device: :cpu
      }

      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe(fake_model, {:pcm_f32, <<0, 0, 0, 0>>}, made_up: true)
    end

    test "rejects bare binary input (typo'd path protection)" do
      fake_model = %WhisperCpp.Model{
        ref: make_ref(),
        path: "fake",
        sampling_rate: 16_000,
        multilingual: false,
        n_vocab: 51_864,
        device: :cpu
      }

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(fake_model, <<1, 2, 3, 4>>)

      assert msg =~ "{:pcm_f32, binary}"
    end

    test "rejects a string path (PCM-only contract)" do
      fake_model = %WhisperCpp.Model{
        ref: make_ref(),
        path: "fake",
        sampling_rate: 16_000,
        multilingual: false,
        n_vocab: 51_864,
        device: :cpu
      }

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(fake_model, "/some/audio.mp3")

      assert msg =~ "{:pcm_f32, binary}"
      assert msg =~ "decode files upstream"
    end
  end

  describe "transcribe/3 VAD option validation" do
    setup do
      {:ok,
       model: %WhisperCpp.Model{
         ref: make_ref(),
         path: "fake",
         sampling_rate: 16_000,
         multilingual: false,
         n_vocab: 51_864,
         device: :cpu
       },
       pcm: {:pcm_f32, <<0.0::little-float-32>>}}
    end

    test "rejects out-of-range vad_threshold", %{model: model, pcm: pcm} do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(model, pcm, vad_model_path: "m.bin", vad_threshold: 1.5)

      assert msg =~ "vad_threshold"
    end

    test "rejects non-positive vad_min_speech_ms", %{model: model, pcm: pcm} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe(model, pcm, vad_model_path: "m.bin", vad_min_speech_ms: 0)
    end

    test "rejects non-string vad_model_path", %{model: model, pcm: pcm} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe(model, pcm, vad_model_path: 123)
    end

    test "rejects vad tuning without a vad model", %{model: model, pcm: pcm} do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe(model, pcm, vad_threshold: 0.7)

      assert msg =~ "no effect without :vad_model_path"
    end

    test "rejects duration_ms of zero", %{model: model, pcm: pcm} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe(model, pcm, duration_ms: 0)
    end
  end

  describe "transcribe_slice/4" do
    setup do
      {:ok,
       model: %WhisperCpp.Model{
         ref: make_ref(),
         path: "fake",
         sampling_rate: 16_000,
         multilingual: false,
         n_vocab: 51_864,
         device: :cpu
       },
       samples: :binary.copy(<<0::little-float-32>>, 16_000 * 5)}
    end

    test "returns empty transcription for slices under 0.3 s", %{model: m, samples: s} do
      assert {:ok, %WhisperCpp.Transcription{text: "", segments: [], duration_s: dur}} =
               WhisperCpp.transcribe_slice(m, s, {0.0, 0.1})

      assert_in_delta dur, 0.1, 1.0e-6
    end

    test "rejects negative start", %{model: m, samples: s} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe_slice(m, s, {-1.0, 2.0})
    end

    test "rejects inverted range (end before start)", %{model: m, samples: s} do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe_slice(m, s, {0.5, 0.4})

      assert msg =~ "greater than start"
    end

    test "rejects zero-length range", %{model: m, samples: s} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe_slice(m, s, {1.0, 1.0})
    end

    test "rejects fully-negative range", %{model: m, samples: s} do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCpp.transcribe_slice(m, s, {-1.0, -0.9})
    end

    test "propagates slice-bounds errors from Pcm.slice", %{model: m, samples: s} do
      # samples = 5 seconds; ask for [4.5, 7.0) which extends past the
      # buffer. Verifies the Pcm.slice -> transcribe_slice handoff.
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCpp.transcribe_slice(m, s, {4.5, 7.0})

      assert msg =~ "past the end"
    end
  end

  describe "build_transcription/2" do
    test "concatenates segments and trims" do
      payload = %{
        language: "en",
        duration_s: 1.5,
        segments: [
          %{
            text: "Hello",
            start: 0.0,
            end: 0.5,
            no_speech_prob: 0.1,
            avg_logprob: -0.3,
            tokens: [1, 2],
            words: nil
          },
          %{
            text: "world",
            start: 0.6,
            end: 1.2,
            no_speech_prob: 0.1,
            avg_logprob: -0.4,
            tokens: [3, 4],
            words: nil
          }
        ]
      }

      transcription = WhisperCpp.build_transcription(payload, 0.0)
      assert transcription.text == "Hello world"
      assert transcription.language == "en"
      assert length(transcription.segments) == 2
    end

    test "shifts timestamps by offset" do
      payload = %{
        language: "en",
        duration_s: 1.0,
        segments: [
          %{
            text: "hi",
            start: 0.1,
            end: 0.5,
            no_speech_prob: 0.0,
            avg_logprob: -0.2,
            tokens: [1],
            words: [%{text: "hi", start: 0.1, end: 0.5, probability: 0.9}]
          }
        ]
      }

      transcription = WhisperCpp.build_transcription(payload, 5.0)
      [seg] = transcription.segments
      assert_in_delta seg.start, 5.1, 1.0e-6
      assert_in_delta seg.end, 5.5, 1.0e-6
      [w] = seg.words
      assert_in_delta w.start, 5.1, 1.0e-6
    end
  end
end
