defmodule WhisperCpp.ErrorTest do
  use ExUnit.Case, async: true

  alias WhisperCpp.Error

  describe "new/3" do
    test "builds a struct with default empty details" do
      assert %Error{reason: :load_error, message: "boom", details: %{}} =
               Error.new(:load_error, "boom")
    end

    test "preserves details" do
      assert %Error{details: %{path: "x"}} =
               Error.new(:invalid_request, "bad", %{path: "x"})
    end
  end

  describe "from_native/1" do
    test "maps known native types to atom reasons" do
      for {type, expected} <- [
            {"invalid_request", :invalid_request},
            {"load_error", :load_error},
            {"inference_error", :inference_error},
            {"runtime_error", :runtime_error},
            {"nif_panic", :nif_panic}
          ] do
        assert %Error{reason: ^expected} =
                 Error.from_native(%{type: type, message: "x", details: %{}})
      end
    end

    test "falls back to :native_error for unknown types" do
      assert %Error{reason: :native_error} =
               Error.from_native(%{type: "weird", message: "x", details: %{}})
    end

    test "handles non-map payloads" do
      assert %Error{reason: :native_error, details: %{raw: _}} =
               Error.from_native(:weird)
    end
  end

  describe "Exception protocol" do
    test "message/1 formats reason + message" do
      err = Error.new(:load_error, "no such file")
      assert Exception.message(err) == "load_error: no such file"
    end

    test "raise/1 works" do
      err = Error.new(:invalid_request, "bad")
      assert_raise WhisperCpp.Error, "invalid_request: bad", fn -> raise err end
    end
  end
end
