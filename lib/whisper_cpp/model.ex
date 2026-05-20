defmodule WhisperCpp.Model do
  @moduledoc """
  Loaded whisper.cpp model handle.

  Holds an opaque NIF reference plus cached metadata. The reference is
  garbage-collected by the BEAM when no longer reachable; whisper.cpp
  frees the model at that point.
  """

  @type device :: :cpu | :cuda | :hipblas | :vulkan | :metal | :coreml | :intel_sycl

  @type t :: %__MODULE__{
          ref: reference(),
          path: Path.t(),
          sampling_rate: pos_integer(),
          multilingual: boolean(),
          n_vocab: pos_integer(),
          device: device()
        }

  @enforce_keys [:ref, :path, :sampling_rate, :multilingual, :n_vocab, :device]
  defstruct [:ref, :path, :sampling_rate, :multilingual, :n_vocab, :device]
end
