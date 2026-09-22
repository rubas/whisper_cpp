defmodule WhisperCpp.Native.BuildEnv do
  @moduledoc false
  # The environment variables that select the NIF artefact at compile
  # time. Mix does not track `System.get_env/1`, so `__mix_recompile__?/0`
  # compares the values against this snapshot. `WhisperCpp.Native` reads
  # them through `get/1` and so recompiles with this module. The hook
  # lives here and not on `WhisperCpp.Native`: Mix loads a module before
  # it calls the hook, and a NIF that cannot load (a cuda artefact on a
  # host without CUDA) would hide the hook exactly when the user switches
  # back.

  @env Map.new(~w(WHISPER_CPP_FEATURES WHISPER_CPP_VARIANT WHISPER_CPP_BUILD), &{&1, System.get_env(&1)})

  @spec get(String.t()) :: String.t() | nil
  def get(name), do: Map.fetch!(@env, name)

  @spec __mix_recompile__?() :: boolean()
  def __mix_recompile__?, do: Enum.any?(@env, fn {name, value} -> System.get_env(name) != value end)
end
