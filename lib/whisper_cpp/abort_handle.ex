defmodule WhisperCpp.AbortHandle do
  @moduledoc """
  Cooperative cancellation handle for `WhisperCpp.transcribe/3`.

  Mint a handle, pass it via the `:abort_handle` option, then signal it
  from another process to ask in-flight inference to return early.
  whisper.cpp polls the abort flag between encoder/decoder steps; the
  returned `%Transcription{}` contains whatever segments had been
  produced before the abort took effect.

      handle = WhisperCpp.AbortHandle.new()

      task =
        Task.async(fn ->
          WhisperCpp.transcribe(model, {:pcm_f32, pcm}, abort_handle: handle)
        end)

      Process.send_after(self(), :timeout, 30_000)
      receive do
        :timeout -> WhisperCpp.AbortHandle.abort(handle)
      end

      Task.await(task)
  """

  alias WhisperCpp.Native

  @enforce_keys [:ref]
  defstruct [:ref]

  @type t :: %__MODULE__{ref: reference()}

  @doc "Mints a fresh, un-aborted handle."
  @spec new() :: t()
  def new, do: %__MODULE__{ref: Native.new_abort_handle()}

  @doc "Asks any in-flight transcribe call using this handle to stop."
  @spec abort(t()) :: :ok
  def abort(%__MODULE__{ref: ref}) do
    Native.abort_handle_signal(ref)
    :ok
  end

  @doc "Returns `true` once `abort/1` has been called for this handle."
  @spec aborted?(t()) :: boolean()
  def aborted?(%__MODULE__{ref: ref}), do: Native.abort_handle_aborted?(ref)
end
