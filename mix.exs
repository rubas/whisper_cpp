defmodule WhisperCpp.MixProject do
  use Mix.Project

  @version "0.1.4"
  @source_url "https://github.com/rubas/whisper_cpp"

  @spec project() :: keyword()
  def project do
    [
      app: :whisper_cpp,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      description: description(),
      package: package(),
      docs: docs(),
      deps: deps()
    ]
  end

  @spec application() :: keyword()
  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  @spec docs() :: keyword()
  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "usage-rules.md"],
      source_url: @source_url,
      source_ref: "v#{@version}",
      homepage_url: @source_url
    ]
  end

  @spec description() :: String.t()
  defp description do
    "Native Elixir bindings for whisper.cpp. Accepts 16 kHz mono f32 PCM " <>
      "and runs on CPU, CUDA, ROCm (hipBLAS), Metal, Vulkan, or CoreML."
  end

  @spec package() :: keyword()
  defp package do
    [
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "whisper.cpp" => "https://github.com/ggerganov/whisper.cpp",
        "whisper-rs" => "https://github.com/tazz4843/whisper-rs"
      },
      files: ~w(lib
           native/whisper_cpp_native/src
           native/whisper_cpp_native/Cargo.toml
           native/whisper_cpp_native/Cargo.lock
           checksum-*.exs
           mix.exs
           README.md
           CHANGELOG.md
           LICENSE*
           usage-rules.md)
    ]
  end

  @spec deps() :: [tuple()]
  defp deps do
    [
      {:rustler_precompiled, "~> 0.9.0"},
      {:rustler, "~> 0.37.3", optional: true},
      {:credo, "~> 1.7.18", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40.2", only: :dev, runtime: false}
    ]
  end
end
