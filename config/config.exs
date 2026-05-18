import Config

# Force source build during this library's own CI / local dev. Downstream
# consumers do *not* see this file - `import_config` in their own
# `config/config.exs` is not transitive across deps - so they continue to
# fetch a precompiled artefact unless they set `WHISPER_CPP_BUILD=1` or the
# equivalent `:rustler_precompiled` config themselves.
config :rustler_precompiled, :force_build, whisper_cpp: System.get_env("CI") == "true"
