//! Attaches an error-kind tag to `anyhow::Error` chains so the NIF
//! entry point can map back to a structured Elixir reason atom.

use anyhow::Context as _;

const KIND_INFERENCE_ERROR: &str = "inference_error";

#[derive(Debug, Clone, Copy)]
struct Kind(&'static str);

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kind={}", self.0)
    }
}

// Required so `anyhow::Error::chain().downcast_ref::<Kind>()` matches
// this tag - `downcast_ref` is bounded by `std::error::Error`.
impl std::error::Error for Kind {}

pub(crate) fn inference_error<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_INFERENCE_ERROR))
}

pub(crate) fn kind_from_chain(err: &anyhow::Error) -> Option<&'static str> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<Kind>().map(|k| k.0))
}

pub(crate) trait ErrorContext<T> {
    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
            .context(Kind(KIND_INFERENCE_ERROR))
    }
}
