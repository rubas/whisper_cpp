//! Attaches an error-kind tag to `anyhow::Error` chains so the NIF
//! entry point can map back to a structured Elixir reason atom.

use anyhow::Context as _;

const KIND_INFERENCE_ERROR: &str = "inference_error";
const KIND_INVALID_REQUEST: &str = "invalid_request";

#[derive(Debug, Clone, Copy)]
struct Kind(&'static str);

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kind={}", self.0)
    }
}

impl std::error::Error for Kind {}

pub(crate) fn inference_error<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_INFERENCE_ERROR))
}

pub(crate) fn invalid_request<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_INVALID_REQUEST))
}

// `anyhow::Error::downcast_ref` reaches context values; iterating
// `err.chain()` does not - contexts sit inside an opaque `ContextError`
// wrapper whose chain elements never downcast to `Kind`.
pub(crate) fn kind_from_chain(err: &anyhow::Error) -> Option<&'static str> {
    err.downcast_ref::<Kind>().map(|k| k.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_from_chain_finds_the_tag() {
        assert_eq!(
            kind_from_chain(&inference_error("boom")),
            Some("inference_error")
        );
        assert_eq!(
            kind_from_chain(&invalid_request("bad")),
            Some("invalid_request")
        );
    }

    #[test]
    fn kind_from_chain_is_none_for_untagged_errors() {
        assert_eq!(kind_from_chain(&anyhow::anyhow!("plain")), None);
    }
}
