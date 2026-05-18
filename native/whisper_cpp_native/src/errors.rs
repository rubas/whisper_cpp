//! Lightweight error categorisation. Attaches an error-kind tag to
//! `anyhow::Error` chains so the NIF entry point can map back to a
//! structured Elixir reason atom.

use anyhow::Context as _;

const KIND_INVALID_REQUEST: &str = "invalid_request";
const KIND_LOAD_ERROR: &str = "load_error";
const KIND_INFERENCE_ERROR: &str = "inference_error";
const KIND_RUNTIME_ERROR: &str = "runtime_error";

#[derive(Debug, Clone, Copy)]
struct Kind(&'static str);

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kind={}", self.0)
    }
}

pub fn invalid_request<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_INVALID_REQUEST))
}

pub fn load_error<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_LOAD_ERROR))
}

pub fn inference_error<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_INFERENCE_ERROR))
}

pub fn runtime_error<E>(err: E) -> anyhow::Error
where
    E: std::fmt::Display + Send + Sync + 'static,
{
    anyhow::anyhow!("{err}").context(Kind(KIND_RUNTIME_ERROR))
}

pub fn kind_from_chain(err: &anyhow::Error) -> Option<&'static str> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<Kind>().map(|k| k.0))
}

pub trait ErrorContext<T> {
    fn invalid_request_ctx(self, msg: &'static str) -> anyhow::Result<T>;
    fn load_error_ctx(self, msg: &'static str) -> anyhow::Result<T>;
    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T>;
    fn runtime_error_ctx(self, msg: &'static str) -> anyhow::Result<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn invalid_request_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
            .context(Kind(KIND_INVALID_REQUEST))
    }

    fn load_error_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
            .context(Kind(KIND_LOAD_ERROR))
    }

    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
            .context(Kind(KIND_INFERENCE_ERROR))
    }

    fn runtime_error_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
            .context(Kind(KIND_RUNTIME_ERROR))
    }
}

pub fn require<T>(opt: Option<T>, msg: &'static str) -> anyhow::Result<T> {
    opt.ok_or_else(|| invalid_request(msg))
}
