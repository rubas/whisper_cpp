//! Tags `anyhow::Error` values with an error kind so the NIF entry point
//! can map back to a structured Elixir reason atom.

const KIND_INFERENCE_ERROR: &str = "inference_error";
const KIND_INVALID_REQUEST: &str = "invalid_request";
const KIND_LOAD_ERROR: &str = "load_error";

/// Root error value that carries the kind next to the message. The kind
/// stays out of `Display`, so the message that reaches Elixir is only
/// the caller-facing text.
#[derive(Debug)]
struct Tagged {
    kind: &'static str,
    message: String,
}

impl std::fmt::Display for Tagged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Tagged {}

fn tagged(kind: &'static str, err: impl std::fmt::Display) -> anyhow::Error {
    anyhow::Error::new(Tagged {
        kind,
        message: err.to_string(),
    })
}

pub(crate) fn inference_error(err: impl std::fmt::Display) -> anyhow::Error {
    tagged(KIND_INFERENCE_ERROR, err)
}

pub(crate) fn invalid_request(err: impl std::fmt::Display) -> anyhow::Error {
    tagged(KIND_INVALID_REQUEST, err)
}

pub(crate) fn load_error(err: impl std::fmt::Display) -> anyhow::Error {
    tagged(KIND_LOAD_ERROR, err)
}

pub(crate) fn kind_of(err: &anyhow::Error) -> Option<&'static str> {
    err.downcast_ref::<Tagged>().map(|t| t.kind)
}

pub(crate) trait ErrorContext<T> {
    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn inference_error_ctx(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| inference_error(format!("{msg}: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_of_finds_the_tag() {
        assert_eq!(kind_of(&inference_error("boom")), Some("inference_error"));
        assert_eq!(kind_of(&invalid_request("bad")), Some("invalid_request"));
    }

    #[test]
    fn kind_of_is_none_for_untagged_errors() {
        assert_eq!(kind_of(&anyhow::anyhow!("plain")), None);
    }
}
