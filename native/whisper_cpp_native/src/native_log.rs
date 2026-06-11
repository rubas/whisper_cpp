//! Filters whisper.cpp / GGML native logging.
//!
//! whisper.cpp prints every model-load and inference detail to stderr -
//! dozens of lines per loaded model. Keep warnings and errors, drop the
//! info/debug chatter; `WHISPER_CPP_NATIVE_LOG` accepts `none`, `error`,
//! `warn` (default), `info`, and `debug`.
#![allow(unsafe_code)] // registering a C log callback has no safe wrapper

use std::cell::Cell;
use std::ffi::CStr;
use std::io::Write as _;
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

use whisper_rs_sys::{
    ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_CONT as LEVEL_CONT,
    ggml_log_level_GGML_LOG_LEVEL_DEBUG as LEVEL_DEBUG,
    ggml_log_level_GGML_LOG_LEVEL_ERROR as LEVEL_ERROR,
    ggml_log_level_GGML_LOG_LEVEL_INFO as LEVEL_INFO,
    ggml_log_level_GGML_LOG_LEVEL_NONE as LEVEL_NONE,
    ggml_log_level_GGML_LOG_LEVEL_WARN as LEVEL_WARN,
};

/// Above [`LEVEL_ERROR`]; nothing reaches it, so everything is dropped.
const SILENCE: u32 = LEVEL_ERROR + 1;

static THRESHOLD: AtomicU32 = AtomicU32::new(LEVEL_WARN);

thread_local! {
    /// Whether this thread's previous chunk was emitted - `CONT` chunks
    /// continue the preceding message and follow its verdict. Native
    /// log calls are synchronous, so a message and its continuations
    /// arrive on one thread.
    static LAST_KEPT: Cell<bool> = const { Cell::new(false) };
}

/// Decide whether a chunk at `level` passes the `threshold`, given the
/// verdict of the thread's previous chunk.
fn keep(level: ggml_log_level, threshold: u32, last_kept: bool) -> bool {
    if level == LEVEL_CONT {
        return last_kept;
    }
    // NONE marks bare prints in the GGML convention: always pass them
    // unless logging is silenced outright.
    if level == LEVEL_NONE {
        return threshold <= LEVEL_ERROR;
    }
    level >= threshold
}

/// # Safety
/// Called by whisper.cpp / GGML from arbitrary native threads; must not
/// panic or unwind. Writes to stderr and ignores failures.
unsafe extern "C" fn filter(level: ggml_log_level, text: *const c_char, _user_data: *mut c_void) {
    let verdict = LAST_KEPT.with(|last| {
        let v = keep(level, THRESHOLD.load(Ordering::Relaxed), last.get());
        last.set(v);
        v
    });

    if verdict && !text.is_null() {
        let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
        let _ = std::io::stderr().write_all(bytes);
    }
}

/// Threshold for a `WHISPER_CPP_NATIVE_LOG` value; `None` for unknown
/// values, which fall back to the default with a warning.
fn threshold_for(value: &str) -> Option<u32> {
    match value {
        "none" => Some(SILENCE),
        "error" => Some(LEVEL_ERROR),
        "warn" => Some(LEVEL_WARN),
        "info" => Some(LEVEL_INFO),
        "debug" => Some(LEVEL_DEBUG),
        _ => None,
    }
}

/// Install the filter for both whisper.cpp and GGML logs. Called once
/// from the NIF `load` hook, before any model can be loaded.
pub(crate) fn install() {
    let threshold = match std::env::var("WHISPER_CPP_NATIVE_LOG") {
        Ok(value) => threshold_for(&value).unwrap_or_else(|| {
            eprintln!(
                "whisper_cpp: unknown WHISPER_CPP_NATIVE_LOG value {value:?} \
                 (expected none|error|warn|info|debug); using \"warn\""
            );
            LEVEL_WARN
        }),
        Err(_) => LEVEL_WARN,
    };
    THRESHOLD.store(threshold, Ordering::Relaxed);

    unsafe {
        whisper_rs_sys::whisper_log_set(Some(filter), std::ptr::null_mut());
        whisper_rs_sys::ggml_log_set(Some(filter), std::ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_for_accepts_each_documented_value() {
        assert_eq!(threshold_for("none"), Some(SILENCE));
        assert_eq!(threshold_for("error"), Some(LEVEL_ERROR));
        assert_eq!(threshold_for("warn"), Some(LEVEL_WARN));
        assert_eq!(threshold_for("info"), Some(LEVEL_INFO));
        assert_eq!(threshold_for("debug"), Some(LEVEL_DEBUG));
        assert_eq!(threshold_for("verbose"), None);
        assert_eq!(threshold_for(""), None);
    }

    #[test]
    fn keep_filters_by_level_and_follows_cont_verdicts() {
        // default threshold: warnings and errors pass, chatter drops
        assert!(keep(LEVEL_WARN, LEVEL_WARN, false));
        assert!(keep(LEVEL_ERROR, LEVEL_WARN, false));
        assert!(!keep(LEVEL_INFO, LEVEL_WARN, false));
        assert!(!keep(LEVEL_DEBUG, LEVEL_WARN, false));

        // continuations inherit the previous chunk's verdict
        assert!(keep(LEVEL_CONT, LEVEL_WARN, true));
        assert!(!keep(LEVEL_CONT, LEVEL_WARN, false));

        // bare prints (NONE) pass unless silenced
        assert!(keep(LEVEL_NONE, LEVEL_WARN, false));
        assert!(keep(LEVEL_NONE, LEVEL_ERROR, false));
        assert!(!keep(LEVEL_NONE, SILENCE, false));

        // "none" drops even errors
        assert!(!keep(LEVEL_ERROR, SILENCE, false));
    }
}
