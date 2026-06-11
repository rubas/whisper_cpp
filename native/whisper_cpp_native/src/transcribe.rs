//! Run-one-transcription glue between the NIF entry point and
//! `whisper-rs`. Owns the decoding strategy, parameter setting, and
//! segment/word collection.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use crate::errors::{ErrorContext as _, inference_error};
use rustler::{Encoder, LocalPid, OwnedEnv};
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

/// Per-call decoding request decoded from the Elixir-side `TranscribeOpts`.
pub(crate) struct TranscribeRequest {
    pub(crate) language: Option<String>,
    pub(crate) translate: bool,
    pub(crate) initial_prompt: Option<String>,
    pub(crate) word_timestamps: bool,
    pub(crate) beam_size: Option<u32>,
    pub(crate) best_of: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) n_threads: Option<u32>,
    pub(crate) n_max_text_ctx: Option<u32>,
    pub(crate) offset_ms: Option<u32>,
    pub(crate) duration_ms: Option<u32>,
    pub(crate) no_speech_thold: Option<f32>,
    pub(crate) logprob_thold: Option<f32>,
    pub(crate) suppress_blank: Option<bool>,
    pub(crate) suppress_non_speech_tokens: Option<bool>,
    pub(crate) single_segment: Option<bool>,
    pub(crate) print_progress: bool,
}

pub(crate) struct WordResult {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) probability: f32,
}

pub(crate) struct SegmentResult {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) no_speech_prob: f32,
    pub(crate) avg_logprob: f32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) words: Option<Vec<WordResult>>,
}

pub(crate) struct TranscriptionResult {
    pub(crate) language: String,
    pub(crate) duration_s: f32,
    pub(crate) segments: Vec<SegmentResult>,
}

/// Saturating cast for `u32` count-like values handed to whisper-rs
/// APIs that use `i32`. Realistic values (thread counts, beam sizes,
/// millisecond offsets) never overflow.
#[inline]
fn u32_to_i32(n: u32) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Build a `FullParams` from the request. Sampling strategy is beam-search
/// when `beam_size > 1`, otherwise greedy.
fn build_params(req: &TranscribeRequest) -> FullParams<'_, '_> {
    let strategy = match req.beam_size {
        Some(n) if n > 1 => SamplingStrategy::BeamSearch {
            beam_size: u32_to_i32(n),
            patience: -1.0,
        },
        _ => SamplingStrategy::Greedy {
            best_of: u32_to_i32(req.best_of.unwrap_or(1)),
        },
    };

    let mut params = FullParams::new(strategy);

    if let Some(t) = req.n_threads {
        params.set_n_threads(u32_to_i32(t));
    }
    if let Some(c) = req.n_max_text_ctx {
        params.set_n_max_text_ctx(u32_to_i32(c));
    }
    if let Some(o) = req.offset_ms {
        params.set_offset_ms(u32_to_i32(o));
    }
    if let Some(d) = req.duration_ms {
        params.set_duration_ms(u32_to_i32(d));
    }
    if let Some(t) = req.temperature {
        params.set_temperature(t);
    }
    if let Some(v) = req.no_speech_thold {
        params.set_no_speech_thold(v);
    }
    if let Some(v) = req.logprob_thold {
        params.set_logprob_thold(v);
    }
    if let Some(b) = req.suppress_blank {
        params.set_suppress_blank(b);
    }
    if let Some(b) = req.suppress_non_speech_tokens {
        params.set_suppress_nst(b);
    }
    if let Some(b) = req.single_segment {
        params.set_single_segment(b);
    }
    if req.translate {
        params.set_translate(true);
    }
    if let Some(ref lang) = req.language {
        params.set_language(Some(lang.as_str()));
    }
    if let Some(ref prompt) = req.initial_prompt {
        params.set_initial_prompt(prompt);
    }

    params.set_token_timestamps(req.word_timestamps);
    params.set_print_progress(req.print_progress);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);

    params
}

/// Shutdown sentinel for the progress sender thread. whisper.cpp only
/// reports progress in `0..=100`, so this can never collide.
const PROGRESS_DONE: i32 = i32::MIN;

/// Wire optional cooperative-cancellation and progress callbacks onto
/// the `FullParams`. Both hooks are no-ops when the caller omits them.
///
/// Progress messages cannot be sent directly from the callback because
/// it fires on the dirty-CPU scheduler thread where
/// `OwnedEnv::send_and_clear` panics. A dedicated sender thread owns
/// the `OwnedEnv` and reads percentages off an `mpsc` channel; the
/// callback only forwards new values.
///
/// whisper-rs 0.16.0 leaks both installed closures (`Box::into_raw`
/// with no matching reclaim), so the `Sender` captured by the progress
/// callback never drops and cannot close the channel. The returned
/// `Sender` is the thread's shutdown path instead: the caller must send
/// [`PROGRESS_DONE`] once `full()` has returned. Until upstream frees
/// the closures, each call still leaks a few dozen heap bytes (and the
/// abort path pins one `Arc<AtomicBool>` clone).
fn install_callbacks(
    params: &mut FullParams<'_, '_>,
    abort_flag: Option<Arc<AtomicBool>>,
    progress_pid: Option<LocalPid>,
) -> Option<mpsc::Sender<i32>> {
    if let Some(flag) = abort_flag {
        // whisper-rs 0.16.0 instantiates its abort trampoline with the
        // caller's closure type `F` but stores the closure as
        // `Box<dyn FnMut() -> bool>`. Passing the boxed trait object
        // makes the two agree; a bare closure is reinterpreted memory
        // (out-of-bounds reads) and the flag is never consulted.
        let callback: Box<dyn FnMut() -> bool> = Box::new(move || flag.load(Ordering::SeqCst));
        params.set_abort_callback_safe(callback);
    }

    let pid = progress_pid?;
    let (tx, rx) = mpsc::channel::<i32>();
    std::thread::spawn(move || {
        while let Ok(pct) = rx.recv() {
            if pct == PROGRESS_DONE {
                break;
            }
            let mut owned = OwnedEnv::new();
            let _ = owned.send_and_clear(&pid, |env| {
                let tag =
                    rustler::Atom::from_str(env, "whisper_progress").expect("atom name is valid");
                (tag, pct).encode(env)
            });
        }
    });

    let callback_tx = tx.clone();
    let mut last: i32 = -1;
    params.set_progress_callback_safe(move |pct: i32| {
        if pct == last {
            return;
        }
        last = pct;
        // Receiver thread has exited if this errors; nothing to do.
        let _ = callback_tx.send(pct);
    });

    Some(tx)
}

/// Transcribe a single PCM buffer. The context mutex is held only long
/// enough to call `create_state()`; `WhisperState` then carries its own
/// `Arc<WhisperInnerContext>`, so parallel transcribes on one loaded
/// model do not serialise.
pub(crate) fn transcribe_one(
    model: &crate::WhisperResource,
    samples: &[f32],
    request: &TranscribeRequest,
    abort_flag: Option<Arc<AtomicBool>>,
    progress_pid: Option<LocalPid>,
) -> anyhow::Result<TranscriptionResult> {
    let token_eot = model.token_eot;
    let mut state: WhisperState = {
        let ctx_guard = model.ctx.lock();
        ctx_guard
            .as_ref()
            .expect("whisper context is only taken in Drop")
            .create_state()
            .inference_error_ctx("failed to create whisper state")?
    };

    let mut params = build_params(request);
    let abort_flag_check = abort_flag.clone();
    let progress_done = install_callbacks(&mut params, abort_flag, progress_pid);

    let full_result = state.full(params, samples);

    // whisper.cpp no longer polls the progress callback once `full()`
    // has returned; stop the sender thread.
    if let Some(tx) = progress_done {
        let _ = tx.send(PROGRESS_DONE);
    }

    if let Err(e) = full_result {
        let aborted = abort_flag_check.is_some_and(|f| f.load(Ordering::SeqCst));
        if !aborted {
            return Err(inference_error(format!("whisper.cpp full() failed: {e}")));
        }
        // Abort was requested by the caller: fall through and return the
        // segments produced before cancellation as a partial result.
    }

    let n_segments = usize::try_from(state.full_n_segments()).unwrap_or(0);
    let mut segments = Vec::with_capacity(n_segments);

    for seg in state.as_iter() {
        segments.push(extract_segment(&seg, request.word_timestamps, token_eot)?);
    }

    let language = {
        let id = state.full_lang_id_from_state();
        whisper_rs::get_lang_str(id)
            .map(str::to_owned)
            .or_else(|| request.language.clone())
            .unwrap_or_else(|| "en".to_owned())
    };

    #[allow(clippy::cast_precision_loss)]
    let duration_s = samples.len() as f32 / 16_000.0_f32;

    Ok(TranscriptionResult {
        language,
        duration_s,
        segments,
    })
}

fn extract_segment(
    seg: &whisper_rs::WhisperSegment<'_>,
    word_timestamps: bool,
    token_eot: u32,
) -> anyhow::Result<SegmentResult> {
    let text = seg
        .to_str_lossy()
        .map(std::borrow::Cow::into_owned)
        .inference_error_ctx("failed to read segment text")?;

    let start = cs_to_s(seg.start_timestamp());
    let end = cs_to_s(seg.end_timestamp());
    let no_speech_prob = seg.no_speech_probability();
    let n_tokens = seg.n_tokens();
    let token_cap = usize::try_from(n_tokens).unwrap_or(0);

    let mut tokens = Vec::with_capacity(token_cap);
    let mut total_logprob = 0.0_f32;
    let mut counted: u32 = 0;
    let mut word_tokens: Option<Vec<WordToken>> = if word_timestamps {
        Some(Vec::new())
    } else {
        None
    };

    for t in 0..n_tokens {
        let Some(tok) = seg.get_token(t) else {
            continue;
        };
        let data = tok.token_data();
        let id = data.id;

        // Keep only text tokens: `id < token_eot` is the text/non-text boundary.
        if let Ok(u) = u32::try_from(id)
            && u < token_eot
        {
            tokens.push(u);
        }

        total_logprob += data.plog;
        counted += 1;

        if let Some(ref mut buf) = word_tokens {
            buf.push(WordToken {
                bytes: tok.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(),
                t0: data.t0,
                t1: data.t1,
                p: data.p,
            });
        }
    }

    let words_acc = word_tokens.map(assemble_words);

    #[allow(clippy::cast_precision_loss)]
    let avg_logprob = if counted > 0 {
        total_logprob / counted as f32
    } else {
        0.0
    };

    Ok(SegmentResult {
        text,
        start,
        end,
        no_speech_prob,
        avg_logprob,
        tokens,
        words: words_acc,
    })
}

/// One decoded token as fed to word assembly: raw text bytes plus
/// whisper.cpp's token-level timing and probability.
struct WordToken {
    bytes: Vec<u8>,
    t0: i64,
    t1: i64,
    p: f32,
}

/// Group decoded tokens into words. Token bytes are accumulated raw and
/// converted to UTF-8 once per finished word: Whisper's BPE regularly
/// splits a multibyte character across two tokens, so converting each
/// token on its own corrupts it (e.g. "schön" becomes "sch\u{FFFD}\u{FFFD}n").
fn assemble_words(tokens: Vec<WordToken>) -> Vec<WordResult> {
    struct WordAcc {
        bytes: Vec<u8>,
        start: f32,
        end: f32,
        probability: f32,
    }

    impl WordAcc {
        fn finish(self) -> WordResult {
            WordResult {
                text: String::from_utf8_lossy(&self.bytes).trim_start().to_owned(),
                start: self.start,
                end: self.end,
                probability: self.probability,
            }
        }
    }

    let mut words = Vec::new();
    let mut current: Option<WordAcc> = None;

    for tok in tokens {
        // Skip whisper.cpp special tokens for the word stream - they
        // carry no acoustic word content.
        if tok.bytes.starts_with(b"[_") || tok.bytes.starts_with(b"<|") {
            continue;
        }

        // A leading space byte cannot be a fragment of a split
        // codepoint: UTF-8 continuation bytes are always >= 0x80.
        let starts_new_word = tok.bytes.first() == Some(&b' ') || current.is_none();

        if starts_new_word {
            if let Some(acc) = current.take() {
                words.push(acc.finish());
            }
            current = Some(WordAcc {
                start: cs_to_s(tok.t0),
                end: cs_to_s(tok.t1),
                probability: tok.p,
                bytes: tok.bytes,
            });
        } else if let Some(ref mut acc) = current {
            acc.bytes.extend_from_slice(&tok.bytes);
            acc.end = cs_to_s(tok.t1);
            // Worst-token probability, matching faster-whisper.
            acc.probability = acc.probability.min(tok.p);
        }
    }

    if let Some(acc) = current.take() {
        words.push(acc.finish());
    }

    words
}

/// whisper.cpp reports timestamps in centiseconds (10 ms units).
#[inline]
fn cs_to_s(cs: i64) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let cs_f = cs as f32;
    cs_f / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_token(bytes: &[u8], t0: i64, t1: i64, p: f32) -> WordToken {
        WordToken {
            bytes: bytes.to_vec(),
            t0,
            t1,
            p,
        }
    }

    #[test]
    fn assemble_words_reassembles_codepoints_split_across_tokens() {
        // "schön" with the "ö" (0xC3 0xB6) split across two BPE tokens.
        let words = assemble_words(vec![
            word_token(b" sch\xC3", 0, 10, 0.9),
            word_token(b"\xB6n", 10, 20, 0.8),
        ]);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "schön");
        assert!((words[0].start - 0.0).abs() < f32::EPSILON);
        assert!((words[0].end - 0.2).abs() < f32::EPSILON);
        assert!((words[0].probability - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn assemble_words_splits_on_leading_space() {
        let words = assemble_words(vec![
            word_token(b" ask", 0, 10, 0.9),
            word_token(b" not", 10, 20, 0.7),
        ]);

        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["ask", "not"]);
    }

    #[test]
    fn assemble_words_skips_special_tokens() {
        let words = assemble_words(vec![
            word_token(b"<|endoftext|>", 0, 0, 1.0),
            word_token(b"[_TT_50]", 0, 0, 1.0),
            word_token(b" hi", 0, 10, 0.9),
        ]);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hi");
    }
}
