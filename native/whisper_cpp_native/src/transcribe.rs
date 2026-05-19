//! Run-one-transcription glue between the NIF entry point and
//! `whisper-rs`. Owns the decoding strategy, parameter setting, and
//! segment/word collection.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use crate::errors::{ErrorContext as _, inference_error};
use parking_lot::Mutex;
use rustler::{Encoder, LocalPid, OwnedEnv};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

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

/// Wire optional cooperative-cancellation and progress callbacks onto
/// the `FullParams`. Both hooks are no-ops when the caller omits them.
///
/// Progress messages cannot be sent directly from the callback because
/// it fires on the dirty-CPU scheduler thread where
/// `OwnedEnv::send_and_clear` panics. A dedicated sender thread owns
/// the `OwnedEnv` and reads percentages off an `mpsc` channel; the
/// callback only forwards new values. When `FullParams` drops, the
/// `Sender` drops, the channel closes, and the thread exits.
fn install_callbacks(
    params: &mut FullParams<'_, '_>,
    abort_flag: Option<Arc<AtomicBool>>,
    progress_pid: Option<LocalPid>,
) {
    if let Some(flag) = abort_flag {
        params.set_abort_callback_safe(move || flag.load(Ordering::SeqCst));
    }
    if let Some(pid) = progress_pid {
        let (tx, rx) = mpsc::channel::<i32>();
        std::thread::spawn(move || {
            while let Ok(pct) = rx.recv() {
                let mut owned = OwnedEnv::new();
                let _ = owned.send_and_clear(&pid, |env| {
                    let tag = rustler::Atom::from_str(env, "whisper_progress")
                        .expect("atom name is valid");
                    (tag, pct).encode(env)
                });
            }
        });

        let mut last: i32 = -1;
        params.set_progress_callback_safe(move |pct: i32| {
            if pct == last {
                return;
            }
            last = pct;
            // Receiver thread has exited if this errors; nothing to do.
            let _ = tx.send(pct);
        });
    }
}

/// Transcribe a single PCM buffer. The context mutex is held only long
/// enough to call `create_state()`; `WhisperState` then carries its own
/// `Arc<WhisperInnerContext>`, so parallel transcribes on one loaded
/// model do not serialise.
pub(crate) fn transcribe_one(
    ctx: &Mutex<WhisperContext>,
    samples: &[f32],
    request: &TranscribeRequest,
    abort_flag: Option<Arc<AtomicBool>>,
    progress_pid: Option<LocalPid>,
) -> anyhow::Result<TranscriptionResult> {
    let mut state: WhisperState = {
        let ctx_guard = ctx.lock();
        ctx_guard
            .create_state()
            .inference_error_ctx("failed to create whisper state")?
    };

    let mut params = build_params(request);
    install_callbacks(&mut params, abort_flag, progress_pid);

    state
        .full(params, samples)
        .map_err(|e| inference_error(format!("whisper.cpp full() failed: {e}")))?;

    let n_segments = usize::try_from(state.full_n_segments()).unwrap_or(0);
    let mut segments = Vec::with_capacity(n_segments);

    for seg in state.as_iter() {
        segments.push(extract_segment(&seg, request.word_timestamps)?);
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
) -> anyhow::Result<SegmentResult> {
    let text = seg
        .to_str_lossy()
        .map(std::borrow::Cow::into_owned)
        .inference_error_ctx("failed to read segment text")?;

    // whisper.cpp reports times in 10-ms units (`t0`, `t1`).
    let start = cs_to_s(seg.start_timestamp());
    let end = cs_to_s(seg.end_timestamp());
    let no_speech_prob = seg.no_speech_probability();
    let n_tokens = seg.n_tokens();
    let token_cap = usize::try_from(n_tokens).unwrap_or(0);

    let mut tokens = Vec::with_capacity(token_cap);
    let mut total_logprob = 0.0_f32;
    let mut counted: u32 = 0;
    let mut words_acc: Option<Vec<WordResult>> = if word_timestamps {
        Some(Vec::new())
    } else {
        None
    };
    let mut current_word: Option<WordResult> = None;

    for t in 0..n_tokens {
        let Some(tok) = seg.get_token(t) else {
            continue;
        };
        let data = tok.token_data();
        let id = data.id;

        // Filter timestamp / special tokens. Whisper text tokens occupy
        // [0, 50_257); everything above is reserved.
        if (0..50_257).contains(&id) {
            if let Ok(u) = u32::try_from(id) {
                tokens.push(u);
            }
        }

        total_logprob += data.plog;
        counted += 1;

        if let Some(ref mut buf) = words_acc {
            let tok_text = tok
                .to_str_lossy()
                .map(std::borrow::Cow::into_owned)
                .unwrap_or_default();

            // Skip whisper.cpp special tokens for the word stream - they
            // carry no acoustic word content.
            if tok_text.starts_with("[_") || tok_text.starts_with("<|") {
                continue;
            }

            let starts_new_word = tok_text.starts_with(' ') || current_word.is_none();

            if starts_new_word {
                if let Some(word) = current_word.take() {
                    buf.push(word);
                }
                current_word = Some(WordResult {
                    text: tok_text.trim_start().to_owned(),
                    start: cs_to_s(data.t0),
                    end: cs_to_s(data.t1),
                    probability: data.p,
                });
            } else if let Some(ref mut word) = current_word {
                word.text.push_str(&tok_text);
                word.end = cs_to_s(data.t1);
                // Worst-token probability, matching faster-whisper.
                word.probability = word.probability.min(data.p);
            }
        }
    }

    if let Some(ref mut buf) = words_acc
        && let Some(word) = current_word.take()
    {
        buf.push(word);
    }

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

/// whisper.cpp reports timestamps in centiseconds (10 ms units).
#[inline]
fn cs_to_s(cs: i64) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let cs_f = cs as f32;
    cs_f / 100.0
}
