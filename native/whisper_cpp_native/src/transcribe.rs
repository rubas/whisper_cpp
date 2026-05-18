//! Run-one-transcription glue between the NIF entry point and
//! `whisper-rs` 0.16. Owns the decoding strategy, parameter setting,
//! and segment/word collection.

use crate::errors::{inference_error, ErrorContext as _};
use parking_lot::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

/// Per-call decoding request decoded from the Elixir-side `TranscribeOpts`.
pub struct TranscribeRequest {
    pub language: Option<String>,
    pub translate: bool,
    pub initial_prompt: Option<String>,
    pub word_timestamps: bool,
    pub beam_size: Option<u32>,
    pub best_of: Option<u32>,
    pub temperature: Option<f32>,
    pub n_threads: Option<u32>,
    pub n_max_text_ctx: Option<u32>,
    pub offset_ms: Option<u32>,
    pub duration_ms: Option<u32>,
    pub no_speech_thold: Option<f32>,
    pub logprob_thold: Option<f32>,
    pub suppress_blank: Option<bool>,
    pub suppress_non_speech_tokens: Option<bool>,
    pub single_segment: Option<bool>,
    pub print_progress: bool,
}

pub struct WordResult {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

pub struct SegmentResult {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub no_speech_prob: f32,
    pub avg_logprob: f32,
    pub tokens: Vec<u32>,
    pub words: Option<Vec<WordResult>>,
}

pub struct TranscriptionResult {
    pub language: String,
    pub duration_s: f32,
    pub segments: Vec<SegmentResult>,
}

/// Build a `FullParams` from the request. Sampling strategy is beam-search
/// when `beam_size > 1`, otherwise greedy.
fn build_params<'a>(req: &'a TranscribeRequest) -> FullParams<'a, 'a> {
    let strategy = match req.beam_size {
        Some(n) if n > 1 => SamplingStrategy::BeamSearch {
            beam_size: n as i32,
            patience: -1.0,
        },
        _ => SamplingStrategy::Greedy {
            best_of: req.best_of.unwrap_or(1) as i32,
        },
    };

    let mut params = FullParams::new(strategy);

    if let Some(t) = req.n_threads {
        params.set_n_threads(t as i32);
    }
    if let Some(c) = req.n_max_text_ctx {
        params.set_n_max_text_ctx(c as i32);
    }
    if let Some(o) = req.offset_ms {
        params.set_offset_ms(o as i32);
    }
    if let Some(d) = req.duration_ms {
        params.set_duration_ms(d as i32);
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

/// Transcribe a single PCM buffer. The whisper.cpp engine is reentrant
/// for inference; we still wrap the context in a [`parking_lot::Mutex`]
/// to serialise state creation under a single resource handle.
pub fn transcribe_one(
    ctx: &Mutex<WhisperContext>,
    samples: &[f32],
    request: &TranscribeRequest,
) -> anyhow::Result<TranscriptionResult> {
    let ctx_guard = ctx.lock();
    let mut state: WhisperState = ctx_guard
        .create_state()
        .inference_error_ctx("failed to create whisper state")?;

    let params = build_params(request);

    state
        .full(params, samples)
        .map_err(|e| inference_error(format!("whisper.cpp full() failed: {e}")))?;

    let n_segments = state.full_n_segments();
    let mut segments = Vec::with_capacity(n_segments as usize);

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
        .map(|cow| cow.into_owned())
        .inference_error_ctx("failed to read segment text")?;

    // whisper.cpp reports times in 10-ms units (`t0`, `t1`).
    let start = cs_to_s(seg.start_timestamp());
    let end = cs_to_s(seg.end_timestamp());
    let no_speech_prob = seg.no_speech_probability();
    let n_tokens = seg.n_tokens();

    let mut tokens = Vec::with_capacity(n_tokens as usize);
    let mut total_logprob = 0.0_f32;
    let mut counted = 0_i32;
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

        // Filter timestamp / special tokens out of the public `tokens`
        // list. The Whisper multilingual vocab caps text tokens at
        // 50_257; everything above that is reserved (timestamps, lang
        // tokens, control). The monolingual `*.en` checkpoints share
        // the same range for normal text.
        if id >= 0 && id < 50_257 {
            tokens.push(id as u32);
        }

        total_logprob += data.plog;
        counted += 1;

        if let Some(ref mut buf) = words_acc {
            let tok_text = tok
                .to_str_lossy()
                .map(|cow| cow.into_owned())
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
                // Keep the worst-token probability as the word
                // probability, mirroring faster-whisper's per-word
                // confidence reduction.
                word.probability = word.probability.min(data.p);
            }
        }
    }

    if let Some(ref mut buf) = words_acc {
        if let Some(word) = current_word.take() {
            buf.push(word);
        }
    }

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

#[inline]
fn cs_to_s(cs: i64) -> f32 {
    cs as f32 / 100.0
}
