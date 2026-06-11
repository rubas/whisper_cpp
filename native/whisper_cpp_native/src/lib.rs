//! Rustler NIF wrapping whisper.cpp via the `whisper-rs` crate.
//!
//! Every entry point returns `{:ok, value}` or
//! `{:error, %{type, message, details}}`; PCM input is little-endian
//! IEEE-754 `f32` mono at 16 kHz.

#![deny(unsafe_code)]

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use rustler::types::binary::Binary;
use rustler::{Encoder, Env, LocalPid, NifMap, ResourceArc, Term};
use whisper_rs::{WhisperContext, WhisperContextParameters};

mod errors;
mod native_log;
mod transcribe;
mod vad;

use errors::kind_from_chain;
use transcribe::{SegmentResult, TranscribeRequest, TranscriptionResult, WordResult};

#[allow(missing_docs)]
mod atoms {
    rustler::atoms! { ok, error }
}
use atoms::{error, ok};

/// `Some(label)` when this build was compiled with a GPU cargo feature.
/// At most one GPU backend is active per artefact.
const GPU_BACKEND: Option<&str> = if cfg!(feature = "cuda") {
    Some("cuda")
} else if cfg!(feature = "hipblas") {
    Some("hipblas")
} else if cfg!(feature = "vulkan") {
    Some("vulkan")
} else if cfg!(feature = "metal") {
    Some("metal")
} else if cfg!(feature = "coreml") {
    Some("coreml")
} else if cfg!(feature = "intel-sycl") {
    Some("intel_sycl")
} else {
    None
};

#[derive(Debug, NifMap)]
struct NativeError {
    r#type: String,
    message: String,
    details: HashMap<String, String>,
}

impl NativeError {
    fn new(type_name: &str, message: impl Into<String>) -> Self {
        Self {
            r#type: type_name.to_owned(),
            message: message.into(),
            details: HashMap::new(),
        }
    }

    fn with_detail(mut self, key: &str, value: impl Into<String>) -> Self {
        self.details.insert(key.to_owned(), value.into());
        self
    }
}

impl From<anyhow::Error> for NativeError {
    fn from(err: anyhow::Error) -> Self {
        let kind = kind_from_chain(&err).unwrap_or("inference_error");
        NativeError::new(kind, format!("{err:#}"))
    }
}

/// Opaque BEAM resource holding a loaded whisper.cpp context. The mutex
/// only wraps the brief `create_state()` step; inference itself runs
/// without it. See `transcribe::transcribe_one`.
///
/// `ctx` is `Some` from construction until `Drop`, which moves the
/// context out so it can be freed off the scheduler thread.
pub(crate) struct WhisperResource {
    pub(crate) ctx: Mutex<Option<WhisperContext>>,
    sampling_rate: usize,
    pub(crate) multilingual: bool,
    n_vocab: usize,
    /// `<|endoftext|>` token id - text tokens occupy `[0, token_eot)`,
    /// everything above is timestamp / language / control. Read from the
    /// loaded model at load time so the boundary stays correct across
    /// checkpoint variants (en-only vs multilingual vs future vocabs).
    pub(crate) token_eot: u32,
    device: &'static str,
}

impl rustler::Resource for WhisperResource {}

/// The BEAM runs resource destructors on the normal scheduler that
/// garbage-collects the owning process, with a ~1 ms budget. Freeing a
/// loaded whisper context releases gigabytes of tensors (and device
/// buffers on GPU builds), so hand it to a detached thread and let the
/// scheduler pay only for the spawn.
impl Drop for WhisperResource {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.get_mut().take() {
            // `thread::spawn` panics when the OS refuses a thread, and
            // a panic inside a resource destructor aborts the whole VM:
            // fall back to freeing inline (slow, but safe).
            if let Err(e) = std::thread::Builder::new()
                .name("whisper-ctx-reaper".to_owned())
                .spawn(move || drop(ctx))
            {
                eprintln!(
                    "whisper_cpp: freeing the model on the scheduler thread (spawn failed: {e})"
                );
            }
        }
    }
}

/// Cooperative cancellation flag. whisper.cpp polls the abort callback
/// between encoder/decoder steps and returns early when set.
pub(crate) struct AbortHandle {
    pub(crate) flag: Arc<AtomicBool>,
}

impl rustler::Resource for AbortHandle {}

#[derive(NifMap)]
struct LoadOpts {
    device: Option<String>,
}

#[derive(NifMap)]
struct TranscribeOpts {
    language: Option<String>,
    translate: Option<bool>,
    initial_prompt: Option<String>,
    word_timestamps: Option<bool>,
    beam_size: Option<u32>,
    best_of: Option<u32>,
    temperature: Option<f32>,
    n_threads: Option<u32>,
    n_max_text_ctx: Option<u32>,
    offset_ms: Option<u32>,
    duration_ms: Option<u32>,
    no_speech_thold: Option<f32>,
    logprob_thold: Option<f32>,
    suppress_blank: Option<bool>,
    suppress_non_speech_tokens: Option<bool>,
    single_segment: Option<bool>,
    print_progress: Option<bool>,
    vad_model_path: Option<String>,
    vad_threshold: Option<f32>,
    vad_min_speech_ms: Option<u32>,
    vad_min_silence_ms: Option<u32>,
    vad_speech_pad_ms: Option<u32>,
}

#[derive(NifMap)]
struct ModelInfo {
    sampling_rate: usize,
    multilingual: bool,
    n_vocab: usize,
    device: String,
}

#[derive(NifMap)]
struct AvailableDevices {
    backends: Vec<String>,
    gpu_supported: bool,
}

#[derive(NifMap)]
struct NifWord {
    text: String,
    start: f32,
    end: f32,
    probability: f32,
}

#[derive(NifMap)]
struct NifSegment {
    text: String,
    start: f32,
    end: f32,
    no_speech_prob: f32,
    avg_logprob: f32,
    tokens: Vec<u32>,
    words: Option<Vec<NifWord>>,
}

#[derive(NifMap)]
struct NifTranscription {
    language: String,
    duration_s: f32,
    segments: Vec<NifSegment>,
}

impl From<WordResult> for NifWord {
    fn from(w: WordResult) -> Self {
        Self {
            text: w.text,
            start: w.start,
            end: w.end,
            probability: w.probability,
        }
    }
}

impl From<SegmentResult> for NifSegment {
    fn from(s: SegmentResult) -> Self {
        Self {
            text: s.text,
            start: s.start,
            end: s.end,
            no_speech_prob: s.no_speech_prob,
            avg_logprob: s.avg_logprob,
            tokens: s.tokens,
            words: s
                .words
                .map(|ws| ws.into_iter().map(NifWord::from).collect()),
        }
    }
}

impl From<TranscriptionResult> for NifTranscription {
    fn from(t: TranscriptionResult) -> Self {
        Self {
            language: t.language,
            duration_s: t.duration_s,
            segments: t.segments.into_iter().map(NifSegment::from).collect(),
        }
    }
}

fn run_with_panic_protection<T, F>(f: F) -> Result<T, NativeError>
where
    F: FnOnce() -> Result<T, NativeError>,
{
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic_info| {
        let message = panic_info
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic_info.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(NativeError::new("nif_panic", message))
    })
}

fn encode_result<T: Encoder>(env: Env<'_>, result: Result<T, NativeError>) -> Term<'_> {
    match result {
        Ok(value) => (ok(), value).encode(env),
        Err(err) => (error(), err).encode(env),
    }
}

fn resolve_device(requested: Option<&str>) -> Result<(bool, &'static str), NativeError> {
    let lowered = requested.map(str::to_ascii_lowercase);
    match lowered.as_deref() {
        None | Some("auto") => match GPU_BACKEND {
            Some(label) => Ok((true, label)),
            None => Ok((false, "cpu")),
        },
        Some("cpu") => Ok((false, "cpu")),
        Some(other) if Some(other) == GPU_BACKEND => Ok((true, GPU_BACKEND.unwrap())),
        Some(other) => Err(NativeError::new(
            "invalid_request",
            "requested device backend is not enabled in this NIF artefact",
        )
        .with_detail("requested", other)
        .with_detail("enabled", GPU_BACKEND.map_or("cpu", |b| b).to_owned())),
    }
}

/// Reports the active backends compiled into this NIF artefact.
#[rustler::nif]
fn nif_available_devices(env: Env<'_>) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        let mut backends = vec!["cpu".to_owned()];
        if let Some(b) = GPU_BACKEND {
            backends.push(b.to_owned());
        }
        Ok(AvailableDevices {
            backends,
            gpu_supported: GPU_BACKEND.is_some(),
        })
    });
    encode_result(env, result)
}

/// Loads a GGML / GGUF whisper.cpp model file.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)]
fn nif_load_model(env: Env<'_>, path: String, opts: LoadOpts) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        let path_buf = PathBuf::from(&path);
        if !path_buf.is_file() {
            return Err(
                NativeError::new("invalid_request", "model path is not a regular file")
                    .with_detail("path", path.clone()),
            );
        }

        let (use_gpu, device_label) = resolve_device(opts.device.as_deref())?;

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(use_gpu);

        let ctx = WhisperContext::new_with_params(&path_buf, ctx_params).map_err(|reason| {
            NativeError::new("load_error", "failed to load whisper.cpp model")
                .with_detail("reason", reason.to_string())
                .with_detail("path", path.clone())
                .with_detail("device", device_label.to_owned())
        })?;

        // whisper.cpp's published checkpoints all run at 16 kHz; the C
        // API does not expose the rate.
        let sampling_rate = 16_000_usize;
        let multilingual = ctx.is_multilingual();
        let n_vocab = usize::try_from(ctx.n_vocab()).unwrap_or(0);
        let token_eot = u32::try_from(ctx.token_eot()).unwrap_or(u32::MAX);

        Ok(ResourceArc::new(WhisperResource {
            ctx: Mutex::new(Some(ctx)),
            sampling_rate,
            multilingual,
            n_vocab,
            token_eot,
            device: device_label,
        }))
    });

    encode_result(env, result)
}

/// Returns metadata cached at load time.
#[rustler::nif]
#[allow(clippy::needless_pass_by_value)]
fn nif_model_info(env: Env<'_>, model: ResourceArc<WhisperResource>) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        Ok(ModelInfo {
            sampling_rate: model.sampling_rate,
            multilingual: model.multilingual,
            n_vocab: model.n_vocab,
            device: model.device.to_owned(),
        })
    });
    encode_result(env, result)
}

fn decode_pcm_f32(bytes: &[u8]) -> Result<Vec<f32>, NativeError> {
    if bytes.is_empty() {
        return Err(NativeError::new(
            "invalid_request",
            "samples binary is empty",
        ));
    }

    if !bytes.len().is_multiple_of(4) {
        return Err(NativeError::new(
            "invalid_request",
            "samples binary length must be a multiple of 4 (f32)",
        )
        .with_detail("byte_length", bytes.len().to_string()));
    }

    // whisper.cpp takes sample counts as i32; larger buffers would
    // silently truncate at the FFI boundary.
    if bytes.len() / 4 > i32::MAX as usize {
        return Err(NativeError::new(
            "invalid_request",
            "samples binary exceeds the supported length (i32::MAX samples, about 37 hours)",
        )
        .with_detail("samples", (bytes.len() / 4).to_string()));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(NativeError::new(
                "invalid_request",
                "samples binary contains a non-finite sample (NaN or infinity); \
                 the upstream decoder produced corrupted audio",
            )
            .with_detail("sample_index", index.to_string()));
        }
        samples.push(value);
    }
    Ok(samples)
}

fn build_request(opts: TranscribeOpts) -> TranscribeRequest {
    TranscribeRequest {
        language: opts.language,
        translate: opts.translate.unwrap_or(false),
        initial_prompt: opts.initial_prompt,
        word_timestamps: opts.word_timestamps.unwrap_or(false),
        beam_size: opts.beam_size,
        best_of: opts.best_of,
        temperature: opts.temperature,
        n_threads: opts.n_threads,
        n_max_text_ctx: opts.n_max_text_ctx,
        offset_ms: opts.offset_ms,
        duration_ms: opts.duration_ms,
        no_speech_thold: opts.no_speech_thold,
        logprob_thold: opts.logprob_thold,
        suppress_blank: opts.suppress_blank,
        suppress_non_speech_tokens: opts.suppress_non_speech_tokens,
        single_segment: opts.single_segment,
        print_progress: opts.print_progress.unwrap_or(false),
        vad_model_path: opts.vad_model_path,
        vad_threshold: opts.vad_threshold,
        vad_min_speech_ms: opts.vad_min_speech_ms,
        vad_min_silence_ms: opts.vad_min_silence_ms,
        vad_speech_pad_ms: opts.vad_speech_pad_ms,
    }
}

/// Transcribes a single PCM buffer. The buffer may be longer than the
/// 30 s Whisper window; whisper.cpp chunks internally. `abort` and
/// `progress_pid` are optional cooperative hooks.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)]
fn nif_transcribe<'a>(
    env: Env<'a>,
    model: ResourceArc<WhisperResource>,
    samples_bin: Binary,
    opts: TranscribeOpts,
    abort: Option<ResourceArc<AbortHandle>>,
    progress_pid: Option<LocalPid>,
) -> Term<'a> {
    let bytes = samples_bin.as_slice();
    let abort_flag = abort.map(|h| Arc::clone(&h.flag));
    let result = run_with_panic_protection(|| {
        let samples = decode_pcm_f32(bytes)?;
        let request = build_request(opts);
        let transcription =
            transcribe::transcribe_one(&model, samples, &request, abort_flag, progress_pid)?;
        Ok(NifTranscription::from(transcription))
    });

    encode_result(env, result)
}

/// Allocates a fresh cooperative-cancellation flag.
#[rustler::nif]
fn nif_new_abort_handle() -> ResourceArc<AbortHandle> {
    ResourceArc::new(AbortHandle {
        flag: Arc::new(AtomicBool::new(false)),
    })
}

/// Signals an `AbortHandle`; in-flight transcribe calls observing this
/// flag will return early.
#[rustler::nif]
#[allow(clippy::needless_pass_by_value)]
fn nif_abort_handle_signal(handle: ResourceArc<AbortHandle>) -> rustler::Atom {
    handle.flag.store(true, Ordering::SeqCst);
    ok()
}

/// Reads the current state of an `AbortHandle` (`true` once signalled).
#[rustler::nif]
#[allow(clippy::needless_pass_by_value)]
fn nif_abort_handle_aborted(handle: ResourceArc<AbortHandle>) -> bool {
    handle.flag.load(Ordering::SeqCst)
}

fn on_load(env: Env<'_>, _info: Term<'_>) -> bool {
    native_log::install();
    env.register::<WhisperResource>().is_ok() && env.register::<AbortHandle>().is_ok()
}

rustler::init!("Elixir.WhisperCpp.Native", load = on_load);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_pcm_f32_round_trips_samples() {
        let mut bytes = Vec::new();
        for v in [0.0_f32, 1.0, -1.0, 0.5, -0.25] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let decoded = decode_pcm_f32(&bytes).unwrap();
        assert_eq!(decoded, vec![0.0, 1.0, -1.0, 0.5, -0.25]);
    }

    #[test]
    fn decode_pcm_f32_rejects_empty_input() {
        let err = decode_pcm_f32(&[]).unwrap_err();
        assert_eq!(err.r#type, "invalid_request");
    }

    #[test]
    fn decode_pcm_f32_rejects_misaligned_length() {
        let err = decode_pcm_f32(&[1, 2, 3]).unwrap_err();
        assert_eq!(err.r#type, "invalid_request");
        assert_eq!(
            err.details.get("byte_length").map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn decode_pcm_f32_rejects_non_finite_samples() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bytes = Vec::new();
            for v in [0.5_f32, bad, -0.5] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let err = decode_pcm_f32(&bytes).unwrap_err();
            assert_eq!(err.r#type, "invalid_request");
            assert_eq!(
                err.details.get("sample_index").map(String::as_str),
                Some("1")
            );
        }
    }

    #[test]
    fn resolve_device_auto_falls_back_to_cpu_without_gpu() {
        if GPU_BACKEND.is_none() {
            let (use_gpu, label) = resolve_device(None).unwrap();
            assert!(!use_gpu);
            assert_eq!(label, "cpu");

            let (use_gpu, label) = resolve_device(Some("auto")).unwrap();
            assert!(!use_gpu);
            assert_eq!(label, "cpu");
        }
    }

    #[test]
    fn resolve_device_cpu_works_in_any_build() {
        let (use_gpu, label) = resolve_device(Some("cpu")).unwrap();
        assert!(!use_gpu);
        assert_eq!(label, "cpu");
    }

    #[test]
    fn resolve_device_rejects_gpu_when_not_built_in() {
        if GPU_BACKEND.is_none() {
            assert!(resolve_device(Some("cuda")).is_err());
            assert!(resolve_device(Some("hipblas")).is_err());
        }
    }

    #[test]
    fn run_with_panic_protection_catches_string_panic() {
        let result: Result<(), _> = run_with_panic_protection(|| panic!("boom"));
        let err = result.unwrap_err();
        assert_eq!(err.r#type, "nif_panic");
        assert_eq!(err.message, "boom");
    }
}
