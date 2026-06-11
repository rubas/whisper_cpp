//! Voice-activity-detection preprocessing.
//!
//! whisper.cpp only honours `whisper_full_params.vad` in the
//! context-level `whisper_full`, not in `whisper_full_with_state`,
//! which is what whisper-rs (and this NIF) calls. So the NIF runs the
//! silero VAD itself, stitches the detected speech into a filtered
//! buffer the same way `whisper_full` does (segment overlap extension
//! plus short silence gaps), and remaps the resulting timestamps back
//! to the original timeline.

use crate::errors::{inference_error, invalid_request, load_error};
use crate::transcribe::TranscribeRequest;
use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

const SAMPLE_RATE: usize = 16_000;

/// Audio copied from each speech segment into the next one, matching
/// `whisper_vad_default_params().samples_overlap`.
const SEGMENT_OVERLAP_S: f32 = 0.1;

/// Silence inserted between stitched speech segments so whisper does
/// not merge words across a cut, matching `whisper_full`'s stitcher.
const SILENCE_GAP_SAMPLES: usize = SAMPLE_RATE / 10;

/// One contiguous speech span, present in both timelines.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpeechRange {
    /// First sample of the span in the original buffer.
    orig_start: usize,
    /// First sample of the span in the filtered buffer.
    filt_start: usize,
    /// Span length in samples (identical in both timelines).
    len: usize,
}

pub(crate) enum VadOutcome {
    NoSpeech,
    Speech {
        filtered: Vec<f32>,
        ranges: Vec<SpeechRange>,
    },
}

/// Select the sample window that `:offset_ms`/`:duration_ms` describe
/// on the original buffer. whisper.cpp cannot apply them itself when
/// VAD is active - it would window the silence-stripped timeline.
pub(crate) fn window(
    offset_ms: Option<u32>,
    duration_ms: Option<u32>,
    n_samples: usize,
) -> (usize, usize) {
    const SAMPLES_PER_MS: usize = SAMPLE_RATE / 1000;
    let start = (offset_ms.unwrap_or(0) as usize * SAMPLES_PER_MS).min(n_samples);
    let end = match duration_ms {
        Some(d) => start
            .saturating_add(d as usize * SAMPLES_PER_MS)
            .min(n_samples),
        None => n_samples,
    };
    (start, end)
}

/// Run silero VAD over `samples` (a window starting at
/// `origin_offset` samples into the original buffer) and stitch the
/// speech segments into a filtered buffer. Ranges carry original-buffer
/// positions so remapped timestamps stay absolute.
pub(crate) fn filter_speech(
    model_path: &str,
    req: &TranscribeRequest,
    samples: &[f32],
    origin_offset: usize,
) -> anyhow::Result<VadOutcome> {
    // Validate the model path even when the window is empty - a bad
    // path is a caller error regardless of the audio.
    if !std::path::Path::new(model_path).is_file() {
        return Err(invalid_request(format!(
            "vad_model_path is not a regular file: {model_path:?}"
        )));
    }

    if samples.is_empty() {
        return Ok(VadOutcome::NoSpeech);
    }

    let mut ctx_params = WhisperVadContextParams::new();
    if let Some(t) = req.n_threads {
        ctx_params.set_n_threads(crate::transcribe::u32_to_i32(t));
    }

    let mut vad_ctx = WhisperVadContext::new(model_path, ctx_params)
        .map_err(|e| load_error(format!("failed to load VAD model {model_path:?}: {e}")))?;

    let segments = vad_ctx
        .segments_from_samples(vad_params(req), samples)
        .map_err(|e| inference_error(format!("VAD segmentation failed: {e}")))?;

    let n = segments.num_segments();
    let mut spans_cs = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for i in 0..n {
        if let (Some(t0), Some(t1)) = (
            segments.get_segment_start_timestamp(i),
            segments.get_segment_end_timestamp(i),
        ) {
            spans_cs.push((t0, t1));
        }
    }

    let ranges = build_ranges(&spans_cs, samples.len(), origin_offset);
    if ranges.is_empty() {
        return Ok(VadOutcome::NoSpeech);
    }

    let filtered = stitch(samples, &ranges, origin_offset);
    Ok(VadOutcome::Speech { filtered, ranges })
}

fn vad_params(req: &TranscribeRequest) -> WhisperVadParams {
    let mut params = WhisperVadParams::new();
    if let Some(t) = req.vad_threshold {
        params.set_threshold(t);
    }
    if let Some(v) = req.vad_min_speech_ms {
        params.set_min_speech_duration(crate::transcribe::u32_to_i32(v));
    }
    if let Some(v) = req.vad_min_silence_ms {
        params.set_min_silence_duration(crate::transcribe::u32_to_i32(v));
    }
    if let Some(v) = req.vad_speech_pad_ms {
        params.set_speech_pad(crate::transcribe::u32_to_i32(v));
    }
    params
}

/// Convert VAD segments (centisecond timestamps) into sample ranges,
/// extending every segment but the last by [`SEGMENT_OVERLAP_S`] and
/// accounting for the silence gaps inserted between spans - the same
/// layout `whisper_full`'s VAD stitcher produces.
fn build_ranges(
    spans_cs: &[(f32, f32)],
    n_samples: usize,
    origin_offset: usize,
) -> Vec<SpeechRange> {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn cs_to_samples(cs: f32) -> usize {
        // f64 round-half-up, matching whisper.cpp's own conversion.
        ((f64::from(cs) / 100.0) * SAMPLE_RATE as f64)
            .round()
            .max(0.0) as usize
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let overlap_samples = (SEGMENT_OVERLAP_S * SAMPLE_RATE as f32) as usize;

    let mut ranges = Vec::with_capacity(spans_cs.len());
    let mut filt_start = 0_usize;

    for (i, &(t0, t1)) in spans_cs.iter().enumerate() {
        let start = cs_to_samples(t0).min(n_samples);
        let mut end = cs_to_samples(t1);
        if i + 1 < spans_cs.len() {
            end += overlap_samples;
        }
        let end = end.min(n_samples);
        if end <= start {
            continue;
        }

        let len = end - start;
        ranges.push(SpeechRange {
            orig_start: origin_offset + start,
            filt_start,
            len,
        });
        filt_start += len + SILENCE_GAP_SAMPLES;
    }

    ranges
}

/// Copy the speech spans out of `samples`, separated by silence gaps.
fn stitch(samples: &[f32], ranges: &[SpeechRange], origin_offset: usize) -> Vec<f32> {
    let last = ranges.last().expect("stitch is only called with speech");
    let mut filtered = vec![0.0_f32; last.filt_start + last.len];
    for r in ranges {
        let start = r.orig_start - origin_offset;
        filtered[r.filt_start..r.filt_start + r.len]
            .copy_from_slice(&samples[start..start + r.len]);
    }
    filtered
}

/// Map a timestamp from the filtered timeline back to the original one.
/// Timestamps that fall into a silence gap or whisper's end-of-audio
/// padding clamp to the end of the nearest preceding speech span, and a
/// span's mapped values never exceed the next span's original start, so
/// the mapping stays monotonic even where the overlap extension
/// duplicates audio across a span boundary.
pub(crate) fn remap_seconds(ranges: &[SpeechRange], t: f32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    fn s(samples: usize) -> f32 {
        samples as f32 / SAMPLE_RATE as f32
    }

    let mut current: Option<usize> = None;
    for (i, r) in ranges.iter().enumerate() {
        if s(r.filt_start) <= t {
            current = Some(i);
        } else {
            break;
        }
    }

    match current {
        // Before the first span: clamp to its start in the original.
        None => ranges.first().map_or(t, |r| s(r.orig_start)),
        Some(i) => {
            let r = &ranges[i];
            let delta = (t - s(r.filt_start)).clamp(0.0, s(r.len));
            let mapped = s(r.orig_start) + delta;
            match ranges.get(i + 1) {
                Some(next) => mapped.min(s(next.orig_start)),
                None => mapped,
            }
        }
    }
}

#[cfg(test)]
// Exact float equality is the point here - stitch copies samples
// verbatim - and the synthetic sample counts are tiny.
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // 16k samples = 1 s. Spans in centiseconds.
    const N: usize = 16_000 * 10;

    #[test]
    fn window_selects_offset_and_duration_in_original_samples() {
        assert_eq!(window(None, None, N), (0, N));
        assert_eq!(window(Some(2_000), None, N), (32_000, N));
        assert_eq!(window(Some(2_000), Some(4_000), N), (32_000, 96_000));
        // clamped to the buffer
        assert_eq!(window(Some(20_000), Some(1_000), N), (N, N));
    }

    #[test]
    fn build_ranges_bakes_in_the_window_origin() {
        let ranges = build_ranges(&[(100.0, 200.0)], N, 32_000);
        assert_eq!(ranges[0].orig_start, 32_000 + 16_000);
        // remap reports absolute original time: 0 s filtered = 3 s original
        assert!((remap_seconds(&ranges, 0.0) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn build_ranges_extends_all_but_last_segment_by_overlap() {
        let ranges = build_ranges(&[(100.0, 200.0), (500.0, 600.0)], N, 0);
        assert_eq!(ranges.len(), 2);
        // 1 s span + 0.1 s overlap
        assert_eq!(ranges[0].orig_start, 16_000);
        assert_eq!(ranges[0].len, 16_000 + 1_600);
        assert_eq!(ranges[0].filt_start, 0);
        // second span starts after first span + silence gap
        assert_eq!(ranges[1].orig_start, 80_000);
        assert_eq!(ranges[1].len, 16_000);
        assert_eq!(ranges[1].filt_start, 16_000 + 1_600 + 1_600);
    }

    #[test]
    fn build_ranges_clamps_to_buffer_and_drops_empty_spans() {
        let ranges = build_ranges(&[(900.0, 1_500.0), (1_500.0, 1_600.0)], N, 0);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].orig_start, 144_000);
        assert_eq!(ranges[0].len, N - 144_000);
    }

    #[test]
    fn stitch_copies_spans_and_leaves_silence_gaps() {
        let mut samples = vec![0.0_f32; N];
        samples[16_000] = 1.0;
        samples[80_000] = 2.0;
        let ranges = build_ranges(&[(100.0, 200.0), (500.0, 600.0)], N, 0);
        let filtered = stitch(&samples, &ranges, 0);

        assert_eq!(filtered[0], 1.0);
        assert_eq!(filtered[ranges[1].filt_start], 2.0);
        // the gap between the spans is silence
        assert_eq!(filtered[ranges[0].len + 100], 0.0);
        assert_eq!(filtered.len(), ranges[1].filt_start + ranges[1].len);
    }

    #[test]
    fn remap_stays_monotonic_when_overlap_crosses_into_the_next_span() {
        // Spans 0.05 s apart: the 0.1 s overlap extension reaches past
        // the next span's original start (2.0 s extended to 2.1 s vs
        // next start 2.05 s). Mapped times must never go backwards.
        let ranges = build_ranges(&[(100.0, 200.0), (205.0, 300.0)], N, 0);
        let mut prev = 0.0_f32;
        let mut t = 0.0_f32;
        while t < 4.0 {
            let mapped = remap_seconds(&ranges, t);
            assert!(
                mapped >= prev,
                "remap went backwards at t={t}: {mapped} < {prev}"
            );
            prev = mapped;
            t += 0.01;
        }
    }

    #[test]
    fn remap_translates_span_times_and_clamps_gaps() {
        let ranges = build_ranges(&[(100.0, 200.0), (500.0, 600.0)], N, 0);

        // inside the first span: 0.5 s into filtered = 1.5 s original
        assert!((remap_seconds(&ranges, 0.5) - 1.5).abs() < 1e-3);
        // inside the silence gap: clamps to the first span's original end
        let first_end = 1.0 + (ranges[0].len as f32 / 16_000.0);
        assert!((remap_seconds(&ranges, 1.15) - first_end).abs() < 1e-3);
        // inside the second span: 0.05 s into it = 5.05 s original
        let second_filt_s = ranges[1].filt_start as f32 / 16_000.0;
        assert!((remap_seconds(&ranges, second_filt_s + 0.05) - 5.05).abs() < 1e-3);
        // far past the end: clamps to the last span's original end
        assert!((remap_seconds(&ranges, 30.0) - 6.0).abs() < 1e-3);
    }
}
