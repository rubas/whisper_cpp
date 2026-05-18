//! Minimal RIFF/WAVE decoder for the formats whisper.cpp consumes.
//!
//! Accepts 16 kHz audio: mono/stereo 16-bit signed PCM, mono/stereo
//! 32-bit signed PCM, mono/stereo 32-bit IEEE-754 float. Stereo is
//! downmixed by averaging channels. Sample rates other than 16 kHz are
//! rejected so callers resample upstream rather than getting wrong
//! results.
//!
//! Returns the raw little-endian f32 byte buffer; the caller turns it
//! into a BEAM binary.

use crate::NativeError;

const TARGET_RATE: u32 = 16_000;
const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;

struct FmtChunk {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn req(cond: bool, msg: &str) -> Result<(), NativeError> {
    if cond {
        Ok(())
    } else {
        Err(NativeError::new("invalid_request", msg))
    }
}

fn read_u32_le(bytes: &[u8], at: usize) -> Result<u32, NativeError> {
    bytes
        .get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| NativeError::new("invalid_request", "truncated WAV header"))
}

fn read_u16_le(bytes: &[u8], at: usize) -> Result<u16, NativeError> {
    bytes
        .get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| NativeError::new("invalid_request", "truncated WAV header"))
}

/// Decode a WAV byte buffer into little-endian f32 mono PCM at 16 kHz.
pub fn decode(bytes: &[u8]) -> Result<Vec<u8>, NativeError> {
    req(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF/WAVE file",
    )?;

    let mut cursor = 12usize;
    let mut fmt: Option<FmtChunk> = None;
    let mut data: Option<&[u8]> = None;

    while cursor + 8 <= bytes.len() {
        let tag = &bytes[cursor..cursor + 4];
        let size = read_u32_le(bytes, cursor + 4)? as usize;
        let body_start = cursor + 8;
        let body_end = body_start
            .checked_add(size)
            .ok_or_else(|| NativeError::new("invalid_request", "WAV chunk size overflow"))?;

        if body_end > bytes.len() {
            return Err(NativeError::new(
                "invalid_request",
                "WAV chunk extends past end of file",
            ));
        }

        let body = &bytes[body_start..body_end];

        match tag {
            b"fmt " => fmt = Some(parse_fmt(body)?),
            b"data" => data = Some(body),
            _ => {}
        }

        cursor = body_end + (size & 1);
    }

    let fmt = fmt.ok_or_else(|| NativeError::new("invalid_request", "missing fmt chunk"))?;
    let data = data.ok_or_else(|| NativeError::new("invalid_request", "missing data chunk"))?;

    validate_fmt(&fmt)?;
    Ok(to_mono_f32_bytes(data, &fmt))
}

fn parse_fmt(body: &[u8]) -> Result<FmtChunk, NativeError> {
    req(body.len() >= 16, "malformed fmt chunk")?;
    Ok(FmtChunk {
        format_tag: read_u16_le(body, 0)?,
        channels: read_u16_le(body, 2)?,
        sample_rate: read_u32_le(body, 4)?,
        bits_per_sample: read_u16_le(body, 14)?,
    })
}

fn validate_fmt(fmt: &FmtChunk) -> Result<(), NativeError> {
    req(
        fmt.format_tag == FORMAT_PCM || fmt.format_tag == FORMAT_FLOAT,
        "unsupported WAV format tag (PCM=1, IEEE float=3 only)",
    )?;
    req(fmt.sample_rate == TARGET_RATE, "WAV must be 16 kHz")?;
    req(
        fmt.channels == 1 || fmt.channels == 2,
        "unsupported channel count (mono / stereo only)",
    )?;
    req(
        fmt.bits_per_sample == 16 || fmt.bits_per_sample == 32,
        "unsupported bits per sample (16 / 32 only)",
    )?;
    if fmt.format_tag == FORMAT_FLOAT && fmt.bits_per_sample != 32 {
        return Err(NativeError::new(
            "invalid_request",
            "IEEE float WAV must be 32-bit",
        ));
    }
    Ok(())
}

#[allow(clippy::cast_possible_wrap)]
fn to_mono_f32_bytes(data: &[u8], fmt: &FmtChunk) -> Vec<u8> {
    match (fmt.format_tag, fmt.bits_per_sample, fmt.channels) {
        (1, 16, 1) => map_chunks_le(data, 2, |c| {
            f32::from(i16::from_le_bytes([c[0], c[1]])) / 32_768.0
        }),
        (1, 16, 2) => map_chunks_le(data, 4, |c| {
            let l = i16::from_le_bytes([c[0], c[1]]);
            let r = i16::from_le_bytes([c[2], c[3]]);
            (f32::from(l) + f32::from(r)) / 65_536.0
        }),
        (1, 32, 1) => map_chunks_le(data, 4, |c| {
            i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0
        }),
        (1, 32, 2) => map_chunks_le(data, 8, |c| {
            let l = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32;
            let r = i32::from_le_bytes([c[4], c[5], c[6], c[7]]) as f32;
            (l + r) / 4_294_967_296.0
        }),
        (3, 32, 1) => {
            // Already f32 LE — pass through, dropping any tail less than 4 bytes.
            let usable = data.len() - (data.len() % 4);
            data[..usable].to_vec()
        }
        (3, 32, 2) => map_chunks_le(data, 8, |c| {
            let l = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let r = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            (l + r) / 2.0
        }),
        _ => Vec::new(),
    }
}

fn map_chunks_le<F: Fn(&[u8]) -> f32>(data: &[u8], stride: usize, f: F) -> Vec<u8> {
    let n = data.len() / stride;
    let mut out = Vec::with_capacity(n * 4);
    for c in data.chunks_exact(stride) {
        out.extend_from_slice(&f(c).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn riff(payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&((payload.len() + 4) as u32).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(payload);
        v
    }

    fn fmt_chunk(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = u32::from(channels) * u32::from(bits) / 8;
        let byte_rate = rate * block_align;
        let mut v = Vec::new();
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&byte_rate.to_le_bytes());
        v.extend_from_slice(&(block_align as u16).to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v
    }

    fn data_chunk(body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(body.len() as u32).to_le_bytes());
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn decodes_16khz_mono_16bit_pcm() {
        let body = (1..=4i16).flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>();
        let mut payload = fmt_chunk(1, 1, 16_000, 16);
        payload.extend(data_chunk(&body));
        let bytes = riff(&payload);

        let out = decode(&bytes).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn rejects_non_riff() {
        let err = decode(b"NOPE").unwrap_err();
        assert_eq!(err.r#type, "invalid_request");
    }

    #[test]
    fn rejects_44_1khz() {
        let mut payload = fmt_chunk(1, 1, 44_100, 16);
        payload.extend(data_chunk(&[]));
        assert!(decode(&riff(&payload)).is_err());
    }

    #[test]
    fn rejects_24_bit() {
        let mut payload = fmt_chunk(1, 1, 16_000, 24);
        payload.extend(data_chunk(&[]));
        assert!(decode(&riff(&payload)).is_err());
    }

    #[test]
    fn passes_through_f32_mono() {
        let body = [0.0_f32, 0.5, -0.5]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>();
        let mut payload = fmt_chunk(3, 1, 16_000, 32);
        payload.extend(data_chunk(&body));
        let out = decode(&riff(&payload)).unwrap();
        assert_eq!(out, body);
    }

    #[test]
    fn skips_unknown_chunks_and_handles_pad() {
        // junk chunk of odd length forces the +1 padding skip.
        let mut payload = fmt_chunk(1, 1, 16_000, 16);
        payload.extend_from_slice(b"junk");
        payload.extend_from_slice(&3u32.to_le_bytes());
        payload.extend_from_slice(b"abc");
        payload.push(0); // padding byte
        payload.extend(data_chunk(&[0u8; 4]));
        assert!(decode(&riff(&payload)).is_ok());
    }
}
