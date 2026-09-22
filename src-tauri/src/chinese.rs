//! Simplified ⇄ Traditional Chinese conversion for transcription output.
//!
//! ASR models emit whichever script dominates their training data, which for
//! Mandarin is Simplified. Asking for Traditional in the prompt does not
//! reliably change that: Whisper's `initial_prompt` and Qwen3-ASR's system
//! field are *biasing context*, not instruction slots, so "請使用繁體中文輸出"
//! is frequently ignored. The script is therefore fixed after the fact, with
//! the OpenCC rulesets, which is deterministic.

use ferrous_opencc::config::BuiltinConfig;
use ferrous_opencc::OpenCC;
use once_cell::sync::Lazy;

/// Simplified → Traditional, with Taiwan vocabulary (OpenCC `s2twp`:
/// 软件 → 軟體, 服务器 → 伺服器).
static TO_TRADITIONAL: Lazy<Option<OpenCC>> = Lazy::new(|| converter(BuiltinConfig::S2twp));

/// Traditional → Simplified, with mainland vocabulary (OpenCC `tw2sp`).
static TO_SIMPLIFIED: Lazy<Option<OpenCC>> = Lazy::new(|| converter(BuiltinConfig::Tw2sp));

fn converter(config: BuiltinConfig) -> Option<OpenCC> {
    match OpenCC::from_config(config) {
        Ok(cc) => Some(cc),
        Err(e) => {
            // Never fail a transcription over this: fall back to passthrough.
            eprintln!("OpenCC {:?} unavailable, leaving text as-is: {}", config, e);
            None
        }
    }
}

/// Convert to Traditional Chinese (Taiwan). Non-Chinese text passes through.
pub fn to_traditional(text: &str) -> String {
    match TO_TRADITIONAL.as_ref() {
        Some(cc) => cc.convert(text),
        None => text.to_string(),
    }
}

/// Convert to Simplified Chinese. Non-Chinese text passes through.
pub fn to_simplified(text: &str) -> String {
    match TO_SIMPLIFIED.as_ref() {
        Some(cc) => cc.convert(text),
        None => text.to_string(),
    }
}

/// Whether `text` looks like Chinese rather than Japanese or Korean.
///
/// Used by the auto-detect language modes: Whisper does not report the language
/// it detected back through this pipeline, so the script is the only signal.
/// Kana or Hangul means the utterance was not Chinese, and converting it would
/// corrupt shared Han characters (学 → 學 in Japanese text).
pub fn looks_chinese(text: &str) -> bool {
    let mut has_han = false;

    for c in text.chars() {
        match c {
            // Hiragana, katakana, katakana phonetic extensions.
            '\u{3040}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' => return false,
            // Hangul syllables, jamo, compatibility jamo.
            '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' => {
                return false
            }
            // CJK unified ideographs, extension A, compatibility ideographs.
            '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}' => {
                has_han = true
            }
            _ => {}
        }
    }

    has_han
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_simplified_to_traditional() {
        // The exact sentence Qwen3-ASR returns for Qwen's own asr_zh.wav sample.
        assert_eq!(
            to_traditional("甚至出现交易几乎停滞的情况。"),
            "甚至出現交易幾乎停滯的情況。"
        );
    }

    #[test]
    fn converts_taiwan_vocabulary() {
        assert_eq!(to_traditional("这个软件的服务器"), "這個軟體的伺服器");
    }

    #[test]
    fn converts_traditional_to_simplified() {
        assert_eq!(to_simplified("這個軟體的伺服器"), "这个软件的服务器");
    }

    #[test]
    fn leaves_ascii_alone() {
        let text = "Handy transcribes speech at 16 kHz.";
        assert_eq!(to_traditional(text), text);
    }

    #[test]
    fn detects_chinese_but_not_japanese_or_korean() {
        assert!(looks_chinese("这是中文"));
        assert!(looks_chinese("混合 English 的中文"));
        assert!(!looks_chinese("これは日本語です"));
        assert!(!looks_chinese("漢字とかな"));
        assert!(!looks_chinese("한국어입니다"));
        assert!(!looks_chinese("pure english"));
        assert!(!looks_chinese(""));
    }
}
