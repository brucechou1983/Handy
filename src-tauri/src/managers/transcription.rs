use crate::chinese;
use crate::managers::model::ModelManager;
use crate::managers::qwen_asr::QwenAsrManager;
use crate::settings::get_settings;
use anyhow::Result;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use tauri::{App, AppHandle, Emitter, Manager};
use whisper_rs::install_whisper_log_trampoline;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

pub struct TranscriptionManager {
    state: Mutex<Option<WhisperState>>,
    context: Mutex<Option<WhisperContext>>,
    model_manager: Arc<ModelManager>,
    qwen_asr_manager: Arc<QwenAsrManager>,
    app_handle: AppHandle,
    current_model_id: Mutex<Option<String>>,
    /// The backend of the currently loaded model: "whisper" or "qwen-asr"
    current_backend: Mutex<Option<String>>,
}

impl TranscriptionManager {
    pub fn new(
        app: &App,
        model_manager: Arc<ModelManager>,
        qwen_asr_manager: Arc<QwenAsrManager>,
    ) -> Result<Self> {
        let app_handle = app.app_handle().clone();

        let manager = Self {
            state: Mutex::new(None),
            context: Mutex::new(None),
            model_manager,
            qwen_asr_manager,
            app_handle: app_handle.clone(),
            current_model_id: Mutex::new(None),
            current_backend: Mutex::new(None),
        };

        // Try to load the default model from settings, but don't fail if no models are available
        let settings = get_settings(&app_handle);
        let _ = manager.load_model(&settings.selected_model);

        Ok(manager)
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        if model_info.backend == "qwen-asr" {
            return self.load_qwen_asr_model(model_id, &model_info.name);
        }

        // Whisper backend
        let model_path = self.model_manager.get_model_path(model_id)?;

        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid path for model: {}", model_id))?;

        println!(
            "Loading transcription model {} from: {}",
            model_id, path_str
        );

        // Install log trampoline once per model load (safe to call multiple times)
        install_whisper_log_trampoline();

        // Create new context
        let context =
            WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
                .map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "loading_failed".to_string(),
                            model_id: Some(model_id.to_string()),
                            model_name: Some(model_info.name.clone()),
                            error: Some(error_msg.clone()),
                        },
                    );
                    anyhow::anyhow!(error_msg)
                })?;

        // Create new state
        let state = context.create_state().map_err(|e| {
            let error_msg = format!("Failed to create state for model {}: {}", model_id, e);
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.clone()),
                },
            );
            anyhow::anyhow!(error_msg)
        })?;

        // Update the current context and state
        {
            let mut current_context = self.context.lock().unwrap();
            *current_context = Some(context);
        }
        {
            let mut current_state = self.state.lock().unwrap();
            *current_state = Some(state);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }
        {
            let mut backend = self.current_backend.lock().unwrap();
            *backend = Some("whisper".to_string());
        }

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        println!("Successfully loaded transcription model: {}", model_id);
        Ok(())
    }

    fn load_qwen_asr_model(&self, model_id: &str, model_name: &str) -> Result<()> {
        println!("Loading Qwen3-ASR model via sidecar...");

        match self.qwen_asr_manager.load_model() {
            Ok(()) => {
                // Clear whisper state since we're using a different backend
                {
                    let mut current_context = self.context.lock().unwrap();
                    *current_context = None;
                }
                {
                    let mut current_state = self.state.lock().unwrap();
                    *current_state = None;
                }
                {
                    let mut current_model = self.current_model_id.lock().unwrap();
                    *current_model = Some(model_id.to_string());
                }
                {
                    let mut backend = self.current_backend.lock().unwrap();
                    *backend = Some("qwen-asr".to_string());
                }

                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "loading_completed".to_string(),
                        model_id: Some(model_id.to_string()),
                        model_name: Some(model_name.to_string()),
                        error: None,
                    },
                );

                println!("Successfully loaded Qwen3-ASR model");
                Ok(())
            }
            Err(e) => {
                let error_msg = format!("Failed to load Qwen3-ASR: {}", e);
                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "loading_failed".to_string(),
                        model_id: Some(model_id.to_string()),
                        model_name: Some(model_name.to_string()),
                        error: Some(error_msg.clone()),
                    },
                );
                Err(anyhow::anyhow!(error_msg))
            }
        }
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        let st = std::time::Instant::now();

        if audio.is_empty() {
            println!("Empty audio vector");
            return Ok(String::new());
        }

        println!("Audio vector length: {}", audio.len());

        let backend = {
            let b = self.current_backend.lock().unwrap();
            b.clone()
        };

        let settings = get_settings(&self.app_handle);

        match backend.as_deref() {
            Some("qwen-asr") => {
                let language = qwen_language(&settings.selected_language);

                // `system_prompt`/`hotwords` stay None until Handy exposes a
                // custom-vocabulary setting; Qwen3-ASR treats that field as
                // biasing context, not as an instruction to follow.
                let result = self
                    .qwen_asr_manager
                    .transcribe(&audio, language, None, None)?;

                let et = std::time::Instant::now();
                println!(
                    "\nQwen3-ASR took {}ms (language: {})",
                    (et - st).as_millis(),
                    result.language.as_deref().unwrap_or("unreported")
                );

                Ok(apply_chinese_script(
                    result.text,
                    &settings,
                    result.language.as_deref(),
                ))
            }
            _ => {
                // Whisper backend (default)
                self.transcribe_whisper(audio, &settings, st)
            }
        }
    }

    fn transcribe_whisper(
        &self,
        audio: Vec<f32>,
        settings: &crate::settings::AppSettings,
        st: std::time::Instant,
    ) -> Result<String> {
        let mut result = String::new();

        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "No model loaded. Please download and select a model from settings first."
            )
        })?;

        // Initialize parameters
        let mut params = FullParams::new(SamplingStrategy::default());

        // Chinese variants share one Whisper language; which script comes out
        // is decided after transcription, in `apply_chinese_script`.
        let language = match settings.selected_language.as_str() {
            "auto" | "auto-zh-TW" | "auto-zh-CN" => None,
            "zh-TW" | "zh-CN" => Some("zh"),
            lang => Some(lang),
        };

        params.set_language(language);

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);

        if settings.translate_to_english {
            params.set_translate(true);
        }

        state
            .full(params, &audio)
            .expect("failed to convert samples");

        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");

        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            result.push_str(&segment);
        }

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        println!("\ntook {}ms{}", (et - st).as_millis(), translation_note);

        // Whisper does not report the language it detected back through this
        // pipeline, so the auto modes fall back to a script check.
        Ok(apply_chinese_script(
            result.trim().to_string(),
            settings,
            None,
        ))
    }
}

/// Map a Handy language code to a Qwen3-ASR language name.
///
/// `None` means "detect it": that is the right answer both for the auto modes
/// and for the languages Handy offers that Qwen3-ASR does not list, where
/// passing the code through would land in the prompt as a bogus language name.
fn qwen_language(code: &str) -> Option<&'static str> {
    match code {
        "zh" | "zh-TW" | "zh-CN" => Some("Chinese"),
        "yue" => Some("Cantonese"),
        "en" => Some("English"),
        "ar" => Some("Arabic"),
        "cs" => Some("Czech"),
        "da" => Some("Danish"),
        "de" => Some("German"),
        "el" => Some("Greek"),
        "es" => Some("Spanish"),
        "fa" => Some("Persian"),
        "fi" => Some("Finnish"),
        "fil" => Some("Filipino"),
        "fr" => Some("French"),
        "hi" => Some("Hindi"),
        "hu" => Some("Hungarian"),
        "id" => Some("Indonesian"),
        "it" => Some("Italian"),
        "ja" => Some("Japanese"),
        "ko" => Some("Korean"),
        "mk" => Some("Macedonian"),
        "ms" => Some("Malay"),
        "nl" => Some("Dutch"),
        "pl" => Some("Polish"),
        "pt" => Some("Portuguese"),
        "ro" => Some("Romanian"),
        "ru" => Some("Russian"),
        "sv" => Some("Swedish"),
        "th" => Some("Thai"),
        "tr" => Some("Turkish"),
        "vi" => Some("Vietnamese"),
        _ => None,
    }
}

/// Enforce the Chinese script the user selected.
///
/// Neither backend can be talked into a script: Whisper's `initial_prompt` and
/// Qwen3-ASR's system field are biasing context, and both models default to
/// Simplified for Mandarin whatever the prompt says. So the conversion happens
/// here. `reported_language` is the language the backend says it transcribed,
/// when it says; otherwise the script of the text decides, which keeps
/// Japanese and Korean output (shared Han characters) untouched.
fn apply_chinese_script(
    text: String,
    settings: &crate::settings::AppSettings,
    reported_language: Option<&str>,
) -> String {
    let target = match settings.selected_language.as_str() {
        "zh-TW" | "auto-zh-TW" => chinese::to_traditional,
        "zh-CN" | "auto-zh-CN" => chinese::to_simplified,
        // "zh" is "Chinese (Auto)": leave the model's own script alone.
        _ => return text,
    };

    // Translated output is English, and a translation request outranks a
    // script preference.
    if settings.translate_to_english {
        return text;
    }

    let is_chinese = match reported_language {
        Some(language) => matches!(
            language.to_ascii_lowercase().as_str(),
            "chinese" | "cantonese"
        ),
        None => chinese::looks_chinese(&text),
    };

    if is_chinese {
        target(&text)
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::get_default_settings;

    fn settings(language: &str) -> crate::settings::AppSettings {
        let mut settings = get_default_settings();
        settings.selected_language = language.to_string();
        settings
    }

    /// What Qwen3-ASR actually returns for Qwen's own asr_zh.wav sample, and
    /// what it should reach the clipboard as when Traditional is selected.
    const SIMPLIFIED: &str = "甚至出现交易几乎停滞的情况。";
    const TRADITIONAL: &str = "甚至出現交易幾乎停滯的情況。";

    #[test]
    fn forced_traditional_converts_reported_chinese() {
        assert_eq!(
            apply_chinese_script(SIMPLIFIED.to_string(), &settings("zh-TW"), Some("Chinese")),
            TRADITIONAL
        );
    }

    #[test]
    fn forced_traditional_converts_unreported_chinese() {
        assert_eq!(
            apply_chinese_script(SIMPLIFIED.to_string(), &settings("zh-TW"), None),
            TRADITIONAL
        );
    }

    #[test]
    fn forced_simplified_converts_back() {
        assert_eq!(
            apply_chinese_script(TRADITIONAL.to_string(), &settings("zh-CN"), Some("Chinese")),
            SIMPLIFIED
        );
    }

    #[test]
    fn auto_traditional_converts_only_chinese() {
        let japanese = "これは日本語です".to_string();
        assert_eq!(
            apply_chinese_script(japanese.clone(), &settings("auto-zh-TW"), Some("Japanese")),
            japanese
        );
        // Whisper reports nothing, so the script has to decide.
        assert_eq!(
            apply_chinese_script(japanese.clone(), &settings("auto-zh-TW"), None),
            japanese
        );
        assert_eq!(
            apply_chinese_script(SIMPLIFIED.to_string(), &settings("auto-zh-TW"), None),
            TRADITIONAL
        );
    }

    #[test]
    fn other_languages_are_left_alone() {
        for language in ["auto", "zh", "en", "ja"] {
            assert_eq!(
                apply_chinese_script(SIMPLIFIED.to_string(), &settings(language), Some("Chinese")),
                SIMPLIFIED,
                "{} should not touch the script",
                language
            );
        }
    }

    #[test]
    fn translation_outranks_the_script_preference() {
        let mut settings = settings("zh-TW");
        settings.translate_to_english = true;
        assert_eq!(
            apply_chinese_script(SIMPLIFIED.to_string(), &settings, Some("Chinese")),
            SIMPLIFIED
        );
    }

    #[test]
    fn qwen_languages_map_to_names_or_auto_detect() {
        assert_eq!(qwen_language("zh-TW"), Some("Chinese"));
        assert_eq!(qwen_language("zh-CN"), Some("Chinese"));
        assert_eq!(qwen_language("ja"), Some("Japanese"));
        assert_eq!(qwen_language("fil"), Some("Filipino"));
        // Auto modes, and the languages Qwen3-ASR does not support, detect.
        assert_eq!(qwen_language("auto"), None);
        assert_eq!(qwen_language("auto-zh-TW"), None);
        assert_eq!(qwen_language("uk"), None);
        assert_eq!(qwen_language("he"), None);
    }
}
