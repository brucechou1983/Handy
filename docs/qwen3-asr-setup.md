# Qwen3-ASR 0.6B Setup Guide

Handy supports [Qwen3-ASR-0.6B](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) as an alternative speech-to-text backend via [mlx-audio](https://github.com/Blaizzy/mlx-audio) (0.5.5 or newer), running natively on Apple Silicon.

## Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
  ```bash
  brew install uv
  ```

No system Python installation is required. Handy uses `uv` to create a self-contained Python 3.11 virtual environment automatically.

## How It Works

When you click **"Qwen3 ASR 0.6B → Setup"** in the model selector, Handy:

1. Locates `uv` on your system (checks `/opt/homebrew/bin/uv`, `/usr/local/bin/uv`, or resolves via login shell)
2. Creates an isolated Python 3.11 venv at `~/Library/Application Support/com.handy.app/qwen-asr-venv/`
3. Installs `mlx-audio>=0.5.5,<0.6` from PyPI into the venv (upgrading it if an older version is already there)
4. Verifies the installed version
5. On first transcription, downloads the `mlx-community/Qwen3-ASR-0.6B-8bit` model from HuggingFace

A Python sidecar process (`qwen_asr_sidecar.py`) is spawned using the venv's Python and communicates with the Rust backend via stdin/stdout JSON protocol:

```jsonc
// Handy → sidecar. Omitting "language" lets the model detect it.
{"command": "transcribe", "audio_path": "/tmp/handy_qwen_asr_tmp.wav",
 "language": "Chinese", "system_prompt": "...", "hotwords": ["Handy"]}
// sidecar → Handy
{"ok": true, "text": "…", "language": "Chinese"}
```

`system_prompt` and `hotwords` are the model's [context/hotword](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf#context--hotwords) mechanism, which biases the transcription toward names and domain vocabulary. They are *not* an instruction slot — the model will not follow "reply in Traditional Chinese" placed there. No Handy setting fills them in yet.

## Why 0.5.5 or newer

mlx-audio 0.5.x is what made Handy's use of this model straightforward:

- `generate(..., system_prompt=..., hotwords=[...])` are real arguments, so Handy no longer monkey-patches mlx-audio's private prompt builder
- `language=None` auto-detects and reports the language back, so Handy's **Auto** no longer has to fall back to English

The prerequisite check reads the installed version and asks you to re-run Setup if it is older than 0.5.5.

## Chinese Script

Qwen3-ASR takes a language ("Chinese"), not a script, and emits Simplified for Mandarin regardless of the prompt. Handy converts the finished transcription with OpenCC instead — see the **Language** setting. This happens in the Rust backend, so it applies to the Whisper models too.

## Building from Source

```bash
# Install dependencies
bun install

# Build (use CMAKE_POLICY_VERSION_MINIMUM if you hit cmake errors on macOS)
CMAKE_POLICY_VERSION_MINIMUM=3.5 bun run tauri build

# The app bundle is at:
# src-tauri/target/release/bundle/macos/Handy.app
```

### Installing the Build

```bash
# Remove old installation first (quit Handy if running)
rm -rf /Applications/Handy.app

# Copy the new build
cp -R src-tauri/target/release/bundle/macos/Handy.app /Applications/Handy.app
```

## Troubleshooting

### `xattr -cr` error during `tauri build`

```
failed to bundle project: failed to remove extra attributes from app bundle: `failed to run xattr`
```

**Cause:** The pip-installed `xattr` Python package (which has a different CLI interface) shadows the system `/usr/bin/xattr` via pyenv shims. Tauri's bundler calls `xattr -cr` which the pip version doesn't support.

**Fix:** Uninstall the pip `xattr` package:
```bash
pip3 uninstall xattr
pyenv rehash  # if using pyenv
```

### "mlx-audio X is out of date (need 0.5.5+). Run Setup to update it."

**Cause:** The venv still has the mlx-audio that an earlier Handy version installed from the GitHub main branch (0.3.1), which lacks the native `system_prompt`/`hotwords` arguments and language auto-detection.

**Fix:** Click **Setup** again in the model selector; it upgrades in place. Or by hand:
```bash
uv pip install -U "mlx-audio>=0.5.5,<0.6" \
  --python ~/Library/Application\ Support/com.handy.app/qwen-asr-venv/bin/python3
```

### "mlx-audio is not installed" even after setup succeeds

**Cause:** The venv exists but the install did not finish — usually a network failure during `uv pip install`.

**Fix:** Run Setup again, or install by hand with the command above. Qwen3-ASR support has been in the PyPI releases of mlx-audio since 0.4.0, so no git checkout is needed any more.

### `mlx_audio.__version__` AttributeError

**Cause:** The `mlx_audio` package does not expose a `__version__` attribute. Early Handy versions checked the installation with `import mlx_audio; print(mlx_audio.__version__)`, which threw an `AttributeError` even though the package was correctly installed.

**Fix:** The check reads the package metadata instead: `from importlib.metadata import version; print(version('mlx-audio'))`.

### App can't find `python3` or `uv` when launched from Finder

**Cause:** macOS `.app` bundles launched from Finder/Spotlight get a minimal `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`), which doesn't include Homebrew paths like `/opt/homebrew/bin/`.

**Fix:** Handy resolves `uv` by checking well-known paths and falling back to a login shell lookup. The sidecar runs using the venv's absolute Python path, so no system Python dependency exists at runtime.

### Resetting the Qwen ASR Environment

To start fresh, delete the venv directory:
```bash
rm -rf ~/Library/Application\ Support/com.handy.app/qwen-asr-venv/
```
Then click "Setup" again in Handy.
