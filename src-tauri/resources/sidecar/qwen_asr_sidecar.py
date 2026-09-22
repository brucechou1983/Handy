#!/usr/bin/env python3
"""
Qwen3-ASR sidecar process for Handy.

Communicates with the Rust backend via stdin/stdout JSON protocol.
Uses mlx-audio for inference on Apple Silicon.

Protocol:
  Request:  {"command": "transcribe", "audio_path": "/tmp/audio.wav",
             "language": "Chinese", "system_prompt": "...", "hotwords": ["Handy"]}
  Response: {"ok": true, "text": "Hello world", "language": "English"}

  Request:  {"command": "load_model"}
  Response: {"ok": true, "mlx_audio_version": "0.5.5"}

  Request:  {"command": "health"}
  Response: {"ok": true, "model_loaded": true, "mlx_audio_version": "0.5.5"}

  Request:  {"command": "shutdown"}
  (process exits)

Omitting "language" (or passing null / "auto") lets the model detect the
language itself and report it back in the response.
"""

import json
import sys
import os
import traceback

MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"

# mlx-audio grew native `system_prompt` and `hotwords` arguments, plus
# language auto-detection, in 0.5.x. Earlier versions needed the prompt to be
# monkey-patched in and could not auto-detect, so Handy requires 0.5.5+.
MIN_MLX_AUDIO = (0, 5, 5)

model = None


def send_response(data: dict):
    """Send a JSON response to stdout."""
    line = json.dumps(data, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def send_error(message: str):
    send_response({"ok": False, "error": message})


def mlx_audio_version() -> str:
    try:
        from importlib.metadata import version

        return version("mlx-audio")
    except Exception:
        return "unknown"


def _parsed_version(raw: str):
    parts = []
    for piece in raw.split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def load_model():
    global model
    installed = mlx_audio_version()

    if installed == "unknown" or _parsed_version(installed) < MIN_MLX_AUDIO:
        send_error(
            "mlx-audio {} is too old (need {}+). Re-run Setup for Qwen3 ASR "
            "in Handy's model settings to update it.".format(
                installed, ".".join(str(n) for n in MIN_MLX_AUDIO)
            )
        )
        return

    try:
        from mlx_audio.stt import load

        model = load(MODEL_ID)
        send_response({"ok": True, "mlx_audio_version": installed})
    except Exception as e:
        send_error(f"Failed to load model: {e}")


def _first_language(value):
    """`generate` reports the language as a list for batched inputs."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value else None


def handle_transcribe(request: dict):
    if model is None:
        send_error("Model not loaded")
        return

    audio_path = request.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        send_error(f"Audio file not found: {audio_path}")
        return

    # language=None asks the model to detect the language itself.
    language = request.get("language")
    if language in ("", "auto"):
        language = None

    kwargs = {"language": language}

    system_prompt = request.get("system_prompt")
    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    hotwords = request.get("hotwords")
    if hotwords:
        kwargs["hotwords"] = list(hotwords)

    try:
        result = model.generate(audio_path, **kwargs)
        text = result.text if hasattr(result, "text") else str(result)
        send_response(
            {
                "ok": True,
                "text": text.strip(),
                "language": _first_language(getattr(result, "language", None)),
            }
        )
    except Exception as e:
        send_error(f"Transcription failed: {e}\n{traceback.format_exc()}")


def handle_health():
    send_response(
        {
            "ok": True,
            "model_loaded": model is not None,
            "mlx_audio_version": mlx_audio_version(),
        }
    )


def main():
    # Signal readiness
    send_response({"ok": True, "status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            send_error(f"Invalid JSON: {e}")
            continue

        command = request.get("command")

        if command == "load_model":
            load_model()
        elif command == "transcribe":
            handle_transcribe(request)
        elif command == "health":
            handle_health()
        elif command == "shutdown":
            send_response({"ok": True, "status": "shutting_down"})
            break
        else:
            send_error(f"Unknown command: {command}")

    sys.exit(0)


if __name__ == "__main__":
    main()
