from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-live-2.5-flash-native-audio"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3"
DEFAULT_LOCATION = "us-central1"
DEFAULT_INPUT_SAMPLE_RATE_HZ = 16000
DEFAULT_OUTPUT_SAMPLE_RATE_HZ = 24000
DEFAULT_VOICE_NAME = "Aoede"


def _project_id_from_adc_key() -> str | None:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        return None
    path = Path(credentials_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    project_id = data.get("project_id")
    if isinstance(project_id, str) and project_id.strip():
        return project_id.strip()
    return None


def resolve_project_id() -> str:
    explicit = os.environ.get("GEMINI_LIVE_PROJECT_ID")
    if explicit:
        return explicit
    from_adc_key = _project_id_from_adc_key()
    if from_adc_key:
        return from_adc_key
    from_env = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if from_env:
        return from_env
    raise RuntimeError(
        "No project id found. Set GEMINI_LIVE_PROJECT_ID or GOOGLE_APPLICATION_CREDENTIALS."
    )


@dataclass(slots=True)
class LiveTranscriptSettings:
    project_id: str
    location: str = DEFAULT_LOCATION
    model: str = DEFAULT_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    voice_name: str = DEFAULT_VOICE_NAME
    input_sample_rate_hz: int = DEFAULT_INPUT_SAMPLE_RATE_HZ
    output_sample_rate_hz: int = DEFAULT_OUTPUT_SAMPLE_RATE_HZ
    enable_proactive_audio: bool = True
    enable_affective_dialog: bool = True
    debug_events: bool = False
    fallback_to_default_transcription: bool = True
    system_instruction: str | None = None

    @classmethod
    def from_environment(cls) -> "LiveTranscriptSettings":
        return cls(
            project_id=resolve_project_id(),
            location=os.environ.get("GEMINI_LIVE_LOCATION", DEFAULT_LOCATION),
            model=os.environ.get("GEMINI_LIVE_MODEL", DEFAULT_MODEL),
            transcription_model=os.environ.get(
                "GEMINI_LIVE_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
            ),
            voice_name=os.environ.get("GEMINI_LIVE_VOICE_NAME", DEFAULT_VOICE_NAME),
            input_sample_rate_hz=int(
                os.environ.get(
                    "GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ",
                    str(DEFAULT_INPUT_SAMPLE_RATE_HZ),
                )
            ),
            output_sample_rate_hz=int(
                os.environ.get(
                    "GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ",
                    str(DEFAULT_OUTPUT_SAMPLE_RATE_HZ),
                )
            ),
            enable_proactive_audio=os.environ.get(
                "GEMINI_LIVE_PROACTIVE_AUDIO", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            enable_affective_dialog=os.environ.get(
                "GEMINI_LIVE_AFFECTIVE_DIALOG", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            debug_events=os.environ.get("GEMINI_LIVE_DEBUG_EVENTS", "false").lower()
            in {"1", "true", "yes", "on"},
            fallback_to_default_transcription=os.environ.get(
                "GEMINI_LIVE_FALLBACK_TO_DEFAULT_TRANSCRIPTION", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            system_instruction=os.environ.get("GEMINI_LIVE_SYSTEM_INSTRUCTION"),
        )


def create_vertex_client(settings: LiveTranscriptSettings) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=settings.project_id,
        location=settings.location,
    )


def build_live_connect_config(settings: LiveTranscriptSettings) -> dict[str, Any]:
    config: dict[str, Any] = {
        "response_modalities": [types.Modality.AUDIO],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": settings.voice_name},
            },
        },
        "enable_affective_dialog": settings.enable_affective_dialog,
        "proactivity": {"proactive_audio": settings.enable_proactive_audio},
        "input_audio_transcription": {"model": settings.transcription_model},
        "output_audio_transcription": {"model": settings.transcription_model},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "disabled": False,
                "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_HIGH,
                "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 800,
            },
            "activity_handling": types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            "turn_coverage": types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        },
    }
    if settings.system_instruction:
        config["system_instruction"] = settings.system_instruction
    return config
