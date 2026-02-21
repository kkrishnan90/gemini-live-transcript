from __future__ import annotations

import argparse
import asyncio

from .config import LiveTranscriptSettings
from .patches import apply_google_genai_live_patch
from .runtime import LiveTranscriptionRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gemini Live API native-audio with input/output transcription."
    )
    parser.add_argument("--project-id", help="Vertex AI project id.")
    parser.add_argument("--location", default=None, help="Vertex AI region (default: us-central1).")
    parser.add_argument("--model", default=None, help="Live model id.")
    parser.add_argument("--transcription-model", default=None, help="Transcription model id.")
    parser.add_argument("--voice-name", default=None, help="Prebuilt voice name.")
    parser.add_argument("--disable-proactive-audio", action="store_true")
    parser.add_argument("--disable-affective-dialog", action="store_true")
    parser.add_argument(
        "--disable-transcription-model-fallback",
        action="store_true",
        help="Fail fast if Vertex rejects explicit transcription model fields.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = LiveTranscriptSettings.from_environment()
    if args.project_id:
        settings.project_id = args.project_id
    if args.location:
        settings.location = args.location
    if args.model:
        settings.model = args.model
    if args.transcription_model:
        settings.transcription_model = args.transcription_model
    if args.voice_name:
        settings.voice_name = args.voice_name
    if args.disable_proactive_audio:
        settings.enable_proactive_audio = False
    if args.disable_affective_dialog:
        settings.enable_affective_dialog = False
    if args.disable_transcription_model_fallback:
        settings.fallback_to_default_transcription = False

    apply_google_genai_live_patch()

    runner = LiveTranscriptionRunner(settings)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
