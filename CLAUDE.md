# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python package providing drop-in helpers for the Gemini Live API on Vertex AI. Enables real-time audio conversations with `gemini-live-2.5-flash-native-audio`, input/output transcription via Whisper, and interruption-aware transcript logging.

## Setup & Development Commands

```bash
# Create virtual environment and install dependencies
uv venv .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
uv pip install --python .venv-live/bin/python -e .

# Activate the environment
source .venv-live/bin/activate

# Run the CLI
gemini-live-transcribe

# Run with options
gemini-live-transcribe --project-id <ID> --voice-name Aoede --location us-central1
```

There is no test suite in this project.

## Architecture

The package lives in `src/gemini_live_transcript/` with four modules:

- **config.py** — `LiveTranscriptSettings` dataclass holding all runtime config (model, audio rates, VAD, voice, etc.). Settings resolve from CLI args → environment variables → defaults. `resolve_project_id()` checks `GEMINI_LIVE_PROJECT_ID` → ADC credentials → `GOOGLE_CLOUD_PROJECT`. Builds the `LiveConnectConfig` for Vertex AI.

- **patches.py** — Monkey-patches the `google-genai` SDK using `wrapt` to work around a Vertex API limitation: the API currently rejects nested `model` fields in transcription configs. The patch sanitizes these fields before transmission and restores them on serialization. Must be applied via `apply_google_genai_live_patch()` before connecting.

- **runtime.py** — Core async runtime (`LiveTranscriptionRunner`). Runs four concurrent asyncio tasks: `_send_audio_loop` (streams mic input to Gemini), `_receive_loop` (processes server messages — transcriptions, audio, turn events, interruptions), `_keyboard_loop` (handles `/quit`, `/interrupt`, `/text` commands), and `_playback_sync_loop` (synchronizes output transcript timing with the audio playback buffer). Uses `sounddevice` callbacks for real-time audio I/O. Has automatic fallback: retries without transcription model if Vertex rejects the config.

- **cli.py** — Argument parser and `main()` entry point. Merges CLI args with `LiveTranscriptSettings`, applies the SDK patch, and launches the runner.

### Data Flow

1. `cli.main()` → builds settings → applies SDK patch → creates Vertex client
2. `LiveTranscriptionRunner.run()` opens a live session via `client.aio.live.connect()`
3. Four concurrent loops run: mic audio streams out, server responses stream in (transcripts, audio, events), keyboard input handled, playback sync manages output transcript timing
4. `PlaybackBuffer` (thread-safe) queues model audio for the output callback
5. `TranscriptState` tracks partial/final transcripts, interruption counts, and events

### Key Vertex API Constraint

As of 2026-02-21, Vertex Live accepts only empty transcription configs (`"input_audio_transcription": {}`, `"output_audio_transcription": {}`). The `patches.py` module exists solely to work around this SDK validation issue. If/when Google fixes this, the patch can be removed.

## Environment Variables

All settings are configurable via environment variables prefixed with `GEMINI_LIVE_`:
`PROJECT_ID`, `LOCATION`, `MODEL`, `TRANSCRIPTION_MODEL`, `VOICE_NAME`, `INPUT_SAMPLE_RATE_HZ`, `OUTPUT_SAMPLE_RATE_HZ`, `DEBUG_EVENTS`, `PROACTIVE_AUDIO`, `AFFECTIVE_DIALOG`, `FALLBACK_TO_DEFAULT_TRANSCRIPTION`, `SYSTEM_INSTRUCTION`.

## Python Version

Requires Python >=3.11 (uses modern asyncio patterns and dataclass features).
