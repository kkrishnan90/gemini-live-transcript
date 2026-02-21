# gemini-live-transcript

Simple package for Gemini Live API on Vertex AI that enables:
- live audio conversation with `gemini-live-2.5-flash-native-audio`
- input + output transcription
- Whisper v3 large transcription config passthrough (`whisper-large-v3`)
- interruption-aware transcript logging (partial model transcript + user interruption events)

## Setup (uv + separate venv)

```bash
cd /Users/krishnankumar/Desktop/gemini-live-transcript
uv venv .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
uv pip install --python .venv-live/bin/python -e .
```

## Configuration

All tunable runtime config lives in:
- `/Users/krishnankumar/Desktop/gemini-live-transcript/src/gemini_live_transcript/config.py`

Defaults:
- model: `gemini-live-2.5-flash-native-audio`
- location: `us-central1`
- transcription model: `whisper-large-v3`
- output transcript lead cap vs playback: `0ms` (strict sync)
- transcript hold-back before printing partial output: `220ms`
- VAD:
  - start speech sensitivity: `LOW`
  - end speech sensitivity: `HIGH`
  - prefix padding: `300ms`
  - silence duration: `800ms`
- affective dialog: enabled
- proactive audio: enabled

Project id resolution order:
1. `GEMINI_LIVE_PROJECT_ID`
2. `project_id` from `GOOGLE_APPLICATION_CREDENTIALS` JSON
3. `GOOGLE_CLOUD_PROJECT`

## Run CLI test

```bash
source .venv-live/bin/activate
gemini-live-transcribe
```

Or:

```bash
python /Users/krishnankumar/Desktop/gemini-live-transcript/scripts/live_transcribe_cli.py
```

CLI commands:
- `/quit` stop session
- `/interrupt` send explicit `activity_start` + `activity_end` signals via `send_realtime_input`
- `/text <message>` optional typed prompt
- any other typed line is ignored (audio-only mode)

Default CLI output is transcript-only with explicit speaker tags:
- `[USER][PARTIAL] ...`
- `[USER][FINAL] ...`
- `[MODEL][PARTIAL] ...`
- `[MODEL][FINAL] ...`

For accurate interruption behavior, use headphones (or strong echo cancellation) so model playback is not re-captured by the mic as new user speech.

Optional sync tuning env vars:
- `GEMINI_LIVE_MAX_OUTPUT_TRANSCRIPT_LEAD_MS` (default `0`)
- `GEMINI_LIVE_OUTPUT_TRANSCRIPT_HOLD_MS` (default `220`)
- `GEMINI_LIVE_DEBUG_EVENTS` (default `false`, enable event/debug logs)

## Integration in existing app

```python
from google import genai
from gemini_live_transcript import (
    LiveTranscriptSettings,
    apply_google_genai_live_patch,
    build_live_connect_config,
)

settings = LiveTranscriptSettings.from_environment()
apply_google_genai_live_patch()

client = genai.Client(
    vertexai=True,
    project=settings.project_id,
    location=settings.location,
)

config = build_live_connect_config(settings)

try:
    async with client.aio.live.connect(model=settings.model, config=config) as session:
        ...
except Exception as exc:
    # If Vertex rejects transcription model fields, retry with {}
    fallback = dict(config)
    fallback["input_audio_transcription"] = {}
    fallback["output_audio_transcription"] = {}
    async with client.aio.live.connect(model=settings.model, config=fallback) as session:
        ...
```

This keeps native SDK API/class/method names (`send_client_content`, `send_realtime_input`, `turn_complete`, `go_away`, `interrupted`, `voice_activity_detection_signal`, etc.) and only patches SDK validation around transcription config keys.

Session management note:
- `google-genai` `session.receive()` yields a single model turn and ends at `turn_complete`.
- The runner re-enters `session.receive()` in a loop so conversations continue seamlessly across turns.

## Important Vertex behavior (as of 2026-02-21)

Vertex Live currently accepts:

```json
{
  "input_audio_transcription": {},
  "output_audio_transcription": {}
}
```

and rejects a nested `model` field in those objects for setup.  
The CLI retries automatically with empty transcription configs if that server-side validation error happens.
