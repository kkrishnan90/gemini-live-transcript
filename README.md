# gemini-live-transcript

A CLI application for real-time voice conversations with Google's Gemini Live API on Vertex AI. The app streams microphone audio to the model, plays back the model's audio response through the speaker, and prints live transcriptions for both sides. When the user interrupts the model mid-response, the app automatically injects context so the model can seamlessly continue from where it was cut off.

## Setup

```bash
# Clone and set up
git clone https://github.com/kkrishnan90/gemini-live-transcript.git
cd gemini-live-transcript

# Create virtual environment and install
uv venv .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
uv pip install --python .venv-live/bin/python -e .

# Activate
source .venv-live/bin/activate
```

**Prerequisites:**
- Python >= 3.11
- A GCP project with the Vertex AI API enabled
- Google Cloud auth configured via `gcloud auth application-default login` or a service account key (`GOOGLE_APPLICATION_CREDENTIALS`)

## Running the CLI

```bash
gemini-live-transcribe
# or with explicit project
gemini-live-transcribe --project-id your-gcp-project-id
```

Once running, speak into your microphone. The model responds with audio and transcriptions print to the terminal in real time:

```
[14:30:01.123] [USER][FINAL] Tell me a story about a cat.
[14:30:02.456] [MODEL][PARTIAL] Once upon a time,
[14:30:03.789] [MODEL][PARTIAL] there was a cat named Shadow...
```

**CLI commands** (type while running):
- `/quit` — end the session
- `/text <message>` — send a text prompt instead of voice
- Any other typed input is ignored (audio-only mode)

**Tip:** Use headphones so the model's audio playback doesn't get picked up by the microphone as user speech.

## How Interruption and Contextual Resume Works

This is the core feature. When you interrupt the model while it's speaking, the Gemini Live API sends an `interrupted` signal. The app handles this with three steps:

### Step 1: Track what was heard vs. what was not

The app maintains a `PlaybackBuffer` that tracks two byte counters: `total_received` (audio bytes from the server) and `total_played` (audio bytes actually sent to the speaker). Every output transcription segment is tagged with the `total_received` value at the time it arrives, creating a temporal anchor between text and audio.

When `interrupted` fires, the app compares each transcription segment's byte marker against `total_played` to split the transcript into:
- **Heard:** segments whose audio was actually played through the speaker
- **Not heard:** segments that were buffered or generated after the interruption

This split happens in `_handle_server_content()` in `runtime.py` (around line 336).

### Step 2: Find the sentence-level continuation point

Rather than trying to pick up from the exact word where audio was cut (which the model gets wrong — it skips words), the app finds the **beginning of the sentence that was being spoken** when the interruption occurred.

The `_find_continuation_text()` method:
1. Locates where the heard text ends within the full transcript
2. Looks backward from that point for the last sentence boundary (`.` `!` `?` followed by a space)
3. Returns everything from that sentence start onward

Example:
```
Full transcript:  "A scavenger discovers an artifact that threatens to reawaken a power. Hunted by a corporation, she must protect the secret."
User heard:       "A scavenger discovers an artifact that threatens"
Continuation:     "A scavenger discovers an artifact that threatens to reawaken a power. Hunted by a corporation, she must protect the secret."
                   ↑ restarts from beginning of the truncated sentence, not from "threatens"
```

This logic lives in `_find_continuation_text()` as a static method on `LiveTranscriptionRunner` in `runtime.py`.

### Step 3: Inject context so the model knows what to do

Immediately when `interrupted` fires (before the model starts generating its next response), the app sends a context-injection prompt via `send_client_content()` with `turn_complete=True`. This prompt includes:

- **Recent conversation history** (last 4 turns)
- **What the user heard** and **what they did not hear**
- **The user's interrupting speech** (input transcription)
- **Exact continuation text** — the verbatim text the model should say if it needs to continue
- **Routing instructions** telling the model to:
  1. If the user asked a new question — answer it, then offer to continue the interrupted response ("Would you like me to finish the story?")
  2. If the interruption was noise/acknowledgment — seamlessly continue from the sentence start
  3. If it was a follow-up — address it, then continue

This injection happens in `_inject_resume_context()` in `runtime.py`.

### Playback-to-transcript sync

A background loop (`_playback_sync_loop`) runs every 30ms, comparing the playback position against pending transcript segments. Text only appears on screen when its corresponding audio is within ~100ms of being played. This keeps the printed transcript synchronized with what the user actually hears. The sync tolerance is set at line 171 in `runtime.py`:

```python
lead_bytes = int(self.settings.output_sample_rate_hz * 2 * 0.1)  # 100ms
```

## File-by-file breakdown

### `src/gemini_live_transcript/config.py`

All runtime configuration. Contains:

- **`LiveTranscriptSettings`** dataclass — every tunable parameter: model name, audio sample rates, voice, VAD sensitivity, system instruction, etc. All fields can be set via `GEMINI_LIVE_*` environment variables.
- **`resolve_project_id()`** — resolves the GCP project ID from `GEMINI_LIVE_PROJECT_ID` → ADC credentials file → `GOOGLE_CLOUD_PROJECT`.
- **`build_live_connect_config()`** — builds the `LiveConnectConfig` dict sent to the Gemini Live API. Configures audio response mode, voice selection, VAD settings (speech sensitivity, padding, silence thresholds), input/output transcription, and proactive audio.
- **`create_vertex_client()`** — creates the `google-genai` client with Vertex AI mode.

### `src/gemini_live_transcript/runtime.py`

The core async runtime. This is where all the real logic lives.

**`TranscriptState`** (dataclass) — mutable state for the session:
- Tracks partial/final transcription text for both user and model
- Maintains a rolling conversation history (last 4 turns)
- Stores interruption data: the full model transcript, what was heard, what the user said
- Interruption counter and turn state flags

**`PlaybackBuffer`** — thread-safe byte buffer connecting the async receive loop to the `sounddevice` output callback:
- `append()` — called when audio chunks arrive from the server
- `read()` — called by the speaker output callback to drain audio
- `clear()` — called on interruption to stop playback immediately
- Tracks `total_received` and `total_played` byte counters used for transcript-to-audio synchronization

**`LiveTranscriptionRunner`** — the main runner class. `run()` opens audio streams and a live session, then launches four concurrent asyncio tasks:

| Task | What it does |
|------|-------------|
| `_send_audio_loop` | Reads mic input from a queue and streams it to the Gemini session as PCM audio blobs |
| `_receive_loop` | Consumes server messages — routes transcription, audio, VAD, and turn events to their handlers |
| `_playback_sync_loop` | Runs every 30ms to print transcript segments that have caught up with audio playback |
| `_keyboard_loop` | Handles `/quit`, `/text` commands from stdin |

**Key methods in the interrupt flow:**

- **`_handle_server_content()`** — processes every `server_content` message. Tracks input/output transcription, feeds audio to `PlaybackBuffer`, and on `content.interrupted`: snapshots heard vs. full transcripts using byte-position comparison, clears the playback buffer, and sets `needs_resume = True`.
- **`_find_continuation_text(full, heard)`** — static method that computes the sentence-level continuation point. Finds the truncation position in the full text, looks backward for the last sentence boundary, and returns from that sentence start onward.
- **`_inject_resume_context(session)`** — builds and sends the context-injection prompt with conversation history, heard/unheard split, exact continuation text, and routing instructions.
- **`_record_turn(role, text)`** — appends to the rolling conversation history (capped at 4 entries).

### `src/gemini_live_transcript/cli.py`

The entry point. Parses CLI arguments (`--project-id`, `--location`, `--model`, `--voice-name`, etc.), merges them with `LiveTranscriptSettings.from_environment()`, applies the SDK patch, and runs `LiveTranscriptionRunner` via `asyncio.run()`.

### `src/gemini_live_transcript/__init__.py`

Package exports: `LiveTranscriptSettings`, `build_live_connect_config`, `create_vertex_client`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_LIVE_PROJECT_ID` | *(from ADC)* | GCP project ID |
| `GEMINI_LIVE_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_LIVE_MODEL` | `gemini-live-2.5-flash-native-audio` | Live model ID |
| `GEMINI_LIVE_TRANSCRIPTION_MODEL` | `whisper-large-v3` | Server-side transcription model |
| `GEMINI_LIVE_VOICE_NAME` | `Aoede` | Prebuilt voice name |
| `GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ` | `16000` | Mic input sample rate |
| `GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ` | `24000` | Speaker output sample rate |
| `GEMINI_LIVE_PROACTIVE_AUDIO` | `true` | Enable proactive audio |
| `GEMINI_LIVE_AFFECTIVE_DIALOG` | `true` | Enable affective dialog |
| `GEMINI_LIVE_DEBUG_EVENTS` | `false` | Print all event-level logs |
| `GEMINI_LIVE_SYSTEM_INSTRUCTION` | *(none)* | System instruction for the model |

## CLI Flags

| Flag | Description |
|------|-------------|
| `--project-id` | Vertex AI project ID |
| `--location` | Vertex AI region |
| `--model` | Live model ID |
| `--transcription-model` | Transcription model ID |
| `--voice-name` | Prebuilt voice name |
| `--disable-proactive-audio` | Turn off proactive audio |
| `--disable-affective-dialog` | Turn off affective dialog |
| `--disable-transcription-model-fallback` | Fail instead of retrying with empty transcription config |
