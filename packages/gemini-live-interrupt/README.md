# gemini-live-interrupt

Automatic interrupt-resume for the Gemini Live API. When a user interrupts the model mid-sentence, this package injects context behind the scenes so the model knows what it was saying, what the user said, and can decide whether to continue, pivot, or blend both.

Works with any existing `google-genai` application. No code changes required beyond a single function call.

## Install

```bash
pip install gemini-live-interrupt
```

## The Problem

When using the Gemini Live API for real-time audio conversations, users can interrupt the model at any time. The API sends an `interrupted` signal, but the model has no memory of what it was saying or how far it got. On its next response, it starts fresh — losing the context of the interrupted response entirely.

For example, if the model is telling a 5-line story and the user interrupts at line 2, the model won't know to continue from line 2. It either starts over, or moves on as if the story never happened.

## How This Package Solves It

Call `enable_interrupt_resume()` once before you connect to the Gemini Live API. After that, every live session automatically:

1. Tracks what the model was saying (output transcription)
2. Tracks what the user said (input transcription)
3. Detects when an interruption happens
4. Injects a context-recovery prompt to the model with what was said and what to do next

The model then decides the right course of action based on the context.

## Step-by-step Usage

### Step 1: Install the package

```bash
pip install gemini-live-interrupt
```

### Step 2: Add one line to your existing code

Add `enable_interrupt_resume()` **before** you create any live sessions:

```python
from gemini_live_interrupt import enable_interrupt_resume

enable_interrupt_resume()
```

That's it. Your existing `google-genai` code does not need any other changes.

### Step 3: Make sure transcription is enabled in your config

The package reads the transcription text that the Gemini Live API provides. For this to work, your `LiveConnectConfig` must have input and output transcription enabled:

```python
config = {
    "response_modalities": ["AUDIO"],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    # ... your other config
}
```

If transcription is not enabled, the package has nothing to track and interruptions won't be handled.

### Step 4: Connect and use the API as normal

```python
from google import genai
from google.genai import types
from gemini_live_interrupt import enable_interrupt_resume

# Step 2: enable before connecting
enable_interrupt_resume()

# Your existing code — nothing changes below this line
client = genai.Client(vertexai=True, project="my-project", location="us-central1")

config = {
    "response_modalities": [types.Modality.AUDIO],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Aoede"}
        }
    },
}

async with client.aio.live.connect(
    model="gemini-live-2.5-flash-native-audio", config=config
) as session:
    # send audio, receive responses — interruptions are handled automatically
    async for msg in session.receive():
        if msg.server_content and msg.server_content.model_turn:
            for part in msg.server_content.model_turn.parts:
                if part.inline_data:
                    # play or forward audio as usual
                    play_audio(part.inline_data.data)
```

## What Happens When the User Interrupts

When the model is speaking and the user starts talking, the API sends `content.interrupted`. This package intercepts it and sends a context prompt to the model containing:

1. **Recent conversation** — the last few turns of dialogue so the model has context
2. **What was being said** — the output transcription accumulated before the interruption
3. **What the user said** — the input transcription of the interrupting speech
4. **Exact continuation text** — the sentence that was cut off, from its beginning (not mid-word)
5. **Instructions** for how to proceed:
   - **New question:** answer it, then ask if the user wants to continue the interrupted response
   - **Background noise / acknowledgment:** seamlessly continue from the sentence that was cut off
   - **Follow-up or clarification:** address it briefly, then continue the interrupted response

The model receives all of this before generating its next response, so it can make an informed decision.

## Example: FastAPI WebSocket App

```python
from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types
from gemini_live_interrupt import enable_interrupt_resume

app = FastAPI()
enable_interrupt_resume()

client = genai.Client(vertexai=True, project="my-project", location="us-central1")

@app.websocket("/live")
async def live_audio(ws: WebSocket):
    await ws.accept()

    config = {
        "response_modalities": [types.Modality.AUDIO],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }

    async with client.aio.live.connect(
        model="gemini-live-2.5-flash-native-audio", config=config
    ) as session:
        async for msg in session.receive():
            if msg.server_content and msg.server_content.model_turn:
                for part in msg.server_content.model_turn.parts:
                    if part.inline_data:
                        await ws.send_bytes(part.inline_data.data)
```

Interruptions are handled transparently. When a connected client interrupts the model, the resume context is injected automatically.

## Custom Prompt

If you want to control exactly what prompt is injected on interruption, pass a `prompt_builder` function:

```python
def my_prompt(heard: str, user_text: str, history: list) -> str:
    """
    Args:
        heard: what the model was saying (output transcription up to interruption)
        user_text: what the user said to interrupt
        history: list of (role, text) tuples — recent conversation turns
    Returns:
        the prompt string to inject
    """
    return (
        f"You were interrupted. You were saying: {heard}. "
        f"The user said: {user_text}. "
        f"Continue from where you left off."
    )

enable_interrupt_resume(prompt_builder=my_prompt)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_builder` | `callable` or `None` | `None` | Custom `(heard, user_text, history) -> str` function. If `None`, uses the built-in prompt. |
| `max_history` | `int` | `4` | Number of conversation turns to keep (4 = last 2 exchanges). |

## Disabling

To remove the patch and restore original behavior:

```python
from gemini_live_interrupt import disable_interrupt_resume

disable_interrupt_resume()
```

## Requirements

- Python >= 3.11
- `google-genai` SDK
- `wrapt`
