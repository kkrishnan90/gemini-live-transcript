# gemini-live-interrupt

Drop-in interrupt-resume for the [google-genai](https://pypi.org/project/google-genai/) Live API. One function call, zero code changes.

When a user interrupts the model mid-response, the Gemini Live API fires an `interrupted` signal but leaves it up to you to handle context recovery. This package patches the SDK transparently so the model automatically receives context about what it was saying and what the user said, enabling it to seamlessly resume, pivot, or blend.

## Install

```bash
pip install gemini-live-interrupt
```

## Quick start

```python
from gemini_live_interrupt import enable_interrupt_resume

enable_interrupt_resume()  # one call, done

# Your existing google-genai code works unchanged:
async with client.aio.live.connect(model=model, config=config) as session:
    await session.send(input="Hello!", end_of_turn=True)
    async for msg in session.receive():
        # interruptions are now auto-handled behind the scenes
        print(msg)
```

Call `enable_interrupt_resume()` once before connecting. Every `AsyncSession` created after that will automatically:

1. Track input/output transcription per session
2. Detect `content.interrupted` signals
3. Inject a context-recovery prompt via `send_client_content` so the model knows what it was saying and what the user said

## API

### `enable_interrupt_resume(prompt_builder=None, max_history=4)`

Patches `AsyncSession._receive()` to auto-handle interruptions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_builder` | `callable` or `None` | `None` | Custom function `(heard, user_text, history) -> str` that builds the resume prompt. If `None`, uses the built-in template. |
| `max_history` | `int` | `4` | Number of conversation turns to keep in the rolling history window (e.g., 4 = last 2 exchanges). |

**Parameters passed to `prompt_builder`:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `heard` | `str` | Accumulated output transcription text from the model's response up to the interruption point. |
| `user_text` | `str` | Latest user input transcription text (what the user said to interrupt). |
| `history` | `list[tuple[str, str]]` | Rolling conversation history as `(role, text)` tuples where role is `"user"` or `"model"`. |

**Example with custom prompt:**

```python
def my_prompt(heard: str, user_text: str, history: list) -> str:
    return f"You were interrupted. You said: {heard}. User said: {user_text}. Continue."

enable_interrupt_resume(prompt_builder=my_prompt)
```

### `disable_interrupt_resume()`

Removes the patch and restores the original `AsyncSession._receive()`. Takes no arguments.

```python
from gemini_live_interrupt import disable_interrupt_resume

disable_interrupt_resume()
```

## How it works

- Uses [wrapt](https://pypi.org/project/wrapt/) to wrap `AsyncSession._receive()` (the single-message async method, not the `receive()` async generator)
- Attaches an `_InterruptHandler` instance to each session lazily on first `_receive()` call (no `__init__` or `connect()` patching needed)
- Compatible with the existing `gemini-live-transcript` SDK patches (different methods are patched)
- Thread-safe patch application via lock
- Idempotent: calling `enable_interrupt_resume()` multiple times is safe

## Default prompt behavior

When an interruption is detected, the injected prompt includes:

1. **Recent conversation history** (last N turns)
2. **What the model was saying** (output transcription up to interruption)
3. **What the user said** (input transcription)
4. **Smart routing instructions** telling the model to:
   - Address new questions directly
   - Continue seamlessly if the interruption was just background noise
   - Blend both if the interruption was a follow-up

## Requirements

- Python >= 3.11
- `google-genai` SDK
- `wrapt`
