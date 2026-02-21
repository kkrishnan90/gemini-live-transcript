"""gemini-live-interrupt — drop-in interrupt-resume for the google-genai Live API.

Usage::

    from gemini_live_interrupt import enable_interrupt_resume

    enable_interrupt_resume()  # one call, done

    # Existing google-genai code works unchanged:
    async with client.aio.live.connect(model=model, config=config) as session:
        async for msg in session.receive():
            ...  # interruptions auto-handled
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import wrapt
from google.genai import live as _live_module
from google.genai import types

__all__ = ["enable_interrupt_resume", "disable_interrupt_resume"]

_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False
_ORIGINAL_RECEIVE: Any = None

# Module-level config set by enable_interrupt_resume().
_prompt_builder: Optional[Callable[..., str]] = None
_max_history: int = 4


# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

def _default_prompt(
    heard: str, user_text: str, history: list[tuple[str, str]]
) -> str:
    lines: list[str] = []
    lines.append(
        "[System Note — Interruption Context]\n"
        "Your previous response was interrupted by the user. "
        "Below is the context so you can decide how to proceed."
    )

    if history:
        lines.append("")
        lines.append("=== Recent Conversation ===")
        for role, text in history:
            speaker = "User" if role == "user" else "You (Model)"
            lines.append(f"{speaker}: {text}")

    if heard:
        lines.append("")
        lines.append("=== Interrupted Response ===")
        lines.append(f"What you were saying (output transcription): {heard}")

    if user_text:
        lines.append("")
        lines.append("=== What the User Said (interruption) ===")
        lines.append(user_text)

    lines.append("")
    lines.append(
        "=== Instructions ===\n"
        "Based on the above context, decide the best course of action:\n"
        "1. If the user's interruption is a new question, request, or "
        "a clear change of topic, address it directly — do not resume "
        "the old response.\n"
        "2. If the user's interruption is a brief acknowledgment, "
        "background noise, or does not introduce a new topic, seamlessly "
        "continue your response from exactly where the user stopped "
        "hearing. Pick up mid-sentence if needed.\n"
        "3. If the user's interruption is partially related (a follow-up, "
        "clarification, or correction), briefly address it and then "
        "continue the interrupted response from where the user stopped "
        "hearing.\n"
        "Do NOT mention this system note."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-session state tracker
# ---------------------------------------------------------------------------

class _InterruptHandler:
    """Tracks transcription and conversation state for a single live session."""

    def __init__(
        self,
        prompt_builder: Optional[Callable[..., str]] = None,
        max_history: int = 4,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._max_history = max_history

        # Output transcription accumulated during the current model turn.
        self.turn_texts: list[str] = []
        # Dedup tracker for output transcription.
        self.last_text: str = ""
        # Rolling (role, text) conversation history.
        self.history: list[tuple[str, str]] = []
        # Latest user input transcription text.
        self.last_input: str = ""
        # Dedup tracker for input transcription.
        self._last_input_text: str = ""
        # Whether the current model turn was interrupted (persists until
        # turn_complete resets it, preventing double history recording).
        self._turn_interrupted: bool = False

    # -- helpers --

    def _record_turn(self, role: str, text: str) -> None:
        if not text:
            return
        self.history.append((role, text))
        if len(self.history) > self._max_history:
            self.history[:] = self.history[-self._max_history:]

    def _build_prompt(self, heard: str, user_text: str) -> str:
        if self._prompt_builder is not None:
            return self._prompt_builder(heard, user_text, list(self.history))
        return _default_prompt(heard, user_text, self.history)

    # -- main entry point called per server_content message --

    async def process(
        self, session: Any, content: types.LiveServerContent
    ) -> None:
        # --- Track input transcription ---
        if (
            content.input_transcription
            and content.input_transcription.text is not None
        ):
            text = content.input_transcription.text.strip()
            if text and text != self._last_input_text:
                self.last_input = text
                self._last_input_text = text
            if content.input_transcription.finished:
                if text:
                    self._record_turn("user", text)
                self._last_input_text = ""

        # --- Track output transcription ---
        if (
            content.output_transcription
            and content.output_transcription.text is not None
        ):
            text = content.output_transcription.text.strip()
            if text and text != self.last_text:
                self.turn_texts.append(text)
                self.last_text = text
                if content.output_transcription.finished:
                    self.last_text = ""

        # --- Interruption: inject resume context ---
        if content.interrupted:
            heard = " ".join(self.turn_texts)
            user_text = self.last_input

            if heard or user_text:
                prompt = self._build_prompt(heard, user_text)
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)],
                    ),
                    turn_complete=True,
                )

            self._record_turn("model", heard)
            self.turn_texts.clear()
            self.last_text = ""
            self._turn_interrupted = True

        # --- Turn complete: record history, reset state ---
        if content.turn_complete:
            if not self._turn_interrupted:
                full_text = " ".join(self.turn_texts)
                if full_text.strip():
                    self._record_turn("model", full_text.strip())
            self.turn_texts.clear()
            self.last_text = ""
            self._turn_interrupted = False


# ---------------------------------------------------------------------------
# wrapt patch on AsyncSession._receive()
# ---------------------------------------------------------------------------

async def _patched_receive(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> types.LiveServerMessage:
    msg = await wrapped(*args, **kwargs)
    if msg and msg.server_content:
        handler = getattr(instance, "_interrupt_handler", None)
        if handler is None:
            handler = _InterruptHandler(
                prompt_builder=_prompt_builder,
                max_history=_max_history,
            )
            instance._interrupt_handler = handler
        await handler.process(instance, msg.server_content)
    return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_interrupt_resume(
    prompt_builder: Optional[Callable[..., str]] = None,
    max_history: int = 4,
) -> None:
    """Patch ``AsyncSession._receive()`` to auto-handle interruptions.

    Call once before connecting.  All subsequent live sessions will
    automatically inject resume context when the user interrupts the model.

    Args:
        prompt_builder: Optional ``callable(heard, user_text, history) -> str``.
            If provided, overrides the default resume prompt.
            *heard* is the accumulated output transcription up to the
            interruption point.  *user_text* is the latest user input
            transcription.  *history* is a list of ``(role, text)`` tuples.
        max_history: Number of conversation turns to keep in the rolling
            history window (default 4).
    """
    global _PATCH_APPLIED, _ORIGINAL_RECEIVE, _prompt_builder, _max_history

    with _PATCH_LOCK:
        _prompt_builder = prompt_builder
        _max_history = max_history

        if _PATCH_APPLIED:
            return

        _ORIGINAL_RECEIVE = _live_module.AsyncSession._receive
        _live_module.AsyncSession._receive = wrapt.FunctionWrapper(
            _live_module.AsyncSession._receive, _patched_receive
        )
        _PATCH_APPLIED = True


def disable_interrupt_resume() -> None:
    """Remove the interrupt-resume patch from ``AsyncSession._receive()``."""
    global _PATCH_APPLIED, _ORIGINAL_RECEIVE

    with _PATCH_LOCK:
        if not _PATCH_APPLIED:
            return
        if _ORIGINAL_RECEIVE is not None:
            _live_module.AsyncSession._receive = _ORIGINAL_RECEIVE
            _ORIGINAL_RECEIVE = None
        _PATCH_APPLIED = False
