from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import sounddevice as sd
from google.genai import errors
from google.genai import types

from .config import LiveTranscriptSettings, build_live_connect_config, create_vertex_client


_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@dataclass(slots=True)
class TranscriptState:
    last_input_text: str = ""
    last_output_text: str = ""
    model_speaking: bool = False
    model_turn_interrupted: bool = False
    model_turn_audio_started: bool = False
    last_enqueued_output_text: str = ""
    pending_output_segments: deque[tuple[str, bool, int]] = field(default_factory=deque)
    printed_output_segments: list[tuple[str, bool, int]] = field(default_factory=list)
    model_turn_complete: bool = False
    interruption_count: int = 0
    events: list[str] = field(default_factory=list)
    # All output transcription text received during the current model turn,
    # accumulated regardless of interruption state.  This is the model's native
    # transcription (ground truth) and may extend beyond what was played.
    turn_all_output_texts: list[str] = field(default_factory=list)
    last_full_output_text: str = ""
    # Rolling conversation history (kept to last 4 turns = 2 exchanges).
    conversation_history: list[tuple[str, str]] = field(default_factory=list)
    # Interruption resume data — heard is set on interrupt, full on turn_complete.
    interrupted_full_transcript: str = ""
    interrupted_heard_transcript: str = ""
    interrupted_input_text: str = ""
    needs_resume: bool = False


class PlaybackBuffer:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._total_received = 0
        self._total_played = 0

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buf.extend(chunk)
            self._total_received += len(chunk)

    def read(self, n_bytes: int) -> bytes:
        with self._lock:
            if not self._buf:
                return b""
            size = min(n_bytes, len(self._buf))
            out = bytes(self._buf[:size])
            del self._buf[:size]
            self._total_played += size
            return out

    def clear(self) -> None:
        with self._lock:
            self._total_played += len(self._buf)
            self._buf.clear()

    @property
    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    @property
    def total_played(self) -> int:
        with self._lock:
            return self._total_played

    def stats(self) -> tuple[int, int, int]:
        with self._lock:
            return self._total_received, self._total_played, len(self._buf)


class LiveTranscriptionRunner:
    def __init__(self, settings: LiveTranscriptSettings) -> None:
        self.settings = settings
        self.state = TranscriptState()
        self._stop_event = asyncio.Event()
        self._input_audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
        self._playback = PlaybackBuffer()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _log(self, text: str) -> None:
        line = f"[{_ts()}] {text}"
        self.state.events.append(line)
        print(line, flush=True)

    def _log_event(self, text: str) -> None:
        if self.settings.debug_events:
            self._log(f"[EVENT] {text}")

    def _log_transcript(self, speaker: str, text: str, is_final: bool) -> None:
        stage = "FINAL" if is_final else "PARTIAL"
        self._log(f"[{speaker}][{stage}] {text}")

    def _enqueue_input_chunk(self, chunk: bytes) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._input_audio_q.put_nowait(chunk)
            return
        except asyncio.QueueFull:
            pass
        try:
            self._input_audio_q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self._input_audio_q.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def _input_callback(self, indata: bytes, frames: int, time: object, status: sd.CallbackFlags) -> None:
        if status:
            self._log_event(f"audio-input-status {status}")
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._enqueue_input_chunk, bytes(indata))

    def _output_callback(
        self, outdata: bytearray, frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        if status:
            self._log_event(f"audio-output-status {status}")
        needed = frames * 2
        chunk = self._playback.read(needed)
        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))
        outdata[:] = chunk

    async def _send_audio_loop(self, session: object) -> None:
        mime_type = f"audio/pcm;rate={self.settings.input_sample_rate_hz}"
        while not self._stop_event.is_set():
            chunk = await self._input_audio_q.get()
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=mime_type)
            )

    def _playback_backlog_ms(self) -> float:
        _, _, buffered_bytes = self._playback.stats()
        return (buffered_bytes / 2.0) * 1000.0 / float(self.settings.output_sample_rate_hz)

    def _flush_pending_transcripts(self, *, force: bool = False) -> None:
        """Print pending output transcript segments that have synced with audio playback."""
        if self.state.model_turn_interrupted:
            return
        played = self._playback.total_played
        # Allow text to lead audio by ~100ms worth of bytes.
        lead_bytes = int(self.settings.output_sample_rate_hz * 2 * 0.1)
        while self.state.pending_output_segments:
            text, is_final, audio_at = self.state.pending_output_segments[0]
            if not force and played + lead_bytes < audio_at:
                break
            self.state.pending_output_segments.popleft()
            if text and text != self.state.last_output_text:
                self._log_transcript("MODEL", text, is_final)
                self.state.printed_output_segments.append((text, is_final, audio_at))
                self.state.last_output_text = text
            if is_final:
                self.state.last_output_text = ""

    async def _playback_sync_loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush_pending_transcripts()
            if self.state.model_turn_complete and self._playback_backlog_ms() <= 20.0:
                self._flush_pending_transcripts(force=True)
                self.state.model_speaking = False
                self.state.model_turn_complete = False
                self.state.model_turn_audio_started = False
                self.state.printed_output_segments.clear()
                self.state.last_enqueued_output_text = ""
                self.state.turn_all_output_texts.clear()
                self.state.last_full_output_text = ""
            await asyncio.sleep(0.03)

    def _record_turn(self, role: str, text: str) -> None:
        """Append a turn to conversation history, keeping the last 4 entries."""
        if not text:
            return
        self.state.conversation_history.append((role, text))
        if len(self.state.conversation_history) > 4:
            self.state.conversation_history[:] = self.state.conversation_history[-4:]

    async def _inject_resume_context(self, session: object) -> None:
        """Send a context-injection prompt after an interruption so the model
        can decide whether to resume its interrupted response, pivot to the
        user's new question, or blend both."""
        full = self.state.interrupted_full_transcript
        heard = self.state.interrupted_heard_transcript
        user_text = self.state.interrupted_input_text

        if not full and not heard:
            return

        lines: list[str] = []
        lines.append(
            "[System Note — Interruption Context]\n"
            "Your previous response was interrupted by the user. "
            "Below is the full context so you can decide how to proceed."
        )

        # --- Recent conversation (last 2 exchanges) ---
        history = self.state.conversation_history
        if history:
            lines.append("")
            lines.append("=== Recent Conversation ===")
            for role, text in history:
                speaker = "User" if role == "user" else "You (Model)"
                lines.append(f"{speaker}: {text}")

        # --- Transcription comparison ---
        lines.append("")
        lines.append("=== Interrupted Response ===")
        lines.append(
            f"Model native transcription (ground truth, full output): {full}"
        )
        lines.append(
            f"Real-time audio transcription (what user heard before interruption): {heard}"
        )
        if full and heard and full != heard:
            if full.startswith(heard):
                remainder = full[len(heard):].strip()
            else:
                remainder = ""
            if remainder:
                lines.append(f"Unheard remainder (user missed this): {remainder}")

        # --- User's interrupting speech ---
        if user_text:
            lines.append("")
            lines.append(f"=== What the User Said (interruption) ===")
            lines.append(user_text)

        # --- Smart routing instructions ---
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

        prompt = "\n".join(lines)
        self._log(f"{_CYAN}[RESUME CONTEXT]{_RESET} injected ({len(prompt)} chars):")
        for pline in prompt.splitlines():
            self._log(f"{_CYAN}  | {pline}{_RESET}")

        # Inject as a user turn with turn_complete=True so the model
        # processes the context and generates a response.  This is sent at
        # content.interrupted time — before the model starts its next
        # response — so it won't kill an in-progress generation.
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
            turn_complete=True,
        )

    async def _handle_server_content(self, content: types.LiveServerContent) -> None:
        if content.input_transcription and content.input_transcription.text is not None:
            text = content.input_transcription.text.strip()
            if text and text != self.state.last_input_text:
                self._log_transcript(
                    "USER", text, bool(content.input_transcription.finished)
                )
                self.state.last_input_text = text
            if content.input_transcription.finished:
                if text:
                    self._record_turn("user", text)
                self.state.last_input_text = ""

        if content.output_transcription and content.output_transcription.text is not None:
            text = content.output_transcription.text.strip()
            is_final = bool(content.output_transcription.finished)
            # Always accumulate native transcription (ground truth) even after
            # interruption — post-interruption segments represent text the model
            # generated but whose audio was cut off.
            if text and text != self.state.last_full_output_text:
                self.state.turn_all_output_texts.append(text)
                self.state.last_full_output_text = text
                if is_final:
                    self.state.last_full_output_text = ""
            # Only feed playback-sync pipeline when not interrupted.
            if not self.state.model_turn_interrupted and text and text != self.state.last_enqueued_output_text:
                self.state.pending_output_segments.append(
                    (text, is_final, self._playback.total_received)
                )
                self.state.last_enqueued_output_text = text
                if is_final:
                    self.state.last_enqueued_output_text = ""
            self.state.model_speaking = True

        if content.model_turn and content.model_turn.parts:
            for part in content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    if not self.state.model_turn_interrupted:
                        self._playback.append(part.inline_data.data)
                        self.state.model_speaking = True
                        self.state.model_turn_audio_started = True
                if part.text:
                    self._log_event(f"model-text {part.text}")

        if content.interrupted:
            played = self._playback.total_played

            # Flush pending segments up to what was actually played.
            while self.state.pending_output_segments:
                text, is_final, audio_at = self.state.pending_output_segments[0]
                if audio_at > played:
                    break
                self.state.pending_output_segments.popleft()
                if text and text != self.state.last_output_text:
                    self._log_transcript("MODEL", text, is_final)
                    self.state.printed_output_segments.append((text, is_final, audio_at))
                    self.state.last_output_text = text

            # Find the last segment the user actually heard.
            last_heard_text = ""
            for seg_text, _, audio_at in self.state.printed_output_segments:
                if audio_at <= played:
                    last_heard_text = seg_text
                else:
                    break

            # Snapshot BOTH transcripts at interruption time, before state
            # is cleared and before the next turn's transcription can arrive.
            #
            # Full (native): everything the model transcribed for this turn.
            self.state.interrupted_full_transcript = " ".join(
                self.state.turn_all_output_texts
            )
            # Heard (real-time): all printed segments — the sync loop already
            # gates printing to within ~300ms of playback, so every printed
            # segment was heard (or about to be heard) by the user.
            heard_texts = [t for t, _, _ in self.state.printed_output_segments]
            self.state.interrupted_heard_transcript = (
                " ".join(heard_texts) if heard_texts else last_heard_text
            )
            self.state.interrupted_input_text = self.state.last_input_text

            self._playback.clear()
            self.state.interruption_count += 1
            self.state.model_speaking = False
            self.state.model_turn_interrupted = True
            self.state.pending_output_segments.clear()
            self.state.printed_output_segments.clear()
            self.state.last_enqueued_output_text = ""
            # Clear the accumulator so the next model turn starts fresh.
            self.state.turn_all_output_texts.clear()
            self.state.last_full_output_text = ""

            # Inject resume context NOW — before the model starts generating
            # its next response to the user's audio.  If we wait until
            # turn_complete, the model may have already begun responding and
            # our send_client_content would kill that generation.
            self.state.needs_resume = True

            if last_heard_text:
                self._log(f"{_YELLOW}[MODEL][INTERRUPTED] {last_heard_text}{_RESET}")
            self._log_event("model output interrupted by user speech")

        if content.generation_complete:
            self._log_event("generation_complete=True")

        if content.turn_complete:
            # Record the model turn to conversation history and trigger resume
            # if the turn was interrupted — must happen before resetting the flag.
            if self.state.model_turn_interrupted:
                # Resume context was already injected at interrupted time.
                # Just record history here.
                self._record_turn(
                    "model",
                    self.state.interrupted_heard_transcript
                    or self.state.interrupted_full_transcript,
                )
            else:
                full_text = " ".join(self.state.turn_all_output_texts)
                if full_text.strip():
                    self._record_turn("model", full_text.strip())

            self.state.model_turn_complete = True
            self.state.model_turn_interrupted = False
            reason = (
                content.turn_complete_reason.value
                if content.turn_complete_reason is not None
                else "UNKNOWN"
            )
            self._log_event(
                f"turn_complete=True reason={reason} waiting_for_input={content.waiting_for_input}"
            )

    async def _receive_loop(self, session: object) -> None:
        while not self._stop_event.is_set():
            # In google-genai, session.receive() returns one model turn and
            # exits after turn_complete. Re-enter it to keep the session alive.
            got_any_message = False
            async for message in session.receive():
                got_any_message = True
                if self._stop_event.is_set():
                    return

                if message.go_away:
                    self._log_event(f"go_away time_left={message.go_away.time_left}")

                if message.voice_activity_detection_signal:
                    vad = message.voice_activity_detection_signal.vad_signal_type.value
                    self._log_event(f"vad_signal={vad}")

                if message.voice_activity:
                    activity = message.voice_activity.voice_activity_type.value
                    self._log_event(f"voice_activity={activity}")

                if message.session_resumption_update:
                    update = message.session_resumption_update
                    self._log_event(
                        "session_resumption_update "
                        f"resumable={update.resumable} "
                        f"last_consumed_client_message_index={update.last_consumed_client_message_index}"
                    )

                if message.tool_call:
                    self._log_event("tool_call received")
                if message.tool_call_cancellation:
                    self._log_event("tool_call_cancellation received")

                if message.server_content:
                    await self._handle_server_content(message.server_content)
                    if self.state.needs_resume:
                        await self._inject_resume_context(session)
                        self.state.needs_resume = False

            if not got_any_message:
                await asyncio.sleep(0.05)

    async def _keyboard_loop(self, session: object) -> None:
        self._log_event(
            "Audio mode enabled. Speak into mic. Commands: /interrupt, /quit, /text <message>."
        )
        while not self._stop_event.is_set():
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                self._stop_event.set()
                return
            text = line.strip()
            if not text:
                continue
            if text == "/quit":
                self._stop_event.set()
                return
            if text == "/interrupt":
                self._log("Interruption is handled by the Live API's built-in VAD. Speak to interrupt.")
                continue
            if text.startswith("/text "):
                prompt = text[len("/text ") :].strip()
                if not prompt:
                    continue
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                    turn_complete=True,
                )
                self._log_event(f"user text sent: {prompt}")
                continue
            self._log_event("Ignoring typed text in audio-only mode. Use voice or /text <message>.")

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        client = create_vertex_client(self.settings)
        config = build_live_connect_config(self.settings)

        self._log_event(
            f"Connecting: project={self.settings.project_id} "
            f"location={self.settings.location} model={self.settings.model}"
        )
        self._log_event(
            f"Transcription model requested for input/output: {self.settings.transcription_model}"
        )

        input_stream = sd.RawInputStream(
            samplerate=self.settings.input_sample_rate_hz,
            channels=1,
            dtype="int16",
            callback=self._input_callback,
            blocksize=int(self.settings.input_sample_rate_hz / 10),
        )
        output_stream = sd.RawOutputStream(
            samplerate=self.settings.output_sample_rate_hz,
            channels=1,
            dtype="int16",
            callback=self._output_callback,
            blocksize=int(self.settings.output_sample_rate_hz / 10),
        )

        with input_stream, output_stream:
            try:
                async with client.aio.live.connect(model=self.settings.model, config=config) as session:
                    await self._run_connected_session(session)
            except errors.APIError as exc:
                if self._should_retry_without_transcription_model(exc, config):
                    self._log_event(
                        "server rejected transcription model field in Live setup; retrying with default transcription config"
                    )
                    fallback_config = dict(config)
                    fallback_config["input_audio_transcription"] = {}
                    fallback_config["output_audio_transcription"] = {}
                    async with client.aio.live.connect(
                        model=self.settings.model, config=fallback_config
                    ) as session:
                        await self._run_connected_session(session)
                else:
                    raise

        self._log_event(f"Stopped. total_interruptions={self.state.interruption_count}")

    def _should_retry_without_transcription_model(
        self, exc: errors.APIError, config: dict[str, object]
    ) -> bool:
        if not self.settings.fallback_to_default_transcription:
            return False
        if "input_audio_transcription" not in config:
            return False
        if "output_audio_transcription" not in config:
            return False
        message = str(exc).lower()
        return (
            "unknown name \"model\" at 'setup.input_audio_transcription'" in message
            or "unknown name \"model\" at 'setup.output_audio_transcription'" in message
        )

    async def _run_connected_session(self, session: object) -> None:
        send_audio_task = asyncio.create_task(self._send_audio_loop(session))
        receive_task = asyncio.create_task(self._receive_loop(session))
        playback_sync_task = asyncio.create_task(self._playback_sync_loop())
        keyboard_task = asyncio.create_task(self._keyboard_loop(session))
        stop_task = asyncio.create_task(self._stop_event.wait())

        tasks = {send_audio_task, receive_task, playback_sync_task, keyboard_task}
        while not self._stop_event.is_set():
            done, _ = await asyncio.wait(
                tasks | {stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                break

            for task in list(done):
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    self._stop_event.set()
                    break
                if task in tasks:
                    # A worker task ending unexpectedly means connection/input
                    # loop has ended; stop session cleanly.
                    self._log_event("worker task finished; ending session")
                    self._stop_event.set()
                    break

        with contextlib.suppress(Exception):
            await session.send_realtime_input(audio_stream_end=True)

        for task in tasks | {stop_task}:
            task.cancel()
        results = await asyncio.gather(*(tasks | {stop_task}), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                raise result
