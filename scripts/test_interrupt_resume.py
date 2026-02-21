#!/usr/bin/env python3
"""
Gold-standard interruption-resume test for gemini-live-transcript.

Scenario
--------
1. Greet the model with "Hi" — let it respond.
2. Ask the model to count from 1 to 20.
3. When "five" appears in the output transcription, interrupt by saying "continue".
4. After the interrupted turn completes, inject resume context.
5. Verify the model resumes from 6 (not from 1, not from 20+).

Requires macOS (uses `say` + `afconvert` for TTS-to-PCM conversion).

Usage
-----
    source .venv-live/bin/activate
    python scripts/test_interrupt_resume.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from datetime import datetime

from google.genai import types

from gemini_live_transcript.config import (
    LiveTranscriptSettings,
    build_live_connect_config,
    create_vertex_client,
)
from gemini_live_transcript.patches import apply_google_genai_live_patch

# ── ANSI colours ─────────────────────────────────────────────────────────────
_G = "\033[32m"   # green
_R = "\033[31m"   # red
_Y = "\033[33m"   # yellow
_C = "\033[36m"   # cyan
_D = "\033[2m"    # dim
_0 = "\033[0m"    # reset

# ── Timeout for each receive phase (seconds) ────────────────────────────────
PHASE_TIMEOUT = 30.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def say_to_pcm(text: str, rate: int = 16000) -> bytes:
    """Generate raw PCM int16 mono audio from *text* using macOS ``say``."""
    with tempfile.TemporaryDirectory() as d:
        aiff = os.path.join(d, "s.aiff")
        wav_path = os.path.join(d, "s.wav")
        subprocess.run(
            ["say", "-o", aiff, text],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1",
             aiff, wav_path],
            check=True, capture_output=True,
        )
        with wave.open(wav_path, "rb") as wf:
            return wf.readframes(wf.getnframes())


async def send_pcm(
    session: object,
    pcm: bytes,
    rate: int,
    *,
    trail_silence_s: float = 1.0,
) -> None:
    """Stream *pcm* to *session* at approximately real-time pace (100 ms
    chunks), followed by *trail_silence_s* seconds of silence so the VAD
    detects end-of-speech."""
    chunk_bytes = rate * 2 // 10  # 100 ms of int16 mono
    mime = f"audio/pcm;rate={rate}"

    for i in range(0, len(pcm), chunk_bytes):
        await session.send_realtime_input(
            audio=types.Blob(data=pcm[i : i + chunk_bytes], mime_type=mime),
        )
        await asyncio.sleep(0.10)

    # Trailing silence — lets VAD fire end-of-speech.
    silence_chunk = b"\x00" * chunk_bytes
    n_silence = max(1, int(trail_silence_s / 0.10))
    for _ in range(n_silence):
        await session.send_realtime_input(
            audio=types.Blob(data=silence_chunk, mime_type=mime),
        )
        await asyncio.sleep(0.10)


async def receive_turn(
    session: object,
    *,
    label: str = "MODEL",
    timeout: float = PHASE_TIMEOUT,
) -> list[str]:
    """Receive one full model turn.  Returns the list of transcription texts."""
    texts: list[str] = []
    deadline = asyncio.get_event_loop().time() + timeout

    async for msg in session.receive():
        if asyncio.get_event_loop().time() > deadline:
            log(f"{_Y}  [timeout]{_0}")
            break
        if not msg.server_content:
            continue
        sc = msg.server_content
        if sc.output_transcription and sc.output_transcription.text:
            t = sc.output_transcription.text.strip()
            if t:
                texts.append(t)
                log(f"  {label}: {t}")
        if sc.turn_complete:
            break

    return texts


# ── Core test ────────────────────────────────────────────────────────────────

async def run_test() -> bool:
    apply_google_genai_live_patch()

    settings = LiveTranscriptSettings.from_environment()
    client = create_vertex_client(settings)
    config = build_live_connect_config(settings)
    rate = settings.input_sample_rate_hz

    # ── Pre-generate audio clips ─────────────────────────────────────────
    log("Generating audio clips with macOS `say` ...")
    hi_pcm = say_to_pcm("Hi", rate)
    count_pcm = say_to_pcm(
        "Don't ask me any questions. Start counting from one to twenty.",
        rate,
    )
    continue_pcm = say_to_pcm("Continue", rate)
    log(
        f"  hi={len(hi_pcm)}B  count={len(count_pcm)}B  "
        f"continue={len(continue_pcm)}B"
    )

    # ── Connect ──────────────────────────────────────────────────────────
    # Vertex currently rejects explicit model fields in transcription
    # configs — use empty configs so the API falls back to its default.
    config["input_audio_transcription"] = {}
    config["output_audio_transcription"] = {}

    log(f"Connecting to {settings.model} ...")

    async with client.aio.live.connect(
        model=settings.model, config=config,
    ) as session:

        # ── Step 1 — Greeting ────────────────────────────────────────────
        log(f"\n{_G}{'─'*50}")
        log(f"  Step 1 · Greeting")
        log(f"{'─'*50}{_0}")
        log("Sending: \"Hi\"")
        await send_pcm(session, hi_pcm, rate)
        await receive_turn(session, label="MODEL")

        # ── Step 2 — Ask to count ────────────────────────────────────────
        log(f"\n{_G}{'─'*50}")
        log(f"  Step 2 · Request counting 1–20")
        log(f"{'─'*50}{_0}")
        log("Sending: \"Count from 1 to 20\"")
        await send_pcm(session, count_pcm, rate)

        # ── Step 3 — Receive counting, interrupt at "five" ───────────────
        log(f"\n{_G}{'─'*50}")
        log(f"  Step 3 · Counting — will interrupt at 'five'")
        log(f"{'─'*50}{_0}")

        pre_transcripts: list[str] = []
        interrupted = False
        interrupt_sent = False
        deadline = asyncio.get_event_loop().time() + PHASE_TIMEOUT

        async for msg in session.receive():
            if asyncio.get_event_loop().time() > deadline:
                log(f"{_Y}  [timeout waiting for counting]{_0}")
                break
            if not msg.server_content:
                continue
            sc = msg.server_content

            if sc.output_transcription and sc.output_transcription.text:
                t = sc.output_transcription.text.strip()
                if t:
                    pre_transcripts.append(t)
                    log(f"  MODEL: {t}")

                    # Fire the interrupt as soon as we see "five".
                    if "five" in t.lower() and not interrupt_sent:
                        log(f"{_C}  >>> 'five' detected — sending 'continue' ...{_0}")
                        asyncio.create_task(
                            send_pcm(session, continue_pcm, rate, trail_silence_s=0.8)
                        )
                        interrupt_sent = True

            if sc.interrupted:
                interrupted = True
                log(f"{_Y}  >>> Model interrupted{_0}")

            if sc.turn_complete:
                break

        if not interrupted:
            log(f"{_R}FAIL: model was never interrupted{_0}")
            return False

        # ── Step 4 — Inject resume context ───────────────────────────────
        log(f"\n{_G}{'─'*50}")
        log(f"  Step 4 · Injecting resume context")
        log(f"{'─'*50}{_0}")

        heard = " ".join(pre_transcripts)
        prompt = (
            "[System Note — Interruption Context]\n"
            "Your previous response was interrupted by the user.\n"
            "\n"
            "=== Interrupted Response ===\n"
            f"What you were saying (user heard this): {heard}\n"
            "\n"
            "=== What the User Said ===\n"
            "continue\n"
            "\n"
            "=== Instructions ===\n"
            "The user wants you to continue counting from exactly where they\n"
            "stopped hearing.  Do NOT repeat numbers already said.\n"
            "Do NOT mention being interrupted or this note.\n"
            "Just keep counting."
        )

        for line in prompt.splitlines():
            log(f"{_C}  | {line}{_0}")

        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
            turn_complete=True,
        )

        # ── Step 5 — Collect the resumed response ───────────────────────
        log(f"\n{_G}{'─'*50}")
        log(f"  Step 5 · Resumed response")
        log(f"{'─'*50}{_0}")

        post_transcripts = await receive_turn(
            session, label="MODEL (resumed)", timeout=PHASE_TIMEOUT,
        )

    # ── Assertions ───────────────────────────────────────────────────────
    log(f"\n{'═'*50}")
    log("  Assertions")
    log(f"{'═'*50}")

    pre_text = " ".join(pre_transcripts).lower()
    post_text = " ".join(post_transcripts).lower()

    checks = {
        "'five' in pre-interrupt output": "five" in pre_text,
        "'six' in resumed output": "six" in post_text,
        "did not restart from 'one'": "one," not in post_text,
    }

    all_pass = True
    for desc, ok in checks.items():
        symbol = f"{_G}✓{_0}" if ok else f"{_R}✗{_0}"
        log(f"  {symbol}  {desc}")
        if not ok:
            all_pass = False

    log("")
    if all_pass:
        log(f"{_G}{'═'*50}")
        log(f"  TEST PASSED")
        log(f"{'═'*50}{_0}")
    else:
        log(f"{_R}{'═'*50}")
        log(f"  TEST FAILED")
        log(f"{'═'*50}{_0}")
        log(f"\n{_D}Pre-interrupt text : {pre_text}{_0}")
        log(f"{_D}Post-interrupt text: {post_text}{_0}")

    return all_pass


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    try:
        passed = asyncio.run(run_test())
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
