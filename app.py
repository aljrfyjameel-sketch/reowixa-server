import os
import io
import re
import json
import wave
import base64
import asyncio
import logging
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types


# ============================================================
# RIWEXA - Persistent Gemini Live Session
# Native Audio | Arabic + English | ESP32-S3
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA")

app = FastAPI(title="RIWEXA Saudi Cultural Robot")

API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

client = genai.Client(api_key=API_KEY) if API_KEY else None

MAX_PCM_BYTES = 320_000
MAX_TEXT_LENGTH = 4000
AUDIO_CHUNK_SIZE = 2048
SUPPORTED_SAMPLE_RATES = (16000, 24000)

SYSTEM_PROMPT = """
You are RIWEXA, a Saudi cultural guide robot.

You help visitors discover:
- Saudi history and heritage.
- Saudi regions and traditions.
- Museums, historical sites and landmarks.
- Traditional arts, crafts, clothing and food.
- Saudi culture and national identity.

LANGUAGE:
- Automatically identify the language spoken by the visitor.
- If the visitor speaks Arabic, answer in natural Arabic.
- If the visitor speaks English, answer in natural English.
- If the visitor changes language, immediately follow the latest language.
- Never ask the visitor to choose a language.
- Keep the conversation context across all questions in this visit.

BEHAVIOR:
- Be welcoming, clear, friendly and informative.
- Do not invent historical facts, dates or sources.
- If you do not know something, say so honestly.
- Do not claim to see the visitor or surroundings.
- Keep spoken answers appropriate for a public exhibition.

AUDIO:
- Always answer using native audio.
- Do not replace spoken answers with text.
- Speak naturally and clearly.
"""

# ============================================================
# HTTP
# ============================================================

@app.get("/")
async def home():
    return {
        "status": "ok",
        "service": "RIWEXA",
        "mode": "persistent-native-audio"
    }


@app.head("/")
async def health_head():
    return JSONResponse(content=None, status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if client else "missing_api_key",
        "service": "RIWEXA",
        "model": LIVE_MODEL
    }


# ============================================================
# Audio helpers
# ============================================================

def make_wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


def read_wav_audio(data: bytes):
    if not data.startswith(b"RIFF"):
        raise ValueError("Invalid WAV file")

    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono")

        if wav.getsampwidth() != 2:
            raise ValueError("Audio must be PCM16")

        rate = wav.getframerate()

        if rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("Unsupported sample rate")

        pcm = wav.readframes(wav.getnframes())

    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid or empty PCM audio")

    return pcm, rate


def get_output_rate(mime_type):
    match = re.search(
        r"rate=(\d+)",
        mime_type or "",
        re.IGNORECASE
    )

    rate = int(match.group(1)) if match else 24000

    if rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported output sample rate: {rate}")

    return rate


# ============================================================
# Safe WebSocket sending
# ============================================================

async def send_json(ws, lock, payload):
    async with lock:
        await ws.send_json(payload)


async def send_binary(ws, lock, payload):
    async with lock:
        await ws.send_bytes(payload)


# ============================================================
# Gemini Live response receiver
# The same session is reused for every visitor turn.
# ============================================================

async def receive_live_turn(
    session,
    websocket,
    send_lock,
    visitor_active
):
    audio_started = False
    audio_ended = False
    audio_bytes_sent = 0
    pending_byte = b""
    output_rate = 24000
    transcript_parts = []
    
    # قائمة انتظار لفصل سرعة Gemini عن سرعة إرسال ESP32
    audio_queue = asyncio.Queue()

    # عامل خلفية يقوم بإرسال الصوت بهدوء للروبوت
    async def pacer_worker():
        bytes_pushed = 0
        prebuffer = 24000  # إرسال أول نصف ثانية بسرعة لملء ذاكرة الروبوت
        
        while visitor_active.is_set():
            chunk = await audio_queue.get()
            if chunk == b"END":
                break
                
            await send_binary(websocket, send_lock, chunk)
            bytes_pushed += len(chunk)
            
            if bytes_pushed > prebuffer:
                duration = (len(chunk) / 2) / output_rate
                # تنظيم التدفق دون إيقاف Gemini
                await asyncio.sleep(duration * 0.90) 

    pacer_task = asyncio.create_task(pacer_worker())

    try:
        async for response in session.receive():
            content = response.server_content
            if content is None:
                continue

            transcription = getattr(content, "output_transcription", None)
            if transcription and transcription.text:
                transcript_parts.append(transcription.text)

            model_turn = getattr(content, "model_turn", None)
            if model_turn:
                for part in model_turn.parts or []:
                    inline = getattr(part, "inline_data", None)
                    if not inline or not inline.data:
                        continue

                    raw_audio = inline.data
                    if isinstance(raw_audio, str):
                        raw_audio = base64.b64decode(raw_audio)
                    raw_audio = bytes(raw_audio)

                    mime_type = getattr(inline, "mime_type", "") or ""
                    output_rate = get_output_rate(mime_type)

                    raw_audio = pending_byte + raw_audio
                    usable_length = len(raw_audio) & ~1
                    pending_byte = raw_audio[usable_length:]
                    raw_audio = raw_audio[:usable_length]

                    if not raw_audio:
                        continue
                    if not visitor_active.is_set():
                        continue

                    if not audio_started:
                        await send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "audio_start",
                                "format": "pcm_s16le",
                                "sample_rate": output_rate,
                                "channels": 1,
                                "bits": 16
                            }
                        )
                        audio_started = True

                    for offset in range(0, len(raw_audio), AUDIO_CHUNK_SIZE):
                        if not visitor_active.is_set():
                            break
                        chunk = raw_audio[offset:offset + AUDIO_CHUNK_SIZE]
                        if chunk:
                            # نضع الصوت في القائمة بدلاً من إرساله فوراً
                            audio_queue.put_nowait(chunk)
                            audio_bytes_sent += len(chunk)

            if getattr(content, "interrupted", False):
                logger.info("Gemini turn interrupted")
                break
            if getattr(content, "turn_complete", False):
                break
    finally:
        # إغلاق مهام الخلفية بأمان
        audio_queue.put_nowait(b"END")
        await pacer_task

    if audio_started and not audio_ended:
        await send_json(websocket, send_lock, {"type": "audio_end"})
        audio_ended = True

    transcript = "".join(transcript_parts).strip()
    if transcript and visitor_active.is_set():
        await send_json(
            websocket,
            send_lock,
            {"type": "transcript_answer", "text": transcript}
        )

    logger.info(
        "Live turn finished: %d audio bytes at %d Hz",
        audio_bytes_sent,
        output_rate
    )

    return audio_bytes_sent


# ============================================================
# Gemini turn sender
# ============================================================

async def run_text_turn(
    session,
    websocket,
    send_lock,
    visitor_active,
    prompt
):
    await session.send_realtime_input(text=prompt)

    return await receive_live_turn(
        session,
        websocket,
        send_lock,
        visitor_active
    )


async def run_audio_turn(
    session,
    websocket,
    send_lock,
    visitor_active,
    pcm,
    sample_rate
):
    if not pcm:
        raise ValueError("Empty audio recording")

    if len(pcm) % 2:
        raise ValueError("PCM16 data has an incomplete sample")

    # The client controls speech boundaries because
    # automatic activity detection is disabled.
    await session.send_realtime_input(
        activity_start=types.ActivityStart()
    )

    chunk_size = 4096
    mime_type = f"audio/pcm;rate={sample_rate}"

    for offset in range(0, len(pcm), chunk_size):

        if not visitor_active.is_set():
            break

        chunk = pcm[offset:offset + chunk_size]

        await session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type=mime_type
            )
        )

    await session.send_realtime_input(
        activity_end=types.ActivityEnd()
    )

    if not visitor_active.is_set():
        return 0

    return await receive_live_turn(
        session,
        websocket,
        send_lock,
        visitor_active
    )


# ============================================================
# Persistent visitor session worker
# One connection for the whole visit.
# ============================================================

async def visitor_worker(
    websocket,
    send_lock,
    queue,
    visitor_active
):
    if client is None:
        await send_json(
            websocket,
            send_lock,
            {
                "type": "error",
                "message": "GEMINI_API_KEY is missing on the server."
            }
        )
        return

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=TTS_VOICE
                )
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            )
        )
    )

    try:
        async with client.aio.live.connect(
            model=LIVE_MODEL,
            config=config
        ) as session:

            logger.info("Persistent visitor session opened")

            await send_json(
                websocket,
                send_lock,
                {
                    "type": "session",
                    "status": "started"
                }
            )

            while True:
                job = await queue.get()

                job_type = job.get("type")

                try:
                    if job_type == "welcome":

                        visitor_active.set()

                        await run_text_turn(
                            session,
                            websocket,
                            send_lock,
                            visitor_active,
                            (
                                "A visitor has arrived. "
                                "Welcome them warmly in Arabic. "
                                "Introduce yourself briefly as RIWEXA, "
                                "their Saudi cultural guide. "
                                "Then remain ready for their questions. "
                                "Follow the visitor's spoken language "
                                "in all later replies."
                            )
                        )

                    elif job_type == "audio":

                        if visitor_active.is_set():
                            await run_audio_turn(
                                session,
                                websocket,
                                send_lock,
                                visitor_active,
                                job["pcm"],
                                job["rate"]
                            )

                    elif job_type == "text":

                        if visitor_active.is_set():
                            await run_text_turn(
                                session,
                                websocket,
                                send_lock,
                                visitor_active,
                                job["message"]
                            )

                    elif job_type == "goodbye":

                        # The ESP32 has confirmed the visitor left.
                        # The same Gemini session retains the context.
                        visitor_active.set()

                        await run_text_turn(
                            session,
                            websocket,
                            send_lock,
                            visitor_active,
                            (
                                "The visitor is leaving now. "
                                "Give them a brief, warm goodbye. "
                                "Use the language they most recently spoke."
                            )
                        )

                        await send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "session",
                                "status": "ended"
                            }
                        )

                        break

                except Exception as exc:
                    logger.exception("Visitor turn failed")

                    with suppress(Exception):
                        await send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "error",
                                "message": str(exc)[:250]
                            }
                        )

                if job_type == "goodbye":
                    break

    except asyncio.CancelledError:
        raise

    except Exception:
        logger.exception("Persistent Gemini session failed")

        with suppress(Exception):
            await send_json(
                websocket,
                send_lock,
                {
                    "type": "error",
                    "message": "تعذر الاتصال بخدمة Gemini Live."
                }
            )

    finally:
        visitor_active.clear()
        logger.info("Persistent visitor session closed")


# ============================================================
# WebSocket endpoint
# Receives ESP32 messages while Gemini is responding.
# ============================================================

@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    logger.info("ESP32 connected")

    send_lock = asyncio.Lock()

    pcm_buffer = bytearray()
    pcm_rate = 16000
    receiving_pcm = False

    visitor_active = asyncio.Event()
    visitor_queue = asyncio.Queue()

    worker_task = None

    async def start_session():
        nonlocal worker_task

        if worker_task and not worker_task.done():
            return

        visitor_active.set()

        worker_task = asyncio.create_task(
            visitor_worker(
                websocket,
                send_lock,
                visitor_queue,
                visitor_active
            )
        )

    try:
        await send_json(
            websocket,
            send_lock,
            {
                "type": "ready",
                "protocol": 1
            }
        )

        while True:

            packet = await websocket.receive()

            if packet.get("type") == "websocket.disconnect":
                break

            # ------------------------------------------------
            # Binary audio
            # ------------------------------------------------

            if packet.get("bytes") is not None:

                if not receiving_pcm:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Unexpected binary audio data."
                        }
                    )
                    continue

                chunk = packet["bytes"]

                if len(pcm_buffer) + len(chunk) > MAX_PCM_BYTES:

                    pcm_buffer.clear()
                    receiving_pcm = False

                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Audio recording exceeded the limit."
                        }
                    )
                    continue

                pcm_buffer.extend(chunk)
                continue

            # ------------------------------------------------
            # JSON messages
            # ------------------------------------------------

            raw_text = packet.get("text")

            if raw_text is None:
                continue

            try:
                message = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "error",
                        "message": "Invalid JSON."
                    }
                )
                continue

            msg_type = message.get("type")

            # PING
            if msg_type == "ping":
                await send_json(
                    websocket,
                    send_lock,
                    {"type": "pong"}
                )
                continue

            # VISITOR ARRIVED
            if (
                msg_type == "event"
                and message.get("event") == "visitor_arrived"
            ):

                if worker_task and not worker_task.done():
                    logger.warning("Arrival received while session active")
                    continue

                visitor_active.set()
                await start_session()

                await visitor_queue.put(
                    {"type": "welcome"}
                )
                continue

            # VISITOR LEFT
            if (
                msg_type == "event"
                and message.get("event") == "visitor_left"
            ):

                if worker_task and not worker_task.done():

                    visitor_active.clear()

                    # The worker finishes consuming the current
                    # Live turn, discards further audio, then says goodbye.
                    await visitor_queue.put(
                        {"type": "goodbye"}
                    )

                continue

            # AUDIO START
            if msg_type == "audio_start":

                if receiving_pcm:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "A recording is already in progress."
                        }
                    )
                    continue

                if message.get("format") != "pcm_s16le":
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Unsupported audio format."
                        }
                    )
                    continue

                if message.get("channels") != 1:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Audio must be mono."
                        }
                    )
                    continue

                try:
                    pcm_rate = int(
                        message.get("sample_rate", 16000)
                    )
                except (TypeError, ValueError):
                    pcm_rate = 0

                if pcm_rate not in SUPPORTED_SAMPLE_RATES:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Unsupported sample rate."
                        }
                    )
                    continue

                if not visitor_active.is_set():
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "No active visitor session."
                        }
                    )
                    continue

                pcm_buffer.clear()
                receiving_pcm = True
                continue

            # AUDIO END
            if msg_type == "audio_end":

                if not receiving_pcm:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "No recording is active."
                        }
                    )
                    continue

                receiving_pcm = False

                captured = bytes(pcm_buffer)
                pcm_buffer.clear()

                if len(captured) < pcm_rate:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Recording is too short."
                        }
                    )
                    continue

                if len(captured) % 2:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Incomplete PCM16 sample."
                        }
                    )
                    continue

                if visitor_active.is_set() and worker_task and not worker_task.done():

                    await visitor_queue.put(
                        {
                            "type": "audio",
                            "pcm": captured,
                            "rate": pcm_rate
                        }
                    )
                continue

            # TEXT QUESTION
            if msg_type == "text":

                question = str(
                    message.get("message", "")
                ).strip()

                if not question or len(question) > MAX_TEXT_LENGTH:
                    await send_json(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Invalid text question."
                        }
                    )
                    continue

                if visitor_active.is_set() and worker_task and not worker_task.done():
                    await visitor_queue.put(
                        {
                            "type": "text",
                            "message": question
                        }
                    )
                continue

            await send_json(
                websocket,
                send_lock,
                {
                    "type": "error",
                    "message": "Unknown message type."
                }
            )

    except WebSocketDisconnect:
        logger.info("ESP32 disconnected")

    except Exception:
        logger.exception("WebSocket handler failed")

    finally:
        visitor_active.clear()

        if worker_task and not worker_task.done():
            worker_task.cancel()

            with suppress(asyncio.CancelledError):
                await worker_task

        logger.info("ESP32 WebSocket closed")


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup_event():

    if client is None:
        logger.error("GEMINI_API_KEY is missing")
    else:
        logger.info("Gemini client configured")

    logger.info("RIWEXA server started")
