
import os
import io
import re
import wave
import json
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

# ==========================================
# RIWEXA - Saudi Cultural Robot
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA")

app = FastAPI(title="RIWEXA Saudi Cultural Robot")

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
if TTS_MODEL == "gemini-3.1-flash-tts-preview":
    TTS_MODEL = "gemini-3.8-flash-lite-tts"  # Migrate legacy Render env settings automatically.
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
MAX_PCM_BYTES = 320_000  # ten seconds, 16 kHz, mono PCM16

SYSTEM_PROMPT = """
أنت RIWEXA، روبوت ثقافي سعودي ذكي.

أنت سفير رقمي للثقافة السعودية، تساعد الزوار
على اكتشاف تاريخ المملكة وتراثها وهويتها الثقافية.

أهدافك:
1. التعريف بتاريخ المملكة العربية السعودية.
2. التعريف بالمناطق السعودية وتراثها.
3. شرح المواقع التاريخية والأثرية.
4. التعريف بالعادات والتقاليد والفنون الشعبية.
5. التعريف بالحرف اليدوية والأزياء والأكلات السعودية.
6. التعريف بالمعالم والمتاحف والمواقع الثقافية.
7. مساعدة الزوار من مختلف اللغات والقدرات.

أسلوبك:
- تحدث بلغة الزائر.
- استخدم لغة واضحة وودودة.
- اجعل إجاباتك مختصرة ومفيدة.
- احترم جميع الثقافات.
- لا تخترع معلومات تاريخية.
- إذا لم تعرف الإجابة، وضح ذلك.
- لا تدّعِ أنك ترى الزائر أو المكان دون بيانات.

أنت RIWEXA، الروبوت الثقافي السعودي.
مهمتك جعل الثقافة السعودية أقرب إلى الجميع.
"""


# ==========================================
# HTTP
# ==========================================

@app.get("/")
async def home():
    return {
        "status": "ok",
        "service": "RIWEXA",
        "chat": "ready",
        "voice": "ready",
        "websocket": "ready"
    }


@app.head("/")
async def health_head():
    return JSONResponse(content=None, status_code=200)


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ==========================================
# Gemini text response
# ==========================================

def generate_answer(message: str) -> str:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            SYSTEM_PROMPT,
            f"رسالة الزائر:\n{message}"
        ]
    )

    answer = response.text

    if not answer:
        return "عذراً، لم أتمكن من إعداد إجابة الآن."

    return answer.strip()


# ==========================================
# Understand incoming WAV
# ==========================================

def understand_audio(audio_bytes: bytes) -> str:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Invalid WAV file")

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono")

        if wav.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit PCM")

        if wav.getframerate() not in (16000, 24000):
            raise ValueError("Audio must be 16000 or 24000 Hz")

        if wav.getnframes() == 0:
            raise ValueError("Audio is empty")

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            SYSTEM_PROMPT,
            (
                "استمع إلى تسجيل الزائر وافهم سؤاله. "
                "أجب بلغة السؤال نفسه، بإجابة واضحة ومختصرة."
            ),
            audio_part
        ]
    )

    if not response.text:
        return "عذراً، لم أتمكن من فهم التسجيل."

    return response.text.strip()


# ==========================================
# Gemini Text To Speech
# ==========================================

def make_speech(text: str) -> tuple[bytes, int]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE
                    )
                )
            )
        )
    )

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = part.inline_data

            if not inline or not inline.data:
                continue

            audio = inline.data
            mime = inline.mime_type or ""

            # Provider returned WAV
            if audio.startswith(b"RIFF"):
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    if wav.getnchannels() != 1:
                        raise ValueError("TTS WAV must be mono")

                    if wav.getsampwidth() != 2:
                        raise ValueError("TTS WAV must be 16-bit PCM")

                    rate = wav.getframerate()
                    pcm = wav.readframes(wav.getnframes())

                return pcm, rate

            # Raw PCM
            rate = 24000
            match = re.search(r"rate=(\d+)", mime, re.IGNORECASE)

            if match:
                rate = int(match.group(1))

            if mime and not (
                "pcm" in mime.lower()
                or "l16" in mime.lower()
            ):
                raise ValueError(f"Unsupported audio MIME: {mime}")

            if len(audio) % 2:
                raise ValueError("Invalid 16-bit PCM length")

            return audio, rate

    raise RuntimeError("Gemini returned no audio")


# ==========================================
# Send PCM to ESP32
# ==========================================

async def send_audio_stream(
    websocket: WebSocket,
    pcm: bytes,
    sample_rate: int
):
    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid PCM audio")

    await websocket.send_json({
        "type": "audio_start",
        "format": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": 1,
        "bits": 16
    })

    # Keep frames small for ESP32
    chunk_size = 2048

    for offset in range(0, len(pcm), chunk_size):
        await websocket.send_bytes(
            pcm[offset:offset + chunk_size]
        )

    await websocket.send_json({
        "type": "audio_end"
    })


# ==========================================
# Speak a fixed message
# Welcome / Goodbye
# ==========================================

async def speak_message(
    websocket: WebSocket,
    message: str
):
    await websocket.send_json({
        "type": "status",
        "message": "RIWEXA تتحدث..."
    })

    pcm, rate = await asyncio.to_thread(
        make_speech,
        message
    )

    await send_audio_stream(
        websocket,
        pcm,
        rate
    )


# ==========================================
# Low-latency audio-to-audio response
# ==========================================

async def answer_with_live(websocket: WebSocket, wav_bytes: bytes = None, question: str = None) -> bool:
    """Stream a recorded visitor question through Gemini Live and relay its audio.

    Returns False only when Live fails before sending any audio, so the caller can
    fall back to the existing transcription + TTS path.
    """
    if not client:
        return False

    pcm = b""
    sample_rate = 16000
    if wav_bytes is not None:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Live input must be mono PCM16 WAV")
            sample_rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
        if sample_rate not in (16000, 24000) or not pcm:
            raise ValueError("Unsupported or empty Live audio input")
    elif not question:
        return False

    system_instruction = (
        SYSTEM_PROMPT
        + "\n\nأجب بلغة الزائر مباشرة وبإجابة صوتية قصيرة لا تتجاوز جملتين."
    )
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=TTS_VOICE
                )
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    audio_started = False
    transcript_parts = []
    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            if pcm:
                chunk_size = 4096
                mime_type = f"audio/pcm;rate={sample_rate}"
                for offset in range(0, len(pcm), chunk_size):
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm[offset:offset + chunk_size],
                            mime_type=mime_type,
                        )
                    )
                await session.send_realtime_input(audio_stream_end=True)
            else:
                await session.send_client_content(
                    turns=types.Content(
                        role="user", parts=[types.Part.from_text(text=question)]
                    ),
                    turn_complete=True,
                )

            async for response in session.receive():
                content = response.server_content
                if content is None:
                    continue

                transcription = content.output_transcription
                if transcription and transcription.text:
                    transcript_parts.append(transcription.text)

                turn = content.model_turn
                if turn:
                    for part in turn.parts or []:
                        inline = part.inline_data
                        if not inline or not inline.data:
                            continue

                        audio = inline.data
                        if isinstance(audio, str):
                            audio = audio.encode("latin-1")
                        audio = audio[:len(audio) & ~1]
                        if not audio:
                            continue

                        if not audio_started:
                            await websocket.send_json({
                                "type": "audio_start",
                                "format": "pcm_s16le",
                                "sample_rate": 24000,
                                "channels": 1,
                                "bits": 16,
                            })
                            audio_started = True
                        await websocket.send_bytes(audio)

                if content.turn_complete:
                    break

        if not audio_started:
            return False

        answer = "".join(transcript_parts).strip()
        if answer:
            await websocket.send_json({"type": "response", "message": answer})
            await websocket.send_json({"type": "transcript_answer", "text": answer})
        await websocket.send_json({"type": "audio_end"})
        return True

    except Exception as exc:
        provider_status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        logger.exception("Gemini Live audio response failed (%s, %s)", type(exc).__name__, provider_status)
        try:
            await websocket.send_json({
                "type": "diagnostic",
                "source": "live",
                "error_type": type(exc).__name__,
                "status": provider_status,
            })
        except Exception:
            pass
        if audio_started:
            try:
                await websocket.send_json({"type": "audio_end"})
            except Exception:
                pass
            raise
        return False


# ==========================================
# Process visitor question
# ==========================================

async def process_question(
    websocket: WebSocket,
    question: str = None,
    audio_bytes: bytes = None
):
    try:
        await websocket.send_json({
            "type": "status",
            "message": "جارٍ فهم سؤالك..."
        })

        if await answer_with_live(websocket, wav_bytes=audio_bytes, question=question):
            return
        if audio_bytes is not None:
            logger.warning("Falling back to standard audio question path")
            answer = await asyncio.to_thread(understand_audio, audio_bytes)
        else:
            answer = await asyncio.to_thread(generate_answer, question)

        await websocket.send_json({
            "type": "response",
            "message": answer
        })
        await websocket.send_json({
            "type": "transcript_answer",
            "text": answer
        })

        pcm, rate = await asyncio.to_thread(
            make_speech,
            answer
        )

        await send_audio_stream(
            websocket,
            pcm,
            rate
        )

    except Exception as exc:
        provider_status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        logger.exception("Question processing failed (%s, %s)", type(exc).__name__, provider_status)

        await websocket.send_json({
            "type": "error",
            "message": "تعذر معالجة السؤال. حاول مرة أخرى.",
            "diagnostic": {"error_type": type(exc).__name__, "status": provider_status},
        })


# ==========================================
# WebSocket
# ==========================================

@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info("RIWEXA connected")
    pcm_upload = bytearray()
    pcm_rate = 16000
    receiving_pcm = False

    try:
        await websocket.send_json({"type": "ready", "protocol": 1})
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                logger.info("RIWEXA disconnected")
                break

            # ------------------------------
            # Incoming WAV from ESP32
            # ------------------------------

            if data.get("bytes") is not None:
                audio_bytes = data["bytes"]

                # Protocol v1 upload: binary PCM16 chunks between audio_start/audio_end.
                if receiving_pcm:
                    if len(pcm_upload) + len(audio_bytes) > MAX_PCM_BYTES:
                        pcm_upload.clear()
                        receiving_pcm = False
                        await websocket.send_json({
                            "type": "error",
                            "message": "التسجيل أطول من الحد المسموح."
                        })
                        continue
                    pcm_upload.extend(audio_bytes)
                    continue

                # Backwards compatibility with the previous one-frame WAV upload.
                if len(audio_bytes) < 44:
                    await websocket.send_json({
                        "type": "error",
                        "message": "ملف الصوت غير صالح."
                    })
                    continue

                if len(audio_bytes) > 2 * 1024 * 1024:
                    await websocket.send_json({
                        "type": "error",
                        "message": "التسجيل كبير جداً."
                    })
                    continue

                await process_question(
                    websocket,
                    audio_bytes=audio_bytes
                )

            # ------------------------------
            # Incoming JSON
            # ------------------------------

            elif data.get("text") is not None:
                try:
                    message = json.loads(data["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "صيغة الرسالة غير صحيحة."
                    })
                    continue

                message_type = message.get("type")

                # Start a chunked PCM16 upload.
                if message_type == "audio_start":
                    if receiving_pcm:
                        await websocket.send_json({
                            "type": "error",
                            "message": "يوجد تسجيل قيد الإرسال بالفعل."
                        })
                        continue
                    if (
                        message.get("format") != "pcm_s16le"
                        or message.get("channels") != 1
                    ):
                        await websocket.send_json({
                            "type": "error",
                            "message": "تنسيق الصوت غير مدعوم."
                        })
                        continue
                    try:
                        pcm_rate = int(message.get("sample_rate", 16000))
                    except (TypeError, ValueError):
                        pcm_rate = 0
                    if pcm_rate not in (16000, 24000):
                        await websocket.send_json({
                            "type": "error",
                            "message": "معدل العينة غير مدعوم."
                        })
                        continue
                    pcm_upload.clear()
                    receiving_pcm = True
                    continue

                if message_type == "audio_end":
                    if not receiving_pcm:
                        await websocket.send_json({
                            "type": "error",
                            "message": "لا يوجد تسجيل لاستكماله."
                        })
                        continue
                    receiving_pcm = False
                    captured_pcm = bytes(pcm_upload)
                    pcm_upload.clear()
                    if len(captured_pcm) < pcm_rate * 2 // 5:
                        await websocket.send_json({
                            "type": "error",
                            "message": "التسجيل قصير جداً."
                        })
                        continue
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(pcm_rate)
                        wav_file.writeframes(captured_pcm)
                    await process_question(
                        websocket,
                        audio_bytes=wav_buffer.getvalue()
                    )
                    continue

                # Ping
                if message_type == "ping":
                    await websocket.send_json({
                        "type": "pong"
                    })
                    continue

                # Visitor arrived
                if (
                    message_type == "event"
                    and message.get("event") == "visitor_arrived"
                ):
                    language = message.get("language", "ar")

                    if language == "en":
                        greeting = (
                            "Hello and welcome! "
                            "I am RIWEXA, your Saudi cultural guide. "
                            "You can ask me about Saudi history and heritage."
                        )
                    else:
                        greeting = (
                            "أهلاً وسهلاً بكم! "
                            "أنا ريوكسا، دليلكم الثقافي السعودي. "
                            "اسألوني عن تاريخ المملكة وتراثها."
                        )

                    try:
                        await speak_message(websocket, greeting)
                    except Exception:
                        logger.exception("Welcome speech failed")
                        await websocket.send_json({
                            "type": "error",
                            "message": "تعذر تشغيل الترحيب الصوتي."
                        })

                    continue

                # Visitor left
                if (
                    message_type == "event"
                    and message.get("event") == "visitor_left"
                ):
                    language = message.get("language", "ar")

                    if language == "en":
                        goodbye = (
                            "Thank you for visiting. "
                            "We hope you enjoyed discovering Saudi culture. "
                            "Have a wonderful day!"
                        )
                    else:
                        goodbye = (
                            "شكراً لزيارتكم! "
                            "سعدتُ بمشاركتكم التعرف على الثقافة السعودية. "
                            "نتمنى لكم يوماً سعيداً."
                        )

                    try:
                        await speak_message(websocket, goodbye)
                    except Exception:
                        logger.exception("Goodbye speech failed")
                        await websocket.send_json({
                            "type": "error",
                            "message": "تعذر تشغيل رسالة الوداع."
                        })

                    continue

                # Text question
                if message_type == "text":
                    user_message = str(
                        message.get("message", "")
                    ).strip()

                    if not user_message:
                        await websocket.send_json({
                            "type": "error",
                            "message": "اكتب سؤالك أولاً."
                        })
                        continue

                    if len(user_message) > 4000:
                        await websocket.send_json({
                            "type": "error",
                            "message": "الرسالة طويلة جداً."
                        })
                        continue

                    await process_question(
                        websocket,
                        question=user_message
                    )

                    continue

                await websocket.send_json({
                    "type": "error",
                    "message": "نوع الرسالة غير معروف."
                })

    except WebSocketDisconnect:
        logger.info("RIWEXA disconnected")

    except Exception:
        logger.exception("WebSocket connection error")


@app.on_event("startup")
async def startup_event():
    logger.info("RIWEXA Saudi Cultural Robot started")

