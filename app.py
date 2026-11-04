
import os
import io
import re
import wave
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

if not API_KEY:
    logger.warning("GEMINI_API_KEY is missing")

CHAT_MODEL = "gemini-3.5-flash-lite"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

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
8. تقديم تجربة ثقافية تفاعلية وممتعة.

أسلوبك:
- تحدث بلغة الزائر.
- استخدم لغة واضحة وودودة.
- اجعل إجاباتك مختصرة ومفيدة.
- احترم جميع الثقافات.
- لا تخترع معلومات تاريخية أو حقائق.
- إذا لم تعرف الإجابة، وضح ذلك.
- لا تدّعِ أنك ترى الزائر أو المكان دون بيانات متاحة.

تذكر:
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
        "identity": "Saudi Cultural Robot",
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
        raise RuntimeError("Gemini API key is not configured")

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
# Understand incoming WAV audio
# ==========================================

def understand_audio(audio_bytes: bytes) -> str:
    if not client:
        raise RuntimeError("Gemini API key is not configured")

    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Audio must be a valid WAV file")

    # Validate WAV structure
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Input audio must be mono")

        if wav_file.getsampwidth() != 2:
            raise ValueError("Input audio must be 16-bit PCM")

        if wav_file.getframerate() != 16000:
            raise ValueError("Input audio must be 16000 Hz")

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            SYSTEM_PROMPT,
            (
                "استمع إلى تسجيل الزائر، وافهم سؤاله، "
                "ثم أجب بلغة السؤال. اجعل الإجابة واضحة ومختصرة."
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
        raise RuntimeError("Gemini API key is not configured")

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            )
        )
    )

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if not part.inline_data:
                continue

            audio = part.inline_data.data
            mime = part.inline_data.mime_type or ""

            if not audio:
                continue

            # If provider already returned WAV, extract PCM.
            if audio.startswith(b"RIFF"):
                with wave.open(io.BytesIO(audio), "rb") as wf:
                    if wf.getnchannels() != 1:
                        raise ValueError("TTS WAV must be mono")

                    if wf.getsampwidth() != 2:
                        raise ValueError("TTS WAV must be 16-bit PCM")

                    sample_rate = wf.getframerate()
                    pcm = wf.readframes(wf.getnframes())

                return pcm, sample_rate

            # Gemini TTS normally returns raw PCM.
            # Default rate is 24000 Hz if MIME omits it.
            sample_rate = 24000

            match = re.search(r"rate=(\d+)", mime, re.IGNORECASE)
            if match:
                sample_rate = int(match.group(1))

            if mime and not (
                "pcm" in mime.lower()
                or "l16" in mime.lower()
            ):
                raise ValueError(
                    f"Unsupported TTS MIME type: {mime}"
                )

            if len(audio) % 2:
                raise ValueError("PCM data has an incomplete sample")

            return audio, sample_rate

    raise RuntimeError("No audio returned by Gemini")


# ==========================================
# Stream PCM in small WebSocket frames
# ==========================================

async def send_audio_stream(
    websocket: WebSocket,
    pcm: bytes,
    sample_rate: int
):
    if not pcm:
        raise ValueError("Empty TTS audio")

    if len(pcm) % 2:
        raise ValueError("PCM must contain 16-bit samples")

    await websocket.send_json({
        "type": "audio_start",
        "sample_rate": sample_rate,
        "channels": 1,
        "bits": 16
    })

    # Small frames for ESP32 WebSocket client.
    chunk_size = 4096

    for offset in range(0, len(pcm), chunk_size):
        await websocket.send_bytes(
            pcm[offset:offset + chunk_size]
        )

    await websocket.send_json({
        "type": "audio_end"
    })


# ==========================================
# Process a text or audio question
# ==========================================

async def process_question(
    websocket: WebSocket,
    question: str = None,
    audio_bytes: bytes = None
):
    try:
        await websocket.send_json({
            "type": "status",
            "message": "أستمع إلى سؤالك..."
            if audio_bytes else "أبحث عن الإجابة..."
        })

        if audio_bytes is not None:
            answer = await asyncio.to_thread(
                understand_audio,
                audio_bytes
            )
        else:
            answer = await asyncio.to_thread(
                generate_answer,
                question
            )

        await websocket.send_json({
            "type": "response",
            "message": answer
        })

        pcm, sample_rate = await asyncio.to_thread(
            make_speech,
            answer
        )

        await send_audio_stream(
            websocket,
            pcm,
            sample_rate
        )

    except Exception:
        logger.exception("Question processing failed")

        await websocket.send_json({
            "type": "error",
            "message": "تعذر معالجة السؤال. حاول مرة أخرى."
        })


# ==========================================
# WebSocket
# ==========================================

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info("RIWEXA client connected")

    try:
        while True:
            data = await websocket.receive()

            # Binary WAV from ESP32
            if data.get("bytes") is not None:
                audio_bytes = data["bytes"]

                if len(audio_bytes) < 44:
                    await websocket.send_json({
                        "type": "error",
                        "message": "ملف الصوت غير صالح."
                    })
                    continue

                if len(audio_bytes) > 2 * 1024 * 1024:
                    await websocket.send_json({
                        "type": "error",
                        "message": "حجم التسجيل كبير جداً."
                    })
                    continue

                await process_question(
                    websocket,
                    audio_bytes=audio_bytes
                )

            # JSON text / ping
            elif data.get("text") is not None:
                import json

                try:
                    message = json.loads(data["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "صيغة الرسالة غير صحيحة."
                    })
                    continue

                message_type = message.get("type")

                if message_type == "ping":
                    await websocket.send_json({
                        "type": "pong"
                    })
                    continue

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

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "نوع الرسالة غير معروف."
                    })

    except WebSocketDisconnect:
        logger.info("RIWEXA client disconnected")

    except Exception:
        logger.exception("WebSocket connection error")


@app.on_event("startup")
async def startup_event():
    logger.info("RIWEXA Saudi Cultural Robot started")
