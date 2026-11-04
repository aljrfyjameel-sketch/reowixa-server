
import os
import io
import wave
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

# ==============================
# RIWEXA - Saudi Cultural Robot
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA")

app = FastAPI(title="RIWEXA Saudi Cultural Robot")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logger.warning("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY) if API_KEY else None

CHAT_MODEL = "gemini-3.5-flash-lite"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

SYSTEM_PROMPT = """
أنت RIWEXA، روبوت ثقافي سعودي ذكي.

هويتك:
أنت سفير رقمي للثقافة السعودية، تساعد الزوار
على اكتشاف تاريخ المملكة وتراثها وهويتها الثقافية.

أهدافك:
1. التعريف بتاريخ المملكة العربية السعودية.
2. التعريف بالمناطق السعودية وتراثها.
3. شرح المواقع التاريخية والأثرية.
4. التعريف بالعادات والتقاليد والفنون الشعبية.
5. التعريف بالحرف اليدوية والأزياء والأكلات السعودية.
6. تقديم معلومات عن المعالم والمتاحف والمواقع الثقافية.
7. مساعدة الزوار من مختلف اللغات والقدرات.
8. تقديم تجربة ثقافية تفاعلية وممتعة.

أسلوبك:
- تحدث بلغة الزائر.
- استخدم لغة واضحة وودودة.
- اجعل إجاباتك مختصرة ومفيدة.
- قدم أمثلة ثقافية عند الحاجة.
- احترم جميع الثقافات.
- لا تخترع معلومات تاريخية أو حقائق.
- إذا لم تعرف الإجابة، وضح ذلك.
- لا تدّعِ أنك ترى الزائر أو المكان أو القطعة الأثرية
  دون وجود صورة أو بيانات متاحة لك.

تذكر:
أنت RIWEXA، الروبوت الثقافي السعودي.
مهمتك جعل الثقافة السعودية أقرب إلى الجميع.
"""

# ==============================
# HTTP HEALTH CHECK
# ==============================

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


# ==============================
# TEXT AI
# ==============================

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


# ==============================
# AUDIO UNDERSTANDING
# ==============================

def understand_audio(audio_bytes: bytes) -> str:
    if not client:
        raise RuntimeError("Gemini API key is not configured")

    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Audio must be a valid WAV file")

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            SYSTEM_PROMPT,
            "استمع إلى تسجيل الزائر، وافهم سؤاله، ثم أجب عنه بالعربية أو بلغة السؤال.",
            audio_part
        ]
    )

    if not response.text:
        return "عذراً، لم أتمكن من فهم التسجيل."

    return response.text.strip()


# ==============================
# TEXT TO SPEECH
# ==============================

def make_speech(text: str) -> bytes:
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
            if part.inline_data and part.inline_data.data:
                audio = part.inline_data.data
                mime = part.inline_data.mime_type or ""

                # إذا كانت البيانات WAV جاهزة
                if audio.startswith(b"RIFF"):
                    return audio

                # Gemini TTS عادة يعيد PCM خاماً.
                # نغلفه في WAV: 24kHz / 16-bit / mono
                if "L16" in mime or "pcm" in mime.lower() or not mime:
                    wav_buffer = io.BytesIO()

                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(24000)
                        wav_file.writeframes(audio)

                    return wav_buffer.getvalue()

                return audio

    raise RuntimeError("No audio returned by Gemini")


# ==============================
# WEBSOCKET
# ==============================

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info("RIWEXA client connected")

    try:
        while True:

            data = await websocket.receive()

            # --------------------------
            # Binary audio from ESP32
            # --------------------------

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

                await websocket.send_json({
                    "type": "status",
                    "message": "أستمع إلى سؤالك..."
                })

                try:
                    answer = await asyncio.to_thread(
                        understand_audio,
                        audio_bytes
                    )

                    await websocket.send_json({
                        "type": "response",
                        "message": answer
                    })

                    speech = await asyncio.to_thread(
                        make_speech,
                        answer
                    )

                    await websocket.send_bytes(speech)

                except Exception as error:
                    logger.exception("Audio processing failed")

                    await websocket.send_json({
                        "type": "error",
                        "message": "تعذر معالجة الصوت. حاول مرة أخرى."
                    })

            # --------------------------
            # Text / JSON messages
            # --------------------------

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

                    await websocket.send_json({
                        "type": "status",
                        "message": "أبحث عن الإجابة..."
                    })

                    try:
                        answer = await asyncio.to_thread(
                            generate_answer,
                            user_message
                        )

                        await websocket.send_json({
                            "type": "response",
                            "message": answer
                        })

                        speech = await asyncio.to_thread(
                            make_speech,
                            answer
                        )

                        await websocket.send_bytes(speech)

                    except Exception:
                        logger.exception("Text processing failed")

                        await websocket.send_json({
                            "type": "error",
                            "message": "تعذر الحصول على الإجابة من الذكاء الاصطناعي."
                        })

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
    logger.info("RIWEXA Saudi Cultural Robot started successfully")
