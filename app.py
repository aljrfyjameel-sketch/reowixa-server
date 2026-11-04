
import os
import json
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

# -----------------------------
# RIWEXA SERVER CONFIGURATION
# -----------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RIWEXA")

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

client = genai.Client(api_key=API_KEY)

CHAT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"

SYSTEM_PROMPT = """
أنت RIWEXA، مساعد زوار ذكي في المتاحف والمعارض والمواقع الثقافية.

أجب بالعربية الواضحة والمختصرة.
كن ودودًا ودقيقًا.
ساعد الزوار في فهم المعروضات والاتجاهات والخدمات.
راعِ احتياجات المكفوفين وضعاف البصر وكبار السن.
إذا لم تعرف معلومة عن الموقع فلا تخترعها.
اجعل إجابتك مناسبة لشاشة LCD صغيرة.
"""

app = FastAPI(title="RIWEXA Server")


# -----------------------------
# HTTP HEALTH CHECK
# -----------------------------

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {
        "status": "ok",
        "service": "RIWEXA",
        "websocket": "ready"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# -----------------------------
# JSON COMMUNICATION
# -----------------------------

async def send_json(ws: WebSocket, data):
    await ws.send_text(
        json.dumps(data, ensure_ascii=False)
    )


# -----------------------------
# GEMINI TEXT RESPONSE
# -----------------------------

def make_answer(prompt):

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=220,
        ),
    )

    return (
        response.text or
        "عذرًا، لم أتمكن من إعداد إجابة."
    ).strip()


# -----------------------------
# GEMINI AUDIO UNDERSTANDING
# -----------------------------

def understand_audio(audio_bytes):

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav",
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            (
                "استمع إلى كلام الزائر وأجب عن طلبه مباشرة. "
                "إذا لم يكن الكلام واضحًا فاطلب منه إعادته."
            ),
            audio_part,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=220,
        ),
    )

    return (
        response.text or
        "لم أتمكن من فهم الصوت، حاول مرة أخرى."
    ).strip()


# -----------------------------
# GEMINI TEXT TO SPEECH
# -----------------------------

def make_speech(text):

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
            ),
        ),
    )

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data

    return None


# -----------------------------
# AUDIO PROCESSING
# -----------------------------

async def handle_audio(ws, audio_bytes):

    if len(audio_bytes) < 44:
        await send_json(ws, {
            "type": "error",
            "message": "ملف الصوت غير مكتمل"
        })
        return

    await send_json(ws, {
        "type": "status",
        "message": "أستمع إليك..."
    })

    try:

        answer = await asyncio.to_thread(
            understand_audio,
            audio_bytes
        )

        await send_json(ws, {
            "type": "response",
            "message": answer
        })

        await send_json(ws, {
            "type": "status",
            "message": "أجهز الرد الصوتي..."
        })

        audio_reply = await asyncio.to_thread(
            make_speech,
            answer
        )

        if audio_reply:

            await ws.send_bytes(audio_reply)

            await send_json(ws, {
                "type": "status",
                "message": "اكتمل الرد"
            })

        else:

            await send_json(ws, {
                "type": "error",
                "message": "تعذر توليد الرد الصوتي"
            })

    except Exception:

        log.exception("Audio processing failed")

        await send_json(ws, {
            "type": "error",
            "message": "حدث خطأ أثناء معالجة الصوت"
        })


# -----------------------------
# WEBSOCKET ENDPOINT
# -----------------------------

@app.websocket("/")
async def websocket_handler(ws: WebSocket):

    await ws.accept()

    log.info("RIWEXA robot connected")

    try:

        while True:

            packet = await ws.receive()

            if packet.get("type") == "websocket.disconnect":
                break

            # Incoming audio
            audio_bytes = packet.get("bytes")

            if audio_bytes is not None:

                await handle_audio(ws, audio_bytes)
                continue

            # Incoming text / JSON
            message = packet.get("text")

            if not message:
                continue

            try:

                data = json.loads(message)

            except json.JSONDecodeError:

                await send_json(ws, {
                    "type": "error",
                    "message": "صيغة الرسالة غير صحيحة"
                })

                continue

            message_type = data.get("type")

            # Ping / Pong
            if message_type == "ping":

                await send_json(ws, {
                    "type": "pong"
                })

                continue

            # Text question
            if message_type == "text":

                user_text = str(
                    data.get("message", "")
                ).strip()[:1000]

                if not user_text:

                    await send_json(ws, {
                        "type": "error",
                        "message": "اكتب سؤالًا أولًا"
                    })

                    continue

                await send_json(ws, {
                    "type": "status",
                    "message": "أفكر..."
                })

                try:

                    answer = await asyncio.to_thread(
                        make_answer,
                        user_text
                    )

                    await send_json(ws, {
                        "type": "response",
                        "message": answer
                    })

                except Exception:

                    log.exception("Text processing failed")

                    await send_json(ws, {
                        "type": "error",
                        "message": "تعذر الاتصال بخدمة الذكاء الاصطناعي"
                    })

                continue

            await send_json(ws, {
                "type": "error",
                "message": "نوع الرسالة غير مدعوم"
            })

    except WebSocketDisconnect:

        log.info("RIWEXA robot disconnected")

    except Exception:

        log.exception("WebSocket connection error")

        try:
            await ws.close()
        except Exception:
            pass


# -----------------------------
# STARTUP
# -----------------------------

@app.on_event("startup")
async def startup_event():
    log.info("RIWEXA server started successfully")
