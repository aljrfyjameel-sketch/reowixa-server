import os
import json
import io
import wave
import asyncio
import logging
import websockets

from google import genai
from google.genai import types

# ---------------- CONFIG ----------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RIWEXA")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

CHAT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"

MAX_AUDIO_BYTES = 1_500_000
AUDIO_CHUNK_SIZE = 4096

SYSTEM_PROMPT = """
أنت RIWEXA، مساعد زوار ذكي في المتاحف والمعارض والمواقع الثقافية.
أجب بالعربية الواضحة والمختصرة والودودة.
ساعد الزائر في فهم المعروضات والاتجاهات والخدمات وإتاحة الوصول.
لا تخترع معلومات عن الموقع إذا لم تكن متأكدًا منها.
اجعل إجابتك مناسبة لشاشة LCD صغيرة.
"""


# ---------------- GEMINI TEXT ----------------

def make_answer(prompt):
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=180,
        ),
    )

    return (
        response.text or
        "عذرًا، لم أتمكن من إعداد إجابة."
    ).strip()


# ---------------- GEMINI AUDIO INPUT ----------------

def understand_audio(audio_bytes):
    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Incoming audio must be a WAV file")

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav",
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            "استمع إلى كلام الزائر وأجب عن طلبه مباشرة. "
            "إذا لم يكن الكلام واضحًا فاطلب منه إعادته.",
            audio_part,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=180,
        ),
    )

    return (
        response.text or
        "لم أتمكن من فهم الصوت، حاول مرة أخرى."
    ).strip()


# ---------------- GEMINI TTS ----------------

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

    for part in response.parts:
        if part.inline_data and part.inline_data.data:
            wav_bytes = part.inline_data.data

            # Gemini unary TTS returns WAV.
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                pcm_data = wav.readframes(wav.getnframes())

            if channels != 1 or sample_width != 2:
                raise ValueError(
                    "Unexpected TTS audio format; expected mono 16-bit PCM"
                )

            return pcm_data, sample_rate

    raise RuntimeError("Gemini returned no audio data")


# ---------------- WEBSOCKET HELPERS ----------------

async def send_json(ws, data):
    await ws.send(
        json.dumps(data, ensure_ascii=False)
    )


async def send_audio(ws, pcm_data, sample_rate):
    await send_json(ws, {
        "type": "audio_start",
        "format": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": 1,
        "bits_per_sample": 16,
        "total_bytes": len(pcm_data),
    })

    # Send small binary WebSocket messages.
    for offset in range(0, len(pcm_data), AUDIO_CHUNK_SIZE):
        chunk = pcm_data[offset:offset + AUDIO_CHUNK_SIZE]
        await ws.send(chunk)

    await send_json(ws, {
        "type": "audio_end"
    })


# ---------------- RESPONSE PIPELINE ----------------

async def answer_and_speak(ws, user_text):
    await send_json(ws, {
        "type": "status",
        "message": "أفكر في إجابتك..."
    })

    answer = await asyncio.to_thread(
        make_answer,
        user_text
    )

    # Text response first, for the LCD.
    await send_json(ws, {
        "type": "response",
        "message": answer
    })

    await send_json(ws, {
        "type": "status",
        "message": "أجهز الرد الصوتي..."
    })

    pcm_data, sample_rate = await asyncio.to_thread(
        make_speech,
        answer
    )

    await send_audio(ws, pcm_data, sample_rate)


async def handle_audio(ws, audio_bytes):
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        await send_json(ws, {
            "type": "error",
            "message": "حجم التسجيل أكبر من الحد المسموح"
        })
        return

    if len(audio_bytes) < 44 or not audio_bytes.startswith(b"RIFF"):
        await send_json(ws, {
            "type": "error",
            "message": "أرسل تسجيل WAV صحيحًا"
        })
        return

    await send_json(ws, {
        "type": "status",
        "message": "أستمع إلى سؤالك..."
    })

    try:
        user_text = await asyncio.to_thread(
            understand_audio,
            audio_bytes
        )

        await answer_and_speak(ws, user_text)

    except Exception:
        log.exception("Audio processing failed")
        await send_json(ws, {
            "type": "error",
            "message": "حدث خطأ أثناء معالجة الصوت"
        })


# ---------------- CONNECTION HANDLER ----------------

async def handler(ws):
    log.info("RIWEXA connected")

    try:
        async for message in ws:

            # Binary message = complete WAV recording.
            if isinstance(message, bytes):
                await handle_audio(ws, message)
                continue

            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                await send_json(ws, {
                    "type": "error",
                    "message": "صيغة الرسالة غير صحيحة"
                })
                continue

            message_type = data.get("type")

            if message_type == "ping":
                await send_json(ws, {
                    "type": "pong"
                })
                continue

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

                try:
                    await answer_and_speak(ws, user_text)

                except Exception:
                    log.exception("Text processing failed")
                    await send_json(ws, {
                        "type": "error",
                        "message": "تعذر إعداد الرد"
                    })

            else:
                await send_json(ws, {
                    "type": "error",
                    "message": "نوع الرسالة غير مدعوم"
                })

    except websockets.exceptions.ConnectionClosed:
        log.info("RIWEXA disconnected")


# ---------------- SERVER ----------------

async def main():
    port = int(os.environ.get("PORT", "10000"))

    async with websockets.serve(
        handler,
        "0.0.0.0",
        port,
        ping_interval=20,
        ping_timeout=30,
        max_size=MAX_AUDIO_BYTES,
    ):
        log.info("RIWEXA server listening on port %s", port)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
