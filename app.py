
import os
import json
import asyncio
import logging
import websockets

from google import genai
from google.genai import types

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
أجب بالعربية الواضحة والمختصرة، وكن ودودًا ودقيقًا.
ساعد الزائر في فهم المعروضات والاتجاهات والخدمات وإتاحة الوصول.
إذا لم تكن تعرف معلومة عن الموقع، فلا تخترعها.
اجعل الرد مناسبًا للعرض على شاشة صغيرة.
"""

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
    return (response.text or "عذرًا، لم أتمكن من إعداد إجابة.").strip()

def understand_audio(audio_bytes):
    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav",
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            "استمع إلى كلام الزائر وأجب عن طلبه مباشرة. إذا لم يكن الكلام واضحًا فاطلب منه إعادته.",
            audio_part,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=220,
        ),
    )
    return (response.text or "لم أتمكن من فهم الصوت، حاول مرة أخرى.").strip()

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
            return part.inline_data.data

    return None

async def send_json(ws, data):
    await ws.send(json.dumps(data, ensure_ascii=False))

async def handle_audio(ws, audio_bytes):
    if len(audio_bytes) < 44:
        await send_json(ws, {
            "type": "error",
            "message": "ملف الصوت غير مكتمل"
        })
        return

    await send_json(ws, {"type": "status", "message": "أحلل الصوت..."})

    try:
        answer = await asyncio.to_thread(understand_audio, audio_bytes)

        await send_json(ws, {
            "type": "response",
            "message": answer
        })

        await send_json(ws, {"type": "status", "message": "أجهز الرد الصوتي..."})

        audio_reply = await asyncio.to_thread(make_speech, answer)

        if audio_reply:
            await ws.send(audio_reply)
        else:
            await send_json(ws, {
                "type": "error",
                "message": "وصل الرد النصي لكن تعذر توليد الصوت"
            })

    except Exception:
        log.exception("Audio processing failed")
        await send_json(ws, {
            "type": "error",
            "message": "حدث خطأ أثناء معالجة الصوت"
        })

async def handler(ws):
    log.info("RIWEXA connected")

    try:
        async for message in ws:
            if isinstance(message, bytes):
                await handle_audio(ws, message)
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await send_json(ws, {
                    "type": "error",
                    "message": "صيغة الرسالة غير صحيحة"
                })
                continue

            if data.get("type") == "ping":
                await send_json(ws, {"type": "pong"})
                continue

            if data.get("type") == "text":
                user_text = str(data.get("message", "")).strip()[:1000]
                if not user_text:
                    await send_json(ws, {
                        "type": "error",
                        "message": "اكتب سؤالًا أولًا"
                    })
                    continue

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

    except websockets.exceptions.ConnectionClosed:
        log.info("RIWEXA disconnected")

async def main():
    port = int(os.environ.get("PORT", "10000"))

    async with websockets.serve(
        handler,
        "0.0.0.0",
        port,
        ping_interval=20,
        ping_timeout=30,
        max_size=2 * 1024 * 1024,
    ):
        log.info("RIWEXA server listening on port %s", port)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
