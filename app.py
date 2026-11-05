import os
import io
import json
import wave
import base64
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA-Global")

app = FastAPI(title="RIWEXA Global Edition - SAIF 2026")

API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-exp")
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

client = genai.Client(api_key=API_KEY) if API_KEY else None

CULTURAL_KNOWLEDGE_BASE = """
[معالم المملكة وتاريخها - مرجع أساسي لـ RIWEXA في SAIF 2026]
1. الدرعية التاريخية (الرياض): عاصمة الدولة السعودية الأولى، ومسجلّة في اليونسكو (حي الطريف).
2. العلا (المدينة المنورة): تضم موقع الحجر الأثري (مدائن صالح)، أول موقع مسجل في اليونسكو بآثار نبطية منحوتة.
3. منطقة عسير (أبها وخميس مشيط): التراث العمراني، قصر شدا، وفن القط العسيري المسجل في اليونسكو.
4. جدة التاريخية (البلد): الرواشين الخشبية وبوابة الحجاج على البحر الأحمر.
5. قصر المصمك (الرياض): حصن انطلاق ملحمة توحيد المملكة على يد الملك عبد العزيز 1902م.
6. رؤية السعودية 2030: استراتيجية تنويع الاقتصاد والتنمية المستدامة.
"""

GROUNDED_SYSTEM_PROMPT = f"""
You are RIWEXA, an advanced Saudi cultural guide robot representing global innovation at SAIF 2026.

STRICT GROUNDING & BEHAVIOR RULES:
- Answer questions about Saudi history, landmarks, and culture strictly based on the Knowledge Base below.
- Do not invent historical facts or dates. If a fact is missing, state it professionally.
- Speak naturally, fluently, and warmly in Arabic or English based on the visitor.

KNOWLEDGE BASE:
{CULTURAL_KNOWLEDGE_BASE}
"""

@app.get("/")
async def home():
    return {"status": "ok", "service": "RIWEXA Global Edition", "event": "SAIF 2026"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 Connected to Global Server")

    if not client:
        await websocket.send_text(json.dumps({"type": "error", "message": "API Key missing"}))
        await websocket.close()
        return

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=GROUNDED_SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
            )
        )
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            logger.info("Gemini Live Session Established")
            await websocket.send_text(json.dumps({"type": "ready", "protocol": 1}))

            async def upstream_task():
                try:
                    while True:
                        packet = await websocket.receive()
                        if packet.get("type") == "websocket.disconnect":
                            break
                        
                        if packet.get("bytes") is not None:
                            audio_data = packet["bytes"]
                            await session.send_realtime_input(
                                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
                            )
                            continue

                        raw_text = packet.get("text")
                        if raw_text:
                            data = json.loads(raw_text)
                            if data.get("type") == "event" and data.get("event") == "visitor_arrived":
                                await session.send_realtime_input(text="أهلاً بالزائر في معرض SAIF 2026. رحب به بحرارة وعرف عن نفسك باختصار كمرشد سياحي سعودي.")
                            elif data.get("type") == "text":
                                await session.send_realtime_input(text=data.get("message", ""))
                except Exception as e:
                    logger.error(f"Upstream error: {e}")

            async def downstream_task():
                try:
                    async for response in session.receive():
                        server_content = response.server_content
                        if not server_content or not server_content.model_turn:
                            continue

                        for part in server_content.model_turn.parts or []:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                audio_bytes = inline.data
                                if isinstance(audio_bytes, str):
                                    audio_bytes = base64.b64decode(audio_bytes)
                                
                                await websocket.send_json({"type": "audio_start", "sample_rate": 24000})
                                await websocket.send_bytes(bytes(audio_bytes))
                                await websocket.send_json({"type": "audio_end"})
                except Exception as e:
                    logger.error(f"Downstream error: {e}")

            await asyncio.gather(upstream_task(), downstream_task())

    except WebSocketDisconnect:
        logger.info("ESP32 Disconnected")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        logger.info("Connection Closed")
