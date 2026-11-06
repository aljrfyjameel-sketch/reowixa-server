import json
import base64
import asyncio
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# إعداد نظام المراقبة (Logging) عشان تتابع السيرفر من منصة Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RIWEXA_SERVER")

app = FastAPI()

# تفعيل الـ CORS لتسهيل الاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# التأكد من المفتاح
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logger.error("CRITICAL: GEMINI_API_KEY is missing in Render Environment Variables!")

client = genai.Client()

SYSTEM_INSTRUCTION = """You are RIWEXA, a Saudi cultural guide robot. You help visitors discover Saudi history, heritage, regions, traditions, museums, historical sites, landmarks, traditional arts, crafts, clothing and food, Saudi culture and national identity. Automatically identify language. If Arabic answer in Arabic, if English answer in English. If visitor changes language follow latest. Never ask to choose language. Keep conversation context. Be welcoming, clear, friendly. Do not invent historical facts. If you don't know say so honestly. Do not claim to see visitor. Keep answers appropriate for public exhibition. Always answer using native audio."""

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-3.8-live", "gemini_configured": bool(api_key)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 Connected to WebSocket successfully!")
    
    try:
        # الحلقة هذي تخلي السيرفر صامد، لو جمناي قطع الاتصال يرجع يشبك فوراً بدون ما يفصل الروبوت
        while True:
            try:
                logger.info("Connecting to Gemini 3.8 Live API...")
                async with client.aio.live.connect(
                    model="gemini-3.8-live",
                    config=types.LiveConnectConfig(
                        system_instruction=types.Content(parts=[types.Part.from_text(SYSTEM_INSTRUCTION)]),
                        response_modalities=["AUDIO"]
                    )
                ) as session:
                    logger.info("Gemini Live Session Established!")

                    async def receive_from_esp32():
                        while True:
                            data = await websocket.receive_text()
                            msg = json.loads(data)
                            msg_type = msg.get("type")
                            
                            if msg_type == "audio":
                                # إرسال الصوت لجمناي
                                pcm_data = base64.b64decode(msg["data"])
                                await session.send(input=types.LiveClientRealtimeInput(
                                    media_chunks=[types.Blob(
                                        mime_type="audio/pcm;rate=16000",
                                        data=pcm_data
                                    )]
                                ))
                            elif msg_type == "visitor_detected":
                                logger.info("Event: Visitor Detected")
                                await session.send(input="A visitor has just arrived. Greet them warmly and briefly.", end_of_turn=True)
                            elif msg_type == "visitor_left":
                                logger.info("Event: Visitor Left")
                                await session.send(input="The visitor has left. Say a brief, polite goodbye.", end_of_turn=True)

                    async def receive_from_gemini():
                        async for response in session.receive():
                            server_content = response.server_content
                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        # تشفير الصوت وإرساله للـ ESP32
                                        b64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                        await websocket.send_text(json.dumps({
                                            "type": "audio", 
                                            "data": b64_audio
                                        }))
                                    elif part.text:
                                        # إرسال النص لعرضه على شاشة LCD
                                        await websocket.send_text(json.dumps({
                                            "type": "transcript",
                                            "text": part.text
                                        }))

                    # تشغيل الاستقبال من الروبوت والاستقبال من جمناي في نفس الوقت
                    recv_esp_task = asyncio.create_task(receive_from_esp32())
                    recv_gem_task = asyncio.create_task(receive_from_gemini())
                    
                    # ننتظر لحد ما وحدة من المهام توقف (بسبب خطأ أو انقطاع)
                    done, pending = await asyncio.wait(
                        [recv_esp_task, recv_gem_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # نلغي المهمة الثانية المعلقة
                    for task in pending:
                        task.cancel()
                        
                    # إذا انقطع الـ WebSocket الأساسي (الروبوت طفى أو فصل)
                    if recv_esp_task in done and recv_esp_task.exception():
                        raise recv_esp_task.exception()
                        
            except WebSocketDisconnect:
                logger.info("ESP32 Disconnected. Waiting for new connection...")
                break # نطلع من الـ while لأن الروبوت فصل
            except Exception as e:
                logger.error(f"Gemini Session Interrupted: {str(e)}. Reconnecting in 1 second...")
                await asyncio.sleep(1) # لو جمناي فصل، ننتظر ثانية ونشبك من جديد بدون ما نفصل الروبوت
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed cleanly.")
    except Exception as e:
        logger.error(f"Critical WebSocket Error: {str(e)}")
