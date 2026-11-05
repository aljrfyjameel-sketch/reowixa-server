
import os
import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Live - استخدم مفتاح GEMINI_API_KEY من Render
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

MODEL_ID = "gemini-2.0-flash-live-001"

SYSTEM_INSTRUCTION = """
انت رويكسا RIWEXA، روبوت تفاعلي ذكي لاستقبال زوار المتاحف.
تستقبل الزائر تلقائيا، ترحب به صوتيا بجملة قصيرة، تفهم عربي وانجليزي بدون سؤال اللغة.
تحافظ على سياق المحادثة طالما الزائر موجود.
ترد بجمل قصيرة 15-20 كلمة، لهجة بيضاء، تركز على الثقافة السعودية: العلا، الدرعية، جدة التاريخية، SAIF.
"""

@app.get("/health")
def health():
    return {"status": "healthy", "service": "RIWEXA", "model": MODEL_ID, "key_set": bool(GEMINI_KEY)}

@app.get("/")
def root():
    return {"service": "RIWEXA", "ws": "/ws", "health": "/health"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to /ws")
    
    if not client:
        await websocket.send_text(json.dumps({"type": "error", "message": "GEMINI_API_KEY missing on server"}))
        await websocket.close()
        return

    try:
        # اعداد جلسة Gemini Live
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
        )

        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            # مهمة ترسل صوت Gemini الى ESP32
            async def receive_from_gemini():
                async for response in session:
                    if response.data:
                        # صوت PCM16LE من Gemini
                        b64_audio = base64.b64encode(response.data).decode()
                        await websocket.send_text(json.dumps({
                            "type": "audio",
                            "data": b64_audio,
                            "sr": 24000
                        }))
                    if response.server_content and response.server_content.turn_complete:
                        await websocket.send_text(json.dumps({"type": "audio_end"}))

            # مهمة تستقبل صوت ESP32 وترسله لـ Gemini
            async def receive_from_esp32():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "text" in msg:
                            data = json.loads(msg["text"])
                            if data.get("type") == "audio_start":
                                await websocket.send_text(json.dumps({"type": "audio_start_ack"}))
                            elif data.get("type") == "audio":
                                # صوت PCM من INMP441 على G6/G7/G13
                                pcm_bytes = base64.b64decode(data["data"])
                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
                                )
                            elif data.get("type") == "visitor_detected":
                                # HC-SR04 اكتشف زائر على G10/G11
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part(text="زائر جديد وصل، رحب به بجملة ترحيبية قصيرة عن SAIF")]
                                    ),
                                    turn_complete=True
                                )
                                await websocket.send_text(json.dumps({"type": "audio_start"}))
                            elif data.get("type") == "visitor_left":
                                await websocket.send_text(json.dumps({"type": "session_end"}))
                        elif "bytes" in msg:
                            # اذا ESP32 ارسل binary مباشر
                            await session.send_realtime_input(
                                audio=types.Blob(data=msg["bytes"], mime_type="audio/pcm;rate=16000")
                            )
                except WebSocketDisconnect:
                    print("ESP32 disconnected")

            # شغل المهمتين مع بعض
            await asyncio.gather(receive_from_gemini(), receive_from_esp32())

    except Exception as e:
        print(f"WS Error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
