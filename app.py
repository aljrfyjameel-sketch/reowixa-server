# RIWEXA - Gemini Live 3.8 - Full Production Server
# ESP32-S3 <-> WebSocket /ws <-> Gemini Live 3.8 <-> Audio PCM16
# SAIF 2026 - Live bidirectional, no REST, no gTTS, no Whisper
# الصوت Native من جمناي نفسه
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os, json, logging, asyncio, base64
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RIWEXA-LIVE")

# ================= RIWEXA SYSTEM PROMPT - المعتمد =================
RIWEXA_SYSTEM_PROMPT = """
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

app = FastAPI(title="RIWEXA Live")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ================= CONFIG - Gemini Live 3.8 =================
MODEL_ID = "gemini-3.8-live"  # <-- Live 3.8 - الحديث، مو 1.5 ولا 2.5 العادي
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=RIWEXA_SYSTEM_PROMPT)]
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    # يكتشف متى الزائر خلص كلامه تلقائيا
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
            prefix_padding_ms=20,
            silence_duration_ms=500,
        )
    ),
    # يحفظ سياق المحادثة داخل الجلسة نفسها
    session_resumption=types.SessionResumptionConfig(handle=None),
)

@app.get("/")
def root():
    return {"service": "RIWEXA", "status": "online", "model": MODEL_ID, "mode": "gemini-live", "ws": "/ws"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "RIWEXA",
        "model": MODEL_ID,
        "mode": "gemini-live-3.8",
        "gemini_configured": client is not None,
        "endpoints": ["/ws", "/health"]
    }

@app.get("/api/text")
def api_text(text: str = "هلا"):
    # للتجربة السريعة من المتصفح بدون ESP32 - يرد نص فقط
    return {"answer": f"Live mode: افتح /ws للصوت. سؤالك كان: {text}", "model": MODEL_ID, "note": "Use WebSocket /ws for audio"}

# ================= WEBSOCKET /ws - القلب =================
# بروتوكول ESP32:
# ESP32 -> Server: {"type":"audio_start"} / {"type":"audio","data":"base64 PCM16 16kHz"} / {"type":"audio_end"}
# ESP32 -> Server: {"type":"visitor_detected"} / {"type":"visitor_left"} / {"type":"session_end"}
# Server -> ESP32: {"type":"audio","data":"base64 PCM16 24kHz"} / {"type":"status","msg":"..."} / {"type":"transcript","text":"..."}
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("ESP32 connected to /ws")
    if client is None:
        await ws.send_json({"type": "error", "msg": "GEMINI_API_KEY missing in Render"})
        await ws.close()
        return

    session_id = f"live-{id(ws)}"
    try:
        async with client.aio.live.connect(model=MODEL_ID, config=LIVE_CONFIG) as session:
            logger.info(f"Gemini Live {MODEL_ID} session started {session_id}")

            # ارسل ترحيب تلقائي اول ما يتصل الزائر
            await ws.send_json({"type": "status", "msg": "live_connected", "model": MODEL_ID})

            async def esp32_to_gemini():
                """يستقبل صوت ESP32 ويحوله لجمناي"""
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("text"):
                            try:
                                data = json.loads(msg["text"])
                            except:
                                continue
                            mtype = data.get("type")
                            if mtype == "audio":
                                # ESP32 يرسل PCM16 16kHz base64
                                pcm = base64.b64decode(data["data"])
                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                                )
                            elif mtype == "audio_start":
                                logger.info("Visitor started speaking")
                            elif mtype == "audio_end":
                                logger.info("Visitor finished speaking")
                            elif mtype == "visitor_detected":
                                # الزائر وصل - خلي جمناي يرحب
                                await session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part.from_text(text="[Visitor just arrived. Greet them warmly in the language they will use, keep it short.]")]),
                                    turn_complete=True
                                )
                            elif mtype == "visitor_left" or mtype == "session_end":
                                logger.info("Visitor left - ending session")
                                await session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part.from_text(text="[Visitor is leaving. Say a short warm goodbye.]")]),
                                    turn_complete=True
                                )
                                # ننتظر ثانيتين للوداع ثم نغلق
                                await asyncio.sleep(2)
                                break
                            elif mtype == "text":
                                # للتجربة من المتصفح
                                await session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part.from_text(text=data.get("text", ""))]),
                                    turn_complete=True
                                )
                        elif msg.get("bytes"):
                            # لو ESP32 ارسل صوت خام بدون JSON
                            await session.send_realtime_input(
                                audio=types.Blob(data=msg["bytes"], mime_type="audio/pcm;rate=16000")
                            )
                except WebSocketDisconnect:
                    logger.info("ESP32 disconnected")
                except Exception as e:
                    logger.error(f"esp32_to_gemini: {e}")

            async def gemini_to_esp32():
                """يستقبل صوت جمناي ويرسله لـ ESP32"""
                try:
                    async for response in session.receive():
                        if response.server_content:
                            sc = response.server_content
                            # نص للـ LCD
                            if sc.output_transcription:
                                txt = sc.output_transcription.text
                                if txt:
                                    await ws.send_json({"type": "transcript", "text": txt, "role": "riwexa"})
                            if sc.input_transcription:
                                txt = sc.input_transcription.text
                                if txt:
                                    await ws.send_json({"type": "transcript", "text": txt, "role": "visitor"})
                            # صوت PCM16 24kHz - يروح للسماعة مباشرة
                            if sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        b64 = base64.b64encode(part.inline_data.data).decode()
                                        await ws.send_json({"type": "audio", "data": b64, "rate": 24000})
                            if sc.turn_complete:
                                await ws.send_json({"type": "turn_complete"})
                            if sc.interrupted:
                                await ws.send_json({"type": "interrupted"})
                        elif response.session_resumption_update:
                            logger.info(f"Session resumed: {response.session_resumption_update}")
                        elif response.go_away:
                            logger.info("Gemini asked to go away")
                            break
                except Exception as e:
                    logger.error(f"gemini_to_esp32: {e}")

            # شغل الاتجاهين مع بعض - صوت رايح جاي بنفس الوقت
            await asyncio.gather(esp32_to_gemini(), gemini_to_esp32())

    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as e:
        logger.error(f"Live session error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "msg": str(e), "model": MODEL_ID})
        except:
            pass
    finally:
        try:
            await ws.close()
        except:
            pass
        logger.info(f"Live session closed {session_id}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
