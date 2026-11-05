import os
import io
import re
import json
import wave
import base64
import asyncio
import logging
import time
from contextlib import suppress
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from google import genai
from google.genai import types

# ============================================================
# RIWEXA - Enterprise Edition (SAIF 2026)
# Persistent Gemini Live Session | VAD | Telemetry | RAG
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA")

app = FastAPI(title="RIWEXA Saudi Cultural Robot - Enterprise")

API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

client = genai.Client(api_key=API_KEY) if API_KEY else None

MAX_PCM_BYTES = 320_000
MAX_TEXT_LENGTH = 4000
AUDIO_CHUNK_SIZE = 2048
SUPPORTED_SAMPLE_RATES = (16000, 24000)

# ============================================================
# Telemetry & Global State
# ============================================================
server_start_time = time.time()

telemetry = {
    "status": "OFFLINE",
    "total_visitors": 0,
    "total_questions": 0,
    "audio_bytes_streamed": 0,
    "last_visitor_time": "None",
    "current_visitor_active": False,
    "last_transcript": "Waiting for input..."
}

# ============================================================
# Enterprise Knowledge Base (RAG & Grounding)
# ============================================================
CULTURAL_KNOWLEDGE_BASE = """
[معالم المملكة وتاريخها - مرجع أساسي موثوق لـ RIWEXA]
1. الدرعية التاريخية (منطقة الرياض): عاصمة الدولة السعودية الأولى، ومسجلّة في قائمة التراث العالمي لليونسكو، وتضم حي الطريف الأثري.
2. العلا (منطقة المدينة المنورة): تضم موقع الحجر الأثري (مدائن صالح)، وهو أول موقع مسجل في المملكة ضمن قائمة التراث العالمي لليونسكو، ويتميز بآثار نبطية منحوتة في الصخور.
3. منطقة عسير (أبها وخميس مشيط): تتميز بالتراث العمراني الفريد مثل قصر شدا، وفن العسير (القط العسيري) المسجل في اليونسكو، وطبيعتها الجبلية الخلابة.
4. جدة التاريخية (البلد): تتميز بيوتها التاريخية الرواشين الخشبية القديمة، وتعتبر بوابة الحجاج التاريخية على البحر الأحمر.
5. قصر المصمك (الرياض): حصن طيني تاريخي شهد انطلاق ملحمة توحيد المملكة العربية السعودية على يد الملك عبد العزيز آل سعود عام 1902م (1319هـ).
6. رؤية السعودية 2030: خطة استراتيجية وطنية أطلقها ولي العهد الأمير محمد بن سلمان لتنويع الاقتصاد السعودي وتنمية المجالات المختلفة.
"""

GROUNDED_SYSTEM_PROMPT = f"""
You are RIWEXA, a Saudi cultural guide robot participating in SAIF 2026.

STRICT GROUNDING RULES:
- You must answer questions about Saudi history, landmarks, and culture strictly based on the Knowledge Base provided below.
- Do not invent historical facts, dates, or locations. If a fact is not in the knowledge base, state honestly that you don't have its specific details.
- Be welcoming, clear, and professional. Speak naturally and clearly in Arabic or English based on the visitor.

KNOWLEDGE BASE:
{CULTURAL_KNOWLEDGE_BASE}
"""

# ============================================================
# HTTP & Dashboard Endpoints
# ============================================================

@app.get("/")
async def home():
    return {
        "status": "ok",
        "service": "RIWEXA Enterprise",
        "mode": "telemetry-grounded-rag"
    }

@app.head("/")
async def health_head():
    return JSONResponse(content=None, status_code=200)

@app.get("/health")
async def health():
    return {
        "status": "healthy" if client else "missing_api_key",
        "service": "RIWEXA",
        "model": LIVE_MODEL,
        "uptime_seconds": int(time.time() - server_start_time)
    }

@app.get("/api/telemetry")
async def get_telemetry():
    uptime = str(timedelta(seconds=int(time.time() - server_start_time)))
    return {**telemetry, "uptime": uptime}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RIWEXA Command Center - SAIF 2026</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
            h1 { color: #58a6ff; text-align: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }
            .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;}
            .card:hover { transform: translateY(-5px); }
            .card h3 { color: #8b949e; font-size: 14px; text-transform: uppercase; margin-top: 0; }
            .card .value { font-size: 28px; font-weight: bold; color: #58a6ff; }
            .status-online { color: #3fb950; font-weight: bold; }
            .status-offline { color: #f85149; font-weight: bold; }
            .transcript-box { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin-top: 20px; color: #7ee787; font-family: monospace; text-align: left; direction: rtl;}
            .footer { text-align: center; margin-top: 40px; color: #484f58; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>RIWEXA Command Center 🚀</h1>
        <div style="text-align: center; margin-bottom: 20px;">
            Robot Status: <span id="robot-status" class="status-offline">OFFLINE</span>
            | Uptime: <span id="uptime">0:00:00</span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Total Visitors Greeted</h3>
                <div class="value" id="total-visitors">0</div>
            </div>
            <div class="card">
                <h3>Questions Answered</h3>
                <div class="value" id="total-questions">0</div>
            </div>
            <div class="card">
                <h3>Audio Processed (MB)</h3>
                <div class="value" id="audio-bytes">0.00</div>
            </div>
            <div class="card">
                <h3>Current Visitor</h3>
                <div class="value" id="visitor-active" style="color: #8b949e;">Idle</div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h3>Live Conversation Log</h3>
            <div class="transcript-box" id="last-transcript">Waiting for connection...</div>
        </div>

        <div class="footer">Engineered for SAIF 2026 | Enterprise Telemetry Module</div>

        <script>
            async function fetchTelemetry() {
                try {
                    const response = await fetch('/api/telemetry');
                    const data = await response.json();
                    
                    const statusEl = document.getElementById('robot-status');
                    statusEl.innerText = data.status;
                    statusEl.className = data.status === 'ONLINE' ? 'status-online' : 'status-offline';
                    
                    document.getElementById('uptime').innerText = data.uptime;
                    document.getElementById('total-visitors').innerText = data.total_visitors;
                    document.getElementById('total-questions').innerText = data.total_questions;
                    document.getElementById('audio-bytes').innerText = (data.audio_bytes_streamed / (1024 * 1024)).toFixed(2);
                    
                    const activeEl = document.getElementById('visitor-active');
                    if (data.current_visitor_active) {
                        activeEl.innerText = 'Engaged 🎙️';
                        activeEl.style.color = '#3fb950';
                    } else {
                        activeEl.innerText = 'Idle ⏳';
                        activeEl.style.color = '#8b949e';
                    }

                    document.getElementById('last-transcript').innerText = data.last_transcript;
                } catch (error) {
                    console.error('Failed to fetch telemetry', error);
                }
            }
            setInterval(fetchTelemetry, 2000);
            fetchTelemetry();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================================================
# Audio helpers
# ============================================================

def make_wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()

def get_output_rate(mime_type):
    match = re.search(r"rate=(\d+)", mime_type or "", re.IGNORECASE)
    rate = int(match.group(1)) if match else 24000
    if rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported output sample rate: {rate}")
    return rate

# ============================================================
# Safe WebSocket sending
# ============================================================

async def send_json(ws, lock, payload):
    async with lock:
        await ws.send_json(payload)

async def send_binary(ws, lock, payload):
    async with lock:
        await ws.send_bytes(payload)

# ============================================================
# Gemini Live response receiver with Pacing & Queue
# ============================================================

async def receive_live_turn(
    session,
    websocket,
    send_lock,
    visitor_active
):
    audio_started = False
    audio_ended = False
    audio_bytes_sent = 0
    pending_byte = b""
    output_rate = 24000
    transcript_parts = []
    
    audio_queue = asyncio.Queue()

    async def pacer_worker():
        bytes_pushed = 0
        prebuffer = 24000
        
        while visitor_active.is_set():
            chunk = await audio_queue.get()
            if chunk == b"END":
                break
                
            await send_binary(websocket, send_lock, chunk)
            bytes_pushed += len(chunk)
            telemetry["audio_bytes_streamed"] += len(chunk)
            
            if bytes_pushed > prebuffer:
                duration = (len(chunk) / 2) / output_rate
                await asyncio.sleep(duration * 0.90) 

    pacer_task = asyncio.create_task(pacer_worker())

    try:
        async for response in session.receive():
            content = response.server_content
            if content is None:
                continue

            transcription = getattr(content, "output_transcription", None)
            if transcription and transcription.text:
                transcript_parts.append(transcription.text)

            model_turn = getattr(content, "model_turn", None)
            if model_turn:
                for part in model_turn.parts or []:
                    inline = getattr(part, "inline_data", None)
                    if not inline or not inline.data:
                        continue

                    raw_audio = inline.data
                    if isinstance(raw_audio, str):
                        raw_audio = base64.b64decode(raw_audio)
                    raw_audio = bytes(raw_audio)

                    mime_type = getattr(inline, "mime_type", "") or ""
                    output_rate = get_output_rate(mime_type)

                    raw_audio = pending_byte + raw_audio
                    usable_length = len(raw_audio) & ~1
                    pending_byte = raw_audio[usable_length:]
                    raw_audio = raw_audio[:usable_length]

                    if not raw_audio or not visitor_active.is_set():
                        continue

                    if not audio_started:
                        await send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "audio_start",
                                "format": "pcm_s16le",
                                "sample_rate": output_rate,
                                "channels": 1,
                                "bits": 16
                            }
                        )
                        audio_started = True

                    for offset in range(0, len(raw_audio), AUDIO_CHUNK_SIZE):
                        if not visitor_active.is_set():
                            break
                        chunk = raw_audio[offset:offset + AUDIO_CHUNK_SIZE]
                        if chunk:
                            audio_queue.put_nowait(chunk)
                            audio_bytes_sent += len(chunk)

            if getattr(content, "interrupted", False):
                break
            if getattr(content, "turn_complete", False):
                break
    finally:
        audio_queue.put_nowait(b"END")
        await pacer_task

    if audio_started and not audio_ended:
        await send_json(websocket, send_lock, {"type": "audio_end"})
        audio_ended = True

    transcript = "".join(transcript_parts).strip()
    if transcript and visitor_active.is_set():
        telemetry["last_transcript"] = transcript
        await send_json(
            websocket,
            send_lock,
            {"type": "transcript_answer", "text": transcript}
        )

    return audio_bytes_sent

# ============================================================
# Gemini turn sender
# ============================================================

async def run_text_turn(session, websocket, send_lock, visitor_active, prompt):
    await session.send_realtime_input(text=prompt)
    return await receive_live_turn(session, websocket, send_lock, visitor_active)

async def run_audio_turn(session, websocket, send_lock, visitor_active, pcm, sample_rate):
    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid PCM audio")

    await session.send_realtime_input(activity_start=types.ActivityStart())
    chunk_size = 4096
    mime_type = f"audio/pcm;rate={sample_rate}"

    for offset in range(0, len(pcm), chunk_size):
        if not visitor_active.is_set():
            break
        chunk = pcm[offset:offset + chunk_size]
        await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type=mime_type))

    await session.send_realtime_input(activity_end=types.ActivityEnd())
    if not visitor_active.is_set():
        return 0

    return await receive_live_turn(session, websocket, send_lock, visitor_active)

# ============================================================
# Persistent visitor session worker
# ============================================================

async def visitor_worker(websocket, send_lock, queue, visitor_active):
    if client is None:
        await send_json(websocket, send_lock, {"type": "error", "message": "GEMINI_API_KEY is missing."})
        return

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=GROUNDED_SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        )
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            logger.info("Persistent visitor session opened")
            await send_json(websocket, send_lock, {"type": "session", "status": "started"})

            while True:
                job = await queue.get()
                job_type = job.get("type")

                try:
                    if job_type == "welcome":
                        visitor_active.set()
                        telemetry["last_transcript"] = "[System] Greet visitor..."
                        await run_text_turn(
                            session, websocket, send_lock, visitor_active,
                            "A visitor has arrived. Welcome them warmly in Arabic using the Grounding rules. Introduce yourself briefly as RIWEXA, their Saudi cultural guide."
                        )

                    elif job_type == "audio":
                        if visitor_active.is_set():
                            telemetry["total_questions"] += 1
                            telemetry["last_transcript"] = "[System] Processing audio question..."
                            await run_audio_turn(
                                session, websocket, send_lock, visitor_active,
                                job["pcm"], job["rate"]
                            )

                    elif job_type == "text":
                        if visitor_active.is_set():
                            telemetry["total_questions"] += 1
                            telemetry["last_transcript"] = f"[Text Question] {job['message']}"
                            await run_text_turn(
                                session, websocket, send_lock, visitor_active,
                                job["message"]
                            )

                    elif job_type == "goodbye":
                        visitor_active.set()
                        telemetry["last_transcript"] = "[System] Saying goodbye..."
                        await run_text_turn(
                            session, websocket, send_lock, visitor_active,
                            "The visitor is leaving now. Give them a brief, warm goodbye."
                        )
                        await send_json(websocket, send_lock, {"type": "session", "status": "ended"})
                        break

                except Exception as exc:
                    logger.exception("Visitor turn failed")
                    with suppress(Exception):
                        await send_json(websocket, send_lock, {"type": "error", "message": str(exc)[:250]})

                if job_type == "goodbye":
                    break

    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Persistent Gemini session failed")
        with suppress(Exception):
            await send_json(websocket, send_lock, {"type": "error", "message": "تعذر الاتصال بخدمة Gemini Live."})
    finally:
        visitor_active.clear()
        telemetry["current_visitor_active"] = False
        logger.info("Persistent visitor session closed")

# ============================================================
# WebSocket endpoint
# ============================================================

@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 connected")
    
    telemetry["status"] = "ONLINE"
    telemetry["last_transcript"] = "Robot connected. Waiting for visitor..."

    send_lock = asyncio.Lock()
    pcm_buffer = bytearray()
    pcm_rate = 16000
    receiving_pcm = False

    visitor_active = asyncio.Event()
    visitor_queue = asyncio.Queue()
    worker_task = None

    async def start_session():
        nonlocal worker_task
        if worker_task and not worker_task.done():
            return
        visitor_active.set()
        telemetry["current_visitor_active"] = True
        worker_task = asyncio.create_task(visitor_worker(websocket, send_lock, visitor_queue, visitor_active))

    try:
        await send_json(websocket, send_lock, {"type": "ready", "protocol": 1})

        while True:
            packet = await websocket.receive()
            if packet.get("type") == "websocket.disconnect":
                break

            if packet.get("bytes") is not None:
                if not receiving_pcm:
                    continue
                chunk = packet["bytes"]
                if len(pcm_buffer) + len(chunk) > MAX_PCM_BYTES:
                    pcm_buffer.clear()
                    receiving_pcm = False
                    continue
                pcm_buffer.extend(chunk)
                continue

            raw_text = packet.get("text")
            if raw_text is None:
                continue

            try:
                message = json.loads(raw_text)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type")

            if msg_type == "ping":
                await send_json(websocket, send_lock, {"type": "pong"})
                continue

            if msg_type == "event" and message.get("event") == "visitor_arrived":
                if worker_task and not worker_task.done():
                    continue
                
                telemetry["total_visitors"] += 1
                telemetry["last_visitor_time"] = datetime.now().strftime("%H:%M:%S")
                
                visitor_active.set()
                await start_session()
                await visitor_queue.put({"type": "welcome"})
                continue

            if msg_type == "event" and message.get("event") == "visitor_left":
                if worker_task and not worker_task.done():
                    visitor_active.clear()
                    telemetry["current_visitor_active"] = False
                    await visitor_queue.put({"type": "goodbye"})
                continue

            if msg_type == "audio_start":
                if receiving_pcm:
                    continue
                pcm_rate = int(message.get("sample_rate", 16000))
                pcm_buffer.clear()
                receiving_pcm = True
                continue

            if msg_type == "audio_end":
                if not receiving_pcm:
                    continue
                receiving_pcm = False
                captured = bytes(pcm_buffer)
                pcm_buffer.clear()

                if visitor_active.is_set() and worker_task and not worker_task.done():
                    await visitor_queue.put({"type": "audio", "pcm": captured, "rate": pcm_rate})
                continue

            if msg_type == "text":
                question = str(message.get("message", "")).strip()
                if visitor_active.is_set() and worker_task and not worker_task.done():
                    await visitor_queue.put({"type": "text", "message": question})
                continue

    except WebSocketDisconnect:
        logger.info("ESP32 disconnected")
    except Exception:
        logger.exception("WebSocket handler failed")
    finally:
        visitor_active.clear()
        telemetry["status"] = "OFFLINE"
        telemetry["current_visitor_active"] = False
        telemetry["last_transcript"] = "Connection to ESP32 lost."
        
        if worker_task and not worker_task.done():
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        logger.info("ESP32 WebSocket closed")

@app.on_event("startup")
async def startup_event():
    if client is None:
        logger.error("GEMINI_API_KEY is missing")
    logger.info("RIWEXA Enterprise server started")
