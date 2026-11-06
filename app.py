import os
import io
import re
import json
import wave
import base64
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types


# ==========================================
# RIWEXA - Saudi Cultural Robot
# Native Voice-to-Voice
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RIWEXA")

app = FastAPI(title="RIWEXA Saudi Cultural Robot")

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None

LIVE_MODEL = os.getenv(
    "GEMINI_LIVE_MODEL",
    "gemini-3.8-live"
)

TTS_VOICE = os.getenv(
    "GEMINI_TTS_VOICE",
    "Kore"
)

MAX_PCM_BYTES = 320_000

MAX_TEXT_LENGTH = 4000

AUDIO_CHUNK_SIZE = 2048

SUPPORTED_SAMPLE_RATES = (16000, 24000)


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

أسلوبك:
- تحدث بلغة الزائر.
- استخدم لغة واضحة وودودة.
- اجعل إجاباتك مفيدة ومكتملة.
- لا تختصر المعلومات الضرورية.
- احترم جميع الثقافات.
- لا تخترع معلومات تاريخية.
- إذا لم تعرف الإجابة، وضح ذلك.
- لا تدّعِ أنك ترى الزائر أو المكان دون بيانات.

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
        "voice": "Gemini Live Native Audio",
        "websocket": "ready"
    }


@app.head("/")
async def health_head():
    return JSONResponse(content=None, status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "RIWEXA",
        "gemini": "configured" if client else "missing_api_key"
    }


# ==========================================
# Audio utilities
# ==========================================

def read_wav_audio(audio_bytes: bytes):
    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Invalid WAV file")

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:

        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono")

        if wav.getsampwidth() != 2:
            raise ValueError("Audio must be PCM16")

        rate = wav.getframerate()

        if rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("Unsupported audio sample rate")

        if wav.getnframes() == 0:
            raise ValueError("Audio is empty")

        pcm = wav.readframes(wav.getnframes())

    if not pcm:
        raise ValueError("Audio is empty")

    if len(pcm) % 2:
        raise ValueError("Invalid PCM16 data")

    return pcm, rate


def make_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


# ==========================================
# Gemini Live - Native Audio Response
# ==========================================

async def answer_with_live(
    websocket: WebSocket,
    wav_bytes: bytes = None,
    question: str = None
) -> bool:

    """
    Recorded PCM/WAV input or text input
    -> Gemini Live
    -> Native audio output streamed directly to ESP32.

    No text-to-speech fallback.
    """

    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    pcm = b""
    input_rate = 16000

    if wav_bytes is not None:
        pcm, input_rate = read_wav_audio(wav_bytes)

    elif question:
        question = question.strip()

        if not question:
            raise ValueError("Question is empty")

        if len(question) > MAX_TEXT_LENGTH:
            raise ValueError("Question is too long")

    else:
        return False

    system_instruction = (
        SYSTEM_PROMPT
        + "\n\n"
        + "أجب بلغة الزائر نفسها. "
        + "استخدم إجابة صوتية واضحة وطبيعية ومكتملة. "
        + "لا تحذف المعلومات المهمة من الإجابة."
    )

    config_fields = {
        "response_modalities": ["AUDIO"],

        "system_instruction": system_instruction,

        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=TTS_VOICE
                )
            )
        ),

        "output_audio_transcription": (
            types.AudioTranscriptionConfig()
        ),
    }

    if pcm:
        config_fields["realtime_input_config"] = (
            types.RealtimeInputConfig(
                automatic_activity_detection=(
                    types.AutomaticActivityDetection(
                        disabled=True
                    )
                )
            )
        )

    config = types.LiveConnectConfig(**config_fields)

    audio_started = False
    audio_ended = False

    audio_bytes_sent = 0

    pending_byte = b""

    transcript_parts = []

    output_rate = 24000

    try:

        async with client.aio.live.connect(
            model=LIVE_MODEL,
            config=config
        ) as session:

            # ----------------------------------
            # Send visitor's recorded question
            # ----------------------------------

            if pcm:

                chunk_size = 4096

                mime_type = (
                    f"audio/pcm;rate={input_rate}"
                )

                await session.send_realtime_input(
                    activity_start=types.ActivityStart()
                )

                for offset in range(
                    0,
                    len(pcm),
                    chunk_size
                ):

                    chunk = pcm[
                        offset:offset + chunk_size
                    ]

                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type=mime_type
                        )
                    )

                await session.send_realtime_input(
                    activity_end=types.ActivityEnd()
                )

            # ----------------------------------
            # Send text question, if applicable
            # ----------------------------------

            else:

                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=question
                            )
                        ]
                    ),
                    turn_complete=True
                )

            # ----------------------------------
            # Receive Gemini audio continuously
            # ----------------------------------

            async for response in session.receive():

                content = response.server_content

                if content is None:
                    continue

                # Optional transcription only.
                # It is NOT used to generate speech.

                transcription = getattr(
                    content,
                    "output_transcription",
                    None
                )

                if (
                    transcription
                    and transcription.text
                ):

                    transcript_parts.append(
                        transcription.text
                    )

                # ----------------------------------
                # Process generated audio parts
                # ----------------------------------

                turn = content.model_turn

                if turn:

                    for part in turn.parts or []:

                        inline = part.inline_data

                        if (
                            not inline
                            or not inline.data
                        ):
                            continue

                        audio = inline.data

                        # The SDK normally provides bytes.
                        # Handle base64 strings if returned.

                        if isinstance(audio, str):
                            audio = base64.b64decode(audio)

                        audio = bytes(audio)

                        # Preserve PCM16 sample alignment.

                        audio = pending_byte + audio

                        usable_length = (
                            len(audio) & ~1
                        )

                        pending_byte = (
                            audio[usable_length:]
                        )

                        audio = audio[:usable_length]

                        if not audio:
                            continue

                        # Start speaker on first audio chunk.

                        if not audio_started:

                            mime = (
                                getattr(inline, "mime_type", "")
                                or ""
                            )

                            match = re.search(
                                r"rate=(\d+)",
                                mime,
                                re.IGNORECASE
                            )

                            if match:
                                output_rate = int(
                                    match.group(1)
                                )

                            if output_rate not in SUPPORTED_SAMPLE_RATES:
                                raise ValueError(
                                    "Unsupported Gemini output sample rate"
                                )

                            await websocket.send_json({
                                "type": "audio_start",
                                "format": "pcm_s16le",
                                "sample_rate": output_rate,
                                "channels": 1,
                                "bits": 16
                            })

                            audio_started = True

                        # Send audio immediately.
                        # No full-answer accumulation.

                        for offset in range(
                            0,
                            len(audio),
                            AUDIO_CHUNK_SIZE
                        ):

                            chunk = audio[
                                offset:offset + AUDIO_CHUNK_SIZE
                            ]

                            await websocket.send_bytes(
                                chunk
                            )

                            audio_bytes_sent += len(chunk)

                # ----------------------------------
                # Handle interruption and completion
                # ----------------------------------

                if getattr(content, "interrupted", False):
                    logger.warning(
                        "Gemini Live response interrupted"
                    )
                    break

                if content.turn_complete:
                    break

        # ----------------------------------
        # Validate completed audio
        # ----------------------------------

        if pending_byte:
            raise ValueError(
                "Incomplete PCM16 audio sample"
            )

        if not audio_started or audio_bytes_sent == 0:
            raise RuntimeError(
                "Gemini Live returned no audio"
            )

        # ----------------------------------
        # End ESP32 speaker playback
        # ----------------------------------

        await websocket.send_json({
            "type": "audio_end"
        })

        audio_ended = True

        # Optional transcript for display/debugging.

        answer = "".join(
            transcript_parts
        ).strip()

        if answer:

            await websocket.send_json({
                "type": "response",
                "message": answer
            })

            await websocket.send_json({
                "type": "transcript_answer",
                "text": answer
            })

        logger.info(
            "Gemini Live audio streamed: %d bytes, %d Hz",
            audio_bytes_sent,
            output_rate
        )

        return True

    except Exception as exc:

        provider_status = (
            getattr(exc, "status_code", None)
            or getattr(exc, "code", None)
        )

        logger.exception(
            "Gemini Live failed (%s, %s)",
            type(exc).__name__,
            provider_status
        )

        try:

            await websocket.send_json({
                "type": "diagnostic",
                "source": "live",
                "error_type": type(exc).__name__,
                "status": provider_status
            })

        except Exception:
            pass

        # If audio already started, close the playback stream.
        # Do not trigger another TTS response.

        if audio_started and not audio_ended:

            try:

                await websocket.send_json({
                    "type": "audio_end"
                })

                audio_ended = True

            except Exception:
                pass

        raise


# ==========================================
# Fixed messages - Welcome / Goodbye
# ==========================================

async def speak_message(
    websocket: WebSocket,
    message: str
):

    await websocket.send_json({
        "type": "status",
        "message": "RIWEXA تتحدث..."
    })

    await answer_with_live(
        websocket,
        question=(
            "انطق النص التالي كما هو وبصوت ودود، "
            "دون إضافة شرح: " + message
        )
    )


# ==========================================
# Process visitor question
# ==========================================

async def process_question(
    websocket: WebSocket,
    question: str = None,
    audio_bytes: bytes = None
):

    try:

        await websocket.send_json({
            "type": "status",
            "message": "جارٍ فهم سؤالك..."
        })

        success = await answer_with_live(
            websocket,
            wav_bytes=audio_bytes,
            question=question
        )

        if not success:
            raise RuntimeError(
                "Gemini Live did not produce an audio response"
            )

    except Exception as exc:

        provider_status = (
            getattr(exc, "status_code", None)
            or getattr(exc, "code", None)
        )

        logger.exception(
            "Question processing failed (%s, %s)",
            type(exc).__name__,
            provider_status
        )

        try:

            await websocket.send_json({
                "type": "error",
                "message": (
                    "تعذر معالجة السؤال صوتياً. "
                    "حاول مرة أخرى."
                ),
                "diagnostic": {
                    "error_type": type(exc).__name__,
                    "status": provider_status
                }
            })

        except Exception:
            logger.warning(
                "Could not send error to disconnected ESP32"
            )


# ==========================================
# WebSocket
# ==========================================

@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    logger.info("RIWEXA connected")

    pcm_upload = bytearray()

    pcm_rate = 16000

    receiving_pcm = False

    try:

        await websocket.send_json({
            "type": "ready",
            "protocol": 1
        })

        while True:

            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":

                logger.info(
                    "RIWEXA disconnected"
                )

                break

            # ==================================
            # Incoming binary audio
            # ==================================

            if data.get("bytes") is not None:

                audio_bytes = data["bytes"]

                if receiving_pcm:

                    if (
                        len(pcm_upload) + len(audio_bytes)
                        > MAX_PCM_BYTES
                    ):

                        pcm_upload.clear()

                        receiving_pcm = False

                        await websocket.send_json({
                            "type": "error",
                            "message": (
                                "التسجيل أطول من الحد المسموح."
                            )
                        })

                        continue

                    pcm_upload.extend(audio_bytes)

                    continue

                # Backwards compatibility:
                # previous one-frame WAV upload.

                if len(audio_bytes) < 44:

                    await websocket.send_json({
                        "type": "error",
                        "message": "ملف الصوت غير صالح."
                    })

                    continue

                if len(audio_bytes) > 2 * 1024 * 1024:

                    await websocket.send_json({
                        "type": "error",
                        "message": "التسجيل كبير جداً."
                    })

                    continue

                await process_question(
                    websocket,
                    audio_bytes=audio_bytes
                )

            # ==================================
            # Incoming JSON
            # ==================================

            elif data.get("text") is not None:

                try:

                    message = json.loads(
                        data["text"]
                    )

                except json.JSONDecodeError:

                    await websocket.send_json({
                        "type": "error",
                        "message": "صيغة الرسالة غير صحيحة."
                    })

                    continue

                message_type = message.get("type")

                # ----------------------------------
                # Start PCM recording upload
                # ----------------------------------

                if message_type == "audio_start":

                    if receiving_pcm:

                        await websocket.send_json({
                            "type": "error",
                            "message": (
                                "يوجد تسجيل قيد الإرسال بالفعل."
                            )
                        })

                        continue

                    if (
                        message.get("format") != "pcm_s16le"
                        or message.get("channels") != 1
                    ):

                        await websocket.send_json({
                            "type": "error",
                            "message": "تنسيق الصوت غير مدعوم."
                        })

                        continue

                    try:

                        pcm_rate = int(
                            message.get(
                                "sample_rate",
                                16000
                            )
                        )

                    except (TypeError, ValueError):

                        pcm_rate = 0

                    if pcm_rate not in SUPPORTED_SAMPLE_RATES:

                        await websocket.send_json({
                            "type": "error",
                            "message": "معدل العينة غير مدعوم."
                        })

                        continue

                    pcm_upload.clear()

                    receiving_pcm = True

                    continue

                # ----------------------------------
                # Finish PCM recording upload
                # ----------------------------------

                if message_type == "audio_end":

                    if not receiving_pcm:

                        await websocket.send_json({
                            "type": "error",
                            "message": (
                                "لا يوجد تسجيل لاستكماله."
                            )
                        })

                        continue

                    receiving_pcm = False

                    captured_pcm = bytes(
                        pcm_upload
                    )

                    pcm_upload.clear()

                    minimum_length = (
                        pcm_rate * 2 // 5
                    )

                    if len(captured_pcm) < minimum_length:

                        await websocket.send_json({
                            "type": "error",
                            "message": "التسجيل قصير جداً."
                        })

                        continue

                    if len(captured_pcm) % 2:

                        await websocket.send_json({
                            "type": "error",
                            "message": "بيانات التسجيل غير مكتملة."
                        })

                        continue

                    wav_bytes = make_wav(
                        captured_pcm,
                        pcm_rate
                    )

                    await process_question(
                        websocket,
                        audio_bytes=wav_bytes
                    )

                    continue

                # ----------------------------------
                # Ping
                # ----------------------------------

                if message_type == "ping":

                    await websocket.send_json({
                        "type": "pong"
                    })

                    continue

                # ----------------------------------
                # Visitor arrived
                # ----------------------------------

                if (
                    message_type == "event"
                    and message.get("event") == "visitor_arrived"
                ):

                    language = message.get(
                        "language",
                        "ar"
                    )

                    if language == "en":

                        greeting = (
                            "Hello and welcome! "
                            "I am RIWEXA, your Saudi cultural guide. "
                            "You can ask me about Saudi history and heritage."
                        )

                    else:

                        greeting = (
                            "أهلاً وسهلاً بكم! "
                            "أنا ريوكسا، دليلكم الثقافي السعودي. "
                            "اسألوني عن تاريخ المملكة وتراثها."
                        )

                    try:

                        await speak_message(
                            websocket,
                            greeting
                        )

                    except Exception:

                        logger.exception(
                            "Welcome speech failed"
                        )

                        await websocket.send_json({
                            "type": "error",
                            "message": (
                                "تعذر تشغيل الترحيب الصوتي."
                            )
                        })

                    continue

                # ----------------------------------
                # Visitor left
                # ----------------------------------

                if (
                    message_type == "event"
                    and message.get("event") == "visitor_left"
                ):

                    language = message.get(
                        "language",
                        "ar"
                    )

                    if language == "en":

                        goodbye = (
                            "Thank you for visiting. "
                            "We hope you enjoyed discovering Saudi culture. "
                            "Have a wonderful day!"
                        )

                    else:

                        goodbye = (
                            "شكراً لزيارتكم! "
                            "سعدتُ بمشاركتكم التعرف على الثقافة السعودية. "
                            "نتمنى لكم يوماً سعيداً."
                        )

                    try:

                        await speak_message(
                            websocket,
                            goodbye
                        )

                    except Exception:

                        logger.exception(
                            "Goodbye speech failed"
                        )

                        await websocket.send_json({
                            "type": "error",
                            "message": (
                                "تعذر تشغيل رسالة الوداع."
                            )
                        })

                    continue

                # ----------------------------------
                # Text question
                # ----------------------------------

                if message_type == "text":

                    user_message = str(
                        message.get(
                            "message",
                            ""
                        )
                    ).strip()

                    if not user_message:

                        await websocket.send_json({
                            "type": "error",
                            "message": "اكتب سؤالك أولاً."
                        })

                        continue

                    if len(user_message) > MAX_TEXT_LENGTH:

                        await websocket.send_json({
                            "type": "error",
                            "message": "الرسالة طويلة جداً."
                        })

                        continue

                    await process_question(
                        websocket,
                        question=user_message
                    )

                    continue

                # ----------------------------------
                # Unknown message
                # ----------------------------------

                await websocket.send_json({
                    "type": "error",
                    "message": "نوع الرسالة غير معروف."
                })

    except WebSocketDisconnect:

        logger.info(
            "RIWEXA disconnected"
        )

    except Exception:

        logger.exception(
            "WebSocket connection error"
        )


# ==========================================
# Startup
# ==========================================

@app.on_event("startup")
async def startup_event():

    if not client:

        logger.error(
            "GEMINI_API_KEY is missing"
        )

    else:

        logger.info(
            "Gemini client configured"
        )

    logger.info(
        "RIWEXA Saudi Cultural Robot started"
    )
