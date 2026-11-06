MAX_PCM_BYTES = 320_000  # ten seconds, 16 kHz, mono PCM16

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
- اجعل إجاباتك مختصرة ومفيدة.
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


# ==========================================
# Gemini text response
# ==========================================

def generate_answer(message: str) -> str:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

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


# ==========================================
# Understand incoming WAV
# ==========================================

def understand_audio(audio_bytes: bytes) -> str:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Invalid WAV file")

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono")

        if wav.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit PCM")

        if wav.getframerate() not in (16000, 24000):
            raise ValueError("Audio must be 16000 or 24000 Hz")

        if wav.getnframes() == 0:
            raise ValueError("Audio is empty")

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            SYSTEM_PROMPT,
            (
                "استمع إلى تسجيل الزائر وافهم سؤاله. "
                "أجب بلغة السؤال نفسه، بإجابة واضحة ومختصرة."
            ),
            audio_part
        ]
    )

    if not response.text:
        return "عذراً، لم أتمكن من فهم التسجيل."

    return response.text.strip()


# ==========================================
# Gemini Text To Speech
# ==========================================

def make_speech(text: str) -> tuple[bytes, int]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is missing")

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE
                    )
                )
            )
        )
    )

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = part.inline_data

            if not inline or not inline.data:
                continue

            audio = inline.data
            mime = inline.mime_type or ""


