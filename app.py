from flask import Flask, request, jsonify, send_file
import os, logging, io, re, gc, threading, time, uuid, tempfile
import numpy as np
from 【entity-transformers¦canonical_name=transformers】 import pipeline
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== برومبتك المعتمد كامل - لا يتغير =====
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

# ===== اعداد جمناي =====
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=RIWEXA_SYSTEM_PROMPT)

app = Flask(__name__)
app.config['TEMP_AUDIO_DIR'] = '/tmp/audio_responses'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

stt_pipeline = None
conversation_history = {} # session_id -> chat session

def detect_lang(text):
    return 'ar' if re.search(r'[\u0600-\u06FF]', text) else 'en'

def init_stt():
    global stt_pipeline
    if stt_pipeline is None:
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        stt_pipeline = pipeline("automatic-speech-recognition", model="openai/【entity-whisper¦canonical_name=Whisper】-tiny", device=device)
        logger.info("【entity-Whisper¦canonical_name=Whisper】 loaded")

def get_chat(session_id):
    if session_id not in conversation_history:
        conversation_history[session_id] = gemini_model.start_chat(history=[])
    return conversation_history[session_id]

def ask_gemini(text, session_id):
    chat = get_chat(session_id)
    resp = chat.send_message(text)
    return resp.text.strip()

def make_tts(text):
    lang = detect_lang(text)
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tts.save(tmp.name)
        seg = AudioSegment.from_file(tmp.name, format="mp3")
        seg = seg.set_frame_rate(16000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        os.unlink(tmp.name)
        return buf.getvalue()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    init_stt()
    session_id = request.headers.get('X-Session-ID', 'default')
    if not request.data:
        return jsonify({"error": "No audio"}), 400
    # فك WAV
    data = request.data
    if data[:4] == b'RIFF':
        data = data[44:]
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0
    stt = stt_pipeline({"sampling_rate": 16000, "raw": audio})
    user_text = stt.get('text','').strip()
    if not user_text:
        user_text = "هلا"
    # جمناي يرد بالبرومبت المعتمد
    answer = ask_gemini(user_text, session_id)
    wav = make_tts(answer)
    if not os.path.exists(app.config['TEMP_AUDIO_DIR']):
        os.makedirs(app.config['TEMP_AUDIO_DIR'])
    fid = str(uuid.uuid4())
    path = os.path.join(app.config['TEMP_AUDIO_DIR'], f"{fid}.wav")
    open(path,'wb').write(wav)
    return jsonify({"status":"success","file_id":fid,"stream_url":f"/stream_audio/{fid}","message":answer,"transcribed":user_text,"lang":detect_lang(answer)})

@app.route('/stream_audio/<fid>')
def stream(fid):
    path = os.path.join(app.config['TEMP_AUDIO_DIR'], f"{fid}.wav")
    if os.path.exists(path):
        return send_file(path, mimetype='audio/wav')
    return jsonify({"error":"not found"}),404

@app.route('/health')
def health():
    return jsonify({"status":"healthy","service":"RIWEXA","model":"gemini-1.5-flash","stt":"whisper-tiny","tts":"gtts"})

@app.route('/')
def home():
    return jsonify({"service":"RIWEXA with Gemini","status":"online"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
