from flask import Flask, request, jsonify, send_file
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import logging
import io
import numpy as np
import re
import gc
import psutil
import threading
import time
import uuid
import tempfile
import atexit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# RIWEXA SYSTEM PROMPT - النسخة المعتمدة من المستخدم
# =====================================================
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

IS_HF_SPACE = os.environ.get('SPACE_ID') is not None
HF_TOKEN = os.environ.get('HF_TOKEN')

if IS_HF_SPACE:
    device = "cpu"
    torch.set_num_threads(2)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)

logger.info(f"Using device: {device}")

app = Flask(__name__)
app.config['TEMP_AUDIO_DIR'] = '/tmp/audio_responses'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

stt_pipeline = None
llm_model = None
llm_tokenizer = None
tts_pipeline = None
tts_type = None

active_files = {}
file_cleanup_lock = threading.Lock()
cleanup_thread = None
# ذاكرة المحادثة - يحفظ سياق الزيارة
conversation_history = {}  # session_id -> list

def cleanup_old_files():
    while True:
        try:
            with file_cleanup_lock:
                current_time = time.time()
                for file_id in list(active_files.keys()):
                    if current_time - active_files[file_id]['created_time'] > 300:
                        try:
                            if os.path.exists(active_files[file_id]['filepath']):
                                os.remove(active_files[file_id]['filepath'])
                            del active_files[file_id]
                        except: pass
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        time.sleep(60)

def start_cleanup_thread():
    global cleanup_thread
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
        cleanup_thread.start()

def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        mi = process.memory_info()
        return {"rss_mb": mi.rss/1024/1024, "available_mb": psutil.virtual_memory().available/1024/1024, "percent": psutil.virtual_memory().percent}
    except:
        return {"rss_mb":0, "available_mb":0, "percent":0}

def detect_language(text):
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    return 'en'

def initialize_models():
    global stt_pipeline, llm_model, llm_tokenizer, tts_pipeline, tts_type
    try:
        if stt_pipeline is None:
            logger.info("Loading Whisper-tiny STT...")
            stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device, torch_dtype=torch.float16 if device=="cuda" else torch.float32, token=HF_TOKEN, return_timestamps=False)
            logger.info("STT loaded")
            gc.collect()
        if llm_model is None:
            logger.info("Loading flan-t5-base LLM...")
            model_name = "google/flan-t5-base"
            llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32, token=HF_TOKEN, trust_remote_code=True).to(device)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            logger.info("LLM loaded")
            gc.collect()
        if tts_pipeline is None:
            try:
                from gtts import gTTS
                tts_pipeline = "gtts"; tts_type = "gtts"
            except:
                tts_pipeline = "silent"; tts_type = "silent"
        start_cleanup_thread()
        logger.info("All models loaded")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise e

def generate_llm_response(text, session_id="default"):
    try:
        if len(text) > 200:
            text = text[:200]
        if not text.strip():
            lang = detect_language(text)
            return "يا هلا بك في رويكسا، اسألني عن العلا أو الدرعية" if lang=='ar' else "Welcome to RIWEXA, ask me about AlUla or Diriyah"

        # سياق المحادثة - اخر 4 اسئلة
        history = conversation_history.get(session_id, [])[-4:]
        conv = ""
        for h in history:
            conv += f"Visitor: {h['q']}\nRIWEXA: {h['a']}\n"

        # هنا نحط التعليمات المعتمدة
        full_prompt = f"{RIWEXA_SYSTEM_PROMPT}\n\nConversation history:\n{conv}\nVisitor now: {text}\nRIWEXA answer:"

        inputs = llm_tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs_ids = llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                use_cache=True
            )
        response = llm_tokenizer.decode(outputs_ids[0], skip_special_tokens=True).strip()
        del inputs, input_ids, outputs_ids
        gc.collect()

        if not response or len(response) < 3:
            lang = detect_language(text)
            return "ما أعرف هالمعلومة، بس اسألني عن العلا أو الدرعية" if lang=='ar' else "I don't know that, but ask me about AlUla or Diriyah"

        # حفظ في ذاكرة الجلسة
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({"q": text, "a": response})
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
        return response
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ما أعرف هالمعلومة، اسألني عن شي ثاني" if detect_language(text)=='ar' else "I don't know that, please ask me something else."

def preprocess_audio_optimized(audio_bytes):
    try:
        if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF':
            audio_bytes = audio_bytes[44:]
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_samples = 30 * 16000
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        if len(audio_data) < int(0.5*16000):
            return None, None
        return 16000, audio_data
    except Exception as e:
        logger.error(f"Audio preprocess error: {e}")
        raise e

def generate_tts_audio(text):
    try:
        text = text.replace('\n',' ').strip()
        if len(text) > 200:
            text = text[:200]
        if not text:
            text = "يا هلا بك في رويكسا"
        # يكتشف اللغة تلقائيا - عربي او انجليزي
        lang = detect_language(text)
        logger.info(f"TTS generating lang={lang}: '{text[:50]}...'")
        if tts_type == "gtts":
            from gtts import gTTS
            from pydub import AudioSegment
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(tmp_file.name)
                audio_segment = AudioSegment.from_file(tmp_file.name, format="mp3")
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_data = wav_buffer.getvalue()
                os.unlink(tmp_file.name)
                return wav_data
        return b''
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b''

@app.route('/process_audio', methods=['POST'])
def process_audio():
    start_time = time.time()
    if not all([stt_pipeline, llm_model, llm_tokenizer, tts_pipeline]):
        return jsonify({"error": "Models are still loading"}), 503
    if not request.data:
        return jsonify({"error": "No audio data"}), 400
    if len(request.data) < 1000:
        return jsonify({"error": "Audio too small"}), 400
    try:
        # session_id من الهيدر او من ESP32 - يحفظ سياق الزيارة
        session_id = request.headers.get('X-Session-ID', 'default')
        rate, audio_data = preprocess_audio_optimized(request.data)
        if audio_data is None:
            return jsonify({"error": "Invalid or too short audio"}), 400
        stt_result = stt_pipeline({"sampling_rate": rate, "raw": audio_data}, generate_kwargs={"language": "ar"})
        # Whisper يكتشف اللغة تلقائيا - نحاول عربي ثم انجليزي
        transcribed_text = stt_result.get('text','').strip()
        del audio_data; gc.collect()
        if not transcribed_text or len(transcribed_text) < 2:
            transcribed_text = "هلا"
        assistant_response = generate_llm_response(transcribed_text, session_id)
        audio_response = generate_tts_audio(assistant_response)
        if not audio_response:
            return jsonify({"error": "TTS failed"}), 500
        if not os.path.exists(app.config['TEMP_AUDIO_DIR']):
            os.makedirs(app.config['TEMP_AUDIO_DIR'])
        file_id = str(uuid.uuid4())
        temp_filename = os.path.join(app.config['TEMP_AUDIO_DIR'], f"{file_id}.wav")
        with open(temp_filename, 'wb') as f:
            f.write(audio_response)
        with file_cleanup_lock:
            active_files[file_id] = {'filepath': temp_filename, 'created_time': time.time(), 'accessed': False}
        return jsonify({
            'status': 'success',
            'file_id': file_id,
            'stream_url': f'/stream_audio/{file_id}',
            'message': assistant_response,
            'transcribed': transcribed_text,
            'lang': detect_language(assistant_response),
            'processing_time': round(time.time()-start_time, 2)
        })
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        gc.collect()
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/stream_audio/<file_id>')
def stream_audio(file_id):
    try:
        with file_cleanup_lock:
            if file_id in active_files:
                filepath = active_files[file_id]['filepath']
                if os.path.exists(filepath):
                    return send_file(filepath, mimetype='audio/wav', as_attachment=False, download_name='response.wav')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': 'Stream failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    memory = get_memory_usage()
    return jsonify({
        "status": "ready" if all([stt_pipeline, llm_model, llm_tokenizer, tts_pipeline]) else "loading",
        "service": "RIWEXA",
        "model": "gemini-style prompt + flan-t5-base",
        "models": {"stt": stt_pipeline is not None, "llm": llm_model is not None, "tts": tts_pipeline is not None, "tts_type": tts_type},
        "system": {"device": device, "memory_mb": round(memory['rss_mb'],1)}
    })

@app.route('/status', methods=['GET'])
def simple_status():
    return jsonify({"ready": all([stt_pipeline, llm_model, llm_tokenizer, tts_pipeline])})

@app.route('/', methods=['GET'])
def home():
    return jsonify({"service": "RIWEXA", "status": "online", "endpoints": ["/process_audio", "/stream_audio/<id>", "/health"]})

@app.errorhandler(413)
def handle_large(e):
    return jsonify({"error": "Audio too large"}), 413

if __name__ == '__main__':
    try:
        logger.info("Starting RIWEXA Server")
        initialize_models()
        logger.info("Server ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        exit(1)
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
