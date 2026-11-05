import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
انت رويكسا RIWEXA، جهاز ثقافي محمول باليد.
مرشد سياحي سعودي، ترد باللهجة البيضاء، مختصر جدا 12-18 كلمة فقط.
مواضيعك: العلا، الدرعية، جدة التاريخية، سيف 2026.
"""

@app.route('/')
def home():
    return "RIWEXA Server ONLINE"

@app.route('/api/text', methods=['POST'])
def ask_text():
    try:
        data = request.get_json(force=True)
        user_q = data.get('text', 'رحب بالزائر')
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_q}
            ],
            max_tokens=80
        )
        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer, "status": "ok"})
    except Exception as e:
        return jsonify({"answer": "يا هلا بك في رويكسا مرشدك الثقافي", "error": str(e)})

@app.route('/api/roixa', methods=['GET'])
def ask_get():
    q = request.args.get('text', 'معلومة عن العلا')
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ],
            max_tokens=80
        )
        return jsonify({"answer": completion.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"answer": "أهلا بك في رويكسا", "error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
