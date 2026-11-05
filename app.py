import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ياخذ المفتاح من Render Environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = "انت رويكسا، مرشد ثقافي سعودي، ترد ب 15 كلمة فقط عن سيف والعلا والدرعية."

# === كل الروابط تشتغل GET و POST عشان ما يطلع 404 ===
@app.route('/')
def home():
    return "RIWEXA Server ONLINE - use /api/text"

@app.route('/api/text', methods=['GET', 'POST'])
@app.route('/api/roixa', methods=['GET', 'POST'])
def universal_api():
    try:
        # يشتغل سواء ارسلت ?text= او JSON
        if request.method == 'GET':
            q = request.args.get('text', 'رحب بالزائر')
        else:
            data = request.get_json(silent=True) or {}
            q = data.get('text') or request.form.get('text') or 'رحب بالزائر'

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}
            ],
            max_tokens=80
        )
        ans = completion.choices[0].message.content.strip()
        return jsonify({"answer": ans, "status": "ok", "query": q})

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"answer": "يا هلا بك في رويكسا، مرشدك في سيف 2026", "error": str(e)}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
