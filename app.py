from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# مفتاح Gemini من Render Environment
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM = "انت رويكسا، جهاز ثقافي محمول باليد، مرشد سياحي سعودي، ترد باللهجة البيضاء مختصر جدا 12-18 كلمة فقط عن العلا والدرعية وسيف 2026."

@app.get("/")
def home():
    return {"status": "RIWEXA Gemini ONLINE"}

@app.api_route("/api/text", methods=["GET", "POST"])
@app.api_route("/api/roixa", methods=["GET", "POST"])
async def universal_api(request: Request):
    try:
        if request.method == "GET":
            q = request.query_params.get("text", "رحب بالزائر")
        else:
            try:
                body = await request.json()
                q = body.get("text", "رحب بالزائر")
            except:
                form = await request.form()
                q = form.get("text", "رحب بالزائر")

        # Gemini
        prompt = f"{SYSTEM}\n\nسؤال الزائر: {q}\nرد رويكسا:"
        response = model.generate_content(prompt)
        ans = response.text.strip()
        
        return {"answer": ans, "status": "ok", "query": q}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"answer": "يا هلا والله في رويكسا مرشدك الثقافي في سيف", "error": str(e)}
