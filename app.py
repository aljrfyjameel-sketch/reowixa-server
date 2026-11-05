import os
import json
import asyncio
from fastapi import FastAPI, WebSocket
from google import genai

app = FastAPI()

# مفتاح Gemini من متغير البيئة
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("ضع مفتاح Gemini في متغير البيئة GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# برومبت النظام — مرشد سياحي ثقافي لـ SAIF 2026
SYSTEM_INSTRUCTION = """
أنت مرشد سياحي ثقافي ذكي لمشروع RIWEXA في معرض SAIF 2026 بالمملكة العربية السعودية.
تتحدث العربية والإنجليزية بطلاقة.
تقدم معلومات دقيقة عن معالم المملكة وتاريخها وابتكاراتها.
كن ودوداً ومحترفاً ومختصراً في ردودك.
"""

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ ESP32 connected")

    try:
        # فتح جلسة مباشرة مع Gemini Live
        async with client.live.connect(
            model="gemini-2.0-flash-live",
            config={
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "language_code": "ar",
                        "voice": "Kore"  # أو أي صوت متاح
                    }
                },
                "system_instruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION}]
                }
            }
        ) as session:
            
            # إرسال رسالة ترحيب نصية
            await session.send(input="السلام عليكم، قل مرحباً للزائر")

            async def esp_to_gemini():
                """نقل الرسائل من ESP32 إلى Gemini"""
                while True:
                    data = await ws.receive()
                    
                    if isinstance(data, str):
                        # رسالة نصية (مثل visitor_arrived)
                        msg = json.loads(data)
                        if msg.get("type") == "visitor_arrived":
                            # إرسال طلب إلى Gemini لاستقبال الزائر
                            await session.send(input="وصل زائر جديد، رحّب به بلطف")
                    else:
                        # بيانات صوتية خام (PCM16)
                        await session.send(input=data)

            async def gemini_to_esp():
                """نقل الردود من Gemini إلى ESP32"""
                async for response in session.receive():
                    if response.data is not None:
                        # بيانات صوتية → أرسلها للسماعة
                        await ws.send_bytes(response.data)
                    elif response.text is not None:
                        # نص → أرسله للشاشة
                        await ws.send_text(json.dumps({
                            "type": "text",
                            "text": response.text
                        }))

            # تشغيل النقل المتزامن في الاتجاهين
            await asyncio.gather(esp_to_gemini(), gemini_to_esp())

    except Exception as e:
        print(f"❌ خطأ: {e}")
    finally:
        await ws.close()
        print("❌ ESP32 disconnected")
