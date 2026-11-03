import os
import asyncio
import websockets
from google import genai

# جلب مفتاح الـ API من متغيرات البيئة في ريندر بأمان
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

async def handler(websocket):
    print("Client connected!")
    try:
        async for message in websocket:
            print("Received data from Robot...")
            # استخدام الموديل الصحيح والسريع للرد الفوري
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="أهلاً بك يا ريويكسا، الصوت واصل وجاهز!"
            )
            await websocket.send(response.text)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    port = int(os.environ.get("PORT", 10000))
    # تعديل طريقة تشغيل الـ WebSocket لتتوافق مع أحدث إصدار
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"Server started on port {port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
