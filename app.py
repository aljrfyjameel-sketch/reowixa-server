import os
import asyncio
import websockets
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")client = genai.Client(api_key=GEMINI_API_KEY)

async def handler(websocket, path):
    print("Client connected!")
    try:
        async for message in websocket:
            response = client.models.generate_content(
                model="gemini-3.8-live",
                contents="أهلاً بك يا ريويكسا، هل الصوت واضح؟"
            )
            await websocket.send(response.text)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    port = int(os.environ.get("PORT", 10000))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"Server started on port {port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
